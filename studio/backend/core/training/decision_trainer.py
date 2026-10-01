# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Full fine-tuning of Laya decision models (RLCD + per-type temperature calibration)."""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loggers import get_logger

logger = get_logger(__name__)

HOLDOUT_MAX = 400
MIN_CALIBRATION_ITEMS = 10
GROUP_SIZE = 4
SIGMA_START, SIGMA_END = 0.4, 0.1
HEAD_LR_SCALE = 4.0
LORA_HEAD_LR = 1e-4
LORA_TARGET_MODULES = ["Wqkv", "Wo", "Wi"]


class DecisionDataError(ValueError):
    pass


def _parsed(value):
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _option_keys(internal: dict) -> list[str]:
    if internal["t"] == "choice":
        return [str(key) for key in internal["crit"]]
    if internal["t"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(internal["crit"]))]


def _target(internal: dict, gold) -> tuple[list[float], int]:
    keys = _option_keys(internal)
    label = gold.get("label") if isinstance(gold, dict) else gold
    if isinstance(label, bool) or (internal["t"] == "noul" and isinstance(label, str)):
        label = str(label).lower()
    elif isinstance(label, (int, float)) and internal["t"] == "score":
        label = str(int(label))
    elif label is not None:
        label = str(label)
    probabilities = gold.get("probabilities") if isinstance(gold, dict) else None
    if isinstance(probabilities, dict):
        try:
            target = [max(0.0, float(probabilities.get(key, 0.0))) for key in keys]
        except (TypeError, ValueError):
            target = []
        total = sum(target)
        if total > 0:
            target = [value / total for value in target]
            return target, keys.index(label) if label in keys else target.index(max(target))
    if label in keys:
        return [1.0 if key == label else 0.0 for key in keys], keys.index(label)
    raise DecisionDataError("gold has no usable label or probabilities")


def _build_items(rows, tok, cfg: dict) -> tuple[list[dict], int, int, str | None]:
    from fastapi import HTTPException
    from pydantic import ValidationError

    from core.systemone import laya_runtime
    from routes.systemone import QuestionIn, _validate

    laya = laya_runtime._laya()
    build_sequence, render_options = laya.common.build_sequence, laya.common.render_options
    to_internal = laya.agent.Agent._to_internal
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))

    items: list[dict] = []
    total = skipped = 0
    first_reason: str | None = None

    def skip(reason: str) -> None:
        nonlocal skipped, first_reason
        skipped += 1
        first_reason = first_reason or reason

    for index, row in enumerate(rows):
        state = _parsed(row.get("state"))
        questions = _parsed(row.get("questions"))
        gold = _parsed(row["gold"] if row.get("gold") is not None else row.get("answers"))
        if state is None or not isinstance(questions, dict) or not isinstance(gold, dict):
            total += 1
            skip(f"row {index + 1} needs state, questions and gold")
            continue
        for name, question in questions.items():
            total += 1
            if name not in gold:
                skip(f'row {index + 1} has no gold for "{name}"')
                continue
            try:
                parsed = QuestionIn.model_validate(question)
                _validate(name, parsed)
                internal = to_internal(laya_runtime._to_laya(parsed.model_dump()))
                target, label = _target(internal, _parsed(gold[name]))
            except HTTPException as exc:
                skip(f"row {index + 1}: {exc.detail['message']}")
                continue
            except ValidationError:
                skip(f'row {index + 1}: question "{name}" is not a valid question')
                continue
            except DecisionDataError as exc:
                skip(f'row {index + 1}: "{name}" {exc}')
                continue
            ids, markers = build_sequence(tok, state, internal, max_len, head_max_len)
            if len(markers) != len(render_options(internal)) or len(markers) != len(target):
                skip(f'row {index + 1}: "{name}" options exceed the {max_len}-token context')
                continue
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": laya.common.QTYPES[internal["t"]],
                    "target": target,
                    "label": label,
                }
            )
    return items, total, skipped, first_reason


def _read_local_rows(paths: list[str], load_dataset) -> list[dict]:
    from utils.paths import dataset_files_in_dir, datasets_root

    files: list[Path] = []
    for entry in paths:
        path = Path(entry if os.path.isabs(entry) else os.path.join(str(datasets_root()), entry))
        files.extend(dataset_files_in_dir(path) if path.is_dir() else [path])
    if not files:
        raise ValueError("No local dataset files found")
    rows: list[dict] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix in (".json", ".jsonl"):
            # Read directly: Arrow would merge every question's criteria keys into one struct.
            text = path.read_text(encoding = "utf-8-sig")
            try:
                data = json.loads(text)
                rows.extend(data if isinstance(data, list) else [data])
            except ValueError:
                rows.extend(json.loads(line) for line in text.splitlines() if line.strip())
        elif suffix in (".csv", ".parquet"):
            rows.extend(load_dataset(suffix[1:], data_files = [str(path)], split = "train"))
        else:
            raise ValueError(f"Unsupported local dataset format: {path.name}")
    return [row for row in rows if isinstance(row, dict)]


def _load_rows(config: dict, should_stop: Callable[[], bool], status) -> tuple[list, list | None]:
    from core.training.eval_dataset import evaluation_enabled
    from core.training.worker import _load_embedding_hf_dataset, _worker_hf_token
    from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset

    hf_dataset = str(config.get("hf_dataset") or "").strip()
    local_datasets = config.get("local_datasets") or []
    evaluate = evaluation_enabled(config.get("eval_steps"))
    eval_rows = None
    if hf_dataset:
        rows = list(_load_embedding_hf_dataset(config, load_dataset, status))
        if evaluate and config.get("eval_split"):
            eval_rows = list(
                load_dataset(
                    hf_dataset,
                    config.get("subset") or None,
                    split = config["eval_split"],
                    token = _worker_hf_token(config),
                )
            )
    elif local_datasets:
        rows = _read_local_rows(local_datasets, load_dataset)
    elif config.get("s3_config"):
        from core.training.s3_dataset import prepare_s3_dataset_download

        status("Downloading dataset from S3...")
        download = prepare_s3_dataset_download(config["s3_config"], cancel_callback = should_stop)
        try:
            rows = _read_local_rows(download.files, load_dataset)
        finally:
            download.cleanup()
    else:
        raise ValueError("No dataset specified for decision training.")
    if evaluate and config.get("local_eval_datasets"):
        eval_rows = _read_local_rows(config["local_eval_datasets"], load_dataset)

    start, end = config.get("dataset_slice_start"), config.get("dataset_slice_end")
    if start is not None or end is not None:
        rows = rows[start or 0 : (len(rows) if end is None else end + 1)]
    return rows, eval_rows


def _autocast(torch, device):
    if device.type == "cuda":
        bf16 = (
            torch.cuda.is_bf16_supported()
            if torch.version.hip
            else torch.cuda.get_device_capability(device)[0] >= 8
        )
        return torch.bfloat16 if bf16 else torch.float16
    if device.type == "xpu":
        return torch.bfloat16
    return None


def _forward(model, batch, device, amp_dtype):
    import torch
    with torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
        logits, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
    return logits.float()


def _heldout_logits(
    model,
    items,
    collate,
    device,
    amp_dtype,
    batch_size: int = 16,
):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            logits = _forward(model, collate([chunk]), device, amp_dtype).cpu()
            out.extend(logits[row, : len(item["markers"])] for row, item in enumerate(chunk))
    model.train()
    return out


def _soft_ce(logits, items) -> float:
    import torch
    losses = [
        -(torch.tensor(item["target"]) * torch.log_softmax(z, -1)).sum().item()
        for z, item in zip(logits, items)
    ]
    return sum(losses) / max(1, len(losses))


def _metrics(logits, items, temperature_for, ece_score) -> tuple[float, float]:
    import numpy as np
    import torch

    conf, correct = [], []
    for z, item in zip(logits, items):
        p = torch.softmax(z / temperature_for(item["qtype"], len(z)), -1)
        conf.append(float(p.max()))
        correct.append(float(int(p.argmax()) == item["label"]))
    return float(np.mean(correct)), ece_score(np.array(conf), np.array(correct))


def _fit_temperature(logits, items) -> float:
    import torch

    kmax = max(len(z) for z in logits)
    z = torch.full((len(logits), kmax), -1e4)
    target = torch.zeros((len(logits), kmax))
    for i, (row, item) in enumerate(zip(logits, items)):
        z[i, : len(row)] = row
        target[i, : len(row)] = torch.tensor(item["target"])
    log_t = torch.zeros(1, requires_grad = True)
    optimizer = torch.optim.LBFGS([log_t], lr = 0.1, max_iter = 100)

    def closure():
        optimizer.zero_grad()
        loss = -(target * torch.log_softmax(z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def _optimizer(groups: list[dict], name: str, weight_decay: float, device):
    import torch
    if name == "adamw_8bit" and device.type == "cuda":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, weight_decay = weight_decay)
        except Exception as exc:
            logger.warning("8-bit AdamW unavailable (%s); using torch AdamW", exc)
    return torch.optim.AdamW(groups, weight_decay = weight_decay)


def _length_grouped_batches(items: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    import torch
    from transformers.trainer_pt_utils import get_length_grouped_indices

    order = get_length_grouped_indices(
        [len(item["ids"]) for item in items],
        batch_size,
        generator = torch.Generator().manual_seed(seed),
    )
    batches = [
        [items[i] for i in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]
    # The longest batch stays first so an out-of-memory shows up on the first step.
    rest = batches[1:]
    random.Random(seed).shuffle(rest)
    return batches[:1] + rest


def _save(output_dir: Path, model, tok, cfg: dict, fix_tokenizer_config) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents = True, exist_ok = True)
    config_path = output_dir / "rl_agent_config.json"
    config_path.unlink(missing_ok = True)
    save_file(
        {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()},
        str(output_dir / "model.safetensors"),
    )
    model.encoder.config.save_pretrained(str(output_dir / "encoder"))
    tok.save_pretrained(str(output_dir / "tokenizer"))
    fix_tokenizer_config(str(output_dir))
    # Written last: a folder with rl_agent_config.json is a complete, servable checkpoint.
    partial = output_dir / "rl_agent_config.json.tmp"
    partial.write_text(json.dumps(cfg, indent = 2), encoding = "utf-8")
    os.replace(partial, config_path)


def run_decision_training(event_queue: Any, stop_queue: Any, config: dict) -> None:
    import torch
    try:
        _run(event_queue, stop_queue, config)
    except torch.cuda.OutOfMemoryError:
        event_queue.put(
            {
                "type": "error",
                "error": (
                    "Out of GPU memory while training the decision model. Lower the batch size "
                    "and raise gradient accumulation to keep the same effective batch."
                ),
                "stack": "",
                "ts": time.time(),
            }
        )


def _run(event_queue: Any, stop_queue: Any, config: dict) -> None:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, get_scheduler

    from core.systemone import laya_runtime
    from core.systemone.catalog import Checkpoint
    from core.training.resume import session_eta_seconds
    from core.training.worker import (
        _emit_output_dir,
        _send_status,
        _start_worker_stop_poller,
        _worker_hf_token,
    )
    from utils.paths import is_local_path, resolve_output_dir
    from utils.training_runs import build_default_output_dir_name

    def send(kind: str, **payload) -> None:
        event_queue.put({"type": kind, **payload, "ts": time.time()})

    def status(message: str) -> None:
        _send_status(event_queue, message)

    stop = {"requested": False, "save": True}

    def on_stop(save: bool) -> None:
        stop["requested"], stop["save"] = True, save
        logger.info("Decision training: stop signal received (save=%s)", save)

    _start_worker_stop_poller(stop_queue, on_stop)

    def stopped_before_training() -> bool:
        if stop["requested"]:
            message = "Training stopped" if stop["save"] else "Training cancelled"
            send("complete", output_dir = None, status_message = message)
        return stop["requested"]

    model_name = config["model_name"]
    subfolder = config.get("model_subfolder") or None
    hf_token = _worker_hf_token(config)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    seed = int(config.get("random_seed") or 3407)
    random.seed(seed)
    torch.manual_seed(seed)

    status("Loading decision model...")
    laya = laya_runtime._laya()
    base = Checkpoint("base", model_name, subfolder, "")
    try:
        root = laya_runtime._checkpoint_dir(base)
    except FileNotFoundError as exc:
        send("error", error = f"Not a Laya decision checkpoint: {exc}", stack = "")
        return
    folder = root / subfolder if subfolder else root
    cfg = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
    laya.agent._fix_tokenizer_config(str(folder))
    tok = AutoTokenizer.from_pretrained(str(folder / "tokenizer"))
    model = laya.common.build_model(cfg, encoder_dir = str(folder / "encoder"))
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict = True)
    # As laya does at load: ModernBERT would otherwise torch.compile parts of the encoder.
    model.encoder.config.reference_compile = False
    if stopped_before_training():
        return

    status("Loading dataset...")
    rows, eval_rows = _load_rows(config, lambda: stop["requested"], status)
    if stopped_before_training():
        return
    status("Preparing decisions...")
    items, total, skipped, reason = _build_items(rows, tok, cfg)
    eval_items = None
    if eval_rows is not None:
        eval_items, eval_total, eval_skipped, eval_reason = _build_items(eval_rows, tok, cfg)
        total, skipped, reason = total + eval_total, skipped + eval_skipped, reason or eval_reason
    if not items:
        send(
            "error",
            error = (
                "No usable decisions in the dataset"
                + (f" ({reason})" if reason else "")
                + ". Each row needs state, questions and gold."
            ),
            stack = "",
        )
        return
    if skipped:
        send("warning", message = f"Skipped {skipped:,} of {total:,} decisions: {reason}.")

    if eval_items is None:
        order = list(range(len(items)))
        random.Random(seed).shuffle(order)
        held = set(order[: min(HOLDOUT_MAX, len(items) // 10)])
        eval_items = [item for i, item in enumerate(items) if i in held]
        items = [item for i, item in enumerate(items) if i not in held]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    amp_dtype = _autocast(torch, device)
    # Laya checkpoints are fp16, so frozen weights kept in fp16 are saved back bit-exact.
    frozen_dtype = torch.float16 if amp_dtype is not None else torch.float32
    use_lora = config.get("training_type") == "LoRA/QLoRA"
    checkpointing = str(config.get("gradient_checkpointing") or "none").lower()
    if checkpointing == "unsloth":
        from core.import_guards import ensure_real_packages

        ensure_real_packages("unsloth_zoo", "unsloth")
        import unsloth  # noqa: F401
        from unsloth_zoo.gradient_checkpointing import patch_unsloth_smart_gradient_checkpointing

        patch_unsloth_smart_gradient_checkpointing(dtype = amp_dtype)
    if checkpointing not in ("none", "false"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": checkpointing == "unsloth"}
        )
        if checkpointing == "unsloth":
            model.encoder.enable_input_require_grads()
    if use_lora:
        from peft import LoraConfig, get_peft_model

        model.encoder.requires_grad_(False)
        model.encoder.to(frozen_dtype)
        model.encoder = get_peft_model(
            model.encoder,
            LoraConfig(
                r = int(config.get("lora_r") or 64),
                lora_alpha = int(config.get("lora_alpha") or 64),
                lora_dropout = float(config.get("lora_dropout") or 0.0),
                target_modules = LORA_TARGET_MODULES,
                bias = "none",
            ),
        )
    else:
        embeddings = model.encoder.get_input_embeddings()
        embeddings.weight.requires_grad_(False)
        embeddings.to(frozen_dtype)
    model.to(device).train()

    collate = lambda batch: laya.common.collate_items(batch, tok.pad_token_id)
    clamp, temp_bucket = laya.common.clamp_temperature, laya.common.temp_bucket
    base_temperature = [clamp(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
    base_buckets = {k: clamp(v) for k, v in cfg.get("temperature_by_options", {}).items()}
    base_metrics = None
    if eval_items:
        status(f"Evaluating the base model on {len(eval_items):,} held-out decisions...")
        logits = _heldout_logits(model, eval_items, collate, device, amp_dtype)
        base_metrics = _metrics(
            logits,
            eval_items,
            lambda qt, k: base_buckets.get(temp_bucket(qt, k), base_temperature[qt]),
            laya.common.ece_score,
        )
        logger.info("Base held-out accuracy %.4f, ECE %.4f", *base_metrics)
    if stopped_before_training():
        return

    batch_size = max(1, int(config.get("batch_size") or 8))
    accumulation = max(1, int(config.get("gradient_accumulation_steps") or 1))
    micro_batches = math.ceil(len(items) / batch_size)
    steps_per_epoch = max(1, math.ceil(micro_batches / accumulation))
    max_steps = int(config.get("max_steps") or 0)
    epochs = max(1, int(config.get("num_epochs") or 1))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * epochs
    epochs = math.ceil(total_steps / steps_per_epoch)
    learning_rate = float(config.get("learning_rate") or 2.5e-5)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = _optimizer(
        [
            {"params": [p for n, p in trainable if "encoder." in n], "lr": learning_rate},
            {
                "params": [p for n, p in trainable if "encoder." not in n],
                "lr": LORA_HEAD_LR if use_lora else learning_rate * HEAD_LR_SCALE,
            },
        ],
        str(config.get("optim") or "adamw_torch").lower(),
        float(config.get("weight_decay") or 0.0),
        device,
    )
    warmup = int(config.get("warmup_steps") or 0) or round(
        float(config.get("warmup_ratio") or 0.0) * total_steps
    )
    scheduler = get_scheduler(
        config.get("lr_scheduler_type") or "cosine",
        optimizer,
        num_warmup_steps = warmup,
        num_training_steps = total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled = amp_dtype == torch.float16)
    max_grad_norm = float(config.get("max_grad_norm") or 1.0)

    output_dir = str(
        resolve_output_dir(
            config.get("output_dir")
            or build_default_output_dir_name(
                model_name
                if is_local_path(model_name) or not subfolder
                else f"{model_name}-{subfolder}",
                config.get("project_name"),
            )
        )
    )
    _emit_output_dir(event_queue, output_dir)

    wandb = None
    if config.get("enable_wandb"):
        try:
            import wandb
            if config.get("wandb_token"):
                os.environ["WANDB_API_KEY"] = config["wandb_token"]
            wandb.init(
                project = config.get("wandb_project") or "unsloth-training",
                name = Path(output_dir).name,
                config = {k: config.get(k) for k in ("model_name", "model_subfolder")},
            )
        except Exception as exc:
            wandb = None
            send("warning", message = f"Weights & Biases logging is off: {exc}")

    status("Training in progress...")
    start = time.time()
    step = 0
    peak_gb = None
    try:
        for epoch in range(epochs):
            if step >= total_steps or stop["requested"]:
                break
            sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, epochs - 1))
            chunks = _length_grouped_batches(items, batch_size, seed + epoch)
            losses: list[float] = []
            for index, chunk in enumerate(chunks):
                if stop["requested"] or step >= total_steps:
                    break
                batch = collate([chunk])
                logits = _forward(model, batch, device, amp_dtype)
                mask = batch["marker_mask"].to(device)
                target = batch["target"].to(device)
                options = mask.sum(-1, keepdim = True).float()
                noise = torch.randn((GROUP_SIZE,) + logits.shape, device = device) * sigma * mask
                noise = (noise - noise.sum(-1, keepdim = True) / options) * mask
                sampled = logits.detach().unsqueeze(0) + noise
                with torch.no_grad():
                    reward = laya.common.proper_reward(
                        torch.softmax(sampled.masked_fill(~mask, -1e4), -1),
                        target.unsqueeze(0),
                        batch["qtype"].to(device),
                        mask,
                        w_sph = 0.75,
                        w_rps = 1.0,
                    )
                    advantage = reward - reward.mean(0, keepdim = True)
                    advantage = advantage / (advantage.std() + 1e-6)
                log_prob = -(((sampled - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
                soft_ce = (
                    -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1))
                    .sum(-1)
                    .mean()
                )
                loss = -(advantage * log_prob).mean() + soft_ce
                scaler.scale(loss / accumulation).backward()
                losses.append(loss.item())
                if (index + 1) % accumulation and index + 1 < len(chunks):
                    continue
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none = True)
                step += 1
                elapsed = time.time() - start
                if device.type == "cuda":
                    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                metrics = {
                    "step": step,
                    "epoch": round(epoch + (index + 1) / len(chunks), 2),
                    "loss": sum(losses) / len(losses),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                }
                last_of_epoch = index + 1 == len(chunks) or step >= total_steps
                eval_loss = None
                if last_of_epoch and eval_items:
                    eval_loss = _soft_ce(
                        _heldout_logits(model, eval_items, collate, device, amp_dtype),
                        eval_items,
                    )
                send(
                    "progress",
                    **metrics,
                    total_steps = total_steps,
                    elapsed_seconds = elapsed,
                    eta_seconds = session_eta_seconds(elapsed, step, 0, total_steps),
                    session_start_step = 0,
                    eval_loss = eval_loss,
                    peak_memory_gb = peak_gb,
                    status_message = "",
                )
                if wandb is not None:
                    wandb.log(
                        {**metrics, **({"eval_loss": eval_loss} if eval_loss is not None else {})},
                        step = step,
                    )
                losses = []
    finally:
        if wandb is not None:
            wandb.finish()

    if stop["requested"] and not stop["save"]:
        send("complete", output_dir = None, status_message = "Training cancelled")
        return
    if use_lora:
        model.encoder = model.encoder.merge_and_unload()

    status("Calibrating confidence...")
    temperature = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    tuned_metrics = None
    if eval_items:
        logits = _heldout_logits(model, eval_items, collate, device, amp_dtype)
        for qtype in range(3):
            chosen = [i for i, item in enumerate(eval_items) if item["qtype"] == qtype]
            if len(chosen) >= MIN_CALIBRATION_ITEMS:
                temperature[qtype] = clamp(
                    _fit_temperature([logits[i] for i in chosen], [eval_items[i] for i in chosen])
                )
        tuned_metrics = _metrics(
            logits, eval_items, lambda qt, k: temperature[qt], laya.common.ece_score
        )
        logger.info("Fine-tuned held-out accuracy %.4f, ECE %.4f", *tuned_metrics)

    status("Saving model...")
    cfg["fine_tuned"] = True
    cfg["temperature"] = temperature
    # Inherited per-bucket temperatures take precedence at load and would mask the new fit.
    cfg.pop("temperature_by_options", None)
    cfg["training"] = {
        "base": model_name,
        "subfolder": subfolder,
        "method": "lora" if use_lora else "full",
        "dataset": config.get("hf_dataset")
        or [Path(path).name for path in config.get("local_datasets") or []],
        "steps": step,
        "epochs": epochs,
        "heldout_decisions": len(eval_items or []),
        "heldout_accuracy_base": base_metrics and round(base_metrics[0], 4),
        "heldout_accuracy": tuned_metrics and round(tuned_metrics[0], 4),
        "heldout_ece_base": base_metrics and round(base_metrics[1], 4),
        "heldout_ece": tuned_metrics and round(tuned_metrics[1], 4),
        "seconds": round(time.time() - start, 1),
        "peak_memory_gb": peak_gb and round(peak_gb, 2),
        "date": datetime.now(timezone.utc).isoformat(timespec = "seconds"),
    }
    _save(Path(output_dir), model, tok, cfg, laya.agent._fix_tokenizer_config)
    logger.info("Decision model saved to %s: %s", output_dir, cfg["training"])

    message = "Decision training completed"
    if base_metrics and tuned_metrics:
        message = (
            f"Held-out accuracy {base_metrics[0]:.2f} -> {tuned_metrics[0]:.2f}, "
            f"calibration error {tuned_metrics[1]:.2f}"
        )
    send("complete", output_dir = output_dir, status_message = message)
