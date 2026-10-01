# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's decision model training: loads the dataset, then trains with unsloth's FastDecisionModel."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loggers import get_logger

logger = get_logger(__name__)

EVAL_MAX = 2000
MIN_REPORTED_ITEMS = 50


def _studio_validate(name: str, question) -> None:
    from fastapi import HTTPException
    from pydantic import ValidationError

    from routes.systemone import QuestionIn, _validate

    # The Decision API's own checks, so every trained question is one it will serve.
    try:
        _validate(name, QuestionIn.model_validate(question))
    except ValidationError:
        raise ValueError("is not a valid question") from None
    except HTTPException as exc:
        raise ValueError(exc.detail["message"]) from None


def _without_nulls(value):
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    return value


def _from_arrow(rows) -> list[dict]:
    # Arrow gives every row the union of all rows' keys, filled with None; a null criterion
    # description is therefore read as absent, and "" means an option without one.
    return [
        {
            **row,
            **{
                key: _without_nulls(row[key])
                for key in ("questions", "gold", "answers")
                if isinstance(row.get(key), dict)
            },
        }
        for row in rows
    ]


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
            rows.extend(
                _from_arrow(load_dataset(suffix[1:], data_files = [str(path)], split = "train"))
            )
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
        rows = _from_arrow(_load_embedding_hf_dataset(config, load_dataset, status))
        if evaluate and config.get("eval_split"):
            eval_rows = _from_arrow(
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
    import random

    from core.import_guards import ensure_real_packages
    from core.systemone import laya_runtime

    # unsloth's decision support imports `laya`; register Studio's vendored copy under that name first.
    laya_runtime._laya()
    ensure_real_packages("unsloth_zoo", "unsloth")
    import torch
    from transformers import TrainingArguments
    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported

    from core.systemone.catalog import Checkpoint
    from core.training.eval_dataset import evaluation_enabled
    from core.training.trainer import _drop_hf_stdout_callbacks, _hf_stdout_progress_disabled
    from core.training.training import apply_save_strategy
    from core.training.worker import (
        _create_embedding_progress_callback,
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
    use_lora = config.get("training_type") == "LoRA/QLoRA"

    status("Loading decision model...")
    try:
        root = laya_runtime._checkpoint_dir(Checkpoint("base", model_name, subfolder, ""))
    except FileNotFoundError as exc:
        send("error", error = f"Not a Laya decision checkpoint: {exc}", stack = "")
        return
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(root / subfolder if subfolder else root),
        full_finetuning = not use_lora,
        use_gradient_checkpointing = config.get("gradient_checkpointing") or "none",
    )
    if use_lora:
        model = FastDecisionModel.get_peft_model(
            model,
            r = int(config.get("lora_r") or 64),
            lora_alpha = int(config.get("lora_alpha") or 64),
            lora_dropout = float(config.get("lora_dropout") or 0.0),
            random_state = seed,
            use_rslora = bool(config.get("use_rslora")),
        )
    if stopped_before_training():
        return

    status("Loading dataset...")
    rows, eval_rows = _load_rows(config, lambda: stop["requested"], status)
    if stopped_before_training():
        return
    status("Preparing decisions...")
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model, _studio_validate)
    eval_items = None
    if eval_rows is not None:
        eval_items, eval_report = FastDecisionModel.build_dataset(
            eval_rows, tokenizer, model, _studio_validate
        )
        report = {
            "total": report["total"] + eval_report["total"],
            "skipped": report["skipped"] + eval_report["skipped"],
            "reason": report["reason"] or eval_report["reason"],
        }
    if not items:
        reason = f" ({report['reason']})" if report["reason"] else ""
        send(
            "error",
            error = f"No usable decisions in the dataset{reason}. Each row needs state, questions and gold.",
            stack = "",
        )
        return
    if report["skipped"]:
        send(
            "warning",
            message = f"Skipped {report['skipped']:,} of {report['total']:,} decisions: {report['reason']}.",
        )
    if eval_items is None:
        items, eval_items = FastDecisionModel.split_holdout(items, seed)
        if eval_items:
            status(f"Holding out {len(eval_items):,} decisions to calibrate confidence...")
    elif len(eval_items) > EVAL_MAX:
        eval_items = random.Random(seed).sample(eval_items, EVAL_MAX)

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

    batch_size = max(1, int(config.get("batch_size") or 8))
    accumulation = max(1, int(config.get("gradient_accumulation_steps") or 1))
    max_steps = int(config.get("max_steps") or 0)
    epochs = max(1, int(config.get("num_epochs") or 1))
    steps_per_epoch = max(1, math.ceil(math.ceil(len(items) / batch_size) / accumulation))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * epochs
    bf16 = is_bfloat16_supported() if torch.cuda.is_available() else False
    if config.get("enable_wandb"):
        if config.get("wandb_token"):
            os.environ["WANDB_API_KEY"] = config["wandb_token"]
        os.environ["WANDB_PROJECT"] = config.get("wandb_project") or "unsloth-training"
    arguments = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": float(config.get("learning_rate") or 8e-4),
        "weight_decay": float(config.get("weight_decay") or 0.0),
        "lr_scheduler_type": config.get("lr_scheduler_type") or "cosine",
        "optim": config.get("optim") or "adamw_torch",
        "max_grad_norm": float(config.get("max_grad_norm") or 1.0),
        "seed": seed,
        "fp16": torch.cuda.is_available() and not bf16,
        "bf16": bf16,
        "logging_steps": 1,
        "report_to": ["wandb"] if config.get("enable_wandb") else "none",
        "disable_tqdm": _hf_stdout_progress_disabled(),
        "warmup_steps": int(config.get("warmup_steps") or 0)
        or round(float(config.get("warmup_ratio") or 0.0) * total_steps),
    }
    if max_steps > 0:
        arguments["max_steps"] = max_steps
    else:
        arguments["num_train_epochs"] = epochs
    if eval_items and evaluation_enabled(config.get("eval_steps")):
        arguments["eval_strategy"] = "steps"
        arguments["eval_steps"] = float(config["eval_steps"])
    apply_save_strategy(arguments, 0)

    trainer = DecisionTrainer(
        model = model,
        args = TrainingArguments(**arguments),
        train_dataset = items,
        eval_dataset = eval_items or None,
        processing_class = tokenizer,
        callbacks = [
            _create_embedding_progress_callback(
                event_queue,
                total_steps = total_steps,
                training_start_time = time.time(),
                should_stop = lambda: stop["requested"],
            )
        ],
    )
    _drop_hf_stdout_callbacks(trainer)

    base_metrics = None
    if eval_items:
        status(f"Evaluating the base model on {len(eval_items):,} held-out decisions...")
        base_metrics = FastDecisionModel.evaluate(trainer.model, tokenizer, eval_items)
        logger.info("Base held-out metrics: %s", base_metrics)
    if stopped_before_training():
        return

    start = time.time()
    trainer.train()
    if stop["requested"] and not stop["save"]:
        send("complete", output_dir = None, status_message = "Training cancelled")
        return

    tuned_metrics = None
    if eval_items:
        status("Calibrating confidence...")
        tuned_metrics = FastDecisionModel.calibrate(trainer.model, tokenizer, eval_items)
        logger.info("Fine-tuned held-out metrics: %s", tuned_metrics)

    status("Saving model...")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    metadata = {
        "base": model_name,
        "subfolder": subfolder,
        "method": "lora" if use_lora else "full",
        "objective": "soft_cross_entropy",
        "dataset": config.get("hf_dataset")
        or [Path(path).name for path in config.get("local_datasets") or []],
        "steps": trainer.state.global_step,
        "epochs": round(trainer.state.epoch or 0, 2),
        "heldout_decisions": len(eval_items or []),
        "heldout_accuracy_base": base_metrics and round(base_metrics["accuracy"], 4),
        "heldout_accuracy": tuned_metrics and round(tuned_metrics["accuracy"], 4),
        "heldout_ece_base": base_metrics and round(base_metrics["ece"], 4),
        "heldout_ece": tuned_metrics and round(tuned_metrics["ece"], 4),
        "seconds": round(time.time() - start, 1),
        "peak_memory_gb": peak_gb and round(peak_gb, 2),
        "date": datetime.now(timezone.utc).isoformat(timespec = "seconds"),
    }
    FastDecisionModel.save_pretrained(trainer.model, tokenizer, output_dir, metadata)
    logger.info("Decision model saved to %s: %s", output_dir, metadata)

    message = "Decision training completed"
    if base_metrics and tuned_metrics and len(eval_items) >= MIN_REPORTED_ITEMS:
        message = (
            f"Held-out accuracy {base_metrics['accuracy']:.2f} -> {tuned_metrics['accuracy']:.2f}, "
            f"calibration error {tuned_metrics['ece']:.2f}"
        )
    send("complete", output_dir = output_dir, status_message = message)
