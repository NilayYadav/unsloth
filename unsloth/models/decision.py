# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "FastDecisionModel",
    "DecisionTrainer",
    "DecisionDataCollator",
]

import json
import os
import random
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import Trainer

TRAIN_MAX_LEN, TRAIN_HEAD_MAX_LEN = 1024, 256
HOLDOUT_MAX = 400
MIN_CALIBRATION_ITEMS = 10
HEAD_LR_SCALE = 4.0
LORA_HEAD_LR = 1e-4
QUESTION_TYPES = ("choice", "score", "noul")
_FILES = ("rl_agent_config.json", "model.safetensors")
_DIRS = ("encoder", "tokenizer")


class DecisionDataError(ValueError):
    pass


def _laya():
    try:
        import laya
    except ImportError as exc:
        raise ImportError(
            "Unsloth: decision models need the `laya` package: pip install laya"
        ) from exc
    return laya


def is_decision_checkpoint(folder) -> bool:
    folder = Path(folder)
    return all((folder / name).is_file() for name in _FILES) and all(
        (folder / name).is_dir() for name in _DIRS
    )


def _checkpoint_folder(model_name, subfolder, token, local_files_only) -> Path:
    root = Path(model_name).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        prefix = f"{subfolder}/" if subfolder else ""
        root = Path(
            snapshot_download(
                model_name,
                token = token,
                local_files_only = local_files_only,
                allow_patterns = [prefix + name for name in _FILES]
                + [f"{prefix}{name}/*" for name in _DIRS],
            )
        )
    folder = root / subfolder if subfolder else root
    if not is_decision_checkpoint(folder):
        raise ValueError(
            f"Unsloth: {folder} is not a decision model checkpoint "
            "(rl_agent_config.json, model.safetensors, encoder/ and tokenizer/)."
        )
    return folder


def _amp_dtype(device):
    if device.type == "cuda":
        from ._utils import is_bfloat16_supported
        return torch.bfloat16 if is_bfloat16_supported() else torch.float16
    return torch.bfloat16 if device.type == "xpu" else None


def _parsed(value):
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _internal(question) -> dict:
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        raise DecisionDataError("is not a valid question")
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice" and not (isinstance(criteria, (dict, list)) and criteria):
        raise DecisionDataError("needs criteria naming its options")
    if kind == "score" and not (isinstance(criteria, list) and criteria):
        raise DecisionDataError("needs a list of criteria levels")
    if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
        raise DecisionDataError('criteria may only have "true" and "false"')
    laya_question = {"type": kind, "instructions": question.get("instructions") or ""}
    if criteria is not None:
        laya_question["criteria"] = criteria
    return _laya().agent.Agent._to_internal(laya_question)


def _option_keys(internal: dict) -> list:
    if internal["t"] == "choice":
        return [str(key) for key in internal["crit"]]
    if internal["t"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(internal["crit"]))]


def _target(internal: dict, gold) -> tuple:
    keys = _option_keys(internal)
    label = gold.get("label") if isinstance(gold, dict) else gold
    if isinstance(label, bool) or (internal["t"] == "noul" and isinstance(label, str)):
        label = str(label).lower()
    elif isinstance(label, (int, float)) and internal["t"] == "score":
        label = str(round(label))
    elif label is not None:
        label = str(label)
    probabilities = gold.get("probabilities") if isinstance(gold, dict) else None
    if internal["t"] == "noul" and isinstance(gold, dict):
        noul = gold.get("noul")
        if not isinstance(probabilities, dict) and isinstance(noul, (int, float)):
            probabilities = {"true": noul}
        if isinstance(probabilities, dict) and len(probabilities.keys() & {"false", "true"}) == 1:
            known = "true" if "true" in probabilities else "false"
            try:
                probabilities = {
                    known: float(probabilities[known]),
                    ({"false", "true"} - {known}).pop(): 1.0 - float(probabilities[known]),
                }
            except (TypeError, ValueError):
                probabilities = None
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


def _soft_cross_entropy(logits, target, mask):
    logits = logits.float().masked_fill(~mask, -1e4)
    return -(target * torch.log_softmax(logits, -1)).sum(-1).mean()


class DecisionDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: list) -> dict:
        rows, length = len(items), max(len(item["input_ids"]) for item in items)
        options = max(len(item["markers"]) for item in items)
        batch = {
            "input_ids": torch.full((rows, length), self.pad_token_id, dtype = torch.long),
            "attention_mask": torch.zeros((rows, length), dtype = torch.long),
            "marker_pos": torch.zeros((rows, options), dtype = torch.long),
            "marker_mask": torch.zeros((rows, options), dtype = torch.bool),
            "qtype": torch.tensor([item["qtype"] for item in items]),
            "target": torch.zeros((rows, options), dtype = torch.float32),
        }
        for i, item in enumerate(items):
            ids, markers = item["input_ids"], item["markers"]
            batch["input_ids"][i, : len(ids)] = torch.tensor(ids)
            batch["attention_mask"][i, : len(ids)] = 1
            batch["marker_pos"][i, : len(markers)] = torch.tensor(markers)
            batch["marker_mask"][i, : len(markers)] = True
            batch["target"][i, : len(item["target"])] = torch.tensor(item["target"])
        return batch


class _LengthGroupedBatches(torch.utils.data.Sampler):
    # Micro-batches of similar length in shuffled order, so each optimizer step still mixes lengths
    # (and with them question types); the longest goes first so an out-of-memory shows at once.
    def __init__(self, lengths: list, batch_size: int, seed: int):
        self.lengths, self.batch_size, self.seed, self.epoch = lengths, batch_size, seed, 0

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        from transformers.trainer_pt_utils import get_length_grouped_indices

        seed = self.seed + self.epoch
        self.epoch += 1
        order = get_length_grouped_indices(
            self.lengths, self.batch_size, generator = torch.Generator().manual_seed(seed)
        )
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        short = batches.pop() if len(batches) > 1 and len(batches[-1]) < self.batch_size else []
        rest = batches[1:]
        random.Random(seed).shuffle(rest)
        return iter([index for batch in batches[:1] + rest + [short] for index in batch])


class DecisionTrainer(Trainer):
    def __init__(
        self,
        *args,
        head_learning_rate: Optional[float] = None,
        **kwargs,
    ):
        self.head_learning_rate = head_learning_rate
        args_ = kwargs.get("args") or (args[1] if len(args) > 1 else None)
        if args_ is not None:
            args_.remove_unused_columns = False
            args_.prediction_loss_only = True
            if args_.label_names is None:
                args_.label_names = ["target"]
        if kwargs.get("data_collator") is None and kwargs.get("processing_class") is not None:
            kwargs["data_collator"] = DecisionDataCollator(kwargs["processing_class"].pad_token_id)
        super().__init__(*args, **kwargs)
        # Trainer already divides this loss by the accumulation steps and accelerate's backward divides
        # again, which leaves every update 1/accumulation of the mean gradient; undo the second division.
        backward = self.accelerator.backward

        def _backward(loss, **backward_kwargs):
            from accelerate.utils import DistributedType
            if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
                loss = loss * self.accelerator.gradient_accumulation_steps
            return backward(loss, **backward_kwargs)

        self.accelerator.backward = _backward

    def _get_train_sampler(self, *args, **kwargs):
        dataset = (args[0] if args else kwargs.get("train_dataset")) or self.train_dataset
        if dataset is None:
            return None
        return _LengthGroupedBatches(
            [len(item["input_ids"]) for item in dataset], self.args.train_batch_size, self.args.seed
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        target = inputs.pop("target")
        logits, _ = model(**inputs)
        loss = _soft_cross_entropy(logits, target, inputs["marker_mask"])
        return (loss, (logits,)) if return_outputs else loss

    def create_optimizer(self, model = None):
        if self.optimizer is not None:
            return self.optimizer
        model = model or self.model
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        lora = any("lora_" in n for n, _ in trainable)
        head_lr = self.head_learning_rate or (
            LORA_HEAD_LR if lora else self.args.learning_rate * HEAD_LR_SCALE
        )
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)
        groups = [
            {"params": [p for n, p in trainable if n.startswith("encoder.")]},
            {"params": [p for n, p in trainable if not n.startswith("encoder.")], "lr": head_lr},
        ]
        for group in groups:
            group["weight_decay"] = self.args.weight_decay
        self.optimizer = optimizer_cls([g for g in groups if g["params"]], **optimizer_kwargs)
        return self.optimizer


@torch.no_grad()
def _logits(
    model,
    items: list,
    pad_token_id: int,
    batch_size: int = 16,
) -> list:
    device = next(model.parameters()).device
    amp_dtype = _amp_dtype(device)
    collate = DecisionDataCollator(pad_token_id)
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = collate(chunk)
        batch.pop("target")
        with torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
            logits, _ = model(**{k: v.to(device) for k, v in batch.items()})
        logits = logits.float().cpu()
        out.extend(logits[row, : len(item["markers"])] for row, item in enumerate(chunk))
    model.train(was_training)
    return out


def _metrics(logits, items, temperatures) -> dict:
    import numpy as np

    conf, correct, loss = [], [], []
    for z, item, temperature in zip(logits, items, temperatures):
        log_p = torch.log_softmax(z / temperature, -1)
        conf.append(float(log_p.exp().max()))
        correct.append(float(int(log_p.argmax()) == item["label"]))
        loss.append(float(-(torch.tensor(item["target"]) * log_p).sum()))
    return {
        "accuracy": float(np.mean(correct)),
        "ece": _laya().common.ece_score(np.array(conf), np.array(correct)),
        "loss": float(np.mean(loss)),
    }


def _fit_temperature(logits, items) -> float:
    options = max(len(z) for z in logits)
    z = torch.full((len(logits), options), -1e4)
    target = torch.zeros((len(logits), options))
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


def _fit_temperatures(logits, items, indices, fallback: list) -> tuple:
    clamp = _laya().common.clamp_temperature
    temperature, fitted = list(fallback), set()
    for qtype in range(3):
        chosen = [i for i in indices if items[i]["qtype"] == qtype]
        if len(chosen) >= MIN_CALIBRATION_ITEMS:
            temperature[qtype] = clamp(
                _fit_temperature([logits[i] for i in chosen], [items[i] for i in chosen])
            )
            fitted.add(qtype)
    return temperature, fitted


def _served_temperatures(config: dict, logits, items) -> list:
    common = _laya().common
    per_type = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
    buckets = {
        key: common.clamp_temperature(value)
        for key, value in (config.get("temperature_by_options") or {}).items()
    }
    return [
        buckets.get(common.temp_bucket(item["qtype"], len(z)), per_type[item["qtype"]])
        for z, item in zip(logits, items)
    ]


def _enable_gradient_checkpointing(model, mode) -> None:
    mode = str(mode).lower() if mode not in (None, False, True) else mode
    if mode in (None, False, "none", "false", ""):
        return
    unsloth = mode == "unsloth"
    if unsloth and torch.cuda.is_available():
        from unsloth_zoo.gradient_checkpointing import patch_unsloth_smart_gradient_checkpointing
        patch_unsloth_smart_gradient_checkpointing(dtype = _amp_dtype(torch.device("cuda")))
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": unsloth}
    )
    if unsloth:
        model.encoder.enable_input_require_grads()


# Decision models in Laya's rl_agent_config.json layout: any encoder plus a typed decision head.
class FastDecisionModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        subfolder: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        full_finetuning: bool = False,
        use_gradient_checkpointing = "unsloth",
        token: Optional[str] = None,
        local_files_only: bool = False,
    ):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        laya = _laya()
        folder = _checkpoint_folder(model_name, subfolder, token, local_files_only)
        config = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
        laya.agent._fix_tokenizer_config(str(folder))
        tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"))
        model = laya.common.build_model(config, encoder_dir = str(folder / "encoder"))
        model.load_state_dict(load_file(str(folder / "model.safetensors")), strict = True)
        # ModernBERT would otherwise torch.compile parts of the encoder.
        model.encoder.config.reference_compile = False

        positions = int(getattr(model.encoder.config, "max_position_embeddings", TRAIN_MAX_LEN))
        wanted = max_seq_length or max(int(config.get("max_len", 512)), TRAIN_MAX_LEN)
        config["max_len"] = min(positions, int(wanted))
        config["head_max_len"] = min(
            config["max_len"] // 2, max(int(config.get("head_max_len", 192)), TRAIN_HEAD_MAX_LEN)
        )
        model.decision_config = config
        if full_finetuning:
            embeddings = model.encoder.get_input_embeddings()
            embeddings.weight.requires_grad_(False)
        _enable_gradient_checkpointing(model, use_gradient_checkpointing)
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r: int = 64,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        target_modules = "all-linear",
        bias: str = "none",
        random_state: int = 3407,
        use_rslora: bool = False,
    ):
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(random_state)
        model.encoder.requires_grad_(False)
        model.encoder = get_peft_model(
            model.encoder,
            LoraConfig(
                r = r,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                target_modules = target_modules,
                bias = bias,
                use_rslora = use_rslora,
            ),
        )
        return model

    @staticmethod
    def build_dataset(
        rows,
        tokenizer,
        model,
        validate: Optional[Callable[[str, dict], None]] = None,
    ) -> tuple:
        common = _laya().common
        max_len = int(model.decision_config.get("max_len", 512))
        head_max_len = int(model.decision_config.get("head_max_len", 192))
        items, report = [], {"total": 0, "skipped": 0, "reason": None}

        def skip(reason):
            report["skipped"] += 1
            report["reason"] = report["reason"] or reason

        for index, row in enumerate(rows):
            state = _parsed(row.get("state"))
            questions = _parsed(row.get("questions"))
            gold = _parsed(row["gold"] if row.get("gold") is not None else row.get("answers"))
            if state is None or not isinstance(questions, dict) or not isinstance(gold, dict):
                report["total"] += 1
                skip(f"row {index + 1} needs state, questions and gold")
                continue
            for name, question in questions.items():
                report["total"] += 1
                if name not in gold:
                    skip(f'row {index + 1} has no gold for "{name}"')
                    continue
                try:
                    if validate is not None:
                        validate(name, question)
                    internal = _internal(question)
                    target, label = _target(internal, _parsed(gold[name]))
                except ValueError as exc:
                    skip(f'row {index + 1}: "{name}" {exc}')
                    continue
                ids, markers = common.build_sequence(
                    tokenizer, state, internal, max_len, head_max_len
                )
                if len(markers) != len(common.render_options(internal)) or len(markers) != len(
                    target
                ):
                    skip(f'row {index + 1}: "{name}" options exceed the {max_len}-token context')
                    continue
                items.append(
                    {
                        "input_ids": ids,
                        "markers": markers,
                        "qtype": common.QTYPES[internal["t"]],
                        "target": target,
                        "label": label,
                        "row": index,
                    }
                )
        return items, report

    @staticmethod
    def split_holdout(
        items: list,
        seed: int = 3407,
        fraction: float = 0.1,
        max_items: int = HOLDOUT_MAX,
    ):
        target = min(max_items, int(len(items) * fraction))
        rows = sorted({item["row"] for item in items})
        random.Random(seed).shuffle(rows)
        sizes = {}
        for item in items:
            sizes[item["row"]] = sizes.get(item["row"], 0) + 1
        held, count = set(), 0
        for row in rows[:-1]:
            if count >= target:
                break
            held.add(row)
            count += sizes[row]
        return (
            [item for item in items if item["row"] not in held],
            [item for item in items if item["row"] in held],
        )

    @staticmethod
    def evaluate(model, tokenizer, items: list) -> dict:
        logits = _logits(model, items, tokenizer.pad_token_id)
        return _metrics(logits, items, _served_temperatures(model.decision_config, logits, items))

    @staticmethod
    def calibrate(model, tokenizer, items: list) -> dict:
        common = _laya().common
        config = model.decision_config
        fallback = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
        logits = _logits(model, items, tokenizer.pad_token_id)
        everything = range(len(items))
        temperature, fitted = _fit_temperatures(logits, items, everything, fallback)
        # Reported numbers score each half of the rows with temperatures fitted on the other half.
        half = {row: i % 2 for i, row in enumerate(sorted({item["row"] for item in items}))}
        per_item = [1.0] * len(items)
        for side in (0, 1):
            other = [i for i in everything if half[items[i]["row"]] != side]
            side_temperature, _ = _fit_temperatures(logits, items, other, fallback)
            for i in everything:
                if half[items[i]["row"]] == side:
                    per_item[i] = side_temperature[items[i]["qtype"]]
        config["temperature"] = temperature
        buckets = {
            key: value
            for key, value in (config.pop("temperature_by_options", None) or {}).items()
            if common.QTYPES.get(key.split(":")[0]) not in fitted
        }
        if buckets:
            config["temperature_by_options"] = buckets
        return {**_metrics(logits, items, per_item), "fitted_types": sorted(fitted)}

    @staticmethod
    def save_pretrained(
        model,
        tokenizer,
        save_directory,
        metadata: Optional[dict] = None,
    ) -> None:
        from safetensors.torch import save_file

        if hasattr(model.encoder, "merge_and_unload"):
            model.encoder = model.encoder.merge_and_unload()
        output = Path(save_directory)
        output.mkdir(parents = True, exist_ok = True)
        config_path = output / "rl_agent_config.json"
        config_path.unlink(missing_ok = True)
        save_file(
            {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()},
            str(output / "model.safetensors"),
        )
        model.encoder.config.save_pretrained(str(output / "encoder"))
        tokenizer.save_pretrained(str(output / "tokenizer"))
        _laya().agent._fix_tokenizer_config(str(output))
        config = {**model.decision_config, "fine_tuned": True}
        if metadata is not None:
            config["training"] = metadata
        # Written last: a folder with rl_agent_config.json is a complete checkpoint.
        partial = output / "rl_agent_config.json.tmp"
        partial.write_text(json.dumps(config, indent = 2), encoding = "utf-8")
        os.replace(partial, config_path)
