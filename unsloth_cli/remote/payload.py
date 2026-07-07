# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mapping from the CLI training Config to the agent's TrainingStartRequest payload."""

from __future__ import annotations

from typing import List, Optional

from unsloth_cli.config import Config
from unsloth_cli.remote import RemoteError

_TRAINING_TYPE_MAP = {
    "lora": "LoRA/QLoRA",
    "full": "Full Finetuning",
}

# Relative to the agent's dataset roots; resolve_dataset_path() on the server
# maps it under $STUDIO_HOME/assets/datasets/.
REMOTE_UPLOADS_SUBDIR = "uploads/remote"


def remote_dataset_dir(tag: str) -> str:
    return f"{REMOTE_UPLOADS_SUBDIR}/{tag}"


def build_training_payload(
    cfg: Config,
    *,
    project_name: Optional[str] = None,
    local_dataset_remote_paths: Optional[List[str]] = None,
    hf_token: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    """Build a TrainingStartRequest-shaped dict from a CLI Config.

    `local_dataset_remote_paths` are the agent-relative paths the datasets were
    uploaded to (see remote_dataset_dir); they replace cfg.data.local_dataset.
    """
    if not cfg.model:
        raise RemoteError(
            "No model specified.",
            hint = "Set `model:` in the config file or pass --model.",
        )
    training_type = _TRAINING_TYPE_MAP.get(cfg.training.training_type)
    if training_type is None:
        raise RemoteError(
            f"Unsupported training_type for remote runs: {cfg.training.training_type!r}"
        )

    target_modules = [
        m.strip() for m in str(cfg.lora.target_modules).split(",") if m and m.strip()
    ]

    payload = {
        "model_name": cfg.model,
        "project_name": project_name,
        "training_type": training_type,
        "use_lora": cfg.training.training_type == "lora",
        "hf_token": hf_token,
        "load_in_4bit": cfg.training.load_in_4bit,
        "max_seq_length": cfg.training.max_seq_length,
        "hf_dataset": cfg.data.dataset,
        "local_datasets": local_dataset_remote_paths or [],
        "format_type": cfg.data.format_type,
        "num_epochs": cfg.training.num_epochs,
        # The request schema types learning_rate as a string.
        "learning_rate": str(cfg.training.learning_rate),
        "batch_size": cfg.training.batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "warmup_steps": cfg.training.warmup_steps,
        # CLI uses 0 for "no step cap"; the request expects None for that.
        "max_steps": cfg.training.max_steps or None,
        "save_steps": cfg.training.save_steps,
        "weight_decay": cfg.training.weight_decay,
        "random_seed": cfg.training.random_seed,
        "packing": cfg.training.packing,
        "train_on_completions": cfg.training.train_on_completions,
        "gradient_checkpointing": cfg.training.gradient_checkpointing,
        "lora_r": cfg.lora.lora_r,
        "lora_alpha": cfg.lora.lora_alpha,
        "lora_dropout": cfg.lora.lora_dropout,
        "target_modules": target_modules,
        "use_rslora": cfg.lora.use_rslora,
        "use_loftq": cfg.lora.use_loftq,
        "finetune_vision_layers": cfg.lora.finetune_vision_layers,
        "finetune_language_layers": cfg.lora.finetune_language_layers,
        "finetune_attention_modules": cfg.lora.finetune_attention_modules,
        "finetune_mlp_modules": cfg.lora.finetune_mlp_modules,
        "enable_wandb": cfg.logging.enable_wandb,
        "wandb_project": cfg.logging.wandb_project,
        "wandb_token": cfg.logging.wandb_token,
        "enable_tensorboard": cfg.logging.enable_tensorboard,
        "tensorboard_dir": cfg.logging.tensorboard_dir,
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    return {k: v for k, v in payload.items() if v is not None}


def training_payload_warnings(cfg: Config) -> List[str]:
    warnings = []
    if cfg.training.save_steps == 0:
        warnings.append(
            "save_steps is 0: no periodic checkpoints will be written, so this job "
            "cannot be resumed after a stop or crash. Set save_steps in the config "
            "to make it resumable."
        )
    from pathlib import Path
    if Path(cfg.training.output_dir) != Path("./outputs"):
        warnings.append(
            "training.output_dir is ignored for remote runs; the agent chooses the "
            "output directory and the adapter is pulled back automatically."
        )
    return warnings
