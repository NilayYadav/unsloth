# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from typing import Literal, Optional, List

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset: Optional[str] = Field(default = None, description = "Hugging Face dataset name.")
    local_dataset: Optional[List[str]] = Field(
        default = None,
        description = "Local dataset file or directory; repeat the option for multiple paths.",
    )
    format_type: Literal["auto", "alpaca", "chatml", "sharegpt"] = Field(
        default = "auto", description = "Dataset prompt format."
    )


class TrainingConfig(BaseModel):
    training_type: Literal["lora", "full"] = Field(
        default = "lora", description = "Fine-tuning method."
    )
    max_seq_length: int = Field(default = 2048, description = "Maximum sequence length in tokens.")
    load_in_4bit: bool = Field(default = True, description = "Load the model in 4-bit precision.")
    output_dir: Path = Field(
        default = Path("./outputs"), description = "Directory for checkpoints and logs."
    )
    num_epochs: int = Field(
        default = 3, description = "Number of passes over the training dataset."
    )
    learning_rate: float = Field(default = 2e-4, description = "Optimizer learning rate.")
    batch_size: int = Field(default = 2, description = "Training examples per device batch.")
    gradient_accumulation_steps: int = Field(
        default = 4, description = "Batches to accumulate before each optimizer step."
    )
    warmup_steps: int = Field(default = 5, description = "Learning-rate warmup steps.")
    max_steps: int = Field(default = 0, description = "Maximum optimizer steps; 0 uses num_epochs.")
    save_steps: int = Field(
        default = 0, description = "Save a checkpoint every N steps; 0 disables periodic saves."
    )
    weight_decay: float = Field(default = 0.01, description = "Optimizer weight decay.")
    random_seed: int = Field(default = 3407, description = "Random seed for reproducible training.")
    packing: bool = Field(
        default = False, description = "Pack multiple short examples into each sequence."
    )
    train_on_completions: bool = Field(
        default = False, description = "Train only on assistant completions in chat datasets."
    )
    gradient_checkpointing: Literal["unsloth", "true", "none"] = Field(
        default = "unsloth", description = "Gradient checkpointing mode."
    )


class LoraConfig(BaseModel):
    lora_r: int = Field(default = 64, description = "LoRA rank.")
    lora_alpha: int = Field(default = 16, description = "LoRA scaling factor.")
    lora_dropout: float = Field(default = 0.0, description = "Dropout applied to LoRA layers.")
    target_modules: str = Field(
        default = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        description = "Comma-separated model modules to adapt with LoRA.",
    )
    vision_all_linear: bool = Field(
        default = False, description = "Adapt all linear layers in vision models."
    )
    use_rslora: bool = Field(default = False, description = "Use rank-stabilized LoRA.")
    use_loftq: bool = Field(default = False, description = "Initialize LoRA weights with LoftQ.")
    finetune_vision_layers: bool = Field(
        default = True, description = "Fine-tune vision encoder layers."
    )
    finetune_language_layers: bool = Field(
        default = True, description = "Fine-tune language-model layers."
    )
    finetune_attention_modules: bool = Field(
        default = True, description = "Fine-tune attention modules."
    )
    finetune_mlp_modules: bool = Field(default = True, description = "Fine-tune MLP modules.")


class LoggingConfig(BaseModel):
    enable_wandb: bool = Field(
        default = False, description = "Log training metrics to Weights & Biases."
    )
    wandb_project: str = Field(
        default = "unsloth-training", description = "Weights & Biases project name."
    )
    wandb_token: Optional[str] = Field(default = None, description = "Weights & Biases API key.")
    enable_tensorboard: bool = Field(
        default = False, description = "Log training metrics to TensorBoard."
    )
    tensorboard_dir: str = Field(default = "runs", description = "Directory for TensorBoard logs.")
    hf_token: Optional[str] = Field(
        default = None, description = "Hugging Face token for private models or datasets."
    )


class Config(BaseModel):
    model: Optional[str] = Field(
        default = None, description = "Base model name or local path to fine-tune."
    )
    data: DataConfig = Field(default_factory = DataConfig)
    training: TrainingConfig = Field(default_factory = TrainingConfig)
    lora: LoraConfig = Field(default_factory = LoraConfig)
    logging: LoggingConfig = Field(default_factory = LoggingConfig)

    def apply_overrides(self, **kwargs):
        """Apply CLI overrides by matching arg names to config fields."""
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                for section in (self.data, self.training, self.lora, self.logging):
                    if hasattr(section, key):
                        setattr(section, key, value)
                        break

    def model_kwargs(self, use_lora: bool, is_vision: bool) -> dict:
        """Return kwargs for trainer.prepare_model_for_training()."""
        if use_lora and is_vision:
            # Vision models expect a string (e.g. "all-linear"); None uses trainer defaults
            target_modules = "all-linear" if self.lora.vision_all_linear else None
        else:
            parsed = [
                m.strip() for m in str(self.lora.target_modules).split(",") if m and m.strip()
            ]
            target_modules = parsed or None

        return {
            "use_lora": use_lora,
            "finetune_vision_layers": self.lora.finetune_vision_layers,
            "finetune_language_layers": self.lora.finetune_language_layers,
            "finetune_attention_modules": self.lora.finetune_attention_modules,
            "finetune_mlp_modules": self.lora.finetune_mlp_modules,
            "target_modules": target_modules,
            "lora_r": self.lora.lora_r,
            "lora_alpha": self.lora.lora_alpha,
            "lora_dropout": self.lora.lora_dropout,
            "use_gradient_checkpointing": self.training.gradient_checkpointing,
            "use_rslora": self.lora.use_rslora,
            "use_loftq": self.lora.use_loftq,
        }

    def training_kwargs(self) -> dict:
        """Return kwargs for trainer.start_training()."""
        return {
            "output_dir": str(self.training.output_dir),
            "num_epochs": self.training.num_epochs,
            "learning_rate": self.training.learning_rate,
            "batch_size": self.training.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "warmup_steps": self.training.warmup_steps,
            "max_steps": self.training.max_steps,
            "save_steps": self.training.save_steps,
            "weight_decay": self.training.weight_decay,
            "random_seed": self.training.random_seed,
            "packing": self.training.packing,
            "train_on_completions": self.training.train_on_completions,
            "max_seq_length": self.training.max_seq_length,
            "enable_wandb": self.logging.enable_wandb,
            "wandb_project": self.logging.wandb_project,
            "wandb_token": self.logging.wandb_token,
            "enable_tensorboard": self.logging.enable_tensorboard,
            "tensorboard_dir": self.logging.tensorboard_dir,
        }


def load_config(path: Optional[Path]) -> Config:
    """Load config from YAML/JSON file, or return defaults if no path given."""
    if not path:
        return Config()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding = "utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        import json
        data = json.loads(text or "{}")

    return Config(**data)
