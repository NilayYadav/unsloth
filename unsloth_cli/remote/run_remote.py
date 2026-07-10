# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# `unsloth train --remote`: payload build, upload, submit, attach, pull.

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from unsloth_cli.config import Config
from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.ssh import AgentClient
from unsloth_cli.remote.ssh import SSHRunner
from unsloth_cli.remote.state import (
    ADAPTER_SUBDIR,
    JobRecord,
    RemoteRecord,
    get_remote,
    job_registry_dir,
    register_artifact,
    save_job,
    utc_now,
)

_REMOTE_DATASETS_ROOT = "~/.unsloth/studio/assets/datasets"


_TRAINING_TYPE_MAP = {
    "lora": "LoRA/QLoRA",
    "full": "Full Finetuning",
}

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


def _format_eta(eta_seconds) -> str:
    if not eta_seconds or eta_seconds < 0:
        return "--:--"
    eta = int(eta_seconds)
    hours, remainder = divmod(eta, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def attach_progress(
    client: AgentClient,
    console: Console,
    job_id: str,
    last_event_id: Optional[str] = None,
) -> Tuple[str, dict]:
    # Returns ("completed" | "error", last_payload); Ctrl-C propagates.
    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[metrics]}"),
        TimeElapsedColumn(),
    )
    outcome = "error"
    last_payload: dict = {}
    with Progress(*columns, console = console, transient = False) as progress:
        task_id = progress.add_task(
            f"[bold]{job_id}[/bold] starting...", total = None, metrics = "",
        )
        for event in client.stream_progress(last_event_id = last_event_id):
            data = event.data
            if event.event == "heartbeat":
                message = data.get("message")
                if message:
                    progress.update(task_id, description = f"[bold]{job_id}[/bold] {message}")
                continue
            if event.event == "progress":
                last_payload = data
                step = data.get("step") or 0
                total = data.get("total_steps") or None
                loss = data.get("loss")
                lr = data.get("learning_rate")
                epoch = data.get("epoch")
                metrics = " ".join(filter(None, [
                    f"loss {loss:.4f}" if loss is not None else None,
                    f"lr {lr:.2e}" if lr else None,
                    f"epoch {epoch:.2f}" if epoch is not None else None,
                    f"eta {_format_eta(data.get('eta_seconds'))}",
                ]))
                progress.update(
                    task_id,
                    description = f"[bold]{job_id}[/bold] training",
                    completed = step,
                    total = total,
                    metrics = metrics,
                )
                continue
            if event.event == "complete":
                last_payload = data or last_payload
                progress.update(task_id, description = f"[bold]{job_id}[/bold] [green]completed[/green]")
                outcome = "completed"
                break
            if event.event == "error":
                last_payload = data or last_payload
                progress.update(task_id, description = f"[bold]{job_id}[/bold] [red]error[/red]")
                outcome = "error"
                break
    return outcome, last_payload


def _upload_local_datasets(
    runner: SSHRunner, console: Console, local_files: List[str],
) -> List[str]:
    files = [Path(f).expanduser() for f in local_files]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise RemoteError(f"Local dataset file(s) not found: {', '.join(missing)}")
    tag = f"up_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    remote_relative_dir = remote_dataset_dir(tag)
    remote_absolute_dir = f"{_REMOTE_DATASETS_ROOT}/{remote_relative_dir}"
    runner.run(f"mkdir -p {remote_absolute_dir}")
    for f in files:
        console.print(f"  uploading {f.name}...")
        runner.rsync_up(str(f), f"{remote_absolute_dir}/")
    return [f"{remote_relative_dir}/{f.name}" for f in files]


def pull_job_artifact(
    record: RemoteRecord, job: JobRecord, console: Console, force: bool = False,
) -> Path:
    if not job.output_dir:
        raise RemoteError(
            f"Job {job.job_id} has no known output directory yet.",
            hint = "Refresh with: unsloth jobs",
        )
    destination = job_registry_dir(job.job_id) / ADAPTER_SUBDIR
    if destination.exists() and job.pulled and not force:
        console.print(f"Already pulled to {destination} (use --force to re-pull).")
        return destination
    destination.mkdir(parents = True, exist_ok = True)
    runner = SSHRunner.for_remote(record)
    console.print(f"Pulling adapter from {record.name}...")
    runner.rsync_down(
        f"{job.output_dir.rstrip('/')}/",
        f"{destination}/",
        excludes = ("checkpoint-*", "runs", "*.pth"),
    )
    register_artifact(job.job_id, base_model = job.model_name, remote = record.name)
    job.pulled = True
    job.registry_path = str(destination)
    save_job(job)
    return destination


def _refresh_job_from_run(job: JobRecord, run: dict) -> JobRecord:
    job.status = run.get("status") or job.status
    job.output_dir = run.get("output_dir") or job.output_dir
    job.model_name = run.get("model_name") or job.model_name
    job.final_step = run.get("final_step", job.final_step)
    job.total_steps = run.get("total_steps", job.total_steps)
    job.final_loss = run.get("final_loss", job.final_loss)
    return job


def run_remote_training(
    cfg: Config,
    remote_name: str,
    console: Console,
    config_file: Optional[Path] = None,
    detach: bool = False,
    hf_token: Optional[str] = None,
    wandb_token: Optional[str] = None,
) -> Tuple[str, Optional[Path]]:
    # Returns (job_id, pulled_adapter_path_or_None).
    record = get_remote(remote_name)
    runner = SSHRunner.for_remote(record)

    for warning in training_payload_warnings(cfg):
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    remote_paths: Optional[List[str]] = None
    if cfg.data.local_dataset:
        console.print(f"Uploading {len(cfg.data.local_dataset)} local dataset file(s)...")
        remote_paths = _upload_local_datasets(runner, console, cfg.data.local_dataset)

    if wandb_token:
        cfg.logging.wandb_token = wandb_token
    payload = build_training_payload(
        cfg,
        project_name = config_file.stem if config_file else None,
        local_dataset_remote_paths = remote_paths,
        hf_token = hf_token,
    )

    tunnel = runner.open_tunnel(record.agent_port)
    try:
        client = AgentClient(tunnel.url, api_key = record.api_key)
        response = client.start_training(payload)
        job_id = response["job_id"]
        console.print(f"Submitted [bold]{job_id}[/bold] to {record.name} ({record.destination}).")

        job = JobRecord(
            job_id = job_id,
            remote = record.name,
            kind = "train",
            submitted_at = utc_now(),
            config_file = str(config_file) if config_file else None,
            payload = dict(payload),
            uploaded_datasets = remote_paths or [],
            status = "running",
            model_name = cfg.model,
        )
        save_job(job)

        if detach:
            console.print(
                f"Detached. Follow with: [bold]unsloth logs {job_id} -f[/bold]  |  "
                f"cancel with: [bold]unsloth cancel {job_id}[/bold]"
            )
            return job_id, None

        try:
            outcome, _ = attach_progress(client, console, job_id)
        except KeyboardInterrupt:
            console.print(
                f"\nDetached — [bold]{job_id}[/bold] keeps running on {record.name}.\n"
                f"Reattach: [bold]unsloth logs {job_id} -f[/bold]  |  "
                f"cancel: [bold]unsloth cancel {job_id}[/bold]"
            )
            return job_id, None

        run_detail = client.run_detail(job_id)
        run = run_detail.get("run") or {}
        _refresh_job_from_run(job, run)
        save_job(job)

        if outcome != "completed" or job.status == "error":
            message = run.get("error_message") or "Training failed on the remote."
            raise RemoteError(
                message,
                hint = f"Full logs: unsloth logs {job_id} --raw",
            )

        console.print(
            f"[green]Training completed[/green] — final loss "
            f"{f'{job.final_loss:.4f}' if job.final_loss is not None else '?'} "
            f"at step {job.final_step if job.final_step is not None else '?'}."
        )
        adapter_path = pull_job_artifact(record, job, console)
        console.print(f"Adapter pulled to [bold]{adapter_path}[/bold]")
        console.print(f"Chat with it: [bold]unsloth chat {job_id}[/bold]")
        return job_id, adapter_path
    finally:
        tunnel.close()
