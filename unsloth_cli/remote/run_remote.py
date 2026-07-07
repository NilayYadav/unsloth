# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Orchestration for `unsloth train --remote`: upload → submit → attach → pull."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console

from unsloth_cli.config import Config
from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.agent import AgentClient, RemoteBusyError
from unsloth_cli.remote.jobs import JobRecord, save_job, utc_now
from unsloth_cli.remote.payload import (
    build_training_payload,
    remote_dataset_dir,
    training_payload_warnings,
)
from unsloth_cli.remote.progress import attach_progress
from unsloth_cli.remote.registry import (
    ADAPTER_SUBDIR,
    job_registry_dir,
    register_artifact,
)
from unsloth_cli.remote.ssh import SSHRunner
from unsloth_cli.remote.state import RemoteRecord, get_remote

# Under the agent's $STUDIO_HOME; keep in sync with payload.remote_dataset_dir.
_REMOTE_DATASETS_ROOT = "~/.unsloth/studio/assets/datasets"


def _upload_local_datasets(
    runner: SSHRunner, console: Console, local_files: List[str],
) -> List[str]:
    """rsync local dataset files to the agent's uploads dir; return agent-relative paths."""
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
    """Pull the job's adapter into the local registry and register it."""
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
    console.print(f"Pulling adapter from {record.name}:{job.output_dir} ...")
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
    """Submit a training job to a remote and (unless detached) attach to it.

    Returns (job_id, pulled_adapter_path_or_None). Raises RemoteError on
    submission problems; raises SystemExit-like typer flows in the command.
    """
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

        # Confirm the terminal state against the durable run record.
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
            f"{job.final_loss if job.final_loss is not None else '?'} "
            f"at step {job.final_step if job.final_step is not None else '?'}."
        )
        adapter_path = pull_job_artifact(record, job, console)
        console.print(f"Adapter pulled to [bold]{adapter_path}[/bold]")
        console.print(f"Chat with it: [bold]unsloth chat {job_id}[/bold]")
        return job_id, adapter_path
    finally:
        tunnel.close()
