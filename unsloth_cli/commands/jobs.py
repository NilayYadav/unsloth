# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Top-level job commands for remote runs: jobs, logs, cancel, resume, pull."""

from __future__ import annotations

import os
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.agent import AgentClient
from unsloth_cli.remote.jobs import JobRecord, list_jobs, resolve_job, save_job, utc_now
from unsloth_cli.remote.progress import attach_progress
from unsloth_cli.remote.run_remote import _refresh_job_from_run, pull_job_artifact
from unsloth_cli.remote.ssh import SSHRunner
from unsloth_cli.remote.state import RemoteRecord, get_remote

_SERVER_LOG_GLOB = "~/.unsloth/studio/logs/server/server-*.log"
_ACTIVE_PHASES = ("training", "loading_model", "configuring")


def _fail(error: RemoteError) -> "typer.Exit":
    console = Console(stderr = True)
    console.print(f"[red]Error:[/red] {error}")
    if error.hint:
        console.print(f"[yellow]Hint:[/yellow] {error.hint}")
    return typer.Exit(code = 1)


def _connect(record: RemoteRecord):
    runner = SSHRunner.for_remote(record)
    tunnel = runner.open_tunnel(record.agent_port)
    return runner, tunnel, AgentClient(tunnel.url, api_key = record.api_key)


def _refresh_remote_jobs(records: list, console: Console) -> set:
    """Refresh local mirrors from each reachable remote; returns unreachable names."""
    unreachable = set()
    by_remote = {}
    for job in records:
        by_remote.setdefault(job.remote, []).append(job)
    for remote_name, jobs_for_remote in by_remote.items():
        try:
            record = get_remote(remote_name)
            _, tunnel, client = _connect(record)
            with tunnel:
                runs = {r.get("id"): r for r in client.list_runs(limit = 200).get("runs", [])}
        except RemoteError:
            unreachable.add(remote_name)
            continue
        for job in jobs_for_remote:
            run = runs.get(job.job_id)
            if run:
                _refresh_job_from_run(job, run)
                save_job(job)
    return unreachable


def jobs(
    remote: Optional[str] = typer.Option(None, "--remote", help = "Only this remote."),
    all_jobs: bool = typer.Option(False, "--all", help = "Show all jobs, not the latest 20."),
):
    """List remote training jobs (refreshes status from reachable remotes)."""
    console = Console()
    records = [j for j in list_jobs() if remote is None or j.remote == remote]
    if not records:
        console.print("No remote jobs yet. Start one with: unsloth train --config <cfg> --remote <name>")
        return
    unreachable = _refresh_remote_jobs(records, console)
    records.sort(key = lambda j: j.submitted_at or "", reverse = True)
    if not all_jobs:
        records = records[:20]

    table = Table()
    # Job ids must stay copyable: fold instead of ellipsis-truncate.
    table.add_column("Job", overflow = "fold")
    for column in ("Remote", "Status", "Model", "Step", "Loss", "Submitted", "Pulled"):
        table.add_column(column)
    for job in records:
        status = job.status
        if job.remote in unreachable:
            status = f"{status} (cached)"
        step = (
            f"{job.final_step}/{job.total_steps}"
            if job.final_step is not None and job.total_steps else "?"
        )
        loss = f"{job.final_loss:.4f}" if job.final_loss is not None else "?"
        table.add_row(
            job.job_id, job.remote, status, job.model_name or "?",
            step, loss, job.submitted_at or "?", "yes" if job.pulled else "no",
        )
    console.print(table)


def logs(
    job_id: str = typer.Argument(..., help = "Job id (or unique prefix)."),
    follow: bool = typer.Option(False, "--follow", "-f", help = "Keep streaming."),
    raw: bool = typer.Option(False, "--raw", help = "Raw server log instead of structured progress."),
):
    """Show a remote job's progress; reattaches to live runs."""
    console = Console()
    try:
        job = resolve_job(job_id)
        record = get_remote(job.remote)

        if raw:
            runner = SSHRunner.for_remote(record)
            tail_flags = "-n 200 -F" if follow else "-n 200"
            code = runner.stream(
                f'tail {tail_flags} "$(ls -t {_SERVER_LOG_GLOB} | head -1)"'
            )
            raise typer.Exit(code = 0 if code in (0, 130) else code)

        runner, tunnel, client = _connect(record)
        with tunnel:
            status = client.train_status()
            is_current = status.get("job_id") == job.job_id
            is_active = status.get("phase") in _ACTIVE_PHASES

            if is_current and is_active and follow:
                try:
                    attach_progress(client, console, job.job_id)
                except KeyboardInterrupt:
                    console.print(
                        f"\nDetached — [bold]{job.job_id}[/bold] keeps running on {record.name}."
                    )
                    raise typer.Exit(code = 0)

            run = client.run_detail(job.job_id).get("run") or {}
        _refresh_job_from_run(job, run)
        save_job(job)
        console.print(
            f"[bold]{job.job_id}[/bold] on {record.name}: {job.status}"
            + (f" — step {job.final_step}/{job.total_steps}" if job.final_step is not None else "")
            + (f", loss {job.final_loss:.4f}" if job.final_loss is not None else "")
        )
        if run.get("error_message"):
            console.print(f"[red]{run['error_message']}[/red]")
        if is_current and is_active and not follow:
            details = status.get("details") or {}
            console.print(
                f"Currently {status.get('phase')} at step {details.get('step', '?')} — "
                f"follow with: unsloth logs {job.job_id} -f"
            )
    except RemoteError as e:
        raise _fail(e)


def _wait_for_settled_run(client, job_id: str, timeout: float = 60.0) -> dict:
    """Poll the run record until the worker leaves 'running' (stop is async)."""
    deadline = time.monotonic() + timeout
    run = client.run_detail(job_id).get("run") or {}
    while run.get("status") == "running" and time.monotonic() < deadline:
        time.sleep(2)
        run = client.run_detail(job_id).get("run") or {}
    return run


def cancel(
    job_id: str = typer.Argument(..., help = "Job id (or unique prefix)."),
    no_save: bool = typer.Option(
        False, "--no-save", help = "Discard the checkpoint instead of saving it.",
    ),
):
    """Stop a running remote job (saves a resumable checkpoint by default)."""
    console = Console()
    try:
        job = resolve_job(job_id)
        record = get_remote(job.remote)
        _, tunnel, client = _connect(record)
        with tunnel:
            status = client.train_status()
            if status.get("job_id") != job.job_id or status.get("phase") not in _ACTIVE_PHASES:
                raise RemoteError(
                    f"{job.job_id} is not the active job on {record.name} "
                    f"(phase: {status.get('phase')}).",
                    hint = "Check with: unsloth jobs",
                )
            client.stop_training(save = not no_save)
            run = _wait_for_settled_run(client, job.job_id)
        _refresh_job_from_run(job, run)
        job.status = job.status if job.status != "running" else "stopped"
        save_job(job)
        console.print(
            f"Stopped [bold]{job.job_id}[/bold]"
            + ("." if no_save else f" — resumable with: unsloth resume {job.job_id}")
        )
    except RemoteError as e:
        raise _fail(e)


def resume(
    job_id: str = typer.Argument(..., help = "Job id (or unique prefix) to resume."),
    detach: bool = typer.Option(False, "--detach", help = "Submit and return immediately."),
):
    """Resume a stopped remote job from its last checkpoint (same remote)."""
    console = Console()
    try:
        job = resolve_job(job_id)
        record = get_remote(job.remote)
        if not job.payload:
            raise RemoteError(
                f"No stored config for {job.job_id}; cannot rebuild the request.",
            )
        _, tunnel, client = _connect(record)
        with tunnel:
            # A just-cancelled run needs a few seconds to checkpoint and settle.
            run = _wait_for_settled_run(client, job.job_id)
            _refresh_job_from_run(job, run)
            save_job(job)
            if not run.get("can_resume"):
                raise RemoteError(
                    f"{job.job_id} cannot be resumed (status: {run.get('status')}).",
                    hint = "Only stopped runs with saved checkpoints (save_steps > 0) resume.",
                )

            payload = dict(job.payload)
            payload["resume_from_checkpoint"] = job.output_dir
            # Secrets are never persisted in job records; re-inject from env.
            if os.environ.get("HF_TOKEN"):
                payload["hf_token"] = os.environ["HF_TOKEN"]
            if os.environ.get("WANDB_API_KEY"):
                payload["wandb_token"] = os.environ["WANDB_API_KEY"]

            response = client.start_training(payload)
            new_job_id = response["job_id"]
            new_job = JobRecord(
                job_id = new_job_id,
                remote = record.name,
                kind = "train",
                submitted_at = utc_now(),
                config_file = job.config_file,
                payload = dict(job.payload),
                uploaded_datasets = list(job.uploaded_datasets),
                status = "running",
                model_name = job.model_name,
                resumed_from = job.job_id,
            )
            save_job(new_job)
            console.print(f"Resumed as [bold]{new_job_id}[/bold] from {job.job_id}'s checkpoint.")

            if detach:
                console.print(f"Follow with: unsloth logs {new_job_id} -f")
                return
            try:
                attach_progress(client, console, new_job_id)
            except KeyboardInterrupt:
                console.print(f"\nDetached — [bold]{new_job_id}[/bold] keeps running.")
                return
            run = client.run_detail(new_job_id).get("run") or {}
        _refresh_job_from_run(new_job, run)
        save_job(new_job)
        if new_job.status == "completed":
            adapter = pull_job_artifact(record, new_job, console)
            console.print(f"Adapter pulled to [bold]{adapter}[/bold]")
    except RemoteError as e:
        raise _fail(e)


def pull(
    job_id: str = typer.Argument(..., help = "Job id (or unique prefix)."),
    force: bool = typer.Option(False, "--force", help = "Re-pull even if already pulled."),
):
    """Fetch a job's trained adapter into the local registry."""
    console = Console()
    try:
        job = resolve_job(job_id)
        record = get_remote(job.remote)
        if not job.output_dir:
            _, tunnel, client = _connect(record)
            with tunnel:
                run = client.run_detail(job.job_id).get("run") or {}
            _refresh_job_from_run(job, run)
            save_job(job)
        adapter = pull_job_artifact(record, job, console, force = force)
        console.print(f"Adapter at [bold]{adapter}[/bold] — try: unsloth chat {job.job_id}")
    except RemoteError as e:
        raise _fail(e)
