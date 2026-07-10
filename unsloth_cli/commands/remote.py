# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# `unsloth remote` plus the top-level job commands (jobs/logs/cancel/resume/pull).

from __future__ import annotations

import os
import re
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.bootstrap import (
    ProbeResult,
    initialize_agent_auth,
    run_bootstrap,
    run_probe,
)
from unsloth_cli.remote.run_remote import (
    _refresh_job_from_run,
    attach_progress,
    pull_job_artifact,
)
from unsloth_cli.remote.ssh import AgentClient, SSHRunner, pin_host_key, scan_host_key
from unsloth_cli.remote.state import (
    JobRecord,
    RemoteRecord,
    get_remote,
    list_jobs,
    load_remotes,
    remove_remote,
    resolve_job,
    save_job,
    upsert_remote,
    utc_now,
)

remote_app = typer.Typer(help = "Manage remote GPU machines for cloud training.")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SERVER_LOG_GLOB = "~/.unsloth/studio/logs/server/server-*.log"
_ACTIVE_PHASES = ("training", "loading_model", "configuring")


def _fail(error: RemoteError) -> "typer.Exit":
    console = Console(stderr = True)
    console.print(f"[red]Error:[/red] {error}")
    if error.hint:
        console.print(f"[yellow]Hint:[/yellow] {error.hint}")
    return typer.Exit(code = 1)


def _print_probe(console: Console, result: ProbeResult) -> None:
    problems = [c for c in result.checks if not c.ok]
    if not problems:
        by_name = {c.name: c for c in result.checks}
        caps = result.capabilities
        parts = [
            ", ".join(f"{n} ({v:g} GB)" for n, v in zip(caps.gpu_names, caps.vram_gb))
            or "no GPU",
            f"driver {caps.driver}" if caps.driver else None,
            f"{caps.disk_free_gb:g} GB free" if caps.disk_free_gb is not None else None,
            f"agent via {by_name['Agent supervision'].value}"
            if "Agent supervision" in by_name else None,
            by_name["System packages"].value if "System packages" in by_name else None,
        ]
        console.print("Probe ok: " + " · ".join(p for p in parts if p))
        return
    table = Table(title = "Remote machine probe", show_lines = False)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")
    for check in result.checks:
        status = "[green]ok[/green]" if check.ok else (
            "[red]FAIL[/red]" if check.fatal else "[yellow]warn[/yellow]"
        )
        table.add_row(check.name, status, check.value)
    console.print(table)
    for check in result.checks:
        if not check.ok and check.hint:
            style = "red" if check.fatal else "yellow"
            console.print(f"  [{style}]{check.name}:[/{style}] {check.hint}")


def _connect(record: RemoteRecord, runner: Optional[SSHRunner] = None):
    runner = runner or SSHRunner.for_remote(record)
    tunnel = runner.open_tunnel(record.agent_port)
    return runner, tunnel, AgentClient(tunnel.url, api_key = record.api_key)


def _existing_valid_key(name: str, agent_url: str) -> Optional[str]:
    existing = load_remotes().get(name)
    if existing is None or not existing.api_key:
        return None
    try:
        AgentClient(agent_url, api_key = existing.api_key).train_status()
    except RemoteError:
        return None
    return existing.api_key


def _resolve(identifier: Optional[str]) -> JobRecord:
    if identifier:
        return resolve_job(identifier)
    records = list_jobs()
    if not records:
        raise RemoteError(
            "No remote jobs yet.",
            hint = "Start one with: unsloth train --config <cfg> --remote <name>",
        )
    return max(records, key = lambda j: j.submitted_at or "")


@remote_app.command()
def add(
    name: str = typer.Argument(..., help = "Name for this remote (e.g. gpu1)."),
    destination: str = typer.Argument(..., help = "SSH destination, user@host."),
    key: Optional[str] = typer.Option(None, "--key", "-i", help = "SSH private key path."),
    port: int = typer.Option(22, "--port", "-p", help = "SSH port."),
):
    """Probe, bootstrap, and register a remote GPU machine.

    The machine only needs SSH access and an NVIDIA driver; everything else
    is installed into ~/.unsloth/env on the VM.
    """
    console = Console()
    if not _NAME_RE.match(name):
        raise _fail(RemoteError(f"Invalid remote name {name!r} (use letters, digits, - or _)."))
    if "@" not in destination:
        raise _fail(RemoteError(
            f"Destination {destination!r} must be user@host.",
            hint = "Example: unsloth remote add gpu1 ubuntu@203.0.113.7",
        ))

    try:
        host = destination.split("@", 1)[1]
        with console.status(f"Fetching host key of {host}..."):
            fingerprint, known_hosts_text = scan_host_key(host, port = port)
        pin_host_key(known_hosts_text)
        console.print(f"Pinned host key {fingerprint}")

        runner = SSHRunner(destination, port = port, identity_file = key)
        with console.status("Probing hardware and tooling..."):
            probe = run_probe(runner)
        _print_probe(console, probe)
        if probe.fatal_failures:
            raise RemoteError(
                "The machine is not ready (see failures above).",
                hint = "Fix the issues and re-run `unsloth remote add`.",
            )

        with console.status("Bootstrapping...") as status:
            result = run_bootstrap(
                runner, probe.capabilities, agent_port = 8877,
                on_step = lambda label: status.update(f"[bold]{label}[/bold]"),
            )
        console.print(
            f"Environment ready (unsloth {result.remote_unsloth_version or 'unknown'}, "
            f"{result.supervision} agent)."
        )

        with runner.open_tunnel(8877) as tunnel:
            api_key = _existing_valid_key(name, tunnel.url)
            if api_key is None:
                with console.status("Setting up agent access..."):
                    api_key = initialize_agent_auth(runner, AgentClient(tunnel.url))
            client = AgentClient(tunnel.url, api_key = api_key)
            client.health()
            status_payload = client.train_status()

        upsert_remote(RemoteRecord(
            name = name,
            host = host,
            user = destination.split("@", 1)[0],
            port = port,
            identity_file = key,
            host_key_fingerprint = fingerprint,
            api_key = api_key,
            supervision = result.supervision,
            unsloth_version = result.remote_unsloth_version,
            added_at = utc_now(),
            capabilities = probe.capabilities,
        ))
        busy = ""
        if status_payload.get("phase") in _ACTIVE_PHASES:
            busy = f" (currently busy: {status_payload.get('job_id') or 'a job'})"
        console.print(f"\n[green]Remote '{name}' is ready.[/green]{busy}")
        console.print(
            f"Try: [bold]unsloth train --config path/to/train.yaml --remote {name}[/bold]"
        )
    except RemoteError as e:
        raise _fail(e)


@remote_app.command("list")
def list_remotes():
    """List configured remotes."""
    console = Console()
    remotes = load_remotes()
    if not remotes:
        console.print("No remotes configured. Add one with: unsloth remote add <name> <user@host>")
        return
    table = Table()
    for column in ("Name", "Destination", "GPU", "Driver", "Added"):
        table.add_column(column)
    for name, record in sorted(remotes.items()):
        table.add_row(
            name, f"{record.destination}:{record.port}",
            ", ".join(record.capabilities.gpu_names) or "?",
            record.capabilities.driver or "?", record.added_at or "?",
        )
    console.print(table)


@remote_app.command()
def status(name: str = typer.Argument(..., help = "Remote name.")):
    """Show agent health, active job, GPU, and disk for a remote."""
    console = Console()
    try:
        record = get_remote(name)
        runner, tunnel, client = _connect(record)
        with tunnel:
            client.health()
            train_status = client.train_status()
            hardware = client.hardware()
        disk = runner.run('df -Pk "$HOME" | tail -1', check = False).stdout.split()
        disk_free = f"{int(disk[3]) / 1024 / 1024:.0f} GB free" if len(disk) > 3 and disk[3].isdigit() else "?"

        console.print(f"[bold]{name}[/bold] ({record.destination})")
        console.print(f"  agent:  [green]up[/green] — phase {train_status.get('phase')}")
        if train_status.get("job_id"):
            details = train_status.get("details") or {}
            step = details.get("step")
            loss = details.get("loss")
            console.print(
                f"  job:    {train_status['job_id']} "
                f"(step {step if step is not None else '?'}/{details.get('total_steps') or '?'}, "
                f"loss {f'{loss:.4f}' if loss is not None else '?'})"
            )
        for gpu in (hardware.get("gpus") or []) if isinstance(hardware, dict) else []:
            used = gpu.get("memory_used_mb") or gpu.get("vram_used_mb")
            total = gpu.get("memory_total_mb") or gpu.get("vram_total_mb")
            util = gpu.get("utilization_percent") or gpu.get("utilization")
            console.print(f"  gpu:    {gpu.get('name', 'GPU')} — {used}/{total} MB, {util}% util")
        console.print(f"  disk:   {disk_free}")
    except RemoteError as e:
        raise _fail(e)


@remote_app.command()
def rm(name: str = typer.Argument(..., help = "Remote name.")):
    """Forget a remote (the machine itself is untouched)."""
    try:
        remove_remote(name)
    except RemoteError as e:
        raise _fail(e)
    Console().print(f"Removed remote '{name}'.")


def _refresh_remote_jobs(records: list) -> set:
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


def jobs():
    """List remote training jobs (refreshes status from reachable remotes)."""
    console = Console()
    records = list_jobs()
    if not records:
        console.print("No remote jobs yet. Start one with: unsloth train --config <cfg> --remote <name>")
        return
    unreachable = _refresh_remote_jobs(records)
    records.sort(key = lambda j: j.submitted_at or "", reverse = True)
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
    job_id: Optional[str] = typer.Argument(None, help = "Job id or prefix (default: latest job)."),
    follow: bool = typer.Option(False, "--follow", "-f", help = "Keep streaming."),
    raw: bool = typer.Option(False, "--raw", help = "Raw server log instead of structured progress."),
):
    """Show a remote job's progress; reattaches to live runs."""
    console = Console()
    try:
        job = _resolve(job_id)
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
                f"follow with: unsloth logs -f"
            )
    except RemoteError as e:
        raise _fail(e)


def _wait_for_settled_run(client, job_id: str, timeout: float = 60.0) -> dict:
    # Stop is async; poll until the run leaves 'running'.
    deadline = time.monotonic() + timeout
    run = client.run_detail(job_id).get("run") or {}
    while run.get("status") == "running" and time.monotonic() < deadline:
        time.sleep(2)
        run = client.run_detail(job_id).get("run") or {}
    return run


def cancel(
    job_id: Optional[str] = typer.Argument(None, help = "Job id or prefix (default: latest job)."),
):
    """Stop a running remote job (saves a resumable checkpoint)."""
    console = Console()
    try:
        job = _resolve(job_id)
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
            client.stop_training(save = True)
            run = _wait_for_settled_run(client, job.job_id)
        _refresh_job_from_run(job, run)
        job.status = job.status if job.status != "running" else "stopped"
        save_job(job)
        console.print(f"Stopped [bold]{job.job_id}[/bold] — resumable with: unsloth resume")
    except RemoteError as e:
        raise _fail(e)


def resume(
    job_id: Optional[str] = typer.Argument(None, help = "Job id or prefix (default: latest job)."),
):
    """Resume a stopped remote job from its last checkpoint."""
    console = Console()
    try:
        job = _resolve(job_id)
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
            new_job = JobRecord(
                job_id = response["job_id"],
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
            console.print(f"Resumed as [bold]{new_job.job_id}[/bold] from {job.job_id}'s checkpoint.")

            try:
                attach_progress(client, console, new_job.job_id)
            except KeyboardInterrupt:
                console.print(f"\nDetached — [bold]{new_job.job_id}[/bold] keeps running.")
                return
            run = client.run_detail(new_job.job_id).get("run") or {}
        _refresh_job_from_run(new_job, run)
        save_job(new_job)
        if new_job.status == "completed":
            adapter = pull_job_artifact(record, new_job, console)
            console.print(f"Adapter pulled to [bold]{adapter}[/bold]")
    except RemoteError as e:
        raise _fail(e)


def pull(
    job_id: Optional[str] = typer.Argument(None, help = "Job id or prefix (default: latest job)."),
    force: bool = typer.Option(False, "--force", help = "Re-pull even if already pulled."),
):
    """Fetch a job's trained adapter into the local registry."""
    console = Console()
    try:
        job = _resolve(job_id)
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
