# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth remote` — manage remote GPU machines for cloud training."""

from __future__ import annotations

import re
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.agent import AgentClient
from unsloth_cli.remote.jobs import utc_now
from unsloth_cli.remote.bootstrap import AGENT_SERVICE, ENV_DIR, run_bootstrap
from unsloth_cli.remote.probe import ProbeResult, run_probe
from unsloth_cli.remote.ssh import SSHRunner, pin_host_key, scan_host_key
from unsloth_cli.remote.state import (
    DEFAULT_AGENT_PORT,
    RemoteRecord,
    get_remote,
    load_remotes,
    remove_remote,
    upsert_remote,
)

remote_app = typer.Typer(help = "Manage remote GPU machines for cloud training.")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Remote-side paths (under the agent's studio home).
_REMOTE_OUTPUTS = "~/.unsloth/studio/outputs"
_REMOTE_UPLOADS = "~/.unsloth/studio/assets/datasets/uploads/remote"


def _fail(error: RemoteError) -> "typer.Exit":
    console = Console(stderr = True)
    console.print(f"[red]Error:[/red] {error}")
    if error.hint:
        console.print(f"[yellow]Hint:[/yellow] {error.hint}")
    return typer.Exit(code = 1)


def _print_probe(console: Console, result: ProbeResult) -> None:
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


def _agent_client(record: RemoteRecord, runner: Optional[SSHRunner] = None):
    """Open a tunnel to the record's agent; returns (tunnel, client)."""
    runner = runner or SSHRunner.for_remote(record)
    tunnel = runner.open_tunnel(record.agent_port)
    return tunnel, AgentClient(tunnel.url, api_key = record.api_key)


@remote_app.command()
def add(
    name: str = typer.Argument(..., help = "Name for this remote (e.g. gpu1)."),
    destination: str = typer.Argument(..., help = "SSH destination, user@host."),
    key: Optional[str] = typer.Option(None, "--key", "-i", help = "SSH private key path."),
    port: int = typer.Option(22, "--port", "-p", help = "SSH port."),
    agent_port: int = typer.Option(
        DEFAULT_AGENT_PORT, "--agent-port",
        help = "Port the agent binds on the remote's localhost.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help = "Skip confirmation prompts."),
    skip_install: bool = typer.Option(
        False, "--skip-install", hidden = True,
        help = "Skip uv/env/package installation (assumes an existing env).",
    ),
):
    """Probe, bootstrap, and register a remote GPU machine.

    The machine only needs SSH access and an NVIDIA driver; everything else
    (Python, unsloth, the agent) is installed into ~/.unsloth/env on the VM.
    """
    console = Console()
    if not _NAME_RE.match(name):
        raise _fail(RemoteError(f"Invalid remote name {name!r} (use letters, digits, - or _)."))
    if "@" not in destination:
        raise _fail(RemoteError(
            f"Destination {destination!r} must be user@host.",
            hint = "Example: unsloth remote add gpu1 ubuntu@203.0.113.7",
        ))
    if name in load_remotes() and not yes:
        typer.confirm(f"Remote '{name}' already exists. Overwrite?", abort = True)

    try:
        host = destination.split("@", 1)[1]
        console.print(f"Fetching host key of [bold]{host}[/bold]...")
        fingerprint, known_hosts_text = scan_host_key(host, port = port)
        console.print(f"  pinned {fingerprint}")
        pin_host_key(known_hosts_text)

        runner = SSHRunner(destination, port = port, identity_file = key)
        console.print("Probing hardware and tooling...")
        probe = run_probe(runner)
        _print_probe(console, probe)
        if probe.fatal_failures:
            raise RemoteError(
                "The machine is not ready (see failures above).",
                hint = "Fix the issues and re-run `unsloth remote add`.",
            )
        if probe.warnings and not yes:
            typer.confirm("Continue despite the warnings above?", abort = True)

        result = None
        with console.status("Bootstrapping...") as status:
            def on_step(label: str):
                status.update(f"[bold]{label}[/bold]")
                console.print(f"  → {label}")

            result = run_bootstrap(
                runner, probe.capabilities, agent_port,
                skip_install = skip_install, on_step = on_step,
            )

        record = RemoteRecord(
            name = name,
            host = host,
            user = destination.split("@", 1)[0],
            port = port,
            identity_file = key,
            host_key_fingerprint = fingerprint,
            agent_port = agent_port,
            api_key = result.api_key,
            supervision = result.supervision,
            unsloth_version = result.remote_unsloth_version,
            added_at = utc_now(),
            capabilities = probe.capabilities,
        )

        console.print("Verifying the agent through an SSH tunnel...")
        with runner.open_tunnel(agent_port) as tunnel:
            client = AgentClient(tunnel.url, api_key = record.api_key)
            client.health()
            status_payload = client.train_status()
            hardware = client.hardware()
        gpus = hardware.get("gpus") if isinstance(hardware, dict) else None
        if isinstance(gpus, list) and not gpus:
            console.print("[yellow]Warning:[/yellow] the agent reports no visible GPUs.")

        upsert_remote(record)
        console.print(f"\n[green]Remote '{name}' is ready[/green] "
                      f"(agent phase: {status_payload.get('phase', 'unknown')}).")
        console.print(f"Try: [bold]unsloth train --config train.yaml --remote {name}[/bold]")
    except RemoteError as e:
        raise _fail(e)


@remote_app.command("list")
def list_remotes(
    offline: bool = typer.Option(
        False, "--offline", help = "Skip liveness checks (faster).",
    ),
):
    """List configured remotes."""
    console = Console()
    remotes = load_remotes()
    if not remotes:
        console.print("No remotes configured. Add one with: unsloth remote add <name> <user@host>")
        return
    table = Table()
    for column in ("Name", "Destination", "GPU", "Driver", "Agent"):
        table.add_column(column)
    for name, record in sorted(remotes.items()):
        gpu = ", ".join(record.capabilities.gpu_names) or "?"
        if offline:
            agent = "-"
        else:
            agent = _liveness(record)
        table.add_row(name, f"{record.destination}:{record.port}", gpu,
                      record.capabilities.driver or "?", agent)
    console.print(table)


def _liveness(record: RemoteRecord) -> str:
    try:
        tunnel, client = _agent_client(record)
        with tunnel:
            status = client.train_status()
        phase = status.get("phase", "unknown")
        job_id = status.get("job_id") or ""
        if phase in ("training", "loading_model", "configuring") and job_id:
            return f"[green]busy[/green] ({job_id})"
        return f"[green]up[/green] ({phase})"
    except RemoteError:
        return "[red]unreachable[/red]"


@remote_app.command()
def status(name: str = typer.Argument(..., help = "Remote name.")):
    """Show agent health, active job, GPU utilization, and disk usage."""
    console = Console()
    try:
        record = get_remote(name)
        runner = SSHRunner.for_remote(record)
        tunnel, client = _agent_client(record, runner)
        with tunnel:
            train_status = client.train_status()
            hardware = client.hardware()
        disk = runner.run('df -Pk "$HOME" | tail -1', check = False).stdout.split()
        disk_free = f"{int(disk[3]) / 1024 / 1024:.0f} GB free" if len(disk) > 3 and disk[3].isdigit() else "?"

        console.print(f"[bold]{name}[/bold] ({record.destination})")
        console.print(f"  agent:  up — phase {train_status.get('phase')}")
        if train_status.get("job_id"):
            details = train_status.get("details") or {}
            console.print(
                f"  job:    {train_status['job_id']} "
                f"(step {details.get('step', '?')}/{details.get('total_steps', '?')}, "
                f"loss {details.get('loss', '?')})"
            )
        for gpu in (hardware.get("gpus") or []) if isinstance(hardware, dict) else []:
            gpu_name = gpu.get("name", "GPU")
            used = gpu.get("memory_used_mb") or gpu.get("vram_used_mb")
            total = gpu.get("memory_total_mb") or gpu.get("vram_total_mb")
            util = gpu.get("utilization_percent") or gpu.get("utilization")
            console.print(f"  gpu:    {gpu_name} — {used}/{total} MB, {util}% util")
        console.print(f"  disk:   {disk_free}")
    except RemoteError as e:
        raise _fail(e)


@remote_app.command()
def doctor(name: str = typer.Argument(..., help = "Remote name.")):
    """Re-probe the machine and verify agent health, auth, and versions."""
    console = Console()
    try:
        record = get_remote(name)
        runner = SSHRunner.for_remote(record)
        probe = run_probe(runner)
        _print_probe(console, probe)

        checks_ok = not probe.fatal_failures
        agent_ok = False
        auth_ok = False
        phase = "?"
        try:
            tunnel, client = _agent_client(record, runner)
            with tunnel:
                client.health()
                agent_ok = True
                phase = client.train_status().get("phase", "?")
                auth_ok = True
        except RemoteError as e:
            console.print(f"[red]Agent check failed:[/red] {e}")

        console.print(f"  agent reachable: {'[green]yes[/green]' if agent_ok else '[red]no[/red]'}")
        console.print(f"  API key valid:   {'[green]yes[/green]' if auth_ok else '[red]no[/red]'} (phase: {phase})")
        if record.unsloth_version:
            console.print(f"  remote unsloth:  {record.unsloth_version}")
        if not (checks_ok and agent_ok and auth_ok):
            raise typer.Exit(code = 1)
        console.print("[green]All checks passed.[/green]")
    except RemoteError as e:
        raise _fail(e)


@remote_app.command()
def rm(
    name: str = typer.Argument(..., help = "Remote name."),
    purge: bool = typer.Option(
        False, "--purge",
        help = "Also stop the agent and remove ~/.unsloth/env on the machine.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Remove a remote from this machine (optionally cleaning up the VM)."""
    console = Console()
    try:
        record = get_remote(name)
        if not yes:
            typer.confirm(f"Remove remote '{name}' ({record.destination})?", abort = True)
        if purge:
            runner = SSHRunner.for_remote(record)
            console.print("Stopping the agent on the VM...")
            if record.supervision == "systemd":
                runner.run(
                    f"systemctl --user disable --now {AGENT_SERVICE} 2>/dev/null; "
                    f"rm -f ~/.config/systemd/user/{AGENT_SERVICE}.service; "
                    "systemctl --user daemon-reload",
                    check = False,
                )
            elif record.supervision == "tmux":
                runner.run(f"tmux kill-session -t {AGENT_SERVICE} 2>/dev/null || true", check = False)
            if not yes:
                typer.confirm(f"Also delete {ENV_DIR} on the VM?", abort = True)
            runner.run(f"rm -rf {ENV_DIR}", check = False, timeout = 300)
            runner.close_master()
        remove_remote(name)
        console.print(f"Removed remote '{name}'.")
    except RemoteError as e:
        raise _fail(e)


@remote_app.command(
    "exec",
    context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
)
def exec_command(
    ctx: typer.Context,
    name: str = typer.Argument(..., help = "Remote name."),
):
    """Run any unsloth command on the remote exactly as it runs locally.

    Example: unsloth remote exec gpu1 -- export ./outputs/run1 ./out -f gguf
    """
    import shlex

    from unsloth_cli.remote.bootstrap import ENV_UNSLOTH

    args = list(ctx.args)
    if args and args[0] == "unsloth":
        args = args[1:]
    if not args:
        raise _fail(RemoteError(
            "No command given.",
            hint = "Example: unsloth remote exec gpu1 -- train --help",
        ))
    try:
        record = get_remote(name)
        runner = SSHRunner.for_remote(record)
        code = runner.stream(f"{ENV_UNSLOTH} {shlex.join(args)}")
        raise typer.Exit(code = code)
    except RemoteError as e:
        raise _fail(e)


@remote_app.command()
def gc(
    name: str = typer.Argument(..., help = "Remote name."),
    keep_days: int = typer.Option(
        7, "--keep-days", help = "Keep checkpoints/uploads newer than this many days.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Free disk space on the remote: old checkpoints and dataset uploads."""
    console = Console()
    try:
        record = get_remote(name)
        runner = SSHRunner.for_remote(record)
        report = runner.run(
            f"du -sh {_REMOTE_OUTPUTS} {_REMOTE_UPLOADS} 2>/dev/null || true",
            timeout = 300,
        ).stdout.strip()
        console.print("Current usage:")
        for line in report.splitlines():
            console.print(f"  {line}")

        find_targets = (
            f"find {_REMOTE_OUTPUTS} -maxdepth 2 -type d -name 'checkpoint-*' -mtime +{keep_days}; "
            f"find {_REMOTE_UPLOADS} -mindepth 1 -maxdepth 1 -type d -mtime +{keep_days}"
        )
        targets = [
            l for l in runner.run(f"({find_targets}) 2>/dev/null || true", timeout = 300)
            .stdout.splitlines() if l.strip()
        ]
        if not targets:
            console.print(f"Nothing older than {keep_days} days to clean.")
            return
        console.print(f"\nWill delete {len(targets)} directories older than {keep_days} days:")
        for target in targets[:20]:
            console.print(f"  {target}")
        if len(targets) > 20:
            console.print(f"  ... and {len(targets) - 20} more")
        if not yes:
            typer.confirm("Proceed?", abort = True)
        delete_script = "\n".join(f"rm -rf '{t}'" for t in targets)
        runner.run("bash -s", input_text = delete_script, timeout = 600)
        console.print("[green]Done.[/green]")
    except RemoteError as e:
        raise _fail(e)
