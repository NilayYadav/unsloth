# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Terminal rendering of the remote training progress stream."""

from __future__ import annotations

from typing import Optional, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from unsloth_cli.remote.agent import AgentClient


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
    """Render the SSE progress stream until the job completes or errors.

    Returns (outcome, last_payload) where outcome is "completed" or "error".
    KeyboardInterrupt propagates to the caller (detach is the caller's UX).
    """
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
