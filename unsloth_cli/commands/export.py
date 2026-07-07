# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import shlex
from pathlib import Path
from typing import Optional

import typer


EXPORT_FORMATS = ["merged-16bit", "merged-4bit", "gguf", "lora"]
GGUF_QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]

# Remote-side scratch dir the verbatim `unsloth export` writes into before the
# result is pulled back to the user's local output_dir.
_REMOTE_EXPORTS_DIR = "~/.unsloth/studio/exports/remote"


def _remote_export(
    remote: str,
    checkpoint: Path,
    output_dir: Path,
    extra_flags: list,
    hf_token: Optional[str],
) -> None:
    """Run the same `unsloth export` CLI on the remote and pull the result back."""
    from rich.console import Console

    from unsloth_cli.remote import RemoteError, looks_like_job_id
    from unsloth_cli.remote.agent import AgentClient
    from unsloth_cli.remote.bootstrap import ENV_UNSLOTH
    from unsloth_cli.remote.jobs import resolve_job, utc_now
    from unsloth_cli.remote.registry import register_artifact
    from unsloth_cli.remote.ssh import SSHRunner
    from unsloth_cli.remote.state import get_remote

    console = Console()
    record = get_remote(remote)
    runner = SSHRunner.for_remote(record)

    job = None
    if looks_like_job_id(str(checkpoint)):
        job = resolve_job(str(checkpoint))
        if job.remote != remote:
            raise RemoteError(
                f"Job {job.job_id} ran on remote '{job.remote}', not '{remote}'.",
                hint = f"Use: unsloth export {job.job_id} ... --remote {job.remote}",
            )
        if not job.output_dir:
            raise RemoteError(
                f"Job {job.job_id} has no known output directory.",
                hint = "Refresh with: unsloth jobs",
            )
        remote_checkpoint = job.output_dir
    else:
        remote_checkpoint = str(checkpoint)

    # Exporting loads the model; refuse while a training job owns the GPU.
    try:
        tunnel = runner.open_tunnel(record.agent_port)
        try:
            status = AgentClient(tunnel.url, api_key = record.api_key).train_status()
        finally:
            tunnel.close()
        if status.get("phase") in ("training", "loading_model", "configuring"):
            raise RemoteError(
                f"Remote '{remote}' is busy training {status.get('job_id', '')}.",
                hint = "Wait for it to finish or cancel it first.",
            )
    except RemoteError as e:
        if "busy training" in str(e):
            raise
        console.print("[yellow]Warning:[/yellow] could not check the agent; continuing.")

    tag = job.job_id if job else Path(remote_checkpoint).name
    remote_out = f"{_REMOTE_EXPORTS_DIR}/{tag}_{utc_now().replace(':', '')}"
    command = " ".join([
        f"HF_TOKEN={shlex.quote(hf_token)}" if hf_token else "",
        ENV_UNSLOTH, "export",
        shlex.quote(remote_checkpoint), shlex.quote(remote_out),
        *extra_flags,
    ]).strip()

    console.print(f"Running export on {remote} ({record.destination})...")
    code = runner.stream(command)
    if code != 0:
        raise RemoteError(f"Remote export failed (exit {code}).")

    output_dir.mkdir(parents = True, exist_ok = True)
    console.print(f"Pulling result into {output_dir} ...")
    runner.rsync_down(f"{remote_out}/", f"{output_dir}/")
    pulled = sorted(p.name for p in output_dir.iterdir())
    for name in pulled:
        console.print(f"  {name}")
    if job is not None:
        register_artifact(
            job.job_id,
            remote = remote,
            gguf = [str(output_dir / n) for n in pulled if n.endswith(".gguf")],
        )
    console.print("[green]Export complete.[/green]")


def list_checkpoints(
    outputs_dir: Path = typer.Option(
        Path("./outputs"), "--outputs-dir", help = "Directory that holds training runs."
    ),
):
    """List checkpoints detected in the outputs directory."""
    from studio.backend.core.export import ExportBackend

    backend = ExportBackend()
    checkpoints = backend.scan_checkpoints(outputs_dir = str(outputs_dir))
    if not checkpoints:
        typer.echo("No checkpoints found.")
        raise typer.Exit()

    for model_name, ckpt_list, metadata in checkpoints:
        typer.echo(f"\n{model_name}:")
        for display, path, loss in ckpt_list:
            loss_str = f" (loss: {loss:.4f})" if loss is not None else ""
            typer.echo(f"  {display}{loss_str}: {path}")


def export(
    checkpoint: Path = typer.Argument(
        ..., help = "Path to checkpoint directory (or a job id with --remote)."
    ),
    output_dir: Path = typer.Argument(..., help = "Directory to save exported model."),
    remote: Optional[str] = typer.Option(
        None,
        "--remote",
        help = "Run the export on a configured remote and pull the result back.",
    ),
    format: str = typer.Option(
        "merged-16bit",
        "--format",
        "-f",
        help = f"Export format: {', '.join(EXPORT_FORMATS)}",
    ),
    quantization: str = typer.Option(
        "q4_k_m",
        "--quantization",
        "-q",
        help = f"GGUF quantization method: {', '.join(GGUF_QUANTS)}",
    ),
    push_to_hub: bool = typer.Option(
        False, "--push-to-hub", help = "Push exported model to HuggingFace Hub."
    ),
    repo_id: Optional[str] = typer.Option(
        None, "--repo-id", help = "HuggingFace repo ID (username/model-name)."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "HuggingFace token."
    ),
    private: bool = typer.Option(False, "--private", help = "Make the HuggingFace repo private."),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit"),
):
    """Export a checkpoint to various formats (merged, GGUF, LoRA adapter)."""
    if format not in EXPORT_FORMATS:
        typer.echo(
            f"Error: Invalid format '{format}'. Choose from: {', '.join(EXPORT_FORMATS)}",
            err = True,
        )
        raise typer.Exit(code = 2)

    if push_to_hub and not repo_id:
        typer.echo("Error: --repo-id required when using --push-to-hub", err = True)
        raise typer.Exit(code = 2)

    if remote:
        from unsloth_cli.remote import RemoteError

        flags = ["--format", format, "-q", quantization,
                 "--max-seq-length", str(max_seq_length),
                 "--load-in-4bit" if load_in_4bit else "--no-load-in-4bit"]
        if push_to_hub:
            flags += ["--push-to-hub", "--repo-id", str(repo_id)]
            if private:
                flags.append("--private")
        try:
            _remote_export(remote, checkpoint, output_dir, flags, hf_token)
        except RemoteError as e:
            typer.echo(f"Error: {e}", err = True)
            if e.hint:
                typer.echo(f"Hint: {e.hint}", err = True)
            raise typer.Exit(code = 1)
        raise typer.Exit(code = 0)

    from studio.backend.core.export import ExportBackend

    backend = ExportBackend()

    typer.echo(f"Loading checkpoint: {checkpoint}")
    success, message = backend.load_checkpoint(
        checkpoint_path = str(checkpoint),
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )
    if not success:
        typer.echo(f"Error: {message}", err = True)
        raise typer.Exit(code = 1)
    typer.echo(message)

    typer.echo(f"Exporting as {format}...")
    output_path: Optional[str] = None
    if format == "merged-16bit":
        success, message, output_path = backend.export_merged_model(
            save_directory = str(output_dir),
            format_type = "16-bit (FP16)",
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
        )
    elif format == "merged-4bit":
        success, message, output_path = backend.export_merged_model(
            save_directory = str(output_dir),
            format_type = "4-bit (FP4)",
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
        )
    elif format == "gguf":
        success, message, output_path = backend.export_gguf(
            save_directory = str(output_dir),
            quantization_method = quantization.upper(),
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
        )
    elif format == "lora":
        success, message, output_path = backend.export_lora_adapter(
            save_directory = str(output_dir),
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
        )

    if not success:
        typer.echo(f"Error: {message}", err = True)
        raise typer.Exit(code = 1)

    typer.echo(message)
    if output_path:
        typer.echo(f"Saved to: {output_path}")
