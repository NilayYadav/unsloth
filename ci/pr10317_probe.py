"""PR 10317 repro probe: how many times does a GGUF Hub export convert the model?

Drives the REAL studio/backend/core/export/export.py:ExportBackend.export_gguf.
Only the model conversion leg and the Hub network leg are doubles; the double for
push_to_hub_gguf mirrors unsloth/save.py:unsloth_push_to_hub_gguf, which does
tempfile.mkdtemp(prefix="unsloth_gguf_") and re-runs unsloth_save_pretrained_gguf
before uploading from that temp directory.

Prints PROBE_JSON with measured values, then asserts the intended post-fix shape.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from fnmatch import fnmatch
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "studio" / "backend"
sys.path.insert(0, str(BACKEND))


def _preseed_utils_packages() -> None:
    """Register utils and utils.paths by path, without running their __init__.

    The repo's stub installer execs utils/paths/storage_roots.py under its own
    module name. If utils.paths has not been imported yet, that exec re-enters
    utils/paths/__init__.py, which imports the still-half-built storage_roots
    back and raises ImportError. Registering the two packages with a real
    __path__ lets storage_roots resolve utils.paths.path_utils directly and
    never runs the __init__, so the runner needs nothing beyond pytest.
    """
    import types

    for name, relative in (("utils", "utils"), ("utils.paths", "utils/paths")):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(BACKEND / relative)]
        module.__package__ = name
        sys.modules[name] = module


_preseed_utils_packages()

_SPEC = importlib.util.spec_from_file_location(
    "pr10317_helpers", BACKEND / "tests" / "test_export_absolute_paths.py"
)
_H = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_H)

# Size of one "quantized" artifact. Real bytes, really written, so the disk and
# wall-clock numbers below are measured rather than described. A real Q4_K_M of a
# 8B model is ~4.9GB; this is scaled down to keep the runner honest about time.
ARTIFACT_BYTES = 64 * 1024 * 1024
CHUNK = b"GGUF" + os.urandom(1020)

PRESEEDED: set = set()

STATE = {
    "conversions": [],
    "hub_uploads": [],
    "model_card": None,
    "token_seen": None,
    "repo_created": None,
}


def _convert(dest_dir: Path, label: str) -> Path:
    """Stand-in for merge + convert_hf_to_gguf + llama-quantize."""
    started = time.time()
    dest_dir.mkdir(parents = True, exist_ok = True)
    gguf = dest_dir / "model.Q4_K_M.gguf"
    written = 0
    with open(gguf, "wb") as handle:
        while written < ARTIFACT_BYTES:
            handle.write(CHUNK)
            written += len(CHUNK)
    (dest_dir / "Modelfile").write_text("FROM ./model.Q4_K_M.gguf\n")
    (dest_dir / "config.json").write_text('{"model_type": "llama"}\n')
    STATE["conversions"].append(
        {
            "label": label,
            "output_dir": str(dest_dir),
            "bytes": written,
            "seconds": round(time.time() - started, 3),
            "under_system_temp": str(dest_dir).startswith(
                str(Path(tempfile.gettempdir()).resolve())
            )
            or str(Path(dest_dir).resolve()).startswith(
                str(Path(tempfile.gettempdir()).resolve())
            ),
        }
    )
    return gguf


class Model:
    def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method, **kwargs):
        out = Path(f"{model_save_path}_gguf")
        gguf = _convert(out, "save_pretrained_gguf (local export)")
        return {"gguf_files": [str(gguf)], "modelfile_location": str(out / "Modelfile")}

    def push_to_hub_gguf(self, repo_id, tokenizer = None, **kwargs):
        # Mirrors unsloth_push_to_hub_gguf: a fresh system-temp dir, a full re-convert,
        # then the upload reads from THAT directory.
        temp_dir = Path(tempfile.mkdtemp(prefix = "unsloth_gguf_"))
        gguf = _convert(Path(f"{temp_dir}_gguf"), "push_to_hub_gguf (re-convert)")
        STATE["hub_uploads"].append(
            {"source_dir": str(gguf.parent), "files": sorted(p.name for p in gguf.parent.iterdir())}
        )


class _RepoUrl(str):
    repo_id = "unsloth-ci/pr10317-gguf"


class HfApi:
    def __init__(self, token = None):
        STATE["token_seen"] = "present" if token else "absent"

    def create_repo(self, repo_id, private = False, exist_ok = False, **kwargs):
        STATE["repo_created"] = {"repo_id": repo_id, "private": private, "exist_ok": exist_ok}
        return _RepoUrl("https://huggingface.co/unsloth-ci/pr10317-gguf")

    def upload_folder(
        self,
        folder_path,
        repo_id,
        repo_type,
        allow_patterns = None,
        ignore_patterns = None,
        **kwargs,
    ):
        # Mirror huggingface_hub: fnmatch over repo-relative paths across the whole tree.
        root = Path(folder_path)
        paths = [str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()]
        if allow_patterns is not None:
            paths = [f for f in paths if any(fnmatch(f, a) for a in allow_patterns)]
        for pattern in ignore_patterns or ():
            paths = [f for f in paths if not fnmatch(f, pattern)]
        STATE["hub_uploads"].append({"source_dir": str(folder_path), "files": sorted(paths)})


class ModelCard:
    def __init__(self, content):
        STATE["model_card"] = content

    def push_to_hub(self, repo_id, token = None, commit_message = None):
        pass


def main() -> int:
    import pytest

    mp = pytest.MonkeyPatch()
    try:
        _H._install_export_backend_stubs(mp)
        export_mod = _H._load_module("pr10317_export_backend", "core/export/export.py", mp)

        # Deliberately NOT under tempfile.gettempdir(): the whole point is that the
        # second conversion lands in the system temp dir while the user asked for this
        # workspace folder.
        root = Path.cwd() / "pr10317_export_root"
        if root.exists():
            import shutil as _shutil

            _shutil.rmtree(root)
        root.mkdir(parents = True)
        save_dir = root / "export"

        # The export panel has a folder browser and accepts absolute paths, so the
        # chosen folder can already hold the user's own files, and a failed earlier
        # export deliberately leaves its merged _tmp_model_* behind.
        save_dir.mkdir(parents = True)
        (save_dir / "notes.txt").write_text("unrelated")
        (save_dir / "dataset.jsonl").write_text('{"a": 1}')
        leftover = save_dir / "_tmp_model_earlier" / "model"
        leftover.mkdir(parents = True)
        (leftover / "model-00001-of-00002.safetensors").write_bytes(b"W" * 4096)
        PRESEEDED.update(
            {"notes.txt", "dataset.jsonl", "_tmp_model_earlier/model/model-00001-of-00002.safetensors"}
        )
        mp.setattr(export_mod, "resolve_export_write_dir", lambda _v: save_dir)
        mp.setattr(export_mod, "HfApi", HfApi, raising = False)
        mp.setattr(export_mod, "ModelCard", ModelCard, raising = False)

        backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
        backend.current_model = Model()
        backend.current_tokenizer = object()
        backend.current_checkpoint = None

        started = time.time()
        success, message, output_path = backend.export_gguf(
            str(save_dir),
            "Q4_K_M",
            push_to_hub = True,
            repo_id = "unsloth-ci/pr10317-gguf",
            hf_token = "hf_probe_token",
            private = False,
        )
        elapsed = round(time.time() - started, 3)
    finally:
        mp.undo()

    uploads = STATE["hub_uploads"]
    conversions = STATE["conversions"]
    on_hub = uploads[-1]["files"] if uploads else []
    hub_source = uploads[-1]["source_dir"] if uploads else None
    local_files = sorted(p.name for p in Path(output_path).iterdir()) if output_path else []

    report = {
        "export_succeeded": success,
        "message": message,
        "output_path": output_path,
        "conversion_count": len(conversions),
        "conversions": conversions,
        "total_bytes_converted": sum(c["bytes"] for c in conversions),
        "total_convert_seconds": round(sum(c["seconds"] for c in conversions), 3),
        "wall_clock_seconds": elapsed,
        "hub_upload_source": hub_source,
        "hub_upload_came_from_local_export": bool(output_path)
        and hub_source is not None
        and Path(hub_source).resolve() == Path(output_path).resolve(),
        "files_on_hub": on_hub,
        "files_in_local_export": local_files,
        "repo_created": STATE["repo_created"],
        "token_seen": STATE["token_seen"],
        "model_card_written": STATE["model_card"] is not None,
        "system_temp_root": str(Path(tempfile.gettempdir()).resolve()),
        "conversions_outside_the_requested_folder": [
            c["output_dir"] for c in conversions if c["under_system_temp"]
        ],
        "extra_bytes_outside_the_requested_folder": sum(
            c["bytes"] for c in conversions if c["under_system_temp"]
        ),
        "preseeded_user_files_in_the_folder": sorted(PRESEEDED),
        "unrelated_user_files_published": sorted(set(on_hub) & PRESEEDED),
    }
    print("PROBE_JSON " + json.dumps(report, indent = 2))

    failures = []
    if not success:
        failures.append(f"export_gguf failed: {message}")
    if report["conversion_count"] != 1:
        failures.append(
            f"expected exactly 1 conversion, observed {report['conversion_count']}: "
            + "; ".join(f"{c['label']} -> {c['output_dir']}" for c in conversions)
        )
    if not report["hub_upload_came_from_local_export"]:
        failures.append(
            f"expected the Hub upload to come from the local export {output_path!r}, "
            f"it came from {hub_source!r}"
        )
    if "model.Q4_K_M.gguf" not in on_hub:
        failures.append(f"expected model.Q4_K_M.gguf on the Hub, got {on_hub}")
    published = sorted(set(on_hub) & PRESEEDED)
    if published:
        failures.append(
            "the export published files it did not create, from the user's chosen "
            f"folder: {published}"
        )

    for line in failures:
        print(f"ASSERT_FAIL {line}")
    if failures:
        print("PROBE_RESULT FAIL")
        return 1
    print("PROBE_RESULT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
