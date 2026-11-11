# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from types import SimpleNamespace

import pytest

import core.inference.llama_cpp as llama_cpp
import routes.inference as inference_route
from core.inference import llama_keepwarm as kw
from core.inference.llama_cpp import LlamaCppBackend
from utils import openai_auto_switch_settings as settings
from utils.paths import storage_roots


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


@pytest.fixture
def slot_dir(monkeypatch, tmp_path):
    root = tmp_path / "llama-slots"
    root.mkdir()
    monkeypatch.setattr(storage_roots, "llama_slot_cache_root", lambda: root)
    monkeypatch.setattr(storage_roots, "mlx_prompt_cache_root", lambda: tmp_path / "mlx")
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    monkeypatch.setattr(llama_cpp.shutil, "disk_usage", lambda _p: SimpleNamespace(free = 1 << 40))
    monkeypatch.setattr(kw, "_unload_kv", [])
    monkeypatch.setattr(kw, "_kv_resume", None)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_pending", 0)
    return root


@pytest.fixture
def server(monkeypatch, slot_dir):
    calls = []

    def fake_post(url, **kwargs):
        action = kwargs["params"]["action"]
        calls.append((action, url.rsplit("/", 1)[-1], kwargs["json"]["filename"]))
        if action == "save":
            (slot_dir / kwargs["json"]["filename"]).write_bytes(b"kv")
        return _Resp({"n_saved": 40, "n_restored": 40})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    return calls


def _loaded_backend(slot_dir, tmp_path):
    backend = LlamaCppBackend()
    backend._healthy = True
    backend._process = SimpleNamespace(poll = lambda: None, pid = 0)
    backend._port = 8081
    backend._slot_save_dir = str(slot_dir)
    backend._slot_save_binary = ("/bin/llama-server", 1)
    gguf = tmp_path / "model.gguf"
    if not gguf.exists():
        gguf.write_bytes(b"gguf")
    backend._gguf_path = str(gguf)
    backend._effective_parallel_slots = 1
    backend._estimate_kv_cache_bytes = lambda *a, **k: 0
    backend._kill_process = lambda **_k: None
    return backend


def _stop_server(monkeypatch, backend):
    import run
    monkeypatch.setattr(inference_route, "_llama_cpp_backend", backend)
    run._graceful_shutdown()


def _start_server(monkeypatch):
    monkeypatch.setattr(kw, "_unload_kv", [])
    monkeypatch.setattr(kw, "_kv_resume", None)
    kw.sweep_slot_save_dir()


def _actions(calls):
    return [action for action, _slot, _name in calls]


def test_chat_kv_saved_at_shutdown_is_restored_after_a_restart(
    monkeypatch, tmp_path, slot_dir, server
):
    killed = []
    backend = _loaded_backend(slot_dir, tmp_path)
    backend._kill_process = lambda **_k: killed.append(list(_actions(server)))
    _stop_server(monkeypatch, backend)
    assert killed == [["save"]]
    saved = server[0][2]
    assert (slot_dir / saved).exists()

    _start_server(monkeypatch)
    assert (slot_dir / saved).exists()
    assert not list(slot_dir.glob("*.json"))

    kw.note_model_loaded(_loaded_backend(slot_dir, tmp_path))
    assert server[1:] == [("restore", "0", saved)]
    assert not (slot_dir / saved).exists()


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX file modes")
def test_shutdown_manifest_is_private(monkeypatch, tmp_path, slot_dir, server):
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    (manifest,) = slot_dir.glob("*.json")
    assert manifest.stat().st_mode & 0o077 == 0


def test_restart_does_not_restore_onto_rewritten_weights(monkeypatch, tmp_path, slot_dir, server):
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    saved = server[0][2]
    (tmp_path / "model.gguf").write_bytes(b"different weights")

    _start_server(monkeypatch)
    kw.note_model_loaded(_loaded_backend(slot_dir, tmp_path))
    assert _actions(server) == ["save"]
    assert not (slot_dir / saved).exists()


def test_restart_does_not_restore_onto_a_different_launch(monkeypatch, tmp_path, slot_dir, server):
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    saved = server[0][2]

    _start_server(monkeypatch)
    relaunched = _loaded_backend(slot_dir, tmp_path)
    relaunched._effective_context_length = 4096
    kw.note_model_loaded(relaunched)
    assert _actions(server) == ["save"]
    assert not (slot_dir / saved).exists()


def test_restart_does_not_restore_onto_another_llama_server(
    monkeypatch, tmp_path, slot_dir, server
):
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    saved = server[0][2]

    _start_server(monkeypatch)
    updated = _loaded_backend(slot_dir, tmp_path)
    updated._slot_save_binary = ("/bin/llama-server", 2)
    kw.note_model_loaded(updated)
    assert _actions(server) == ["save"]
    assert not (slot_dir / saved).exists()


def test_slot_files_from_an_interrupted_save_are_swept(monkeypatch, slot_dir):
    orphan = slot_dir / "resume-dead-slot0.bin"
    orphan.write_bytes(b"kv")
    (slot_dir / "resume-manifest.tmp").write_text('{"slots": [{"filename": "' + orphan.name + '"}]')

    _start_server(monkeypatch)
    assert not orphan.exists()
    assert not list(slot_dir.iterdir())
    assert kw._unload_kv == []


def test_slot_files_are_deleted_when_the_manifest_cannot_be_written(
    monkeypatch, tmp_path, slot_dir, server
):
    monkeypatch.setattr(kw.os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("disk full")))
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    assert _actions(server) == ["save"]
    assert not list(slot_dir.glob("resume-*.bin"))

    _start_server(monkeypatch)
    assert not list(slot_dir.iterdir())


def test_manifest_naming_a_missing_slot_file_restores_nothing(
    monkeypatch, tmp_path, slot_dir, server
):
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    (slot_dir / server[0][2]).unlink()

    _start_server(monkeypatch)
    kw.note_model_loaded(_loaded_backend(slot_dir, tmp_path))
    assert _actions(server) == ["save"]
    assert not list(slot_dir.iterdir())


def test_keep_kv_off_saves_nothing_at_shutdown(monkeypatch, tmp_path, slot_dir, server):
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    assert server == []
    assert not list(slot_dir.iterdir())


def test_keep_kv_turned_off_drops_the_save_at_startup(monkeypatch, tmp_path, slot_dir, server):
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)

    _start_server(monkeypatch)
    assert not list(slot_dir.iterdir())
    kw.note_model_loaded(_loaded_backend(slot_dir, tmp_path))
    assert _actions(server) == ["save"]


def test_shutdown_skips_the_save_while_a_request_is_in_flight(
    monkeypatch, tmp_path, slot_dir, server
):
    monkeypatch.setattr(kw, "_inflight", 1)
    _stop_server(monkeypatch, _loaded_backend(slot_dir, tmp_path))
    assert server == []


def test_shutdown_with_no_model_loaded_writes_nothing(monkeypatch, tmp_path, slot_dir, server):
    backend = _loaded_backend(slot_dir, tmp_path)
    backend._healthy = False
    _stop_server(monkeypatch, backend)
    assert server == []
    assert not list(slot_dir.iterdir())
