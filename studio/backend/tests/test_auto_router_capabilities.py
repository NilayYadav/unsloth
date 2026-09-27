# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

import core.inference.auto_router_capabilities as capabilities
from core.inference.auto_router_capabilities import detect_capabilities, suggest_tasks

TOOL_TEMPLATE = (
    "{% if tools %}{% for tool in tools %}{{ tool | tojson }}{% endfor %}{% endif %}"
    "{% for message in messages %}{{ message.content }}{% endfor %}"
)
PLAIN_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"


def _string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def _write_gguf(path: Path, *, template: str | None, context_length: int | None) -> Path:
    fields = [_string("general.architecture") + struct.pack("<I", 8) + _string("llama")]
    if context_length is not None:
        fields.append(_string("llama.context_length") + struct.pack("<II", 4, context_length))
    if template is not None:
        fields.append(_string("tokenizer.chat_template") + struct.pack("<I", 8) + _string(template))
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, len(fields)) + b"".join(fields))
    return path


def _bump(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns = (stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))


@pytest.fixture(autouse = True)
def _clear_cache():
    capabilities._cache.clear()


def _resolve_to(monkeypatch, load_path, variant = None):
    calls = []

    def resolve(model_id):
        calls.append(model_id)
        return (str(load_path), variant, False)

    monkeypatch.setattr(capabilities, "_resolve", resolve)
    return calls


@pytest.mark.parametrize(
    ("model_id", "vision", "expected"),
    [
        ("unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", False, ["code"]),
        ("mistralai/Devstral-Small-2505", False, ["code"]),
        ("mistralai/Codestral-22B-v0.1", False, ["code"]),
        ("google/codegemma-7b-it", False, ["code"]),
        ("unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", False, ["reasoning"]),
        ("Qwen/QwQ-32B", False, ["reasoning"]),
        ("unsloth/Qwen3-4B-Thinking-2507-GGUF", False, ["reasoning"]),
        ("microsoft/Phi-4-reasoning", False, ["reasoning"]),
        ("Qwen/Qwen2.5-Math-7B-Instruct", False, ["reasoning"]),
        ("sam-paech/Creative-Writer-9B", False, ["writing"]),
        ("unsloth/Llama-3.2-3B-Instruct-GGUF", False, ["general"]),
        ("unsloth/gemma-3-4b-it-GGUF", True, ["vision", "general"]),
        ("unsloth/Qwen3-VL-8B-Thinking-GGUF", True, ["vision", "reasoning"]),
        ("some/sentence-encoder-chat", False, ["general"]),
        ("", False, ["general"]),
    ],
)
def test_suggest_tasks(model_id, vision, expected):
    assert suggest_tasks(model_id, vision) == expected


def test_gguf_with_tool_template_reports_tools_and_context(tmp_path, monkeypatch):
    _write_gguf(tmp_path / "Model-Q4_K_M.gguf", template = TOOL_TEMPLATE, context_length = 32768)
    _resolve_to(monkeypatch, tmp_path, "Q4_K_M")

    assert detect_capabilities("org/Model-GGUF") == {
        "vision": False,
        "tools": True,
        "context_length": 32768,
    }


def test_gguf_picks_the_resolved_variant_file(tmp_path, monkeypatch):
    _write_gguf(tmp_path / "Model-Q8_0.gguf", template = PLAIN_TEMPLATE, context_length = 4096)
    _write_gguf(tmp_path / "Model-Q4_K_M.gguf", template = TOOL_TEMPLATE, context_length = 8192)
    _resolve_to(monkeypatch, tmp_path, "Q8_0")

    assert detect_capabilities("org/Model-GGUF") == {
        "vision": False,
        "tools": False,
        "context_length": 4096,
    }


def test_gguf_mmproj_companion_reports_vision(tmp_path, monkeypatch):
    _write_gguf(tmp_path / "Model-Q4_K_M.gguf", template = PLAIN_TEMPLATE, context_length = 8192)
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"\0" * 32)
    _resolve_to(monkeypatch, tmp_path, "Q4_K_M")

    assert detect_capabilities("org/Model-GGUF")["vision"] is True


def test_standalone_gguf_file_without_template(tmp_path, monkeypatch):
    weight = _write_gguf(tmp_path / "model.gguf", template = None, context_length = None)
    _resolve_to(monkeypatch, weight)

    assert detect_capabilities("model") == {"vision": False, "tools": False, "context_length": None}


def _write_weights_dir(root: Path, *, template, config) -> Path:
    root.mkdir(parents = True, exist_ok = True)
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    (root / "model.safetensors").write_bytes(b"\0" * 16)
    return root


def test_safetensors_dir_reads_tokenizer_template_and_config(tmp_path, monkeypatch):
    root = _write_weights_dir(
        tmp_path / "Model",
        template = TOOL_TEMPLATE,
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "max_position_embeddings": 131072,
        },
    )
    _resolve_to(monkeypatch, root)

    assert detect_capabilities("org/Model") == {
        "vision": False,
        "tools": True,
        "context_length": 131072,
    }


def test_safetensors_vision_config_and_named_templates(tmp_path, monkeypatch):
    root = _write_weights_dir(
        tmp_path / "VL",
        template = [
            {"name": "default", "template": PLAIN_TEMPLATE},
            {"name": "tool_use", "template": TOOL_TEMPLATE},
        ],
        config = {
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "vision_config": {"image_size": 896},
            "text_config": {"max_position_embeddings": 128000},
        },
    )
    _resolve_to(monkeypatch, root)

    assert detect_capabilities("org/VL") == {
        "vision": True,
        "tools": True,
        "context_length": 128000,
    }


def test_chat_template_jinja_file_is_read(tmp_path, monkeypatch):
    root = _write_weights_dir(
        tmp_path / "Jinja",
        template = None,
        config = {"architectures": ["LlamaForCausalLM"], "max_position_embeddings": 8192},
    )
    (root / "chat_template.jinja").write_text(TOOL_TEMPLATE)
    _resolve_to(monkeypatch, root)

    assert detect_capabilities("org/Jinja")["tools"] is True


def test_results_are_cached_until_a_watched_file_changes(tmp_path, monkeypatch):
    root = _write_weights_dir(
        tmp_path / "Model",
        template = TOOL_TEMPLATE,
        config = {"architectures": ["LlamaForCausalLM"], "max_position_embeddings": 8192},
    )
    _resolve_to(monkeypatch, root)
    probes = []
    real_probe = capabilities._probe
    monkeypatch.setattr(
        capabilities, "_probe", lambda *args: probes.append(args) or real_probe(*args)
    )

    assert detect_capabilities("org/Model")["tools"] is True
    assert detect_capabilities("org/Model")["tools"] is True
    assert len(probes) == 1

    tokenizer_config = root / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps({"chat_template": PLAIN_TEMPLATE}))
    _bump(tokenizer_config)
    assert detect_capabilities("org/Model")["tools"] is False
    assert len(probes) == 2


def test_unresolvable_or_failing_models_report_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities, "_resolve", lambda model_id: None)
    assert detect_capabilities("missing") == {"vision": False, "tools": False, "context_length": None}

    def boom(model_id):
        raise OSError("unreadable")

    monkeypatch.setattr(capabilities, "_resolve", boom)
    assert detect_capabilities("broken") == {"vision": False, "tools": False, "context_length": None}

    _resolve_to(monkeypatch, tmp_path / "gone")
    assert detect_capabilities("gone") == {"vision": False, "tools": False, "context_length": None}


def test_implausible_context_lengths_are_ignored(tmp_path, monkeypatch):
    root = _write_weights_dir(
        tmp_path / "Model",
        template = PLAIN_TEMPLATE,
        config = {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 4096,
            "model_max_length": 1_000_000_000_000_000_000,
        },
    )
    _resolve_to(monkeypatch, root)

    assert detect_capabilities("org/Model")["context_length"] == 4096


def test_validation_message_drops_pydantic_noise():
    from pydantic import ValidationError

    from core.inference.auto_router import RouterProfile
    from routes.settings import _validation_message

    with pytest.raises(ValidationError) as caught:
        RouterProfile.model_validate({"models": [{"id": "a", "tasks": ["general"]}]})
    assert _validation_message(caught.value) == "Choose a default model for Auto"


def test_laya_status(monkeypatch):
    import threading

    import core.inference.auto_router as auto_router
    from core.inference.auto_router import RouterProfile
    from routes.settings import _auto_router_laya_status

    single = RouterProfile.model_validate(
        {"models": [{"id": "a", "tasks": ["code", "vision"]}], "default_model": "a"}
    )
    pool = RouterProfile.model_validate(
        {
            "models": [{"id": "a", "tasks": ["code"]}, {"id": "b", "tasks": ["general"]}],
            "default_model": "b",
        }
    )
    warms = []
    monkeypatch.setattr(auto_router, "warm_laya", lambda: warms.append(1))
    monkeypatch.setattr(auto_router, "_laya_agent", None)
    monkeypatch.setattr(auto_router, "_laya_loader", None)
    monkeypatch.setattr(auto_router, "_laya_retry_at", 0.0)

    assert _auto_router_laya_status(single) == "not_needed"
    assert _auto_router_laya_status(pool) == "loading"
    assert warms == [1]

    monkeypatch.setattr(auto_router, "_laya_retry_at", float("inf"))
    assert _auto_router_laya_status(pool) == "unavailable"

    gate = threading.Event()
    loader = threading.Thread(target = gate.wait)
    loader.start()
    monkeypatch.setattr(auto_router, "_laya_loader", loader)
    try:
        assert _auto_router_laya_status(pool) == "loading"
    finally:
        gate.set()
        loader.join()

    monkeypatch.setattr(auto_router, "_laya_agent", object())
    assert _auto_router_laya_status(pool) == "ready"
    assert warms == [1]
