# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_MAX = 512
_cache_lock = threading.Lock()
_cache: dict[str, tuple[tuple, tuple[tuple[str, int], ...], dict]] = {}

_CONTEXT_KEY = re.compile(
    r"(?:position(?:s|_embeddings)|(?:^|_)ctx|context_len(?:gth)?"
    r"|seq(?:uence)?_len(?:gth)?|model_max_length)$"
)
_CONTEXT_KEY_EXCLUDE = re.compile(r"(?:^|_)image_")
_MIN_CONTEXT = 256
_MAX_CONTEXT = 1 << 24

_CODE = re.compile(
    r"(?<!en)coder|codestral|devstral|codellama|codegemma|codeqwen|(?<![a-z])code(?![a-z])"
)
_REASONING = re.compile(
    r"math|reason|think|qwq|magistral|(?<![a-z0-9])r1(?![0-9])"
)
_WRITING = re.compile(r"writer|writing|creative|story|novel|roleplay")


def suggest_tasks(model_id: str, vision: bool) -> list[str]:
    name = (model_id or "").lower()
    tasks = [
        task
        for task, pattern in (("code", _CODE), ("reasoning", _REASONING), ("writing", _WRITING))
        if pattern.search(name)
    ]
    if vision:
        return ["vision", *(tasks or ["general"])]
    return tasks or ["general"]


def _empty() -> dict:
    return {"vision": False, "tools": False, "context_length": None}


def _mtime(path: str) -> int:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return -1


def _resolve(model_id: str) -> Optional[tuple[str, Optional[str], bool]]:
    from core.inference.local_model_resolver import resolve_local_gguf

    resolved = resolve_local_gguf(model_id, include_companion_scope = True)
    if not resolved:
        return None
    load_path, variant, _loader_id, repo_level = resolved[:4]
    return load_path, variant, bool(repo_level)


def _gguf_weight_file(load_path: str, variant: Optional[str]) -> Optional[str]:
    from hub.services.models.ollama import is_ollama_manifest_ref

    if is_ollama_manifest_ref(load_path):
        from hub.services.models.ollama import ollama_model_ref_files

        return ollama_model_ref_files(load_path)[0]
    from utils.models.model_config import _find_local_gguf_by_variant, detect_gguf_model

    path = Path(load_path)
    if path.is_file():
        return load_path if path.suffix.lower() == ".gguf" else None
    if variant and path.is_dir():
        return _find_local_gguf_by_variant(load_path, variant)
    return detect_gguf_model(load_path)


def _gguf_vision(load_path: str, variant: Optional[str], repo_level: bool) -> bool:
    from hub.services.models.ollama import is_ollama_manifest_ref
    from utils.models.gguf_metadata import mmproj_accepts_image

    if is_ollama_manifest_ref(load_path):
        from hub.services.models.ollama import ollama_model_ref_files

        projector = ollama_model_ref_files(load_path)[1]
        return projector is not None and mmproj_accepts_image(projector)
    from core.inference.local_model_resolver import local_gguf_companion_roots
    from utils.models.model_config import is_vision_model

    roots = local_gguf_companion_roots(load_path, repo_level = repo_level)
    return bool(
        is_vision_model(
            load_path,
            local_files_only = True,
            gguf_variant = variant,
            gguf_companion_roots = roots or None,
        )
    )


def _gguf_capabilities(
    load_path: str, weight: str, variant: Optional[str], repo_level: bool
) -> dict:
    from core.inference.template_capabilities import template_supports_tools
    from utils.models.gguf_metadata import (
        _read_gguf_string,
        read_gguf_chat_template,
        read_gguf_context_length,
    )

    result = _empty()
    try:
        result["vision"] = _gguf_vision(load_path, variant, repo_level)
    except Exception as exc:
        logger.debug("auto router: vision probe failed for %s: %s", load_path, exc)
    try:
        templates = [
            read_gguf_chat_template(weight),
            _read_gguf_string(weight, "tokenizer.chat_template.tool_use"),
        ]
        result["tools"] = any(template_supports_tools(template) for template in templates)
    except Exception as exc:
        logger.debug("auto router: template probe failed for %s: %s", weight, exc)
    try:
        result["context_length"] = _plausible_context(read_gguf_context_length(weight))
    except Exception as exc:
        logger.debug("auto router: context probe failed for %s: %s", weight, exc)
    return result


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding = "utf-8-sig"))
    except (OSError, ValueError):
        return None


def _plausible_context(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if _MIN_CONTEXT <= value <= _MAX_CONTEXT else None


def _config_context_length(config) -> Optional[int]:
    if not isinstance(config, dict):
        return None
    lengths = []
    for cfg in (config.get("text_config"), config):
        if not isinstance(cfg, dict):
            continue
        for name, value in cfg.items():
            if _CONTEXT_KEY.search(str(name)) and not _CONTEXT_KEY_EXCLUDE.search(str(name)):
                length = _plausible_context(value)
                if length is not None:
                    lengths.append(length)
    return max(lengths) if lengths else None


def _weights_templates(load_dir: Path) -> list:
    templates = []
    tokenizer_config = _read_json(load_dir / "tokenizer_config.json")
    if isinstance(tokenizer_config, dict):
        template = tokenizer_config.get("chat_template")
        if isinstance(template, list):
            templates += [item.get("template") for item in template if isinstance(item, dict)]
        else:
            templates.append(template)
    chat_template_json = _read_json(load_dir / "chat_template.json")
    if isinstance(chat_template_json, dict):
        templates.append(chat_template_json.get("chat_template"))
    jinja_files = [load_dir / "chat_template.jinja"]
    extra = load_dir / "additional_chat_templates"
    if extra.is_dir():
        jinja_files += sorted(extra.glob("*.jinja"))
    for path in jinja_files:
        try:
            templates.append(path.read_text(encoding = "utf-8"))
        except OSError:
            pass
    return templates


def _weights_capabilities(load_path: str) -> dict:
    from core.inference.template_capabilities import template_supports_tools

    load_dir = Path(load_path)
    result = _empty()
    try:
        from utils.models.model_config import is_vision_model

        result["vision"] = bool(is_vision_model(load_path, local_files_only = True))
    except Exception as exc:
        logger.debug("auto router: vision probe failed for %s: %s", load_path, exc)
    try:
        result["tools"] = any(template_supports_tools(t) for t in _weights_templates(load_dir))
    except Exception as exc:
        logger.debug("auto router: template probe failed for %s: %s", load_path, exc)
    try:
        result["context_length"] = _config_context_length(_read_json(load_dir / "config.json"))
    except Exception as exc:
        logger.debug("auto router: context probe failed for %s: %s", load_path, exc)
    return result


def _watched_weights_files(load_path: str) -> list[str]:
    names = ("config.json", "tokenizer_config.json", "chat_template.json", "chat_template.jinja")
    return [load_path, *(os.path.join(load_path, name) for name in names)]


def _probe(load_path: str, variant: Optional[str], repo_level: bool) -> tuple[dict, list[str]]:
    from hub.services.models.ollama import is_ollama_manifest_ref

    weight = _gguf_weight_file(load_path, variant)
    if weight:
        watched = [weight, os.path.dirname(weight)]
        if not is_ollama_manifest_ref(load_path):
            watched.append(load_path)
        return _gguf_capabilities(load_path, weight, variant, repo_level), watched
    return _weights_capabilities(load_path), _watched_weights_files(load_path)


def detect_capabilities(model_id: str) -> dict:
    try:
        target = _resolve(model_id)
    except Exception as exc:
        logger.debug("auto router: could not resolve %s: %s", model_id, exc)
        target = None
    if target is None:
        return _empty()
    with _cache_lock:
        cached = _cache.get(model_id)
    if (
        cached is not None
        and cached[0] == target
        and all(_mtime(path) == stamp for path, stamp in cached[1])
    ):
        return dict(cached[2])
    try:
        result, watched = _probe(*target)
    except Exception as exc:
        logger.debug("auto router: capability probe failed for %s: %s", model_id, exc)
        return _empty()
    stamps = tuple((path, _mtime(path)) for path in watched)
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)), None)
        _cache[model_id] = (target, stamps, result)
    return dict(result)
