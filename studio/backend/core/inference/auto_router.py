# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel, Field, model_validator

from utils.account_context import OWNER, run_as

logger = logging.getLogger(__name__)

SETTING_KEY = "auto_router_profile"
TASKS = ("code", "reasoning", "writing", "general", "vision")
CRITERIA = {
    "code": "code",
    "reasoning": "math and reasoning",
    "writing": "creative writing",
    "general": "general questions",
}
LAYA_LABELS = {
    "code": "code",
    "reasoning": "math",
    "writing": "creative writing",
    "general": "general",
}


class RouterModel(BaseModel):
    id: str = Field(min_length=1, max_length=300)
    tasks: list[str] = Field(default_factory=list)
    vision: bool = False
    tools: bool = False
    context_length: int | None = Field(default=None, ge=256)

    @model_validator(mode="after")
    def validate_tasks(self):
        if not self.tasks or any(task not in TASKS for task in self.tasks):
            raise ValueError(f"Choose at least one task from: {', '.join(TASKS)}")
        if len(self.tasks) != len(set(self.tasks)):
            raise ValueError("Model tasks must be unique")
        return self


class RouterRule(BaseModel):
    contains: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=300)


class RouterProfile(BaseModel):
    models: list[RouterModel] = Field(default_factory=list, max_length=256)
    default_model: str | None = None
    rules: list[RouterRule] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_models(self):
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("Auto models must be unique")
        if self.default_model is not None and self.default_model not in ids:
            raise ValueError("Default model must be in the Auto group")
        if any(rule.model not in ids for rule in self.rules):
            raise ValueError("Rule target must be in the Auto group")
        if self.models and self.default_model is None:
            raise ValueError("Choose a default model for Auto")
        return self


@dataclass(frozen=True)
class RouterDecision:
    model: str
    reason: str
    task: str | None = None
    fallbacks: tuple[str, ...] = ()


def get_profile() -> RouterProfile:
    from storage.studio_db import get_app_setting

    saved = run_as(OWNER, get_app_setting, SETTING_KEY, None)
    return RouterProfile.model_validate(saved or {})


def build_profile(candidates: list[dict], resident: frozenset[str] = frozenset()) -> RouterProfile:
    models = []
    for candidate in candidates:
        try:
            models.append(RouterModel.model_validate(candidate))
        except ValueError:
            continue
    if not models:
        return RouterProfile()
    default = next((model.id for model in models if model.id in resident), models[0].id)
    return RouterProfile(models=models, default_model=default)


def save_profile(profile: RouterProfile) -> RouterProfile:
    from storage.studio_db import upsert_app_settings

    run_as(OWNER, upsert_app_settings, {SETTING_KEY: profile.model_dump()})
    if needs_laya(profile):
        warm_laya()
    return profile


def needs_laya(profile: RouterProfile) -> bool:
    return len({task for model in profile.models for task in model.tasks if task in CRITERIA}) > 1


def warm_if_configured() -> None:
    if needs_laya(get_profile()):
        warm_laya()


_laya_lock = threading.Lock()
_predict_lock = threading.Lock()
_laya_agent: Any = None
_laya_loader: threading.Thread | None = None
_laya_retry_at = 0.0
_sessions_lock = threading.Lock()
_sessions: dict[tuple[str, str], tuple[str, float]] = {}


def session_model(subject: str, session_id: str) -> str | None:
    with _sessions_lock:
        entry = _sessions.get((subject, session_id))
        if entry is None or time.monotonic() - entry[1] > 3600:
            return None
        return entry[0]


def remember_session(subject: str, session_id: str, model: str) -> None:
    with _sessions_lock:
        if len(_sessions) >= 1024:
            oldest = min(_sessions, key=lambda key: _sessions[key][1])
            _sessions.pop(oldest, None)
        _sessions[(subject, session_id)] = (model, time.monotonic())


def warm_laya() -> None:
    global _laya_loader
    with _laya_lock:
        if (
            _laya_agent is not None
            or time.monotonic() < _laya_retry_at
            or (_laya_loader is not None and _laya_loader.is_alive())
        ):
            return
        _laya_loader = threading.Thread(target=_load_laya, name="auto-router-laya", daemon=True)
        _laya_loader.start()


def _load_laya() -> None:
    global _laya_agent, _laya_retry_at
    from core.systemone import laya_runtime
    from core.systemone.catalog import CHECKPOINTS

    try:
        laya_runtime.ensure_package()
        import laya

        checkpoint = CHECKPOINTS["laya-typed-decisions"]
        folder = laya_runtime._checkpoint_dir(checkpoint)
        agent = laya.load(str(folder), subfolder=checkpoint.subfolder, device="cpu")
    except Exception as exc:
        logger.warning("Auto router Laya is unavailable: %s", exc)
        with _laya_lock:
            _laya_retry_at = time.monotonic() + 60
        return
    with _laya_lock:
        _laya_agent = agent


def _classify(prompt: str, choices: list[str]) -> tuple[str, float]:
    from core.systemone import laya_runtime

    if _laya_agent is None:
        warm_laya()
        raise RuntimeError("Laya is loading")

    with _predict_lock:
        question = {
            "task": {
                "type": "choice",
                "instructions": "Choose the main task requested by the user.",
                "criteria": {LAYA_LABELS[task]: CRITERIA[task] for task in choices},
            }
        }
        answers, _ = laya_runtime._predict(_laya_agent, prompt[:4000], question)
        result = answers["answers"]["task"]
    choice = result["choice"]
    task = next(task for task, label in LAYA_LABELS.items() if label == choice)
    return task, float(result["probabilities"][choice])


def _prefer(ids: list[str], resident: frozenset[str], default: str | None) -> str:
    loaded = [model for model in ids if model in resident]
    if default in loaded:
        return default
    if loaded:
        return loaded[0]
    return default if default in ids else ids[0]


def choose_model(
    profile: RouterProfile,
    *,
    prompt: str,
    image: bool,
    tools: bool,
    estimated_tokens: int,
    current_model: str | None,
    pinned_model: str | None = None,
    follow_up: bool = False,
    tool_turn: bool = False,
    resident: frozenset[str] = frozenset(),
) -> RouterDecision:
    if not profile.models:
        raise ValueError("Auto has no models. Download a model first.")
    capable = [
        model
        for model in profile.models
        if (not image or model.vision)
        and (not tools or model.tools)
    ]
    eligible = [
        model for model in capable
        if model.context_length is None or estimated_tokens <= model.context_length
    ]
    decision = _decide(
        profile, eligible, capable, prompt=prompt, image=image, tools=tools,
        estimated_tokens=estimated_tokens, current_model=current_model, pinned_model=pinned_model,
        follow_up=follow_up, tool_turn=tool_turn, resident=resident,
    )
    if pinned_model:
        return decision
    ids = [model.id for model in eligible]
    order = [current_model, *[m for m in ids if m in resident], profile.default_model, *ids]
    fallbacks = tuple(dict.fromkeys(m for m in order if m in ids and m != decision.model))
    return replace(decision, fallbacks=fallbacks)


def _decide(
    profile: RouterProfile,
    eligible: list[RouterModel],
    capable: list[RouterModel],
    *,
    prompt: str,
    image: bool,
    tools: bool,
    estimated_tokens: int,
    current_model: str | None,
    pinned_model: str | None,
    follow_up: bool,
    tool_turn: bool,
    resident: frozenset[str],
) -> RouterDecision:
    if not eligible:
        if image and not any(model.vision for model in profile.models):
            raise ValueError("No Auto model accepts images. Mark a vision model as 'Accepts images' in Router settings.")
        if tools and not any(model.tools for model in profile.models):
            raise ValueError("No Auto model supports tools. Turn off Search/Code or mark a tool-capable model as 'Supports tools' in Router settings.")
        if not capable:
            raise ValueError("No Auto model supports this combination of images and tools.")
        raise ValueError(
            f"The request is about {estimated_tokens} tokens, beyond every compatible Auto model's context limit. "
            "Shorten the conversation or correct the context lengths in Router settings."
        )
    by_id = {model.id: model for model in eligible}
    if pinned_model:
        if pinned_model not in by_id:
            raise ValueError("The pinned model cannot handle this request.")
        return RouterDecision(pinned_model, "pinned by user")
    if tool_turn and current_model in by_id:
        return RouterDecision(current_model, "continuing a tool call")
    if image:
        vision = [model.id for model in eligible if model.vision]
        if current_model in vision:
            return RouterDecision(current_model, "vision model already serving", "vision")
        chosen = _prefer(vision, resident, profile.default_model)
        return RouterDecision(chosen, "image requires a vision model", "vision")
    for rule in profile.rules:
        if rule.model in by_id and rule.contains.casefold() in prompt.casefold():
            return RouterDecision(rule.model, "matched a user rule")
    if (
        follow_up
        and current_model in by_id
        and re.match(
            r"\s*(continue\b|go on\b|keep going\b|finish (it|that)\b|same task\b)", prompt, re.I
        )
    ):
        return RouterDecision(current_model, "continuing this conversation")
    tasks = [task for task in CRITERIA if any(task in model.tasks for model in eligible)]
    laya_unavailable = False
    if len(tasks) == 1:
        task, confidence = tasks[0], 1.0
    elif tasks:
        try:
            task, confidence = _classify(prompt, tasks)
        except Exception:
            task, confidence = "general", 0.0
            laya_unavailable = True
    else:
        task, confidence = "general", 0.0
    matches = [model.id for model in eligible if task in model.tasks]
    no_signal = laya_unavailable or not tasks
    if current_model in by_id and (
        current_model in matches or no_signal or (follow_up and confidence < 0.7)
    ):
        return RouterDecision(current_model, "continuing with current model", task)
    if matches and not no_signal:
        if current_model in by_id and current_model in resident and not any(m in resident for m in matches):
            return RouterDecision(current_model, f"{task} model is not loaded, keeping the current model", task)
        return RouterDecision(_prefer(matches, resident, profile.default_model), f"{task} task", task)
    if profile.default_model in by_id:
        return RouterDecision(
            profile.default_model,
            "default while Laya is unavailable" if laya_unavailable else "default model",
        )
    return RouterDecision(eligible[0].id, "eligible fallback")
