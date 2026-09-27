# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

import core.inference.auto_router as auto_router
from core.inference.auto_router import RouterProfile, choose_model


@pytest.fixture(autouse=True)
def classify_general(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("general", 0.9))


def profile():
    return RouterProfile.model_validate(
        {
            "models": [
                {"id": "coder", "tasks": ["code"], "tools": True, "context_length": 4096},
                {"id": "vision", "tasks": ["vision"], "vision": True, "tools": False},
                {"id": "general", "tasks": ["general"], "tools": True},
            ],
            "default_model": "general",
            "rules": [{"contains": "project atlas", "model": "coder"}],
        }
    )


def route(
    *, prompt="hello", image=False, tools=False, tokens=100, current=None, pin=None, follow_up=False
):
    return choose_model(
        profile(),
        prompt=prompt,
        image=image,
        tools=tools,
        estimated_tokens=tokens,
        current_model=current,
        pinned_model=pin,
        follow_up=follow_up,
    )


def test_image_requires_vision_even_when_a_text_model_is_pinned():
    assert route(image=True).model == "vision"
    with pytest.raises(ValueError, match="pinned model cannot"):
        route(image=True, pin="coder")


def test_user_rule_selects_model_before_task_guess():
    decision = route(prompt="explain project atlas")
    assert decision.model == "coder"
    assert decision.reason == "matched a user rule"


def test_rule_cannot_override_capability_or_context_limit():
    with pytest.raises(ValueError, match="combination of images and tools"):
        route(image=True, tools=True)
    assert route(prompt="project atlas", tokens=5000).model == "general"


def test_missing_tool_capability_explains_how_to_fix_auto():
    pool = RouterProfile.model_validate(
        {"models": [{"id": "coder", "tasks": ["code"]}], "default_model": "coder"}
    )
    assert choose_model(
        pool, prompt="Write a Python program", image=False, tools=False,
        estimated_tokens=20, current_model=None,
    ).model == "coder"
    with pytest.raises(ValueError, match="Turn off Search/Code"):
        choose_model(
            pool, prompt="Write a Python program", image=False, tools=True,
            estimated_tokens=20, current_model=None,
        )


def test_context_error_reports_estimated_size():
    with pytest.raises(ValueError, match="about 5000 tokens"):
        choose_model(
            RouterProfile.model_validate({
                "models": [{"id": "coder", "tasks": ["code"], "context_length": 4096}],
                "default_model": "coder",
            }),
            prompt="Write a Python program", image=False, tools=False,
            estimated_tokens=5000, current_model=None,
        )


def test_short_follow_up_stays_on_current_model():
    decision = route(prompt="Continue fixing it", current="coder", follow_up=True)
    assert decision.model == "coder"
    assert decision.reason == "continuing this conversation"


def test_short_new_task_can_switch_models():
    decision = route(prompt="Now write a story", current="coder", follow_up=True)
    assert decision.model == "general"


def test_invalid_profile_cannot_name_a_model_outside_pool():
    with pytest.raises(ValueError, match="Default model must be"):
        RouterProfile.model_validate(
            {"models": [{"id": "coder", "tasks": ["code"]}], "default_model": "missing"}
        )
