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
    *, prompt="hello", image=False, tools=False, tokens=100, current=None, pin=None, follow_up=False,
    tool_turn=False,
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
        tool_turn=tool_turn,
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


def test_new_chat_follows_an_unsure_task_guess_over_the_loaded_model(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.55))
    decision = route(prompt="Write a Python function", current="general")
    assert decision.model == "coder"
    assert decision.reason == "code task"


def test_new_chat_follows_an_unsure_task_guess_with_nothing_loaded(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.55))
    assert route(prompt="Write a Python function").model == "coder"


def test_follow_up_stays_unless_the_task_clearly_changes(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("general", 0.55))
    assert route(prompt="make it faster", current="coder", follow_up=True).model == "coder"


def test_tool_result_turn_keeps_the_serving_model(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("general", 0.95))
    decision = route(prompt="[tool_result]", current="coder", follow_up=True, tool_turn=True)
    assert decision.model == "coder"
    assert decision.reason == "continuing a tool call"


def test_laya_unavailable_keeps_the_loaded_model(monkeypatch):
    def unavailable(prompt, choices):
        raise RuntimeError("Laya is loading")

    monkeypatch.setattr(auto_router, "_classify", unavailable)
    assert route(prompt="Write a Python function", current="coder").model == "coder"
    assert route(prompt="Write a Python function").reason == "default while Laya is unavailable"


def test_prefers_a_loaded_model_that_serves_the_task(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.8))
    pool = RouterProfile.model_validate(
        {
            "models": [
                {"id": "coder-a", "tasks": ["code"]},
                {"id": "coder-b", "tasks": ["code"]},
                {"id": "general", "tasks": ["general"]},
            ],
            "default_model": "general",
        }
    )
    decision = choose_model(
        pool, prompt="Write a Python function", image=False, tools=False,
        estimated_tokens=20, current_model=None, resident=frozenset({"coder-b", "general"}),
    )
    assert decision.model == "coder-b"


def test_fallbacks_prefer_loaded_then_default_and_skip_incapable_models(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.8))
    decision = route(prompt="Write a Python function", tools=True, current="general")
    assert decision.model == "coder"
    assert decision.fallbacks == ("general",)


def test_task_model_is_loaded_over_a_resident_model_of_another_task(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.95))
    decision = choose_model(
        profile(), prompt="Write a Python function", image=False, tools=False,
        estimated_tokens=20, current_model="general", resident=frozenset({"general"}),
    )
    assert decision.model == "coder"
    assert decision.reason == "code task"
    assert decision.task == "code"


def test_loaded_task_model_takes_over_for_free(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.95))
    decision = choose_model(
        profile(), prompt="Write a Python function", image=False, tools=False,
        estimated_tokens=20, current_model="general", resident=frozenset({"general", "coder"}),
    )
    assert decision.model == "coder"
    assert decision.reason == "code task"


def test_hard_requirement_still_loads_over_a_resident_model():
    decision = choose_model(
        profile(), prompt="what is in this picture", image=True, tools=False,
        estimated_tokens=20, current_model="general", resident=frozenset({"general"}),
    )
    assert decision.model == "vision"


def test_build_profile_defaults_to_a_loaded_model_and_skips_bad_rows():
    built = auto_router.build_profile(
        [
            {"id": "coder", "tasks": ["code"], "tools": True, "context_length": 4096},
            {"id": "general", "tasks": ["general"], "tools": True, "context_length": None},
            {"id": "broken", "tasks": [], "context_length": 4},
        ],
        resident=frozenset({"general"}),
    )
    assert [model.id for model in built.models] == ["coder", "general"]
    assert built.default_model == "general"
    assert auto_router.build_profile([]).models == []


def test_build_profile_handles_a_large_download_folder():
    rows = [{"id": f"model-{i}", "tasks": ["general"], "context_length": 4096} for i in range(40)]
    built = auto_router.build_profile(rows, resident=frozenset({"model-7"}))
    assert len(built.models) == 40
    assert built.default_model == "model-7"


def named_router(**slots):
    return auto_router.NamedRouter(id="coding-setup", name="Coding setup", slots=slots)


def test_named_router_needs_a_general_model():
    with pytest.raises(ValueError, match="General model"):
        named_router(code="glm")
    with pytest.raises(ValueError, match="unique"):
        auto_router.NamedRouters(routers=[named_router(general="a"), named_router(general="b")])


def test_named_router_turns_slots_into_a_profile():
    profile = auto_router.router_profile(
        named_router(code="glm", general="qwen", vision="qwen-vl", writing="qwen"),
        {"glm": {"tools": True, "context_length": 131072}, "qwen-vl": {"vision": False}},
    )
    by_id = {model.id: model for model in profile.models}
    assert by_id["glm"].tasks == ["code"]
    assert by_id["qwen"].tasks == ["writing", "general"]
    assert by_id["qwen-vl"].vision
    assert by_id["glm"].context_length == 131072
    assert profile.default_model == "qwen"


def test_named_router_loads_the_model_the_user_chose_for_the_task(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.95))
    profile = auto_router.router_profile(named_router(code="glm", general="qwen"), {})
    decision = choose_model(
        profile, prompt="Fix this Python bug", image=False, tools=False,
        estimated_tokens=20, current_model="qwen", resident=frozenset({"qwen"}),
    )
    assert decision.model == "glm"
    assert decision.reason == "code task"


def test_router_model_ids():
    assert auto_router.is_router_model("auto")
    assert auto_router.is_router_model("router/coding-setup")
    assert not auto_router.is_router_model("unsloth/Qwen3-4B")
    assert not auto_router.is_router_model(None)


def test_pinned_choice_has_no_fallbacks():
    assert route(pin="coder").fallbacks == ()
