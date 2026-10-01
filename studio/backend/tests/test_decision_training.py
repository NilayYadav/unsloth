# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import shutil
import sys
import time

import pytest

torch = pytest.importorskip("torch")
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors import safe_open

from auth.authentication import get_current_subject
from core.systemone import catalog, laya_runtime

WORDS = (
    "the server is down again refund my card charge twice please help now "
    "angry calm fine login broken slow outage billing account password"
).split()
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this need a reply now?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {"outage": "service down", "billing": "charges and refunds"},
    },
    "mood": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "angry"],
    },
}


def _row(i):
    outage = i % 2 == 0
    return {
        "state": "the server is down again help now" if outage else "refund my card charge twice",
        "questions": json.dumps(QUESTIONS),
        "gold": json.dumps(
            {
                "urgent": {"label": "true" if outage else "false"},
                "team": {
                    "label": "outage" if outage else "billing",
                    "probabilities": {"outage": 0.9, "billing": 0.1}
                    if outage
                    else {"outage": 0.2, "billing": 0.8},
                },
                "mood": {"label": 2 if outage else 0},
            }
        ),
    }


def _base_checkpoint(folder):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, PreTrainedTokenizerFast
    from safetensors.torch import save_file

    laya = laya_runtime._laya()
    specials = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    vocab = {token: i for i, token in enumerate(specials + sorted(set(WORDS)))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        pad_token = "[PAD]",
        sep_token = "[SEP]",
        cls_token = "[CLS]",
        unk_token = "[UNK]",
        mask_token = "[MASK]",
    ).save_pretrained(str(folder / "tokenizer"))
    ModernBertConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        pad_token_id = 0,
        cls_token_id = 2,
        sep_token_id = 1,
        global_attn_every_n_layers = 2,
        local_attention = 16,
    ).save_pretrained(str(folder / "encoder"))
    cfg = {
        "encoder": "tiny",
        "head_layers": 1,
        "act_costs": {"escalate": 0.5},
        "max_len": 96,
        "head_max_len": 48,
        "temperature": [1.2, 1.1, 1.3],
        "temperature_by_options": {"noul:2": 0.1},
    }
    torch.manual_seed(0)
    model = laya.common.build_model(cfg, encoder_dir = str(folder / "encoder"))
    save_file(
        {k: v.half().contiguous() for k, v in model.state_dict().items()},
        str(folder / "model.safetensors"),
    )
    (folder / "rl_agent_config.json").write_text(json.dumps(cfg), encoding = "utf-8")
    return folder


@pytest.fixture
def studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    for name in (
        "UNSLOTH_SYSTEMONE_MODEL",
        "UNSLOTH_SYSTEMONE_DISABLE",
        "UNSLOTH_SYSTEMONE_DEVICE",
    ):
        monkeypatch.delenv(name, raising = False)
    for name in ("_agent", "_loaded", "_device_name", "_loader", "_loading", "_failure"):
        monkeypatch.setattr(laya_runtime, name, None)
    yield tmp_path
    if laya_runtime._loader is not None:
        laya_runtime._loader.join(30)


@pytest.fixture
def base(studio_home):
    return _base_checkpoint(studio_home / "base")


def _dataset(folder, rows):
    path = folder / "decisions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding = "utf-8")
    return str(path)


def _config(base, dataset, **overrides):
    return {
        "model_name": str(base),
        "model_subfolder": None,
        "is_decision": True,
        "hf_dataset": "",
        "local_datasets": [dataset],
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_epochs": 1,
        "max_steps": 3,
        "learning_rate": "1e-3",
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "random_seed": 3407,
        "gradient_checkpointing": "none",
        "eval_steps": 0,
        "project_name": None,
        **overrides,
    }


needs_worker = pytest.mark.skipif(sys.platform == "darwin", reason = "Apple Silicon trains with MLX")


def _train(config, stop = None):
    import multiprocessing as mp

    from core.training.training import _build_training_worker_config
    from core.training.worker import run_training_process

    context = mp.get_context("spawn")
    events, stops = context.Queue(), context.Queue()
    process = context.Process(
        target = run_training_process,
        kwargs = {
            "event_queue": events,
            "stop_queue": stops,
            "config": _build_training_worker_config({"training_type": "Full Finetuning", **config}),
        },
    )
    process.start()
    received = []
    try:
        while not received or received[-1]["type"] not in ("complete", "error"):
            received.append(events.get(timeout = 600))
            if stop is not None and received[-1]["type"] == "progress":
                stops.put({"type": "stop", "save": stop})
                stop = None
    finally:
        process.join(60)
    return received


def _of(events, kind):
    return [e for e in events if e["type"] == kind]


@pytest.fixture
def client(studio_home):
    from routes import systemone
    from routes.settings import router as settings_router

    app = FastAPI()
    app.include_router(systemone.router, prefix = "/v1")
    app.include_router(settings_router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def _decide(client, model = "default"):
    for _ in range(120):
        response = client.post(
            "/v1/systemone",
            json = {"model": model, "state": "the server is down again", "questions": QUESTIONS},
        )
        if response.status_code != 503:
            return response
        time.sleep(0.5)
    return response


@needs_worker
def test_fine_tune_is_calibrated_saved_and_served(base, studio_home, client):
    rows = [_row(i) for i in range(120)]
    rows += [
        {"state": "slow login", "questions": "not json", "gold": "{}"},
        {"state": "slow login", "questions": json.dumps(QUESTIONS), "gold": json.dumps({})},
        {
            "state": "slow login",
            "questions": json.dumps({"q": {"type": "maybe", "instructions": "?"}}),
            "gold": json.dumps({"q": "yes"}),
        },
    ]
    events = _train(_config(base, _dataset(studio_home, rows)))

    assert not _of(events, "error"), _of(events, "error")
    progress = _of(events, "progress")
    assert [e["step"] for e in progress] == [1, 2, 3]
    assert all(e["total_steps"] == 3 and e["loss"] > 0 for e in progress)
    assert progress[-1]["eval_loss"] is not None
    assert _of(events, "warning")[0]["message"].startswith("Skipped 5 of 365 decisions: row 121")
    complete = _of(events, "complete")[-1]
    assert complete["status_message"].startswith("Held-out accuracy ")
    output = complete["output_dir"]
    assert _of(events, "output_dir")[0]["output_dir"] == output

    from utils.paths import outputs_root

    folder = outputs_root() / output.rsplit("/", 1)[-1]
    assert str(folder) == output
    cfg = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
    assert cfg["fine_tuned"] is True
    assert "temperature_by_options" not in cfg
    assert cfg["max_len"] == 96 and cfg["head_max_len"] == 48
    assert all(0.5 <= t <= 5.0 for t in cfg["temperature"])
    assert cfg["temperature"] != [1.2, 1.1, 1.3]
    assert cfg["training"]["heldout_decisions"] == 36
    assert cfg["training"]["steps"] == 3
    with safe_open(str(folder / "model.safetensors"), "pt") as weights:
        assert {weights.get_tensor(k).dtype for k in weights.keys()} == {torch.float16}

    name = catalog.FINE_TUNE_PREFIX + folder.name
    settings = client.get("/api/settings/systemone").json()
    assert {"name": name, "kind": "fine_tune", "label": folder.name}.items() <= next(
        m for m in settings["models"] if m["name"] == name
    ).items()
    response = client.put("/api/settings/systemone", json = {"enabled": True, "model": name})
    assert response.status_code == 200, response.text
    assert response.json()["model"] == name
    plan = client.get("/api/settings/systemone/resolve").json()
    assert plan["cached"] is True and plan["repo"] is None

    response = _decide(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == name
    assert set(body["answers"]) == set(QUESTIONS)
    assert body["answers"]["team"]["choice"] in ("outage", "billing")
    assert _decide(client, name).json()["model"] == name


@needs_worker
def test_lora_run_merges_into_the_served_layout(base, studio_home, client):
    from safetensors.torch import load_file

    rows = [_row(i) for i in range(120)]
    events = _train(
        _config(
            base, _dataset(studio_home, rows), training_type = "LoRA/QLoRA", lora_r = 4, lora_alpha = 4
        )
    )

    assert not _of(events, "error"), _of(events, "error")
    folder = catalog.fine_tune(
        catalog.FINE_TUNE_PREFIX + _of(events, "complete")[-1]["output_dir"].rsplit("/", 1)[-1]
    )
    cfg = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
    assert cfg["training"]["method"] == "lora"
    tuned = load_file(str(folder / "model.safetensors"))
    original = load_file(str(base / "model.safetensors"))
    assert tuned.keys() == original.keys()
    assert {t.dtype for t in tuned.values()} == {torch.float16}
    assert not torch.equal(
        tuned["encoder.layers.0.attn.Wqkv.weight"], original["encoder.layers.0.attn.Wqkv.weight"]
    )
    assert torch.equal(
        tuned["encoder.embeddings.tok_embeddings.weight"],
        original["encoder.embeddings.tok_embeddings.weight"],
    )

    name = catalog.FINE_TUNE_PREFIX + folder.name
    response = client.put("/api/settings/systemone", json = {"enabled": True, "model": name})
    assert response.status_code == 200, response.text
    response = _decide(client)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == name


@needs_worker
def test_all_invalid_rows_are_an_error_and_save_nothing(base, studio_home):
    from utils.paths import outputs_root

    rows = [{"state": "slow", "questions": json.dumps(QUESTIONS)}] * 4
    events = _train(_config(base, _dataset(studio_home, rows)))

    assert "No usable decisions" in _of(events, "error")[0]["error"]
    assert not _of(events, "complete")
    assert not outputs_root().exists() or not any(outputs_root().iterdir())


@needs_worker
@pytest.mark.parametrize("training_type", ["Full Finetuning", "LoRA/QLoRA"])
def test_stop_with_save_leaves_a_servable_checkpoint(base, studio_home, training_type):
    rows = [_row(i) for i in range(120)]
    events = _train(
        _config(
            base,
            _dataset(studio_home, rows),
            max_steps = 0,
            num_epochs = 3,
            training_type = training_type,
        ),
        True,
    )

    complete = _of(events, "complete")[-1]
    last = _of(events, "progress")[-1]
    assert last["step"] < last["total_steps"]
    folder = catalog.fine_tune(catalog.FINE_TUNE_PREFIX + complete["output_dir"].rsplit("/", 1)[-1])
    assert folder is not None and laya_runtime.is_cached(folder)
    with (
        safe_open(str(folder / "model.safetensors"), "pt") as tuned,
        safe_open(str(base / "model.safetensors"), "pt") as original,
    ):
        assert set(tuned.keys()) == set(original.keys())


@needs_worker
def test_cancel_leaves_no_weights(base, studio_home):
    from utils.paths import outputs_root

    rows = [_row(i) for i in range(120)]
    events = _train(_config(base, _dataset(studio_home, rows), max_steps = 0, num_epochs = 3), False)

    complete = _of(events, "complete")[-1]
    assert complete["output_dir"] is None
    assert complete["status_message"] == "Training cancelled"
    assert not outputs_root().exists() or not any(outputs_root().rglob("model.safetensors"))


def _fake_output(
    root,
    name,
    complete = True,
):
    folder = root / name
    for sub in ("encoder", "tokenizer"):
        (folder / sub).mkdir(parents = True)
    (folder / "model.safetensors").write_bytes(b"x")
    if complete:
        (folder / "rl_agent_config.json").write_text("{}", encoding = "utf-8")
    return folder


def test_settings_accept_only_complete_owner_fine_tunes(studio_home, client, tmp_path):
    from utils.paths import outputs_root

    root = outputs_root()
    _fake_output(root, "laya_done_1")
    _fake_output(root, "laya_half_2", complete = False)
    outside = _fake_output(tmp_path, "elsewhere")
    (root / "linked").symlink_to(outside, target_is_directory = True)

    names = [m["name"] for m in client.get("/api/settings/systemone").json()["models"]]
    assert "laya-ft:laya_done_1" in names
    assert not any(n in names for n in ("laya-ft:laya_half_2", "laya-ft:linked"))
    for bad in (
        "laya-ft:laya_half_2",
        "laya-ft:linked",
        "laya-ft:../elsewhere",
        f"laya-ft:{outside}",
        "laya-ft:",
    ):
        response = client.put("/api/settings/systemone", json = {"model": bad})
        assert response.status_code == 400, bad

    response = client.put("/api/settings/systemone", json = {"model": "laya-ft:laya_done_1"})
    assert response.status_code == 200
    assert client.get("/api/settings/systemone").json()["model"] == "laya-ft:laya_done_1"
    shutil.rmtree(root / "laya_done_1")
    assert client.get("/api/settings/systemone").json()["model"] == "laya-multilingual"


def test_llm_output_scans_skip_decision_outputs(studio_home):
    from utils.models.checkpoints import scan_checkpoints
    from utils.models.model_config import scan_trained_models
    from utils.paths import outputs_root

    root = outputs_root()
    _fake_output(root, "laya_done_1")
    merged = root / "llama_merged_1"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding = "utf-8")
    (merged / "model.safetensors").write_bytes(b"x")

    assert [name for name, _, _ in scan_trained_models(str(root))] == ["llama_merged_1"]
    assert [name for name, _, _ in scan_checkpoints(str(root))] == ["llama_merged_1"]


def test_model_config_classifies_a_local_laya_folder(base):
    import asyncio

    from routes.models import get_model_config

    result = asyncio.run(
        get_model_config(model_name = str(base), hf_token = None, current_subject = "tester")
    )
    assert result.model_type == "decision" and result.is_decision is True
    assert float(result.config["training"]["learning_rate"]) == 8e-4
    assert result.config["lora"]["lora_r"] == 64
    assert result.decision_checkpoints is None


def test_start_preflight_accepts_a_local_laya_folder(base):
    from models.training import TrainingStartRequest
    from routes.training import _reject_untrainable_model_request

    request = TrainingStartRequest(
        model_name = str(base),
        training_type = "Full Finetuning",
        is_decision = True,
        hf_dataset = "org/decisions",
        format_type = "auto",
    )
    assert _reject_untrainable_model_request(request).model_name == str(base.resolve())
