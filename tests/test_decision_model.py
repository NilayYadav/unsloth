import json

import pytest

torch = pytest.importorskip("torch")
laya = pytest.importorskip("laya")

from unsloth import DecisionTrainer, FastDecisionModel
from unsloth.models.decision import DecisionDataCollator, _target

WORDS = "the server is down again refund my card charge twice please help now".split()
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
        "questions": QUESTIONS,
        "gold": {
            "urgent": {"label": "true" if outage else "false"},
            "team": {"label": "outage" if outage else "billing"},
            "mood": 2 if outage else 0,
        },
    }


@pytest.fixture
def checkpoint(tmp_path):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, PreTrainedTokenizerFast

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
    ).save_pretrained(str(tmp_path / "base" / "tokenizer"))
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
    ).save_pretrained(str(tmp_path / "base" / "encoder"))
    config = {
        "encoder": "tiny",
        "head_layers": 1,
        "act_costs": {"escalate": 0.5},
        "max_len": 96,
        "head_max_len": 48,
        "temperature": [1.2, 1.1, 1.3],
        "temperature_by_options": {"noul:2": 0.7, "score:3-5": 0.9},
    }
    torch.manual_seed(0)
    model = laya.common.build_model(config, encoder_dir = str(tmp_path / "base" / "encoder"))
    save_file(
        {k: v.half().contiguous() for k, v in model.state_dict().items()},
        str(tmp_path / "base" / "model.safetensors"),
    )
    (tmp_path / "base" / "rl_agent_config.json").write_text(json.dumps(config))
    return tmp_path / "base"


def _internal(question):
    return laya.agent.Agent._to_internal(question)


def test_targets_accept_labels_soft_gold_and_one_sided_noul():
    noul = _internal(QUESTIONS["urgent"])
    assert _target(noul, {"probabilities": {"true": 0.3}}) == ([pytest.approx(0.7), 0.3], 0)
    assert _target(noul, {"noul": 0.8}) == ([pytest.approx(0.2), 0.8], 1)
    assert _target(_internal(QUESTIONS["mood"]), {"label": 1.6})[1] == 2
    team = _internal(QUESTIONS["team"])
    assert _target(team, {"probabilities": {"outage": 3, "billing": 1}}) == ([0.75, 0.25], 0)


def test_dataset_holdout_and_collator(checkpoint):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    assert model.decision_config["max_len"] == 1024
    assert model.decision_config["head_max_len"] == 256

    rows = [_row(i) for i in range(50)] + [{"state": "x", "questions": "{}", "gold": {}}]
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    assert len(items) == 150 and report["skipped"] == 0

    bad = [{"state": "x", "questions": {"q": {"type": "maybe"}}, "gold": {"q": "yes"}}]
    assert FastDecisionModel.build_dataset(bad, tokenizer, model)[1]["skipped"] == 1

    train, held = FastDecisionModel.split_holdout(items, 3407)
    assert len(held) == 15
    assert not {i["row"] for i in train} & {i["row"] for i in held}

    batch = DecisionDataCollator(tokenizer.pad_token_id)(items[:3])
    assert batch["marker_mask"].sum(-1).tolist() == [2, 2, 3]
    assert torch.allclose(batch["target"].sum(-1), torch.ones(3))


@pytest.mark.parametrize("lora", [True, False])
def test_train_calibrate_save_and_load_with_laya(checkpoint, tmp_path, lora):
    from transformers import TrainingArguments

    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = not lora, use_gradient_checkpointing = False
    )
    if lora:
        model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 4)
        targets = {
            n.split(".")[-1] for n, m in model.encoder.named_modules() if hasattr(m, "lora_A")
        }
        assert targets == {"Wqkv", "Wo", "Wi"}
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(80)], tokenizer, model)
    train, held = FastDecisionModel.split_holdout(items, 3407, fraction = 0.5)
    trainer = DecisionTrainer(
        model = model,
        args = TrainingArguments(
            output_dir = str(tmp_path / "run"),
            per_device_train_batch_size = 8,
            max_steps = 4,
            learning_rate = 1e-3,
            report_to = "none",
            save_strategy = "no",
        ),
        train_dataset = train,
        eval_dataset = held,
        processing_class = tokenizer,
    )
    trainer.train()
    assert trainer.evaluate()["eval_loss"] > 0

    metrics = FastDecisionModel.calibrate(trainer.model, tokenizer, held)
    assert 0 <= metrics["accuracy"] <= 1 and metrics["fitted_types"] == [0, 1, 2]
    assert "temperature_by_options" not in trainer.model.decision_config

    FastDecisionModel.save_pretrained(trainer.model, tokenizer, tmp_path / "out", {"steps": 4})
    saved = json.loads((tmp_path / "out" / "rl_agent_config.json").read_text())
    assert saved["fine_tuned"] is True and saved["training"] == {"steps": 4}
    agent = laya.load(str(tmp_path / "out"), device = "cpu")
    answer = agent.predict("the server is down again", QUESTIONS)["answers"]
    assert answer["team"]["choice"] in ("outage", "billing")


def test_gradient_accumulation_matches_one_large_batch(checkpoint, tmp_path):
    from transformers import TrainerCallback, TrainingArguments

    grads = {}

    class Grab(TrainerCallback):
        def __init__(self, key):
            self.key = key

        def on_pre_optimizer_step(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            grads[self.key] = torch.cat(
                [p.grad.flatten() for p in model.parameters() if p.grad is not None]
            ).norm()

    for accumulation in (1, 4):
        torch.manual_seed(0)
        model, tokenizer = FastDecisionModel.from_pretrained(
            str(checkpoint), full_finetuning = True, use_gradient_checkpointing = False
        )
        model.eval()
        items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(16)], tokenizer, model)
        trainer = DecisionTrainer(
            model = model,
            args = TrainingArguments(
                output_dir = str(tmp_path / f"ga{accumulation}"),
                per_device_train_batch_size = 48 // accumulation,
                gradient_accumulation_steps = accumulation,
                max_steps = 1,
                max_grad_norm = 0.0,
                report_to = "none",
                save_strategy = "no",
                use_cpu = True,
            ),
            train_dataset = items,
            processing_class = tokenizer,
            callbacks = [Grab(accumulation)],
        )
        trainer._get_train_sampler = lambda *a, **k: torch.utils.data.SequentialSampler(items)
        trainer.model.train = lambda mode = True: trainer.model
        trainer.train()
    assert torch.allclose(grads[1], grads[4], rtol = 1e-3)


def test_steps_mix_lengths_while_micro_batches_stay_grouped():
    from unsloth.models.decision import _LengthGroupedBatches

    lengths = [10] * 64 + [500] * 64
    sampler = _LengthGroupedBatches(lengths, 8, 3407)
    order = list(sampler)
    assert sorted(order) == list(range(128))
    batches = [order[i : i + 8] for i in range(0, 128, 8)]
    assert all(len({lengths[i] for i in batch}) == 1 for batch in batches)
    assert lengths[batches[0][0]] == 500
    first_step = {lengths[i] for batch in batches[:8] for i in batch}
    assert first_step == {10, 500}
    assert list(sampler) != order
