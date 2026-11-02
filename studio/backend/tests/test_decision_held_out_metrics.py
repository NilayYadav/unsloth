# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

import core.training.training as tr
from core.training import decision_trainer
from core.training.training import TrainingBackend
from storage import studio_db

QUESTIONS_SHOWN = 8
OPTIONS_SHOWN = 20
REPORT = {
    "decisions": 60,
    "accuracy_base": 0.5,
    "accuracy": 0.9,
    "ece_base": 0.3,
    "ece": 0.05,
    "questions_total": 1,
    "questions": [
        {"name": "urgent", "confusion": {"true": {"true": 27, "false": 3}, "false": {"false": 30}}}
    ],
}


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(tr, "account_is_retired", lambda: False)
    monkeypatch.setattr(TrainingBackend, "_start_stop_watchdog", lambda *a, **k: None)
    studio_db.create_run(
        id = "run-1",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00+00:00",
        total_steps = 3,
    )
    b = TrainingBackend()
    b.current_job_id = "run-1"
    b._db_run_created = True
    yield b
    studio_db.close_wal_keeper()


def _complete(**extra):
    return {"type": "complete", "output_dir": "/tmp/run", "status_message": "done", **extra}


def test_finished_run_keeps_its_held_out_results(backend):
    backend._handle_event(_complete(decision_metrics = REPORT))

    assert backend._progress.decision_metrics == REPORT
    run = studio_db.get_run("run-1")
    assert run["status"] == "completed" and run["decision_metrics"] == REPORT


def test_a_watchdog_finish_keeps_the_held_out_results_too(backend):
    backend._progress.decision_metrics = REPORT

    backend._finish_stopped_run(
        "run-1", None, [], 3, None, None, [], status = "stopped", decision_metrics = REPORT
    )

    assert studio_db.get_run("run-1")["decision_metrics"] == REPORT


def test_a_run_without_held_out_results_has_none(backend):
    backend._handle_event(_complete())

    assert studio_db.get_run("run-1")["decision_metrics"] is None


def test_a_database_from_an_older_install_gains_the_column(backend):
    conn = studio_db.get_connection()
    conn.execute("ALTER TABLE training_runs DROP COLUMN decision_metrics")
    conn.commit()
    conn.close()
    studio_db._schema_ready.clear()

    backend._handle_event(_complete(decision_metrics = REPORT))

    assert studio_db.get_run("run-1")["decision_metrics"] == REPORT


def _report(base, tuned, decisions):
    return decision_trainer._heldout_report(base, tuned, decisions)


def _metrics(confusion, ece = 0.1):
    return {"accuracy": 0.87654, "ece": ece, "confusion": confusion}


def test_report_rounds_the_scores_and_keeps_the_counts():
    confusion = {"urgent": {"true": {"true": 9, "false": 1}}}

    report = _report(_metrics({}), _metrics(confusion), 10)

    assert report["accuracy"] == 0.8765
    assert report["questions_total"] == 1
    assert report["questions"] == [{"name": "urgent", "confusion": confusion["urgent"]}]


def test_report_keeps_the_most_asked_questions():
    confusion = {f"q{i}": {"a": {"a": i + 1}} for i in range(QUESTIONS_SHOWN + 3)}

    report = _report(_metrics({}), _metrics(confusion), 100)

    assert report["questions_total"] == QUESTIONS_SHOWN + 3
    names = [question["name"] for question in report["questions"]]
    assert names == [f"q{i}" for i in range(QUESTIONS_SHOWN + 2, 2, -1)]


def test_report_leaves_out_a_question_too_wide_to_table():
    wide = {str(i): {"0": 1} for i in range(OPTIONS_SHOWN + 1)}

    report = _report(_metrics({}), _metrics({"pick": wide, "ok": {"a": {"a": 1}}}), 100)

    assert report["questions_total"] == 2
    assert [question["name"] for question in report["questions"]] == ["ok"]


def test_report_survives_a_run_whose_scores_went_non_finite():
    report = _report(_metrics({}, float("nan")), _metrics({}, float("inf")), 100)

    assert report["ece_base"] is None and report["ece"] is None
    json.dumps(report, allow_nan = False)
