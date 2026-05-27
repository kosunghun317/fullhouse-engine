"""Unit tests for league-training orchestration helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.strong_mocks.league_train import build_eval_tasks, should_promote, summarize_eval


def test_build_eval_tasks_rotates_pool_opponents(tmp_path):
    candidate = tmp_path / "candidate"
    tasks = build_eval_tasks(
        candidate_path=candidate,
        pool_names=["public_holdout"],
        seeds=[11, 12],
        hands=7,
        players=3,
        match_prefix="unit",
    )

    assert len(tasks) == 2
    assert tasks[0]["n_hands"] == 7
    assert tasks[0]["bot_paths"]["candidate"] == str(candidate)
    assert len(tasks[0]["bot_paths"]) == 3
    assert tasks[0]["bot_paths"] != tasks[1]["bot_paths"]


def test_summarize_eval_penalizes_busts_and_errors():
    results = [
        {
            "chip_delta": {"candidate": 1000},
            "final_stacks": {"candidate": 11000},
            "bot_errors": {"candidate": []},
        },
        {
            "chip_delta": {"candidate": -10000},
            "final_stacks": {"candidate": 0},
            "bot_errors": {"candidate": ["exception"]},
        },
    ]

    summary = summarize_eval(results)

    assert summary["runs"] == 2
    assert summary["bust_count"] == 1
    assert summary["error_count"] == 1
    assert summary["score"] < summary["mean_delta"]


def test_should_promote_requires_margin_and_zero_errors():
    incumbent = {"score": 1000, "error_count": 0}
    candidate = {"score": 1300, "error_count": 0}
    broken = {"score": 5000, "error_count": 1}

    assert should_promote(candidate, incumbent, margin=250)
    assert not should_promote(candidate, incumbent, margin=400)
    assert not should_promote(broken, incumbent, margin=250)
