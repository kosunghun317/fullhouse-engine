"""Unit tests for coevolution training helpers."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.coevolve_training import (
    _summary_for,
    load_resume_state,
    resolve_ppo_init_mode,
    save_resume_state,
)
from tools.plot_training_progress import render_svg
from tools.strong_mocks.self_train_heuristic import parse_extra_opponents


def test_summary_for_penalizes_busts_and_errors():
    results = [
        {
            "chip_delta": {"candidate": 1200},
            "final_stacks": {"candidate": 11200},
            "bot_errors": {"candidate": []},
        },
        {
            "chip_delta": {"candidate": -10000},
            "final_stacks": {"candidate": 0},
            "bot_errors": {"candidate": ["exception"]},
        },
    ]

    summary = _summary_for(results, "candidate")

    assert summary["runs"] == 2
    assert summary["bust_count"] == 1
    assert summary["error_count"] == 1
    assert summary["score"] < summary["mean_delta"]


def test_parse_extra_opponents_accepts_labelled_and_plain_paths():
    specs = parse_extra_opponents(["latest=/tmp/ppo", "/tmp/plain_bot"])

    assert specs[0] == ("latest", "/tmp/ppo")
    assert specs[1][1] == "/tmp/plain_bot"


def test_render_svg_from_metrics_jsonl(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"cycle": 0, "series": "ppo_stable", "mean_delta": -100}\n'
        '{"cycle": 1, "series": "ppo_stable", "mean_delta": 250}\n'
        '{"cycle": 1, "series": "heuristic_selected", "mean_delta": 400}\n',
        encoding="utf-8",
    )
    output = tmp_path / "plot.svg"

    report = render_svg(metrics, output)

    assert output.is_file()
    assert "ppo_stable" in report["series"]
    assert "<svg" in output.read_text(encoding="utf-8")


def test_auto_ppo_init_bootstraps_then_resumes_latest():
    assert resolve_ppo_init_mode("auto", bootstrap_cycle=True) == "oracle"
    assert resolve_ppo_init_mode("auto", bootstrap_cycle=False) == "latest"
    assert resolve_ppo_init_mode("fresh", bootstrap_cycle=True) == "fresh"


def test_resume_state_round_trip(tmp_path: Path):
    run_root = tmp_path / "run"
    state_path = run_root / "state.json"
    rng = random.Random(123)
    population = [{"name": "candidate", "env": {"A": "1"}}]

    saved = save_resume_state(
        state_path,
        args=SimpleNamespace(run_id="resume-test"),
        run_root=run_root,
        metrics_path=run_root / "metrics.jsonl",
        plot_path=run_root / "ev_progress.svg",
        latest_ppo=run_root / "ppo",
        latest_heuristic=run_root / "heuristic",
        next_cycle=3,
        population=population,
        rng=rng,
    )
    loaded = load_resume_state(run_root)

    assert loaded == saved
    assert loaded["next_cycle"] == 3
    assert loaded["population"] == population
    assert loaded["latest_ppo"].endswith("/ppo")
