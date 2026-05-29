"""Unit tests for heuristic self-training output artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.strong_mocks.self_train_heuristic as self_train


def _args(**overrides):
    defaults = {
        "run_id": "unit-self-train",
        "run_root": None,
        "generated_root": None,
        "result_root": None,
        "generations": 2,
        "population": 4,
        "elite": 2,
        "matches_per_generation": 2,
        "hands": 8,
        "min_heuristics": 2,
        "max_heuristics": 3,
        "workers": 1,
        "parallel_backend": "thread",
        "seed": 17,
        "extra_opponent": [],
        "allow_smoke": True,
        "json": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_output_paths_prefers_exact_run_root(tmp_path: Path):
    args = _args(
        run_root=str(tmp_path / "run"),
        result_root=str(tmp_path / "legacy_results"),
        generated_root=str(tmp_path / "custom_generated"),
    )

    paths = self_train.resolve_output_paths(args)

    assert paths["run_root"] == tmp_path / "run"
    assert paths["generated_dir"] == tmp_path / "custom_generated"
    assert paths["summary"] == tmp_path / "run" / "summary.json"
    assert paths["metrics"] == tmp_path / "run" / "metrics.jsonl"
    assert paths["plot"] == tmp_path / "run" / "ev_progress.svg"


def test_resolve_output_paths_preserves_legacy_base_roots(tmp_path: Path):
    args = _args(
        run_root=None,
        result_root=str(tmp_path / "results"),
        generated_root=str(tmp_path / "generated"),
    )

    paths = self_train.resolve_output_paths(args)

    assert paths["run_root"] == tmp_path / "results" / "unit-self-train"
    assert paths["generated_dir"] == tmp_path / "generated" / "unit-self-train"


def test_self_training_budget_rejects_tiny_non_smoke_run():
    args = _args(allow_smoke=False)

    with pytest.raises(SystemExit):
        self_train.validate_training_budget(args)


def test_run_self_training_writes_summary_metrics_and_plot(tmp_path: Path, monkeypatch):
    def fake_evaluate_generation(generated_dir, run_id, generation, population, args, rng, extra_opponents=None):
        ranked = []
        for index, item in enumerate(population):
            mean_delta = 1000.0 + generation * 100.0 - index * 50.0
            ranked.append({
                "name": item["name"],
                "env": item.get("env", {}),
                "games": 2,
                "mean_delta": mean_delta,
                "min_delta": int(mean_delta - 75),
                "values": [int(mean_delta - 75), int(mean_delta + 75)],
            })
        ranked.sort(key=lambda row: (row["mean_delta"], row["min_delta"]), reverse=True)
        return {"generation": generation, "ranked": ranked, "matches": []}

    monkeypatch.setattr(self_train, "evaluate_generation", fake_evaluate_generation)
    args = _args(run_root=str(tmp_path / "run"))

    report = self_train.run_self_training(args)

    run_root = tmp_path / "run"
    assert report["run_root"] == str(run_root)
    assert report["generated_dir"] == str(run_root / "generated")
    assert (run_root / "summary.json").is_file()
    assert (run_root / "metrics.jsonl").is_file()
    assert (run_root / "ev_progress.svg").is_file()
    assert (run_root / "generation_000.json").is_file()
    assert (run_root / "generation_001.json").is_file()
    assert (run_root / "population_001.json").is_file()
    assert (run_root / "population_002.json").is_file()

    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    metric_lines = (run_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    svg = (run_root / "ev_progress.svg").read_text(encoding="utf-8")

    assert summary["plot"] == str(run_root / "ev_progress.svg")
    assert len(metric_lines) == 4
    assert any('"series": "generation_best"' in line for line in metric_lines)
    assert "<svg" in svg
    assert "generation_best" in svg
