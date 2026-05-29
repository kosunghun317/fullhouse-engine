"""Unit tests for the all-strong-mock training orchestrator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.check_strong_mocks as gate
from tools.strong_mocks import train_all_strong_mocks as all_mocks


def test_strong_mock_gate_accepts_custom_candidate_paths():
    tasks = gate.build_tasks(
        candidates=["ensemble"],
        baselines=["shark"],
        seeds=[10],
        hands=25,
        modes=["heads-up"],
        candidate_paths={"ensemble": "runs/tmp/ensemble"},
    )

    assert tasks[0]["bot_paths"]["candidate"] == "runs/tmp/ensemble"


def test_all_strong_mock_budget_rejects_tiny_serious_run():
    args = all_mocks.build_parser().parse_args([
        "--ppo-generations",
        "1",
        "--ppo-matches-per-generation",
        "2",
        "--ppo-hands",
        "40",
    ])

    with pytest.raises(SystemExit):
        all_mocks.validate_orchestration_budget(args)


def test_all_strong_mock_smoke_orchestrates_train_tune_and_gate(tmp_path, monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def fake_imitation(samples, seed, style, output, hidden=48):
        calls.append(("oracle_imitation", str(output)))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        return {"output": str(output), "samples": samples, "seed": seed, "style": style, "train_accuracy": 1.0}

    def fake_real(args):
        calls.append((args.kind, str(args.output)))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        return {
            "kind": args.kind,
            "output": str(args.output),
            "generations": args.generations,
            "matches_per_generation": args.matches_per_generation,
            "hands": args.hands,
            "best_mean_train_delta": 1000.0,
        }

    def fake_deep(args):
        calls.append(("ppo_deep_policy", str(args.output)))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        return {"kind": "deep_ppo", "output": str(args.output), "generations": args.generations}

    def fake_arm(args):
        calls.append(("heuristic_rl_selector", str(args.output)))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        return {"kind": "heuristic_arm_selector", "output": str(args.output), "generations": args.generations}

    def fake_tune(args):
        calls.append(("arm_selector_params", str(args.promote_output)))
        return {
            "mode": "cem_racing_arm_selector_param_tuning",
            "best_params": str(Path(args.output_root) / args.run_id / "best_params.npz"),
            "passed": True,
        }

    def fake_gate(args):
        assert args.candidate_paths is not None
        assert args.candidate_paths["ensemble"].startswith(str(tmp_path))
        calls.append(("strong_mock_gate", None))
        return {"passed": True, "tasks": 7, "candidates": {}}

    monkeypatch.setattr(all_mocks.train_imitation, "train", fake_imitation)
    monkeypatch.setattr(all_mocks.train_real_policy, "train", fake_real)
    monkeypatch.setattr(all_mocks.train_deep_ppo, "train", fake_deep)
    monkeypatch.setattr(all_mocks.train_arm_selector, "train", fake_arm)
    monkeypatch.setattr(all_mocks.tune_arm_selector_params, "tune", fake_tune)
    monkeypatch.setattr(all_mocks.strong_gate, "run", fake_gate)

    args = all_mocks.build_parser().parse_args([
        "--run-id",
        "smoke",
        "--output-root",
        str(tmp_path),
        "--allow-smoke",
        "--oracle-samples",
        "10",
        "--ppo-generations",
        "1",
        "--ppo-matches-per-generation",
        "1",
        "--ppo-hands",
        "20",
        "--bucket-generations",
        "1",
        "--bucket-matches-per-generation",
        "1",
        "--bucket-hands",
        "20",
        "--deep-generations",
        "1",
        "--deep-matches-per-generation",
        "1",
        "--deep-hands",
        "20",
        "--arm-generations",
        "1",
        "--arm-matches-per-generation",
        "1",
        "--arm-hands",
        "20",
        "--arm-tune-generations",
        "1",
        "--arm-tune-population",
        "4",
        "--arm-tune-stages",
        "1:20:1.0",
        "--gate-seed-count",
        "1",
        "--gate-hands",
        "20",
        "--gate-min-tasks-per-candidate",
        "1",
        "--workers",
        "1",
        "--parallel-backend",
        "thread",
    ])

    summary = all_mocks.run(args)

    assert summary["passed"] is True
    assert (tmp_path / "smoke" / "summary.json").is_file()
    assert [name for name, _output in calls] == [
        "oracle_imitation",
        "ppo",
        "bucket",
        "ppo_deep_policy",
        "heuristic_rl_selector",
        "arm_selector_params",
        "strong_mock_gate",
    ]
    assert str(tmp_path / "smoke" / "artifacts") in calls[0][1]
