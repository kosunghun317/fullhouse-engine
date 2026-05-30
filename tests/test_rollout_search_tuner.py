"""Tests for rollout-search f/g tuning helpers."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.strong_mocks import policies
from tools.strong_mocks import tune_rollout_search as tuner


def _state(can_check: bool = True) -> dict:
    return {
        "type": "action_request",
        "hand_id": "test",
        "street": "flop",
        "seat_to_act": 0,
        "pot": 1200,
        "current_bet": 0 if can_check else 400,
        "min_raise_to": 800,
        "amount_owed": 0 if can_check else 400,
        "can_check": can_check,
        "your_cards": ["As", "Ks"],
        "community_cards": ["Qs", "7s", "2d"],
        "your_stack": 9000,
        "your_bet_this_street": 0,
        "players": [
            {"seat": 0, "is_folded": False, "stack": 9000},
            {"seat": 1, "is_folded": False, "stack": 9000},
        ],
        "action_log": [],
    }


def test_rollout_fg_functions_are_monotone_and_actionable():
    low = policies.rollout_check_raise_fraction(0.35)
    middle = policies.rollout_check_raise_fraction(0.58)
    high = policies.rollout_check_raise_fraction(0.78)

    assert low == 0.0
    assert 0.0 < middle < high

    assert policies.rollout_facing_bet_raise_fraction(0.20, 0.50, 2, 0.25, "rollout_pressure") < 0.0
    assert policies.rollout_facing_bet_raise_fraction(0.52, 0.47, 2, 0.25, "rollout_pressure") == 0.0
    assert policies.rollout_facing_bet_raise_fraction(0.91, 0.25, 2, 0.10, "rollout_pressure") > 0.0


def test_rollout_pressure_uses_1024_samples(monkeypatch):
    seen = []

    def fake_rollout_equity(state, samples=360, budget_opponents=None, rng=None):
        seen.append(samples)
        return 0.50

    monkeypatch.setattr(policies, "_rollout_equity", fake_rollout_equity)

    action = policies.decide_rollout(_state(can_check=True), style="rollout_pressure")

    assert seen == [1024]
    assert action["action"] in {"check", "raise"}


def test_rollout_tuner_smoke_writes_best_bot(tmp_path, monkeypatch):
    def fake_run_match(match_id, bot_paths, n_hands=400, verbose=False, seed=None):
        assert tuner.CANDIDATE_ID in bot_paths
        return {
            "chip_delta": {tuner.CANDIDATE_ID: 1234},
            "final_stacks": {tuner.CANDIDATE_ID: 11234},
            "bot_errors": {tuner.CANDIDATE_ID: []},
            "duration_s": 0.01,
        }

    monkeypatch.setattr(tuner, "run_match", fake_run_match)

    args = SimpleNamespace(
        run_id="smoke",
        output_root=str(tmp_path),
        suites="heads_up_reference",
        generations=1,
        population=2,
        elite=1,
        stages="1:5:1.0",
        seed_start=10,
        seed=123,
        init_sigma_frac=0.10,
        min_sigma_frac=0.01,
        smoothing=0.5,
        bust_penalty=9000.0,
        error_penalty=25000.0,
        stdev_penalty=0.10,
        workers=1,
        parallel_backend="thread",
        latency_repeats=1,
        max_decision_seconds=2.0,
        skip_latency_check=True,
        clean=False,
        progress=False,
    )

    summary = tuner.tune(args)

    assert summary["best"]["summary"]["score"] == 1234.0
    assert (tmp_path / "smoke" / "best_params.json").is_file()
    assert (tmp_path / "smoke" / "best_bot" / "bot.py").is_file()
