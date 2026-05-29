"""Unit tests for strong-mock statistical gates."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.check_strong_mocks as gate
from tools.strong_mocks.policies import _rollout_equity


def _args(**overrides):
    values = {
        "candidate": ["ensemble"],
        "baseline": ["shark"],
        "mode": ["heads-up"],
        "seed_count": 2,
        "seed_start": 100,
        "hands": 40,
        "min_mean_delta": 500.0,
        "min_win_rate": 0.55,
        "min_tasks_per_candidate": 512,
        "workers": 1,
        "parallel_backend": "process",
        "allow_smoke": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_strong_mock_gate_rejects_tiny_non_smoke_budget():
    with pytest.raises(SystemExit):
        gate.run(_args())


def test_strong_mock_gate_smoke_can_exercise_pass_logic(monkeypatch):
    def fake_map_parallel(fn, tasks, workers, backend):
        return [
            {
                "candidate": task["candidate"],
                "suite": task["suite"],
                "seed": task["seed"],
                "hands": task["hands"],
                "delta": 900,
                "busted": False,
                "error_count": 0,
                "errors": [],
                "duration_s": 0.0,
            }
            for task in tasks
        ]

    monkeypatch.setattr(gate, "map_parallel", fake_map_parallel)

    report = gate.run(_args(allow_smoke=True, min_tasks_per_candidate=1))

    assert report["passed"]
    assert report["candidates"]["ensemble"]["summary"]["runs"] == 2


def test_rollout_equity_is_deterministic_for_same_state():
    state = {
        "community_cards": ["Ah", "7d", "2c"],
        "your_cards": ["As", "Kd"],
        "players": [
            {"is_folded": False},
            {"is_folded": False},
            {"is_folded": False},
        ],
    }

    first = _rollout_equity(state, samples=40)
    second = _rollout_equity(state, samples=40)

    assert first == second
