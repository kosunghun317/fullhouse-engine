"""Unit tests for full-space heuristic tuning helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.tune_heuristic_full_space import (
    env_from_vector,
    load_param_specs,
    parse_stages,
    vector_from_env,
    _check_budget,
)


def test_full_space_loads_most_heuristic_parameters():
    specs = load_param_specs()
    names = {spec.name for spec in specs}

    assert len(specs) >= 85
    assert "HEURISTIC_RNG_SEED" not in names
    assert "HEURISTIC_CALL_MARGIN_BASE" in names
    assert "HEURISTIC_PROFILE_TARGETING_ENABLED" in names
    assert "HEURISTIC_FLOP_SAMPLES" in names


def test_vector_round_trip_decodes_env_overrides():
    specs = load_param_specs()
    env = {
        "HEURISTIC_CALL_MARGIN_BASE": "0.095",
        "HEURISTIC_PROFILE_TARGETING_ENABLED": "0.0",
    }

    decoded = env_from_vector(specs, vector_from_env(specs, env))

    assert decoded["HEURISTIC_CALL_MARGIN_BASE"] == "0.095"
    assert decoded["HEURISTIC_PROFILE_TARGETING_ENABLED"] == "0.0"


def test_budget_guard_rejects_tiny_non_smoke_run():
    args = SimpleNamespace(
        allow_smoke=False,
        skip_final_validation=False,
        population=4,
        generations=1,
        final_validation_seed_count=256,
        final_validation_hands=400,
        min_tasks_per_candidate=512,
        min_final_tasks_per_candidate=1024,
    )

    with pytest.raises(SystemExit):
        _check_budget(args, ["strong_mock_6max"], parse_stages("2:12:0.5"))


def test_budget_guard_requires_final_validation_for_serious_run():
    args = SimpleNamespace(
        allow_smoke=False,
        skip_final_validation=True,
        population=24,
        generations=6,
        final_validation_seed_count=256,
        final_validation_hands=400,
        min_tasks_per_candidate=512,
        min_final_tasks_per_candidate=1024,
    )

    with pytest.raises(SystemExit):
        _check_budget(
            args,
            ["strong_mock_6max", "strong_hybrid_6max", "heads_up_strong_rollout", "heads_up_strong_ensemble"],
            parse_stages("128:400:0.5,256:400:0.25"),
        )
