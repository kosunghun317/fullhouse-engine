"""Unit tests for real-policy training guardrails."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.strong_mocks.train_real_policy import (
    DEFAULT_PPO_OUTPUT,
    Decision,
    _flatten_decisions,
    should_export_policy,
    validate_training_budget,
)
from tools.strong_mocks.actions import strategic_mask


def test_should_export_policy_allows_new_outputs_below_threshold():
    assert should_export_policy(
        selected_mean_delta=-500.0,
        output_exists=False,
        min_export_mean_delta=1500.0,
    )


def test_should_export_policy_blocks_existing_output_below_threshold():
    assert not should_export_policy(
        selected_mean_delta=1100.0,
        output_exists=True,
        min_export_mean_delta=1500.0,
    )


def test_should_export_policy_allows_existing_output_above_threshold():
    assert should_export_policy(
        selected_mean_delta=1600.0,
        output_exists=True,
        min_export_mean_delta=1500.0,
    )


def test_flatten_decisions_preserves_behavior_temperature():
    flat = _flatten_decisions([
        {
            "decisions": [
                Decision(
                    bot_id="train_0",
                    x=np.zeros(32, dtype=np.float32),
                    mask=np.ones(8, dtype=bool),
                    action=1,
                    bucket=7,
                    old_logprob=-0.5,
                    old_value=0.1,
                    temperature=0.62,
                    reward=1.0,
                ).__dict__
            ]
        }
    ])

    assert flat["temperature"].shape == (1,)
    assert flat["temperature"][0] == 0.62


def test_strategic_mask_blocks_weak_large_call_offs():
    state = {
        "street": "preflop",
        "can_check": False,
        "your_stack": 10000,
        "your_bet_this_street": 100,
        "amount_owed": 4200,
        "pot": 3000,
        "min_raise_to": 8500,
        "your_cards": ["7h", "2c"],
    }

    mask = strategic_mask(state)

    assert not mask[1]
    assert not mask[7]


def test_strategic_mask_allows_premium_large_call_offs():
    state = {
        "street": "preflop",
        "can_check": False,
        "your_stack": 10000,
        "your_bet_this_street": 100,
        "amount_owed": 4200,
        "pot": 3000,
        "min_raise_to": 8500,
        "your_cards": ["Ah", "Ac"],
    }

    mask = strategic_mask(state)

    assert mask[1]
    assert mask[7]


def test_real_policy_budget_rejects_tiny_non_smoke_run(tmp_path):
    args = SimpleNamespace(
        kind="ppo",
        allow_smoke=False,
        generations=1,
        matches_per_generation=2,
        hands=40,
        min_training_hands=1_000_000,
    )

    with pytest.raises(SystemExit):
        validate_training_budget(args, tmp_path / "policy.npz")


def test_real_policy_budget_blocks_smoke_overwriting_canonical_artifact():
    args = SimpleNamespace(
        kind="ppo",
        allow_smoke=True,
        generations=1,
        matches_per_generation=2,
        hands=40,
        min_training_hands=1_000_000,
    )

    with pytest.raises(SystemExit):
        validate_training_budget(args, DEFAULT_PPO_OUTPUT)


def test_real_policy_budget_allows_smoke_to_temp_output(tmp_path):
    args = SimpleNamespace(
        kind="ppo",
        allow_smoke=True,
        generations=1,
        matches_per_generation=2,
        hands=40,
        min_training_hands=1_000_000,
    )

    validate_training_budget(args, tmp_path / "policy.npz")
