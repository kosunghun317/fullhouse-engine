"""Tests for the independent deep PPO strong mock."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.strong_mocks import train_deep_ppo
from tools.strong_mocks.deep_actions import ACTION_LABELS, strategic_mask
from tools.strong_mocks.deep_policies import (
    DEEP_FEATURE_NAMES,
    deep_oracle_logits,
    decide_deep_model_sampled,
    extract_deep_features,
)


def _sample_state() -> dict:
    return {
        "type": "action_request",
        "hand_id": "deep_test_001",
        "street": "flop",
        "seat_to_act": 0,
        "pot": 900,
        "community_cards": ["As", "7d", "2c"],
        "current_bet": 0,
        "min_raise_to": 100,
        "amount_owed": 0,
        "can_check": True,
        "your_cards": ["Ah", "Kd"],
        "your_stack": 9600,
        "your_bet_this_street": 0,
        "players": [
            {"seat": 0, "bot_id": "hero", "stack": 9600, "is_folded": False, "is_all_in": False},
            {"seat": 1, "bot_id": "villain", "stack": 9400, "is_folded": False, "is_all_in": False},
        ],
        "action_log": [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 0, "action": "big_blind", "amount": 100},
            {"seat": 0, "action": "raise", "amount": 260},
            {"seat": 1, "action": "call", "amount": 260},
        ],
    }


def test_deep_action_space_has_more_arms_than_original_ppo():
    assert len(ACTION_LABELS) == 14
    assert "raise_220" in ACTION_LABELS


def test_deep_feature_vector_and_oracle_logits_match_action_space():
    x = extract_deep_features(_sample_state())
    logits = deep_oracle_logits(x.reshape(1, -1))

    assert x.shape == (len(DEEP_FEATURE_NAMES),)
    assert logits.shape == (1, len(ACTION_LABELS))


def test_deep_policy_fallback_returns_legal_shape_without_artifact(tmp_path):
    action = decide_deep_model_sampled(_sample_state(), str(tmp_path))

    assert action["action"] in {"fold", "check", "call", "raise", "all_in"}
    if action["action"] == "raise":
        assert isinstance(action["amount"], int)


def test_deep_strategic_mask_blocks_weak_large_call_offs():
    state = _sample_state()
    state.update({
        "street": "preflop",
        "can_check": False,
        "pot": 3000,
        "current_bet": 4300,
        "amount_owed": 4200,
        "min_raise_to": 8500,
        "your_cards": ["7h", "2c"],
    })

    mask = strategic_mask(state)

    assert not mask[1]
    assert not mask[-1]


def test_supervised_bootstrap_moves_deep_policy_toward_teacher():
    hidden = (16, 8)
    params = train_deep_ppo._init_params(seed=123, hidden=hidden)
    mean = np.zeros(len(DEEP_FEATURE_NAMES), dtype=np.float64)
    scale = np.ones(len(DEEP_FEATURE_NAMES), dtype=np.float64)

    result = train_deep_ppo._supervised_bootstrap(
        params=params,
        mean=mean,
        scale=scale,
        seed=123,
        samples=384,
        holdout=128,
        epochs=3,
        batch_size=64,
        learning_rate=0.015,
        max_grad_norm=1.0,
        style="pressure",
    )

    assert result["bootstrap_final_loss"] < result["bootstrap_initial_loss"]
    assert result["bootstrap_final_agreement"] >= result["bootstrap_initial_agreement"]


def test_deep_ppo_budget_rejects_tiny_non_smoke_run(tmp_path):
    args = SimpleNamespace(
        allow_smoke=False,
        generations=1,
        matches_per_generation=2,
        hands=40,
        min_training_hands=1_000_000,
    )

    with pytest.raises(SystemExit):
        train_deep_ppo.validate_training_budget(args, tmp_path / "policy.npz")


def test_deep_ppo_budget_blocks_smoke_overwriting_canonical_artifact():
    args = SimpleNamespace(
        allow_smoke=True,
        generations=1,
        matches_per_generation=2,
        hands=40,
        min_training_hands=1_000_000,
    )

    with pytest.raises(SystemExit):
        train_deep_ppo.validate_training_budget(args, train_deep_ppo.DEFAULT_OUTPUT)
