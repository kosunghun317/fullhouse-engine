"""Tests for the heuristic-guided expert-arm selector mock."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.strong_mocks import train_arm_selector
from tools.strong_mocks.arm_selector_policy import (
    SELECTOR_FEATURE_NAMES,
    candidate_matrix,
    choose_candidate,
    decide_arm_selector,
)
from tools.strong_mocks.expert_arms import ARM_NAMES, generate_candidates


def _sample_state() -> dict:
    return {
        "type": "action_request",
        "hand_id": "selector_test_001",
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
        "match_action_log": [
            {"hand_num": 0, "seat": 1, "bot_id": "villain", "action": "fold"},
            {"hand_num": 1, "seat": 1, "bot_id": "villain", "action": "fold"},
            {"hand_num": 2, "seat": 1, "bot_id": "villain", "action": "call"},
            {"hand_num": 3, "seat": 1, "bot_id": "villain", "action": "fold"},
            {"hand_num": 4, "seat": 1, "bot_id": "villain", "action": "check"},
            {"hand_num": 5, "seat": 1, "bot_id": "villain", "action": "fold"},
            {"hand_num": 6, "seat": 1, "bot_id": "villain", "action": "raise"},
            {"hand_num": 7, "seat": 1, "bot_id": "villain", "action": "fold"},
        ],
    }


def test_expert_arms_generate_named_legal_candidates():
    candidates = generate_candidates(_sample_state())

    assert [candidate.arm for candidate in candidates] == list(ARM_NAMES)
    assert len(candidates) >= 10
    for candidate in candidates:
        assert candidate.action["action"] in {"fold", "check", "call", "raise", "all_in"}
        assert 0.0 <= candidate.risk <= 1.0
        assert 0.0 <= candidate.confidence <= 1.0


def test_candidate_matrix_shape_matches_selector_features():
    candidates, matrix = candidate_matrix(_sample_state())

    assert len(candidates) == len(ARM_NAMES)
    assert matrix.shape == (len(ARM_NAMES), len(SELECTOR_FEATURE_NAMES))


def test_selector_fallback_returns_legal_action_without_artifact(tmp_path):
    action = decide_arm_selector(_sample_state(), str(tmp_path), sample=False)

    assert action["action"] in {"fold", "check", "call", "raise", "all_in"}
    if action["action"] == "raise":
        assert isinstance(action["amount"], int)


def test_selector_policy_file_can_force_arm_choice(tmp_path):
    state = _sample_state()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    weights = np.zeros(len(SELECTOR_FEATURE_NAMES), dtype=np.float32)
    arm_feature_index = list(SELECTOR_FEATURE_NAMES).index("arm_pot_control")
    weights[arm_feature_index] = 5.0
    np.savez_compressed(
        data_dir / "policy.npz",
        weights=weights,
        bias=np.zeros(1, dtype=np.float32),
        mean=np.zeros(len(SELECTOR_FEATURE_NAMES), dtype=np.float32),
        scale=np.ones(len(SELECTOR_FEATURE_NAMES), dtype=np.float32),
        learned_blend=np.asarray([1.0], dtype=np.float32),
        temperature=np.asarray([0.35], dtype=np.float32),
    )

    candidate, info = choose_candidate(state, str(data_dir), sample=False)

    assert info["model_loaded"]
    assert candidate.arm == "pot_control"


def test_supervised_bootstrap_improves_selector_agreement():
    params = train_arm_selector._init_params(seed=123)
    result, _mean, _scale = train_arm_selector._supervised_bootstrap(
        params=params,
        seed=123,
        samples=220,
        holdout=80,
        epochs=3,
        learning_rate=0.025,
        learned_blend=0.80,
        max_grad_norm=1.0,
    )

    assert result["bootstrap_final_loss"] < result["bootstrap_initial_loss"]
    assert result["bootstrap_final_agreement"] >= result["bootstrap_initial_agreement"]

