"""Tests for the heuristic-guided expert-arm selector mock."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.strong_mocks import train_arm_selector, tune_arm_selector_params
from tools.strong_mocks.arm_selector_params import (
    PARAM_NAMES,
    coerce_params,
    load_params_npz,
    params_from_vector,
    save_params_npz,
    vector_from_params,
)
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


def test_selector_params_roundtrip_and_clip(tmp_path):
    params = coerce_params({"call_margin_base": 9.0, "pressure_bluff_fraction": -1.0})
    assert params["call_margin_base"] <= 0.16
    assert params["pressure_bluff_fraction"] >= 0.35

    normalized = vector_from_params(params, normalized=True)
    restored = params_from_vector(normalized, normalized=True)
    assert set(restored) == set(PARAM_NAMES)
    assert abs(restored["call_margin_base"] - params["call_margin_base"]) < 1e-6

    path = tmp_path / "params.npz"
    save_params_npz(path, restored, source="test")
    loaded = load_params_npz(path)
    assert loaded["pressure_bluff_fraction"] == restored["pressure_bluff_fraction"]


def test_expert_candidate_generation_accepts_tuned_params():
    state = _sample_state()
    default_candidates = generate_candidates(state)
    tuned = coerce_params({"value_threshold_base": 0.82})
    tuned_candidates = generate_candidates(state, tuned)

    default_action = next(candidate for candidate in default_candidates if candidate.arm == "default_tag").action
    tuned_action = next(candidate for candidate in tuned_candidates if candidate.arm == "default_tag").action

    assert default_action["action"] == "raise"
    assert tuned_action["action"] == "check"


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


def test_arm_selector_budget_rejects_tiny_non_smoke_run(tmp_path):
    args = SimpleNamespace(
        allow_smoke=False,
        generations=1,
        matches_per_generation=2,
        hands=40,
        min_training_hands=1_000_000,
    )

    with pytest.raises(SystemExit):
        train_arm_selector.validate_training_budget(args, tmp_path / "policy.npz")


def test_arm_selector_budget_blocks_smoke_overwriting_canonical_artifact():
    args = SimpleNamespace(
        allow_smoke=True,
        generations=1,
        matches_per_generation=2,
        hands=40,
        min_training_hands=1_000_000,
    )

    with pytest.raises(SystemExit):
        train_arm_selector.validate_training_budget(args, train_arm_selector.DEFAULT_OUTPUT)


def test_tuning_stage_parser_requires_explicit_budget_shape():
    stages = tune_arm_selector_params._parse_stages("2:10:0.5,4:20:1.0")

    assert [stage.matches for stage in stages] == [2, 4]
    assert [stage.hands for stage in stages] == [10, 20]
    assert stages[-1].keep_frac == 1.0


def test_tuning_default_stage_budget_is_statistically_gated():
    stages = tune_arm_selector_params._parse_stages(tune_arm_selector_params.DEFAULT_STAGE_SPEC)
    final = stages[-1]

    assert final.matches * final.hands >= 6400
