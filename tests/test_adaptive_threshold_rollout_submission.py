"""Coverage for the adaptive threshold-rollout submission candidate."""

from __future__ import annotations

import ast
import importlib.util
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.validator import check_static, validate

BOT_DIR = ROOT / "bots" / "adaptive_threshold_rollout_submission"
BOT_FILE = BOT_DIR / "bot.py"


def load_bot():
    spec = importlib.util.spec_from_file_location("adaptive_threshold_rollout_under_test", BOT_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def state_template(**overrides):
    state = {
        "type": "action_request",
        "hand_id": "test_hand",
        "street": "flop",
        "seat_to_act": 0,
        "pot": 1200,
        "current_bet": 0,
        "min_raise_to": 800,
        "amount_owed": 0,
        "can_check": True,
        "your_cards": ["As", "Ks"],
        "community_cards": ["Qs", "7s", "2d"],
        "your_stack": 9000,
        "your_bet_this_street": 0,
        "players": [
            {"seat": 0, "bot_id": "hero", "stack": 9000, "is_folded": False},
            {"seat": 1, "bot_id": "villain1", "stack": 9000, "is_folded": False},
            {"seat": 2, "bot_id": "villain2", "stack": 9000, "is_folded": False},
            {"seat": 3, "bot_id": "villain3", "stack": 9000, "is_folded": False},
            {"seat": 4, "bot_id": "villain4", "stack": 9000, "is_folded": False},
            {"seat": 5, "bot_id": "villain5", "stack": 9000, "is_folded": False},
        ],
        "action_log": [],
        "match_action_log": [],
    }
    state.update(overrides)
    return state


def heads_up_state(**overrides):
    state = state_template(players=[
        {"seat": 0, "bot_id": "hero", "stack": 9000, "is_folded": False},
        {"seat": 1, "bot_id": "villain1", "stack": 9000, "is_folded": False},
    ])
    state.update(overrides)
    return state


def assert_valid_action(state, action):
    assert isinstance(action, dict)
    assert action.get("action") in {"fold", "check", "call", "raise", "all_in"}
    if state.get("can_check"):
        assert action["action"] != "fold"
    if action["action"] == "raise":
        assert isinstance(action.get("amount"), int)
        assert action["amount"] >= int(state.get("min_raise_to", 0) or 0)
        assert action["amount"] <= int(state.get("your_bet_this_street", 0) or 0) + int(state.get("your_stack", 0) or 0)


def test_submission_file_is_standalone_and_static_safe():
    assert BOT_FILE.is_file()
    assert not (BOT_DIR / "data").exists()
    assert sorted(path.name for path in BOT_DIR.glob("*.py")) == ["bot.py"]

    source = BOT_FILE.read_text(encoding="utf-8")
    assert "tools." not in source
    assert "bots." not in source
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots == {"eval7", "random", "time"}

    static_errors = [issue for issue in check_static(str(BOT_FILE)) if issue["level"] == "error"]
    assert static_errors == []


def test_threshold_math_and_noise_buffer_shape():
    bot = load_bot()

    assert bot._threshold_from_fraction(0.50) == pytest.approx(0.25)
    assert bot._fraction_from_threshold(0.30) == pytest.approx(0.75)
    assert bot._mc_buffer(0.30, 1024) > bot._mc_buffer(0.30, 10000)

    value_candidates = bot._candidate_fractions(0.82, state_template(), "value")
    bluff_candidates = bot._candidate_fractions(0.28, state_template(), "bluff")
    assert any(abs(item - 0.63) < 0.01 for item in value_candidates)
    assert max(value_candidates) <= bot._effective_fraction_cap(state_template())
    assert max(bluff_candidates) <= bot._effective_fraction_cap(state_template())


def test_mechanical_weight_uses_observed_pressure_responses():
    bot = load_bot()
    folding_log = [
        {"hand_num": 0, "seat": 0, "bot_id": "hero", "action": "raise", "amount": 700},
        {"hand_num": 0, "seat": 1, "bot_id": "villain1", "action": "fold", "amount": None},
        {"hand_num": 1, "seat": 0, "bot_id": "hero", "action": "raise", "amount": 900},
        {"hand_num": 1, "seat": 1, "bot_id": "villain1", "action": "call", "amount": None},
        {"hand_num": 2, "seat": 0, "bot_id": "hero", "action": "raise", "amount": 1200},
        {"hand_num": 2, "seat": 1, "bot_id": "villain1", "action": "fold", "amount": None},
    ]
    raising_log = [
        {"hand_num": 0, "seat": 0, "bot_id": "hero", "action": "raise", "amount": 700},
        {"hand_num": 0, "seat": 1, "bot_id": "villain1", "action": "raise", "amount": 2100},
        {"hand_num": 1, "seat": 0, "bot_id": "hero", "action": "raise", "amount": 900},
        {"hand_num": 1, "seat": 1, "bot_id": "villain1", "action": "all_in", "amount": 9000},
        {"hand_num": 2, "seat": 0, "bot_id": "hero", "action": "raise", "amount": 1200},
        {"hand_num": 2, "seat": 1, "bot_id": "villain1", "action": "raise", "amount": 3600},
    ]

    folding_weight = bot._mechanical_weight(heads_up_state(match_action_log=folding_log))
    raising_weight = bot._mechanical_weight(heads_up_state(match_action_log=raising_log))

    assert folding_weight > 0.60
    assert raising_weight < 0.35
    assert folding_weight > raising_weight


def test_threshold_call_model_responds_to_size_and_weight():
    bot = load_bot()
    mechanical_state = heads_up_state(match_action_log=[
        {"hand_num": 0, "seat": 0, "bot_id": "hero", "action": "raise", "amount": 1000},
        {"hand_num": 0, "seat": 1, "bot_id": "villain1", "action": "fold", "amount": None},
        {"hand_num": 1, "seat": 0, "bot_id": "hero", "action": "raise", "amount": 1000},
        {"hand_num": 1, "seat": 1, "bot_id": "villain1", "action": "fold", "amount": None},
    ])
    low_weight = 0.20
    high_weight = bot._mechanical_weight(mechanical_state)

    loose_call = bot._mechanical_call_probability(0.27, 0.30, mechanical_state, high_weight)
    pressured_call = bot._mechanical_call_probability(1.27, 0.30, mechanical_state, high_weight)
    generic_pressure = bot._mechanical_call_probability(1.27, 0.30, mechanical_state, low_weight)

    assert pressured_call < loose_call
    assert pressured_call > generic_pressure


def test_decide_returns_legal_actions_for_representative_states():
    bot = load_bot()
    states = [
        state_template(street="preflop", your_cards=["As", "Kh"], community_cards=[], pot=150, current_bet=100, min_raise_to=200, amount_owed=100, can_check=False),
        state_template(street="flop", your_cards=["As", "Ks"], community_cards=["Qs", "7s", "2d"], can_check=True),
        state_template(street="turn", your_cards=["Ah", "Qh"], community_cards=["Ad", "7h", "2c", "Jh"], pot=4800, current_bet=2400, min_raise_to=4800, amount_owed=2400, can_check=False),
        state_template(street="river", your_cards=["9c", "9d"], community_cards=["9s", "7d", "2h", "Jc", "3s"], pot=8200, current_bet=0, min_raise_to=400, amount_owed=0, can_check=True),
        state_template(street="flop", your_cards=["2c", "7d"], community_cards=["As", "Kd", "Qh"], your_stack=350, pot=3000, current_bet=600, min_raise_to=1200, amount_owed=350, can_check=False),
        state_template(street="flop", your_cards=["bad"], community_cards=["As", "Kd", "Qh"], can_check=True),
    ]

    for state in states:
        assert_valid_action(state, bot.decide(state))


def test_rollout_equity_is_deterministic_and_time_guarded(monkeypatch):
    bot = load_bot()
    state = state_template()

    first = bot._rollout_equity(state, 128, bot._state_rng(state))
    second = bot._rollout_equity(state, 128, bot._state_rng(state))
    assert 0.0 <= first <= 1.0
    assert first == second

    calls = {"evaluate": 0}

    def fake_evaluate(cards):
        calls["evaluate"] += 1
        return len(cards)

    monkeypatch.setattr(bot, "TIME_BUDGET_SECONDS", -1.0)
    monkeypatch.setattr(bot.eval7, "evaluate", fake_evaluate)
    equity = bot._rollout_equity(state, 1024, bot._state_rng(state))
    assert 0.0 <= equity <= 1.0
    assert calls["evaluate"] <= (bot.MIN_ROLLOUT_SAMPLES + bot.TIME_CHECK_INTERVAL) * 6


def test_decide_latency_stays_under_action_budget():
    bot = load_bot()
    states = [
        state_template(hand_id=f"lat_{index}", street="preflop", your_cards=["As", "Kh"], community_cards=[], pot=150, current_bet=100, min_raise_to=200, amount_owed=100, can_check=False)
        for index in range(2)
    ]
    states.extend([
        state_template(hand_id="lat_flop"),
        state_template(hand_id="lat_turn", street="turn", community_cards=["Ad", "7h", "2c", "Jh"], pot=4800, current_bet=2400, min_raise_to=4800, amount_owed=2400, can_check=False),
        state_template(hand_id="lat_river", street="river", community_cards=["9s", "7d", "2h", "Jc", "3s"], pot=8200),
    ])

    elapsed = []
    for state in states:
        started = time.perf_counter()
        action = bot.decide(state)
        elapsed.append(time.perf_counter() - started)
        assert_valid_action(state, action)

    assert max(elapsed) < 2.0


def test_official_validator_accepts_submission_directory():
    result = validate(str(BOT_DIR))
    assert result["passed"], result
    assert result["errors"] == []
    assert all(item["elapsed"] < 2.0 for item in result["tests"])


def test_decide_handles_warmup_and_non_dict_state():
    bot = load_bot()
    assert bot.decide({"type": "warmup"}) == {"action": "check"}
    assert bot.decide(None) == {"action": "check"}
