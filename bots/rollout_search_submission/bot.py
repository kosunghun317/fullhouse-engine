"""Standalone rollout-search submission candidate.

The file is intentionally self-contained: no repo-local imports, no data files,
and no write-time state. It uses the tuned f/g parameters from
runs/rollout_search_tuning/rollout-final-20260531-004635/best_params.json.
"""

import random
import time

import eval7


RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = tuple(eval7.Card(rank + suit) for rank in RANKS for suit in SUITS)

PARAMS = {
    "samples": 1024.0,
    "deep_samples": 1024.0,
    "f_min_equity": 0.38600570637776854,
    "f_center": 0.5536016465664075,
    "f_scale": 0.10075311307590826,
    "f_min_raise": 0.35886009720451495,
    "f_max_raise": 0.9085659785544977,
    "g_call_edge_base": 0.06366738517571167,
    "g_call_edge_active": 0.03899529132884684,
    "g_pressure_discount": 0.024391452794899727,
    "g_thin_call_edge": 0.01937116600090552,
    "g_raise_edge": 0.22216495168990685,
    "g_raise_equity": 0.9198701106956259,
    "g_raise_max_owed_pot": 0.20878296621524373,
    "g_raise_center": 0.9198701106956259,
    "g_raise_scale": 0.041550034474822965,
    "g_min_raise": 0.9369258774068039,
    "g_max_raise": 0.9369258774068039,
}

TIME_BUDGET_SECONDS = 1.55
MIN_ROLLOUT_SAMPLES = 128
TIME_CHECK_INTERVAL = 32
BIG_BLIND = 100


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _sigmoid(value):
    value = _clip(float(value), -60.0, 60.0)
    return 1.0 / (1.0 + pow(2.718281828459045, -value))


def _rank_value(card):
    text = str(card)
    return RANKS.index(text[0]) + 2 if text and text[0] in RANKS else 2


def _preflop_strength(cards):
    cards = list(cards or [])
    if len(cards) < 2:
        return 0.15
    values = sorted((_rank_value(card) for card in cards[:2]), reverse=True)
    pair = values[0] == values[1]
    suited = len(cards[0]) > 1 and len(cards[1]) > 1 and cards[0][1] == cards[1][1]
    score = 0.42 * values[0] / 14.0 + 0.20 * values[1] / 14.0
    if pair:
        score += 0.30 + values[0] / 120.0
    if suited:
        score += 0.06
    if values[0] == 14:
        score += 0.06
    return _clip(score, 0.03, 0.96)


def _active_players(state):
    players = list(state.get("players", []) or [])
    if not players:
        return 2
    active = 0
    for player in players:
        if not player.get("is_folded", False):
            active += 1
    return max(2, active)


def _state_rng(state, salt="rollout_pressure"):
    action_log = state.get("action_log", [])
    parts = [
        salt,
        str(state.get("hand_id", "")),
        str(state.get("seat_to_act", "")),
        str(len(action_log)),
        str(state.get("street", "")),
        str(state.get("pot", "")),
        str(state.get("amount_owed", "")),
        ",".join(str(card) for card in (state.get("your_cards", []) or [])),
        ",".join(str(card) for card in (state.get("community_cards", []) or [])),
    ]
    seed = 0
    for ch in "|".join(parts):
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    return random.Random(seed)


def _rollout_equity(state, samples, rng):
    board = list(state.get("community_cards", []) or [])
    cards = list(state.get("your_cards", []) or [])
    try:
        hero = [eval7.Card(card) for card in cards[:2]]
        community = [eval7.Card(card) for card in board[:5]]
    except Exception:
        active = _active_players(state)
        return max(0.03, _preflop_strength(cards) - 0.055 * max(0, active - 2))

    if len(hero) < 2:
        active = _active_players(state)
        return max(0.03, _preflop_strength(cards) - 0.055 * max(0, active - 2))

    active = _active_players(state)
    opponents = max(1, min(5, active - 1))
    dead = set(hero + community)
    deck = [card for card in FULL_DECK if card not in dead]
    board_needed = max(0, 5 - len(community))
    need = opponents * 2 + board_needed
    if len(deck) < need:
        return 0.0

    target = max(1, int(samples))
    deadline = time.perf_counter() + TIME_BUDGET_SECONDS
    wins = 0.0
    completed = 0
    for index in range(target):
        if (
            index >= MIN_ROLLOUT_SAMPLES
            and index % TIME_CHECK_INTERVAL == 0
            and time.perf_counter() >= deadline
        ):
            break
        draw = rng.sample(deck, need)
        offset = 0
        opp_hands = []
        for _ in range(opponents):
            opp_hands.append([draw[offset], draw[offset + 1]])
            offset += 2
        runout = list(community)
        if board_needed:
            runout.extend(draw[offset:offset + board_needed])

        hero_score = eval7.evaluate(hero + runout)
        best_score = hero_score
        tied = 1
        for opp_hand in opp_hands:
            score = eval7.evaluate(opp_hand + runout)
            if score > best_score:
                best_score = score
                tied = 1
            elif score == best_score:
                tied += 1
        if hero_score == best_score:
            wins += 1.0 / tied
        completed += 1

    return wins / max(1, completed)


def _check_raise_fraction(equity):
    if equity < PARAMS["f_min_equity"]:
        return 0.0
    weight = _sigmoid((equity - PARAMS["f_center"]) / PARAMS["f_scale"])
    return PARAMS["f_min_raise"] + weight * (PARAMS["f_max_raise"] - PARAMS["f_min_raise"])


def _facing_bet_raise_fraction(equity, pot_odds, active_players, owed_pot_ratio):
    edge = float(equity) - float(pot_odds)
    active_extra = max(0, int(active_players) - 2)
    call_edge = PARAMS["g_call_edge_base"] + PARAMS["g_call_edge_active"] * active_extra
    call_edge -= PARAMS["g_pressure_discount"]
    if edge < call_edge:
        if edge > PARAMS["g_thin_call_edge"]:
            return 0.0
        return -1.0
    if (
        edge < PARAMS["g_raise_edge"]
        or equity < PARAMS["g_raise_equity"]
        or owed_pot_ratio > PARAMS["g_raise_max_owed_pot"]
    ):
        return 0.0
    weight = _sigmoid((equity - PARAMS["g_raise_center"]) / PARAMS["g_raise_scale"])
    return PARAMS["g_min_raise"] + weight * (PARAMS["g_max_raise"] - PARAMS["g_min_raise"])


def _sanitize_action(state, action):
    act = str(action.get("action", "")).lower()
    can_check = bool(state.get("can_check", False))
    stack = max(0, int(state.get("your_stack", 0) or 0))
    invested = max(0, int(state.get("your_bet_this_street", 0) or 0))
    owed = max(0, int(state.get("amount_owed", 0) or 0))

    if stack <= 0:
        return {"action": "check"} if can_check else {"action": "fold"}
    if act == "fold":
        return {"action": "check"} if can_check else {"action": "fold"}
    if act == "check":
        return {"action": "check"} if can_check else {"action": "call"}
    if act == "call":
        return {"action": "check"} if can_check else {"action": "call"}
    if act == "all_in":
        return {"action": "all_in"}
    if act == "raise":
        amount = int(action.get("amount", 0) or 0)
        min_raise = max(0, int(state.get("min_raise_to", 0) or 0))
        max_total = invested + stack
        if max_total <= min_raise:
            if can_check:
                return {"action": "check"}
            return {"action": "all_in"} if stack <= owed else {"action": "call"}
        return {"action": "raise", "amount": max(min_raise, min(amount, max_total))}
    return {"action": "check"} if can_check else {"action": "fold"}


def _raise_to_fraction(state, fraction):
    pot = max(1, int(state.get("pot", 0) or 0))
    current_bet = max(0, int(state.get("current_bet", 0) or 0))
    min_raise = max(0, int(state.get("min_raise_to", 0) or 0))
    amount = max(min_raise, current_bet + int(pot * float(fraction)))
    return _sanitize_action(state, {"action": "raise", "amount": amount})


def _passive_action(state):
    return _sanitize_action(state, {"action": "check"} if state.get("can_check") else {"action": "call"})


def decide(state):
    if not isinstance(state, dict) or state.get("type") == "warmup":
        return {"action": "check"}

    samples = int(PARAMS["samples"])
    equity = _rollout_equity(state, samples=samples, rng=_state_rng(state))
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    active = _active_players(state)

    if bool(state.get("can_check", False)):
        fraction = _check_raise_fraction(equity)
        if fraction > 0.0:
            return _raise_to_fraction(state, fraction)
        return _passive_action(state)

    odds = owed / max(1.0, float(pot + owed))
    owed_pot_ratio = owed / max(1.0, float(pot))
    fraction = _facing_bet_raise_fraction(equity, odds, active, owed_pot_ratio)
    if fraction > 0.0:
        return _raise_to_fraction(state, fraction)
    if fraction == 0.0:
        return _passive_action(state)
    return _sanitize_action(state, {"action": "fold"})
