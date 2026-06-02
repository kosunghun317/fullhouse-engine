"""Replay-conditioned portal behavior-clone mock competitor."""

import json
import os


DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
ACTION_LABELS = ("fold", "check", "call", "raise", "all_in")
RANK_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

DEFAULT_POLICY = {
    "policy_type": "portal_behavior_clone_v1",
    "group": "unknown_mixed",
    "fallback": {
        "count": 0,
        "action_probs": {"fold": 0.30, "check": 0.30, "call": 0.24, "raise": 0.14, "all_in": 0.02},
        "raise_to_pot": {"count": 1, "mean": 0.85, "p25": 0.55, "p50": 0.75, "p75": 1.10},
    },
    "cells": {},
}


def _load_policy():
    path = os.path.join(DATA_DIR, "policy.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return dict(DEFAULT_POLICY)
        merged = dict(DEFAULT_POLICY)
        merged.update(data)
        if not isinstance(merged.get("fallback"), dict):
            merged["fallback"] = dict(DEFAULT_POLICY["fallback"])
        if not isinstance(merged.get("cells"), dict):
            merged["cells"] = {}
        return merged
    except Exception:
        return dict(DEFAULT_POLICY)


POLICY = _load_policy()
BOT_NAME = "PortalClone_" + str(POLICY.get("group", "unknown_mixed"))[:32]
BOT_AVATAR = "robot_3"


def _clamp(value, lo, hi):
    return min(hi, max(lo, value))


def _num(state, key, default=0):
    value = state.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return default


def _state_seed(state, salt):
    text = (
        str(state.get("hand_id", ""))
        + "|"
        + str(state.get("street", ""))
        + "|"
        + str(state.get("seat_to_act", ""))
        + "|"
        + str(len(state.get("action_log") or []))
        + "|"
        + "".join(state.get("your_cards") or [])
        + "|"
        + salt
    )
    acc = 2166136261
    for ch in text:
        acc = ((acc ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return acc


def _roll(state, salt):
    return (_state_seed(state, salt) % 1_000_000) / 1_000_000.0


def _active_opponents(state):
    seat = state.get("seat_to_act")
    count = 0
    for player in state.get("players") or []:
        if player.get("seat") == seat:
            continue
        if player.get("state") in {"active", "all_in"} and not player.get("is_folded"):
            count += 1
    return max(1, count)


def _opponent_bucket(active_opponents):
    if active_opponents <= 1:
        return "hu"
    if active_opponents <= 2:
        return "short"
    return "multi"


def _pot_odds_bucket(pot_odds):
    if pot_odds <= 0.0:
        return "free"
    if pot_odds < 0.18:
        return "cheap"
    if pot_odds < 0.34:
        return "medium"
    return "expensive"


def _risk_bucket(risk):
    if risk <= 0.0:
        return "none"
    if risk < 0.12:
        return "low"
    if risk < 0.35:
        return "medium"
    if risk < 0.70:
        return "high"
    return "terminal"


def _raise_bucket(raises_this_street):
    if raises_this_street <= 0:
        return "r0"
    if raises_this_street == 1:
        return "r1"
    return "r2plus"


def _raises_this_street(state):
    actions = [
        str(entry.get("action", "")).lower()
        for entry in state.get("action_log") or []
        if isinstance(entry, dict)
    ]
    raises = sum(1 for action in actions if action in {"raise", "all_in"})
    if state.get("street") == "preflop":
        return raises
    if int(_num(state, "current_bet", 0)) <= 0:
        return 0
    return max(1, min(raises, 2))


def _context_key(state):
    owed = max(0.0, float(_num(state, "amount_owed", 0)))
    pot = max(0.0, float(_num(state, "pot", 0)))
    stack = max(0.0, float(_num(state, "your_stack", 0)))
    current = max(0.0, float(_num(state, "your_bet_this_street", 0)))
    facing = owed > 0.0
    pot_odds = owed / max(1.0, pot + owed)
    stack_risk = owed / max(1.0, stack + current)
    mode = "facing" if facing else "free"
    return "|".join(
        [
            str(state.get("street") or "preflop"),
            mode,
            _opponent_bucket(_active_opponents(state)),
            _pot_odds_bucket(pot_odds),
            _risk_bucket(stack_risk),
            _raise_bucket(_raises_this_street(state)),
        ]
    )


def _context_backoff_keys(key):
    parts = str(key).split("|")
    if len(parts) != 6:
        return [key]
    street, mode, opponents, pot_odds, risk, raises = parts
    return [
        key,
        "|".join([street, mode, opponents, pot_odds, risk, "any"]),
        "|".join([street, mode, opponents, "any", "any", "any"]),
        "|".join([street, mode, "any", "any", "any", "any"]),
    ]


def _select_cell(state):
    cells = POLICY.get("cells") if isinstance(POLICY.get("cells"), dict) else {}
    for key in _context_backoff_keys(_context_key(state)):
        cell = cells.get(key)
        if isinstance(cell, dict):
            return cell
    return POLICY.get("fallback") if isinstance(POLICY.get("fallback"), dict) else DEFAULT_POLICY["fallback"]


def _action_probs(cell):
    probs = cell.get("action_probs") if isinstance(cell, dict) else None
    if not isinstance(probs, dict):
        return DEFAULT_POLICY["fallback"]["action_probs"]
    cleaned = {}
    for action in ACTION_LABELS:
        value = probs.get(action, 0.0)
        cleaned[action] = max(0.0, float(value) if isinstance(value, (int, float)) else 0.0)
    total = sum(cleaned.values())
    if total <= 0.0:
        return DEFAULT_POLICY["fallback"]["action_probs"]
    return {action: cleaned[action] / total for action in ACTION_LABELS}


def _sample_action(state, cell):
    probs = _action_probs(cell)
    roll = _roll(state, "policy-action")
    acc = 0.0
    for action in ACTION_LABELS:
        acc += probs.get(action, 0.0)
        if roll <= acc:
            return action
    return "fold"


def _preflop_score(cards):
    if len(cards) < 2:
        return 0.28
    r1 = RANK_VALUE.get(cards[0][0], 2)
    r2 = RANK_VALUE.get(cards[1][0], 2)
    high = max(r1, r2)
    low = min(r1, r2)
    suited = len(cards[0]) > 1 and len(cards[1]) > 1 and cards[0][1] == cards[1][1]
    gap = high - low
    if r1 == r2:
        return _clamp(0.48 + high / 28.0, 0.05, 0.98)
    score = 0.16 + high / 23.0 + low / 74.0
    if suited:
        score += 0.055
    if gap <= 1:
        score += 0.045
    elif gap >= 5:
        score -= 0.055
    if high >= 13 and low >= 10:
        score += 0.075
    return _clamp(score, 0.04, 0.93)


def _made_hand_score(cards, board):
    all_cards = list(cards) + list(board)
    if not all_cards:
        return 0.20
    ranks = [RANK_VALUE.get(card[0], 2) for card in all_cards if card]
    suits = [card[1] for card in all_cards if len(card) > 1]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}
    made = 0.28
    pairs = sum(1 for count in rank_counts.values() if count >= 2)
    if max(rank_counts.values(), default=0) >= 4:
        made = 0.94
    elif max(rank_counts.values(), default=0) >= 3:
        made = 0.78
    elif pairs >= 2:
        made = 0.67
    elif pairs == 1:
        made = 0.50
    if max(suit_counts.values(), default=0) >= 5:
        made = max(made, 0.86)
    elif max(suit_counts.values(), default=0) == 4:
        made = max(made, 0.48)
    unique = sorted(set(ranks))
    run = 1
    best_run = 1
    for index in range(1, len(unique)):
        if unique[index] == unique[index - 1] + 1:
            run += 1
        else:
            run = 1
        best_run = max(best_run, run)
    if best_run >= 5:
        made = max(made, 0.90)
    elif best_run == 4:
        made = max(made, 0.50)
    return made


def _strength(state):
    cards = list(state.get("your_cards") or [])
    preflop = _preflop_score(cards)
    board = list(state.get("community_cards") or [])
    if not board:
        base = preflop
    else:
        base = max(preflop * 0.55, _made_hand_score(cards, board))
    penalty = max(0, _active_opponents(state) - 1) * 0.035
    return _clamp(base - penalty, 0.02, 0.98)


def _max_total_bet(state):
    return int(_num(state, "your_stack", 0) + _num(state, "your_bet_this_street", 0))


def _sizing_ratio(cell):
    sizing = cell.get("raise_to_pot") if isinstance(cell, dict) else None
    if not isinstance(sizing, dict):
        sizing = POLICY.get("fallback", {}).get("raise_to_pot")
    if not isinstance(sizing, dict):
        return 0.75
    for key in ("p50", "mean", "p75", "p25"):
        value = sizing.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return _clamp(float(value), 0.20, 4.00)
    return 0.75


def _raise_amount(state, cell):
    pot = max(1, int(_num(state, "pot", 1)))
    owed = max(0, int(_num(state, "amount_owed", 0)))
    current_bet = max(0, int(_num(state, "current_bet", 0)))
    min_raise = int(_num(state, "min_raise_to", 0))
    max_total = _max_total_bet(state)
    jitter = 0.85 + _roll(state, "raise-size") * 0.30
    target = current_bet + int((pot + owed) * _sizing_ratio(cell) * jitter)
    target = max(min_raise, target)
    target = min(max_total, target)
    return target


def _sanitize(state, action, amount=None):
    can_check = bool(state.get("can_check"))
    max_total = _max_total_bet(state)
    min_raise = int(_num(state, "min_raise_to", 0))
    if action == "check":
        return {"action": "check"} if can_check else {"action": "fold"}
    if action == "fold":
        return {"action": "check"} if can_check else {"action": "fold"}
    if action == "call":
        return {"action": "check"} if can_check else {"action": "call"}
    if action == "all_in":
        return {"action": "all_in"} if max_total > 0 else ({"action": "check"} if can_check else {"action": "fold"})
    if action == "raise":
        if min_raise <= 0 or max_total <= min_raise:
            return {"action": "call"} if not can_check else {"action": "check"}
        target = int(amount if amount is not None else min_raise)
        target = max(min_raise, min(max_total, target))
        if target >= max_total:
            return {"action": "all_in"}
        return {"action": "raise", "amount": target}
    return {"action": "check"} if can_check else {"action": "fold"}


def _guard_action(state, cell, action, strength):
    can_check = bool(state.get("can_check"))
    owed = max(0.0, float(_num(state, "amount_owed", 0)))
    pot = max(1.0, float(_num(state, "pot", 1)))
    pot_odds = owed / max(1.0, pot + owed)
    probs = _action_probs(cell)
    raiseish = probs.get("raise", 0.0) + probs.get("all_in", 0.0)
    bluff_ok = _roll(state, "guard-bluff") < min(0.18, raiseish * 0.32)

    if can_check and action in {"fold", "call", "check"}:
        return _sanitize(state, "check")

    if action == "all_in" and strength < 0.78:
        if strength >= 0.58 and bluff_ok:
            action = "raise"
        elif not can_check and strength >= pot_odds + 0.08:
            return _sanitize(state, "call")
        else:
            return _sanitize(state, "check" if can_check else "fold")

    if action == "raise":
        if strength < 0.34 and not bluff_ok:
            return _sanitize(state, "check" if can_check else "fold")
        if not can_check and strength < pot_odds + 0.04 and not bluff_ok:
            return _sanitize(state, "fold")
        return _sanitize(state, "raise", _raise_amount(state, cell))

    if action == "call" and not can_check:
        if pot_odds >= 0.34 and strength < pot_odds + 0.06:
            return _sanitize(state, "fold")
        return _sanitize(state, "call")

    if action == "check":
        return _sanitize(state, "check")

    if action == "fold" and not can_check and strength >= max(0.58, pot_odds + 0.22):
        if strength > 0.82 and raiseish > 0.08:
            return _sanitize(state, "raise", _raise_amount(state, cell))
        return _sanitize(state, "call")

    return _sanitize(state, action)


def decide(state):
    if state.get("type") == "warmup":
        return {"action": "check"}
    cell = _select_cell(state)
    action = _sample_action(state, cell)
    return _guard_action(state, cell, action, _strength(state))
