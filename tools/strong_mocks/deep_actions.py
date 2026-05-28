"""Expanded action abstraction for the benchmark-only deep PPO mock."""

from __future__ import annotations

import numpy as np


ACTION_LABELS = (
    "fold",
    "check_call",
    "raise_min",
    "raise_025",
    "raise_033",
    "raise_045",
    "raise_055",
    "raise_067",
    "raise_085",
    "raise_100",
    "raise_125",
    "raise_160",
    "raise_220",
    "all_in",
)

RAISE_FRACTIONS = {
    3: 0.25,
    4: 0.33,
    5: 0.45,
    6: 0.55,
    7: 0.67,
    8: 0.85,
    9: 1.00,
    10: 1.25,
    11: 1.60,
    12: 2.20,
}

RANKS = "23456789TJQKA"
BIG_BLIND = 100


def _rank_value(card: str) -> int:
    text = str(card)
    return RANKS.index(text[0]) + 2 if text and text[0] in RANKS else 2


def _preflop_strength(cards: list[str]) -> float:
    if len(cards or []) < 2:
        return 0.10
    values = sorted((_rank_value(card) for card in cards[:2]), reverse=True)
    pair = values[0] == values[1]
    suited = len(cards[0]) > 1 and len(cards[1]) > 1 and cards[0][1] == cards[1][1]
    gap = abs(values[0] - values[1])
    score = 0.42 * values[0] / 14.0 + 0.20 * values[1] / 14.0
    if pair:
        score += 0.30 + values[0] / 120.0
    if suited:
        score += 0.06
    if gap <= 1:
        score += 0.04
    elif gap >= 5:
        score -= 0.06
    if values[0] == 14:
        score += 0.06
    return float(max(0.02, min(0.98, score)))


def _postflop_commit_signal(state: dict) -> float:
    cards = list(state.get("your_cards", []) or [])
    board = list(state.get("community_cards", []) or [])
    all_cards = cards + board
    ranks: dict[int, int] = {}
    suits: dict[str, int] = {}
    for card in all_cards:
        ranks[_rank_value(card)] = ranks.get(_rank_value(card), 0) + 1
        if len(str(card)) > 1:
            suits[str(card)[1]] = suits.get(str(card)[1], 0) + 1
    pairish = 0.32 if any(count >= 2 for count in ranks.values()) else 0.0
    trips = 0.30 if any(count >= 3 for count in ranks.values()) else 0.0
    flush_draw = 0.17 if len(board) < 5 and max(suits.values(), default=0) >= 4 else 0.0
    vals = set(ranks)
    if 14 in vals:
        vals.add(1)
    straight_draw = 0.0
    for start in range(1, 11):
        if len(vals & {start, start + 1, start + 2, start + 3, start + 4}) >= 4:
            straight_draw = 0.13
            break
    return min(1.0, pairish + trips + flush_draw + straight_draw)


def legal_mask(state: dict) -> np.ndarray:
    mask = np.ones(len(ACTION_LABELS), dtype=bool)
    if state.get("can_check"):
        mask[0] = False
    stack = int(state.get("your_stack", 0) or 0)
    min_raise = int(state.get("min_raise_to", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    if stack <= 0:
        mask[2:] = False
    elif invested + stack <= min_raise:
        mask[2:13] = False
    return mask


def strategic_mask(state: dict) -> np.ndarray:
    """Legal mask plus risk guardrails for a wider neural action space."""
    mask = legal_mask(state)
    stack = int(state.get("your_stack", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    total_stack = stack + invested

    if owed > 0 and not state.get("can_check"):
        pressure_call = owed >= max(3 * BIG_BLIND, int(stack * 0.25), int(pot * 0.45))
        if pressure_call and total_stack > 15 * BIG_BLIND:
            if state.get("street") == "preflop":
                call_allowed = _preflop_strength(list(state.get("your_cards", []) or [])) >= 0.78
            else:
                call_allowed = _postflop_commit_signal(state) >= 0.50
            if not call_allowed:
                mask[1] = False

    if total_stack <= 15 * BIG_BLIND:
        return mask

    if mask[-1]:
        if state.get("street") == "preflop":
            all_in_allowed = _preflop_strength(list(state.get("your_cards", []) or [])) >= 0.86
        else:
            all_in_allowed = _postflop_commit_signal(state) >= 0.62 and stack <= pot * 2.2
        if not all_in_allowed:
            mask[-1] = False

    # Larger arms are useful for bucket/threshold pressure, but weak deep-stack
    # overbets make the mock too brittle. Gate them by a cheap strength signal.
    if state.get("street") == "preflop":
        strength = _preflop_strength(list(state.get("your_cards", []) or []))
        raise_strength = 0.78
    else:
        strength = _postflop_commit_signal(state)
        raise_strength = 0.55
    medium = strength >= 0.64
    strong = strength >= 0.82 if state.get("street") == "preflop" else strength >= 0.55
    if owed > 0 and not state.get("can_check") and total_stack > 15 * BIG_BLIND:
        if strength < raise_strength:
            mask[2:13] = False
            mask[-1] = False
        elif not strong:
            mask[8:13] = False
    if total_stack > 25 * BIG_BLIND:
        if not medium:
            mask[8:13] = False
        elif not strong:
            mask[10:13] = False
    return mask


def sanitize_action(state: dict, action: dict) -> dict:
    act = str(action.get("action", "")).lower()
    can_check = bool(state.get("can_check", False))
    stack = int(state.get("your_stack", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    owed = int(state.get("amount_owed", 0) or 0)
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
        min_raise = int(state.get("min_raise_to", 0) or 0)
        max_total = invested + stack
        if max_total <= min_raise:
            return {"action": "all_in"} if max_total > owed else ({"action": "check"} if can_check else {"action": "call"})
        return {"action": "raise", "amount": max(min_raise, min(amount, max_total))}
    return {"action": "check"} if can_check else {"action": "fold"}


def raise_to_fraction(state: dict, fraction: float) -> dict:
    pot = max(1, int(state.get("pot", 0) or 0))
    current_bet = int(state.get("current_bet", 0) or 0)
    min_raise = int(state.get("min_raise_to", 0) or 0)
    amount = max(min_raise, current_bet + int(pot * fraction))
    return sanitize_action(state, {"action": "raise", "amount": amount})


def action_index_to_action(state: dict, index: int) -> dict:
    index = int(index)
    if index == 0:
        return sanitize_action(state, {"action": "fold"})
    if index == 1:
        return sanitize_action(state, {"action": "check"} if state.get("can_check") else {"action": "call"})
    if index == 2:
        return sanitize_action(state, {"action": "raise", "amount": int(state.get("min_raise_to", 0) or 0)})
    if index == len(ACTION_LABELS) - 1:
        return sanitize_action(state, {"action": "all_in"})
    if index in RAISE_FRACTIONS:
        return raise_to_fraction(state, RAISE_FRACTIONS[index])
    return sanitize_action(state, {"action": "check"} if state.get("can_check") else {"action": "fold"})


def masked_argmax(logits: np.ndarray, state: dict) -> int:
    values = np.asarray(logits, dtype=float).copy()
    mask = strategic_mask(state)
    values[~mask] = -1e9
    return int(np.argmax(values))
