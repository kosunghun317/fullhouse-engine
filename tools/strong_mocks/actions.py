"""Shared action abstraction for benchmark-only strong mock bots."""

from __future__ import annotations

import numpy as np


ACTION_LABELS = (
    "fold",
    "check_call",
    "raise_033",
    "raise_050",
    "raise_075",
    "raise_100",
    "raise_150",
    "all_in",
)

RAISE_FRACTIONS = {
    2: 0.33,
    3: 0.50,
    4: 0.75,
    5: 1.00,
    6: 1.50,
}


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
        mask[2:7] = False
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
    if index == 7:
        return sanitize_action(state, {"action": "all_in"})
    if index in RAISE_FRACTIONS:
        return raise_to_fraction(state, RAISE_FRACTIONS[index])
    return sanitize_action(state, {"action": "check"} if state.get("can_check") else {"action": "fold"})


def masked_argmax(logits: np.ndarray, state: dict) -> int:
    values = np.asarray(logits, dtype=float).copy()
    mask = legal_mask(state)
    values[~mask] = -1e9
    return int(np.argmax(values))
