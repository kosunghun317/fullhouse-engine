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


RANKS = "23456789TJQKA"
BIG_BLIND = 100


def _rank_value(card: str) -> int:
    return RANKS.index(str(card)[0]) + 2 if card and str(card)[0] in RANKS else 2


def _preflop_strength(cards: list[str]) -> float:
    if len(cards or []) < 2:
        return 0.10
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
    return float(max(0.02, min(0.98, score)))


def _postflop_commit_signal(state: dict) -> float:
    cards = list(state.get("your_cards", []) or [])
    board = list(state.get("community_cards", []) or [])
    all_cards = cards + board
    ranks = {}
    suits = {}
    for card in all_cards:
        ranks[_rank_value(card)] = ranks.get(_rank_value(card), 0) + 1
        if len(str(card)) > 1:
            suits[str(card)[1]] = suits.get(str(card)[1], 0) + 1
    pairish = 0.36 if any(count >= 2 for count in ranks.values()) else 0.0
    trips = 0.26 if any(count >= 3 for count in ranks.values()) else 0.0
    flush_draw = 0.16 if len(board) < 5 and max(suits.values(), default=0) >= 4 else 0.0
    vals = set(ranks)
    if 14 in vals:
        vals.add(1)
    straight_draw = 0.0
    for start in range(1, 11):
        if len(vals & {start, start + 1, start + 2, start + 3, start + 4}) >= 4:
            straight_draw = 0.12
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
        mask[2:7] = False
    return mask


def strategic_mask(state: dict) -> np.ndarray:
    """Legal mask with a benchmark-policy all-in guard.

    Strong mock neural policies are allowed to explore, but treating every
    legal all-in as a normal action creates brittle benchmark opponents. This
    keeps all-in available for short-stack, already-committed, and clearly
    strong/draw-heavy states while removing it from ordinary deep-stack spots.
    """
    mask = legal_mask(state)
    if not mask[7]:
        return mask
    stack = int(state.get("your_stack", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    total_stack = stack + invested
    if total_stack <= 15 * BIG_BLIND or owed >= max(1, int(stack * 0.55)):
        return mask
    if state.get("street") == "preflop":
        if _preflop_strength(list(state.get("your_cards", []) or [])) >= 0.86:
            return mask
    elif _postflop_commit_signal(state) >= 0.62 and stack <= pot * 2.2:
        return mask
    mask[7] = False
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
    mask = strategic_mask(state)
    values[~mask] = -1e9
    return int(np.argmax(values))
