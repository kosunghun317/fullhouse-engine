"""Mock competitor: coarse CFR-like bucket policy with fixed action buckets."""

import random


def _rank_value(card):
    return "23456789TJQKA".index(card[0]) + 2


def _bucket(state):
    cards = state.get("your_cards", [])
    board = state.get("community_cards", [])
    values = sorted((_rank_value(c) for c in cards), reverse=True) if len(cards) >= 2 else [2, 2]
    pair = len(cards) >= 2 and cards[0][0] == cards[1][0]
    suited = len(cards) >= 2 and cards[0][1] == cards[1][1]
    board_suits = {}
    board_vals = []
    for card in board:
        board_suits[card[1]] = board_suits.get(card[1], 0) + 1
        board_vals.append(_rank_value(card))
    wet = max(board_suits.values(), default=0) >= 3 or (max(board_vals, default=2) - min(board_vals, default=2) <= 5 and len(board_vals) >= 3)
    if pair and values[0] >= 10:
        return "premium"
    if values[0] == 14 and values[1] >= 10:
        return "premium"
    if pair or suited or values[0] >= 12:
        return "medium_wet" if wet else "medium_dry"
    return "trash_wet" if wet else "trash_dry"


def _raise_to(state, frac):
    pot = max(1, int(state.get("pot", 0) or 0))
    total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
    amount = max(int(state.get("min_raise_to", 0) or 0), int(state.get("current_bet", 0) or 0) + int(pot * frac))
    return {"action": "raise", "amount": min(total, amount)}


def decide(state):
    bucket = _bucket(state)
    if state.get("street") == "preflop":
        if bucket == "premium":
            return _raise_to(state, 0.80)
        if bucket.startswith("medium") and state.get("can_check"):
            return _raise_to(state, 0.40)
        if bucket.startswith("medium") and int(state.get("amount_owed", 0) or 0) <= int(max(100, state.get("pot", 0) * 0.16)):
            return {"action": "call"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if state.get("can_check"):
        if bucket == "premium":
            return _raise_to(state, 0.75 if random.random() < 0.65 else 0.35)
        if bucket == "medium_dry" and random.random() < 0.55:
            return _raise_to(state, 0.50)
        if bucket == "trash_dry" and random.random() < 0.18:
            return _raise_to(state, 0.67)
        return {"action": "check"}

    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    threshold = 0.34 if bucket == "premium" else 0.24 if bucket.startswith("medium") else 0.10
    if owed <= pot * threshold:
        return {"action": "call"}
    return {"action": "fold"}
