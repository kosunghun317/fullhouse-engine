"""Benchmark opponent: applies frequent half-pot pressure."""

import random


def _high_card(cards):
    order = "23456789TJQKA"
    return max((order.index(card[0]) + 2 for card in cards), default=2)


def decide(state):
    pot = max(1, int(state.get("pot", 0) or 0))
    stack_total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
    target = min(stack_total, max(int(state.get("min_raise_to", 0) or 0), int(state.get("current_bet", 0) or 0) + int(pot * 0.5)))
    strongish = _high_card(state.get("your_cards", [])) >= 12

    if state.get("can_check") and (strongish or random.random() < 0.45):
        return {"action": "raise", "amount": target}
    if not state.get("can_check"):
        owed = int(state.get("amount_owed", 0) or 0)
        if strongish and owed <= pot * 0.45:
            return {"action": "call"}
        if random.random() < 0.20 and owed <= pot * 0.25:
            return {"action": "call"}
    return {"action": "check"} if state.get("can_check") else {"action": "fold"}
