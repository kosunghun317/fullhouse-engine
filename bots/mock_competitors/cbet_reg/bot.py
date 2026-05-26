"""Mock competitor: chart-based regular that overuses continuation bets."""

import random


STRONG = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs", "KQs"}
PLAY = STRONG | {"99", "88", "77", "AJo", "ATs", "KQo", "KJs", "QJs", "JTs", "T9s", "98s"}


def _class(cards):
    order = {rank: index + 2 for index, rank in enumerate("23456789TJQKA")}
    if len(cards) < 2:
        return "72o"
    c1, c2 = cards[0], cards[1]
    r1, r2 = c1[0], c2[0]
    if order[r2] > order[r1]:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if c1[1] == c2[1] else "o")


def _raise_to(state, frac):
    pot = max(1, int(state.get("pot", 0) or 0))
    total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
    amount = max(int(state.get("min_raise_to", 0) or 0), int(state.get("current_bet", 0) or 0) + int(pot * frac))
    return {"action": "raise", "amount": min(total, amount)}


def _was_preflop_raiser(state):
    seat = state.get("seat_to_act")
    for action in state.get("action_log", []):
        if action.get("seat") == seat and action.get("action") in ("raise", "all_in"):
            return True
    return False


def decide(state):
    cls = _class(state.get("your_cards", []))
    if state.get("street") == "preflop":
        if cls in STRONG:
            return _raise_to(state, 0.65)
        if cls in PLAY and state.get("can_check"):
            return _raise_to(state, 0.45)
        if cls in PLAY and int(state.get("amount_owed", 0) or 0) <= max(150, int(state.get("pot", 0) * 0.18)):
            return {"action": "call"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if state.get("can_check"):
        if _was_preflop_raiser(state) and random.random() < 0.72:
            return _raise_to(state, 0.50)
        if random.random() < 0.22:
            return _raise_to(state, 0.38)
        return {"action": "check"}

    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= max(100, int(pot * 0.22)):
        return {"action": "call"}
    return {"action": "fold"}
