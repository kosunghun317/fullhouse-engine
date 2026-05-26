"""Mock competitor: modified Shark-style tight regular."""

import random


STRONG = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs", "KQs"}
MEDIUM = STRONG | {"99", "88", "77", "AJo", "ATs", "KQo", "KJs", "QJs", "JTs"}


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


def _position(state):
    players = state.get("players", [])
    seat = int(state.get("seat_to_act", 0) or 0)
    n = max(1, len(players))
    sb = None
    for action in state.get("action_log", []):
        if action.get("action") == "small_blind":
            sb = int(action.get("seat", 0))
            break
    if sb is None:
        return seat / max(1, n - 1)
    dealer = (sb - 1) % n
    return ((seat - dealer) % n) / max(1, n - 1)


def _raise_to(state, frac):
    pot = max(1, int(state.get("pot", 0) or 0))
    total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
    amount = max(int(state.get("min_raise_to", 0) or 0), int(state.get("current_bet", 0) or 0) + int(pot * frac))
    return {"action": "raise", "amount": min(total, amount)}


def decide(state):
    cls = _class(state.get("your_cards", []))
    pos = _position(state)
    if state.get("street") == "preflop":
        if cls in STRONG:
            return _raise_to(state, 0.70)
        if cls in MEDIUM and pos > 0.45:
            return _raise_to(state, 0.45) if state.get("can_check") else {"action": "call"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if state.get("can_check"):
        if pos > 0.55 and random.random() < 0.38:
            return _raise_to(state, 0.55)
        return {"action": "check"}

    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    threshold = 0.30 if pos > 0.50 else 0.18
    if owed <= pot * threshold:
        return {"action": "call"}
    return {"action": "fold"}
