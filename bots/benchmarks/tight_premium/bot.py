"""Benchmark opponent: tight premium-heavy opens and strong folds."""


PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"}
PLAYABLE = PREMIUM | {"TT", "99", "AQo", "AJs", "KQs"}


def _hand_class(cards):
    order = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
    if len(cards) < 2:
        return "72o"
    c1, c2 = cards[0], cards[1]
    r1, r2 = c1[0], c2[0]
    if order[r2] > order[r1]:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if c1[1] == c2[1] else "o")


def decide(state):
    cls = _hand_class(state.get("your_cards", []))
    if state.get("street") == "preflop":
        if cls in PREMIUM:
            stack_total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
            amount = min(stack_total, max(int(state.get("min_raise_to", 0) or 0), 350))
            return {"action": "raise", "amount": amount}
        if cls in PLAYABLE and int(state.get("amount_owed", 0) or 0) <= 250:
            return {"action": "call"} if not state.get("can_check") else {"action": "check"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if state.get("can_check"):
        return {"action": "check"}
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= max(100, int(pot * 0.12)):
        return {"action": "call"}
    return {"action": "fold"}
