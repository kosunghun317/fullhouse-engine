"""Benchmark opponent: shoves often with only minimal hand awareness."""


def decide(state):
    cards = state.get("your_cards", [])
    ranks = [c[0] for c in cards]
    strong = len(ranks) == 2 and (ranks[0] == ranks[1] or "A" in ranks or "K" in ranks)

    if strong or state.get("pot", 0) >= 600:
        return {"action": "all_in"}
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"}
