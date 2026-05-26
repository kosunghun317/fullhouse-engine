"""Benchmark opponent: creates short-stack all-in pressure."""


def decide(state):
    stack = int(state.get("your_stack", 0) or 0)
    cards = state.get("your_cards", [])
    ranks = [card[0] for card in cards]
    pair = len(ranks) == 2 and ranks[0] == ranks[1]
    ace_or_king = "A" in ranks or "K" in ranks

    if stack <= 2500 or pair or ace_or_king:
        return {"action": "all_in"}
    if state.get("can_check"):
        return {"action": "check"}
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= max(100, int(pot * 0.18)):
        return {"action": "call"}
    return {"action": "fold"}
