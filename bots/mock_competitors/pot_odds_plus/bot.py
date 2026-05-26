"""Mock competitor: pot-odds threshold bot with street-aware thresholds."""


def decide(state):
    if state.get("can_check"):
        return {"action": "check"}
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= 0:
        return {"action": "check"}
    ratio = pot / owed
    street = state.get("street")
    threshold = 3.2
    if street == "preflop":
        threshold = 2.4
    elif street == "river":
        threshold = 3.8
    if ratio >= threshold:
        return {"action": "call"}
    return {"action": "fold"}
