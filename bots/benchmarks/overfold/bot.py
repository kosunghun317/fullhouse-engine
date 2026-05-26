"""Benchmark opponent: folds to pressure and only checks/calls tiny prices."""


def decide(state):
    if state.get("can_check"):
        return {"action": "check"}
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= max(100, int(pot * 0.08)):
        return {"action": "call"}
    return {"action": "fold"}
