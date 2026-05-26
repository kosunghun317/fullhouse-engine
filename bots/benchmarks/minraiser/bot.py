"""Benchmark opponent: min-raises most openings and calls otherwise."""


def decide(state):
    if state.get("your_stack", 0) > state.get("amount_owed", 0):
        return {"action": "raise", "amount": state.get("min_raise_to", 0)}
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"}
