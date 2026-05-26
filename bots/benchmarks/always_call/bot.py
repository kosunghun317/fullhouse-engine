"""Benchmark opponent: calls everything, checks when free."""


def decide(state):
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"}
