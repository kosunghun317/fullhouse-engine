import time


def decide(game_state):
    if game_state.get("type") == "warmup":
        return {"action": "check"}
    time.sleep(0.05)
    if game_state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"}
