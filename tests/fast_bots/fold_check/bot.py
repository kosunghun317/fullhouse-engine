def decide(game_state):
    if game_state.get("type") == "warmup":
        return {"action": "check"}
    if game_state.get("can_check"):
        return {"action": "check"}
    return {"action": "fold"}
