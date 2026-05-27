def decide(game_state):
    if game_state.get("type") == "warmup":
        return {"action": "check"}
    if game_state.get("can_check"):
        return {"action": "raise", "amount": int(game_state.get("min_raise_to", 0))}
    if game_state.get("min_raise_to"):
        return {"action": "raise", "amount": int(game_state["min_raise_to"])}
    return {"action": "call"}
