"""Mock competitor: classifies opponents from match_action_log and counters."""


def _stats(state):
    stats = {}
    for action in state.get("match_action_log", []):
        bot_id = action.get("bot_id")
        act = action.get("action")
        if not bot_id or act in (None, "small_blind", "big_blind"):
            continue
        item = stats.setdefault(bot_id, {"actions": 0, "raises": 0, "calls": 0, "folds": 0})
        item["actions"] += 1
        if act in ("raise", "all_in"):
            item["raises"] += 1
        elif act == "call":
            item["calls"] += 1
        elif act == "fold":
            item["folds"] += 1
    return stats


def _table_profile(state):
    seat = state.get("seat_to_act")
    stats = _stats(state)
    profiles = []
    for player in state.get("players", []):
        if player.get("seat") == seat or player.get("is_folded"):
            continue
        item = stats.get(player.get("bot_id"), {})
        actions = item.get("actions", 0)
        if actions < 8:
            profiles.append("unknown")
            continue
        raise_rate = (item.get("raises", 0) + 1) / (actions + 4)
        call_rate = (item.get("calls", 0) + 1) / (actions + 4)
        fold_rate = (item.get("folds", 0) + 1) / (actions + 4)
        if raise_rate > 0.32:
            profiles.append("maniac")
        elif call_rate > 0.45 and fold_rate < 0.30:
            profiles.append("station")
        elif fold_rate > 0.45 and raise_rate < 0.20:
            profiles.append("folder")
        else:
            profiles.append("regular")
    for profile in ("station", "maniac", "folder", "regular"):
        if profile in profiles:
            return profile
    return "unknown"


def _strength(cards):
    if len(cards) < 2:
        return 0.0
    order = "23456789TJQKA"
    vals = sorted((order.index(c[0]) + 2 for c in cards), reverse=True)
    score = vals[0] / 14 * 0.55 + vals[1] / 14 * 0.25
    if vals[0] == vals[1]:
        score += 0.25
    if cards[0][1] == cards[1][1]:
        score += 0.05
    return min(1.0, score)


def _raise_to(state, frac):
    pot = max(1, int(state.get("pot", 0) or 0))
    total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
    amount = max(int(state.get("min_raise_to", 0) or 0), int(state.get("current_bet", 0) or 0) + int(pot * frac))
    return {"action": "raise", "amount": min(total, amount)}


def decide(state):
    profile = _table_profile(state)
    strength = _strength(state.get("your_cards", []))
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    if state.get("can_check"):
        if profile == "folder" and strength > 0.35:
            return _raise_to(state, 0.58)
        if profile == "station" and strength > 0.70:
            return _raise_to(state, 0.85)
        if profile == "maniac" and strength > 0.82:
            return {"action": "check"}
        if strength > 0.78:
            return _raise_to(state, 0.60)
        return {"action": "check"}
    if profile == "maniac" and strength > 0.55 and owed <= pot * 0.70:
        return {"action": "call"}
    if profile == "station" and owed > pot * 0.35:
        return {"action": "fold"}
    if strength > 0.68 and owed <= pot * 0.40:
        return {"action": "call"}
    return {"action": "fold"}
