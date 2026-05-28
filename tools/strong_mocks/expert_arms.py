"""Human-heuristic expert arms for benchmark-only selector bots.

These arms deliberately generate poker-shaped candidate actions instead of raw
action buckets. A learned selector can rank them, but it cannot invent an
unjustified all-in or overbet outside the expert contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tools.strong_mocks.deep_actions import raise_to_fraction, sanitize_action
from tools.strong_mocks.features import extract_features


BIG_BLIND = 100

ARM_NAMES = (
    "default_tag",
    "pot_control",
    "value_station",
    "anti_maniac",
    "pressure_folder",
    "pot_odds_breaker",
    "spr_commit",
    "short_stack_pushfold",
    "range_advantage_cbet",
    "blocker_bluff",
    "showdown_value",
    "strong_unknown_avoidance",
)

INTENTS = (
    "fold",
    "check_call",
    "value",
    "bluff",
    "semi_bluff",
    "trap",
    "pot_control",
    "commit",
)

TARGET_TYPES = ("unknown", "folder", "station", "maniac", "strong")


@dataclass(frozen=True)
class Candidate:
    arm: str
    action: dict
    intent: str
    risk: float
    confidence: float
    target_type: str
    equity_estimate: float
    score_hint: float

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "action": dict(self.action),
            "intent": self.intent,
            "risk": float(self.risk),
            "confidence": float(self.confidence),
            "target_type": self.target_type,
            "equity_estimate": float(self.equity_estimate),
            "score_hint": float(self.score_hint),
        }


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _active_players(state: dict) -> int:
    return sum(1 for player in state.get("players", []) if not player.get("is_folded"))


def _hero_bot_id(state: dict) -> str:
    players = state.get("players", [])
    seat = int(state.get("seat_to_act", 0) or 0)
    if 0 <= seat < len(players):
        return str(players[seat].get("bot_id", "hero"))
    return "hero"


def _seat_to_bot_id(state: dict, seat: int) -> str:
    players = state.get("players", [])
    if 0 <= seat < len(players):
        return str(players[seat].get("bot_id", f"seat_{seat}"))
    return f"seat_{seat}"


def _player_stats(entries: Iterable[dict]) -> dict[str, dict[str, float]]:
    raw: dict[str, dict[str, float]] = {}
    for item in entries:
        action = str(item.get("action", ""))
        if action in ("small_blind", "big_blind", "post_blind", "ante"):
            continue
        bot_id = item.get("bot_id")
        if bot_id is None:
            continue
        stats = raw.setdefault(
            str(bot_id),
            {"actions": 0.0, "raise": 0.0, "call": 0.0, "fold": 0.0, "check": 0.0, "all_in": 0.0},
        )
        stats["actions"] += 1.0
        if action in stats:
            stats[action] += 1.0
    return raw


def _classify(stats: dict[str, float] | None) -> tuple[str, dict[str, float]]:
    if not stats:
        return "unknown", {
            "raise_rate": 0.20,
            "call_rate": 0.25,
            "fold_rate": 0.25,
            "all_in_rate": 0.02,
            "actions": 0.0,
        }
    actions = float(stats.get("actions", 0.0))
    raise_rate = (stats.get("raise", 0.0) + 1.0) / (actions + 4.0)
    call_rate = (stats.get("call", 0.0) + 1.0) / (actions + 4.0)
    fold_rate = (stats.get("fold", 0.0) + 1.0) / (actions + 4.0)
    all_in_rate = (stats.get("all_in", 0.0) + 0.5) / (actions + 4.0)
    rates = {
        "raise_rate": float(raise_rate),
        "call_rate": float(call_rate),
        "fold_rate": float(fold_rate),
        "all_in_rate": float(all_in_rate),
        "actions": float(actions),
    }
    if actions < 7:
        return "unknown", rates
    if all_in_rate > 0.06 or raise_rate > 0.33:
        return "maniac", rates
    if call_rate > 0.42 and fold_rate < 0.30:
        return "station", rates
    if fold_rate > 0.42 and raise_rate < 0.20:
        return "folder", rates
    if raise_rate < 0.17 and call_rate < 0.28 and fold_rate > 0.34:
        return "strong", rates
    return "unknown", rates


def _target_profile(state: dict) -> dict:
    seat_to_bot = {
        int(player.get("seat", idx)): str(player.get("bot_id", f"seat_{idx}"))
        for idx, player in enumerate(state.get("players", []))
    }
    match_entries = list(state.get("match_action_log", []) or [])
    hand_entries = []
    for item in state.get("action_log", []) or []:
        row = dict(item)
        if "bot_id" not in row and "seat" in row:
            row["bot_id"] = seat_to_bot.get(int(row["seat"]), f"seat_{row['seat']}")
        hand_entries.append(row)
    stats_by_bot = _player_stats(match_entries + hand_entries)
    hero = _hero_bot_id(state)

    target_bot = None
    if int(state.get("amount_owed", 0) or 0) > 0:
        for item in reversed(hand_entries):
            if item.get("action") in ("raise", "all_in"):
                bot_id = str(item.get("bot_id", ""))
                if bot_id and bot_id != hero:
                    target_bot = bot_id
                    break

    if target_bot is None:
        active_profiles = []
        for idx, player in enumerate(state.get("players", [])):
            bot_id = str(player.get("bot_id", f"seat_{idx}"))
            if bot_id == hero or player.get("is_folded"):
                continue
            kind, rates = _classify(stats_by_bot.get(bot_id))
            active_profiles.append((kind, rates, bot_id))
        priority = {"station": 0, "maniac": 1, "folder": 2, "strong": 3, "unknown": 4}
        active_profiles.sort(key=lambda row: priority.get(row[0], 9))
        if active_profiles:
            kind, rates, target_bot = active_profiles[0]
        else:
            kind, rates = _classify(None)
            target_bot = "unknown"
    else:
        kind, rates = _classify(stats_by_bot.get(target_bot))

    fold_pressure = _clip(rates["fold_rate"] + 0.18 * (kind == "folder") - 0.18 * (kind == "station"))
    station_score = _clip(rates["call_rate"] - rates["fold_rate"] + 0.20 * (kind == "station"))
    maniac_score = _clip(rates["raise_rate"] + 1.5 * rates["all_in_rate"] + 0.20 * (kind == "maniac"))
    return {
        "bot_id": target_bot,
        "target_type": kind,
        "rates": rates,
        "fold_pressure": fold_pressure,
        "station_score": station_score,
        "maniac_score": maniac_score,
    }


def _hero_was_preflop_aggressor(state: dict) -> bool:
    hero_seat = int(state.get("seat_to_act", 0) or 0)
    last_raiser = None
    for item in state.get("action_log", []) or []:
        if item.get("action") == "raise":
            last_raiser = int(item.get("seat", -1) or -1)
    return last_raiser == hero_seat


def _equity_proxy(features: np.ndarray, state: dict) -> float:
    strength = float(features[17])
    active = max(2, _active_players(state))
    if state.get("street") == "preflop":
        return _clip(strength + 0.03 * features[11] - 0.055 * max(0, active - 2))
    made = float(features[22])
    draw = float(max(features[23], features[24]))
    wet = float(features[18])
    paired = float(features[19])
    broadway = float(features[21])
    equity = 0.17 + 0.44 * strength + 0.20 * made + 0.11 * draw + 0.04 * broadway
    equity -= 0.045 * wet + 0.025 * paired + 0.055 * max(0, active - 2)
    return _clip(equity, 0.03, 0.97)


def build_context(state: dict) -> dict:
    features = extract_features(state)
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    stack = max(0, int(state.get("your_stack", 0) or 0))
    invested = max(0, int(state.get("your_bet_this_street", 0) or 0))
    total_stack = max(1, stack + invested)
    active = max(1, _active_players(state))
    target = _target_profile(state)
    equity = _equity_proxy(features, state)
    draw = float(max(features[23], features[24]))
    made = float(features[22])
    return {
        "features": features,
        "street": str(state.get("street", "preflop")),
        "owed": owed,
        "pot": pot,
        "stack": stack,
        "invested": invested,
        "total_stack": total_stack,
        "pot_odds": owed / max(1, pot + owed),
        "can_check": bool(state.get("can_check", False)),
        "active": active,
        "multiway": active >= 3,
        "spr": stack / max(1, pot),
        "position": float(features[11]),
        "strength": float(features[17]),
        "equity": equity,
        "made": made,
        "draw": draw,
        "wet": float(features[18]),
        "dry": 1.0 - float(features[18]),
        "paired": float(features[19]),
        "monotone": float(features[20]),
        "broadway": float(features[21]),
        "target_type": target["target_type"],
        "target_rates": target["rates"],
        "fold_pressure": target["fold_pressure"],
        "station_score": target["station_score"],
        "maniac_score": target["maniac_score"],
        "hero_was_preflop_aggressor": _hero_was_preflop_aggressor(state),
    }


def _action_risk(state: dict, action: dict) -> float:
    stack = max(1, int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0))
    act = action.get("action")
    if act in ("fold", "check"):
        return 0.0
    if act == "call":
        return _clip(int(state.get("amount_owed", 0) or 0) / stack)
    if act == "all_in":
        return 1.0
    if act == "raise":
        invested = int(state.get("your_bet_this_street", 0) or 0)
        amount = int(action.get("amount", 0) or 0)
        return _clip(max(0, amount - invested) / stack)
    return 0.0


def _passive_action(state: dict) -> dict:
    return sanitize_action(state, {"action": "check"} if state.get("can_check") else {"action": "fold"})


def _call_action(state: dict) -> dict:
    return sanitize_action(state, {"action": "check"} if state.get("can_check") else {"action": "call"})


def _cand(
    state: dict,
    ctx: dict,
    arm: str,
    action: dict,
    intent: str,
    confidence: float,
    score_hint: float = 0.0,
    equity: float | None = None,
) -> Candidate:
    clean = sanitize_action(state, action)
    risk = _action_risk(state, clean)
    return Candidate(
        arm=arm,
        action=clean,
        intent=intent if intent in INTENTS else "check_call",
        risk=risk,
        confidence=_clip(confidence),
        target_type=ctx["target_type"] if ctx["target_type"] in TARGET_TYPES else "unknown",
        equity_estimate=_clip(ctx["equity"] if equity is None else equity),
        score_hint=float(score_hint),
    )


def _value_threshold(ctx: dict) -> float:
    return 0.66 + (0.07 if ctx["multiway"] else 0.0) - 0.04 * (ctx["target_type"] == "station")


def _call_margin(ctx: dict) -> float:
    margin = 0.075 + (0.04 if ctx["multiway"] else 0.0)
    margin += 0.035 if ctx["street"] == "river" else 0.0
    margin += 0.035 if ctx["target_type"] in ("strong", "folder") else 0.0
    margin -= 0.045 if ctx["target_type"] == "maniac" else 0.0
    return max(0.02, margin)


def _default_tag(state: dict, ctx: dict) -> Candidate:
    equity = ctx["equity"]
    if ctx["can_check"]:
        if equity >= _value_threshold(ctx):
            return _cand(state, ctx, "default_tag", raise_to_fraction(state, 0.55), "value", 0.66 + equity * 0.25, 0.15)
        if ctx["draw"] and equity >= 0.39 and ctx["fold_pressure"] > 0.38 and not ctx["multiway"]:
            return _cand(state, ctx, "default_tag", raise_to_fraction(state, 0.45), "semi_bluff", 0.55, 0.05)
        return _cand(state, ctx, "default_tag", _passive_action(state), "check_call", 0.58, 0.0)
    if equity >= ctx["pot_odds"] + _call_margin(ctx):
        if equity > 0.80 and ctx["owed"] < ctx["pot"] * 0.35:
            return _cand(state, ctx, "default_tag", raise_to_fraction(state, 0.75), "value", 0.75, 0.12)
        return _cand(state, ctx, "default_tag", _call_action(state), "check_call", 0.62, 0.04)
    return _cand(state, ctx, "default_tag", _passive_action(state), "fold", 0.66, -0.02)


def _pot_control(state: dict, ctx: dict) -> Candidate:
    if ctx["can_check"]:
        return _cand(state, ctx, "pot_control", _passive_action(state), "pot_control", 0.68, 0.04)
    threshold = ctx["pot_odds"] + max(0.055, _call_margin(ctx) - 0.025)
    if ctx["equity"] >= threshold and ctx["owed"] <= ctx["pot"] * 0.45:
        return _cand(state, ctx, "pot_control", _call_action(state), "pot_control", 0.58, 0.03)
    return _cand(state, ctx, "pot_control", _passive_action(state), "fold", 0.64, 0.0)


def _value_station(state: dict, ctx: dict) -> Candidate:
    equity = ctx["equity"]
    made_or_strong = ctx["made"] or equity > 0.61 or ctx["strength"] > 0.78
    if ctx["can_check"]:
        if made_or_strong and equity > 0.54:
            frac = 0.85 if equity > 0.72 else 0.67
            return _cand(state, ctx, "value_station", raise_to_fraction(state, frac), "value", 0.68 + 0.20 * ctx["station_score"], 0.18)
        return _cand(state, ctx, "value_station", _passive_action(state), "check_call", 0.54, -0.01)
    if equity >= ctx["pot_odds"] + 0.08:
        if equity > 0.83 and ctx["owed"] < ctx["pot"] * 0.30:
            return _cand(state, ctx, "value_station", raise_to_fraction(state, 0.85), "value", 0.78, 0.16)
        return _cand(state, ctx, "value_station", _call_action(state), "check_call", 0.62, 0.04)
    return _cand(state, ctx, "value_station", _passive_action(state), "fold", 0.60, -0.04)


def _anti_maniac(state: dict, ctx: dict) -> Candidate:
    equity = ctx["equity"]
    if ctx["can_check"]:
        if equity > 0.82 and ctx["spr"] <= 3.0:
            return _cand(state, ctx, "anti_maniac", raise_to_fraction(state, 0.75), "value", 0.76, 0.13)
        return _cand(state, ctx, "anti_maniac", _passive_action(state), "trap", 0.62 + 0.18 * ctx["maniac_score"], 0.07)
    if equity >= ctx["pot_odds"] + max(0.02, _call_margin(ctx) - 0.05):
        if equity > 0.84 and ctx["owed"] < ctx["pot"] * 0.40:
            return _cand(state, ctx, "anti_maniac", raise_to_fraction(state, 0.85), "value", 0.73, 0.11)
        return _cand(state, ctx, "anti_maniac", _call_action(state), "trap", 0.67, 0.08)
    return _cand(state, ctx, "anti_maniac", _passive_action(state), "fold", 0.60, -0.03)


def _pressure_folder(state: dict, ctx: dict) -> Candidate:
    if ctx["can_check"] and not ctx["multiway"]:
        board_story = 0.25 * ctx["dry"] + 0.20 * ctx["broadway"] + 0.18 * ctx["position"]
        if ctx["equity"] > 0.61:
            return _cand(state, ctx, "pressure_folder", raise_to_fraction(state, 0.67), "value", 0.64, 0.10)
        if ctx["fold_pressure"] + board_story > 0.63 and ctx["equity"] > 0.26:
            return _cand(state, ctx, "pressure_folder", raise_to_fraction(state, 0.56), "bluff", 0.61, 0.12)
    if not ctx["can_check"] and ctx["equity"] >= ctx["pot_odds"] + _call_margin(ctx):
        return _cand(state, ctx, "pressure_folder", _call_action(state), "check_call", 0.54, 0.01)
    return _cand(state, ctx, "pressure_folder", _passive_action(state), "fold" if not ctx["can_check"] else "check_call", 0.55, 0.0)


def _pot_odds_breaker(state: dict, ctx: dict) -> Candidate:
    if ctx["can_check"]:
        if ctx["equity"] > 0.62 or (ctx["made"] and ctx["equity"] > 0.55):
            return _cand(state, ctx, "pot_odds_breaker", raise_to_fraction(state, 0.48), "value", 0.64, 0.13)
        if ctx["fold_pressure"] > 0.38 and ctx["target_type"] != "station" and not ctx["multiway"] and ctx["equity"] > 0.24:
            return _cand(state, ctx, "pot_odds_breaker", raise_to_fraction(state, 0.56), "bluff", 0.59, 0.11)
    if not ctx["can_check"] and ctx["equity"] >= ctx["pot_odds"] + 0.065:
        return _cand(state, ctx, "pot_odds_breaker", _call_action(state), "check_call", 0.55, 0.02)
    return _cand(state, ctx, "pot_odds_breaker", _passive_action(state), "pot_control", 0.53, 0.0)


def _spr_commit(state: dict, ctx: dict) -> Candidate:
    commit = ctx["spr"] <= 2.5 and (ctx["equity"] > 0.67 or ctx["strength"] > 0.83 or (ctx["draw"] and ctx["equity"] > 0.52))
    if commit:
        if ctx["stack"] <= ctx["pot"] * 1.15 or ctx["equity"] > 0.76:
            return _cand(state, ctx, "spr_commit", {"action": "all_in"}, "commit", 0.72, 0.17)
        return _cand(state, ctx, "spr_commit", raise_to_fraction(state, 0.85), "commit", 0.66, 0.11)
    if not ctx["can_check"] and ctx["equity"] >= ctx["pot_odds"] + 0.05 and ctx["owed"] < ctx["stack"] * 0.28:
        return _cand(state, ctx, "spr_commit", _call_action(state), "check_call", 0.56, 0.02)
    return _cand(state, ctx, "spr_commit", _passive_action(state), "pot_control", 0.50, -0.02)


def _short_stack_pushfold(state: dict, ctx: dict) -> Candidate:
    short = ctx["total_stack"] <= 15 * BIG_BLIND
    if short and ctx["street"] == "preflop":
        threshold = 0.62 - 0.09 * ctx["position"] + (0.08 if not ctx["can_check"] else 0.0)
        if ctx["strength"] >= threshold:
            return _cand(state, ctx, "short_stack_pushfold", {"action": "all_in"}, "commit", 0.74, 0.19)
        return _cand(state, ctx, "short_stack_pushfold", _passive_action(state), "fold", 0.68, 0.02)
    if short and ctx["equity"] > 0.62:
        return _cand(state, ctx, "short_stack_pushfold", {"action": "all_in"}, "commit", 0.70, 0.12)
    return _cand(state, ctx, "short_stack_pushfold", _default_tag(state, ctx).action, "check_call", 0.47, -0.03)


def _range_advantage_cbet(state: dict, ctx: dict) -> Candidate:
    good_board = bool(ctx["broadway"] or (ctx["paired"] and ctx["dry"]))
    if ctx["can_check"] and not ctx["multiway"] and ctx["street"] in ("flop", "turn"):
        if ctx["hero_was_preflop_aggressor"] and good_board:
            frac = 0.33 if ctx["dry"] else 0.45
            intent = "value" if ctx["equity"] > 0.58 else "bluff"
            conf = 0.58 + 0.12 * ctx["broadway"] + 0.08 * ctx["fold_pressure"]
            return _cand(state, ctx, "range_advantage_cbet", raise_to_fraction(state, frac), intent, conf, 0.10)
    return _cand(state, ctx, "range_advantage_cbet", _passive_action(state), "check_call", 0.50, 0.0)


def _blocker_bluff(state: dict, ctx: dict) -> Candidate:
    draw_or_blocker = bool(ctx["draw"] or (ctx["monotone"] and ctx["strength"] > 0.55))
    if (
        ctx["can_check"]
        and not ctx["multiway"]
        and draw_or_blocker
        and ctx["target_type"] in ("folder", "unknown", "strong")
        and ctx["fold_pressure"] > 0.36
        and 0.24 <= ctx["equity"] <= 0.62
    ):
        return _cand(state, ctx, "blocker_bluff", raise_to_fraction(state, 0.67), "semi_bluff", 0.57, 0.10)
    if not ctx["can_check"] and ctx["draw"] and ctx["equity"] >= ctx["pot_odds"] + 0.035:
        return _cand(state, ctx, "blocker_bluff", _call_action(state), "check_call", 0.51, 0.01)
    return _cand(state, ctx, "blocker_bluff", _passive_action(state), "check_call", 0.48, -0.01)


def _showdown_value(state: dict, ctx: dict) -> Candidate:
    if ctx["can_check"]:
        if ctx["equity"] > 0.69 and ctx["target_type"] != "strong":
            return _cand(state, ctx, "showdown_value", raise_to_fraction(state, 0.45), "value", 0.60, 0.06)
        return _cand(state, ctx, "showdown_value", _passive_action(state), "pot_control", 0.66, 0.05)
    if ctx["equity"] >= ctx["pot_odds"] + 0.045 and ctx["owed"] < ctx["pot"] * 0.65:
        return _cand(state, ctx, "showdown_value", _call_action(state), "check_call", 0.63, 0.05)
    return _cand(state, ctx, "showdown_value", _passive_action(state), "fold", 0.61, 0.01)


def _strong_unknown_avoidance(state: dict, ctx: dict) -> Candidate:
    if ctx["can_check"]:
        if ctx["equity"] > 0.76 and ctx["target_type"] != "strong":
            return _cand(state, ctx, "strong_unknown_avoidance", raise_to_fraction(state, 0.55), "value", 0.56, 0.04)
        return _cand(state, ctx, "strong_unknown_avoidance", _passive_action(state), "pot_control", 0.67, 0.06)
    if ctx["equity"] >= ctx["pot_odds"] + 0.12 and ctx["owed"] < ctx["stack"] * 0.22:
        return _cand(state, ctx, "strong_unknown_avoidance", _call_action(state), "check_call", 0.61, 0.03)
    return _cand(state, ctx, "strong_unknown_avoidance", _passive_action(state), "fold", 0.68, 0.04)


ARM_FUNCTIONS = {
    "default_tag": _default_tag,
    "pot_control": _pot_control,
    "value_station": _value_station,
    "anti_maniac": _anti_maniac,
    "pressure_folder": _pressure_folder,
    "pot_odds_breaker": _pot_odds_breaker,
    "spr_commit": _spr_commit,
    "short_stack_pushfold": _short_stack_pushfold,
    "range_advantage_cbet": _range_advantage_cbet,
    "blocker_bluff": _blocker_bluff,
    "showdown_value": _showdown_value,
    "strong_unknown_avoidance": _strong_unknown_avoidance,
}


def generate_candidates(state: dict) -> list[Candidate]:
    if state.get("type") == "warmup":
        return [
            Candidate(
                arm="default_tag",
                action={"action": "check"},
                intent="check_call",
                risk=0.0,
                confidence=1.0,
                target_type="unknown",
                equity_estimate=0.5,
                score_hint=0.0,
            )
        ]
    ctx = build_context(state)
    candidates = []
    for arm in ARM_NAMES:
        try:
            candidates.append(ARM_FUNCTIONS[arm](state, ctx))
        except Exception:
            candidates.append(_cand(state, ctx, arm, _passive_action(state), "pot_control", 0.35, -0.10))
    return candidates


def default_candidate_score(candidate: Candidate) -> float:
    intent_bonus = {
        "value": 0.08,
        "semi_bluff": 0.04,
        "trap": 0.05,
        "pot_control": 0.03,
        "commit": 0.02,
        "bluff": -0.01,
        "fold": 0.0,
        "check_call": 0.02,
    }.get(candidate.intent, 0.0)
    action = candidate.action.get("action")
    legality_bonus = 0.02 if action in ("check", "call", "raise", "all_in", "fold") else -0.15
    return (
        0.95 * candidate.confidence
        + 0.55 * candidate.equity_estimate
        - 0.42 * candidate.risk
        + candidate.score_hint
        + intent_bonus
        + legality_bonus
    )
