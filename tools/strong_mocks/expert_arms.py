"""Human-heuristic expert arms for benchmark-only selector bots.

These arms deliberately generate poker-shaped candidate actions instead of raw
action buckets. A learned selector can rank them, but it cannot invent an
unjustified all-in or overbet outside the expert contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tools.strong_mocks.arm_selector_params import coerce_params
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


def _classify(stats: dict[str, float] | None, params: dict[str, float]) -> tuple[str, dict[str, float]]:
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
    if actions < int(round(params["classify_min_actions"])):
        return "unknown", rates
    if all_in_rate > params["maniac_allin_rate"] or raise_rate > params["maniac_raise_rate"]:
        return "maniac", rates
    if call_rate > params["station_call_rate"] and fold_rate < params["station_fold_rate_max"]:
        return "station", rates
    if fold_rate > params["folder_fold_rate"] and raise_rate < params["folder_raise_rate_max"]:
        return "folder", rates
    if (
        raise_rate < params["strong_raise_rate_max"]
        and call_rate < params["strong_call_rate_max"]
        and fold_rate > params["strong_fold_rate_min"]
    ):
        return "strong", rates
    return "unknown", rates


def _target_profile(state: dict, params: dict[str, float]) -> dict:
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
            kind, rates = _classify(stats_by_bot.get(bot_id), params)
            active_profiles.append((kind, rates, bot_id))
        priority = {"station": 0, "maniac": 1, "folder": 2, "strong": 3, "unknown": 4}
        active_profiles.sort(key=lambda row: priority.get(row[0], 9))
        if active_profiles:
            kind, rates, target_bot = active_profiles[0]
        else:
            kind, rates = _classify(None, params)
            target_bot = "unknown"
    else:
        kind, rates = _classify(stats_by_bot.get(target_bot), params)

    fold_pressure = _clip(
        rates["fold_rate"]
        + params["fold_pressure_folder_bonus"] * (kind == "folder")
        - params["fold_pressure_station_penalty"] * (kind == "station")
    )
    station_score = _clip(rates["call_rate"] - rates["fold_rate"] + params["station_score_bonus"] * (kind == "station"))
    maniac_score = _clip(
        rates["raise_rate"]
        + params["maniac_allin_weight"] * rates["all_in_rate"]
        + params["maniac_score_bonus"] * (kind == "maniac")
    )
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


def _equity_proxy(features: np.ndarray, state: dict, params: dict[str, float]) -> float:
    strength = float(features[17])
    active = max(2, _active_players(state))
    if state.get("street") == "preflop":
        return _clip(
            strength
            + params["equity_preflop_position_bonus"] * features[11]
            - params["equity_active_penalty"] * max(0, active - 2)
        )
    made = float(features[22])
    draw = float(max(features[23], features[24]))
    wet = float(features[18])
    paired = float(features[19])
    broadway = float(features[21])
    equity = (
        params["equity_postflop_base"]
        + params["equity_strength_weight"] * strength
        + params["equity_made_weight"] * made
        + params["equity_draw_weight"] * draw
        + params["equity_broadway_weight"] * broadway
    )
    equity -= (
        params["equity_wet_penalty"] * wet
        + params["equity_paired_penalty"] * paired
        + params["equity_active_penalty"] * max(0, active - 2)
    )
    return _clip(equity, 0.03, 0.97)


def build_context(state: dict, params: dict[str, float] | None = None) -> dict:
    params = coerce_params(params)
    features = extract_features(state)
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    stack = max(0, int(state.get("your_stack", 0) or 0))
    invested = max(0, int(state.get("your_bet_this_street", 0) or 0))
    total_stack = max(1, stack + invested)
    active = max(1, _active_players(state))
    target = _target_profile(state, params)
    equity = _equity_proxy(features, state, params)
    draw = float(max(features[23], features[24]))
    made = float(features[22])
    return {
        "features": features,
        "params": params,
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
    params = ctx["params"]
    return (
        params["value_threshold_base"]
        + (params["value_threshold_multiway_bonus"] if ctx["multiway"] else 0.0)
        - params["value_threshold_station_discount"] * (ctx["target_type"] == "station")
    )


def _call_margin(ctx: dict) -> float:
    params = ctx["params"]
    margin = params["call_margin_base"] + (params["call_margin_multiway_bonus"] if ctx["multiway"] else 0.0)
    margin += params["call_margin_river_bonus"] if ctx["street"] == "river" else 0.0
    margin += params["call_margin_strong_bonus"] if ctx["target_type"] in ("strong", "folder") else 0.0
    margin -= params["call_margin_maniac_discount"] if ctx["target_type"] == "maniac" else 0.0
    return max(params["call_margin_min"], margin)


def _default_tag(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    equity = ctx["equity"]
    if ctx["can_check"]:
        if equity >= _value_threshold(ctx):
            return _cand(
                state,
                ctx,
                "default_tag",
                raise_to_fraction(state, params["default_value_fraction"]),
                "value",
                params["default_value_conf_base"] + equity * params["default_value_conf_equity_weight"],
                params["default_value_hint"],
            )
        if (
            ctx["draw"]
            and equity >= params["default_semibluff_equity"]
            and ctx["fold_pressure"] > params["default_semibluff_fold_pressure"]
            and not ctx["multiway"]
        ):
            return _cand(
                state,
                ctx,
                "default_tag",
                raise_to_fraction(state, params["default_semibluff_fraction"]),
                "semi_bluff",
                0.55,
                0.05,
            )
        return _cand(state, ctx, "default_tag", _passive_action(state), "check_call", 0.58, 0.0)
    if equity >= ctx["pot_odds"] + _call_margin(ctx):
        if equity > params["default_raise_value_equity"] and ctx["owed"] < ctx["pot"] * params["default_raise_owed_pot_max"]:
            return _cand(state, ctx, "default_tag", raise_to_fraction(state, params["default_raise_fraction"]), "value", 0.75, 0.12)
        return _cand(state, ctx, "default_tag", _call_action(state), "check_call", 0.62, 0.04)
    return _cand(state, ctx, "default_tag", _passive_action(state), "fold", 0.66, -0.02)


def _pot_control(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    if ctx["can_check"]:
        return _cand(state, ctx, "pot_control", _passive_action(state), "pot_control", 0.68, 0.04)
    threshold = ctx["pot_odds"] + max(params["pot_control_call_margin_floor"], _call_margin(ctx) - params["pot_control_call_margin_discount"])
    if ctx["equity"] >= threshold and ctx["owed"] <= ctx["pot"] * params["pot_control_owed_pot_max"]:
        return _cand(state, ctx, "pot_control", _call_action(state), "pot_control", 0.58, 0.03)
    return _cand(state, ctx, "pot_control", _passive_action(state), "fold", 0.64, 0.0)


def _value_station(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    equity = ctx["equity"]
    made_or_strong = ctx["made"] or equity > params["value_station_strong_equity"] or ctx["strength"] > params["value_station_strong_strength"]
    if ctx["can_check"]:
        if made_or_strong and equity > params["value_station_bet_equity"]:
            frac = params["value_station_big_fraction"] if equity > params["value_station_high_equity"] else params["value_station_normal_fraction"]
            return _cand(state, ctx, "value_station", raise_to_fraction(state, frac), "value", 0.68 + 0.20 * ctx["station_score"], 0.18)
        return _cand(state, ctx, "value_station", _passive_action(state), "check_call", 0.54, -0.01)
    if equity >= ctx["pot_odds"] + params["value_station_call_margin"]:
        if equity > params["value_station_raise_equity"] and ctx["owed"] < ctx["pot"] * params["value_station_raise_owed_pot_max"]:
            return _cand(state, ctx, "value_station", raise_to_fraction(state, params["value_station_big_fraction"]), "value", 0.78, 0.16)
        return _cand(state, ctx, "value_station", _call_action(state), "check_call", 0.62, 0.04)
    return _cand(state, ctx, "value_station", _passive_action(state), "fold", 0.60, -0.04)


def _anti_maniac(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    equity = ctx["equity"]
    if ctx["can_check"]:
        if equity > params["anti_maniac_value_equity"] and ctx["spr"] <= params["anti_maniac_value_spr_max"]:
            return _cand(state, ctx, "anti_maniac", raise_to_fraction(state, params["anti_maniac_value_fraction"]), "value", 0.76, 0.13)
        return _cand(state, ctx, "anti_maniac", _passive_action(state), "trap", 0.62 + 0.18 * ctx["maniac_score"], 0.07)
    if equity >= ctx["pot_odds"] + max(params["call_margin_min"], _call_margin(ctx) - params["anti_maniac_call_margin_discount"]):
        if equity > params["anti_maniac_raise_equity"] and ctx["owed"] < ctx["pot"] * params["anti_maniac_raise_owed_pot_max"]:
            return _cand(state, ctx, "anti_maniac", raise_to_fraction(state, params["value_station_big_fraction"]), "value", 0.73, 0.11)
        return _cand(state, ctx, "anti_maniac", _call_action(state), "trap", 0.67, 0.08)
    return _cand(state, ctx, "anti_maniac", _passive_action(state), "fold", 0.60, -0.03)


def _pressure_folder(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    if ctx["can_check"] and not ctx["multiway"]:
        board_story = (
            params["pressure_dry_weight"] * ctx["dry"]
            + params["pressure_broadway_weight"] * ctx["broadway"]
            + params["pressure_position_weight"] * ctx["position"]
        )
        if ctx["equity"] > params["pressure_value_equity"]:
            return _cand(state, ctx, "pressure_folder", raise_to_fraction(state, params["pressure_value_fraction"]), "value", 0.64, 0.10)
        if ctx["fold_pressure"] + board_story > params["pressure_bluff_threshold"] and ctx["equity"] > params["pressure_bluff_equity"]:
            return _cand(state, ctx, "pressure_folder", raise_to_fraction(state, params["pressure_bluff_fraction"]), "bluff", 0.61, 0.12)
    if not ctx["can_check"] and ctx["equity"] >= ctx["pot_odds"] + _call_margin(ctx):
        return _cand(state, ctx, "pressure_folder", _call_action(state), "check_call", 0.54, 0.01)
    return _cand(state, ctx, "pressure_folder", _passive_action(state), "fold" if not ctx["can_check"] else "check_call", 0.55, 0.0)


def _pot_odds_breaker(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    if ctx["can_check"]:
        if ctx["equity"] > params["pot_odds_value_equity"] or (ctx["made"] and ctx["equity"] > params["pot_odds_value_made_equity"]):
            return _cand(state, ctx, "pot_odds_breaker", raise_to_fraction(state, params["pot_odds_value_fraction"]), "value", 0.64, 0.13)
        if (
            ctx["fold_pressure"] > params["pot_odds_bluff_fold_pressure"]
            and ctx["target_type"] != "station"
            and not ctx["multiway"]
            and ctx["equity"] > params["pot_odds_bluff_equity"]
        ):
            return _cand(state, ctx, "pot_odds_breaker", raise_to_fraction(state, params["pot_odds_bluff_fraction"]), "bluff", 0.59, 0.11)
    if not ctx["can_check"] and ctx["equity"] >= ctx["pot_odds"] + params["pot_odds_call_margin"]:
        return _cand(state, ctx, "pot_odds_breaker", _call_action(state), "check_call", 0.55, 0.02)
    return _cand(state, ctx, "pot_odds_breaker", _passive_action(state), "pot_control", 0.53, 0.0)


def _spr_commit(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    commit = (
        ctx["spr"] <= params["spr_commit_spr_max"]
        and (
            ctx["equity"] > params["spr_commit_equity"]
            or ctx["strength"] > params["spr_commit_strength"]
            or (ctx["draw"] and ctx["equity"] > params["spr_commit_draw_equity"])
        )
    )
    if commit:
        if ctx["stack"] <= ctx["pot"] * params["spr_jam_stack_pot_max"] or ctx["equity"] > params["spr_jam_equity"]:
            return _cand(state, ctx, "spr_commit", {"action": "all_in"}, "commit", 0.72, 0.17)
        return _cand(state, ctx, "spr_commit", raise_to_fraction(state, params["spr_raise_fraction"]), "commit", 0.66, 0.11)
    if (
        not ctx["can_check"]
        and ctx["equity"] >= ctx["pot_odds"] + params["spr_call_margin"]
        and ctx["owed"] < ctx["stack"] * params["spr_call_owed_stack_max"]
    ):
        return _cand(state, ctx, "spr_commit", _call_action(state), "check_call", 0.56, 0.02)
    return _cand(state, ctx, "spr_commit", _passive_action(state), "pot_control", 0.50, -0.02)


def _short_stack_pushfold(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    short = ctx["total_stack"] <= params["short_stack_bb"] * BIG_BLIND
    if short and ctx["street"] == "preflop":
        threshold = (
            params["short_stack_push_base"]
            - params["short_stack_position_discount"] * ctx["position"]
            + (params["short_stack_facing_bonus"] if not ctx["can_check"] else 0.0)
        )
        if ctx["strength"] >= threshold:
            return _cand(state, ctx, "short_stack_pushfold", {"action": "all_in"}, "commit", 0.74, 0.19)
        return _cand(state, ctx, "short_stack_pushfold", _passive_action(state), "fold", 0.68, 0.02)
    if short and ctx["equity"] > params["short_stack_postflop_equity"]:
        return _cand(state, ctx, "short_stack_pushfold", {"action": "all_in"}, "commit", 0.70, 0.12)
    return _cand(state, ctx, "short_stack_pushfold", _default_tag(state, ctx).action, "check_call", 0.47, -0.03)


def _range_advantage_cbet(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    good_board = bool(ctx["broadway"] or (ctx["paired"] and ctx["dry"]))
    if ctx["can_check"] and not ctx["multiway"] and ctx["street"] in ("flop", "turn"):
        if ctx["hero_was_preflop_aggressor"] and good_board:
            frac = params["cbet_dry_fraction"] if ctx["dry"] else params["cbet_wet_fraction"]
            intent = "value" if ctx["equity"] > params["cbet_value_equity"] else "bluff"
            conf = (
                params["cbet_conf_base"]
                + params["cbet_broadway_conf_weight"] * ctx["broadway"]
                + params["cbet_fold_conf_weight"] * ctx["fold_pressure"]
            )
            return _cand(state, ctx, "range_advantage_cbet", raise_to_fraction(state, frac), intent, conf, 0.10)
    return _cand(state, ctx, "range_advantage_cbet", _passive_action(state), "check_call", 0.50, 0.0)


def _blocker_bluff(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    draw_or_blocker = bool(ctx["draw"] or (ctx["monotone"] and ctx["strength"] > params["blocker_strength_threshold"]))
    if (
        ctx["can_check"]
        and not ctx["multiway"]
        and draw_or_blocker
        and ctx["target_type"] in ("folder", "unknown", "strong")
        and ctx["fold_pressure"] > params["blocker_fold_pressure"]
        and params["blocker_equity_min"] <= ctx["equity"] <= params["blocker_equity_max"]
    ):
        return _cand(state, ctx, "blocker_bluff", raise_to_fraction(state, params["blocker_fraction"]), "semi_bluff", 0.57, 0.10)
    if not ctx["can_check"] and ctx["draw"] and ctx["equity"] >= ctx["pot_odds"] + params["blocker_call_margin"]:
        return _cand(state, ctx, "blocker_bluff", _call_action(state), "check_call", 0.51, 0.01)
    return _cand(state, ctx, "blocker_bluff", _passive_action(state), "check_call", 0.48, -0.01)


def _showdown_value(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    if ctx["can_check"]:
        if ctx["equity"] > params["showdown_value_equity"] and ctx["target_type"] != "strong":
            return _cand(state, ctx, "showdown_value", raise_to_fraction(state, params["showdown_value_fraction"]), "value", 0.60, 0.06)
        return _cand(state, ctx, "showdown_value", _passive_action(state), "pot_control", 0.66, 0.05)
    if ctx["equity"] >= ctx["pot_odds"] + params["showdown_call_margin"] and ctx["owed"] < ctx["pot"] * params["showdown_owed_pot_max"]:
        return _cand(state, ctx, "showdown_value", _call_action(state), "check_call", 0.63, 0.05)
    return _cand(state, ctx, "showdown_value", _passive_action(state), "fold", 0.61, 0.01)


def _strong_unknown_avoidance(state: dict, ctx: dict) -> Candidate:
    params = ctx["params"]
    if ctx["can_check"]:
        if ctx["equity"] > params["strong_unknown_value_equity"] and ctx["target_type"] != "strong":
            return _cand(
                state,
                ctx,
                "strong_unknown_avoidance",
                raise_to_fraction(state, params["strong_unknown_value_fraction"]),
                "value",
                0.56,
                0.04,
            )
        return _cand(state, ctx, "strong_unknown_avoidance", _passive_action(state), "pot_control", 0.67, 0.06)
    if ctx["equity"] >= ctx["pot_odds"] + params["strong_unknown_call_margin"] and ctx["owed"] < ctx["stack"] * params["strong_unknown_owed_stack_max"]:
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


def generate_candidates(state: dict, params: dict[str, float] | None = None) -> list[Candidate]:
    params = coerce_params(params)
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
    ctx = build_context(state, params)
    candidates = []
    for arm in ARM_NAMES:
        try:
            candidates.append(ARM_FUNCTIONS[arm](state, ctx))
        except Exception:
            candidates.append(_cand(state, ctx, arm, _passive_action(state), "pot_control", 0.35, -0.10))
    return candidates


def default_candidate_score(candidate: Candidate, params: dict[str, float] | None = None) -> float:
    params = coerce_params(params)
    intent_bonus = {
        "value": params["score_value_bonus"],
        "semi_bluff": params["score_semibluff_bonus"],
        "trap": params["score_trap_bonus"],
        "pot_control": params["score_pot_control_bonus"],
        "commit": params["score_commit_bonus"],
        "bluff": params["score_bluff_bonus"],
        "fold": 0.0,
        "check_call": params["score_check_call_bonus"],
    }.get(candidate.intent, 0.0)
    action = candidate.action.get("action")
    legality_bonus = params["score_legal_bonus"] if action in ("check", "call", "raise", "all_in", "fold") else params["score_illegal_penalty"]
    return (
        params["score_confidence_weight"] * candidate.confidence
        + params["score_equity_weight"] * candidate.equity_estimate
        - params["score_risk_weight"] * candidate.risk
        + candidate.score_hint
        + intent_bonus
        + legality_bonus
    )
