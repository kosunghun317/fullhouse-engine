"""Build replay-conditioned behavior-clone mocks from portal match records."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import DECISION_ACTIONS, STARTING_STACK, load_json, load_ndjson
from .mock_generation import safe_profile_name
from .profiles import assign_profile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "bots" / "mock_competitors" / "portal_behavior_clone" / "bot.py"
ACTION_LABELS = ("fold", "check", "call", "raise", "all_in")
AGGRESSIVE_ACTIONS = {"raise", "all_in"}
STREETS = ("preflop", "flop", "turn", "river")


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _round(value: float, places: int = 4) -> float:
    return round(float(value), places)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _street_name(street_index: int) -> str:
    return STREETS[min(max(0, street_index), len(STREETS) - 1)]


def _pot_odds_bucket(pot_odds: float) -> str:
    if pot_odds <= 0.0:
        return "free"
    if pot_odds < 0.18:
        return "cheap"
    if pot_odds < 0.34:
        return "medium"
    return "expensive"


def _risk_bucket(risk: float) -> str:
    if risk <= 0.0:
        return "none"
    if risk < 0.12:
        return "low"
    if risk < 0.35:
        return "medium"
    if risk < 0.70:
        return "high"
    return "terminal"


def _opponent_bucket(active_opponents: int) -> str:
    if active_opponents <= 1:
        return "hu"
    if active_opponents <= 2:
        return "short"
    return "multi"


def _raise_bucket(raises_this_street: int) -> str:
    if raises_this_street <= 0:
        return "r0"
    if raises_this_street == 1:
        return "r1"
    return "r2plus"


def context_key(
    street: str,
    facing: bool,
    active_opponents: int,
    pot_odds: float,
    stack_risk: float,
    raises_this_street: int,
) -> str:
    mode = "facing" if facing else "free"
    return "|".join(
        [
            street,
            mode,
            _opponent_bucket(active_opponents),
            _pot_odds_bucket(pot_odds),
            _risk_bucket(stack_risk),
            _raise_bucket(raises_this_street),
        ]
    )


def context_backoff_keys(key: str) -> list[str]:
    street, mode, opponents, pot_odds, risk, raises = key.split("|")
    return [
        key,
        "|".join([street, mode, opponents, pot_odds, risk, "any"]),
        "|".join([street, mode, opponents, "any", "any", "any"]),
        "|".join([street, mode, "any", "any", "any", "any"]),
    ]


def _action_amount(entry: dict) -> int:
    value = entry.get("amount", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _normalize(counter: Counter, smoothing: float) -> dict[str, float]:
    total = sum(counter.get(action, 0) + smoothing for action in ACTION_LABELS)
    if total <= 0:
        return {action: _round(1.0 / len(ACTION_LABELS)) for action in ACTION_LABELS}
    return {
        action: _round((counter.get(action, 0) + smoothing) / total)
        for action in ACTION_LABELS
    }


def _size_summary(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": _round(sum(values) / len(values)),
        "p25": _round(_percentile(values, 0.25) or 0.0),
        "p50": _round(_percentile(values, 0.50) or 0.0),
        "p75": _round(_percentile(values, 0.75) or 0.0),
    }


def _load_rows(directory: Path, stem: str) -> list[dict]:
    json_path = directory / f"{stem}.json"
    if json_path.exists():
        rows = load_json(json_path)
        return rows if isinstance(rows, list) else []
    ndjson_path = directory / f"{stem}.ndjson"
    if ndjson_path.exists():
        return load_ndjson(ndjson_path)
    return []


def _load_rank_rows(directory: Path) -> dict[str, dict]:
    path = directory / "selected_leaderboard.json"
    if not path.exists():
        path = directory / "leaderboard.json"
    if path.exists():
        rows = load_json(path)
    else:
        rows = _load_rows(directory, "leaderboard")
    return {row["bot_id"]: row for row in rows if isinstance(row.get("bot_id"), str)}


def _load_bot_names(directory: Path) -> dict[str, str]:
    rows = _load_rows(directory, "bots")
    return {
        row["id"]: str(row.get("bot_name") or row["id"][:8])
        for row in rows
        if isinstance(row.get("id"), str)
    }


def _load_profile_rows(directory: Path) -> dict[str, str]:
    path = directory / "strategy_report.json"
    if not path.exists():
        return {}
    rows = load_json(path).get("bots", [])
    return {
        row["bot_id"]: assign_profile(row)
        for row in rows
        if isinstance(row.get("bot_id"), str)
    }


def _selected_bot_ids(
    rank_rows: dict[str, dict],
    rank_lte: int,
    rank_gte: int = 1,
) -> set[str]:
    return {
        bot_id
        for bot_id, row in rank_rows.items()
        if isinstance(row.get("rank"), int) and rank_gte <= int(row["rank"]) <= rank_lte
    }


def _group_name(
    bot_id: str,
    group_by: str,
    rank_rows: dict[str, dict],
    bot_names: dict[str, str],
    profiles: dict[str, str],
    rank_lte: int,
) -> str:
    if group_by == "bot":
        rank = rank_rows.get(bot_id, {}).get("rank", "na")
        return f"rank_{rank}_{safe_profile_name(bot_names.get(bot_id, bot_id[:8]))}"
    if group_by == "profile":
        return profiles.get(bot_id, "unknown_mixed")
    return f"top{rank_lte}"


def _new_group() -> dict:
    return {
        "actions": Counter(),
        "sizes": [],
        "cells": defaultdict(lambda: {"actions": Counter(), "sizes": []}),
        "bot_ids": set(),
    }


def _record(group: dict, key: str, action: str, raise_to_pot: float | None) -> None:
    group["actions"][action] += 1
    if raise_to_pot is not None:
        group["sizes"].append(raise_to_pot)
    for candidate_key in context_backoff_keys(key):
        cell = group["cells"][candidate_key]
        cell["actions"][action] += 1
        if raise_to_pot is not None:
            cell["sizes"].append(raise_to_pot)


def _active_opponents(active_seats: set[int], folded: set[int], seat: int) -> int:
    return sum(1 for other in active_seats if other != seat and other not in folded)


def _observe_match(
    payload: dict,
    selected: set[str],
    groups: dict[str, dict],
    rank_rows: dict[str, dict],
    bot_names: dict[str, str],
    profiles: dict[str, str],
    group_by: str,
    rank_lte: int,
) -> None:
    bots = sorted(payload.get("bots") or [], key=lambda row: row.get("seat", 0))
    seat_to_bot = {row["seat"]: row["bot_id"] for row in bots if isinstance(row.get("seat"), int)}
    bot_to_seat = {bot_id: seat for seat, bot_id in seat_to_bot.items()}
    stacks = {seat: STARTING_STACK for seat in seat_to_bot}

    for hand in sorted(payload.get("hands") or [], key=lambda row: row.get("hand_num", 0)):
        active_seats = set(seat_to_bot)
        folded: set[int] = set()
        all_in: set[int] = set()
        bets: dict[int, int] = defaultdict(int)
        invested: dict[int, int] = defaultdict(int)
        current_bet = 0
        needs_to_act = set(active_seats)
        street_index = 0
        raises_this_street = 0

        for idx, entry in enumerate(hand.get("action_log") or []):
            action = str(entry.get("action", "")).lower()
            seat = entry.get("seat")
            if not isinstance(seat, int) or seat not in seat_to_bot:
                continue
            amount = _action_amount(entry)

            if action in {"small_blind", "big_blind"}:
                paid = amount
                bets[seat] += paid
                invested[seat] += paid
                current_bet = max(current_bet, bets[seat])
                continue
            if action not in DECISION_ACTIONS:
                continue

            bot_id = seat_to_bot[seat]
            pot_before = sum(invested.values())
            owed = max(0, current_bet - bets[seat])
            stack_before = max(0, stacks.get(seat, STARTING_STACK) - invested[seat])
            facing = owed > 0
            pot_odds = _safe_div(owed, pot_before + owed)
            stack_risk = _safe_div(owed, stack_before + bets[seat])
            if bot_id in selected:
                key = context_key(
                    _street_name(street_index),
                    facing=facing,
                    active_opponents=_active_opponents(active_seats, folded, seat),
                    pot_odds=pot_odds,
                    stack_risk=stack_risk,
                    raises_this_street=raises_this_street,
                )
                raise_to_pot = None
                if action in AGGRESSIVE_ACTIONS and pot_before > 0:
                    raise_to_pot = max(0.0, amount / max(1.0, float(pot_before)))
                name = _group_name(bot_id, group_by, rank_rows, bot_names, profiles, rank_lte)
                groups[name]["bot_ids"].add(bot_id)
                _record(groups[name], key, action, raise_to_pot)

            needs_to_act.discard(seat)
            paid = 0
            if action == "fold":
                folded.add(seat)
            elif action == "call":
                paid = amount
                bets[seat] += paid
            elif action == "raise":
                paid = max(0, amount - bets[seat])
                bets[seat] += paid
                current_bet = max(current_bet, bets[seat])
                raises_this_street += 1
                needs_to_act = {
                    other for other in active_seats
                    if other != seat and other not in folded and other not in all_in
                }
            elif action == "all_in":
                paid = max(0, amount - bets[seat])
                bets[seat] += paid
                current_bet = max(current_bet, bets[seat])
                raises_this_street += 1
                all_in.add(seat)
                needs_to_act = {
                    other for other in active_seats
                    if other != seat and other not in folded and other not in all_in
                }
            if paid:
                invested[seat] += paid

            remaining = [other for other in active_seats if other not in folded]
            if len(remaining) <= 1:
                continue
            active = [other for other in remaining if other not in all_in]
            street_done = bool(active) and not needs_to_act and all(bets[other] >= current_bet for other in active)
            if street_done and idx < len(hand.get("action_log") or []) - 1:
                street_index += 1
                bets = defaultdict(int)
                current_bet = 0
                raises_this_street = 0
                needs_to_act = set(active)

        for seat, amount in invested.items():
            stacks[seat] = stacks.get(seat, STARTING_STACK) - amount
        for winner in hand.get("hand_winners") or []:
            bot_id = winner.get("bot_id")
            amount = winner.get("amount")
            if isinstance(bot_id, str) and isinstance(amount, int) and bot_id in bot_to_seat:
                stacks[bot_to_seat[bot_id]] = stacks.get(bot_to_seat[bot_id], STARTING_STACK) + amount


def _policy_from_group(
    name: str,
    group: dict,
    rank_rows: dict[str, dict],
    bot_names: dict[str, str],
    min_cell_actions: int,
    smoothing: float,
) -> dict:
    cells = {}
    for key, cell in sorted(group["cells"].items()):
        count = sum(cell["actions"].values())
        if count < min_cell_actions:
            continue
        cells[key] = {
            "count": count,
            "action_probs": _normalize(cell["actions"], smoothing),
            "raise_to_pot": _size_summary(cell["sizes"]),
        }

    bot_ids = sorted(group["bot_ids"], key=lambda bot_id: rank_rows.get(bot_id, {}).get("rank", 10**9))
    return {
        "policy_type": "portal_behavior_clone_v1",
        "group": name,
        "action_count": sum(group["actions"].values()),
        "bot_count": len(bot_ids),
        "bot_ids": bot_ids,
        "bot_names": [bot_names.get(bot_id, bot_id[:8]) for bot_id in bot_ids],
        "official_ranks": [rank_rows.get(bot_id, {}).get("rank") for bot_id in bot_ids],
        "fallback": {
            "count": sum(group["actions"].values()),
            "action_probs": _normalize(group["actions"], smoothing),
            "raise_to_pot": _size_summary(group["sizes"]),
        },
        "cell_count": len(cells),
        "cells": cells,
    }


def jitter_policy(policy: dict, seed: int, action_noise: float = 0.04, size_noise: float = 0.12) -> dict:
    """Return a deterministic policy variant for mock-pool diversification."""

    rng = random.Random(int(seed))
    cloned = json.loads(json.dumps(policy))
    for cell in [cloned.get("fallback", {}), *cloned.get("cells", {}).values()]:
        probs = cell.get("action_probs")
        if isinstance(probs, dict):
            noisy = {
                action: max(0.001, float(value) + rng.gauss(0.0, action_noise))
                for action, value in probs.items()
            }
            total = sum(noisy.values())
            cell["action_probs"] = {action: _round(value / total) for action, value in noisy.items()}
        sizing = cell.get("raise_to_pot")
        if isinstance(sizing, dict):
            for key in ("mean", "p25", "p50", "p75"):
                if isinstance(sizing.get(key), (int, float)):
                    sizing[key] = _round(max(0.05, float(sizing[key]) * rng.gauss(1.0, size_noise)))
    return cloned


def build_behavior_clone_summary(
    directory: Path,
    rank_lte: int = 64,
    rank_gte: int = 1,
    group_by: str = "profile",
    min_actions: int = 200,
    min_cell_actions: int = 20,
    smoothing: float = 0.25,
) -> dict:
    if group_by not in {"all", "profile", "bot"}:
        raise ValueError("group_by must be one of: all, profile, bot")
    rank_rows = _load_rank_rows(directory)
    bot_names = _load_bot_names(directory)
    profiles = _load_profile_rows(directory)
    selected = _selected_bot_ids(rank_rows, rank_lte=rank_lte, rank_gte=rank_gte)
    groups: dict[str, dict] = defaultdict(_new_group)

    for path in sorted((directory / "matches").glob("*.json")):
        _observe_match(
            load_json(path),
            selected,
            groups,
            rank_rows,
            bot_names,
            profiles,
            group_by,
            rank_lte,
        )

    policies = [
        _policy_from_group(name, group, rank_rows, bot_names, min_cell_actions, smoothing)
        for name, group in sorted(groups.items())
        if sum(group["actions"].values()) >= min_actions
    ]
    policies.sort(key=lambda item: (-item["action_count"], item["group"]))
    return {
        "source_directory": str(directory),
        "rank_gte": rank_gte,
        "rank_lte": rank_lte,
        "group_by": group_by,
        "min_actions": min_actions,
        "min_cell_actions": min_cell_actions,
        "smoothing": smoothing,
        "selected_bot_count": len(selected),
        "policy_count": len(policies),
        "policies": policies,
    }


def materialize_behavior_clones(
    summary: dict,
    output: Path,
    template: Path = DEFAULT_TEMPLATE,
    source: Path | None = None,
    variants_per_policy: int = 1,
    variant_seed: int = 1729,
    action_noise: float = 0.04,
    size_noise: float = 0.12,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    profiles = []
    variant_count = max(1, int(variants_per_policy))
    for policy in summary.get("policies", []):
        safe_group = safe_profile_name(str(policy["group"]))
        for variant_index in range(variant_count):
            if variant_count == 1:
                variant = {"index": 0, "name": "base", "action_noise": 0.0, "size_noise": 0.0}
                variant_policy = json.loads(json.dumps(policy))
                dir_name = safe_group
            else:
                variant = {
                    "index": variant_index + 1,
                    "name": f"v{variant_index + 1:02d}",
                    "action_noise": float(action_noise),
                    "size_noise": float(size_noise),
                }
                variant_policy = jitter_policy(
                    policy,
                    int(variant_seed) + variant_index + len(profiles) * 9973,
                    action_noise=float(action_noise),
                    size_noise=float(size_noise),
                )
                dir_name = f"{safe_group}_{variant['name']}"
            variant_policy["variant"] = variant
            bot_id = "portal_clone_" + dir_name
            bot_dir = output / dir_name
            data_dir = bot_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template, bot_dir / "bot.py")
            (data_dir / "policy.json").write_text(
                json.dumps(variant_policy, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            profiles.append({
                "profile": str(policy["group"]),
                "variant": variant,
                "bot_id": bot_id,
                "path": str(bot_dir),
                "bot_count": policy.get("bot_count"),
                "action_count": policy.get("action_count"),
                "official_ranks": policy.get("official_ranks", []),
                "clone_policy": str(data_dir / "policy.json"),
            })

    manifest = {
        "source": str(source or summary.get("source_directory")),
        "template": str(template),
        "output": str(output),
        "profile_count": len(profiles),
        "source_policy_count": summary.get("policy_count", len(summary.get("policies", []))),
        "policy_count": len(profiles),
        "variants_per_policy": variant_count,
        "variant_seed": int(variant_seed),
        "action_noise": float(action_noise),
        "size_noise": float(size_noise),
        "group_by": summary.get("group_by"),
        "rank_gte": summary.get("rank_gte"),
        "rank_lte": summary.get("rank_lte"),
        "profiles": profiles,
        "suite_bots": {row["bot_id"]: row["path"] for row in profiles},
    }
    (output / "behavior_clone_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
