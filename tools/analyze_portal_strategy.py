"""Analyze downloaded Fullhouse portal strategy tendencies.

Input is an output directory from download_portal_targeted_history.py. The
portal action_log stores seats local to the current alive-player list, so this
tool reconstructs that list from match_bots order and hand-by-hand stack deltas.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


STARTING_STACK = 10_000
DECISION_ACTIONS = {"fold", "check", "call", "raise", "all_in"}
AGGRESSIVE_ACTIONS = {"raise", "all_in"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_ndjson(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_div(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return round(ordered[idx], 3)


def bot_name(bot_id: str, bots: dict[str, dict]) -> str:
    return str(bots.get(bot_id, {}).get("bot_name") or bot_id[:8])


def action_amount(entry: dict) -> int:
    value = entry.get("amount", 0)
    return value if isinstance(value, int) else 0


def resolve_seat(seat_to_bot: dict[int, str], seat: int) -> str | None:
    return seat_to_bot.get(seat)


def simulate_hand(
    hand: dict,
    seat_to_bot: dict[int, str],
    stacks: dict[str, int],
    stats: dict[str, Counter],
) -> dict:
    active_seats = set(seat_to_bot)
    seats_seen: set[int] = set()
    folded: set[int] = set()
    all_in: set[int] = set()
    bets = defaultdict(int)
    invested = defaultdict(int)
    current_bet = 0
    needs_to_act = set(active_seats)
    street_index = 0
    last_aggressor: int | None = None
    saw_action_by_bot = defaultdict(set)
    preflop_vpip = set()
    preflop_pfr = set()
    hand_raisers = set()
    action_rows = []

    entries = list(hand.get("action_log") or [])
    for idx, entry in enumerate(entries):
        action = str(entry.get("action", "")).lower()
        seat = entry.get("seat")
        if not isinstance(seat, int):
            continue
        bot_id = resolve_seat(seat_to_bot, seat)
        if bot_id is None:
            stats["_errors"]["unmapped_seat"] += 1
            continue
        seats_seen.add(seat)

        amount = action_amount(entry)
        if action in {"small_blind", "big_blind"}:
            paid = amount
            bets[seat] += paid
            invested[bot_id] += paid
            current_bet = max(current_bet, bets[seat])
            continue

        if action not in DECISION_ACTIONS:
            continue

        stats[bot_id]["actions"] += 1
        stats[bot_id][f"action_{action}"] += 1
        saw_action_by_bot[bot_id].add(street_index)
        needs_to_act.discard(seat)

        pot_before = sum(invested.values())
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
            needs_to_act = {
                s for s in active_seats
                if s != seat and s not in folded and s not in all_in
            }
            last_aggressor = seat
            hand_raisers.add(bot_id)
        elif action == "all_in":
            paid = max(0, amount - bets[seat])
            bets[seat] += paid
            current_bet = max(current_bet, bets[seat])
            all_in.add(seat)
            needs_to_act = {
                s for s in active_seats
                if s != seat and s not in folded and s not in all_in
            }
            last_aggressor = seat
            hand_raisers.add(bot_id)

        if paid:
            invested[bot_id] += paid
            if street_index == 0 and action in {"call", "raise", "all_in"}:
                preflop_vpip.add(bot_id)
            if street_index == 0 and action in AGGRESSIVE_ACTIONS:
                preflop_pfr.add(bot_id)

        if action in AGGRESSIVE_ACTIONS:
            stats[bot_id]["aggressive_actions"] += 1
            if street_index == 0:
                stats[bot_id]["preflop_aggressive_actions"] += 1
            if action == "all_in" and street_index == 0:
                stats[bot_id]["preflop_all_in"] += 1
            if pot_before > 0:
                stats[bot_id]["raise_ratio_count"] += 1
                stats[bot_id]["raise_ratio_sum"] += amount / pot_before

        if last_aggressor is not None and last_aggressor != seat and action in {"fold", "call"}:
            stats[bot_id]["faced_pressure"] += 1
            stats[bot_id][f"pressure_{action}"] += 1

        action_rows.append((bot_id, action, street_index))

        remaining = [s for s in active_seats if s not in folded]
        if len(remaining) <= 1:
            continue
        active = [
            s for s in remaining
            if s not in all_in
        ]
        street_done = bool(active) and not needs_to_act and all(bets[s] >= current_bet for s in active)
        if street_done and idx < len(entries) - 1:
            street_index += 1
            bets = defaultdict(int)
            current_bet = 0
            last_aggressor = None
            needs_to_act = set(active)

    for seat in seats_seen:
        stats[seat_to_bot[seat]]["hands_alive"] += 1
    for bot_id in preflop_vpip:
        stats[bot_id]["vpip_hands"] += 1
    for bot_id in preflop_pfr:
        stats[bot_id]["pfr_hands"] += 1
    for bot_id in hand_raisers:
        stats[bot_id]["hands_with_raise"] += 1
    for bot_id in hand.get("revealed_cards") or {}:
        stats[bot_id]["showdowns"] += 1

    for bot_id, amount in invested.items():
        stacks[bot_id] = stacks.get(bot_id, 0) - amount
    for winner in hand.get("hand_winners") or []:
        bot_id = winner.get("bot_id")
        amount = winner.get("amount", 0)
        if isinstance(bot_id, str) and isinstance(amount, int):
            stacks[bot_id] = stacks.get(bot_id, 0) + amount
            stats[bot_id]["hands_won"] += 1
            stats[bot_id]["winnings"] += amount

    return {"invested": dict(invested), "action_rows": action_rows}


def analyze_match(path: Path, stats: dict[str, Counter], match_results: dict[str, Counter]) -> None:
    payload = load_json(path)
    bots = sorted(payload.get("bots") or [], key=lambda row: row.get("seat", 0))
    initial_order = [row["bot_id"] for row in bots]
    seat_to_bot = {row["seat"]: row["bot_id"] for row in bots if isinstance(row.get("seat"), int)}
    stacks = {bot_id: STARTING_STACK for bot_id in initial_order}
    for row in bots:
        bot_id = row["bot_id"]
        match_results[bot_id]["matches"] += 1
        if isinstance(row.get("chip_delta"), int):
            match_results[bot_id]["chip_delta"] += row["chip_delta"]

    for hand in sorted(payload.get("hands") or [], key=lambda row: row.get("hand_num", 0)):
        simulate_hand(hand, seat_to_bot, stacks, stats)


def build_report(directory: Path, top: int) -> dict:
    bots = {row["id"]: row for row in load_json(directory / "bots.json")}
    selected_leaderboard = load_json(directory / "selected_leaderboard.json")
    official = {row["bot_id"]: row for row in selected_leaderboard}
    stats: dict[str, Counter] = defaultdict(Counter)
    match_results: dict[str, Counter] = defaultdict(Counter)

    for path in sorted((directory / "matches").glob("*.json")):
        analyze_match(path, stats, match_results)

    rows = []
    for bot_id, counts in stats.items():
        if bot_id == "_errors":
            continue
        actions = counts["actions"]
        raises = counts["action_raise"] + counts["action_all_in"]
        pressure = counts["faced_pressure"]
        raise_ratio_count = counts["raise_ratio_count"]
        official_row = official.get(bot_id, {})
        rows.append({
            "bot_id": bot_id,
            "bot_name": bot_name(bot_id, bots),
            "official_rank": official_row.get("rank"),
            "official_delta": official_row.get("cumulative_delta"),
            "official_matches": official_row.get("matches_played"),
            "sample_matches": match_results[bot_id]["matches"],
            "sample_delta": match_results[bot_id]["chip_delta"],
            "hands_alive": counts["hands_alive"],
            "actions": actions,
            "vpip": safe_div(counts["vpip_hands"], counts["hands_alive"]),
            "pfr": safe_div(counts["pfr_hands"], counts["hands_alive"]),
            "raise_rate": safe_div(raises, actions),
            "call_rate": safe_div(counts["action_call"], actions),
            "fold_rate": safe_div(counts["action_fold"], actions),
            "check_rate": safe_div(counts["action_check"], actions),
            "all_in_rate": safe_div(counts["action_all_in"], actions),
            "preflop_all_in_per_hand": safe_div(counts["preflop_all_in"], counts["hands_alive"]),
            "pressure_fold_rate": safe_div(counts["pressure_fold"], pressure),
            "pressure_call_rate": safe_div(counts["pressure_call"], pressure),
            "showdown_rate": safe_div(counts["showdowns"], counts["hands_alive"]),
            "win_hand_rate": safe_div(counts["hands_won"], counts["hands_alive"]),
            "avg_raise_to_pot": round(counts["raise_ratio_sum"] / raise_ratio_count, 3)
            if raise_ratio_count else 0.0,
            "stack_mismatches": match_results[bot_id]["stack_mismatch"],
        })

    rows.sort(key=lambda row: (
        row["official_rank"] is None,
        row["official_rank"] if row["official_rank"] is not None else 10**9,
        -row["sample_delta"],
    ))
    return {
        "directory": str(directory),
        "bot_count": len(rows),
        "errors": dict(stats.get("_errors", {})),
        "bots": rows[:top],
    }


def print_text(report: dict) -> None:
    print(f"directory={report['directory']} bots={report['bot_count']} errors={report['errors']}")
    fields = [
        "official_rank", "bot_name", "official_delta", "sample_matches", "sample_delta",
        "hands_alive", "vpip", "pfr", "raise_rate", "call_rate", "fold_rate",
        "all_in_rate", "pressure_fold_rate", "showdown_rate", "avg_raise_to_pot",
    ]
    print("\t".join(fields))
    for row in report["bots"]:
        print("\t".join(str(row.get(field, "")) for field in fields))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = build_report(args.directory, args.top)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
