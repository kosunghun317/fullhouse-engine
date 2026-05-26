"""Summarize exported Fullhouse hand histories.

The Day 1 export schema may differ from local development structures, so this
tool intentionally accepts nested JSON or newline-delimited JSON and extracts
only stable poker concepts: actions, pressure responses, winners, and showdowns.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ACTION_KEYS = ("action_log", "match_action_log", "events", "actions")
SHOWDOWN_KEYS = ("showdown", "showdowns", "revealed_cards", "hole_cards")
WINNER_KEYS = ("winner", "winners", "pot_awards", "awards")


def _load_json(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def _looks_like_hand(node):
    if not isinstance(node, dict):
        return False
    for key in ACTION_KEYS + SHOWDOWN_KEYS + WINNER_KEYS:
        if key in node:
            return True
    return False


def _iter_hands(node):
    if isinstance(node, dict):
        if _looks_like_hand(node):
            yield node
            return
        for value in node.values():
            yield from _iter_hands(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_hands(item)


def _iter_action_entries(value):
    if isinstance(value, list):
        for item in value:
            yield from _iter_action_entries(item)
    elif isinstance(value, dict):
        if "action" in value:
            yield value
        for key in ("events", "actions"):
            if key in value:
                yield from _iter_action_entries(value[key])


def _actions_from_hand(hand):
    actions = []
    for key in ACTION_KEYS:
        if key in hand:
            actions.extend(_iter_action_entries(hand[key]))
    streets = hand.get("streets")
    if isinstance(streets, dict):
        for value in streets.values():
            actions.extend(_iter_action_entries(value))
    return actions


def _player_id(entry):
    for key in ("bot_id", "player_id", "player", "name", "seat"):
        if key in entry and entry[key] is not None:
            return str(entry[key])
    return "unknown"


def _amount(entry):
    for key in ("amount", "total_bet", "bet", "raise_to", "chips"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _extract_winners(hand):
    winners = []
    for key in WINNER_KEYS:
        value = hand.get(key)
        if isinstance(value, str):
            winners.append(value)
        elif isinstance(value, dict):
            winners.append(_player_id(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    winners.append(item)
                elif isinstance(item, dict):
                    winners.append(_player_id(item))
    return winners


def _collect_strengths(node, counter):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("hand_type", "hand_rank", "strength") and isinstance(value, str):
                counter[value] += 1
            else:
                _collect_strengths(value, counter)
    elif isinstance(node, list):
        for item in node:
            _collect_strengths(item, counter)


def _rate(count, total):
    return round(count / total, 4) if total else 0.0


def summarize(paths):
    action_counts = defaultdict(Counter)
    raise_amounts = defaultdict(list)
    pressure = defaultdict(Counter)
    winner_counts = Counter()
    hand_strengths = Counter()
    hands = 0
    showdown_hands = 0

    for path in paths:
        for hand in _iter_hands(_load_json(path)):
            hands += 1
            if any(key in hand for key in SHOWDOWN_KEYS):
                showdown_hands += 1
            _collect_strengths(hand, hand_strengths)
            winner_counts.update(_extract_winners(hand))

            previous = None
            for entry in _actions_from_hand(hand):
                player = _player_id(entry)
                action = str(entry.get("action", "")).lower()
                if not action:
                    continue
                action_counts[player][action] += 1
                if action in ("raise", "all_in"):
                    raise_amounts[player].append(_amount(entry))
                if (
                    previous
                    and previous.get("action") in ("raise", "all_in")
                    and _player_id(previous) != player
                    and action in ("fold", "call")
                ):
                    pressure[player]["events"] += 1
                    pressure[player][action] += 1
                previous = {"action": action, "player": player}

    players = {}
    for player, counts in sorted(action_counts.items()):
        actions = sum(counts.values())
        pressure_events = pressure[player]["events"]
        raises = counts["raise"] + counts["all_in"]
        players[player] = {
            "actions": actions,
            "raise_rate": _rate(raises, actions),
            "call_rate": _rate(counts["call"], actions),
            "fold_rate": _rate(counts["fold"], actions),
            "all_in_rate": _rate(counts["all_in"], actions),
            "pressure_events": pressure_events,
            "pressure_fold_rate": _rate(pressure[player]["fold"], pressure_events),
            "pressure_call_rate": _rate(pressure[player]["call"], pressure_events),
            "avg_raise_amount": round(sum(raise_amounts[player]) / len(raise_amounts[player]), 2)
            if raise_amounts[player] else 0,
            "actions_by_type": dict(sorted(counts.items())),
        }

    return {
        "files": [str(path) for path in paths],
        "hands": hands,
        "showdown_hands": showdown_hands,
        "winner_counts": dict(sorted(winner_counts.items())),
        "hand_strengths": dict(sorted(hand_strengths.items())),
        "players": players,
    }


def _print_text(report):
    print(f"hands={report['hands']} showdown_hands={report['showdown_hands']}")
    print("players:")
    for player, stats in report["players"].items():
        print(
            f"  {player}: actions={stats['actions']} raise={stats['raise_rate']} "
            f"call={stats['call_rate']} fold={stats['fold_rate']} "
            f"pressure_fold={stats['pressure_fold_rate']} "
            f"avg_raise={stats['avg_raise_amount']}"
        )
    if report["winner_counts"]:
        print("winners:", report["winner_counts"])
    if report["hand_strengths"]:
        print("hand_strengths:", report["hand_strengths"])


def main():
    parser = argparse.ArgumentParser(description="Summarize Fullhouse hand-history JSON")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = summarize([Path(path) for path in args.paths])
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)


if __name__ == "__main__":
    main()
