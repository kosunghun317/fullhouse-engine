"""Portal replay download, selection, and file writers."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from .client import build_url, fetch_table, normalize_page_size, request_json
from .schema import HANDS_SELECT, METADATA_TABLES


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_ndjson(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def match_output_name(match: dict) -> str:
    return f"{match['round']:02d}_{match['table_index']:03d}_{match['id']}.json"


def fetch_metadata(token: str, page_size: int, timeout: int, retries: int) -> dict[str, list[dict]]:
    return {
        table: fetch_table(table, token, page_size, timeout, retries)
        for table in METADATA_TABLES
    }


def write_table_outputs(output: Path, tables: dict[str, list[dict]]) -> None:
    for table, rows in tables.items():
        write_json(output / f"{table}.json", rows)
        write_ndjson(output / f"{table}.ndjson", rows)


def write_per_match_from_tables(output: Path, tables: dict[str, list[dict]]) -> None:
    bots_by_match = defaultdict(list)
    hands_by_match = defaultdict(list)
    winners_by_hand = defaultdict(list)
    for row in tables.get("match_bots", []):
        bots_by_match[row["match_id"]].append(row)
    for row in tables.get("hands", []):
        hands_by_match[row["match_id"]].append(row)
    for row in tables.get("hand_winners", []):
        winners_by_hand[row["hand_id"]].append(row)

    match_dir = output / "matches"
    for match in tables.get("matches", []):
        match_id = match["id"]
        hands = []
        for hand in hands_by_match.get(match_id, []):
            enriched = dict(hand)
            enriched["winners"] = winners_by_hand.get(hand["id"], [])
            hands.append(enriched)
        write_json(
            match_dir / match_output_name(match),
            {
                "match": match,
                "bots": bots_by_match.get(match_id, []),
                "hands": hands,
            },
        )


def select_leaderboard_tournament(tournaments: list[dict], name: str) -> dict:
    matches = [t for t in tournaments if t.get("name") == name]
    if matches:
        return matches[-1]
    lowered = name.lower()
    partial = [t for t in tournaments if lowered in str(t.get("name", "")).lower()]
    if partial:
        return partial[-1]
    raise ValueError(f"no leaderboard tournament matched {name!r}")


def source_tournament_ids(
    tournaments: list[dict],
    leaderboard_tournament_id: str,
    contains: str,
) -> set[str]:
    needle = contains.lower()
    return {
        t["id"]
        for t in tournaments
        if t["id"] != leaderboard_tournament_id
        and needle in str(t.get("name", "")).lower()
    }


def select_bot_ids(
    leaderboard: list[dict],
    bots: list[dict],
    leaderboard_tournament_id: str,
    rank_gte: int | None,
    rank_lte: int | None,
    bot_ids: list[str],
    bot_names: list[str],
) -> set[str]:
    selected = set(bot_ids)
    bot_name_lookup = {str(b.get("bot_name", "")).lower(): b["id"] for b in bots}
    for name in bot_names:
        bot_id = bot_name_lookup.get(name.lower())
        if not bot_id:
            raise ValueError(f"no bot matched name {name!r}")
        selected.add(bot_id)
    if rank_gte is not None or rank_lte is not None:
        lo = rank_gte if rank_gte is not None else 1
        hi = rank_lte if rank_lte is not None else 10**9
        for row in leaderboard:
            if row["tournament_id"] != leaderboard_tournament_id:
                continue
            if lo <= row["rank"] <= hi:
                selected.add(row["bot_id"])
    if not selected:
        raise ValueError("select at least one bot with rank range, bot id, bot name, or all-match mode")
    return selected


def select_source_matches(
    matches: list[dict],
    match_bots: list[dict],
    source_ids: set[str],
    selected_bot_ids: set[str],
    explicit_match_ids: list[str],
    all_complete_source_matches: bool = False,
) -> list[dict]:
    matches_by_id = {row["id"]: row for row in matches}
    selected_match_ids = set(explicit_match_ids)
    if all_complete_source_matches:
        selected_match_ids.update(
            row["id"]
            for row in matches
            if row.get("status") == "complete" and row.get("tournament_id") in source_ids
        )
    else:
        for row in match_bots:
            match = matches_by_id.get(row["match_id"])
            if not match:
                continue
            if match.get("status") != "complete" or match.get("tournament_id") not in source_ids:
                continue
            if row["bot_id"] in selected_bot_ids:
                selected_match_ids.add(row["match_id"])

    selected_matches = [matches_by_id[mid] for mid in selected_match_ids if mid in matches_by_id]
    selected_matches.sort(key=lambda m: (m["tournament_id"], m["round"], m["table_index"], m["id"]))
    return selected_matches


def selected_rows(
    metadata: dict[str, list[dict]],
    leaderboard_tournament_id: str,
    selected_bot_ids: set[str],
    selected_match_ids: set[str],
) -> dict[str, list[dict]]:
    return {
        "match_bots": [
            row for row in metadata["match_bots"] if row["match_id"] in selected_match_ids
        ],
        "leaderboard": [
            row
            for row in metadata["leaderboard"]
            if row["tournament_id"] == leaderboard_tournament_id and row["bot_id"] in selected_bot_ids
        ],
        "bots": [row for row in metadata["bots"] if row["id"] in selected_bot_ids],
    }


def fetch_hands_to_disk(
    match_ids: list[str],
    matches_by_id: dict[str, dict],
    match_bots_by_match: dict[str, list[dict]],
    output: Path,
    token: str,
    page_size: int,
    timeout: int,
    retries: int,
    batch_size: int,
) -> int:
    hands_path = output / "hands.ndjson"
    hands_path.parent.mkdir(parents=True, exist_ok=True)
    match_dir = output / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)

    page_size = normalize_page_size(page_size)
    hand_count = 0
    current_match_id: str | None = None
    current_hands: list[dict] = []

    def flush_current() -> None:
        nonlocal current_match_id, current_hands
        if current_match_id is None:
            return
        match = matches_by_id[current_match_id]
        write_json(
            match_dir / match_output_name(match),
            {
                "match": match,
                "bots": match_bots_by_match.get(current_match_id, []),
                "hands": current_hands,
            },
        )
        current_match_id = None
        current_hands = []

    with hands_path.open("w", encoding="utf-8") as handle:
        handle.write("")
    for start in range(0, len(match_ids), batch_size):
        batch = match_ids[start : start + batch_size]
        offset = 0
        while True:
            params = {
                "select": HANDS_SELECT,
                "match_id": f"in.({','.join(batch)})",
                "order": "match_id.asc,hand_num.asc",
                "limit": page_size,
                "offset": offset,
            }
            page = request_json(build_url("hands", params), token, timeout, retries)
            with hands_path.open("a", encoding="utf-8") as handle:
                for row in page:
                    match_id = row["match_id"]
                    if current_match_id is not None and match_id != current_match_id:
                        flush_current()
                    if current_match_id is None:
                        current_match_id = match_id
                    current_hands.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            hand_count += len(page)
            print(
                f"hands: batch {start // batch_size + 1} fetched {len(page)} "
                f"rows ({hand_count} total)",
                flush=True,
            )
            if len(page) < page_size:
                break
            offset += page_size
            time.sleep(0.05)
    flush_current()
    return hand_count
