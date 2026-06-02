"""Download targeted Fullhouse portal hand histories.

This is a narrower companion to download_portal_match_history.py. It pulls the
small public metadata tables, selects bots from the official leaderboard, maps
those bots to the internal qualifier match rows, and downloads only the relevant
hands with hand_winners embedded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from download_portal_match_history import ANON_KEY, TABLES, _request_json, _url, write_json, write_ndjson


METADATA_TABLES = ("tournaments", "matches", "match_bots", "bots", "leaderboard")
HANDS_SELECT = (
    "id,match_id,hand_num,street,pot,community_cards,action_log,"
    "revealed_cards,played_at,hand_winners(bot_id,amount)"
)


def fetch_table(table: str, token: str, page_size: int, timeout: int, retries: int) -> list[dict]:
    spec = TABLES[table]
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "select": spec["select"],
            "order": spec["order"],
            "limit": page_size,
            "offset": offset,
        }
        page = _request_json(_url(table, params), token, timeout, retries)
        rows.extend(page)
        print(f"{table}: fetched {len(rows)} rows", flush=True)
        if len(page) < page_size:
            return rows
        offset += page_size


def _match_output_name(match: dict) -> str:
    return f"{match['round']:02d}_{match['table_index']:03d}_{match['id']}.json"


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

    hand_count = 0
    current_match_id: str | None = None
    current_hands: list[dict] = []

    def flush_current() -> None:
        nonlocal current_match_id, current_hands
        if current_match_id is None:
            return
        match = matches_by_id[current_match_id]
        write_json(
            match_dir / _match_output_name(match),
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
            page = _request_json(_url("hands", params), token, timeout, retries)
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


def select_leaderboard_tournament(tournaments: list[dict], name: str) -> dict:
    matches = [t for t in tournaments if t.get("name") == name]
    if matches:
        return matches[-1]
    lowered = name.lower()
    partial = [t for t in tournaments if lowered in str(t.get("name", "")).lower()]
    if partial:
        return partial[-1]
    raise SystemExit(f"no leaderboard tournament matched {name!r}")


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
            raise SystemExit(f"no bot matched name {name!r}")
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
        raise SystemExit("select at least one bot with --rank-lte/--rank-gte, --bot-id, or --bot-name")
    return selected


def write_per_match(output: Path, matches: list[dict], match_bots: list[dict], hands: list[dict]) -> None:
    bots_by_match = defaultdict(list)
    hands_by_match = defaultdict(list)
    for row in match_bots:
        bots_by_match[row["match_id"]].append(row)
    for row in hands:
        hands_by_match[row["match_id"]].append(row)

    match_dir = output / "matches"
    for match in matches:
        write_json(
            match_dir / f"{match['round']:02d}_{match['table_index']:03d}_{match['id']}.json",
            {
                "match": match,
                "bots": bots_by_match.get(match["id"], []),
                "hands": hands_by_match.get(match["id"], []),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/portal_history/targeted")
    parser.add_argument("--jwt", default=os.environ.get("FULLHOUSE_SUPABASE_JWT", ANON_KEY))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--leaderboard-name", default="Fullhouse 2026 Qualifier")
    parser.add_argument("--source-name-contains", default="Qualifier")
    parser.add_argument("--rank-gte", type=int)
    parser.add_argument("--rank-lte", type=int)
    parser.add_argument("--bot-id", action="append", default=[])
    parser.add_argument("--bot-name", action="append", default=[])
    parser.add_argument("--match-id", action="append", default=[])
    parser.add_argument("--limit-matches", type=int)
    parser.add_argument("--hand-batch-size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.page_size > 1000:
        print("Supabase REST caps pages at 1000 rows; using --page-size 1000", flush=True)
        args.page_size = 1000

    output = Path(args.output)
    metadata = {
        table: fetch_table(table, args.jwt, args.page_size, args.timeout, args.retries)
        for table in METADATA_TABLES
    }
    for table, rows in metadata.items():
        write_json(output / f"{table}.json", rows)
        write_ndjson(output / f"{table}.ndjson", rows)

    leaderboard_tournament = select_leaderboard_tournament(
        metadata["tournaments"],
        args.leaderboard_name,
    )
    source_ids = source_tournament_ids(
        metadata["tournaments"],
        leaderboard_tournament["id"],
        args.source_name_contains,
    )
    selected_bot_ids = select_bot_ids(
        metadata["leaderboard"],
        metadata["bots"],
        leaderboard_tournament["id"],
        args.rank_gte,
        args.rank_lte,
        args.bot_id,
        args.bot_name,
    )

    matches_by_id = {row["id"]: row for row in metadata["matches"]}
    selected_match_ids = set(args.match_id)
    for row in metadata["match_bots"]:
        match = matches_by_id.get(row["match_id"])
        if not match:
            continue
        if match["status"] != "complete" or match["tournament_id"] not in source_ids:
            continue
        if row["bot_id"] in selected_bot_ids:
            selected_match_ids.add(row["match_id"])

    selected_matches = [matches_by_id[mid] for mid in selected_match_ids if mid in matches_by_id]
    selected_matches.sort(key=lambda m: (m["tournament_id"], m["round"], m["table_index"], m["id"]))
    if args.limit_matches is not None:
        selected_matches = selected_matches[: args.limit_matches]
        selected_match_ids = {m["id"] for m in selected_matches}

    selected_match_bots = [
        row for row in metadata["match_bots"] if row["match_id"] in selected_match_ids
    ]
    selected_leaderboard = [
        row
        for row in metadata["leaderboard"]
        if row["tournament_id"] == leaderboard_tournament["id"] and row["bot_id"] in selected_bot_ids
    ]
    selected_bots = [row for row in metadata["bots"] if row["id"] in selected_bot_ids]

    write_json(output / "selection.json", {
        "leaderboard_tournament": leaderboard_tournament,
        "source_tournament_ids": sorted(source_ids),
        "selected_bot_ids": sorted(selected_bot_ids),
        "selected_match_ids": [m["id"] for m in selected_matches],
        "selected_bot_count": len(selected_bot_ids),
        "selected_match_count": len(selected_matches),
    })
    write_json(output / "selected_bots.json", selected_bots)
    write_json(output / "selected_leaderboard.json", selected_leaderboard)
    write_json(output / "selected_matches.json", selected_matches)
    write_json(output / "selected_match_bots.json", selected_match_bots)

    print(
        f"selected {len(selected_bot_ids)} bots and {len(selected_matches)} matches",
        flush=True,
    )
    if args.dry_run:
        print(f"wrote {output}")
        return 0

    match_bots_by_match = defaultdict(list)
    for row in selected_match_bots:
        match_bots_by_match[row["match_id"]].append(row)

    hand_count = fetch_hands_to_disk(
        [m["id"] for m in selected_matches],
        matches_by_id,
        match_bots_by_match,
        output,
        args.jwt,
        args.page_size,
        args.timeout,
        args.retries,
        args.hand_batch_size,
    )
    write_json(output / "summary.json", {
        "selected_bot_count": len(selected_bot_ids),
        "selected_match_count": len(selected_matches),
        "hands": hand_count,
    })
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
