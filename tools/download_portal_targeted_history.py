"""Download targeted or full public Fullhouse portal hand histories."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from portal.client import normalize_page_size
from portal.download import (
    fetch_hands_to_disk,
    fetch_metadata,
    select_bot_ids,
    select_leaderboard_tournament,
    select_source_matches,
    selected_rows,
    source_tournament_ids,
    write_json,
    write_ndjson,
    write_table_outputs,
)
from portal.schema import (
    ANON_KEY,
    DEFAULT_LEADERBOARD_NAME,
    DEFAULT_SOURCE_NAME_CONTAINS,
)


def _selected_bot_ids_from_args(args, metadata, leaderboard_tournament_id: str) -> set[str]:
    has_bot_selection = bool(args.bot_id or args.bot_name or args.rank_gte is not None or args.rank_lte is not None)
    if args.all_complete_source_matches or (args.match_id and not has_bot_selection):
        return set()
    return select_bot_ids(
        metadata["leaderboard"],
        metadata["bots"],
        leaderboard_tournament_id,
        args.rank_gte,
        args.rank_lte,
        args.bot_id,
        args.bot_name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/portal_history/targeted")
    parser.add_argument("--jwt", default=os.environ.get("FULLHOUSE_SUPABASE_JWT", ANON_KEY))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--leaderboard-name", default=DEFAULT_LEADERBOARD_NAME)
    parser.add_argument("--source-name-contains", default=DEFAULT_SOURCE_NAME_CONTAINS)
    parser.add_argument("--rank-gte", type=int)
    parser.add_argument("--rank-lte", type=int)
    parser.add_argument("--bot-id", action="append", default=[])
    parser.add_argument("--bot-name", action="append", default=[])
    parser.add_argument("--match-id", action="append", default=[])
    parser.add_argument(
        "--all-complete-source-matches",
        action="store_true",
        help="Download every complete match from source tournaments matching --source-name-contains.",
    )
    parser.add_argument("--limit-matches", type=int)
    parser.add_argument("--hand-batch-size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.page_size > 1000:
        print("Supabase REST caps pages at 1000 rows; using --page-size 1000", flush=True)
    page_size = normalize_page_size(args.page_size)
    output = Path(args.output)

    try:
        metadata = fetch_metadata(args.jwt, page_size, args.timeout, args.retries)
        write_table_outputs(output, metadata)

        leaderboard_tournament = select_leaderboard_tournament(
            metadata["tournaments"],
            args.leaderboard_name,
        )
        source_ids = source_tournament_ids(
            metadata["tournaments"],
            leaderboard_tournament["id"],
            args.source_name_contains,
        )
        selected_bot_ids = _selected_bot_ids_from_args(args, metadata, leaderboard_tournament["id"])
        selected_matches = select_source_matches(
            metadata["matches"],
            metadata["match_bots"],
            source_ids,
            selected_bot_ids,
            args.match_id,
            all_complete_source_matches=args.all_complete_source_matches,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.limit_matches is not None:
        selected_matches = selected_matches[: args.limit_matches]
    selected_match_ids = {m["id"] for m in selected_matches}
    selected_match_bots = [
        row for row in metadata["match_bots"] if row["match_id"] in selected_match_ids
    ]
    if args.all_complete_source_matches or not selected_bot_ids:
        selected_bot_ids = {row["bot_id"] for row in selected_match_bots}

    rows = selected_rows(
        metadata,
        leaderboard_tournament["id"],
        selected_bot_ids,
        selected_match_ids,
    )
    write_json(output / "selection.json", {
        "leaderboard_tournament": leaderboard_tournament,
        "source_tournament_ids": sorted(source_ids),
        "selected_bot_ids": sorted(selected_bot_ids),
        "selected_match_ids": [m["id"] for m in selected_matches],
        "selected_bot_count": len(selected_bot_ids),
        "selected_match_count": len(selected_matches),
        "all_complete_source_matches": args.all_complete_source_matches,
    })
    write_json(output / "selected_bots.json", rows["bots"])
    write_json(output / "selected_leaderboard.json", rows["leaderboard"])
    write_json(output / "selected_matches.json", selected_matches)
    write_json(output / "selected_match_bots.json", rows["match_bots"])
    write_ndjson(output / "selected_matches.ndjson", selected_matches)

    print(
        f"selected {len(selected_bot_ids)} bots and {len(selected_matches)} matches",
        flush=True,
    )
    if args.dry_run:
        print(f"wrote {output}")
        return 0

    matches_by_id = {row["id"]: row for row in metadata["matches"]}
    match_bots_by_match = defaultdict(list)
    for row in rows["match_bots"]:
        match_bots_by_match[row["match_id"]].append(row)

    hand_count = fetch_hands_to_disk(
        [m["id"] for m in selected_matches],
        matches_by_id,
        match_bots_by_match,
        output,
        args.jwt,
        page_size,
        args.timeout,
        args.retries,
        args.hand_batch_size,
    )
    write_json(output / "summary.json", {
        "selected_bot_count": len(selected_bot_ids),
        "selected_match_count": len(selected_matches),
        "hands": hand_count,
        "all_complete_source_matches": args.all_complete_source_matches,
    })
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
