"""Download Fullhouse portal match and hand history via Supabase REST.

The portal frontend exposes a Supabase browser client. The schema in this repo
marks matches, match_bots, hands, hand_winners, and leaderboard as public-read,
so the anon key should be enough. If production RLS differs, pass a logged-in
access token with --jwt or FULLHOUSE_SUPABASE_JWT.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


SUPABASE_URL = "https://zqarejzswyaxsieulhps.supabase.co"
ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpxYXJlanpzd3lheHNpZXVsaHBzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUzODczMjMsImV4cCI6MjA5MDk2MzMyM30."
    "sGkrhovXW7l3TClRM7rjkcXhITNFDGjILiDB99uL5rc"
)

TABLES = {
    "tournaments": {
        "select": "id,name,phase,current_round,total_rounds,n_finalists,starts_at,created_at",
        "order": "starts_at.asc",
    },
    "matches": {
        "select": "id,tournament_id,round,table_index,status,n_hands,started_at,completed_at,error_message",
        "order": "round.asc,table_index.asc",
    },
    "match_bots": {
        "select": "match_id,bot_id,seat,final_stack,chip_delta",
        "order": "match_id.asc,seat.asc",
    },
    "hands": {
        "select": "id,match_id,hand_num,street,pot,community_cards,action_log,revealed_cards,played_at",
        "order": "match_id.asc,hand_num.asc",
    },
    "hand_winners": {
        "select": "hand_id,bot_id,amount",
        "order": "hand_id.asc",
    },
    "bots": {
        "select": "id,user_id,bot_name,version,status,error_message,submitted_at",
        "order": "submitted_at.asc",
    },
    "leaderboard": {
        "select": "tournament_id,bot_id,rank,cumulative_delta,matches_played,updated_at",
        "order": "rank.asc",
    },
}


def _url(table: str, params: dict[str, str | int]) -> str:
    query = urllib.parse.urlencode(params, safe=",.*()")
    return f"{SUPABASE_URL}/rest/v1/{table}?{query}"


def _request_json(url: str, token: str, timeout: int, retries: int) -> list[dict]:
    headers = {
        "apikey": ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            if body and hasattr(exc, "add_note"):
                exc.add_note(body)
            elif body:
                print(body, flush=True)
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {url}") from last_error


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


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_ndjson(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_per_match(output: Path, tables: dict[str, list[dict]]) -> None:
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
            match_dir / f"{match['round']:02d}_{match['table_index']:03d}_{match_id}.json",
            {
                "match": match,
                "bots": bots_by_match.get(match_id, []),
                "hands": hands,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/portal_history/latest")
    parser.add_argument("--jwt", default=os.environ.get("FULLHOUSE_SUPABASE_JWT", ANON_KEY))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--table", action="append", choices=sorted(TABLES),
                        help="Table to download. Defaults to all useful replay tables.")
    args = parser.parse_args()

    output = Path(args.output)
    if args.page_size > 1000:
        print("Supabase REST caps pages at 1000 rows; using --page-size 1000", flush=True)
        args.page_size = 1000
    tables = args.table or ["tournaments", "matches", "match_bots", "hands", "hand_winners", "bots", "leaderboard"]
    downloaded: dict[str, list[dict]] = {}
    for table in tables:
        rows = fetch_table(table, args.jwt, args.page_size, args.timeout, args.retries)
        downloaded[table] = rows
        write_ndjson(output / f"{table}.ndjson", rows)
        write_json(output / f"{table}.json", rows)

    if {"matches", "match_bots", "hands", "hand_winners"} <= set(downloaded):
        write_per_match(output, downloaded)

    write_json(
        output / "summary.json",
        {table: len(rows) for table, rows in downloaded.items()},
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
