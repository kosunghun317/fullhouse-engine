"""Download Fullhouse portal tables via Supabase REST.

The portal frontend exposes a Supabase browser client. The public replay tables
are readable with the anon key. If production RLS differs, pass a logged-in
access token with --jwt or FULLHOUSE_SUPABASE_JWT.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from portal.client import build_url as _url
from portal.client import fetch_table, normalize_page_size
from portal.client import request_json as _request_json
from portal.download import write_json, write_ndjson, write_per_match_from_tables
from portal.schema import ANON_KEY, DEFAULT_FULL_TABLES, TABLES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/portal_history/latest")
    parser.add_argument("--jwt", default=os.environ.get("FULLHOUSE_SUPABASE_JWT", ANON_KEY))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--table",
        action="append",
        choices=sorted(TABLES),
        help="Table to download. Defaults to all useful replay tables.",
    )
    args = parser.parse_args()

    if args.page_size > 1000:
        print("Supabase REST caps pages at 1000 rows; using --page-size 1000", flush=True)
    page_size = normalize_page_size(args.page_size)
    output = Path(args.output)
    tables = args.table or list(DEFAULT_FULL_TABLES)

    downloaded: dict[str, list[dict]] = {}
    for table in tables:
        rows = fetch_table(table, args.jwt, page_size, args.timeout, args.retries)
        downloaded[table] = rows
        write_ndjson(output / f"{table}.ndjson", rows)
        write_json(output / f"{table}.json", rows)

    if {"matches", "match_bots", "hands", "hand_winners"} <= set(downloaded):
        write_per_match_from_tables(output, downloaded)

    write_json(output / "summary.json", {table: len(rows) for table, rows in downloaded.items()})
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
