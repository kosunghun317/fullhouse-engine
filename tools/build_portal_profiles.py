"""Build replay-derived opponent profiles from portal strategy reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portal.profiles import build_profiles, write_profile_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-hands", type=int, default=200)
    parser.add_argument("--rank-gte", type=int)
    parser.add_argument("--rank-lte", type=int)
    parser.add_argument("--top", type=int, default=100_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output = args.output or args.directory / "profiles"
    summary = build_profiles(
        args.directory,
        report_json=args.report_json,
        min_hands=args.min_hands,
        rank_gte=args.rank_gte,
        rank_lte=args.rank_lte,
        top=args.top,
    )
    write_profile_artifacts(summary, output)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"wrote {output} profiles={summary['profile_count']} "
            f"bots={summary['profiled_bot_count']}/{summary['input_bot_count']}"
        )
        for profile in summary["profiles"]:
            cfg = profile["mock_config"]
            print(
                f"{profile['profile']}\tcount={profile['bot_count']}\t"
                f"vpip={cfg['vpip']}\tpfr={cfg['pfr']}\t"
                f"pressure_fold={cfg['pressure_fold_rate']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
