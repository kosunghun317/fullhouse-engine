"""Evaluate a candidate bot against generated portal profile mocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portal.mock_generation import DEFAULT_CANDIDATE, evaluate_profile_mocks


def parse_seeds(raw: str | None, seed_start: int, seed_count: int) -> list[int]:
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return list(range(seed_start, seed_start + seed_count))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mocks_dir", type=Path)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--mode", choices=["sixmax", "heads-up"], default="sixmax")
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seeds")
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds, args.seed_start, args.seed_count)
    report = evaluate_profile_mocks(
        args.mocks_dir,
        candidate=args.candidate,
        seeds=seeds,
        hands=args.hands,
        mode=args.mode,
        profiles=args.profile,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = report["summary"]
        print(
            f"portal_profile_mocks: mean={summary['mean_delta']} "
            f"median={summary['median_delta']} min={summary['min_delta']} "
            f"max={summary['max_delta']} positive={summary['positive_runs']}/{summary['runs']} "
            f"errors={summary['candidate_error_count']}"
        )
        for suite in report["suites"]:
            item = suite["summary"]
            print(
                f"{suite['suite']}: mean={item['mean_delta']} "
                f"median={item['median_delta']} min={item['min_delta']} "
                f"max={item['max_delta']} positive={item['positive_runs']}/{item['runs']} "
                f"errors={item['candidate_error_count']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
