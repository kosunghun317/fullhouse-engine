"""Benchmark the preflop-MC heuristic candidate on focused rollout suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.benchmark_adaptive_threshold_rollout import SUITES, render_results_table
from tools.benchmark_submission_heuristic import build_candidates, run_benchmark


DEFAULT_CANDIDATE_NAME = "preflop_mc_4096_1sec"
DEFAULT_BOT_PATH = "runs/preflop_mc_heuristic/preflop-mc-4096-1sec/bot"
DEFAULT_BASELINE_NAME = "heuristic_submission"
DEFAULT_BASELINE_PATH = "dist/heuristic_bot.zip"
DEFAULT_SUITES = [
    "heads_up_aggressor",
    "reference_6max",
    "strong_hu_rollout",
    "strong_hybrid_6max",
]


def default_suite_ids(all_suites: bool = False) -> list[str]:
    return list(SUITES) if all_suites else list(DEFAULT_SUITES)


def maybe_add_baseline(candidates: dict[str, str], include_baseline: bool) -> dict[str, str]:
    if include_baseline:
        return {DEFAULT_BASELINE_NAME: DEFAULT_BASELINE_PATH, **candidates}
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bot-path",
        default=DEFAULT_BOT_PATH,
        help="Path to the preflop-MC bot directory or zip.",
    )
    parser.add_argument(
        "--candidate-name",
        default=DEFAULT_CANDIDATE_NAME,
        help="Name to show for --bot-path in the result table.",
    )
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=9000)
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Suite id to run. Defaults to the focused preflop-MC suite matrix.",
    )
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="Run every suite from benchmark_adaptive_threshold_rollout.py.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help=f"Also run {DEFAULT_BASELINE_NAME} from {DEFAULT_BASELINE_PATH}.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Optional extra comparison candidate as candidate_id=path.",
    )
    parser.add_argument(
        "--table-format",
        default="github",
        help="tabulate tablefmt for the final non-JSON summary table.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    suite_ids = args.suite or default_suite_ids(args.all_suites)
    unknown_suites = sorted(set(suite_ids) - set(SUITES))
    if unknown_suites:
        raise SystemExit("unknown suite(s): " + ", ".join(unknown_suites))

    try:
        candidates = build_candidates(args.bot_path, args.candidate_name, args.candidate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    candidates = maybe_add_baseline(candidates, args.include_baseline)

    seeds = [args.seed_start + index for index in range(args.seeds)]
    rows = run_benchmark(
        candidates=candidates,
        suite_ids=suite_ids,
        seeds=seeds,
        hands=args.hands,
        json_output=args.json,
    )

    payload = {
        "bot_path": args.bot_path,
        "candidate_name": args.candidate_name,
        "hands": args.hands,
        "seeds": seeds,
        "suites": suite_ids,
        "include_baseline": args.include_baseline,
        "results": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif rows:
        print()
        print(render_results_table(rows, tablefmt=args.table_format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
