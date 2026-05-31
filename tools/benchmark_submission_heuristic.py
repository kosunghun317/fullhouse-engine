"""Benchmark the current submission-ready heuristic zip on rollout suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.benchmark_adaptive_threshold_rollout import SUITES, render_results_table, run_suite


DEFAULT_CANDIDATE_NAME = "heuristic_submission"
DEFAULT_BOT_PATH = "dist/heuristic_bot.zip"


def parse_candidate(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError("candidate must be candidate_id=path")
    key, value = item.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError("candidate must be candidate_id=path")
    return key, value


def build_candidates(bot_path: str, candidate_name: str, extra_candidates: list[str]) -> dict[str, str]:
    candidates = {candidate_name: bot_path}
    for item in extra_candidates:
        key, value = parse_candidate(item)
        candidates[key] = value
    return candidates


def _existing_paths(paths: list[str]) -> bool:
    return all((ROOT / path).exists() for path in paths)


def run_benchmark(
    candidates: dict[str, str],
    suite_ids: list[str],
    seeds: list[int],
    hands: int,
    json_output: bool,
) -> list[dict]:
    rows = []
    for candidate_id, candidate_path in candidates.items():
        if not (ROOT / candidate_path).exists():
            raise SystemExit(f"missing candidate {candidate_id}: {candidate_path}")
        for suite_id in suite_ids:
            opponents = SUITES[suite_id]
            if not _existing_paths(opponents):
                print(f"skip missing suite {suite_id}", file=sys.stderr)
                continue
            row = run_suite(candidate_id, candidate_path, suite_id, opponents, seeds, hands)
            rows.append(row)
            if not json_output:
                print(
                    f"completed {candidate_id:28s} {suite_id:24s} "
                    f"mean={row['mean_delta']:8.1f} ci95={row['ci95']:7.1f} "
                    f"pos={row['positive_rate']:.3f} bust={row['bust_rate']:.3f} "
                    f"errors={row['bot_error_count']}"
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-path", default=DEFAULT_BOT_PATH,
                        help="Path to the submission-ready heuristic bot directory or zip.")
    parser.add_argument("--candidate-name", default=DEFAULT_CANDIDATE_NAME,
                        help="Name to show for --bot-path in the result table.")
    parser.add_argument("--hands", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=9000)
    parser.add_argument("--candidate", action="append", default=[],
                        help="Optional extra comparison candidate as candidate_id=path.")
    parser.add_argument("--suite", action="append", default=[],
                        help="Suite id to run. Defaults to the rollout/threshold benchmark suite matrix.")
    parser.add_argument("--table-format", default="github",
                        help="tabulate tablefmt for the final non-JSON summary table.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    unknown_suites = sorted(set(args.suite) - set(SUITES))
    if unknown_suites:
        raise SystemExit("unknown suite(s): " + ", ".join(unknown_suites))

    try:
        candidates = build_candidates(args.bot_path, args.candidate_name, args.candidate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    suite_ids = args.suite or list(SUITES)
    seeds = [args.seed_start + index for index in range(args.seeds)]
    rows = run_benchmark(
        candidates=candidates,
        suite_ids=suite_ids,
        seeds=seeds,
        hands=args.hands,
        json_output=args.json,
    )

    if args.json:
        print(json.dumps({"results": rows}, indent=2, sort_keys=True))
    elif rows:
        print()
        print(render_results_table(rows, tablefmt=args.table_format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
