"""Benchmark adaptive threshold rollout candidates without writing artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match


DEFAULT_CANDIDATES = {
    "adaptive_threshold": "bots/adaptive_threshold_rollout_submission",
    "blend_adaptive_heuristic": "bots/blend_adaptive_threshold_heuristic",
}

SUITES = {
    "heads_up_shark": ["bots/shark"],
    "heads_up_aggressor": ["bots/aggressor"],
    "heads_up_station": ["bots/mock_competitors/numpy_policy_station"],
    "reference_6max": [
        "bots/shark",
        "bots/ref_bot_2",
        "bots/mathematician",
        "bots/aggressor",
        "bots/template",
    ],
    "pressure_6max": [
        "bots/aggressor",
        "bots/benchmarks/half_pot_pressure",
        "bots/mock_competitors/equity_pressure",
        "bots/mock_competitors/bucket_overbet",
        "bots/mock_competitors/pressure_heads_up",
    ],
    "tight_6max": [
        "bots/benchmarks/tight_premium",
        "bots/mock_competitors/equity_tight",
        "bots/ref_bot_2",
        "bots/template",
        "bots/benchmarks/threshold_caller",
    ],
    "mixed_stress_6max": [
        "bots/aggressor",
        "bots/benchmarks/jammer",
        "bots/benchmarks/minraiser",
        "bots/mock_competitors/bucket_mixed",
        "bots/mock_competitors/equity_loose",
    ],
    "mock_equity_family_6max": [
        "bots/mock_competitors/equity_mc",
        "bots/mock_competitors/equity_loose",
        "bots/mock_competitors/equity_tight",
        "bots/mock_competitors/equity_pressure",
        "bots/mock_competitors/pot_odds_plus",
    ],
    "mock_bucket_family_6max": [
        "bots/mock_competitors/bucket_policy",
        "bots/mock_competitors/bucket_halfpot",
        "bots/mock_competitors/bucket_mixed",
        "bots/mock_competitors/bucket_overbet",
        "bots/strong_mocks/cfr_bucket",
    ],
    "mock_policy_family_6max": [
        "bots/mock_competitors/numpy_policy",
        "bots/mock_competitors/numpy_policy_bluff",
        "bots/mock_competitors/numpy_policy_folder",
        "bots/mock_competitors/numpy_policy_pressure",
        "bots/mock_competitors/numpy_policy_value",
    ],
    "strong_hu_rollout": ["bots/strong_mocks/rollout_search"],
    "strong_hu_ensemble": ["bots/strong_mocks/ensemble"],
    "strong_neural_6max": [
        "bots/strong_mocks/ppo_deep_policy",
        "bots/strong_mocks/ppo_policy",
        "bots/strong_mocks/oracle_imitation",
        "bots/strong_mocks/ensemble",
        "bots/strong_mocks/heuristic_rl_selector",
    ],
    "strong_hybrid_6max": [
        "bots/strong_mocks/cfr_bucket",
        "bots/strong_mocks/rollout_search",
        "bots/strong_mocks/ensemble",
        "bots/strong_mocks/heuristic_rl_selector",
        "bots/strong_mocks/ppo_deep_policy",
    ],
}


def _existing(paths: list[str]) -> bool:
    return all((ROOT / path).exists() for path in paths)


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _summarize(values: list[int], busts: list[bool], errors: int) -> dict:
    ordered = sorted(values)
    return {
        "matches": len(values),
        "mean_delta": round(statistics.mean(values), 3) if values else 0.0,
        "median_delta": round(statistics.median(values), 3) if values else 0.0,
        "p10_delta": ordered[max(0, int(0.10 * len(ordered)) - 1)] if ordered else 0,
        "ci95": round(_ci95(values), 3),
        "positive_rate": round(sum(1 for value in values if value > 0) / max(1, len(values)), 4),
        "bust_rate": round(sum(1 for item in busts if item) / max(1, len(busts)), 4),
        "bot_error_count": errors,
    }


def run_suite(candidate_id: str, candidate_path: str, suite_id: str, opponents: list[str], seeds: list[int], hands: int) -> dict:
    values: list[int] = []
    busts: list[bool] = []
    error_count = 0
    bot_paths = {"hero": str(ROOT / candidate_path)}
    for index, opponent in enumerate(opponents):
        bot_paths[f"opp{index + 1}"] = str(ROOT / opponent)

    for seed in seeds:
        result = run_match(
            match_id=f"{candidate_id}_{suite_id}_{seed}",
            bot_paths=bot_paths,
            n_hands=hands,
            verbose=False,
            seed=seed,
        )
        delta = int(result["chip_delta"]["hero"])
        values.append(delta)
        busts.append(result["final_stacks"]["hero"] <= 0)
        error_count += len(result["bot_errors"].get("hero", []))

    summary = _summarize(values, busts, error_count)
    summary.update({"candidate": candidate_id, "suite": suite_id, "hands": hands})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=9000)
    parser.add_argument("--candidate", action="append", default=[],
                        help="candidate_id=path. May be passed multiple times.")
    parser.add_argument("--suite", action="append", default=[],
                        help="Suite id to run. Defaults to all known suites.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    candidates = dict(DEFAULT_CANDIDATES)
    for item in args.candidate:
        if "=" not in item:
            raise SystemExit("--candidate must be candidate_id=path")
        key, value = item.split("=", 1)
        candidates[key] = value

    suite_ids = args.suite or list(SUITES)
    seeds = [args.seed_start + index for index in range(args.seeds)]
    rows = []
    for candidate_id, candidate_path in candidates.items():
        if not (ROOT / candidate_path).exists():
            print(f"skip missing candidate {candidate_id}: {candidate_path}", file=sys.stderr)
            continue
        for suite_id in suite_ids:
            opponents = SUITES[suite_id]
            if not _existing(opponents):
                print(f"skip missing suite {suite_id}", file=sys.stderr)
                continue
            row = run_suite(candidate_id, candidate_path, suite_id, opponents, seeds, args.hands)
            rows.append(row)
            if not args.json:
                print(
                    f"{candidate_id:28s} {suite_id:24s} "
                    f"mean={row['mean_delta']:8.1f} ci95={row['ci95']:7.1f} "
                    f"pos={row['positive_rate']:.3f} bust={row['bust_rate']:.3f} "
                    f"errors={row['bot_error_count']}"
                )

    if args.json:
        print(json.dumps({"results": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
