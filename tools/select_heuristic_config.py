"""Risk-aware selector for heuristic threshold configurations.

This tool is intentionally local-only. It runs named env configurations across
seeded benchmark suites and ranks them with a conservative score so defaults are
not promoted or rejected from smoke tests.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_heuristic import SUITES, run_suite
from tools.tune_heuristic_thresholds import CONFIGS, _restore_env, _set_env


CORE_SUITES = [
    "reference_6max",
    "mutant_6max",
    "heads_up_shark",
    "heads_up_aggressor",
    "heads_up_station",
]

STRESS_SUITES = [
    "sizing_6max",
    "pressure_6max",
    "tight_6max",
    "mixed_stress_6max",
    "heads_up_threshold",
]

MOCK_SUITES = [
    "mock_rl_6max",
    "mock_bucket_6max",
    "mock_adaptive_6max",
    "heads_up_mock_numpy",
    "heads_up_equity_mc",
]

PRESETS = {
    "quick": {
        "seed_count": 3,
        "suites": ["reference_6max", "mock_rl_6max"],
        "note": "integration-only; not enough to promote or reject defaults",
    },
    "candidate": {
        "seed_count": 10,
        "suites": CORE_SUITES + STRESS_SUITES,
        "note": "minimum candidate screen",
    },
    "mock-screen": {
        "seed_count": 10,
        "suites": ["reference_6max", "mutant_6max"] + MOCK_SUITES,
        "note": "compressed-model/mock-opponent screen",
    },
    "promotion": {
        "seed_count": 30,
        "suites": CORE_SUITES + STRESS_SUITES + MOCK_SUITES,
        "note": "minimum default-promotion screen",
    },
    "final": {
        "seed_count": 100,
        "suites": CORE_SUITES + STRESS_SUITES + MOCK_SUITES,
        "note": "final acceptance matrix",
    },
}


def _parse_seeds(args, preset):
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    count = args.seed_count or PRESETS[preset]["seed_count"]
    return list(range(args.seed_start, args.seed_start + count))


def _score_suite(summary, risk_weight, min_weight, bust_penalty, error_penalty):
    runs = max(1, summary["runs"])
    bust_rate = summary["bust_count"] / runs
    error_rate = summary["heuristic_error_count"] / runs
    return (
        summary["mean_delta"]
        - risk_weight * summary["stdev_delta"]
        + min_weight * summary["min_delta"]
        - bust_penalty * bust_rate
        - error_penalty * error_rate
    )


def _aggregate(item, risk_weight, min_weight, bust_penalty, error_penalty):
    suite_scores = []
    means = []
    medians = []
    mins = []
    positives = 0
    nonnegatives = 0
    runs = 0
    busts = 0
    errors = 0

    for suite in item["results"]:
        summary = suite["summary"]
        suite_scores.append(_score_suite(summary, risk_weight, min_weight, bust_penalty, error_penalty))
        means.append(summary["mean_delta"])
        medians.append(summary["median_delta"])
        mins.append(summary["min_delta"])
        positives += summary["positive_runs"]
        nonnegatives += summary["nonnegative_runs"]
        runs += summary["runs"]
        busts += summary["bust_count"]
        errors += summary["heuristic_error_count"]

    return {
        "config": item["config"],
        "score": round(statistics.mean(suite_scores), 2) if suite_scores else 0.0,
        "mean_of_suite_means": round(statistics.mean(means), 2) if means else 0.0,
        "median_of_suite_medians": round(statistics.median(medians), 2) if medians else 0.0,
        "worst_min_delta": min(mins) if mins else 0,
        "positive_runs": positives,
        "nonnegative_runs": nonnegatives,
        "total_runs": runs,
        "bust_count": busts,
        "bust_rate": round(busts / runs, 4) if runs else 0.0,
        "heuristic_error_count": errors,
    }


def rank_configs(report, risk_weight, min_weight, bust_penalty, error_penalty):
    ranking = [
        _aggregate(item, risk_weight, min_weight, bust_penalty, error_penalty)
        for item in report
    ]
    return sorted(ranking, key=lambda item: item["score"], reverse=True)


def evaluate_config_with_progress(name, suites, seeds, hands, progress=False):
    old = _set_env(CONFIGS[name])
    try:
        results = []
        for index, suite in enumerate(suites, 1):
            if progress:
                print(
                    f"[{name}] suite {index}/{len(suites)}: {suite} "
                    f"({len(seeds)} seeds x {hands} hands)",
                    file=sys.stderr,
                    flush=True,
                )
            results.append(run_suite(suite, seeds, hands, summary_only=True))
    finally:
        _restore_env(old)
    return {"config": name, "env": CONFIGS[name], "results": results}


def main():
    parser = argparse.ArgumentParser(description="Run and rank heuristic configs with a risk-aware score")
    parser.add_argument("--config", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--suite", choices=sorted(SUITES), action="append")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-start", type=int, default=7001)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--risk-weight", type=float, default=0.35)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--bust-penalty", type=float, default=8000.0)
    parser.add_argument("--error-penalty", type=float, default=20000.0)
    parser.add_argument("--progress", action="store_true", help="Print config/suite progress to stderr")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    configs = args.config or sorted(CONFIGS)
    suites = args.suite or PRESETS[args.preset]["suites"]
    seeds = _parse_seeds(args, args.preset)

    report = [
        evaluate_config_with_progress(name, suites, seeds, args.hands, progress=args.progress)
        for name in configs
    ]
    ranking = rank_configs(
        report,
        args.risk_weight,
        args.min_weight,
        args.bust_penalty,
        args.error_penalty,
    )
    payload = {
        "preset": args.preset,
        "preset_note": PRESETS[args.preset]["note"],
        "hands": args.hands,
        "seeds": seeds,
        "suites": suites,
        "score_formula": (
            "mean_delta - risk_weight*stdev_delta + min_weight*min_delta "
            "- bust_penalty*bust_rate - error_penalty*error_rate"
        ),
        "score_params": {
            "risk_weight": args.risk_weight,
            "min_weight": args.min_weight,
            "bust_penalty": args.bust_penalty,
            "error_penalty": args.error_penalty,
        },
        "ranking": ranking,
        "results": report,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"preset={args.preset} ({PRESETS[args.preset]['note']})")
    print(f"hands={args.hands} seeds={len(seeds)} suites={len(suites)}")
    for index, item in enumerate(ranking, 1):
        print(
            f"{index}. {item['config']}: score={item['score']} "
            f"mean={item['mean_of_suite_means']} worst={item['worst_min_delta']} "
            f"positive={item['positive_runs']}/{item['total_runs']} "
            f"busts={item['bust_count']} errors={item['heuristic_error_count']}"
        )


if __name__ == "__main__":
    main()
