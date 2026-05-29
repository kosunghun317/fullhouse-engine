"""Compare named heuristic threshold configurations.

This is a local tuning tool. It changes environment variables before starting
bot subprocesses; the submitted bot defaults are unchanged when env vars are
unset.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_heuristic import run_suite


CONFIGS = {
    "baseline": {},
    "profile-targeting-off": {
        "HEURISTIC_PROFILE_TARGETING_ENABLED": "0.0",
    },
    "legacy-baseline": {
        "HEURISTIC_SPR_LOW_VALUE_DISCOUNT": "0.0",
        "HEURISTIC_SPR_LOW_THIN_VALUE_DISCOUNT": "0.0",
        "HEURISTIC_SPR_LOW_CALL_MARGIN_DISCOUNT": "0.0",
        "HEURISTIC_SPR_HIGH_CALL_MARGIN_BONUS": "0.0",
        "HEURISTIC_SPR_LOW_RAISE_FRACTION_BONUS": "0.0",
        "HEURISTIC_SPR_COMMIT_EQUITY": "1.01",
        "HEURISTIC_OFF_BUCKET_SIZING_PROB": "0.0",
        "HEURISTIC_OFF_BUCKET_BLUFF_MIN": "0.56",
        "HEURISTIC_OFF_BUCKET_VALUE_STATION_BONUS": "0.10",
        "HEURISTIC_OFF_BUCKET_VALUE_TIGHT_MAX": "0.49",
    },
    "small-ball": {
        "HEURISTIC_NORMAL_VALUE_FRACTION": "0.42",
        "HEURISTIC_PRESSURE_VALUE_FRACTION": "0.62",
        "HEURISTIC_THIN_VALUE_FRACTION": "0.34",
        "HEURISTIC_DRY_BLUFF_FRACTION": "0.34",
        "HEURISTIC_WET_SEMI_BLUFF_FRACTION": "0.44",
    },
    "large-value": {
        "HEURISTIC_NORMAL_VALUE_FRACTION": "0.62",
        "HEURISTIC_PRESSURE_VALUE_FRACTION": "0.90",
        "HEURISTIC_THIN_VALUE_FRACTION": "0.52",
        "HEURISTIC_HIGH_EQUITY_RAISE_FRACTION": "1.05",
        "HEURISTIC_DRY_BLUFF_PROB": "0.28",
        "HEURISTIC_WET_BLUFF_PROB": "0.12",
    },
    "conservative": {
        "HEURISTIC_CALL_MARGIN_BASE": "0.095",
        "HEURISTIC_CALL_MARGIN_MULTIWAY": "0.045",
        "HEURISTIC_RISK_REQ_LOW": "0.80",
        "HEURISTIC_RISK_REQ_MID": "0.89",
        "HEURISTIC_RISK_REQ_HIGH": "0.94",
        "HEURISTIC_DRY_BLUFF_PROB": "0.32",
        "HEURISTIC_WET_BLUFF_PROB": "0.15",
    },
    "value-heavy": {
        "HEURISTIC_VALUE_THRESHOLD_BASE": "0.63",
        "HEURISTIC_THIN_VALUE_BASE": "0.56",
        "HEURISTIC_DRY_BLUFF_PROB": "0.25",
        "HEURISTIC_WET_BLUFF_PROB": "0.10",
    },
    "pressure": {
        "HEURISTIC_CALL_MARGIN_BASE": "0.065",
        "HEURISTIC_DRY_BLUFF_PROB": "0.58",
        "HEURISTIC_WET_BLUFF_PROB": "0.32",
        "HEURISTIC_VALUE_THRESHOLD_BASE": "0.65",
    },
    "tight-preflop": {
        "HEURISTIC_PREFLOP_OPEN_SCORE": "66",
        "HEURISTIC_PREFLOP_LATE_PLAY_SCORE": "54",
        "HEURISTIC_PREFLOP_CHEAP_CALL_SCORE": "52",
        "HEURISTIC_PREFLOP_LATE_RAISE_CALL_SCORE": "62",
    },
    "loose-position": {
        "HEURISTIC_PREFLOP_OPEN_SCORE": "58",
        "HEURISTIC_PREFLOP_LATE_PLAY_SCORE": "42",
        "HEURISTIC_PREFLOP_CHEAP_CALL_SCORE": "44",
        "HEURISTIC_PREFLOP_HU_MANIAC_OPEN_SCORE": "48",
    },
    "risk-averse": {
        "HEURISTIC_RISK_REQ_LOW": "0.81",
        "HEURISTIC_RISK_REQ_MID": "0.90",
        "HEURISTIC_RISK_REQ_HIGH": "0.95",
        "HEURISTIC_RISK_CUTOFF_LOW": "0.22",
        "HEURISTIC_CALL_MARGIN_BASE": "0.095",
        "HEURISTIC_DRY_BLUFF_PROB": "0.25",
    },
    "spr-aware": {
        "HEURISTIC_SPR_LOW_VALUE_DISCOUNT": "0.035",
        "HEURISTIC_SPR_LOW_THIN_VALUE_DISCOUNT": "0.025",
        "HEURISTIC_SPR_LOW_CALL_MARGIN_DISCOUNT": "0.025",
        "HEURISTIC_SPR_HIGH_CALL_MARGIN_BONUS": "0.025",
        "HEURISTIC_SPR_LOW_RAISE_FRACTION_BONUS": "0.08",
        "HEURISTIC_SPR_COMMIT_EQUITY": "0.74",
    },
    "anti-bucket": {
        "HEURISTIC_OFF_BUCKET_SIZING_PROB": "0.16",
        "HEURISTIC_OFF_BUCKET_BLUFF_MIN": "0.56",
        "HEURISTIC_OFF_BUCKET_VALUE_STATION_BONUS": "0.10",
        "HEURISTIC_OFF_BUCKET_VALUE_TIGHT_MAX": "0.49",
    },
    "spr-anti-bucket": {
        "HEURISTIC_SPR_LOW_VALUE_DISCOUNT": "0.035",
        "HEURISTIC_SPR_LOW_THIN_VALUE_DISCOUNT": "0.025",
        "HEURISTIC_SPR_LOW_CALL_MARGIN_DISCOUNT": "0.025",
        "HEURISTIC_SPR_HIGH_CALL_MARGIN_BONUS": "0.025",
        "HEURISTIC_SPR_LOW_RAISE_FRACTION_BONUS": "0.08",
        "HEURISTIC_SPR_COMMIT_EQUITY": "0.74",
        "HEURISTIC_OFF_BUCKET_SIZING_PROB": "0.12",
        "HEURISTIC_OFF_BUCKET_BLUFF_MIN": "0.56",
        "HEURISTIC_OFF_BUCKET_VALUE_STATION_BONUS": "0.08",
        "HEURISTIC_OFF_BUCKET_VALUE_TIGHT_MAX": "0.49",
    },
    "pressure-control": {
        "HEURISTIC_MIXED_PRESSURE_CALL_MARGIN_BONUS": "0.025",
        "HEURISTIC_MIXED_PRESSURE_RISK_BONUS": "0.045",
        "HEURISTIC_HU_LEAD_MANIAC_MARGIN_BONUS": "0.065",
        "HEURISTIC_HU_LEAD_MANIAC_RISK_BONUS": "0.12",
    },
    "equity-control": {
        "HEURISTIC_EXTRA_LARGE_BET_EQUITY_PENALTY": "0.035",
        "HEURISTIC_EXTRA_LARGE_BET_THRESHOLD": "0.58",
        "HEURISTIC_EQUITY_ADJ_LARGE_BET": "-0.045",
    },
    "trap-control": {
        "HEURISTIC_TRAP_CHECK_PROB": "0.08",
        "HEURISTIC_TRAP_CHECK_MIN_EQUITY": "0.80",
        "HEURISTIC_TRAP_CHECK_SPR_MAX": "3.2",
    },
    "weakspot-control": {
        "HEURISTIC_MIXED_PRESSURE_CALL_MARGIN_BONUS": "0.020",
        "HEURISTIC_MIXED_PRESSURE_RISK_BONUS": "0.035",
        "HEURISTIC_HU_LEAD_MANIAC_MARGIN_BONUS": "0.060",
        "HEURISTIC_HU_LEAD_MANIAC_RISK_BONUS": "0.11",
        "HEURISTIC_EXTRA_LARGE_BET_EQUITY_PENALTY": "0.025",
        "HEURISTIC_EXTRA_LARGE_BET_THRESHOLD": "0.60",
        "HEURISTIC_TRAP_CHECK_PROB": "0.06",
        "HEURISTIC_TRAP_CHECK_MIN_EQUITY": "0.81",
        "HEURISTIC_TRAP_CHECK_SPR_MAX": "3.0",
    },
    "line-aware": {
        "HEURISTIC_DELAYED_PROBE_PROB": "0.18",
        "HEURISTIC_DELAYED_PROBE_MIN_EQUITY": "0.36",
        "HEURISTIC_DELAYED_PROBE_FOLD_PRESSURE": "0.52",
        "HEURISTIC_BLOCKER_BLUFF_PROB": "0.05",
        "HEURISTIC_BLOCKER_BLUFF_MIN_EQUITY": "0.28",
        "HEURISTIC_BLOCKER_BLUFF_FOLD_PRESSURE": "0.60",
        "HEURISTIC_TOP_PAIR_VALUE_DISCOUNT": "0.012",
        "HEURISTIC_OVERPAIR_VALUE_DISCOUNT": "0.018",
        "HEURISTIC_BOARD_PAIR_DANGER_PENALTY": "0.014",
    },
    "blocker-probe": {
        "HEURISTIC_DELAYED_PROBE_PROB": "0.24",
        "HEURISTIC_DELAYED_PROBE_FOLD_PRESSURE": "0.55",
        "HEURISTIC_BLOCKER_BLUFF_PROB": "0.08",
        "HEURISTIC_BLOCKER_BLUFF_MIN_EQUITY": "0.26",
        "HEURISTIC_BLOCKER_BLUFF_FOLD_PRESSURE": "0.62",
        "HEURISTIC_BLOCKER_BLUFF_FRACTION": "0.58",
    },
    "pair-danger": {
        "HEURISTIC_TOP_PAIR_VALUE_DISCOUNT": "0.010",
        "HEURISTIC_OVERPAIR_VALUE_DISCOUNT": "0.016",
        "HEURISTIC_BOARD_PAIR_DANGER_PENALTY": "0.026",
        "HEURISTIC_EXTRA_LARGE_BET_EQUITY_PENALTY": "0.018",
        "HEURISTIC_EXTRA_LARGE_BET_THRESHOLD": "0.62",
    },
}


def _set_env(overrides):
    keys = set()
    for config in CONFIGS.values():
        keys.update(config)
    old = {key: os.environ.get(key) for key in keys}
    for key in keys:
        if key in overrides:
            os.environ[key] = overrides[key]
        else:
            os.environ.pop(key, None)
    return old


def _restore_env(old):
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def evaluate_config(name, suites, seeds, hands, summary_only=False, workers=1, parallel_backend="process"):
    old = _set_env(CONFIGS[name])
    try:
        results = [
            run_suite(
                suite,
                seeds,
                hands,
                summary_only,
                workers=workers,
                parallel_backend=parallel_backend,
            )
            for suite in suites
        ]
    finally:
        _restore_env(old)
    return {"config": name, "env": CONFIGS[name], "results": results}


def _parse_seeds(args):
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    if args.seed_count:
        return list(range(args.seed_start, args.seed_start + args.seed_count))
    return [101, 202, 303]


def validate_budget(args, suites, seeds):
    if args.allow_smoke:
        return
    if int(args.hands) < 400:
        raise SystemExit("--hands must be at least 400 unless --allow-smoke is set")
    tasks_per_config = len(suites) * len(seeds)
    if tasks_per_config < int(args.min_tasks_per_config):
        raise SystemExit(
            f"each config must have at least {args.min_tasks_per_config} suite/seed tasks; "
            "increase --seed-count or use --allow-smoke only for wiring checks"
        )


def main():
    parser = argparse.ArgumentParser(description="Tune heuristic threshold env configurations")
    parser.add_argument("--config", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seed-count", type=int, default=128)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--min-tasks-per-config", type=int, default=512)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    configs = args.config or sorted(CONFIGS)
    suites = args.suite or [
        "reference_6max",
        "mutant_6max",
        "heads_up_shark",
        "heads_up_aggressor",
        "heads_up_station",
        "sizing_6max",
        "pressure_6max",
        "tight_6max",
        "mixed_stress_6max",
        "heads_up_threshold",
    ]
    seeds = _parse_seeds(args)
    validate_budget(args, suites, seeds)

    report = [
        evaluate_config(
            name,
            suites,
            seeds,
            args.hands,
            args.summary_only,
            workers=args.workers,
            parallel_backend=args.parallel_backend,
        )
        for name in configs
    ]
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    for item in report:
        print(item["config"])
        for suite in item["results"]:
            summary = suite["summary"]
            print(
                f"  {suite['suite']}: mean={summary['mean_delta']} "
                f"min={summary['min_delta']} stdev={summary['stdev_delta']} "
                f"positive={summary['positive_runs']}/{summary['runs']} "
                f"busts={summary['bust_count']} errors={summary['heuristic_error_count']}"
            )


if __name__ == "__main__":
    main()
