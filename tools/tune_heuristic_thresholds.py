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


def evaluate_config(name, suites, seeds, hands, summary_only=False):
    old = _set_env(CONFIGS[name])
    try:
        results = [run_suite(suite, seeds, hands, summary_only) for suite in suites]
    finally:
        _restore_env(old)
    return {"config": name, "env": CONFIGS[name], "results": results}


def _parse_seeds(args):
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    if args.seed_count:
        return list(range(args.seed_start, args.seed_start + args.seed_count))
    return [101, 202, 303]


def main():
    parser = argparse.ArgumentParser(description="Tune heuristic threshold env configurations")
    parser.add_argument("--config", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--hands", type=int, default=400)
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

    report = [evaluate_config(name, suites, seeds, args.hands, args.summary_only) for name in configs]
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
