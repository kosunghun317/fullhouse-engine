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


def evaluate_config(name, suites, seeds, hands):
    old = _set_env(CONFIGS[name])
    try:
        results = [run_suite(suite, seeds, hands) for suite in suites]
    finally:
        _restore_env(old)
    return {"config": name, "env": CONFIGS[name], "results": results}


def main():
    parser = argparse.ArgumentParser(description="Tune heuristic threshold env configurations")
    parser.add_argument("--config", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    configs = args.config or sorted(CONFIGS)
    suites = args.suite or ["reference_6max", "mutant_6max", "heads_up_aggressor"]
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]

    report = [evaluate_config(name, suites, seeds, args.hands) for name in configs]
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    for item in report:
        print(item["config"])
        for suite in item["results"]:
            summary = suite["summary"]
            print(
                f"  {suite['suite']}: mean={summary['mean_delta']} "
                f"min={summary['min_delta']} positive={summary['positive_runs']}/{summary['runs']} "
                f"errors={len(summary['heuristic_errors'])}"
            )


if __name__ == "__main__":
    main()
