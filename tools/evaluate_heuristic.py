"""Run seeded local benchmarks for the heuristic bot.

This is a development harness, not submitted bot code.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match


HEURISTIC = "bots/heuristic/bot.py"

SUITES = {
    "reference_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "aggressor": "bots/aggressor/bot.py",
            "mathematician": "bots/mathematician/bot.py",
            "shark": "bots/shark/bot.py",
            "template": "bots/template/bot.py",
            "pot_odds": "bots/ref_bot_2/bot.py",
        },
    },
    "mutant_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "always_call": "bots/benchmarks/always_call/bot.py",
            "overfold": "bots/benchmarks/overfold/bot.py",
            "jammer": "bots/benchmarks/jammer/bot.py",
            "minraiser": "bots/benchmarks/minraiser/bot.py",
            "template": "bots/template/bot.py",
        },
    },
    "heads_up_shark": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "shark": "bots/shark/bot.py",
        },
    },
    "heads_up_aggressor": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "aggressor": "bots/aggressor/bot.py",
        },
    },
    "heads_up_station": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "always_call": "bots/benchmarks/always_call/bot.py",
        },
    },
}


def summarize(results):
    deltas = [r["chip_delta"].get("heuristic", 0) for r in results]
    errors = []
    for result in results:
        errors.extend(result["bot_errors"].get("heuristic", []))
    return {
        "runs": len(results),
        "mean_delta": round(statistics.mean(deltas), 2) if deltas else 0,
        "median_delta": round(statistics.median(deltas), 2) if deltas else 0,
        "min_delta": min(deltas) if deltas else 0,
        "max_delta": max(deltas) if deltas else 0,
        "stdev_delta": round(statistics.pstdev(deltas), 2) if len(deltas) > 1 else 0,
        "positive_runs": sum(1 for d in deltas if d > 0),
        "nonnegative_runs": sum(1 for d in deltas if d >= 0),
        "heuristic_errors": errors,
    }


def run_suite(name, seeds, hands_override=None):
    suite = SUITES[name]
    hands = hands_override or suite["hands"]
    results = []
    for seed in seeds:
        result = run_match(
            match_id=f"{name}_{seed}",
            bot_paths=suite["bots"],
            n_hands=hands,
            verbose=False,
            seed=seed,
        )
        results.append({
            "seed": seed,
            "chip_delta": result["chip_delta"],
            "final_stacks": result["final_stacks"],
            "bot_errors": result["bot_errors"],
            "duration_s": result["duration_s"],
        })
    return {"suite": name, "hands": hands, "summary": summarize(results), "runs": results}


def main():
    parser = argparse.ArgumentParser(description="Benchmark bots/heuristic/bot.py")
    parser.add_argument("--suite", choices=sorted(SUITES), action="append")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--hands", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    suites = args.suite or ["reference_6max", "mutant_6max", "heads_up_shark"]
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    report = [run_suite(name, seeds, args.hands) for name in suites]

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    for item in report:
        summary = item["summary"]
        print(
            f"{item['suite']}: mean={summary['mean_delta']} "
            f"median={summary['median_delta']} min={summary['min_delta']} "
            f"max={summary['max_delta']} positive={summary['positive_runs']}/{summary['runs']} "
            f"errors={len(summary['heuristic_errors'])}"
        )


if __name__ == "__main__":
    main()
