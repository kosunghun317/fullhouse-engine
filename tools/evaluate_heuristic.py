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
from tools.parallel import map_parallel
from tools.portal.mock_generation import build_match_specs, load_mock_manifest


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
    "sizing_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "threshold_caller": "bots/benchmarks/threshold_caller/bot.py",
            "mathematician": "bots/mathematician/bot.py",
            "overfold": "bots/benchmarks/overfold/bot.py",
            "always_call": "bots/benchmarks/always_call/bot.py",
            "half_pot_pressure": "bots/benchmarks/half_pot_pressure/bot.py",
        },
    },
    "pressure_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "aggressor": "bots/aggressor/bot.py",
            "jammer": "bots/benchmarks/jammer/bot.py",
            "short_stacker": "bots/benchmarks/short_stacker/bot.py",
            "minraiser": "bots/benchmarks/minraiser/bot.py",
            "half_pot_pressure": "bots/benchmarks/half_pot_pressure/bot.py",
        },
    },
    "tight_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "shark": "bots/shark/bot.py",
            "tight_premium": "bots/benchmarks/tight_premium/bot.py",
            "template": "bots/template/bot.py",
            "mathematician": "bots/mathematician/bot.py",
            "overfold": "bots/benchmarks/overfold/bot.py",
        },
    },
    "mixed_stress_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "shark": "bots/shark/bot.py",
            "aggressor": "bots/aggressor/bot.py",
            "threshold_caller": "bots/benchmarks/threshold_caller/bot.py",
            "short_stacker": "bots/benchmarks/short_stacker/bot.py",
            "tight_premium": "bots/benchmarks/tight_premium/bot.py",
        },
    },
    "heads_up_threshold": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "threshold_caller": "bots/benchmarks/threshold_caller/bot.py",
        },
    },
    "mock_rl_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "numpy_policy": "bots/mock_competitors/numpy_policy",
            "equity_mc": "bots/mock_competitors/equity_mc/bot.py",
            "bucket_policy": "bots/mock_competitors/bucket_policy/bot.py",
            "cbet_reg": "bots/mock_competitors/cbet_reg/bot.py",
            "opponent_modeler": "bots/mock_competitors/opponent_modeler/bot.py",
        },
    },
    "mock_bucket_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "bucket_policy": "bots/mock_competitors/bucket_policy/bot.py",
            "pot_odds_plus": "bots/mock_competitors/pot_odds_plus/bot.py",
            "copy_shark_plus": "bots/mock_competitors/copy_shark_plus/bot.py",
            "threshold_caller": "bots/benchmarks/threshold_caller/bot.py",
            "tight_premium": "bots/benchmarks/tight_premium/bot.py",
        },
    },
    "mock_adaptive_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "opponent_modeler": "bots/mock_competitors/opponent_modeler/bot.py",
            "cbet_reg": "bots/mock_competitors/cbet_reg/bot.py",
            "copy_shark_plus": "bots/mock_competitors/copy_shark_plus/bot.py",
            "half_pot_pressure": "bots/benchmarks/half_pot_pressure/bot.py",
            "short_stacker": "bots/benchmarks/short_stacker/bot.py",
        },
    },
    "heads_up_mock_numpy": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "numpy_policy": "bots/mock_competitors/numpy_policy",
        },
    },
    "heads_up_equity_mc": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "equity_mc": "bots/mock_competitors/equity_mc/bot.py",
        },
    },
    "mock_equity_family_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "equity_mc": "bots/mock_competitors/equity_mc/bot.py",
            "equity_tight": "bots/mock_competitors/equity_tight/bot.py",
            "equity_loose": "bots/mock_competitors/equity_loose/bot.py",
            "equity_pressure": "bots/mock_competitors/equity_pressure/bot.py",
            "pot_odds_plus": "bots/mock_competitors/pot_odds_plus/bot.py",
        },
    },
    "mock_bucket_family_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "bucket_policy": "bots/mock_competitors/bucket_policy/bot.py",
            "bucket_halfpot": "bots/mock_competitors/bucket_halfpot/bot.py",
            "bucket_overbet": "bots/mock_competitors/bucket_overbet/bot.py",
            "bucket_mixed": "bots/mock_competitors/bucket_mixed/bot.py",
            "threshold_caller": "bots/benchmarks/threshold_caller/bot.py",
        },
    },
    "mock_policy_family_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "policy_value": "bots/mock_competitors/numpy_policy_value",
            "policy_bluff": "bots/mock_competitors/numpy_policy_bluff",
            "policy_station": "bots/mock_competitors/numpy_policy_station",
            "policy_folder": "bots/mock_competitors/numpy_policy_folder",
            "policy_pressure": "bots/mock_competitors/numpy_policy_pressure",
        },
    },
    "mock_anti_heuristic_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "anti_heuristic": "bots/mock_competitors/anti_heuristic/bot.py",
            "opponent_modeler": "bots/mock_competitors/opponent_modeler/bot.py",
            "equity_pressure": "bots/mock_competitors/equity_pressure/bot.py",
            "bucket_overbet": "bots/mock_competitors/bucket_overbet/bot.py",
            "cbet_reg": "bots/mock_competitors/cbet_reg/bot.py",
        },
    },
    "mock_pressure_heads_up": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "pressure_heads_up": "bots/mock_competitors/pressure_heads_up/bot.py",
        },
    },
    "strong_mock_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "oracle_imitation": "bots/strong_mocks/oracle_imitation",
            "ppo_policy": "bots/strong_mocks/ppo_policy",
            "cfr_bucket": "bots/strong_mocks/cfr_bucket",
            "rollout_search": "bots/strong_mocks/rollout_search",
            "ensemble": "bots/strong_mocks/ensemble",
        },
    },
    "strong_hybrid_6max": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "ensemble": "bots/strong_mocks/ensemble",
            "rollout_search": "bots/strong_mocks/rollout_search",
            "oracle_imitation": "bots/strong_mocks/oracle_imitation",
            "equity_pressure": "bots/mock_competitors/equity_pressure/bot.py",
            "bucket_overbet": "bots/mock_competitors/bucket_overbet/bot.py",
        },
    },
    "heads_up_strong_rollout": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "rollout_search": "bots/strong_mocks/rollout_search",
        },
    },
    "heads_up_strong_ensemble": {
        "hands": 400,
        "bots": {
            "heuristic": HEURISTIC,
            "ensemble": "bots/strong_mocks/ensemble",
        },
    },
}


def register_portal_profile_suites(
    mocks_dir,
    candidate=HEURISTIC,
    mode="sixmax",
    profiles=None,
    hands=400,
) -> list[str]:
    manifest = load_mock_manifest(Path(mocks_dir))
    specs = build_match_specs(
        manifest,
        candidate=candidate,
        profiles=profiles,
        mode=mode,
        candidate_id="heuristic",
    )
    names = []
    for spec in specs:
        SUITES[spec["name"]] = {"hands": hands, "bots": spec["bots"]}
        names.append(spec["name"])
    return names


def summarize(results):
    deltas = [r["chip_delta"].get("heuristic", 0) for r in results]
    errors = []
    busts = 0
    for result in results:
        errors.extend(result["bot_errors"].get("heuristic", []))
        final_stack = result.get("final_stacks", {}).get("heuristic")
        if final_stack is not None:
            busts += int(final_stack <= 0)
        else:
            busts += int(result["chip_delta"].get("heuristic", 0) <= -10000)
    durations = [r.get("duration_s", 0) for r in results]
    return {
        "runs": len(results),
        "mean_delta": round(statistics.mean(deltas), 2) if deltas else 0,
        "median_delta": round(statistics.median(deltas), 2) if deltas else 0,
        "min_delta": min(deltas) if deltas else 0,
        "max_delta": max(deltas) if deltas else 0,
        "stdev_delta": round(statistics.pstdev(deltas), 2) if len(deltas) > 1 else 0,
        "positive_runs": sum(1 for d in deltas if d > 0),
        "nonnegative_runs": sum(1 for d in deltas if d >= 0),
        "bust_count": busts,
        "mean_duration_s": round(statistics.mean(durations), 2) if durations else 0,
        "max_duration_s": round(max(durations), 2) if durations else 0,
        "heuristic_error_count": len(errors),
        "heuristic_errors": errors,
    }


def _run_match_task(task):
    name, seed, hands, bots = task
    result = run_match(
        match_id=f"{name}_{seed}",
        bot_paths=bots,
        n_hands=hands,
        verbose=False,
        seed=seed,
    )
    return {
        "seed": seed,
        "chip_delta": result["chip_delta"],
        "final_stacks": result["final_stacks"],
        "bot_errors": result["bot_errors"],
        "duration_s": result["duration_s"],
    }


def _suite_bots(name, candidate=None):
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; available suites: {', '.join(sorted(SUITES))}")
    suite = SUITES[name]
    bots = dict(suite["bots"])
    if candidate is not None:
        bots["heuristic"] = candidate
    return bots


def run_suite(name, seeds, hands_override=None, summary_only=False, workers=1, parallel_backend="process", candidate=None):
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; available suites: {', '.join(sorted(SUITES))}")
    suite = SUITES[name]
    hands = hands_override or suite["hands"]
    bots = _suite_bots(name, candidate)
    tasks = [(name, seed, hands, bots) for seed in seeds]
    results = map_parallel(_run_match_task, tasks, workers=workers, backend=parallel_backend)
    report = {"suite": name, "hands": hands, "summary": summarize(results)}
    if not summary_only:
        report["runs"] = results
    return report


def _parse_seeds(args):
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    if args.seed_count:
        return list(range(args.seed_start, args.seed_start + args.seed_count))
    return [101, 202, 303]


def main():
    parser = argparse.ArgumentParser(description="Benchmark bots/heuristic/bot.py")
    parser.add_argument("--candidate", default=HEURISTIC)
    parser.add_argument("--suite", action="append")
    parser.add_argument("--portal-mocks-dir", type=Path)
    parser.add_argument("--portal-mocks-mode", choices=["sixmax", "heads-up"], default="sixmax")
    parser.add_argument("--portal-profile", action="append", default=[])
    parser.add_argument("--portal-only", action="store_true")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--hands", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Parallel seed workers; use 0 for auto")
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    portal_suites = []
    if args.portal_mocks_dir:
        portal_suites = register_portal_profile_suites(
            args.portal_mocks_dir,
            candidate=args.candidate,
            mode=args.portal_mocks_mode,
            profiles=args.portal_profile,
            hands=args.hands or 400,
        )

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
    if portal_suites:
        suites = portal_suites if args.portal_only else [*suites, *portal_suites]
    seeds = _parse_seeds(args)
    report = [
        run_suite(
            name,
            seeds,
            args.hands,
            args.summary_only,
            workers=args.workers,
            parallel_backend=args.parallel_backend,
            candidate=args.candidate,
        )
        for name in suites
    ]

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    for item in report:
        summary = item["summary"]
        print(
            f"{item['suite']}: mean={summary['mean_delta']} "
            f"median={summary['median_delta']} min={summary['min_delta']} "
            f"max={summary['max_delta']} stdev={summary['stdev_delta']} "
            f"positive={summary['positive_runs']}/{summary['runs']} "
            f"nonnegative={summary['nonnegative_runs']}/{summary['runs']} "
            f"busts={summary['bust_count']} errors={summary['heuristic_error_count']}"
        )


if __name__ == "__main__":
    main()
