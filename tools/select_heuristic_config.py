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
from tools.heuristic_env_overrides import heuristic_env_scope, load_env_overrides
from tools.tune_heuristic_thresholds import CONFIGS


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

MOCK_FAMILY_SUITES = [
    "mock_equity_family_6max",
    "mock_bucket_family_6max",
    "mock_policy_family_6max",
    "mock_anti_heuristic_6max",
    "mock_pressure_heads_up",
]

STRONG_SUITES = [
    "strong_mock_6max",
    "strong_hybrid_6max",
    "heads_up_strong_rollout",
    "heads_up_strong_ensemble",
]

PRESETS = {
    "quick": {
        "seed_count": 3,
        "suites": ["reference_6max", "mock_rl_6max"],
        "note": "integration-only; not enough to promote or reject defaults",
    },
    "candidate": {
        "seed_count": 64,
        "suites": CORE_SUITES + STRESS_SUITES,
        "note": "statistical candidate screen",
    },
    "mock-screen": {
        "seed_count": 64,
        "suites": ["reference_6max", "mutant_6max"] + MOCK_SUITES + MOCK_FAMILY_SUITES,
        "note": "compressed-model/mock-opponent screen with expanded mock families",
    },
    "mock-family": {
        "seed_count": 128,
        "suites": MOCK_FAMILY_SUITES,
        "note": "focused screen against expanded trained/lookup/adversarial mock families",
    },
    "strong-screen": {
        "seed_count": 128,
        "suites": STRONG_SUITES,
        "note": "focused screen against benchmark-only strong mock opponents",
    },
    "promotion": {
        "seed_count": 128,
        "suites": CORE_SUITES + STRESS_SUITES + MOCK_SUITES,
        "note": "default-promotion screen",
    },
    "final": {
        "seed_count": 200,
        "suites": CORE_SUITES + STRESS_SUITES + MOCK_SUITES,
        "note": "final acceptance matrix",
    },
}


def _parse_seeds(args, preset):
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    count = args.seed_count or PRESETS[preset]["seed_count"]
    return list(range(args.seed_start, args.seed_start + count))


def validate_budget(args, suites, seeds):
    if args.allow_smoke or args.preset == "quick":
        return
    if int(args.hands) < 400:
        raise SystemExit("--hands must be at least 400 unless --preset quick or --allow-smoke is set")
    tasks_per_config = len(suites) * len(seeds)
    if tasks_per_config < int(args.min_tasks_per_config):
        raise SystemExit(
            f"each config must have at least {args.min_tasks_per_config} suite/seed tasks; "
            "increase --seed-count or use --allow-smoke only for wiring checks"
        )


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


def evaluate_env_with_progress(name, env, suites, seeds, hands, progress=False, workers=1, parallel_backend="process"):
    with heuristic_env_scope(env):
        results = []
        for index, suite in enumerate(suites, 1):
            if progress:
                print(
                    f"[{name}] suite {index}/{len(suites)}: {suite} "
                    f"({len(seeds)} seeds x {hands} hands)",
                    file=sys.stderr,
                    flush=True,
                )
            results.append(
                run_suite(
                    suite,
                    seeds,
                    hands,
                    summary_only=True,
                    workers=workers,
                    parallel_backend=parallel_backend,
                )
            )
    return {"config": name, "env": env, "results": results}


def _env_file_configs(args) -> dict[str, dict[str, str]]:
    configs = {}
    names = args.env_config_name or []
    for index, env_file in enumerate(args.env_file or []):
        name = names[index] if index < len(names) else f"env-{Path(env_file).stem}"
        configs[name] = load_env_overrides(env_file)
    return configs


def _apply_bot_overrides(suites: dict, overrides: list[str] | None) -> dict[str, str]:
    applied: dict[str, str] = {}
    for raw in overrides or []:
        if "=" not in raw:
            raise SystemExit(f"invalid --bot-override {raw!r}; expected bot_id=path")
        bot_id, path = raw.split("=", 1)
        bot_id = bot_id.strip()
        path = path.strip()
        if not bot_id or not path:
            raise SystemExit(f"invalid --bot-override {raw!r}; expected bot_id=path")
        matched = False
        for suite in suites.values():
            bots = suite.get("bots", {})
            if bot_id in bots:
                bots[bot_id] = path
                matched = True
        if not matched:
            raise SystemExit(f"--bot-override bot id {bot_id!r} did not match any selected suite")
        applied[bot_id] = path
    return applied


def main():
    parser = argparse.ArgumentParser(description="Run and rank heuristic configs with a risk-aware score")
    parser.add_argument("--config", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--env-file", action="append", help="JSON or best_env.sh file to evaluate as a config")
    parser.add_argument("--env-config-name", action="append", help="Display name for the matching --env-file")
    parser.add_argument("--suite", choices=sorted(SUITES), action="append")
    parser.add_argument("--bot-override", action="append", help="Override a suite bot path, e.g. rollout_search=runs/.../best_bot")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-start", type=int, default=7001)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1, help="Parallel seed workers inside each suite; use 0 for auto")
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--risk-weight", type=float, default=0.35)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--bust-penalty", type=float, default=8000.0)
    parser.add_argument("--error-penalty", type=float, default=20000.0)
    parser.add_argument("--min-tasks-per-config", type=int, default=512)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Print config/suite progress to stderr")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    env_configs = _env_file_configs(args)
    configs = list(args.config or [])
    configs.extend(env_configs)
    if not configs:
        configs = sorted(CONFIGS)
    config_envs = {**CONFIGS, **env_configs}
    suites = args.suite or PRESETS[args.preset]["suites"]
    bot_overrides = _apply_bot_overrides(SUITES, args.bot_override)
    seeds = _parse_seeds(args, args.preset)
    validate_budget(args, suites, seeds)

    report = [
        evaluate_env_with_progress(
            name,
            config_envs[name],
            suites,
            seeds,
            args.hands,
            progress=args.progress,
            workers=args.workers,
            parallel_backend=args.parallel_backend,
        )
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
        "bot_overrides": bot_overrides,
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
