"""Paired-seed promotion gate for heuristic bot configurations.

This tool compares candidate env configurations against an incumbent on the
same suite/seed/hands tasks. It is intentionally separate from the risk-aware
leaderboard in ``select_heuristic_config.py`` because default promotion should
be based on paired differences, not unrelated aggregate means.
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match
from tools.evaluate_heuristic import SUITES
from tools.heuristic_env_overrides import heuristic_env_scope, load_env_overrides
from tools.parallel import map_parallel
from tools.select_heuristic_config import PRESETS
from tools.tune_heuristic_thresholds import CONFIGS


def _parse_seeds(args):
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    count = args.seed_count or PRESETS[args.preset]["seed_count"]
    return list(range(args.seed_start, args.seed_start + count))


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(len(ordered) - 1, lo + 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _bootstrap_mean_ci(values, samples=2000, ci_level=0.95, seed=1729):
    if not values:
        return 0.0, 0.0
    mean = float(statistics.mean(values))
    if len(values) == 1 or samples <= 0:
        return mean, mean
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        total = 0.0
        for _idx in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    alpha = max(0.0, min(1.0, 1.0 - ci_level))
    return _percentile(means, alpha / 2.0), _percentile(means, 1.0 - alpha / 2.0)


def _run_task(task):
    suite_name, seed, hands = task
    suite = SUITES[suite_name]
    result = run_match(
        match_id=f"paired_{suite_name}_{seed}",
        bot_paths=suite["bots"],
        n_hands=hands,
        verbose=False,
        seed=seed,
    )
    delta = int(result["chip_delta"].get("heuristic", 0))
    errors = result["bot_errors"].get("heuristic", [])
    final_stack = result.get("final_stacks", {}).get("heuristic")
    busted = bool(final_stack is not None and final_stack <= 0)
    return {
        "suite": suite_name,
        "seed": seed,
        "delta": delta,
        "busted": busted,
        "error_count": len(errors),
        "errors": errors,
    }


def _evaluate_env_config(config_name, env, tasks, workers, backend, progress=False):
    with heuristic_env_scope(env):
        if progress:
            print(
                f"[{config_name}] {len(tasks)} paired tasks",
                file=sys.stderr,
                flush=True,
            )
        rows = map_parallel(_run_task, tasks, workers=workers, backend=backend)
    return {(row["suite"], row["seed"]): row for row in rows}


def _candidate_env_specs(args) -> list[tuple[str, dict[str, str]]]:
    specs = [(name, CONFIGS[name]) for name in args.candidate or []]
    names = args.candidate_env_name or []
    for index, env_file in enumerate(args.candidate_env_file or []):
        name = names[index] if index < len(names) else f"env-{Path(env_file).stem}"
        specs.append((name, load_env_overrides(env_file)))
    return specs


def _summarize_pair(incumbent, candidate, args):
    keys = sorted(set(incumbent) & set(candidate))
    diffs = [candidate[key]["delta"] - incumbent[key]["delta"] for key in keys]
    cand_deltas = [candidate[key]["delta"] for key in keys]
    inc_deltas = [incumbent[key]["delta"] for key in keys]
    candidate_busts = sum(1 for key in keys if candidate[key]["busted"])
    incumbent_busts = sum(1 for key in keys if incumbent[key]["busted"])
    candidate_errors = sum(candidate[key]["error_count"] for key in keys)
    incumbent_errors = sum(incumbent[key]["error_count"] for key in keys)
    mean_diff = statistics.mean(diffs) if diffs else 0.0
    median_diff = statistics.median(diffs) if diffs else 0.0
    p10_diff = _percentile(diffs, 0.10)
    win_rate = sum(1 for diff in diffs if diff > 0) / max(1, len(diffs))
    mean_ci_low, mean_ci_high = _bootstrap_mean_ci(
        diffs,
        samples=args.bootstrap_samples,
        ci_level=args.ci_level,
        seed=args.bootstrap_seed,
    )

    promotable = (
        len(keys) >= args.min_paired_runs
        and mean_diff >= args.min_mean_diff
        and mean_ci_low >= args.min_mean_ci_low
        and median_diff >= args.min_median_diff
        and p10_diff >= args.min_p10_diff
        and win_rate >= args.min_win_rate
        and candidate_errors <= incumbent_errors
        and candidate_busts <= incumbent_busts + args.max_extra_busts
    )

    suite_rows = []
    for suite in sorted({key[0] for key in keys}):
        suite_diffs = [candidate[key]["delta"] - incumbent[key]["delta"] for key in keys if key[0] == suite]
        suite_rows.append({
            "suite": suite,
            "runs": len(suite_diffs),
            "mean_diff": round(statistics.mean(suite_diffs), 2) if suite_diffs else 0.0,
            "median_diff": round(statistics.median(suite_diffs), 2) if suite_diffs else 0.0,
            "p10_diff": round(_percentile(suite_diffs, 0.10), 2),
            "win_rate": round(sum(1 for diff in suite_diffs if diff > 0) / max(1, len(suite_diffs)), 4),
        })

    return {
        "runs": len(keys),
        "mean_candidate_delta": round(statistics.mean(cand_deltas), 2) if cand_deltas else 0.0,
        "mean_incumbent_delta": round(statistics.mean(inc_deltas), 2) if inc_deltas else 0.0,
        "mean_diff": round(mean_diff, 2),
        "mean_diff_ci": [round(mean_ci_low, 2), round(mean_ci_high, 2)],
        "mean_diff_ci_level": args.ci_level,
        "median_diff": round(median_diff, 2),
        "p10_diff": round(p10_diff, 2),
        "win_rate": round(win_rate, 4),
        "min_paired_runs": args.min_paired_runs,
        "candidate_busts": candidate_busts,
        "incumbent_busts": incumbent_busts,
        "candidate_errors": candidate_errors,
        "incumbent_errors": incumbent_errors,
        "promotable": promotable,
        "suite_breakdown": suite_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Run paired-seed heuristic config promotion gate")
    parser.add_argument("--incumbent", choices=sorted(CONFIGS), default="baseline")
    parser.add_argument("--candidate", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--candidate-env-file", action="append", help="JSON or best_env.sh file to compare as a candidate")
    parser.add_argument("--candidate-env-name", action="append", help="Display name for the matching --candidate-env-file")
    parser.add_argument("--suite", choices=sorted(SUITES), action="append")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="candidate")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-start", type=int, default=9001)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--min-mean-diff", type=float, default=200.0)
    parser.add_argument("--min-mean-ci-low", type=float, default=0.0)
    parser.add_argument("--min-median-diff", type=float, default=0.0)
    parser.add_argument("--min-p10-diff", type=float, default=-1000.0)
    parser.add_argument("--min-win-rate", type=float, default=0.55)
    parser.add_argument("--min-paired-runs", type=int, default=100)
    parser.add_argument("--max-extra-busts", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    candidate_specs = _candidate_env_specs(args)
    if not candidate_specs:
        parser.error("at least one --candidate or --candidate-env-file is required")

    suites = args.suite or PRESETS[args.preset]["suites"]
    seeds = _parse_seeds(args)
    tasks = [(suite, seed, args.hands) for suite in suites for seed in seeds]

    incumbent = _evaluate_env_config(
        args.incumbent,
        CONFIGS[args.incumbent],
        tasks,
        workers=args.workers,
        backend=args.parallel_backend,
        progress=args.progress,
    )
    candidates = {}
    for name, env in candidate_specs:
        rows = _evaluate_env_config(
            name,
            env,
            tasks,
            workers=args.workers,
            backend=args.parallel_backend,
            progress=args.progress,
        )
        candidates[name] = _summarize_pair(incumbent, rows, args)

    report = {
        "incumbent": args.incumbent,
        "candidates": candidates,
        "preset": args.preset,
        "hands": args.hands,
        "seeds": seeds,
        "suites": suites,
        "promotion_thresholds": {
            "min_mean_diff": args.min_mean_diff,
            "min_mean_ci_low": args.min_mean_ci_low,
            "min_median_diff": args.min_median_diff,
            "min_p10_diff": args.min_p10_diff,
            "min_win_rate": args.min_win_rate,
            "min_paired_runs": args.min_paired_runs,
            "max_extra_busts": args.max_extra_busts,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "ci_level": args.ci_level,
        },
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"incumbent={args.incumbent} hands={args.hands} tasks={len(tasks)}")
    for name, summary in sorted(candidates.items(), key=lambda item: item[1]["mean_diff"], reverse=True):
        status = "PROMOTABLE" if summary["promotable"] else "hold"
        print(
            f"{name}: {status} mean_diff={summary['mean_diff']} "
            f"ci{int(args.ci_level * 100)}={summary['mean_diff_ci']} "
            f"median_diff={summary['median_diff']} p10={summary['p10_diff']} "
            f"win_rate={summary['win_rate']}"
        )


if __name__ == "__main__":
    main()
