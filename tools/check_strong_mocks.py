"""Statistical strength gate for benchmark-only strong mock opponents."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandbox.match import run_match
from tools.parallel import map_parallel


CANDIDATES = {
    "cfr_bucket": "bots/strong_mocks/cfr_bucket",
    "ensemble": "bots/strong_mocks/ensemble",
    "heuristic_rl_selector": "bots/strong_mocks/heuristic_rl_selector",
    "oracle_imitation": "bots/strong_mocks/oracle_imitation",
    "ppo_deep_policy": "bots/strong_mocks/ppo_deep_policy",
    "ppo_policy": "bots/strong_mocks/ppo_policy",
    "rollout_search": "bots/strong_mocks/rollout_search",
}

BASELINES = {
    "shark": "bots/shark/bot.py",
    "mathematician": "bots/mathematician/bot.py",
    "aggressor": "bots/aggressor/bot.py",
    "template": "bots/template/bot.py",
    "pot_odds": "bots/ref_bot_2/bot.py",
}


def _percentile(values: list[int], pct: float) -> float:
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


def build_tasks(candidates: list[str], baselines: list[str], seeds: list[int], hands: int, modes: list[str]) -> list[dict]:
    tasks = []
    for candidate in candidates:
        candidate_path = CANDIDATES[candidate]
        for seed in seeds:
            if "sixmax" in modes:
                bot_paths = {"candidate": candidate_path}
                for name in baselines[:5]:
                    bot_paths[f"base_{name}"] = BASELINES[name]
                tasks.append({
                    "candidate": candidate,
                    "suite": "sixmax_defaults",
                    "seed": seed,
                    "hands": hands,
                    "bot_paths": bot_paths,
                })
            if "heads-up" in modes:
                for baseline in baselines:
                    tasks.append({
                        "candidate": candidate,
                        "suite": f"hu_{baseline}",
                        "seed": seed,
                        "hands": hands,
                        "bot_paths": {
                            "candidate": candidate_path,
                            f"base_{baseline}": BASELINES[baseline],
                        },
                    })
    return tasks


def _run_task(task: dict) -> dict:
    result = run_match(
        match_id=f"strong_gate_{task['candidate']}_{task['suite']}_{task['seed']}",
        bot_paths=task["bot_paths"],
        n_hands=task["hands"],
        verbose=False,
        seed=task["seed"],
    )
    errors = result["bot_errors"].get("candidate", [])
    final_stack = result.get("final_stacks", {}).get("candidate")
    delta = int(result["chip_delta"].get("candidate", 0))
    return {
        "candidate": task["candidate"],
        "suite": task["suite"],
        "seed": task["seed"],
        "hands": task["hands"],
        "delta": delta,
        "busted": bool(final_stack is not None and final_stack <= 0),
        "error_count": len(errors),
        "errors": errors,
        "duration_s": result.get("duration_s", 0.0),
    }


def summarize(rows: list[dict]) -> dict:
    deltas = [int(row["delta"]) for row in rows]
    errors = sum(int(row["error_count"]) for row in rows)
    busts = sum(1 for row in rows if row["busted"])
    if not deltas:
        return {
            "runs": 0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "p10_delta": 0.0,
            "min_delta": 0,
            "win_rate": 0.0,
            "bust_count": 0,
            "error_count": 0,
        }
    return {
        "runs": len(deltas),
        "mean_delta": round(float(statistics.mean(deltas)), 2),
        "median_delta": round(float(statistics.median(deltas)), 2),
        "p10_delta": round(_percentile(deltas, 0.10), 2),
        "min_delta": min(deltas),
        "win_rate": round(sum(1 for delta in deltas if delta > 0) / len(deltas), 4),
        "bust_count": busts,
        "error_count": errors,
    }


def run(args) -> dict:
    candidates = args.candidate or sorted(CANDIDATES)
    baselines = args.baseline or sorted(BASELINES)
    modes = args.mode or ["sixmax", "heads-up"]
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    tasks = build_tasks(candidates, baselines, seeds, args.hands, modes)
    tasks_per_candidate = len(tasks) // max(1, len(candidates))
    if not args.allow_smoke:
        if args.hands < 400:
            raise SystemExit("--hands must be at least 400 unless --allow-smoke is set")
        if tasks_per_candidate < args.min_tasks_per_candidate:
            raise SystemExit(
                f"each candidate must have at least {args.min_tasks_per_candidate} tasks; "
                "increase --seed-count or use --allow-smoke only for wiring checks"
            )
    rows = map_parallel(_run_task, tasks, workers=args.workers, backend=args.parallel_backend)
    by_candidate = {}
    for candidate in candidates:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        suite_breakdown = {
            suite: summarize([row for row in candidate_rows if row["suite"] == suite])
            for suite in sorted({row["suite"] for row in candidate_rows})
        }
        summary = summarize(candidate_rows)
        passed = (
            summary["runs"] >= args.min_tasks_per_candidate
            and summary["mean_delta"] >= args.min_mean_delta
            and summary["win_rate"] >= args.min_win_rate
            and summary["error_count"] == 0
        )
        by_candidate[candidate] = {
            "summary": summary,
            "suite_breakdown": suite_breakdown,
            "passed": bool(passed),
        }
    return {
        "hands": args.hands,
        "seed_count": args.seed_count,
        "modes": modes,
        "baselines": baselines,
        "min_mean_delta": args.min_mean_delta,
        "min_win_rate": args.min_win_rate,
        "min_tasks_per_candidate": args.min_tasks_per_candidate,
        "tasks": len(tasks),
        "candidates": by_candidate,
        "passed": all(item["passed"] for item in by_candidate.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strong mocks against default/reference bots")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), action="append")
    parser.add_argument("--baseline", choices=sorted(BASELINES), action="append")
    parser.add_argument("--mode", choices=["sixmax", "heads-up"], action="append")
    parser.add_argument("--seed-count", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=42001)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--min-mean-delta", type=float, default=500.0)
    parser.add_argument("--min-win-rate", type=float, default=0.55)
    parser.add_argument("--min-tasks-per-candidate", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else report)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
