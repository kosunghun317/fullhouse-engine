"""End-to-end coevolution training for heuristic configs and PPO strong mocks.

This is local-only infrastructure. It alternates between:

1. training unrestricted PPO candidate opponents against the latest selected
   heuristic wrapper, and
2. evolving heuristic env-config wrappers against the latest selected PPO.

The submitted `bots/heuristic/bot.py` remains unchanged unless a human later
promotes a discovered env configuration into defaults.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.plot_training_progress import render_svg
from tools.strong_mocks.self_train_heuristic import (
    evaluate_generation,
    next_population,
    seed_population,
    write_variant,
)
from tools.strong_mocks.train_imitation import train as train_imitation
from tools.strong_mocks.train_real_policy import train as train_real_policy
from training.fast_match import run_fast_matches_parallel


BOT_ROOT = ROOT / "bots"
DEFAULT_RESULT_ROOT = Path("/private/tmp/fullhouse_coevolution")
PPO_SOURCE = BOT_ROOT / "strong_mocks" / "ppo_policy"
HEURISTIC_SOURCE = BOT_ROOT / "heuristic" / "bot.py"

PPO_ARMS = {
    "stable": {
        "learning_rate": 0.0015,
        "entropy_coef": 0.0025,
        "temperature": 0.82,
        "ppo_clip_ratio": 0.20,
        "ppo_value_coef": 0.35,
    },
    "explore": {
        "learning_rate": 0.0022,
        "entropy_coef": 0.0060,
        "temperature": 0.92,
        "ppo_clip_ratio": 0.22,
        "ppo_value_coef": 0.25,
    },
    "conservative": {
        "learning_rate": 0.0010,
        "entropy_coef": 0.0015,
        "temperature": 0.78,
        "ppo_clip_ratio": 0.16,
        "ppo_value_coef": 0.45,
    },
}

PPO_EVAL_FILLERS = [
    BOT_ROOT / "shark",
    BOT_ROOT / "mathematician",
    BOT_ROOT / "strong_mocks" / "rollout_search",
    BOT_ROOT / "mock_competitors" / "equity_pressure",
]

HEURISTIC_EVAL_FILLERS = [
    BOT_ROOT / "strong_mocks" / "rollout_search",
    BOT_ROOT / "strong_mocks" / "ensemble",
    BOT_ROOT / "mock_competitors" / "bucket_overbet",
    BOT_ROOT / "benchmarks" / "threshold_caller",
]


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _safe_label(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text) or "item"


def _copytree_clean(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def prepare_ppo_candidate(run_root: Path, cycle: int, arm_name: str, latest_ppo: Path, args) -> Path:
    target = run_root / "ppo_candidates" / f"cycle_{cycle:03d}_{_safe_label(arm_name)}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PPO_SOURCE / "bot.py", target / "bot.py")
    data_dir = target / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.ppo_init == "latest":
        source_policy = latest_ppo / "data" / "policy.npz"
        if source_policy.is_file():
            shutil.copyfile(source_policy, data_dir / "policy.npz")
    elif args.ppo_init == "oracle":
        train_imitation(
            samples=args.ppo_init_samples,
            seed=args.seed + cycle * 1000 + sum(ord(ch) for ch in arm_name),
            style="pressure",
            output=data_dir / "policy.npz",
            hidden=args.ppo_hidden,
        )
    return target


def _summary_for(results: list[dict], bot_id: str) -> dict:
    deltas = [int(result["chip_delta"].get(bot_id, 0)) for result in results]
    busts = sum(1 for result in results if int(result["final_stacks"].get(bot_id, 0)) <= 0)
    errors = [err for result in results for err in result["bot_errors"].get(bot_id, [])]
    if not deltas:
        return {
            "runs": 0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "min_delta": 0,
            "max_delta": 0,
            "stdev_delta": 0.0,
            "positive_runs": 0,
            "bust_count": 0,
            "error_count": 0,
            "score": -999999.0,
        }
    mean_delta = float(statistics.mean(deltas))
    stdev_delta = float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0
    score = mean_delta - 0.35 * stdev_delta + 0.10 * min(deltas)
    score -= 8000.0 * (busts / max(1, len(deltas)))
    score -= 20000.0 * (len(errors) / max(1, len(deltas)))
    return {
        "runs": len(deltas),
        "mean_delta": round(mean_delta, 2),
        "median_delta": round(float(statistics.median(deltas)), 2),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "stdev_delta": round(stdev_delta, 2),
        "positive_runs": sum(1 for delta in deltas if delta > 0),
        "bust_count": busts,
        "error_count": len(errors),
        "score": round(score, 2),
    }


def evaluate_path(
    candidate_path: Path,
    candidate_id: str,
    opponents: list[Path],
    seeds: list[int],
    hands: int,
    workers: int,
    backend: str,
    match_prefix: str,
) -> dict:
    tasks = []
    seats = min(6, 1 + len(opponents))
    for seed in seeds:
        selected = [opponents[(seed + offset) % len(opponents)] for offset in range(seats - 1)]
        bot_paths = {candidate_id: str(candidate_path)}
        for index, path in enumerate(selected):
            bot_paths[f"opp_{index}_{path.name}"] = str(path)
        tasks.append({
            "match_id": f"{match_prefix}_{seed}",
            "bot_paths": bot_paths,
            "n_hands": hands,
            "seed": seed,
        })
    results = run_fast_matches_parallel(tasks, workers=workers, backend=backend)
    return {"summary": _summary_for(results, candidate_id), "runs": results}


def train_ppo_candidate(
    candidate_dir: Path,
    cycle: int,
    arm_name: str,
    arm: dict,
    latest_heuristic: Path,
    latest_ppo: Path,
    args,
) -> dict:
    train_args = Namespace(
        kind="ppo",
        generations=args.ppo_generations,
        matches_per_generation=args.ppo_matches_per_generation,
        hands=args.ppo_hands,
        players=6,
        train_seats=2,
        hidden=args.ppo_hidden,
        bucket_count=32768,
        abstraction="feature",
        bucket_update="policy-gradient",
        learning_rate=arm["learning_rate"],
        entropy_coef=arm["entropy_coef"],
        ppo_clip_ratio=arm["ppo_clip_ratio"],
        ppo_value_coef=arm["ppo_value_coef"],
        max_grad_norm=args.ppo_max_grad_norm,
        epochs=args.ppo_epochs,
        batch_size=args.ppo_batch_size,
        feature_norm_momentum=args.ppo_feature_norm_momentum,
        replay_generations=args.ppo_replay_generations,
        replay_max_decisions=args.ppo_replay_max_decisions,
        temperature=arm["temperature"],
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        opponent_pool=args.opponent_pool,
        extra_bot_path=[str(latest_heuristic), str(latest_ppo)],
        snapshot_interval=2,
        max_snapshots=8,
        snapshot_prob=0.35,
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        seed=args.seed + cycle * 1000 + sum(ord(ch) for ch in arm_name),
        output=str(candidate_dir / "data" / "policy.npz"),
        log_jsonl=str(candidate_dir / "training.jsonl"),
        resume=True,
        export_best=True,
        selection_warmup=args.selection_warmup,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        min_export_mean_delta=-1_000_000_000.0,
        progress=args.progress,
        json=True,
    )
    return train_real_policy(train_args)


def _heuristic_args(args) -> Namespace:
    return Namespace(
        matches_per_generation=args.heuristic_matches_per_generation,
        hands=args.heuristic_hands,
        min_heuristics=2,
        max_heuristics=3,
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        seed=args.seed,
    )


def run_coevolution(args) -> dict:
    run_root = Path(args.result_root) / args.run_id
    generated_root = run_root / "heuristic_generated"
    metrics_path = run_root / "metrics.jsonl"
    plot_path = run_root / "ev_progress.svg"
    run_root.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()

    rng = random.Random(args.seed)
    latest_ppo = PPO_SOURCE
    latest_heuristic = HEURISTIC_SOURCE
    population = seed_population(args.heuristic_population, rng)
    arm_names = args.ppo_arm or ["stable", "explore", "conservative"]
    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds))
    cycles = []

    for cycle in range(args.cycles):
        ppo_candidates = []
        incumbent_eval = evaluate_path(
            Path(latest_ppo),
            "candidate",
            [Path(latest_heuristic), *PPO_EVAL_FILLERS],
            eval_seeds,
            args.eval_hands,
            args.workers,
            args.parallel_backend,
            f"coevolve_ppo_incumbent_c{cycle}",
        )
        ppo_candidates.append({
            "arm": "incumbent",
            "path": str(latest_ppo),
            "train": None,
            "eval": incumbent_eval["summary"],
        })
        _append_jsonl(metrics_path, {
            "cycle": cycle,
            "series": "ppo_incumbent",
            "phase": "ppo_eval",
            "mean_delta": incumbent_eval["summary"]["mean_delta"],
            "score": incumbent_eval["summary"]["score"],
        })
        for arm_name in arm_names:
            arm = PPO_ARMS[arm_name]
            candidate_dir = prepare_ppo_candidate(run_root, cycle, arm_name, Path(latest_ppo), args)
            train_report = train_ppo_candidate(candidate_dir, cycle, arm_name, arm, Path(latest_heuristic), Path(latest_ppo), args)
            eval_report = evaluate_path(
                candidate_dir,
                "candidate",
                [Path(latest_heuristic), *PPO_EVAL_FILLERS],
                eval_seeds,
                args.eval_hands,
                args.workers,
                args.parallel_backend,
                f"coevolve_ppo_c{cycle}_{arm_name}",
            )
            item = {
                "arm": arm_name,
                "path": str(candidate_dir),
                "train": train_report,
                "eval": eval_report["summary"],
            }
            ppo_candidates.append(item)
            _append_jsonl(metrics_path, {
                "cycle": cycle,
                "series": f"ppo_{arm_name}",
                "phase": "ppo_eval",
                "mean_delta": eval_report["summary"]["mean_delta"],
                "score": eval_report["summary"]["score"],
            })
        ppo_candidates.sort(key=lambda item: item["eval"]["score"], reverse=True)
        latest_ppo = Path(ppo_candidates[0]["path"])
        _append_jsonl(metrics_path, {
            "cycle": cycle,
            "series": "ppo_selected",
            "phase": "ppo_selected",
            "mean_delta": ppo_candidates[0]["eval"]["mean_delta"],
            "score": ppo_candidates[0]["eval"]["score"],
        })

        hargs = _heuristic_args(args)
        heuristic_report = evaluate_generation(
            generated_root,
            args.run_id,
            cycle,
            population,
            hargs,
            rng,
            extra_opponents=[("latest_ppo", str(latest_ppo))],
        )
        _json_dump(run_root / f"heuristic_generation_{cycle:03d}.json", heuristic_report)
        best_heuristic = heuristic_report["ranked"][0]
        latest_heuristic = Path(write_variant(
            generated_root,
            args.run_id,
            cycle,
            {
                "name": f"selected_cycle_{cycle}_{best_heuristic['name']}",
                "env": best_heuristic["env"],
                "parent": best_heuristic["name"],
            },
        ))
        heuristic_eval = evaluate_path(
            latest_heuristic,
            "candidate",
            [Path(latest_ppo), *HEURISTIC_EVAL_FILLERS],
            eval_seeds,
            args.eval_hands,
            args.workers,
            args.parallel_backend,
            f"coevolve_heuristic_c{cycle}",
        )
        _append_jsonl(metrics_path, {
            "cycle": cycle,
            "series": "heuristic_selected",
            "phase": "heuristic_eval",
            "mean_delta": heuristic_eval["summary"]["mean_delta"],
            "score": heuristic_eval["summary"]["score"],
        })
        population = next_population(heuristic_report["ranked"], args.heuristic_population, args.heuristic_elite, rng)
        cycle_report = {
            "cycle": cycle,
            "selected_ppo": ppo_candidates[0],
            "selected_heuristic": {
                "path": str(latest_heuristic),
                "ranked": best_heuristic,
                "eval": heuristic_eval["summary"],
            },
            "ppo_candidates": ppo_candidates,
        }
        cycles.append(cycle_report)
        _json_dump(run_root / "summary.json", {
            "run_id": args.run_id,
            "run_root": str(run_root),
            "latest_ppo": str(latest_ppo),
            "latest_heuristic": str(latest_heuristic),
            "cycles": cycles,
        })
        if args.progress:
            print(json.dumps({
                "cycle": cycle,
                "selected_ppo": ppo_candidates[0]["arm"],
                "ppo_score": ppo_candidates[0]["eval"]["score"],
                "heuristic_score": heuristic_eval["summary"]["score"],
                "latest_ppo": str(latest_ppo),
                "latest_heuristic": str(latest_heuristic),
            }), file=sys.stderr, flush=True)

    plot_report = render_svg(metrics_path, plot_path)
    result = {
        "run_id": args.run_id,
        "run_root": str(run_root),
        "metrics": str(metrics_path),
        "plot": str(plot_path),
        "plot_report": plot_report,
        "latest_ppo": str(latest_ppo),
        "latest_heuristic": str(latest_heuristic),
        "cycles": cycles,
    }
    _json_dump(run_root / "summary.json", result)
    if args.promote_ppo:
        target = PPO_SOURCE / "data" / "policy.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(latest_ppo) / "data" / "policy.npz", target)
        result["promoted_ppo_to"] = str(target)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternating PPO/heuristic coevolution training")
    parser.add_argument("--run-id", default="coevolve-local")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--ppo-arm", choices=sorted(PPO_ARMS), action="append")
    parser.add_argument("--ppo-generations", type=int, default=4)
    parser.add_argument("--ppo-matches-per-generation", type=int, default=24)
    parser.add_argument("--ppo-hands", type=int, default=100)
    parser.add_argument("--ppo-hidden", type=int, default=128)
    parser.add_argument("--ppo-init", choices=["oracle", "latest", "fresh"], default="oracle")
    parser.add_argument("--ppo-init-samples", type=int, default=50000)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-batch-size", type=int, default=4096)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=0.75)
    parser.add_argument("--ppo-feature-norm-momentum", type=float, default=0.08)
    parser.add_argument("--ppo-replay-generations", type=int, default=4)
    parser.add_argument("--ppo-replay-max-decisions", type=int, default=24000)
    parser.add_argument("--heuristic-population", type=int, default=10)
    parser.add_argument("--heuristic-elite", type=int, default=3)
    parser.add_argument("--heuristic-matches-per-generation", type=int, default=12)
    parser.add_argument("--heuristic-hands", type=int, default=160)
    parser.add_argument("--eval-seeds", type=int, default=4)
    parser.add_argument("--eval-seed-start", type=int, default=15001)
    parser.add_argument("--eval-hands", type=int, default=140)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--opponent-pool", choices=["oracle", "fast", "mixed", "adversarial"], default="adversarial")
    parser.add_argument("--selection-warmup", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--promote-ppo", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_coevolution(args)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
