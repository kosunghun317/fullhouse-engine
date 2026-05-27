"""League-style trainer for benchmark-only strong mock opponents.

The league orchestrator separates training pools from held-out evaluation
pools, archives candidate artifacts, and promotes only when a candidate beats
the incumbent on held-out fast matches.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.strong_mocks.train_real_policy import train as train_real_policy
from training.fast_match import run_fast_matches_parallel


BOT_ROOT = ROOT / "bots"
STRONG_ROOT = BOT_ROOT / "strong_mocks"


KIND_CONFIG = {
    "ppo": {
        "source_bot": STRONG_ROOT / "ppo_policy" / "bot.py",
        "incumbent": STRONG_ROOT / "ppo_policy",
        "output": STRONG_ROOT / "ppo_policy" / "data" / "policy.npz",
        "hidden": 128,
        "bucket_count": 32768,
        "abstraction": "feature",
        "bucket_update": "policy-gradient",
        "learning_rate": 0.002,
        "entropy_coef": 0.002,
        "epochs": 3,
        "batch_size": 4096,
        "temperature": 0.82,
    },
    "bucket": {
        "source_bot": STRONG_ROOT / "cfr_bucket" / "bot.py",
        "incumbent": STRONG_ROOT / "cfr_bucket",
        "output": STRONG_ROOT / "cfr_bucket" / "data" / "policy.npz",
        "hidden": 128,
        "bucket_count": 32768,
        "abstraction": "cfr-pokerbot",
        "bucket_update": "cfr-plus",
        "learning_rate": 0.004,
        "entropy_coef": 0.0,
        "epochs": 1,
        "batch_size": 4096,
        "temperature": 0.80,
    },
}


STAGES = {
    "oracle_bootstrap": {
        "train_pool": "oracle",
        "eval_pools": ["public_holdout"],
        "generations": {"ppo": 8, "bucket": 6},
        "matches": {"ppo": 32, "bucket": 48},
        "hands": 100,
        "snapshot_prob": 0.20,
    },
    "public_adversarial": {
        "train_pool": "adversarial",
        "eval_pools": ["public_holdout", "mock_holdout"],
        "generations": {"ppo": 14, "bucket": 12},
        "matches": {"ppo": 64, "bucket": 80},
        "hands": 140,
        "snapshot_prob": 0.30,
    },
    "league_mixed": {
        "train_pool": "mixed",
        "eval_pools": ["public_holdout", "mock_holdout", "strong_holdout"],
        "generations": {"ppo": 14, "bucket": 12},
        "matches": {"ppo": 80, "bucket": 96},
        "hands": 160,
        "snapshot_prob": 0.45,
    },
}


EVAL_POOLS = {
    "public_holdout": [
        BOT_ROOT / "heuristic",
        BOT_ROOT / "shark",
        BOT_ROOT / "mathematician",
        BOT_ROOT / "aggressor",
        BOT_ROOT / "template",
        BOT_ROOT / "ref_bot_2",
    ],
    "mock_holdout": [
        BOT_ROOT / "mock_competitors" / "equity_mc",
        BOT_ROOT / "mock_competitors" / "equity_pressure",
        BOT_ROOT / "mock_competitors" / "bucket_overbet",
        BOT_ROOT / "mock_competitors" / "opponent_modeler",
        BOT_ROOT / "mock_competitors" / "anti_heuristic",
        BOT_ROOT / "benchmarks" / "threshold_caller",
    ],
    "strong_holdout": [
        BOT_ROOT / "strong_mocks" / "oracle_imitation",
        BOT_ROOT / "strong_mocks" / "rollout_search",
        BOT_ROOT / "strong_mocks" / "ensemble",
        BOT_ROOT / "mock_competitors" / "copy_shark_plus",
        BOT_ROOT / "mock_competitors" / "cbet_reg",
        BOT_ROOT / "benchmarks" / "jammer",
    ],
}


@dataclass
class EvalSummary:
    runs: int
    mean_delta: float
    median_delta: float
    min_delta: int
    max_delta: int
    stdev_delta: float
    positive_runs: int
    nonnegative_runs: int
    bust_count: int
    error_count: int
    score: float


def _scale_int(value: int, scale: float, minimum: int = 1) -> int:
    return max(minimum, int(math.ceil(value * scale)))


def _stable_offset(*parts: str) -> int:
    text = ":".join(parts)
    return sum((index + 1) * ord(ch) for index, ch in enumerate(text)) % 100000


def _candidate_dir(run_root: Path, stage: str, kind: str) -> Path:
    return run_root / "candidates" / f"{stage}_{kind}"


def prepare_candidate_bot(kind: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    data_dir = target_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(KIND_CONFIG[kind]["source_bot"], target_dir / "bot.py")
    return target_dir


def _bot_label(path: Path, index: int) -> str:
    parent = path.parent.name
    name = path.name
    label = name if name != "bot.py" else parent
    label = "".join(ch if ch.isalnum() else "_" for ch in label)
    return f"opp_{index}_{label}"


def build_eval_tasks(
    candidate_path: Path,
    pool_names: list[str],
    seeds: list[int],
    hands: int,
    players: int,
    match_prefix: str,
) -> list[dict]:
    tasks = []
    for pool_name in pool_names:
        opponents = EVAL_POOLS[pool_name]
        seats = max(2, min(players, len(opponents) + 1))
        needed = seats - 1
        for seed in seeds:
            start = seed % len(opponents)
            selected = [opponents[(start + offset) % len(opponents)] for offset in range(needed)]
            bot_paths = {"candidate": str(candidate_path)}
            for index, path in enumerate(selected):
                bot_paths[_bot_label(path, index)] = str(path)
            tasks.append({
                "match_id": f"{match_prefix}_{pool_name}_{seed}",
                "pool": pool_name,
                "bot_paths": bot_paths,
                "n_hands": hands,
                "seed": seed,
            })
    return tasks


def summarize_eval(
    results: list[dict],
    risk_weight: float = 0.35,
    min_weight: float = 0.10,
    bust_penalty: float = 8000.0,
    error_penalty: float = 20000.0,
) -> dict:
    deltas = [int(result["chip_delta"].get("candidate", 0)) for result in results]
    busts = sum(1 for result in results if int(result["final_stacks"].get("candidate", 0)) <= 0)
    errors = [err for result in results for err in result["bot_errors"].get("candidate", [])]
    runs = len(results)
    if not deltas:
        return EvalSummary(0, 0.0, 0.0, 0, 0, 0.0, 0, 0, 0, 0, -999999.0).__dict__
    bust_rate = busts / max(1, runs)
    error_rate = len(errors) / max(1, runs)
    mean_delta = float(statistics.mean(deltas))
    stdev_delta = float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0
    min_delta = min(deltas)
    score = mean_delta - risk_weight * stdev_delta + min_weight * min_delta
    score -= bust_penalty * bust_rate + error_penalty * error_rate
    return EvalSummary(
        runs=runs,
        mean_delta=round(mean_delta, 2),
        median_delta=round(float(statistics.median(deltas)), 2),
        min_delta=min_delta,
        max_delta=max(deltas),
        stdev_delta=round(stdev_delta, 2),
        positive_runs=sum(1 for delta in deltas if delta > 0),
        nonnegative_runs=sum(1 for delta in deltas if delta >= 0),
        bust_count=busts,
        error_count=len(errors),
        score=round(score, 2),
    ).__dict__


def should_promote(candidate: dict, incumbent: dict, margin: float) -> bool:
    if candidate["error_count"] > 0:
        return False
    return float(candidate["score"]) >= float(incumbent["score"]) + margin


def _train_args(args, stage_name: str, stage: dict, kind: str, output: Path, extra_bot_paths: list[str]) -> Namespace:
    cfg = KIND_CONFIG[kind]
    generation_scale = args.generation_scale
    match_scale = args.match_scale
    hand_scale = args.hand_scale
    return Namespace(
        kind=kind,
        generations=_scale_int(stage["generations"][kind], generation_scale),
        matches_per_generation=_scale_int(stage["matches"][kind], match_scale),
        hands=_scale_int(stage["hands"], hand_scale, minimum=4),
        players=args.players,
        train_seats=args.train_seats,
        hidden=cfg["hidden"],
        bucket_count=cfg["bucket_count"],
        abstraction=cfg["abstraction"],
        bucket_update=cfg["bucket_update"],
        learning_rate=cfg["learning_rate"],
        entropy_coef=cfg["entropy_coef"],
        ppo_clip_ratio=args.ppo_clip_ratio,
        ppo_value_coef=args.ppo_value_coef,
        max_grad_norm=args.max_grad_norm,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        feature_norm_momentum=args.feature_norm_momentum,
        replay_generations=args.replay_generations,
        replay_max_decisions=args.replay_max_decisions,
        temperature=cfg["temperature"],
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        opponent_pool=stage["train_pool"],
        extra_bot_path=extra_bot_paths,
        snapshot_interval=args.snapshot_interval,
        max_snapshots=args.max_snapshots,
        snapshot_prob=stage["snapshot_prob"],
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        seed=args.seed + _stable_offset(stage_name, kind),
        output=str(output),
        log_jsonl=str(output.parent.parent / f"{stage_name}_{kind}_training.jsonl"),
        resume=args.resume,
        export_best=True,
        selection_warmup=args.selection_warmup,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        min_export_mean_delta=args.min_export_mean_delta,
        progress=args.progress,
        json=True,
    )


def _evaluate_bot(path: Path, stage_name: str, kind: str, label: str, stage: dict, args) -> dict:
    seeds = list(range(args.seed_start, args.seed_start + args.eval_seeds))
    tasks = build_eval_tasks(
        candidate_path=path,
        pool_names=stage["eval_pools"],
        seeds=seeds,
        hands=args.eval_hands,
        players=args.players,
        match_prefix=f"league_{stage_name}_{kind}",
    )
    results = run_fast_matches_parallel(tasks, workers=args.workers, backend=args.parallel_backend)
    summary = summarize_eval(
        results,
        risk_weight=args.risk_weight,
        min_weight=args.min_weight,
        bust_penalty=args.bust_penalty,
        error_penalty=args.error_penalty,
    )
    by_pool = {}
    for pool_name in stage["eval_pools"]:
        by_pool[pool_name] = summarize_eval(
            [result for result, task in zip(results, tasks) if task["pool"] == pool_name],
            risk_weight=args.risk_weight,
            min_weight=args.min_weight,
            bust_penalty=args.bust_penalty,
            error_penalty=args.error_penalty,
        )
    return {"summary": summary, "by_pool": by_pool}


def _promote_candidate(kind: str, candidate_dir: Path) -> None:
    source = candidate_dir / "data" / "policy.npz"
    target = KIND_CONFIG[kind]["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def run_league(args) -> dict:
    run_root = Path(args.result_root) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    stages = args.stage or list(STAGES)
    kinds = args.kind or ["ppo", "bucket"]
    league_snapshots: list[str] = []
    reports = []

    for stage_name in stages:
        stage = STAGES[stage_name]
        for kind in kinds:
            candidate_dir = prepare_candidate_bot(kind, _candidate_dir(run_root, stage_name, kind))
            output = candidate_dir / "data" / "policy.npz"
            train_report = train_real_policy(
                _train_args(args, stage_name, stage, kind, output, league_snapshots)
            )
            candidate_eval = _evaluate_bot(candidate_dir, stage_name, kind, "candidate", stage, args)
            incumbent_eval = _evaluate_bot(KIND_CONFIG[kind]["incumbent"], stage_name, kind, "incumbent", stage, args)
            promote = should_promote(candidate_eval["summary"], incumbent_eval["summary"], args.promote_margin)
            if promote and args.promote:
                _promote_candidate(kind, candidate_dir)
                league_snapshots.append(str(candidate_dir))
            elif promote:
                league_snapshots.append(str(candidate_dir))

            report = {
                "stage": stage_name,
                "kind": kind,
                "train_pool": stage["train_pool"],
                "eval_pools": stage["eval_pools"],
                "candidate_dir": str(candidate_dir),
                "train": train_report,
                "candidate_eval": candidate_eval,
                "incumbent_eval": incumbent_eval,
                "promote": bool(promote),
                "promoted_to_repo": bool(promote and args.promote),
                "league_snapshot_count": len(league_snapshots),
            }
            reports.append(report)
            with open(run_root / "league_report.json", "w", encoding="utf-8") as handle:
                json.dump({"run_id": args.run_id, "reports": reports}, handle, indent=2, sort_keys=True)
            if args.progress:
                print(json.dumps({
                    "stage": stage_name,
                    "kind": kind,
                    "candidate_score": candidate_eval["summary"]["score"],
                    "incumbent_score": incumbent_eval["summary"]["score"],
                    "promote": promote,
                }), file=sys.stderr, flush=True)

    return {"run_id": args.run_id, "run_root": str(run_root), "reports": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run league-style staged training for strong mock opponents")
    parser.add_argument("--run-id", default="league-local")
    parser.add_argument("--result-root", default="/private/tmp/fullhouse_league_training")
    parser.add_argument("--kind", choices=sorted(KIND_CONFIG), action="append")
    parser.add_argument("--stage", choices=sorted(STAGES), action="append")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--train-seats", type=int, default=2)
    parser.add_argument("--generation-scale", type=float, default=1.0)
    parser.add_argument("--match-scale", type=float, default=1.0)
    parser.add_argument("--hand-scale", type=float, default=1.0)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--eval-hands", type=int, default=160)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--seed-start", type=int, default=11001)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.20)
    parser.add_argument("--ppo-value-coef", type=float, default=0.35)
    parser.add_argument("--max-grad-norm", type=float, default=0.75)
    parser.add_argument("--feature-norm-momentum", type=float, default=0.08)
    parser.add_argument("--replay-generations", type=int, default=4)
    parser.add_argument("--replay-max-decisions", type=int, default=24000)
    parser.add_argument("--snapshot-interval", type=int, default=2)
    parser.add_argument("--max-snapshots", type=int, default=8)
    parser.add_argument("--selection-warmup", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--min-export-mean-delta", type=float, default=-1_000_000_000.0)
    parser.add_argument("--promote-margin", type=float, default=250.0)
    parser.add_argument("--risk-weight", type=float, default=0.35)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--bust-penalty", type=float, default=8000.0)
    parser.add_argument("--error-penalty", type=float, default=20000.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.players < 2:
        raise SystemExit("--players must be at least 2")
    if not 1 <= args.train_seats <= args.players:
        raise SystemExit("--train-seats must be between 1 and --players")
    result = run_league(args)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
