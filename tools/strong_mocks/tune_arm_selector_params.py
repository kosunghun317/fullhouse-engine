"""CEM/racing tuner for heuristic expert-arm selector parameters.

This script is intentionally separate from smoke tests. By default it refuses
small candidate evaluations; pass --allow-smoke only for wiring checks.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.strong_mocks.arm_selector_params import (  # noqa: E402
    PARAM_NAMES,
    coerce_params,
    params_from_vector,
    save_params_npz,
    vector_from_params,
)
from training.fast_match import run_fast_matches_parallel  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "arm_selector_param_tuning"
DEFAULT_BOT = ROOT / "bots" / "strong_mocks" / "heuristic_rl_selector"
DEFAULT_POLICY = DEFAULT_BOT / "data" / "policy.npz"
DEFAULT_STAGE_SPEC = "16:400:0.35,64:400:1.0"
BIG_BLIND = 100


@dataclass(frozen=True)
class Stage:
    matches: int
    hands: int
    keep_frac: float


@dataclass
class CandidateResult:
    candidate_id: str
    vector: np.ndarray
    params_path: str
    score: float
    mean_bb100: float
    cvar20_bb100: float
    stdev_bb100: float
    bust_rate: float
    error_rate: float
    matches: int
    hands_total: int
    deltas: list[int]


SCENARIOS: tuple[tuple[str, tuple[Path, ...]], ...] = (
    (
        "public_mix",
        (
            ROOT / "bots" / "template",
            ROOT / "bots" / "aggressor",
            ROOT / "bots" / "mathematician",
            ROOT / "bots" / "shark",
            ROOT / "bots" / "benchmarks" / "threshold_caller",
        ),
    ),
    (
        "strong_mix",
        (
            ROOT / "bots" / "strong_mocks" / "rollout_search",
            ROOT / "bots" / "strong_mocks" / "ppo_deep_policy",
            ROOT / "bots" / "strong_mocks" / "cfr_bucket",
            ROOT / "bots" / "strong_mocks" / "ensemble",
            ROOT / "bots" / "mock_competitors" / "equity_pressure",
        ),
    ),
    (
        "weak_field",
        (
            ROOT / "bots" / "benchmarks" / "station",
            ROOT / "bots" / "benchmarks" / "nit",
            ROOT / "bots" / "benchmarks" / "threshold_caller",
            ROOT / "bots" / "mock_competitors" / "bucket_overbet",
            ROOT / "bots" / "aggressor",
        ),
    ),
)


def _parse_stages(spec: str) -> list[Stage]:
    stages = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            matches, hands, keep = raw.split(":")
            stages.append(Stage(max(1, int(matches)), max(1, int(hands)), min(1.0, max(0.05, float(keep)))))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("stage spec must be 'matches:hands:keep[,..]'") from exc
    if not stages:
        raise argparse.ArgumentTypeError("at least one stage is required")
    return stages


def _available_scenarios() -> list[tuple[str, list[Path]]]:
    scenarios = []
    for name, paths in SCENARIOS:
        existing = [path for path in paths if (path / "bot.py").is_file()]
        if len(existing) >= 5:
            scenarios.append((name, existing[:5]))
    if not scenarios:
        raise RuntimeError("no complete evaluation scenarios are available")
    return scenarios


def _write_bot(bot_dir: Path, params: dict[str, float], policy_path: Path | None) -> None:
    data_dir = bot_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_BOT / "bot.py", bot_dir / "bot.py")
    if policy_path is not None and policy_path.is_file():
        shutil.copyfile(policy_path, data_dir / "policy.npz")
    elif DEFAULT_POLICY.is_file():
        shutil.copyfile(DEFAULT_POLICY, data_dir / "policy.npz")
    save_params_npz(data_dir / "params.npz", params)


def _evaluate_bot_dir(
    bot_dir: Path,
    stage: Stage,
    seed_base: int,
    workers: int,
    parallel_backend: str,
) -> dict:
    scenarios = _available_scenarios()
    tasks = []
    for index in range(stage.matches):
        scenario_name, opponents = scenarios[index % len(scenarios)]
        seed = seed_base + index
        bot_paths = {"candidate": str(bot_dir)}
        for opp_index, path in enumerate(opponents):
            bot_paths[f"{path.name}_{opp_index}"] = str(path)
        tasks.append({
            "match_id": f"{bot_dir.name}_{scenario_name}_{index:04d}",
            "bot_paths": bot_paths,
            "n_hands": stage.hands,
            "seed": seed,
        })
    results = run_fast_matches_parallel(tasks, workers=workers, backend=parallel_backend)
    deltas = [int(result["chip_delta"].get("candidate", 0)) for result in results]
    stacks = [int(result["final_stacks"].get("candidate", 0)) for result in results]
    errors = sum(len(result["bot_errors"].get("candidate", [])) for result in results)
    bb100 = np.asarray([(delta / BIG_BLIND) * (100.0 / max(1, stage.hands)) for delta in deltas], dtype=np.float64)
    worst_n = max(1, int(math.ceil(0.20 * len(bb100))))
    mean = float(np.mean(bb100)) if bb100.size else 0.0
    stdev = float(np.std(bb100)) if bb100.size else 0.0
    cvar20 = float(np.sort(bb100)[:worst_n].mean()) if bb100.size else 0.0
    bust_rate = sum(1 for stack in stacks if stack <= 0) / max(1, len(stacks))
    error_rate = errors / max(1, len(results))
    score = mean + 0.25 * cvar20 - 0.15 * stdev - 50.0 * bust_rate - 50.0 * error_rate
    return {
        "score": float(score),
        "mean_bb100": mean,
        "cvar20_bb100": cvar20,
        "stdev_bb100": stdev,
        "bust_rate": float(bust_rate),
        "error_rate": float(error_rate),
        "matches": len(results),
        "hands_total": int(len(results) * stage.hands),
        "deltas": deltas,
    }


def _evaluate_candidates(
    run_dir: Path,
    generation: int,
    vectors: list[np.ndarray],
    stage: Stage,
    seed_base: int,
    policy_path: Path | None,
    workers: int,
    parallel_backend: str,
) -> list[CandidateResult]:
    rows = []
    for index, vector in enumerate(vectors):
        candidate_id = f"g{generation:03d}_c{index:03d}"
        bot_dir = run_dir / "candidates" / candidate_id
        params = params_from_vector(vector, normalized=True)
        _write_bot(bot_dir, params, policy_path)
        metrics = _evaluate_bot_dir(bot_dir, stage, seed_base + index * 10_000, workers, parallel_backend)
        rows.append(CandidateResult(
            candidate_id=candidate_id,
            vector=vector.copy(),
            params_path=str(bot_dir / "data" / "params.npz"),
            **metrics,
        ))
    return rows


def _result_json(row: CandidateResult) -> dict:
    return {
        "candidate_id": row.candidate_id,
        "params_path": row.params_path,
        "score": round(row.score, 6),
        "mean_bb100": round(row.mean_bb100, 6),
        "cvar20_bb100": round(row.cvar20_bb100, 6),
        "stdev_bb100": round(row.stdev_bb100, 6),
        "bust_rate": round(row.bust_rate, 6),
        "error_rate": round(row.error_rate, 6),
        "matches": row.matches,
        "hands_total": row.hands_total,
        "deltas": row.deltas,
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def tune(args) -> dict:
    stages = _parse_stages(args.stages)
    final_stage = stages[-1]
    min_stage_hands = final_stage.matches * final_stage.hands
    if not args.allow_smoke and min_stage_hands < args.min_hands_per_candidate:
        raise SystemExit(
            "refusing statistically weak tuning run: final stage has "
            f"{min_stage_hands} hands/candidate, require >= {args.min_hands_per_candidate}. "
            "Use --allow-smoke only for wiring checks."
        )
    run_id = args.run_id or "arm-selector-params-" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    rng = np.random.default_rng(args.seed)
    default_u = vector_from_params(normalized=True)
    mu = default_u.copy()
    sigma = np.full(len(PARAM_NAMES), args.init_sigma, dtype=np.float64)
    best: CandidateResult | None = None
    reports = []
    policy_path = Path(args.policy) if args.policy else None

    for generation in range(args.generations):
        vectors = rng.normal(mu, sigma, size=(args.population, len(PARAM_NAMES)))
        vectors = np.clip(vectors, 0.0, 1.0)
        vectors[0] = mu
        if generation == 0:
            vectors[0] = default_u
        active = [vectors[index].copy() for index in range(args.population)]
        stage_reports = []
        for stage_index, stage in enumerate(stages):
            rows = _evaluate_candidates(
                run_dir,
                generation,
                active,
                stage,
                args.seed * 1_000_000 + generation * 100_000 + stage_index * 10_000,
                policy_path,
                args.workers,
                args.parallel_backend,
            )
            rows.sort(key=lambda row: row.score, reverse=True)
            keep = max(1, int(math.ceil(len(rows) * stage.keep_frac)))
            stage_reports.append({
                "stage": stage_index,
                "matches": stage.matches,
                "hands": stage.hands,
                "keep": keep,
                "best": _result_json(rows[0]),
                "median_score": round(float(np.median([row.score for row in rows])), 6),
            })
            _append_jsonl(metrics_path, {
                "generation": generation,
                "stage": stage_index,
                "rows": [_result_json(row) for row in rows],
            })
            active = [row.vector for row in rows[:keep]]
        final_rows = rows
        generation_best = final_rows[0]
        if best is None or generation_best.score > best.score:
            best = generation_best
            shutil.copyfile(generation_best.params_path, run_dir / "best_params.npz")

        elite_count = max(2, int(math.ceil(len(final_rows) * args.elite_frac)))
        elites = np.stack([row.vector for row in final_rows[:elite_count]], axis=0)
        elite_mu = elites.mean(axis=0)
        elite_sigma = np.maximum(args.min_sigma, elites.std(axis=0))
        mu = np.clip((1.0 - args.smoothing) * mu + args.smoothing * elite_mu, 0.0, 1.0)
        sigma = np.clip((1.0 - args.smoothing) * sigma + args.smoothing * elite_sigma, args.min_sigma, args.max_sigma)
        # Regularize distribution toward the safe default so noisy wins do not
        # immediately drag the search into brittle extremes.
        mu = np.clip((1.0 - args.default_pull) * mu + args.default_pull * default_u, 0.0, 1.0)

        report = {
            "generation": generation,
            "best_so_far": _result_json(best),
            "generation_best": _result_json(generation_best),
            "elite_count": elite_count,
            "mu_distance_from_default": round(float(np.mean((mu - default_u) ** 2)), 8),
            "sigma_mean": round(float(np.mean(sigma)), 6),
            "stages": stage_reports,
        }
        reports.append(report)
        if args.progress:
            print(json.dumps(report), file=sys.stderr)

    assert best is not None
    if args.promote_output:
        if not args.allow_smoke and best.hands_total < args.min_hands_per_candidate:
            raise SystemExit("best candidate does not meet min hands requirement; refusing promotion")
        promote_path = Path(args.promote_output)
        promote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best.params_path, promote_path)

    summary = {
        "mode": "cem_racing_arm_selector_param_tuning",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": str(metrics_path),
        "best_params": str(run_dir / "best_params.npz"),
        "best": _result_json(best),
        "generations": args.generations,
        "population": args.population,
        "stages": [stage.__dict__ for stage in stages],
        "allow_smoke": bool(args.allow_smoke),
        "reports": reports,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune heuristic arm-selector parameters with CEM/racing")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--promote-output", default=None)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite-frac", type=float, default=0.20)
    parser.add_argument("--stages", default=DEFAULT_STAGE_SPEC, help="comma list of matches:hands:keep")
    parser.add_argument("--init-sigma", type=float, default=0.18)
    parser.add_argument("--min-sigma", type=float, default=0.025)
    parser.add_argument("--max-sigma", type=float, default=0.30)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--default-pull", type=float, default=0.04)
    parser.add_argument("--min-hands-per-candidate", type=int, default=6400)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--seed", type=int, default=8383)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.population < 4:
        raise SystemExit("--population must be at least 4")
    if args.allow_smoke and args.promote_output:
        raise SystemExit("--allow-smoke cannot be combined with --promote-output")
    result = tune(args)
    print(json.dumps(result, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
