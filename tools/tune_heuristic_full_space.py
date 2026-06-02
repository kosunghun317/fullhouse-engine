"""Full-space heuristic parameter tuner with statistically gated racing.

This is local-only infrastructure. It searches bounded HEURISTIC_* env
parameters with a diagonal cross-entropy method and common-random-number
successive-halving stages. The goal is a stable local maximum, not a brittle
single-run winner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_heuristic import register_portal_profile_suites, run_suite
from tools.plot_training_progress import render_svg
from tools.select_heuristic_config import PRESETS
from tools.strong_mocks.self_train_heuristic import MUTATION_SPACE
from tools.tune_heuristic_thresholds import CONFIGS


BOT_PATH = ROOT / "bots" / "heuristic" / "bot.py"
DEFAULT_RUN_ROOT = ROOT / "runs" / "heuristic_full_space_tuning"
ENV_PATTERN = re.compile(
    r"_(?P<kind>float|int)_env\(\"(?P<name>HEURISTIC_[A-Z0-9_]+)\",\s*(?P<default>[-+]?[0-9]*\.?[0-9]+)\)"
)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    default: float
    low: float
    high: float

    @property
    def is_binary(self) -> bool:
        return self.name.endswith("_ENABLED")


@dataclass(frozen=True)
class Stage:
    seed_count: int
    hands: int
    keep_fraction: float


def _named_config_values() -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for config in CONFIGS.values():
        for key, value in config.items():
            try:
                values.setdefault(key, []).append(float(value))
            except Exception:
                pass
    return values


def _generic_bounds(name: str, default: float, kind: str) -> tuple[float, float]:
    if name.endswith("_ENABLED"):
        return 0.0, 1.0
    if kind == "int":
        if name.endswith("_SAMPLES"):
            return max(50.0, default * 0.50), min(2200.0, default * 1.80)
        return max(1.0, default * 0.50), max(default + 1.0, default * 1.80)
    if name.endswith("_SCORE") or "_SCORE" in name:
        return max(20.0, default - 24.0), min(100.0, default + 24.0)
    if name.endswith("_BB") or "RAISE_BB" in name:
        return 1.0, 18.0
    if name in {"HEURISTIC_SPR_LOW", "HEURISTIC_SPR_HIGH", "HEURISTIC_TRAP_CHECK_SPR_MAX"}:
        return 0.5, 12.0
    if name.endswith("_ADJ_NIT") or name.endswith("_ADJ_ABC") or name.endswith("_ADJ_MANIAC") or name.endswith("_ADJ_MULTIWAY") or "_ADJ_" in name:
        return -0.14, 0.14
    if "MARGIN" in name:
        return 0.0, 0.18
    if any(token in name for token in ("BONUS", "DISCOUNT", "PENALTY")):
        return 0.0, 0.16
    if "RAISE_FRACTION" in name or "VALUE_FRACTION" in name or "BLUFF_FRACTION" in name:
        return 0.20, 1.25
    if any(token in name for token in ("PROB", "RATE", "REQ", "CUTOFF", "THRESHOLD", "EQUITY", "MIN", "MAX")):
        return 0.0, 1.0
    width = max(0.10, abs(default) * 0.60)
    return default - width, default + width


def _merge_observed_bounds(
    name: str,
    default: float,
    kind: str,
    observed: dict[str, list[float]],
) -> tuple[float, float]:
    generic_low, generic_high = _generic_bounds(name, default, kind)
    if name in MUTATION_SPACE:
        low, high = MUTATION_SPACE[name]
        return float(low), float(high)
    values = [default, *observed.get(name, [])]
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-9:
        return generic_low, generic_high
    pad = max((high - low) * 0.50, (generic_high - generic_low) * 0.08)
    return max(generic_low, low - pad), min(generic_high, high + pad)


def load_param_specs() -> list[ParamSpec]:
    observed = _named_config_values()
    specs = []
    seen = set()
    for match in ENV_PATTERN.finditer(BOT_PATH.read_text(encoding="utf-8")):
        name = match.group("name")
        if name in seen or name == "HEURISTIC_RNG_SEED":
            continue
        seen.add(name)
        kind = match.group("kind")
        default = float(match.group("default"))
        low, high = _merge_observed_bounds(name, default, kind, observed)
        if high <= low:
            high = low + 1.0
        specs.append(ParamSpec(name=name, kind=kind, default=default, low=low, high=high))
    return sorted(specs, key=lambda item: item.name)


def _encode_value(spec: ParamSpec, value: float) -> float:
    return min(1.0, max(0.0, (float(value) - spec.low) / (spec.high - spec.low)))


def _decode_value(spec: ParamSpec, value: float) -> str:
    raw = spec.low + min(1.0, max(0.0, float(value))) * (spec.high - spec.low)
    if spec.is_binary:
        return "1.0" if raw >= 0.5 else "0.0"
    if spec.kind == "int":
        return str(int(round(raw)))
    return f"{raw:.6g}"


def vector_from_env(specs: list[ParamSpec], env: dict[str, str] | None = None) -> np.ndarray:
    env = env or {}
    values = []
    for spec in specs:
        raw = float(env.get(spec.name, spec.default))
        values.append(_encode_value(spec, raw))
    return np.asarray(values, dtype=np.float64)


def env_from_vector(specs: list[ParamSpec], vector: np.ndarray) -> dict[str, str]:
    env = {}
    for spec, value in zip(specs, vector):
        decoded = _decode_value(spec, float(value))
        default = _decode_value(spec, _encode_value(spec, spec.default))
        if decoded != default:
            env[spec.name] = decoded
    return env


def parse_stages(text: str) -> list[Stage]:
    stages = []
    for item in text.split(","):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid stage {item!r}; expected seeds:hands:keep_fraction")
        stages.append(Stage(int(parts[0]), int(parts[1]), float(parts[2])))
    if not stages:
        raise ValueError("at least one stage is required")
    return stages


def _set_env(overrides: dict[str, str], keys: list[str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in keys}
    for key in keys:
        if key in overrides:
            os.environ[key] = str(overrides[key])
        else:
            os.environ.pop(key, None)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _score_suite(summary: dict, risk_weight: float, min_weight: float, bust_penalty: float, error_penalty: float) -> float:
    runs = max(1, int(summary["runs"]))
    bust_rate = float(summary["bust_count"]) / runs
    error_rate = float(summary["heuristic_error_count"]) / runs
    return (
        float(summary["mean_delta"])
        - risk_weight * float(summary["stdev_delta"])
        + min_weight * float(summary["min_delta"])
        - bust_penalty * bust_rate
        - error_penalty * error_rate
    )


def evaluate_env(
    env: dict[str, str],
    specs: list[ParamSpec],
    suites: list[str],
    seeds: list[int],
    hands: int,
    workers: int,
    parallel_backend: str,
    args,
) -> dict:
    old = _set_env(env, [spec.name for spec in specs])
    try:
        results = [
            run_suite(
                suite,
                seeds,
                hands,
                summary_only=True,
                workers=workers,
                parallel_backend=parallel_backend,
            )
            for suite in suites
        ]
    finally:
        _restore_env(old)
    suite_scores = [
        _score_suite(
            item["summary"],
            args.risk_weight,
            args.min_weight,
            args.bust_penalty,
            args.error_penalty,
        )
        for item in results
    ]
    means = [item["summary"]["mean_delta"] for item in results]
    mins = [item["summary"]["min_delta"] for item in results]
    runs = sum(item["summary"]["runs"] for item in results)
    busts = sum(item["summary"]["bust_count"] for item in results)
    errors = sum(item["summary"]["heuristic_error_count"] for item in results)
    return {
        "env": env,
        "score": round(float(statistics.mean(suite_scores)) if suite_scores else 0.0, 3),
        "mean_of_suite_means": round(float(statistics.mean(means)) if means else 0.0, 3),
        "worst_min_delta": int(min(mins) if mins else 0),
        "runs": int(runs),
        "bust_count": int(busts),
        "bust_rate": round(busts / max(1, runs), 4),
        "heuristic_error_count": int(errors),
        "results": results,
    }


def _soft_weights(scores: list[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    scale = max(1.0, float(np.std(arr)))
    arr = (arr - np.max(arr)) / scale
    weights = np.exp(np.clip(arr, -30.0, 0.0))
    return weights / max(1e-9, float(weights.sum()))


def _candidate_pool(
    specs: list[ParamSpec],
    rng: np.random.Generator,
    mean: np.ndarray,
    sigma: np.ndarray,
    generation: int,
    population: int,
    best_env: dict[str, str] | None,
) -> list[dict]:
    candidates = [{"name": f"g{generation:03d}_mean", "vector": mean.copy(), "env": env_from_vector(specs, mean)}]
    if best_env:
        vector = vector_from_env(specs, best_env)
        candidates.append({"name": f"g{generation:03d}_best", "vector": vector, "env": best_env})
    if generation == 0:
        for name, env in CONFIGS.items():
            vector = vector_from_env(specs, env)
            candidates.append({"name": f"seed_{name}", "vector": vector, "env": env_from_vector(specs, vector)})
            if len(candidates) >= max(2, population // 2):
                break
    while len(candidates) < population:
        vector = np.clip(rng.normal(mean, sigma), 0.0, 1.0)
        candidates.append({
            "name": f"g{generation:03d}_sample_{len(candidates):03d}",
            "vector": vector,
            "env": env_from_vector(specs, vector),
        })
    return candidates[:population]


def _check_budget(args, suites: list[str], stages: list[Stage]) -> None:
    if args.allow_smoke:
        return
    if args.population < 16:
        raise SystemExit("--population must be at least 16 unless --allow-smoke is set")
    if args.generations < 3:
        raise SystemExit("--generations must be at least 3 unless --allow-smoke is set")
    if min(stage.hands for stage in stages) < 400:
        raise SystemExit("all stages must use at least 400 hands unless --allow-smoke is set")
    min_tasks = min(stage.seed_count * len(suites) for stage in stages)
    final_tasks = stages[-1].seed_count * len(suites)
    if min_tasks < args.min_tasks_per_candidate:
        raise SystemExit(
            f"every stage must evaluate at least {args.min_tasks_per_candidate} "
            "suite/seed tasks per candidate; use --allow-smoke only for wiring checks"
        )
    if final_tasks < args.min_final_tasks_per_candidate:
        raise SystemExit(
            f"final stage must evaluate at least {args.min_final_tasks_per_candidate} "
            "suite/seed tasks per candidate"
        )
    if args.skip_final_validation:
        raise SystemExit("--skip-final-validation is only allowed with --allow-smoke")
    if int(args.final_validation_hands) < 400:
        raise SystemExit("--final-validation-hands must be at least 400 unless --allow-smoke is set")
    final_validation_tasks = int(args.final_validation_seed_count) * len(suites)
    if final_validation_tasks < args.min_final_tasks_per_candidate:
        raise SystemExit(
            f"final validation must evaluate at least {args.min_final_tasks_per_candidate} "
            "suite/seed tasks for the selected candidate"
        )


def _write_env_script(path: Path, env: dict[str, str]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for key in sorted(env):
        lines.append(f"export {key}={shlex.quote(str(env[key]))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> dict:
    specs = load_param_specs()
    suites = list(args.suite or PRESETS[args.preset]["suites"])
    if args.portal_mocks_dir:
        portal_suites = register_portal_profile_suites(
            args.portal_mocks_dir,
            mode=args.portal_mocks_mode,
            profiles=args.portal_profile,
            hands=args.portal_suite_hands or 400,
        )
        suites = portal_suites if args.portal_only else [*suites, *portal_suites]
    stages = parse_stages(args.stages)
    _check_budget(args, suites, stages)

    run_root = Path(args.run_root or (DEFAULT_RUN_ROOT / args.run_id)).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    metrics_path = run_root / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    (run_root / "param_space.json").write_text(
        json.dumps([spec.__dict__ for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rng = np.random.default_rng(args.seed)
    mean = vector_from_env(specs, CONFIGS.get(args.center_config, {}))
    sigma = np.full(len(specs), float(args.initial_sigma), dtype=np.float64)
    best_env: dict[str, str] | None = None
    best_report: dict | None = None
    generations = []

    for generation in range(args.generations):
        candidates = _candidate_pool(specs, rng, mean, sigma, generation, args.population, best_env)
        survivors = candidates
        stage_reports = []
        for stage_index, stage in enumerate(stages):
            seeds = list(range(args.seed_start + generation * 100000 + stage_index * 10000, args.seed_start + generation * 100000 + stage_index * 10000 + stage.seed_count))
            evaluated = []
            for candidate in survivors:
                report = evaluate_env(
                    candidate["env"],
                    specs,
                    suites,
                    seeds,
                    stage.hands,
                    args.workers,
                    args.parallel_backend,
                    args,
                )
                row = {key: value for key, value in report.items() if key != "results"}
                row.update({
                    "name": candidate["name"],
                    "generation": generation,
                    "stage": stage_index,
                    "seed_count": stage.seed_count,
                    "hands": stage.hands,
                    "suites": suites,
                    "vector": candidate["vector"].round(6).tolist(),
                })
                evaluated.append(row)
            evaluated.sort(key=lambda item: (item["score"], item["mean_of_suite_means"], item["worst_min_delta"]), reverse=True)
            keep_count = max(args.elite, int(math.ceil(len(evaluated) * stage.keep_fraction)))
            keep_count = min(len(evaluated), max(1, keep_count))
            stage_reports.append({"stage": stage.__dict__, "ranking": evaluated})
            survivor_names = {item["name"] for item in evaluated[:keep_count]}
            survivors = [item for item in survivors if item["name"] in survivor_names]
        final_ranking = stage_reports[-1]["ranking"]
        elites = final_ranking[: max(1, min(args.elite, len(final_ranking)))]
        elite_vectors = np.asarray([item["vector"] for item in elites], dtype=np.float64)
        weights = _soft_weights([item["score"] for item in elites])
        elite_mean = np.average(elite_vectors, axis=0, weights=weights) if weights.size else elite_vectors.mean(axis=0)
        elite_sigma = np.sqrt(np.average((elite_vectors - elite_mean) ** 2, axis=0, weights=weights)) if weights.size else elite_vectors.std(axis=0)
        mean = np.clip(args.mean_smoothing * mean + (1.0 - args.mean_smoothing) * elite_mean, 0.0, 1.0)
        sigma = np.maximum(args.min_sigma, args.sigma_decay * (args.mean_smoothing * sigma + (1.0 - args.mean_smoothing) * elite_sigma))

        winner = final_ranking[0]
        if best_report is None or winner["score"] > best_report["score"]:
            best_report = dict(winner)
            best_env = dict(winner["env"])
            (run_root / "best_config.json").write_text(json.dumps(best_report, indent=2, sort_keys=True), encoding="utf-8")
            (run_root / "best_env.json").write_text(json.dumps(best_env, indent=2, sort_keys=True), encoding="utf-8")
            _write_env_script(run_root / "best_env.sh", best_env)
        metric = {
            "cycle": generation,
            "generation": generation,
            "series": "best_score",
            "phase": "full_space_tuning",
            "mean_delta": winner["score"],
            "score": winner["score"],
            "candidate": winner["name"],
        }
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")
        generations.append({
            "generation": generation,
            "winner": winner,
            "stage_reports": stage_reports,
            "next_sigma_mean": round(float(np.mean(sigma)), 6),
        })
        (run_root / f"generation_{generation:03d}.json").write_text(json.dumps(generations[-1], indent=2, sort_keys=True), encoding="utf-8")
        if args.progress:
            print(json.dumps({
                "generation": generation,
                "winner": winner["name"],
                "score": winner["score"],
                "mean_of_suite_means": winner["mean_of_suite_means"],
                "best_score": best_report["score"],
                "sigma_mean": float(np.mean(sigma)),
            }), file=sys.stderr, flush=True)

    final_validation = None
    if best_env is not None and not args.skip_final_validation and not args.allow_smoke:
        final_seeds = list(range(args.final_validation_seed_start, args.final_validation_seed_start + args.final_validation_seed_count))
        final_validation = evaluate_env(
            best_env,
            specs,
            suites,
            final_seeds,
            args.final_validation_hands,
            args.workers,
            args.parallel_backend,
            args,
        )
        final_payload = {
            "seed_start": args.final_validation_seed_start,
            "seed_count": args.final_validation_seed_count,
            "hands": args.final_validation_hands,
            "suites": suites,
            **final_validation,
        }
        (run_root / "final_validation.json").write_text(json.dumps(final_payload, indent=2, sort_keys=True), encoding="utf-8")

    plot_path = run_root / "ev_progress.svg"
    plot_report = render_svg(metrics_path, plot_path)
    summary = {
        "run_id": args.run_id,
        "run_root": str(run_root),
        "param_count": len(specs),
        "preset": args.preset,
        "suites": suites,
        "stages": [stage.__dict__ for stage in stages],
        "population": args.population,
        "elite": args.elite,
        "generations": args.generations,
        "best": best_report,
        "best_env": best_env,
        "final_validation": final_validation,
        "final_validation_path": str(run_root / "final_validation.json") if final_validation is not None else None,
        "metrics": str(metrics_path),
        "plot": str(plot_path),
        "plot_report": plot_report,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the full HEURISTIC_* parameter space with staged CEM racing")
    parser.add_argument("--run-id", default="heuristic-full-space")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="strong-screen")
    parser.add_argument("--suite", action="append")
    parser.add_argument("--portal-mocks-dir", type=Path)
    parser.add_argument("--portal-mocks-mode", choices=["sixmax", "heads-up"], default="sixmax")
    parser.add_argument("--portal-profile", action="append", default=[])
    parser.add_argument("--portal-only", action="store_true")
    parser.add_argument("--portal-suite-hands", type=int)
    parser.add_argument("--stages", default="128:400:0.5,256:400:0.25")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elite", type=int, default=6)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--center-config", choices=sorted(CONFIGS), default="baseline")
    parser.add_argument("--initial-sigma", type=float, default=0.18)
    parser.add_argument("--min-sigma", type=float, default=0.025)
    parser.add_argument("--sigma-decay", type=float, default=0.86)
    parser.add_argument("--mean-smoothing", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=515151)
    parser.add_argument("--seed-start", type=int, default=31001)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--risk-weight", type=float, default=0.35)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--bust-penalty", type=float, default=8000.0)
    parser.add_argument("--error-penalty", type=float, default=20000.0)
    parser.add_argument("--min-tasks-per-candidate", type=int, default=512)
    parser.add_argument("--min-final-tasks-per-candidate", type=int, default=1024)
    parser.add_argument("--final-validation-seed-count", type=int, default=256)
    parser.add_argument("--final-validation-seed-start", type=int, default=91001)
    parser.add_argument("--final-validation-hands", type=int, default=400)
    parser.add_argument("--skip-final-validation", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
