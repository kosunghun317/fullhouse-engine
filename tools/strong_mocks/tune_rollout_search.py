"""Tune rollout-search f/g sizing functions with staged CEM racing.

This is local-only infrastructure for benchmark opponents. It searches a small
smooth parameterization instead of brute-forcing discrete threshold grids.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match  # noqa: E402
from tools.parallel import map_parallel  # noqa: E402
from tools.strong_mocks.policies import (  # noqa: E402
    ROLLOUT_DEFAULT_PARAMS,
    decide_rollout,
    rollout_params,
)


DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "rollout_search_tuning"
CANDIDATE_ID = "rollout_candidate"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float


PARAM_SPECS = (
    ParamSpec("f_min_equity", 0.30, 0.52),
    ParamSpec("f_center", 0.48, 0.74),
    ParamSpec("f_scale", 0.025, 0.18),
    ParamSpec("f_min_raise", 0.25, 0.85),
    ParamSpec("f_max_raise", 0.65, 1.80),
    ParamSpec("g_call_edge_base", 0.015, 0.13),
    ParamSpec("g_call_edge_active", 0.005, 0.07),
    ParamSpec("g_pressure_discount", 0.0, 0.06),
    ParamSpec("g_thin_call_edge", -0.03, 0.05),
    ParamSpec("g_raise_edge", 0.06, 0.30),
    ParamSpec("g_raise_equity", 0.72, 0.94),
    ParamSpec("g_raise_max_owed_pot", 0.08, 0.45),
    ParamSpec("g_raise_center", 0.76, 0.95),
    ParamSpec("g_raise_scale", 0.02, 0.14),
    ParamSpec("g_min_raise", 0.35, 1.05),
    ParamSpec("g_max_raise", 0.75, 2.00),
)


SUITES = {
    "heads_up_reference": {
        "hands": 400,
        "bots": {
            CANDIDATE_ID: "__CANDIDATE__",
            "shark": "bots/shark/bot.py",
        },
    },
    "heads_up_pressure": {
        "hands": 400,
        "bots": {
            CANDIDATE_ID: "__CANDIDATE__",
            "aggressor": "bots/aggressor/bot.py",
        },
    },
    "sixmax_reference": {
        "hands": 400,
        "bots": {
            CANDIDATE_ID: "__CANDIDATE__",
            "shark": "bots/shark/bot.py",
            "aggressor": "bots/aggressor/bot.py",
            "mathematician": "bots/mathematician/bot.py",
            "template": "bots/template/bot.py",
            "pot_odds": "bots/ref_bot_2/bot.py",
        },
    },
    "sixmax_threshold_pressure": {
        "hands": 400,
        "bots": {
            CANDIDATE_ID: "__CANDIDATE__",
            "threshold_caller": "bots/benchmarks/threshold_caller/bot.py",
            "half_pot_pressure": "bots/benchmarks/half_pot_pressure/bot.py",
            "short_stacker": "bots/benchmarks/short_stacker/bot.py",
            "overfold": "bots/benchmarks/overfold/bot.py",
            "always_call": "bots/benchmarks/always_call/bot.py",
        },
    },
}


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_vector() -> np.ndarray:
    return np.asarray([ROLLOUT_DEFAULT_PARAMS[spec.name] for spec in PARAM_SPECS], dtype=np.float64)


def params_from_vector(vector: np.ndarray) -> dict[str, float]:
    values = {}
    for spec, raw in zip(PARAM_SPECS, vector):
        values[spec.name] = float(max(spec.lo, min(spec.hi, raw)))
    values["f_max_raise"] = max(values["f_min_raise"], values["f_max_raise"])
    values["g_max_raise"] = max(values["g_min_raise"], values["g_max_raise"])
    values["g_raise_center"] = max(values["g_raise_equity"], values["g_raise_center"])
    values["samples"] = float(ROLLOUT_DEFAULT_PARAMS["samples"])
    values["deep_samples"] = float(ROLLOUT_DEFAULT_PARAMS["deep_samples"])
    return rollout_params(values)


def vector_from_params(params: dict[str, float]) -> np.ndarray:
    params = rollout_params(params)
    return np.asarray([params[spec.name] for spec in PARAM_SPECS], dtype=np.float64)


def parse_stages(spec: str) -> list[dict[str, float]]:
    stages = []
    for index, raw in enumerate(part.strip() for part in spec.split(",") if part.strip()):
        pieces = raw.split(":")
        if len(pieces) != 3:
            raise argparse.ArgumentTypeError("stages must be seed_count:hands:keep_frac")
        seed_count = int(pieces[0])
        hands = int(pieces[1])
        keep_frac = float(pieces[2])
        if seed_count <= 0 or hands <= 0 or not 0 < keep_frac <= 1:
            raise argparse.ArgumentTypeError("stage values must be positive and keep_frac in (0, 1]")
        stages.append({"stage": index, "seed_count": seed_count, "hands": hands, "keep_frac": keep_frac})
    if not stages:
        raise argparse.ArgumentTypeError("at least one stage is required")
    return stages


def selected_suites(raw: str | None) -> list[str]:
    if not raw:
        return list(SUITES)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(names) - set(SUITES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown suite(s): {', '.join(unknown)}")
    return names


def write_rollout_bot(bot_dir: Path, params: dict[str, float]) -> None:
    bot_dir.mkdir(parents=True, exist_ok=True)
    literal = json.dumps({key: round(float(value), 10) for key, value in sorted(params.items())}, sort_keys=True)
    root_literal = json.dumps(str(ROOT))
    bot_py = (
        '"""Generated rollout-search threshold candidate."""\n\n'
        "import os\n"
        "import sys\n\n"
        f"ROOT = {root_literal}\n"
        "if ROOT not in sys.path:\n"
        "    sys.path.insert(0, ROOT)\n\n"
        "from tools.strong_mocks.policies import decide_rollout\n\n"
        f"PARAMS = {literal}\n\n"
        "def decide(state):\n"
        "    return decide_rollout(state, style='rollout_pressure', params=PARAMS)\n"
    )
    (bot_dir / "bot.py").write_text(bot_py, encoding="utf-8")


def _bot_paths(suite_name: str, candidate_path: str) -> dict[str, str]:
    suite = SUITES[suite_name]
    return {
        name: (candidate_path if path == "__CANDIDATE__" else path)
        for name, path in suite["bots"].items()
    }


def _run_match_task(task: dict) -> dict:
    result = run_match(
        match_id=f"{task['candidate_id']}_{task['suite']}_{task['seed']}",
        bot_paths=_bot_paths(task["suite"], task["candidate_path"]),
        n_hands=task["hands"],
        verbose=False,
        seed=task["seed"],
    )
    errors = result["bot_errors"].get(CANDIDATE_ID, [])
    final_stack = result.get("final_stacks", {}).get(CANDIDATE_ID)
    delta = int(result["chip_delta"].get(CANDIDATE_ID, 0))
    return {
        "candidate_id": task["candidate_id"],
        "suite": task["suite"],
        "seed": task["seed"],
        "hands": task["hands"],
        "delta": delta,
        "final_stack": final_stack,
        "busted": bool(final_stack is not None and final_stack <= 0) or delta <= -10000,
        "error_count": len(errors),
        "errors": errors,
        "duration_s": float(result.get("duration_s", 0.0)),
    }


def summarize_results(results: list[dict], bust_penalty: float, error_penalty: float, stdev_penalty: float) -> dict:
    deltas = [int(item["delta"]) for item in results]
    error_count = sum(int(item["error_count"]) for item in results)
    bust_count = sum(int(item["busted"]) for item in results)
    mean_delta = statistics.mean(deltas) if deltas else 0.0
    stdev_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    bust_rate = bust_count / max(1, len(results))
    error_rate = error_count / max(1, len(results))
    score = mean_delta - stdev_penalty * stdev_delta - bust_penalty * bust_rate - error_penalty * error_rate
    return {
        "runs": len(results),
        "mean_delta": round(float(mean_delta), 3),
        "median_delta": round(float(statistics.median(deltas)), 3) if deltas else 0.0,
        "min_delta": min(deltas) if deltas else 0,
        "max_delta": max(deltas) if deltas else 0,
        "stdev_delta": round(float(stdev_delta), 3),
        "positive_runs": sum(1 for value in deltas if value > 0),
        "bust_count": bust_count,
        "bust_rate": round(float(bust_rate), 5),
        "error_count": error_count,
        "error_rate": round(float(error_rate), 5),
        "mean_duration_s": round(statistics.mean([item["duration_s"] for item in results]), 4) if results else 0.0,
        "score": round(float(score), 3),
    }


def evaluate_candidate(
    candidate_id: str,
    candidate_path: Path,
    suite_names: list[str],
    seed_start: int,
    seed_count: int,
    hands: int,
    workers: int,
    parallel_backend: str,
    bust_penalty: float,
    error_penalty: float,
    stdev_penalty: float,
) -> dict:
    tasks = []
    for suite in suite_names:
        suite_hands = hands or int(SUITES[suite]["hands"])
        for seed in range(seed_start, seed_start + seed_count):
            tasks.append({
                "candidate_id": candidate_id,
                "candidate_path": str(candidate_path),
                "suite": suite,
                "seed": seed,
                "hands": suite_hands,
            })
    results = map_parallel(_run_match_task, tasks, workers=workers, backend=parallel_backend)
    return {
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path),
        "summary": summarize_results(results, bust_penalty, error_penalty, stdev_penalty),
        "runs": results,
    }


def representative_states() -> list[dict]:
    players6 = [
        {"seat": i, "bot_id": f"p{i}", "stack": 10000 - i * 250, "is_folded": False, "is_all_in": False}
        for i in range(6)
    ]
    return [
        {
            "type": "action_request",
            "hand_id": "lat_pre",
            "street": "preflop",
            "seat_to_act": 0,
            "pot": 150,
            "current_bet": 100,
            "min_raise_to": 200,
            "amount_owed": 100,
            "can_check": False,
            "your_cards": ["As", "Kh"],
            "community_cards": [],
            "your_stack": 9900,
            "your_bet_this_street": 0,
            "players": players6,
            "action_log": [],
        },
        {
            "type": "action_request",
            "hand_id": "lat_flop",
            "street": "flop",
            "seat_to_act": 0,
            "pot": 2100,
            "current_bet": 0,
            "min_raise_to": 200,
            "amount_owed": 0,
            "can_check": True,
            "your_cards": ["As", "Ks"],
            "community_cards": ["Qs", "7s", "2d"],
            "your_stack": 8600,
            "your_bet_this_street": 0,
            "players": players6,
            "action_log": [],
        },
        {
            "type": "action_request",
            "hand_id": "lat_turn",
            "street": "turn",
            "seat_to_act": 0,
            "pot": 4800,
            "current_bet": 2400,
            "min_raise_to": 4800,
            "amount_owed": 2400,
            "can_check": False,
            "your_cards": ["Ah", "Qh"],
            "community_cards": ["Ad", "7h", "2c", "Jh"],
            "your_stack": 7600,
            "your_bet_this_street": 0,
            "players": players6,
            "action_log": [],
        },
        {
            "type": "action_request",
            "hand_id": "lat_river",
            "street": "river",
            "seat_to_act": 0,
            "pot": 8200,
            "current_bet": 0,
            "min_raise_to": 400,
            "amount_owed": 0,
            "can_check": True,
            "your_cards": ["9c", "9d"],
            "community_cards": ["9s", "7d", "2h", "Jc", "3s"],
            "your_stack": 5400,
            "your_bet_this_street": 0,
            "players": players6,
            "action_log": [],
        },
    ]


def measure_decision_latency(params: dict[str, float], repeats: int = 3) -> dict:
    timings = []
    for state in representative_states():
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            action = decide_rollout(state, style="rollout_pressure", params=params)
            timings.append({
                "hand_id": state["hand_id"],
                "street": state["street"],
                "elapsed_s": time.perf_counter() - started,
                "action": action,
            })
    elapsed = [item["elapsed_s"] for item in timings]
    return {
        "samples": int(rollout_params(params)["samples"]),
        "repeats": repeats,
        "states": len(representative_states()),
        "max_elapsed_s": round(max(elapsed), 6) if elapsed else 0.0,
        "mean_elapsed_s": round(statistics.mean(elapsed), 6) if elapsed else 0.0,
        "timings": [
            {
                **item,
                "elapsed_s": round(float(item["elapsed_s"]), 6),
            }
            for item in timings
        ],
    }


def _candidate_vectors(mu: np.ndarray, sigma: np.ndarray, population: int, rng: np.random.Generator, generation: int) -> list[np.ndarray]:
    vectors = [default_vector() if generation == 0 else mu.copy()]
    while len(vectors) < population:
        vectors.append(mu + rng.normal(0.0, sigma, size=mu.shape))
    return vectors


def tune(args: SimpleNamespace) -> dict:
    run_id = args.run_id or f"rollout-search-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_root) / run_id
    if run_dir.exists() and args.clean:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    suite_names = selected_suites(args.suites)
    stages = parse_stages(args.stages)
    rng = np.random.default_rng(args.seed)
    widths = np.asarray([spec.hi - spec.lo for spec in PARAM_SPECS], dtype=np.float64)
    mu = default_vector()
    sigma = widths * float(args.init_sigma_frac)
    sigma_floor = widths * float(args.min_sigma_frac)
    generations = []
    best: dict | None = None

    default_params = params_from_vector(default_vector())
    latency = None
    if not args.skip_latency_check:
        latency = measure_decision_latency(default_params, repeats=args.latency_repeats)
        if latency["max_elapsed_s"] > args.max_decision_seconds:
            raise SystemExit(
                f"1024-sample rollout default exceeded decision budget: "
                f"{latency['max_elapsed_s']}s > {args.max_decision_seconds}s"
            )

    for generation in range(int(args.generations)):
        vectors = _candidate_vectors(mu, sigma, int(args.population), rng, generation)
        candidates = []
        for index, vector in enumerate(vectors):
            candidate_id = f"g{generation:03d}_c{index:03d}"
            params = params_from_vector(vector)
            clipped_vector = vector_from_params(params)
            bot_dir = run_dir / "candidates" / candidate_id
            write_rollout_bot(bot_dir, params)
            candidates.append({"candidate_id": candidate_id, "vector": clipped_vector, "params": params, "bot_dir": bot_dir})

        remaining = candidates
        stage_reports = []
        for stage in stages:
            evaluated = []
            for candidate in remaining:
                report = evaluate_candidate(
                    candidate["candidate_id"],
                    candidate["bot_dir"],
                    suite_names,
                    int(args.seed_start),
                    int(stage["seed_count"]),
                    int(stage["hands"]),
                    int(args.workers),
                    args.parallel_backend,
                    float(args.bust_penalty),
                    float(args.error_penalty),
                    float(args.stdev_penalty),
                )
                report["params"] = candidate["params"]
                report["vector"] = candidate["vector"]
                evaluated.append(report)
            evaluated.sort(key=lambda item: item["summary"]["score"], reverse=True)
            keep = max(int(args.elite), int(round(len(evaluated) * float(stage["keep_frac"]))))
            keep = max(1, min(len(evaluated), keep))
            stage_reports.append({
                "stage": int(stage["stage"]),
                "seed_count": int(stage["seed_count"]),
                "hands": int(stage["hands"]),
                "keep": keep,
                "best": _json_safe({k: v for k, v in evaluated[0].items() if k != "runs"}),
                "median_score": round(float(statistics.median(item["summary"]["score"] for item in evaluated)), 3),
            })
            remaining_ids = {item["candidate_id"] for item in evaluated[:keep]}
            remaining = [candidate for candidate in remaining if candidate["candidate_id"] in remaining_ids]
            if stage["stage"] == stages[-1]["stage"]:
                final_evaluated = evaluated

        generation_best = final_evaluated[0]
        if best is None or generation_best["summary"]["score"] > best["summary"]["score"]:
            best = generation_best

        elites = final_evaluated[: max(1, min(int(args.elite), len(final_evaluated)))]
        elite_matrix = np.stack([item["vector"] for item in elites]).astype(np.float64)
        elite_mean = np.mean(elite_matrix, axis=0)
        elite_sigma = np.maximum(np.std(elite_matrix, axis=0), sigma_floor)
        mu = (1.0 - float(args.smoothing)) * mu + float(args.smoothing) * elite_mean
        sigma = np.maximum(
            (1.0 - float(args.smoothing)) * sigma + float(args.smoothing) * elite_sigma,
            sigma_floor,
        )
        generation_report = {
            "generation": generation,
            "best_so_far": _json_safe({k: v for k, v in best.items() if k != "runs"}) if best else None,
            "generation_best": _json_safe({k: v for k, v in generation_best.items() if k != "runs"}),
            "elite_count": len(elites),
            "sigma_mean": round(float(np.mean(sigma)), 6),
            "stages": stage_reports,
        }
        generations.append(generation_report)
        if args.progress:
            print(json.dumps(generation_report, sort_keys=True), file=sys.stderr)

    assert best is not None
    best_params = rollout_params(best["params"])
    best_bot_dir = run_dir / "best_bot"
    write_rollout_bot(best_bot_dir, best_params)
    _write_json(run_dir / "best_params.json", best_params)

    best_latency = None
    if not args.skip_latency_check:
        best_latency = measure_decision_latency(best_params, repeats=args.latency_repeats)
        if best_latency["max_elapsed_s"] > args.max_decision_seconds:
            raise SystemExit(
                f"best rollout candidate exceeded decision budget: "
                f"{best_latency['max_elapsed_s']}s > {args.max_decision_seconds}s"
            )

    summary = {
        "mode": "rollout_search_fg_cem_tuning",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "suites": suite_names,
        "param_names": [spec.name for spec in PARAM_SPECS],
        "samples": int(best_params["samples"]),
        "best_params_path": str(run_dir / "best_params.json"),
        "best_bot": str(best_bot_dir),
        "best": _json_safe({k: v for k, v in best.items() if k != "runs"}),
        "default_latency": latency,
        "best_latency": best_latency,
        "generations": generations,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune rollout-search f/g sizing functions")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--suites", default=None, help="Comma-separated suite names; defaults to all tuner suites")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--stages", default="4:120:0.5,12:240:0.5")
    parser.add_argument("--seed-start", type=int, default=61001)
    parser.add_argument("--seed", type=int, default=9292)
    parser.add_argument("--init-sigma-frac", type=float, default=0.22)
    parser.add_argument("--min-sigma-frac", type=float, default=0.025)
    parser.add_argument("--smoothing", type=float, default=0.45)
    parser.add_argument("--bust-penalty", type=float, default=9000.0)
    parser.add_argument("--error-penalty", type=float, default=25000.0)
    parser.add_argument("--stdev-penalty", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--max-decision-seconds", type=float, default=2.0)
    parser.add_argument("--skip-latency-check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = tune(args)
    if args.json:
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    else:
        best = summary["best"]
        print(
            f"best={best['candidate_id']} score={best['summary']['score']} "
            f"mean_delta={best['summary']['mean_delta']} best_bot={summary['best_bot']}"
        )
        if summary.get("best_latency"):
            print(f"best_latency_max_s={summary['best_latency']['max_elapsed_s']}")


if __name__ == "__main__":
    main()
