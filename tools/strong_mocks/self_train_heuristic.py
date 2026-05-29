"""Evolutionary self-training for heuristic bot configurations.

This trains benchmark-only heuristic variants by running 6-max matches that
contain two or three heuristic variants plus strong/reference opponents. It
does not edit bots/heuristic/bot.py. Promising configs are exported as wrapper
bot directories under the configured run directory.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match
from tools.parallel import map_parallel
from tools.plot_training_progress import render_svg
from tools.tune_heuristic_thresholds import CONFIGS


TEMPLATE = ROOT / "bots" / "self_training" / "heuristic_variant_template" / "bot.py"
DEFAULT_RUN_ROOT = ROOT / "runs" / "fullhouse_self_training"

MUTATION_SPACE = {
    "HEURISTIC_PREFLOP_OPEN_SCORE": (56.0, 70.0),
    "HEURISTIC_PREFLOP_LATE_PLAY_SCORE": (40.0, 58.0),
    "HEURISTIC_PREFLOP_CHEAP_CALL_SCORE": (42.0, 58.0),
    "HEURISTIC_PREFLOP_LATE_RAISE_CALL_SCORE": (50.0, 66.0),
    "HEURISTIC_PREFLOP_OPEN_BB": (2.3, 3.4),
    "HEURISTIC_PREFLOP_PREMIUM_OPEN_BB": (2.8, 4.0),
    "HEURISTIC_PREFLOP_RERAISE_MULT": (1.8, 2.8),
    "HEURISTIC_CALL_MARGIN_BASE": (0.055, 0.105),
    "HEURISTIC_CALL_MARGIN_MULTIWAY": (0.020, 0.060),
    "HEURISTIC_RISK_REQ_LOW": (0.72, 0.83),
    "HEURISTIC_RISK_REQ_MID": (0.82, 0.92),
    "HEURISTIC_RISK_REQ_HIGH": (0.89, 0.97),
    "HEURISTIC_VALUE_THRESHOLD_BASE": (0.61, 0.70),
    "HEURISTIC_THIN_VALUE_BASE": (0.54, 0.64),
    "HEURISTIC_DRY_BLUFF_PROB": (0.20, 0.62),
    "HEURISTIC_WET_BLUFF_PROB": (0.06, 0.38),
    "HEURISTIC_NORMAL_VALUE_FRACTION": (0.40, 0.70),
    "HEURISTIC_PRESSURE_VALUE_FRACTION": (0.58, 0.98),
    "HEURISTIC_OFF_BUCKET_SIZING_PROB": (0.02, 0.22),
    "HEURISTIC_EXTRA_LARGE_BET_EQUITY_PENALTY": (0.0, 0.055),
    "HEURISTIC_MIXED_PRESSURE_CALL_MARGIN_BONUS": (0.0, 0.040),
    "HEURISTIC_MIXED_PRESSURE_RISK_BONUS": (0.0, 0.070),
    "HEURISTIC_BLOCKER_BLUFF_PROB": (0.0, 0.10),
    "HEURISTIC_BLOCKER_BLUFF_MIN_EQUITY": (0.24, 0.34),
    "HEURISTIC_BLOCKER_BLUFF_FOLD_PRESSURE": (0.54, 0.68),
    "HEURISTIC_DELAYED_PROBE_PROB": (0.0, 0.28),
    "HEURISTIC_DELAYED_PROBE_MIN_EQUITY": (0.30, 0.44),
    "HEURISTIC_DELAYED_PROBE_FOLD_PRESSURE": (0.48, 0.64),
    "HEURISTIC_TOP_PAIR_VALUE_DISCOUNT": (0.0, 0.028),
    "HEURISTIC_OVERPAIR_VALUE_DISCOUNT": (0.0, 0.035),
    "HEURISTIC_BOARD_PAIR_DANGER_PENALTY": (0.0, 0.035),
}

FILLER_OPPONENTS = [
    ("oracle", "bots/strong_mocks/oracle_imitation"),
    ("ppo", "bots/strong_mocks/ppo_policy"),
    ("cfr", "bots/strong_mocks/cfr_bucket"),
    ("rollout", "bots/strong_mocks/rollout_search"),
    ("ensemble", "bots/strong_mocks/ensemble"),
    ("shark", "bots/shark/bot.py"),
    ("aggressor", "bots/aggressor/bot.py"),
    ("equity_pressure", "bots/mock_competitors/equity_pressure/bot.py"),
    ("bucket_overbet", "bots/mock_competitors/bucket_overbet/bot.py"),
]


def parse_extra_opponents(items: list[str] | None) -> list[tuple[str, str]]:
    specs = []
    for index, item in enumerate(items or []):
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).name or f"extra_{index}"
        label = "".join(ch if ch.isalnum() else "_" for ch in label) or f"extra_{index}"
        specs.append((label, path))
    return specs


def _floatish(value):
    try:
        return float(value)
    except Exception:
        return None


def _variant_dir(generated_dir: Path, run_id: str, generation: int, name: str) -> Path:
    del run_id
    return generated_dir / f"g{generation:03d}_{name}"


def write_variant(generated_dir: Path, run_id: str, generation: int, item: dict) -> str:
    target = _variant_dir(generated_dir, run_id, generation, item["name"])
    data_dir = target / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, target / "bot.py")
    payload = dict(item)
    payload["repo_root"] = str(ROOT)
    with open(data_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return str(target)


def seed_population(size: int, rng: random.Random) -> list[dict]:
    seeds = [
        {"name": "baseline", "env": {}},
        {"name": "legacy", "env": dict(CONFIGS.get("legacy-baseline", {}))},
        {"name": "pressure", "env": dict(CONFIGS.get("pressure", {}))},
        {"name": "spr_anti_bucket", "env": dict(CONFIGS.get("spr-anti-bucket", {}))},
        {"name": "pressure_control", "env": dict(CONFIGS.get("pressure-control", {}))},
        {"name": "equity_control", "env": dict(CONFIGS.get("equity-control", {}))},
    ]
    while len(seeds) < size:
        seeds.append({"name": f"random_{len(seeds)}", "env": random_env(rng)})
    return seeds[:size]


def random_env(rng: random.Random) -> dict[str, str]:
    env = {}
    for key, (lo, hi) in MUTATION_SPACE.items():
        if rng.random() < 0.55:
            env[key] = f"{rng.uniform(lo, hi):.4f}"
    return env


def mutate(parent: dict, index: int, rng: random.Random) -> dict:
    env = dict(parent.get("env", {}))
    for key, (lo, hi) in MUTATION_SPACE.items():
        if rng.random() > 0.35:
            continue
        current = _floatish(env.get(key))
        if current is None:
            current = rng.uniform(lo, hi)
        width = (hi - lo) * 0.16
        env[key] = f"{max(lo, min(hi, rng.gauss(current, width))):.4f}"
    if rng.random() < 0.18:
        key = rng.choice(list(MUTATION_SPACE))
        env.pop(key, None)
    return {"name": f"{parent['name']}_m{index}", "env": env, "parent": parent["name"]}


def _lineup(
    generated_dir: Path,
    run_id: str,
    generation: int,
    population: list[dict],
    rng: random.Random,
    min_heur: int,
    max_heur: int,
    forced: list[dict] | None = None,
    extra_opponents: list[tuple[str, str]] | None = None,
) -> dict[str, str]:
    forced = forced or []
    heuristic_count = max(len(forced), rng.randint(min_heur, max_heur))
    forced_names = {item["name"] for item in forced}
    remaining = [item for item in population if item["name"] not in forced_names]
    picked = list(forced)
    fill_count = min(max(0, heuristic_count - len(picked)), len(remaining))
    if fill_count:
        picked.extend(rng.sample(remaining, k=fill_count))
    bots = {}
    for offset, item in enumerate(picked):
        bots[f"h{offset}_{item['name']}"] = write_variant(generated_dir, run_id, generation, item)
    extra_opponents = extra_opponents or []
    filler_slots = max(0, 6 - len(bots))
    fillers = list(extra_opponents[:filler_slots])
    remaining_slots = max(0, filler_slots - len(fillers))
    base_pool = [item for item in FILLER_OPPONENTS if item not in fillers]
    if remaining_slots:
        fillers.extend(rng.sample(base_pool, k=min(remaining_slots, len(base_pool))))
    for offset, (name, path) in enumerate(fillers):
        bots[f"opp{offset}_{name}"] = path
    return bots


def _run_self_training_match(task):
    match_index = task["match_index"]
    seed = task["seed"]
    result = run_match(
        match_id=task["match_id"],
        bot_paths=task["bots"],
        n_hands=task["hands"],
        verbose=False,
        seed=seed,
    )
    return {
        "match": match_index,
        "seed": seed,
        "chip_delta": result["chip_delta"],
        "final_stacks": result["final_stacks"],
        "bot_errors": result["bot_errors"],
        "duration_s": result.get("duration_s", 0),
    }


def evaluate_generation(
    generated_dir: Path,
    run_id: str,
    generation: int,
    population: list[dict],
    args,
    rng: random.Random,
    extra_opponents: list[tuple[str, str]] | None = None,
) -> dict:
    scores = {item["name"]: [] for item in population}
    tasks = []
    coverage_queue = list(population)
    rng.shuffle(coverage_queue)
    for match_index in range(args.matches_per_generation):
        forced = []
        while coverage_queue and len(forced) < args.max_heuristics:
            forced.append(coverage_queue.pop())
        bots = _lineup(
            generated_dir,
            run_id,
            generation,
            population,
            rng,
            args.min_heuristics,
            args.max_heuristics,
            forced=forced,
            extra_opponents=extra_opponents,
        )
        seed = args.seed * 100000 + generation * 1000 + match_index
        tasks.append({
            "match_index": match_index,
            "match_id": f"selftrain_{run_id}_g{generation}_{match_index}",
            "bots": bots,
            "hands": args.hands,
            "seed": seed,
        })
    matches = map_parallel(
        _run_self_training_match,
        tasks,
        workers=args.workers,
        backend=args.parallel_backend,
    )
    matches.sort(key=lambda row: row["match"])
    for result in matches:
        for bot_id, delta in result["chip_delta"].items():
            if not bot_id.startswith("h"):
                continue
            name = bot_id.split("_", 1)[1]
            scores.setdefault(name, []).append(delta)
    ranked = []
    for item in population:
        values = scores.get(item["name"], [])
        ranked.append({
            "name": item["name"],
            "env": item.get("env", {}),
            "games": len(values),
            "mean_delta": sum(values) / len(values) if values else -100000.0,
            "min_delta": min(values) if values else -100000,
            "values": values,
        })
    ranked.sort(key=lambda row: (row["mean_delta"], row["min_delta"]), reverse=True)
    return {"generation": generation, "ranked": ranked, "matches": matches}


def next_population(ranked: list[dict], size: int, elite: int, rng: random.Random) -> list[dict]:
    elites = [{"name": row["name"], "env": row["env"]} for row in ranked[:elite]]
    population = list(elites)
    index = 0
    while len(population) < size:
        parent = rng.choice(elites)
        population.append(mutate(parent, index, rng))
        index += 1
    return population


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def resolve_output_paths(args) -> dict[str, Path]:
    """Resolve self-training outputs.

    `--run-root` is the preferred current contract. The older
    `--result-root`/`--generated-root` flags still behave as base directories
    with `run_id` appended, unless `--run-root` is also supplied.
    """
    run_root_arg = getattr(args, "run_root", None)
    result_root_arg = getattr(args, "result_root", None)
    generated_root_arg = getattr(args, "generated_root", None)

    if run_root_arg:
        run_root = _resolve_path(run_root_arg)
        generated_dir = _resolve_path(generated_root_arg) if generated_root_arg else run_root / "generated"
    elif result_root_arg:
        run_root = _resolve_path(result_root_arg) / args.run_id
        generated_dir = (
            _resolve_path(generated_root_arg) / args.run_id
            if generated_root_arg
            else run_root / "generated"
        )
    else:
        run_root = DEFAULT_RUN_ROOT / args.run_id
        generated_dir = (
            _resolve_path(generated_root_arg) / args.run_id
            if generated_root_arg
            else run_root / "generated"
        )

    return {
        "run_root": run_root,
        "result_dir": run_root,
        "generated_dir": generated_dir,
        "summary": run_root / "summary.json",
        "metrics": run_root / "metrics.jsonl",
        "plot": run_root / "ev_progress.svg",
    }


def _write_generation_metrics(metrics_path: Path, generation: int, ranked: list[dict], elite: int) -> None:
    if not ranked:
        return
    best = ranked[0]
    _append_jsonl(metrics_path, {
        "cycle": generation,
        "generation": generation,
        "series": "generation_best",
        "phase": "self_training",
        "candidate": best["name"],
        "games": best.get("games", 0),
        "mean_delta": best.get("mean_delta", 0.0),
        "min_delta": best.get("min_delta", 0),
    })
    elite_rows = ranked[: max(1, min(elite, len(ranked)))]
    elite_mean = sum(float(row.get("mean_delta", 0.0)) for row in elite_rows) / len(elite_rows)
    _append_jsonl(metrics_path, {
        "cycle": generation,
        "generation": generation,
        "series": "elite_mean",
        "phase": "self_training",
        "candidates": [row["name"] for row in elite_rows],
        "mean_delta": elite_mean,
    })


def validate_training_budget(args) -> None:
    if getattr(args, "allow_smoke", False):
        return
    if int(args.generations) < 3:
        raise SystemExit("--generations must be at least 3 unless --allow-smoke is set")
    if int(args.population) < 16:
        raise SystemExit("--population must be at least 16 unless --allow-smoke is set")
    if int(args.matches_per_generation) < 384:
        raise SystemExit("--matches-per-generation must be at least 384 unless --allow-smoke is set")
    if int(args.hands) < 400:
        raise SystemExit("--hands must be at least 400 unless --allow-smoke is set")


def run_self_training(args) -> dict:
    validate_training_budget(args)
    paths = resolve_output_paths(args)
    run_root = paths["run_root"]
    generated_dir = paths["generated_dir"]
    metrics_path = paths["metrics"]
    plot_path = paths["plot"]
    summary_path = paths["summary"]

    run_root.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")

    rng = random.Random(args.seed)
    population = seed_population(args.population, rng)
    extra_opponents = parse_extra_opponents(getattr(args, "extra_opponent", []))
    history = []
    for generation in range(args.generations):
        report = evaluate_generation(generated_dir, args.run_id, generation, population, args, rng, extra_opponents)
        save_json(run_root / f"generation_{generation:03d}.json", report)
        history.append({"generation": generation, "top": report["ranked"][: min(5, len(report["ranked"]))]})
        _write_generation_metrics(metrics_path, generation, report["ranked"], args.elite)
        population = next_population(report["ranked"], args.population, args.elite, rng)
        save_json(run_root / f"population_{generation + 1:03d}.json", {"population": population})

    plot_report = render_svg(metrics_path, plot_path)
    payload = {
        "run_id": args.run_id,
        "run_root": str(run_root),
        "generated_dir": str(generated_dir),
        "result_dir": str(paths["result_dir"]),
        "summary": str(summary_path),
        "metrics": str(metrics_path),
        "plot": str(plot_path),
        "plot_report": plot_report,
        "history": history,
        "final_population": population,
    }
    save_json(summary_path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Evolve heuristic configs through multi-heuristic 6-max self-training")
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--run-root", default=None, help="Preferred exact run output directory")
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--matches-per-generation", type=int, default=384)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--min-heuristics", type=int, default=2)
    parser.add_argument("--max-heuristics", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="Parallel match workers per generation; use 0 for auto")
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--seed", type=int, default=8080)
    parser.add_argument("--generated-root", default=None, help="Legacy generated base; run_id is appended unless --run-root is also set")
    parser.add_argument("--result-root", default=None, help="Legacy result base; run_id is appended")
    parser.add_argument("--extra-opponent", action="append", default=[])
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_self_training(args)
    print(json.dumps(payload, indent=2) if args.json else payload)


if __name__ == "__main__":
    main()
