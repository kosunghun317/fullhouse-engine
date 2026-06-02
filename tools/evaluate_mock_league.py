"""Evaluate mock opponents against each other in local leagues."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import STARTING_STACK  # noqa: E402
from sandbox.match import run_match  # noqa: E402
from tools.parallel import map_parallel  # noqa: E402
from tools.portal.mock_generation import load_mock_manifest, selected_profile_rows  # noqa: E402


BUILTIN_POOLS = {
    "benchmark": {
        "always_call": "bots/benchmarks/always_call/bot.py",
        "overfold": "bots/benchmarks/overfold/bot.py",
        "jammer": "bots/benchmarks/jammer/bot.py",
        "minraiser": "bots/benchmarks/minraiser/bot.py",
        "half_pot_pressure": "bots/benchmarks/half_pot_pressure/bot.py",
        "threshold_caller": "bots/benchmarks/threshold_caller/bot.py",
    },
    "mock_family": {
        "equity_mc": "bots/mock_competitors/equity_mc/bot.py",
        "equity_tight": "bots/mock_competitors/equity_tight/bot.py",
        "equity_loose": "bots/mock_competitors/equity_loose/bot.py",
        "equity_pressure": "bots/mock_competitors/equity_pressure/bot.py",
        "bucket_policy": "bots/mock_competitors/bucket_policy/bot.py",
        "bucket_overbet": "bots/mock_competitors/bucket_overbet/bot.py",
        "bucket_mixed": "bots/mock_competitors/bucket_mixed/bot.py",
        "opponent_modeler": "bots/mock_competitors/opponent_modeler/bot.py",
    },
}


def portal_pool(mocks_dir: Path, profiles: list[str] | None = None) -> dict[str, str]:
    manifest = load_mock_manifest(mocks_dir)
    rows = selected_profile_rows(manifest, profiles)
    return {row["bot_id"]: row["path"] for row in rows}


def _rotate(items: list[tuple[str, str]], shift: int) -> list[tuple[str, str]]:
    if not items:
        return items
    shift = shift % len(items)
    return items[shift:] + items[:shift]


def _percentile(values: list[float], pct: float) -> float:
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


def _bootstrap_mean_ci(values: list[int], samples: int = 1000, seed: int = 1729) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1 or samples <= 0:
        mean = float(values[0])
        return [mean, mean]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        total = 0
        for _idx in range(len(values)):
            total += values[rng.randrange(len(values))]
        means.append(total / len(values))
    return [round(_percentile(means, 0.025), 2), round(_percentile(means, 0.975), 2)]


def sixmax_specs(
    pool: dict[str, str],
    table_size: int = 6,
    max_tables: int | None = None,
    seat_rotations: int = 1,
    non_contiguous: bool = True,
    table_seed: int = 1729,
) -> list[dict]:
    items = list(pool.items())
    if non_contiguous:
        random.Random(int(table_seed)).shuffle(items)
    if len(items) < 2:
        raise ValueError("need at least two bots for a league")
    rotations = max(1, int(seat_rotations))
    if len(items) <= table_size:
        return [
            {"name": f"league_sixmax_1_r{rotation + 1}", "bots": dict(_rotate(items, rotation))}
            for rotation in range(rotations)
        ]
    limit = min(len(items), max_tables or len(items))
    stride = max(1, len(items) // table_size) if non_contiguous else 1
    specs = []
    for start in range(limit):
        chunk = [items[(start + offset * stride) % len(items)] for offset in range(table_size)]
        for rotation in range(rotations):
            specs.append({
                "name": f"league_sixmax_{start + 1}_r{rotation + 1}",
                "bots": dict(_rotate(chunk, rotation)),
            })
    return specs


def heads_up_specs(pool: dict[str, str], max_pairs: int | None = None, seat_rotations: int = 1) -> list[dict]:
    items = list(pool.items())
    pairs = list(combinations(items, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    specs = []
    rotations = max(1, int(seat_rotations))
    for left, right in pairs:
        orders = [(left, right)]
        if rotations > 1:
            orders.append((right, left))
        for index, order in enumerate(orders[:rotations]):
            specs.append({
                "name": f"league_hu_{order[0][0]}_vs_{order[1][0]}_r{index + 1}",
                "bots": {order[0][0]: order[0][1], order[1][0]: order[1][1]},
            })
    return specs


def build_specs(
    pool: dict[str, str],
    mode: str,
    table_size: int = 6,
    max_tables: int | None = None,
    max_pairs: int | None = None,
    seat_rotations: int = 1,
    non_contiguous: bool = True,
    table_seed: int = 1729,
) -> list[dict]:
    if mode == "sixmax":
        return sixmax_specs(
            pool,
            table_size=table_size,
            max_tables=max_tables,
            seat_rotations=seat_rotations,
            non_contiguous=non_contiguous,
            table_seed=table_seed,
        )
    if mode == "heads-up":
        return heads_up_specs(pool, max_pairs=max_pairs, seat_rotations=seat_rotations)
    specs = sixmax_specs(
        pool,
        table_size=table_size,
        max_tables=max_tables,
        seat_rotations=seat_rotations,
        non_contiguous=non_contiguous,
        table_seed=table_seed,
    )
    specs.extend(heads_up_specs(pool, max_pairs=max_pairs, seat_rotations=seat_rotations))
    return specs


def _run_task(task: dict) -> dict:
    result = run_match(
        match_id=f"{task['suite']}_{task['seed']}",
        bot_paths=task["bots"],
        n_hands=task["hands"],
        verbose=False,
        seed=task["seed"],
    )
    return {
        "suite": task["suite"],
        "seed": task["seed"],
        "chip_delta": result["chip_delta"],
        "final_stacks": result["final_stacks"],
        "bot_errors": result["bot_errors"],
        "duration_s": result["duration_s"],
    }


def summarize_bot_results(
    results: list[dict],
    bust_penalty: float = 9000.0,
    stdev_penalty: float = 0.15,
    error_penalty: float = 25000.0,
    bootstrap_samples: int = 1000,
) -> list[dict]:
    bot_ids = sorted({bot_id for row in results for bot_id in row["chip_delta"]})
    ranking = []
    for bot_id in bot_ids:
        rows = [row for row in results if bot_id in row["chip_delta"]]
        deltas = [int(row["chip_delta"][bot_id]) for row in rows]
        stacks = [int(row["final_stacks"].get(bot_id, 0)) for row in rows]
        errors = sum(len(row["bot_errors"].get(bot_id, [])) for row in rows)
        busts = sum(1 for stack in stacks if stack <= 0)
        wins = 0
        for row in rows:
            best_stack = max(int(value) for value in row["final_stacks"].values())
            wins += int(int(row["final_stacks"].get(bot_id, 0)) == best_stack and best_stack > 0)
        mean = statistics.mean(deltas) if deltas else 0.0
        stdev = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        bust_rate = busts / max(1, len(rows))
        error_rate = errors / max(1, len(rows))
        mean_ci = _bootstrap_mean_ci(deltas, samples=bootstrap_samples, seed=1729 + sum(ord(ch) for ch in bot_id))
        reliability_weight = (len(rows) / max(1, len(results))) ** 0.5
        reliability_weight *= max(0.05, 1.0 - min(0.95, stdev / max(1.0, STARTING_STACK * 3)))
        score = mean - stdev_penalty * stdev - bust_penalty * bust_rate - error_penalty * error_rate
        ranking.append({
            "bot_id": bot_id,
            "runs": len(rows),
            "mean_delta": round(float(mean), 2),
            "median_delta": round(float(statistics.median(deltas)), 2) if deltas else 0.0,
            "min_delta": min(deltas) if deltas else 0,
            "max_delta": max(deltas) if deltas else 0,
            "stdev_delta": round(float(stdev), 2),
            "mean_delta_ci": mean_ci,
            "reliability_weight": round(float(reliability_weight), 5),
            "win_count": wins,
            "bust_count": busts,
            "bust_rate": round(float(bust_rate), 5),
            "error_count": errors,
            "score": round(float(score), 3),
            "reliability_adjusted_score": round(float(score * reliability_weight), 3),
        })
    return sorted(ranking, key=lambda item: (item["reliability_adjusted_score"], item["score"]), reverse=True)


def run_league(args) -> dict:
    pool = {}
    if args.builtin_pool:
        pool.update(BUILTIN_POOLS[args.builtin_pool])
    if args.portal_mocks_dir:
        pool.update(portal_pool(Path(args.portal_mocks_dir), args.profile))
    if args.bot:
        for raw in args.bot:
            if "=" not in raw:
                raise SystemExit(f"invalid --bot {raw!r}; expected id=path")
            bot_id, path = raw.split("=", 1)
            pool[bot_id.strip()] = path.strip()
    if len(pool) < 2:
        raise SystemExit("league needs at least two bots")

    specs = build_specs(
        pool,
        args.mode,
        table_size=args.table_size,
        max_tables=args.max_tables,
        max_pairs=args.max_pairs,
        seat_rotations=args.seat_rotations,
        non_contiguous=not args.contiguous_tables,
        table_seed=args.table_seed,
    )
    seeds = [int(part) for part in args.seeds.split(",")] if args.seeds else list(range(args.seed_start, args.seed_start + args.seed_count))
    tasks = [
        {"suite": spec["name"], "bots": spec["bots"], "seed": seed, "hands": args.hands}
        for spec in specs
        for seed in seeds
    ]
    results = map_parallel(_run_task, tasks, workers=args.workers, backend=args.parallel_backend)
    report = {
        "mode": "mock_league",
        "pool_size": len(pool),
        "league_mode": args.mode,
        "hands": args.hands,
        "seeds": seeds,
        "suite_count": len(specs),
        "suites": specs,
        "ranking": summarize_bot_results(
            results,
            bust_penalty=args.bust_penalty,
            stdev_penalty=args.stdev_penalty,
            error_penalty=args.error_penalty,
            bootstrap_samples=args.bootstrap_samples,
        ),
        "runs": [] if args.summary_only else results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builtin-pool", choices=sorted(BUILTIN_POOLS))
    parser.add_argument("--portal-mocks-dir", type=Path)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--bot", action="append", help="Additional bot as id=path")
    parser.add_argument("--mode", choices=["sixmax", "heads-up", "both"], default="sixmax")
    parser.add_argument("--table-size", type=int, default=6)
    parser.add_argument("--max-tables", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--seat-rotations", type=int, default=1)
    parser.add_argument("--contiguous-tables", action="store_true")
    parser.add_argument("--table-seed", type=int, default=1729)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seeds")
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--bust-penalty", type=float, default=9000.0)
    parser.add_argument("--stdev-penalty", type=float, default=0.15)
    parser.add_argument("--error-penalty", type=float, default=25000.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_league(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    for row in report["ranking"]:
        print(
            f"{row['bot_id']}: score={row['score']} mean={row['mean_delta']} "
            f"median={row['median_delta']} wins={row['win_count']}/{row['runs']} "
            f"bust={row['bust_rate']:.3f} errors={row['error_count']}"
        )


if __name__ == "__main__":
    main()
