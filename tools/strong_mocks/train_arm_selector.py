"""Train the heuristic expert-arm selector strong mock.

The policy learned here scores named expert-arm candidates. It does not choose
raw poker actions directly; expert_arms.py owns the poker heuristics and legal
risk shape, while this trainer learns which expert is best in context.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import STARTING_STACK, PokerEngine
from tools.parallel import map_parallel
from tools.strong_mocks.arm_selector_policy import (
    SELECTOR_FEATURE_NAMES,
    candidate_matrix,
    fallback_scores,
)
from tools.strong_mocks.expert_arms import ARM_NAMES, build_context, default_candidate_score
from tools.strong_mocks.policies import decide_rollout
from training.fast_match import FastBot


DEFAULT_OUTPUT = ROOT / "bots" / "strong_mocks" / "heuristic_rl_selector" / "data" / "policy.npz"

BOT_PATH_OPPONENTS = (
    ROOT / "bots" / "heuristic",
    ROOT / "bots" / "shark",
    ROOT / "bots" / "mathematician",
    ROOT / "bots" / "aggressor",
    ROOT / "bots" / "benchmarks" / "threshold_caller",
    ROOT / "bots" / "mock_competitors" / "equity_pressure",
    ROOT / "bots" / "mock_competitors" / "bucket_overbet",
)

RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK = [rank + suit for rank in RANKS for suit in SUITS]


@dataclass
class Decision:
    bot_id: str
    x: np.ndarray
    action: int
    reward: float = 0.0


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64) / max(1e-6, float(temperature))
    z -= float(np.max(z))
    exp = np.exp(z)
    total = float(np.sum(exp))
    if total <= 0:
        return np.ones_like(exp) / max(1, exp.size)
    return exp / total


def _sample_index(probs: np.ndarray, rng: random.Random) -> int:
    draw = rng.random()
    running = 0.0
    for index, prob in enumerate(probs):
        running += float(prob)
        if draw <= running:
            return index
    return int(len(probs) - 1)


def _init_params(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "weights": rng.normal(0.0, 0.01, size=len(SELECTOR_FEATURE_NAMES)).astype(np.float64),
        "bias": np.zeros(1, dtype=np.float64),
    }


def _load_policy(path: Path, seed: int, resume: bool) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mean = np.zeros(len(SELECTOR_FEATURE_NAMES), dtype=np.float64)
    scale = np.ones(len(SELECTOR_FEATURE_NAMES), dtype=np.float64)
    if resume and path.is_file():
        try:
            data = np.load(path, allow_pickle=False)
            weights = data["weights"].astype(np.float64)
            if weights.shape == (len(SELECTOR_FEATURE_NAMES),):
                params = {"weights": weights, "bias": data.get("bias", np.zeros(1, dtype=np.float32)).astype(np.float64)}
                if "mean" in data.files:
                    mean = data["mean"].astype(np.float64)
                if "scale" in data.files:
                    scale = np.maximum(1e-6, data["scale"].astype(np.float64))
                return params, mean, scale
        except Exception:
            pass
    return _init_params(seed), mean, scale


def _linear_scores(
    params: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    x: np.ndarray,
    base_scores: np.ndarray | None = None,
    learned_blend: float = 0.72,
) -> np.ndarray:
    z = (x.astype(np.float64) - mean) / np.maximum(1e-6, scale)
    learned = z @ params["weights"] + float(params["bias"][0])
    if base_scores is None:
        return learned
    return (1.0 - learned_blend) * base_scores + learned_blend * learned


def _make_match_log(target_type: str, rng: random.Random) -> list[dict]:
    patterns = {
        "folder": (0.10, 0.12, 0.58, 0.18, 0.02),
        "station": (0.12, 0.55, 0.12, 0.20, 0.01),
        "maniac": (0.48, 0.22, 0.14, 0.10, 0.06),
        "strong": (0.20, 0.30, 0.28, 0.20, 0.02),
        "unknown": (0.22, 0.28, 0.25, 0.23, 0.02),
    }
    probs = patterns.get(target_type, patterns["unknown"])
    actions = ("raise", "call", "fold", "check", "all_in")
    rows = []
    for hand_num in range(rng.randint(12, 44)):
        for bot_id in ("opp_1", "opp_2", "opp_3"):
            draw = rng.random()
            total = 0.0
            action = "check"
            for item, prob in zip(actions, probs):
                total += prob
                if draw <= total:
                    action = item
                    break
            rows.append({"hand_num": hand_num, "seat": int(bot_id[-1]), "bot_id": bot_id, "action": action})
    return rows[-200:]


def _sample_cards(rng: random.Random, count: int) -> list[str]:
    return rng.sample(DECK, count)


def sample_training_state(rng: random.Random) -> dict:
    players = []
    active_count = rng.randint(2, 6)
    hero_seat = 0
    target_type = rng.choice(["folder", "station", "maniac", "strong", "unknown"])
    for seat in range(6):
        players.append({
            "seat": seat,
            "bot_id": "hero" if seat == hero_seat else f"opp_{seat}",
            "stack": rng.randint(700, 22000),
            "is_folded": seat >= active_count,
            "is_all_in": False,
        })
    players[hero_seat]["is_folded"] = False
    players[1]["is_folded"] = False

    street = rng.choice(["preflop", "flop", "turn", "river"])
    board_count = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]
    cards = _sample_cards(rng, 2 + board_count)
    hero_cards = cards[:2]
    board = cards[2:]

    pot = rng.randint(150, 9000)
    stack = players[hero_seat]["stack"]
    facing = rng.random() < (0.45 if street != "preflop" else 0.55)
    amount_owed = 0 if not facing else min(stack, rng.randint(50, max(100, min(stack, pot + 1800))))
    can_check = amount_owed == 0
    current_bet = amount_owed if facing else 0
    min_raise_to = max(100, current_bet * 2) if facing else 100

    action_log = [
        {"seat": 1, "action": "small_blind", "amount": 50},
        {"seat": 2, "action": "big_blind", "amount": 100},
    ]
    if rng.random() < 0.35:
        action_log.append({"seat": hero_seat, "action": "raise", "amount": rng.choice([240, 280, 320])})
        action_log.append({"seat": 1, "action": "call", "amount": action_log[-1]["amount"]})
    if facing:
        action_log.append({"seat": 1, "action": "raise", "amount": current_bet})
    elif street != "preflop" and rng.random() < 0.50:
        action_log.append({"seat": 1, "action": "check"})

    return {
        "type": "action_request",
        "hand_id": f"selector_synth_{rng.randrange(1_000_000_000)}",
        "street": street,
        "seat_to_act": hero_seat,
        "pot": pot,
        "community_cards": board,
        "current_bet": current_bet,
        "min_raise_to": min_raise_to,
        "amount_owed": amount_owed,
        "can_check": can_check,
        "your_cards": hero_cards,
        "your_stack": stack,
        "your_bet_this_street": 0,
        "players": players,
        "action_log": action_log,
        "match_action_log": _make_match_log(target_type, rng),
    }


def _oracle_arm(state: dict) -> str:
    ctx = build_context(state)
    if ctx["total_stack"] <= 15 * 100 and ctx["street"] == "preflop":
        return "short_stack_pushfold"
    if ctx["spr"] <= 2.2 and (ctx["equity"] > 0.65 or ctx["strength"] > 0.82):
        return "spr_commit"
    if ctx["target_type"] == "station" and (ctx["equity"] > 0.55 or ctx["made"]):
        return "value_station"
    if ctx["target_type"] == "maniac":
        return "anti_maniac"
    if ctx["target_type"] == "folder" and ctx["can_check"] and not ctx["multiway"]:
        if ctx["draw"] and ctx["equity"] < 0.62:
            return "blocker_bluff"
        if ctx["hero_was_preflop_aggressor"] and ctx["street"] in ("flop", "turn"):
            return "range_advantage_cbet"
        return "pressure_folder"
    if ctx["can_check"] and ctx["fold_pressure"] > 0.39 and not ctx["multiway"]:
        return "pot_odds_breaker"
    if ctx["target_type"] in ("strong", "unknown") and not ctx["can_check"] and ctx["owed"] > ctx["pot"] * 0.55:
        return "strong_unknown_avoidance"
    if 0.42 <= ctx["equity"] <= 0.68:
        return "showdown_value" if not ctx["can_check"] else "pot_control"
    return "default_tag"


def _target_index_for_state(state: dict, candidates) -> int:
    arm = _oracle_arm(state)
    for index, candidate in enumerate(candidates):
        if candidate.arm == arm:
            return index
    return int(np.argmax([default_candidate_score(candidate) for candidate in candidates]))


def _bootstrap_dataset(seed: int, samples: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(max(1, samples)):
        state = sample_training_state(rng)
        candidates, matrix = candidate_matrix(state)
        target = _target_index_for_state(state, candidates)
        base = fallback_scores(candidates)
        rows.append((matrix.astype(np.float32), base.astype(np.float32), target))
    return rows


def _dataset_mean_scale(rows: list[tuple[np.ndarray, np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([row[0] for row in rows], axis=0).astype(np.float64)
    mean = flat.mean(axis=0)
    scale = np.maximum(1e-6, flat.std(axis=0))
    return mean, scale


def _eval_supervised(
    params: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    rows: list[tuple[np.ndarray, np.ndarray, int]],
    learned_blend: float,
) -> dict:
    losses = []
    correct = 0
    for matrix, base, target in rows:
        scores = _linear_scores(params, mean, scale, matrix, base, learned_blend)
        probs = _softmax(scores, 1.0)
        losses.append(-float(np.log(max(1e-8, probs[target]))))
        correct += int(np.argmax(scores) == target)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "agreement": correct / max(1, len(rows)),
    }


def _apply_selector_gradient(
    params: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    matrix: np.ndarray,
    base_scores: np.ndarray,
    target: int,
    coeff: float,
    learning_rate: float,
    learned_blend: float,
    temperature: float,
    max_grad_norm: float,
) -> float:
    scores = _linear_scores(params, mean, scale, matrix, base_scores, learned_blend)
    probs = _softmax(scores, temperature)
    selected = max(1e-8, float(probs[target]))
    grad_scores = probs
    grad_scores[target] -= 1.0
    grad_scores *= coeff * learned_blend / max(1e-6, temperature)
    z = (matrix.astype(np.float64) - mean) / np.maximum(1e-6, scale)
    grad_w = z.T @ grad_scores
    grad_b = np.asarray([float(np.sum(grad_scores))], dtype=np.float64)
    norm = float(np.sqrt(np.sum(grad_w * grad_w) + np.sum(grad_b * grad_b)))
    if max_grad_norm > 0 and norm > max_grad_norm:
        grad_w *= max_grad_norm / norm
        grad_b *= max_grad_norm / norm
    params["weights"] -= learning_rate * grad_w
    params["bias"] -= learning_rate * grad_b
    return -float(np.log(selected))


def _supervised_bootstrap(
    params: dict[str, np.ndarray],
    seed: int,
    samples: int,
    holdout: int,
    epochs: int,
    learning_rate: float,
    learned_blend: float,
    max_grad_norm: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if samples <= 0 or epochs <= 0:
        return {}, np.zeros(len(SELECTOR_FEATURE_NAMES), dtype=np.float64), np.ones(len(SELECTOR_FEATURE_NAMES), dtype=np.float64)
    train_rows = _bootstrap_dataset(seed, samples)
    hold_rows = _bootstrap_dataset(seed + 71, max(1, holdout))
    mean, scale = _dataset_mean_scale(train_rows)
    initial = _eval_supervised(params, mean, scale, hold_rows, learned_blend)
    rng = random.Random(seed + 19)
    for _epoch in range(epochs):
        rng.shuffle(train_rows)
        for matrix, base, target in train_rows:
            _apply_selector_gradient(
                params,
                mean,
                scale,
                matrix,
                base,
                target,
                coeff=1.0,
                learning_rate=learning_rate,
                learned_blend=learned_blend,
                temperature=1.0,
                max_grad_norm=max_grad_norm,
            )
    final = _eval_supervised(params, mean, scale, hold_rows, learned_blend)
    return {
        "bootstrap_initial_loss": round(initial["loss"], 6),
        "bootstrap_final_loss": round(final["loss"], 6),
        "bootstrap_initial_agreement": round(initial["agreement"], 4),
        "bootstrap_final_agreement": round(final["agreement"], 4),
        "bootstrap_samples": samples,
        "bootstrap_holdout": holdout,
        "bootstrap_epochs": epochs,
    }, mean, scale


def _selector_action(task: dict, state: dict, rng: random.Random, record: bool) -> tuple[dict, Decision | None]:
    candidates, matrix = candidate_matrix(state)
    base_scores = fallback_scores(candidates)
    scores = _linear_scores(
        task["params"],
        task["mean"],
        task["scale"],
        matrix,
        base_scores,
        task["learned_blend"],
    )
    probs = _softmax(scores, task["temperature"])
    index = _sample_index(probs, rng)
    action = candidates[index].action
    if not record:
        return action, None
    return action, Decision(
        bot_id=state["players"][state["seat_to_act"]]["bot_id"],
        x=matrix.astype(np.float32),
        action=index,
    )


def _snapshot_action(snapshot: dict, state: dict, rng: random.Random) -> dict:
    task = {
        "params": snapshot["params"],
        "mean": snapshot["mean"],
        "scale": snapshot["scale"],
        "temperature": snapshot["temperature"],
        "learned_blend": snapshot["learned_blend"],
    }
    action, _decision = _selector_action(task, state, rng, record=False)
    return action


def _opponent_pool(mode: str) -> list[dict]:
    base = [
        {"kind": "rollout", "style": "rollout_pressure"},
        {"kind": "rollout", "style": "rollout_deep"},
    ]
    if mode == "rollout":
        return base
    bot_specs = [{"kind": "bot_path", "path": str(path)} for path in BOT_PATH_OPPONENTS if path.exists()]
    if mode == "fast":
        return base + bot_specs[:4]
    return base + bot_specs


def _lineup(task: dict, rng: random.Random) -> tuple[list[str], dict[str, dict]]:
    players = int(task["players"])
    train_seats = min(players, max(1, int(task["train_seats"])))
    bot_ids = [f"train_{i}" for i in range(train_seats)]
    specs = {bot_id: {"kind": "current"} for bot_id in bot_ids}
    pool = _opponent_pool(task["opponent_pool"])
    snapshots = task.get("snapshots", [])
    while len(bot_ids) < players:
        bot_id = f"opp_{len(bot_ids)}"
        if snapshots and rng.random() < float(task["snapshot_prob"]):
            specs[bot_id] = {"kind": "snapshot", "snapshot": copy.deepcopy(rng.choice(snapshots))}
        else:
            specs[bot_id] = copy.deepcopy(rng.choice(pool))
        bot_ids.append(bot_id)
    rng.shuffle(bot_ids)
    return bot_ids, specs


def _opponent_action(spec: dict, state: dict, rng: random.Random, fast_bots: dict[str, FastBot], bot_id: str) -> dict:
    if spec["kind"] == "bot_path":
        bot = fast_bots.get(bot_id)
        return bot.act(state) if bot is not None else decide_rollout(state, "rollout_pressure")
    if spec["kind"] == "snapshot":
        return _snapshot_action(spec["snapshot"], state, rng)
    return decide_rollout(state, style=spec.get("style", "rollout_pressure"))


def _inject_match_log(state: dict, match_log: list[dict]) -> dict:
    if state.get("type") == "action_request":
        state["match_action_log"] = match_log[-200:]
    return state


def _run_training_match(task: dict) -> dict:
    rng = random.Random(int(task["seed"]))
    bot_ids, specs = _lineup(task, rng)
    fast_bots: dict[str, FastBot] = {}
    stacks = {bot_id: STARTING_STACK for bot_id in bot_ids}
    dealer = 0
    match_log: list[dict] = []
    decisions: list[Decision] = []
    hand_count = 0
    try:
        for bot_id, spec in specs.items():
            if spec["kind"] == "bot_path":
                fast_bots[bot_id] = FastBot(bot_id, spec["path"])
                fast_bots[bot_id].warmup()
        for hand_num in range(int(task["hands"])):
            alive = [bot_id for bot_id in bot_ids if stacks[bot_id] > 0]
            if len(alive) < 2:
                break
            hand_start = {bot_id: stacks[bot_id] for bot_id in alive}
            hand_decisions: list[Decision] = []
            engine = PokerEngine(
                hand_id=f"{task['match_id']}_h{hand_num:04d}",
                bot_ids=alive,
                dealer_seat=dealer % len(alive),
                starting_stacks={bot_id: stacks[bot_id] for bot_id in alive},
                seed=int(task["seed"]) * 1_000_003 + hand_num,
            )
            state = _inject_match_log(engine.start_hand(), match_log)
            steps = 0
            while state.get("type") == "action_request":
                seat = int(state["seat_to_act"])
                bot_id = alive[seat]
                spec = specs[bot_id]
                if spec["kind"] == "current":
                    action, decision = _selector_action(task, state, rng, record=True)
                    if decision is not None:
                        hand_decisions.append(decision)
                else:
                    action = _opponent_action(spec, state, rng, fast_bots, bot_id)
                match_log.append({
                    "hand_num": hand_num,
                    "seat": seat,
                    "bot_id": bot_id,
                    "action": action.get("action"),
                    "amount": action.get("amount"),
                })
                state = _inject_match_log(engine.apply_action(seat, action), match_log)
                steps += 1
                if steps > 1000:
                    raise RuntimeError(f"hand exceeded 1000 actions: {task['match_id']} {hand_num}")
            final_stacks = state.get("final_stacks", {})
            for bot_id, value in final_stacks.items():
                stacks[bot_id] = int(value)
            for decision in hand_decisions:
                start = hand_start.get(decision.bot_id, STARTING_STACK)
                final = int(final_stacks.get(decision.bot_id, start))
                reward = (final - start) / max(1.0, float(task["reward_scale"]))
                decision.reward = float(max(-task["reward_clip"], min(task["reward_clip"], reward)))
                decisions.append(decision)
            dealer += 1
            hand_count += 1
    finally:
        for bot in fast_bots.values():
            bot.stop()

    train_ids = [bot_id for bot_id in bot_ids if specs[bot_id]["kind"] == "current"]
    train_deltas = {bot_id: stacks.get(bot_id, 0) - STARTING_STACK for bot_id in train_ids}
    return {
        "decisions": [decision.__dict__ for decision in decisions],
        "hands": hand_count,
        "train_deltas": train_deltas,
        "mean_train_delta": float(sum(train_deltas.values()) / max(1, len(train_deltas))),
        "train_bust_count": sum(1 for bot_id in train_ids if stacks.get(bot_id, 0) <= 0),
        "train_seat_count": len(train_ids),
        "bot_error_count": sum(len(bot.errors) for bot in fast_bots.values()),
    }


def _flatten(matches: list[dict]) -> list[dict]:
    return [row for match in matches for row in match["decisions"]]


def _sample_replay(rows: list[dict], max_rows: int, rng: np.random.Generator) -> list[dict]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    indices = rng.choice(len(rows), size=max_rows, replace=False)
    return [rows[int(index)] for index in indices]


def _normalized_advantages(rows: list[dict]) -> np.ndarray:
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    if rewards.size == 0:
        return rewards
    std = float(np.std(rewards))
    if std < 1e-6:
        return rewards - float(np.mean(rewards))
    return (rewards - float(np.mean(rewards))) / std


def _update_policy(
    params: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    rows: list[dict],
    rng: np.random.Generator,
    learning_rate: float,
    learned_blend: float,
    temperature: float,
    epochs: int,
    max_grad_norm: float,
) -> float:
    if not rows:
        return 0.0
    losses = []
    adv = _normalized_advantages(rows)
    for _epoch in range(max(1, epochs)):
        order = rng.permutation(len(rows))
        for offset in order:
            row = rows[int(offset)]
            matrix = np.asarray(row["x"], dtype=np.float32)
            candidates_base = np.zeros(matrix.shape[0], dtype=np.float64)
            # The candidate score hint is already embedded in matrix, but the
            # default scorer is not reconstructable here; use neutral base for
            # policy-gradient updates and keep fallback blending for inference.
            loss = _apply_selector_gradient(
                params,
                mean,
                scale,
                matrix,
                candidates_base,
                int(row["action"]),
                coeff=float(adv[int(offset)]),
                learning_rate=learning_rate,
                learned_blend=learned_blend,
                temperature=temperature,
                max_grad_norm=max_grad_norm,
            )
            losses.append(loss)
    return float(np.mean(losses)) if losses else 0.0


def _snapshot(params: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray, temperature: float, learned_blend: float) -> dict:
    return {
        "params": {key: value.copy() for key, value in params.items()},
        "mean": mean.copy(),
        "scale": scale.copy(),
        "temperature": max(0.35, temperature * 0.92),
        "learned_blend": learned_blend,
    }


def _export_policy(output: Path, params: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_type=np.asarray(["heuristic_arm_selector"]),
        weights=params["weights"].astype(np.float32),
        bias=params["bias"].astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        selector_feature_names=np.asarray(SELECTOR_FEATURE_NAMES),
        arm_names=np.asarray(ARM_NAMES),
        temperature=np.asarray([report.get("temperature", 0.42)], dtype=np.float32),
        learned_blend=np.asarray([report.get("learned_blend", 0.72)], dtype=np.float32),
        generations=np.asarray([report.get("generations", 0)], dtype=np.int32),
        best_selection_score=np.asarray([report.get("best_selection_score", 0.0)], dtype=np.float32),
    )


def train(args) -> dict:
    output = Path(args.output or DEFAULT_OUTPUT)
    params, mean, scale = _load_policy(output, args.seed, args.resume)
    bootstrap, boot_mean, boot_scale = _supervised_bootstrap(
        params,
        args.seed,
        args.bootstrap_samples,
        args.bootstrap_holdout,
        args.bootstrap_epochs,
        args.bootstrap_learning_rate,
        args.learned_blend,
        args.max_grad_norm,
    )
    if bootstrap:
        mean, scale = boot_mean, boot_scale
    rng = np.random.default_rng(args.seed + 313)
    snapshots: list[dict] = []
    replay: list[dict] = []
    reports = []
    best_metric = -float("inf")
    best_state = {key: value.copy() for key, value in params.items()}
    best_row: dict | None = None

    for generation in range(args.generations):
        pre_update_state = {key: value.copy() for key, value in params.items()}
        tasks = []
        for match_index in range(args.matches_per_generation):
            tasks.append({
                "params": params,
                "mean": mean,
                "scale": scale,
                "temperature": args.temperature,
                "learned_blend": args.learned_blend,
                "hands": args.hands,
                "players": args.players,
                "train_seats": args.train_seats,
                "reward_scale": args.reward_scale,
                "reward_clip": args.reward_clip,
                "opponent_pool": args.opponent_pool,
                "snapshot_prob": args.snapshot_prob,
                "snapshots": snapshots,
                "seed": args.seed * 1_000_000 + generation * 10_000 + match_index,
                "match_id": f"arm_selector_g{generation:04d}_{match_index:04d}",
            })
        matches = map_parallel(_run_training_match, tasks, workers=args.workers, backend=args.parallel_backend)
        rows = _flatten(matches)
        train_bust_count = int(sum(match.get("train_bust_count", 0) for match in matches))
        train_seat_count = int(sum(match.get("train_seat_count", 0) for match in matches))
        mean_delta = float(np.mean([match["mean_train_delta"] for match in matches])) if matches else 0.0
        train_bust_rate = train_bust_count / max(1, train_seat_count)
        selection_score = mean_delta - args.selection_bust_penalty * train_bust_rate
        row = {
            "generation": generation,
            "kind": "heuristic_arm_selector",
            "matches": len(matches),
            "hands": int(sum(match["hands"] for match in matches)),
            "decisions": len(rows),
            "mean_train_delta": round(mean_delta, 3),
            "selection_score": round(selection_score, 3),
            "train_bust_count": train_bust_count,
            "train_bust_rate": round(train_bust_rate, 4),
            "mean_decision_reward": round(float(np.mean([item["reward"] for item in rows])) if rows else 0.0, 5),
            "bot_error_count": int(sum(match.get("bot_error_count", 0) for match in matches)),
        }
        if rows:
            action_counts = np.bincount([int(item["action"]) for item in rows], minlength=len(ARM_NAMES))
            row["arm_counts"] = {ARM_NAMES[i]: int(action_counts[i]) for i in range(len(ARM_NAMES)) if int(action_counts[i]) > 0}
        if generation >= args.selection_warmup and selection_score > best_metric:
            best_metric = selection_score
            best_state = pre_update_state
            best_row = dict(row)
        replay.extend(rows)
        replay = replay[-max(1, args.replay_max_decisions):]
        update_rows = _sample_replay(replay, args.replay_max_decisions, rng)
        loss = _update_policy(
            params,
            mean,
            scale,
            update_rows,
            rng,
            args.learning_rate,
            args.learned_blend,
            args.temperature,
            args.epochs,
            args.max_grad_norm,
        )
        row["update_loss"] = round(float(loss), 6)
        reports.append(row)
        if args.progress:
            print(json.dumps(row), file=sys.stderr)
        if args.snapshot_interval > 0 and (generation + 1) % args.snapshot_interval == 0:
            snapshots.append(_snapshot(params, mean, scale, args.temperature, args.learned_blend))
            snapshots = snapshots[-args.max_snapshots:]

    selected_row = best_row if best_row is not None else (reports[-1] if reports else None)
    if best_row is None:
        best_state = {key: value.copy() for key, value in params.items()}
    report = {
        "mode": "heuristic_guided_selector_self_play",
        "kind": "heuristic_arm_selector",
        "output": str(output),
        "seed": args.seed,
        "arms": len(ARM_NAMES),
        "generations": len(reports),
        "matches_per_generation": args.matches_per_generation,
        "hands": args.hands,
        "players": args.players,
        "train_seats": args.train_seats,
        "temperature": args.temperature,
        "learned_blend": args.learned_blend,
        "best_generation": selected_row["generation"] if selected_row else None,
        "best_selection_score": selected_row["selection_score"] if selected_row else None,
        "best_mean_train_delta": selected_row["mean_train_delta"] if selected_row else None,
        "final_mean_train_delta": reports[-1]["mean_train_delta"] if reports else 0.0,
        "reports": reports,
        **bootstrap,
    }
    _export_policy(output, best_state, mean, scale, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heuristic expert-arm selector strong mock")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--matches-per-generation", type=int, default=40)
    parser.add_argument("--hands", type=int, default=160)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--train-seats", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--bootstrap-learning-rate", type=float, default=0.020)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=0.46)
    parser.add_argument("--learned-blend", type=float, default=0.72)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--reward-clip", type=float, default=8.0)
    parser.add_argument("--selection-warmup", type=int, default=0)
    parser.add_argument("--selection-bust-penalty", type=float, default=9000.0)
    parser.add_argument("--opponent-pool", choices=["rollout", "fast", "adversarial"], default="fast")
    parser.add_argument("--snapshot-interval", type=int, default=2)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument("--snapshot-prob", type=float, default=0.25)
    parser.add_argument("--replay-max-decisions", type=int, default=36000)
    parser.add_argument("--bootstrap-samples", type=int, default=12000)
    parser.add_argument("--bootstrap-holdout", type=int, default=2400)
    parser.add_argument("--bootstrap-epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--seed", type=int, default=8181)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.train_seats <= args.players:
        raise SystemExit("--train-seats must be within player count")
    result = train(args)
    print(json.dumps(result, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
