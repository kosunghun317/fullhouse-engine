"""Train strong mock policies from real Fullhouse self-play rollouts.

This is intentionally benchmark-only. It runs the local Fullhouse engine
in-process, collects decisions and chip-delta rewards, then exports policy
artifacts compatible with bots/strong_mocks/{ppo_policy,cfr_bucket}.
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
from tools.strong_mocks.actions import ACTION_LABELS, action_index_to_action, legal_mask
from tools.strong_mocks.abstractions import abstract_bucket_id
from tools.strong_mocks.dataset import oracle_logits
from tools.strong_mocks.features import FEATURE_NAMES, extract_features
from tools.strong_mocks.policies import decide_rollout, logits_from_model


DEFAULT_PPO_OUTPUT = ROOT / "bots" / "strong_mocks" / "ppo_policy" / "data" / "policy.npz"
DEFAULT_BUCKET_OUTPUT = ROOT / "bots" / "strong_mocks" / "cfr_bucket" / "data" / "policy.npz"

ORACLE_STYLES = ("balanced", "value", "bluff", "station", "folder", "pressure")
MODEL_DIRS = {
    "oracle_model": ROOT / "bots" / "strong_mocks" / "oracle_imitation" / "data",
    "ppo_model": ROOT / "bots" / "strong_mocks" / "ppo_policy" / "data",
    "cfr_model": ROOT / "bots" / "strong_mocks" / "cfr_bucket" / "data",
}


@dataclass
class Decision:
    bot_id: str
    x: np.ndarray
    mask: np.ndarray
    action: int
    bucket: int
    reward: float = 0.0


def _softmax_masked(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).copy()
    values[~mask] = -30.0
    values /= max(1e-6, float(temperature))
    values -= np.max(values)
    exp = np.exp(values)
    exp[~mask] = 0.0
    total = float(np.sum(exp))
    if total <= 0:
        probs = np.zeros_like(exp)
        legal = np.flatnonzero(mask)
        probs[legal] = 1.0 / max(1, len(legal))
        return probs
    return exp / total


def _sample_index(probs: np.ndarray, rng: random.Random) -> int:
    draw = rng.random()
    total = 0.0
    for index, prob in enumerate(probs):
        total += float(prob)
        if draw <= total:
            return index
    return int(len(probs) - 1)


def _mlp_init(seed: int, hidden: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, 0.10, size=(len(FEATURE_NAMES), hidden)).astype(np.float64),
        "b1": np.zeros(hidden, dtype=np.float64),
        "w2": rng.normal(0.0, 0.08, size=(hidden, len(ACTION_LABELS))).astype(np.float64),
        "b2": np.zeros(len(ACTION_LABELS), dtype=np.float64),
    }


def _load_ppo(path: Path, hidden: int, seed: int, resume: bool) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mean = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    scale = np.ones(len(FEATURE_NAMES), dtype=np.float64)
    if resume and path.is_file():
        try:
            data = np.load(path, allow_pickle=False)
            params = {
                "w1": data["w1"].astype(np.float64),
                "b1": data["b1"].astype(np.float64),
                "w2": data["w2"].astype(np.float64),
                "b2": data["b2"].astype(np.float64),
            }
            if params["w1"].shape == (len(FEATURE_NAMES), hidden):
                if "mean" in data.files:
                    mean = data["mean"].astype(np.float64)
                if "scale" in data.files:
                    scale = np.maximum(1e-6, data["scale"].astype(np.float64))
                return params, mean, scale
        except Exception:
            pass
    return _mlp_init(seed, hidden), mean, scale


def _load_bucket(path: Path, bucket_count: int, seed: int, resume: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prefs = rng.normal(0.0, 0.002, size=(bucket_count, len(ACTION_LABELS))).astype(np.float64)
    strategy_sum = np.zeros_like(prefs)
    if resume and path.is_file():
        try:
            data = np.load(path, allow_pickle=False)
            if "regrets" in data.files and data["regrets"].shape == prefs.shape:
                prefs = data["regrets"].astype(np.float64)
                if "strategy_sum" in data.files and data["strategy_sum"].shape == prefs.shape:
                    strategy_sum = data["strategy_sum"].astype(np.float64)
                return prefs, strategy_sum
            if "policy_table" in data.files and data["policy_table"].shape == prefs.shape:
                policy = data["policy_table"].astype(np.float64)
                return np.log(np.maximum(1e-6, policy)), policy.copy()
        except Exception:
            pass
    return prefs, strategy_sum


def _forward(params: dict[str, np.ndarray], x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = (x - mean) / np.maximum(1e-6, scale)
    hidden = np.tanh(z @ params["w1"] + params["b1"])
    logits = hidden @ params["w2"] + params["b2"]
    return hidden, logits


def _policy_action(task: dict, state: dict, rng: random.Random, record: bool) -> tuple[dict, Decision | None]:
    kind = task["kind"]
    x = extract_features(state)
    mask = legal_mask(state)
    if kind == "ppo":
        _hidden, logits = _forward(task["params"], x, task["mean"], task["scale"])
    else:
        b = abstract_bucket_id(state, int(task["bucket_count"]), task.get("abstraction", "feature"))
        logits = np.maximum(task["prefs"][b], 0.0) if task.get("bucket_update") == "cfr-plus" else task["prefs"][b]
    probs = _softmax_masked(logits, mask, task["temperature"])
    action_index = _sample_index(probs, rng)
    action = action_index_to_action(state, action_index)
    if not record:
        return action, None
    return action, Decision(
        bot_id=state["players"][state["seat_to_act"]]["bot_id"],
        x=x.astype(np.float32),
        mask=mask.astype(bool),
        action=action_index,
        bucket=abstract_bucket_id(state, int(task["bucket_count"]), task.get("abstraction", "feature")),
    )


def _snapshot_action(snapshot: dict, state: dict, rng: random.Random) -> dict:
    x = extract_features(state)
    mask = legal_mask(state)
    if snapshot["kind"] == "ppo":
        _hidden, logits = _forward(snapshot["params"], x, snapshot["mean"], snapshot["scale"])
    else:
        b = abstract_bucket_id(state, int(snapshot["bucket_count"]), snapshot.get("abstraction", "feature"))
        logits = np.maximum(snapshot["prefs"][b], 0.0) if snapshot.get("bucket_update") == "cfr-plus" else snapshot["prefs"][b]
    probs = _softmax_masked(logits, mask, snapshot.get("temperature", 0.80))
    return action_index_to_action(state, _sample_index(probs, rng))


def _oracle_action(style: str, state: dict, rng: random.Random) -> dict:
    x = extract_features(state)
    logits = oracle_logits(x.reshape(1, -1), style=style)[0]
    probs = _softmax_masked(logits, legal_mask(state), 0.85)
    return action_index_to_action(state, _sample_index(probs, rng))


def _model_action(name: str, state: dict) -> dict:
    data_dir = MODEL_DIRS.get(name)
    if data_dir is None:
        return _oracle_action("balanced", state, random.Random(0))
    logits = logits_from_model(state, str(data_dir), fallback_style="pressure")
    values = np.asarray(logits, dtype=float)
    values[~legal_mask(state)] = -1e9
    return action_index_to_action(state, int(np.argmax(values)))


def _opponent_action(spec: dict, state: dict, rng: random.Random) -> dict:
    kind = spec["kind"]
    if kind == "oracle":
        return _oracle_action(spec["style"], state, rng)
    if kind == "rollout":
        return decide_rollout(state, style=spec["style"])
    if kind == "model":
        return _model_action(spec["name"], state)
    if kind == "snapshot":
        return _snapshot_action(spec["snapshot"], state, rng)
    return _oracle_action("balanced", state, rng)


def _opponent_pool(mode: str) -> list[dict]:
    base = [{"kind": "oracle", "style": style} for style in ORACLE_STYLES]
    if mode == "oracle":
        return base
    if mode == "fast":
        return base + [{"kind": "model", "name": "oracle_model"}]
    return base + [
        {"kind": "rollout", "style": "rollout_pressure"},
        {"kind": "rollout", "style": "rollout_deep"},
        {"kind": "model", "name": "oracle_model"},
        {"kind": "model", "name": "cfr_model"},
        {"kind": "model", "name": "ppo_model"},
    ]


def _lineup(task: dict, rng: random.Random) -> tuple[list[str], dict[str, dict]]:
    players = int(task["players"])
    train_seats = min(players, max(1, int(task["train_seats"])))
    bot_ids = [f"train_{i}" for i in range(train_seats)]
    specs = {bid: {"kind": "current"} for bid in bot_ids}
    pool = _opponent_pool(task["opponent_pool"])
    snapshots = task.get("snapshots", [])
    while len(bot_ids) < players:
        bot_id = f"opp_{len(bot_ids)}"
        if snapshots and rng.random() < float(task["snapshot_prob"]):
            snapshot = rng.choice(snapshots)
            specs[bot_id] = {"kind": "snapshot", "snapshot": snapshot}
        else:
            specs[bot_id] = copy.deepcopy(rng.choice(pool))
        bot_ids.append(bot_id)
    rng.shuffle(bot_ids)
    return bot_ids, specs


def _inject_match_log(state: dict, match_log: list[dict]) -> dict:
    if state.get("type") == "action_request":
        state["match_action_log"] = match_log[-200:]
    return state


def _run_training_match(task: dict) -> dict:
    rng = random.Random(int(task["seed"]))
    bot_ids, specs = _lineup(task, rng)
    stacks = {bot_id: STARTING_STACK for bot_id in bot_ids}
    dealer = 0
    match_log: list[dict] = []
    decisions: list[Decision] = []
    hand_count = 0

    for hand_num in range(int(task["hands"])):
        alive = [bot_id for bot_id in bot_ids if stacks[bot_id] > 0]
        if len(alive) < 2:
            break
        hand_start = {bot_id: stacks[bot_id] for bot_id in alive}
        hand_decisions: list[Decision] = []
        hand_seed = int(task["seed"]) * 1_000_003 + hand_num
        engine = PokerEngine(
            hand_id=f"{task['match_id']}_h{hand_num:04d}",
            bot_ids=alive,
            dealer_seat=dealer % len(alive),
            starting_stacks={bot_id: stacks[bot_id] for bot_id in alive},
            seed=hand_seed,
        )
        state = _inject_match_log(engine.start_hand(), match_log)
        steps = 0
        while state.get("type") == "action_request":
            seat = int(state["seat_to_act"])
            bot_id = alive[seat]
            spec = specs[bot_id]
            if spec["kind"] == "current":
                action, decision = _policy_action(task, state, rng, record=True)
                if decision is not None:
                    hand_decisions.append(decision)
            else:
                action = _opponent_action(spec, state, rng)
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

    train_ids = [bot_id for bot_id in bot_ids if specs[bot_id]["kind"] == "current"]
    train_deltas = {bot_id: stacks.get(bot_id, 0) - STARTING_STACK for bot_id in train_ids}
    return {
        "decisions": [
            {
                "x": decision.x,
                "mask": decision.mask,
                "action": decision.action,
                "bucket": decision.bucket,
                "reward": decision.reward,
            }
            for decision in decisions
        ],
        "hands": hand_count,
        "train_deltas": train_deltas,
        "mean_train_delta": float(sum(train_deltas.values()) / max(1, len(train_deltas))),
    }


def _flatten_decisions(matches: list[dict]) -> dict[str, np.ndarray]:
    rows = [item for match in matches for item in match["decisions"]]
    if not rows:
        return {
            "x": np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64),
            "mask": np.zeros((0, len(ACTION_LABELS)), dtype=bool),
            "action": np.zeros(0, dtype=np.int64),
            "bucket": np.zeros(0, dtype=np.int64),
            "reward": np.zeros(0, dtype=np.float64),
        }
    return {
        "x": np.stack([row["x"] for row in rows]).astype(np.float64),
        "mask": np.stack([row["mask"] for row in rows]).astype(bool),
        "action": np.asarray([row["action"] for row in rows], dtype=np.int64),
        "bucket": np.asarray([row["bucket"] for row in rows], dtype=np.int64),
        "reward": np.asarray([row["reward"] for row in rows], dtype=np.float64),
    }


def _normalized_advantage(rewards: np.ndarray) -> np.ndarray:
    if rewards.size == 0:
        return rewards
    clipped = np.asarray(rewards, dtype=np.float64)
    std = float(np.std(clipped))
    if std < 1e-6:
        return clipped - float(np.mean(clipped))
    return (clipped - float(np.mean(clipped))) / std


def _update_ppo(
    params: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    data: dict[str, np.ndarray],
    rng: np.random.Generator,
    learning_rate: float,
    entropy_coef: float,
    epochs: int,
    batch_size: int,
) -> float:
    if data["x"].shape[0] == 0:
        return 0.0
    x = data["x"]
    masks = data["mask"]
    actions = data["action"]
    adv = _normalized_advantage(data["reward"])
    n = x.shape[0]
    losses = []
    for _epoch in range(max(1, epochs)):
        order = rng.permutation(n)
        for start in range(0, n, max(1, batch_size)):
            idx = order[start:start + max(1, batch_size)]
            xb = x[idx]
            maskb = masks[idx]
            actionb = actions[idx]
            advb = adv[idx]
            hidden, logits = _forward(params, xb, mean, scale)
            probs = np.zeros_like(logits, dtype=np.float64)
            for row in range(logits.shape[0]):
                probs[row] = _softmax_masked(logits[row], maskb[row], 1.0)
            selected = np.maximum(1e-8, probs[np.arange(len(idx)), actionb])
            losses.append(float(-np.mean(advb * np.log(selected))))

            grad_logits = probs.copy()
            grad_logits[np.arange(len(idx)), actionb] -= 1.0
            grad_logits *= advb[:, None]
            if entropy_coef:
                entropy_grad = probs * (np.log(np.maximum(1e-8, probs)) + 1.0)
                entropy_grad[~maskb] = 0.0
                grad_logits += entropy_coef * entropy_grad
            grad_logits /= max(1, len(idx))

            z = (xb - mean) / np.maximum(1e-6, scale)
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ params["w2"].T
            grad_pre = grad_hidden * (1.0 - hidden * hidden)
            grad_w1 = z.T @ grad_pre
            grad_b1 = grad_pre.sum(axis=0)

            params["w2"] -= learning_rate * grad_w2
            params["b2"] -= learning_rate * grad_b2
            params["w1"] -= learning_rate * grad_w1
            params["b1"] -= learning_rate * grad_b1
    return float(np.mean(losses)) if losses else 0.0


def _update_bucket(
    prefs: np.ndarray,
    strategy_sum: np.ndarray,
    data: dict[str, np.ndarray],
    learning_rate: float,
    update_rule: str,
    iteration_weight: float,
) -> float:
    if data["bucket"].shape[0] == 0:
        return 0.0
    adv = _normalized_advantage(data["reward"])
    total_abs = 0.0
    for b, mask, action, value in zip(data["bucket"], data["mask"], data["action"], adv):
        logits = np.maximum(prefs[int(b)], 0.0) if update_rule == "cfr-plus" else prefs[int(b)]
        probs = _softmax_masked(logits, mask, 1.0)
        update = -probs
        update[int(action)] += 1.0
        if update_rule == "cfr-plus":
            prefs[int(b)] = np.maximum(0.0, prefs[int(b)] + learning_rate * float(value) * update)
            strategy_sum[int(b)] += iteration_weight * probs
        else:
            prefs[int(b)] += learning_rate * float(value) * update
            strategy_sum[int(b)] += probs
        total_abs += abs(float(value))
    return total_abs / max(1, len(adv))


def _export_ppo(output: Path, params: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_type=np.asarray(["ppo_mlp"]),
        w1=params["w1"].astype(np.float32),
        b1=params["b1"].astype(np.float32),
        w2=params["w2"].astype(np.float32),
        b2=params["b2"].astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        classes=np.arange(len(ACTION_LABELS), dtype=np.int8),
        feature_names=np.asarray(FEATURE_NAMES),
        action_labels=np.asarray(ACTION_LABELS),
        train_accuracy=np.asarray([0.0], dtype=np.float32),
        average_reward=np.asarray([report.get("mean_train_delta", 0.0)], dtype=np.float32),
        generations=np.asarray([report.get("generations", 0)], dtype=np.int32),
    )


def _export_bucket(output: Path, prefs: np.ndarray, strategy_sum: np.ndarray, report: dict) -> None:
    fallback_logits = np.maximum(prefs, 0.0) if report.get("bucket_update") == "cfr-plus" else prefs
    positive = np.exp(np.clip(fallback_logits - fallback_logits.max(axis=1, keepdims=True), -30, 30))
    fallback_policy = positive / np.maximum(1e-9, positive.sum(axis=1, keepdims=True))
    summed = np.maximum(0.0, strategy_sum)
    denom = summed.sum(axis=1, keepdims=True)
    policy = np.divide(summed, np.maximum(1e-9, denom), out=fallback_policy.copy(), where=denom > 1e-9)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_type=np.asarray(["cfr_table"]),
        policy_table=policy.astype(np.float32),
        regrets=prefs.astype(np.float32),
        strategy_sum=strategy_sum.astype(np.float32),
        action_labels=np.asarray(ACTION_LABELS),
        abstraction=np.asarray([report.get("abstraction", "feature")]),
        bucket_update=np.asarray([report.get("bucket_update", "policy-gradient")]),
        bucket_count=np.asarray([prefs.shape[0]], dtype=np.int32),
        iterations=np.asarray([report.get("generations", 0)], dtype=np.int32),
        batch_size=np.asarray([report.get("matches_per_generation", 0)], dtype=np.int32),
        seed=np.asarray([report.get("seed", 0)], dtype=np.int64),
    )


def _snapshot(kind: str, params: dict | None, mean: np.ndarray, scale: np.ndarray, prefs: np.ndarray | None, args) -> dict:
    if kind == "ppo":
        return {
            "kind": "ppo",
            "params": {key: value.copy() for key, value in params.items()},
            "mean": mean.copy(),
            "scale": scale.copy(),
            "bucket_count": args.bucket_count,
            "temperature": max(0.55, args.temperature * 0.85),
        }
    return {
        "kind": "bucket",
        "prefs": prefs.copy(),
        "bucket_count": args.bucket_count,
        "abstraction": args.abstraction,
        "bucket_update": args.bucket_update,
        "temperature": max(0.55, args.temperature * 0.85),
    }


def _append_jsonl(path: str | None, row: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def train(args) -> dict:
    output = Path(args.output or (DEFAULT_PPO_OUTPUT if args.kind == "ppo" else DEFAULT_BUCKET_OUTPUT))
    params = None
    mean = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    scale = np.ones(len(FEATURE_NAMES), dtype=np.float64)
    prefs = None
    strategy_sum = None
    if args.kind == "ppo":
        params, mean, scale = _load_ppo(output, args.hidden, args.seed, args.resume)
    else:
        prefs, strategy_sum = _load_bucket(output, args.bucket_count, args.seed, args.resume)

    rng = np.random.default_rng(args.seed)
    snapshots: list[dict] = []
    reports = []
    for generation in range(args.generations):
        tasks = []
        for match_index in range(args.matches_per_generation):
            task = {
                "kind": args.kind,
                "params": params,
                "mean": mean,
                "scale": scale,
                "prefs": prefs,
                "strategy_sum": strategy_sum,
                "bucket_count": args.bucket_count,
                "abstraction": args.abstraction,
                "bucket_update": args.bucket_update,
                "temperature": args.temperature,
                "hands": args.hands,
                "players": args.players,
                "train_seats": args.train_seats,
                "reward_scale": args.reward_scale,
                "reward_clip": args.reward_clip,
                "opponent_pool": args.opponent_pool,
                "snapshot_prob": args.snapshot_prob,
                "snapshots": snapshots,
                "seed": args.seed * 1_000_000 + generation * 10_000 + match_index,
                "match_id": f"realtrain_{args.kind}_g{generation:04d}_{match_index:04d}",
            }
            tasks.append(task)
        matches = map_parallel(
            _run_training_match,
            tasks,
            workers=args.workers,
            backend=args.parallel_backend,
        )
        data = _flatten_decisions(matches)
        if args.kind == "ppo":
            loss = _update_ppo(
                params,
                mean,
                scale,
                data,
                rng,
                args.learning_rate,
                args.entropy_coef,
                args.epochs,
                args.batch_size,
            )
        else:
            loss = _update_bucket(
                prefs,
                strategy_sum,
                data,
                args.learning_rate,
                args.bucket_update,
                generation + 1,
            )
        mean_delta = float(np.mean([match["mean_train_delta"] for match in matches])) if matches else 0.0
        row = {
            "generation": generation,
            "kind": args.kind,
            "matches": len(matches),
            "hands": int(sum(match["hands"] for match in matches)),
            "decisions": int(data["reward"].shape[0]),
            "mean_train_delta": round(mean_delta, 3),
            "mean_decision_reward": round(float(np.mean(data["reward"])) if data["reward"].size else 0.0, 5),
            "update_loss": round(float(loss), 6),
        }
        reports.append(row)
        _append_jsonl(args.log_jsonl, row)
        if args.progress:
            print(json.dumps(row), file=sys.stderr)
        if args.snapshot_interval > 0 and (generation + 1) % args.snapshot_interval == 0:
            snapshots.append(_snapshot(args.kind, params, mean, scale, prefs, args))
            snapshots = snapshots[-args.max_snapshots:]

    report = {
        "mode": "real_fullhouse_self_play",
        "kind": args.kind,
        "output": str(output),
        "seed": args.seed,
        "generations": args.generations,
        "matches_per_generation": args.matches_per_generation,
        "hands": args.hands,
        "players": args.players,
        "train_seats": args.train_seats,
        "mean_train_delta": reports[-1]["mean_train_delta"] if reports else 0.0,
        "abstraction": args.abstraction,
        "bucket_update": args.bucket_update,
        "reports": reports[-5:],
    }
    if args.kind == "ppo":
        _export_ppo(output, params, mean, scale, report)
    else:
        _export_bucket(output, prefs, strategy_sum, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strong mock policies from actual Fullhouse self-play")
    parser.add_argument("--kind", choices=["ppo", "bucket"], default="ppo")
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--matches-per-generation", type=int, default=32)
    parser.add_argument("--hands", type=int, default=80)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--train-seats", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--bucket-count", type=int, default=4096)
    parser.add_argument("--abstraction", choices=["feature", "cfr-pokerbot"], default="feature")
    parser.add_argument("--bucket-update", choices=["policy-gradient", "cfr-plus"], default="policy-gradient")
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.92)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument("--opponent-pool", choices=["oracle", "fast", "mixed"], default="mixed")
    parser.add_argument("--snapshot-interval", type=int, default=2)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument("--snapshot-prob", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--seed", type=int, default=6161)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not (2 <= args.players <= 9):
        raise SystemExit("--players must be between 2 and 9")
    if not (1 <= args.train_seats <= args.players):
        raise SystemExit("--train-seats must be between 1 and --players")
    result = train(args)
    print(json.dumps(result, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
