"""Train an independent deep PPO-style strong mock with expanded action arms."""

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
from tools.strong_mocks.dataset import sample_feature_matrix
from tools.strong_mocks.deep_actions import ACTION_LABELS, action_index_to_action, legal_mask, strategic_mask
from tools.strong_mocks.deep_policies import (
    DEEP_FEATURE_NAMES,
    augment_features,
    deep_oracle_labels,
    deep_oracle_logits,
    extract_deep_features,
)
from tools.strong_mocks.policies import decide_rollout
from training.fast_match import FastBot


DEFAULT_OUTPUT = ROOT / "bots" / "strong_mocks" / "ppo_deep_policy" / "data" / "policy.npz"

BOT_PATH_OPPONENTS = (
    ROOT / "bots" / "heuristic",
    ROOT / "bots" / "shark",
    ROOT / "bots" / "mathematician",
    ROOT / "bots" / "benchmarks" / "threshold_caller",
    ROOT / "bots" / "mock_competitors" / "equity_pressure",
    ROOT / "bots" / "mock_competitors" / "bucket_overbet",
)

ORACLE_STYLES = ("pressure", "value", "conservative", "explore")


@dataclass
class Decision:
    bot_id: str
    x: np.ndarray
    mask: np.ndarray
    action: int
    old_logprob: float = 0.0
    old_value: float = 0.0
    temperature: float = 1.0
    reward: float = 0.0


def _parse_hidden(value: str) -> tuple[int, ...]:
    parts = [int(item) for item in str(value).replace(":", ",").split(",") if item.strip()]
    if not parts:
        raise ValueError("hidden must contain at least one layer size")
    return tuple(max(4, item) for item in parts)


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
    legal = np.flatnonzero(probs > 0)
    return int(legal[-1]) if legal.size else 0


def _init_params(seed: int, hidden: tuple[int, ...]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    input_dim = len(DEEP_FEATURE_NAMES)
    prev = input_dim
    for index, size in enumerate(hidden):
        params[f"w{index}"] = rng.normal(0.0, 0.9 / max(1.0, prev ** 0.5), size=(prev, size)).astype(np.float64)
        params[f"b{index}"] = np.zeros(size, dtype=np.float64)
        prev = size
    params["w_policy"] = rng.normal(0.0, 0.5 / max(1.0, prev ** 0.5), size=(prev, len(ACTION_LABELS))).astype(np.float64)
    params["b_policy"] = np.zeros(len(ACTION_LABELS), dtype=np.float64)
    params["w_value"] = rng.normal(0.0, 0.25 / max(1.0, prev ** 0.5), size=(prev,)).astype(np.float64)
    params["b_value"] = np.zeros(1, dtype=np.float64)
    return params


def _param_hidden_sizes(params: dict[str, np.ndarray]) -> tuple[int, ...]:
    sizes = []
    index = 0
    while f"w{index}" in params:
        sizes.append(int(params[f"w{index}"].shape[1]))
        index += 1
    return tuple(sizes)


def _load_policy(path: Path, hidden: tuple[int, ...], seed: int, resume: bool) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mean = np.zeros(len(DEEP_FEATURE_NAMES), dtype=np.float64)
    scale = np.ones(len(DEEP_FEATURE_NAMES), dtype=np.float64)
    if resume and path.is_file():
        try:
            data = np.load(path, allow_pickle=False)
            params: dict[str, np.ndarray] = {}
            index = 0
            while f"w{index}" in data.files:
                params[f"w{index}"] = data[f"w{index}"].astype(np.float64)
                params[f"b{index}"] = data[f"b{index}"].astype(np.float64)
                index += 1
            params["w_policy"] = data["w_policy"].astype(np.float64)
            params["b_policy"] = data["b_policy"].astype(np.float64)
            params["w_value"] = data["w_value"].astype(np.float64) if "w_value" in data.files else np.zeros(params[f"b{index - 1}"].shape[0])
            params["b_value"] = data["b_value"].astype(np.float64) if "b_value" in data.files else np.zeros(1)
            if _param_hidden_sizes(params) == hidden:
                if "mean" in data.files:
                    mean = data["mean"].astype(np.float64)
                if "scale" in data.files:
                    scale = np.maximum(1e-6, data["scale"].astype(np.float64))
                return params, mean, scale
        except Exception:
            pass
    return _init_params(seed, hidden), mean, scale


def _forward(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    z = (x - mean) / np.maximum(1e-6, scale)
    activations = [z]
    hidden = z
    index = 0
    while f"w{index}" in params:
        hidden = np.tanh(hidden @ params[f"w{index}"] + params[f"b{index}"])
        activations.append(hidden)
        index += 1
    logits = hidden @ params["w_policy"] + params["b_policy"]
    value = hidden @ params["w_value"] + float(np.asarray(params["b_value"]).reshape(-1)[0])
    return activations, logits, np.asarray(value, dtype=np.float64)


def _clip_grads(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    if max_norm <= 0:
        return grads
    total = sum(float(np.sum(grad * grad)) for grad in grads)
    norm = total ** 0.5
    if norm <= max_norm or norm <= 1e-12:
        return grads
    scale = max_norm / norm
    return [grad * scale for grad in grads]


def _apply_backprop(
    params: dict[str, np.ndarray],
    activations: list[np.ndarray],
    grad_logits: np.ndarray,
    grad_value: np.ndarray | None,
    learning_rate: float,
    max_grad_norm: float,
) -> None:
    last_hidden = activations[-1]
    grad_w_policy = last_hidden.T @ grad_logits
    grad_b_policy = grad_logits.sum(axis=0)
    grad_hidden = grad_logits @ params["w_policy"].T

    if grad_value is not None:
        grad_w_value = last_hidden.T @ grad_value
        grad_b_value = np.asarray([grad_value.sum()], dtype=np.float64)
        grad_hidden += grad_value[:, None] * params["w_value"][None, :]
    else:
        grad_w_value = np.zeros_like(params["w_value"])
        grad_b_value = np.zeros_like(params["b_value"])

    layer_indices = list(range(len(activations) - 1))
    grad_layers: list[tuple[int, np.ndarray, np.ndarray]] = []
    for layer in reversed(layer_indices):
        hidden = activations[layer + 1]
        grad_pre = grad_hidden * (1.0 - hidden * hidden)
        grad_w = activations[layer].T @ grad_pre
        grad_b = grad_pre.sum(axis=0)
        grad_hidden = grad_pre @ params[f"w{layer}"].T
        grad_layers.append((layer, grad_w, grad_b))

    ordered_grads = [grad_w_policy, grad_b_policy, grad_w_value, grad_b_value]
    for _layer, grad_w, grad_b in grad_layers:
        ordered_grads.extend([grad_w, grad_b])
    clipped = _clip_grads(ordered_grads, max_grad_norm)
    offset = 0
    grad_w_policy, grad_b_policy, grad_w_value, grad_b_value = clipped[:4]
    offset = 4
    params["w_policy"] -= learning_rate * grad_w_policy
    params["b_policy"] -= learning_rate * grad_b_policy
    params["w_value"] -= learning_rate * grad_w_value
    params["b_value"] -= learning_rate * grad_b_value
    for layer, _grad_w, _grad_b in grad_layers:
        params[f"w{layer}"] -= learning_rate * clipped[offset]
        params[f"b{layer}"] -= learning_rate * clipped[offset + 1]
        offset += 2


def _bootstrap_metrics(params: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    _acts, logits, _values = _forward(params, x, mean, scale)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / np.maximum(1e-9, exp.sum(axis=1, keepdims=True))
    selected = np.maximum(1e-8, probs[np.arange(len(y)), y])
    pred = np.argmax(probs, axis=1)
    return {
        "loss": float(-np.mean(np.log(selected))),
        "agreement": float(np.mean(pred == y)),
    }


def _supervised_bootstrap(
    params: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    seed: int,
    samples: int,
    holdout: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_grad_norm: float,
    style: str,
) -> dict:
    if samples <= 0 or epochs <= 0:
        return {}
    train_base = sample_feature_matrix(samples, seed)
    train_x = augment_features(train_base).astype(np.float64)
    train_y = deep_oracle_labels(train_x, style)
    hold_x = augment_features(sample_feature_matrix(max(1, holdout), seed + 17)).astype(np.float64)
    hold_y = deep_oracle_labels(hold_x, style)
    initial = _bootstrap_metrics(params, mean, scale, hold_x, hold_y)
    rng = np.random.default_rng(seed + 29)
    for _epoch in range(epochs):
        order = rng.permutation(train_x.shape[0])
        for start in range(0, train_x.shape[0], max(1, batch_size)):
            idx = order[start:start + max(1, batch_size)]
            xb = train_x[idx]
            yb = train_y[idx]
            activations, logits, _values = _forward(params, xb, mean, scale)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / np.maximum(1e-9, exp.sum(axis=1, keepdims=True))
            grad_logits = probs
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= max(1, len(yb))
            _apply_backprop(params, activations, grad_logits, None, learning_rate, max_grad_norm)
    final = _bootstrap_metrics(params, mean, scale, hold_x, hold_y)
    return {
        "bootstrap_initial_loss": round(initial["loss"], 6),
        "bootstrap_final_loss": round(final["loss"], 6),
        "bootstrap_initial_agreement": round(initial["agreement"], 4),
        "bootstrap_final_agreement": round(final["agreement"], 4),
        "bootstrap_samples": samples,
        "bootstrap_holdout": holdout,
        "bootstrap_epochs": epochs,
    }


def _normalized_advantage(rewards: np.ndarray) -> np.ndarray:
    if rewards.size == 0:
        return rewards
    std = float(np.std(rewards))
    if std < 1e-6:
        return rewards - float(np.mean(rewards))
    return (rewards - float(np.mean(rewards))) / std


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
    clip_ratio: float,
    value_coef: float,
    max_grad_norm: float,
) -> float:
    if data["x"].shape[0] == 0:
        return 0.0
    x = data["x"]
    masks = data["mask"]
    actions = data["action"]
    targets = data["reward"]
    old_logprob = data["old_logprob"]
    old_value = data["old_value"]
    temperatures = np.maximum(1e-6, data["temperature"])
    adv = _normalized_advantage(targets - old_value)
    losses = []
    n = x.shape[0]
    for _epoch in range(max(1, epochs)):
        order = rng.permutation(n)
        for start in range(0, n, max(1, batch_size)):
            idx = order[start:start + max(1, batch_size)]
            xb = x[idx]
            maskb = masks[idx]
            actionb = actions[idx]
            advb = adv[idx]
            targetb = targets[idx]
            old_logprob_b = old_logprob[idx]
            old_value_b = old_value[idx]
            tempb = temperatures[idx]
            activations, logits, values = _forward(params, xb, mean, scale)
            probs = np.zeros_like(logits, dtype=np.float64)
            for row in range(logits.shape[0]):
                probs[row] = _softmax_masked(logits[row], maskb[row], tempb[row])
            selected = np.maximum(1e-8, probs[np.arange(len(idx)), actionb])
            log_selected = np.log(selected)
            ratio = np.exp(np.clip(log_selected - old_logprob_b, -6.0, 6.0))
            clipped_ratio = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -np.mean(np.minimum(ratio * advb, clipped_ratio * advb))
            value_loss = np.mean((values - targetb) ** 2)
            losses.append(float(policy_loss + value_coef * value_loss))

            active = np.ones(len(idx), dtype=bool)
            active[(advb > 0) & (ratio > 1.0 + clip_ratio)] = False
            active[(advb < 0) & (ratio < 1.0 - clip_ratio)] = False
            coeff = np.zeros(len(idx), dtype=np.float64)
            coeff[active] = advb[active] * ratio[active]
            scaled_coeff = coeff / tempb
            grad_logits = probs * scaled_coeff[:, None]
            grad_logits[np.arange(len(idx)), actionb] -= scaled_coeff
            if entropy_coef:
                entropy_grad = probs * (np.log(np.maximum(1e-8, probs)) + 1.0)
                entropy_grad[~maskb] = 0.0
                grad_logits += entropy_coef * entropy_grad / tempb[:, None]
            grad_logits /= max(1, len(idx))
            grad_value = (2.0 * value_coef / max(1, len(idx))) * (values - targetb)
            # Mildly keep the value baseline close to behavior values in tiny samples.
            if old_value_b.size:
                grad_value += (0.05 * value_coef / max(1, len(idx))) * (values - old_value_b)
            _apply_backprop(params, activations, grad_logits, grad_value, learning_rate, max_grad_norm)
    return float(np.mean(losses)) if losses else 0.0


def _policy_action(task: dict, state: dict, rng: random.Random, record: bool) -> tuple[dict, Decision | None]:
    x = extract_deep_features(state).astype(np.float64)
    mask = strategic_mask(state)
    activations, logits, values = _forward(task["params"], x.reshape(1, -1), task["mean"], task["scale"])
    _ = activations
    temperature = float(task["temperature"])
    probs = _softmax_masked(logits[0], mask, temperature)
    action_index = _sample_index(probs, rng)
    action = action_index_to_action(state, action_index)
    if not record:
        return action, None
    return action, Decision(
        bot_id=state["players"][state["seat_to_act"]]["bot_id"],
        x=x.astype(np.float32),
        mask=mask.astype(bool),
        action=action_index,
        old_logprob=float(np.log(max(1e-8, probs[action_index]))),
        old_value=float(values.reshape(-1)[0]),
        temperature=temperature,
    )


def _oracle_action(style: str, state: dict, rng: random.Random) -> dict:
    x = extract_deep_features(state)
    logits = deep_oracle_logits(x.reshape(1, -1), style)[0]
    probs = _softmax_masked(logits, legal_mask(state), 0.80)
    return action_index_to_action(state, _sample_index(probs, rng))


def _opponent_pool(mode: str) -> list[dict]:
    base = [{"kind": "oracle", "style": style} for style in ORACLE_STYLES]
    if mode == "oracle":
        return base
    if mode == "fast":
        return base + [{"kind": "rollout", "style": "rollout_pressure"}]
    bot_specs = [{"kind": "bot_path", "path": str(path)} for path in BOT_PATH_OPPONENTS]
    return base + [
        {"kind": "rollout", "style": "rollout_pressure"},
        {"kind": "rollout", "style": "rollout_deep"},
    ] + bot_specs


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


def _snapshot_action(snapshot: dict, state: dict, rng: random.Random) -> dict:
    x = extract_deep_features(state).astype(np.float64)
    _acts, logits, _values = _forward(snapshot["params"], x.reshape(1, -1), snapshot["mean"], snapshot["scale"])
    probs = _softmax_masked(logits[0], strategic_mask(state), snapshot["temperature"])
    return action_index_to_action(state, _sample_index(probs, rng))


def _opponent_action(spec: dict, state: dict, rng: random.Random, fast_bots: dict[str, FastBot], bot_id: str) -> dict:
    kind = spec["kind"]
    if kind == "bot_path":
        bot = fast_bots.get(bot_id)
        return bot.act(state) if bot is not None else _oracle_action("pressure", state, rng)
    if kind == "rollout":
        return decide_rollout(state, style=spec["style"])
    if kind == "snapshot":
        return _snapshot_action(spec["snapshot"], state, rng)
    return _oracle_action(spec.get("style", "pressure"), state, rng)


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
                    action, decision = _policy_action(task, state, rng, record=True)
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


def _flatten(matches: list[dict]) -> dict[str, np.ndarray]:
    rows = [item for match in matches for item in match["decisions"]]
    if not rows:
        return {
            "x": np.zeros((0, len(DEEP_FEATURE_NAMES)), dtype=np.float64),
            "mask": np.zeros((0, len(ACTION_LABELS)), dtype=bool),
            "action": np.zeros(0, dtype=np.int64),
            "old_logprob": np.zeros(0, dtype=np.float64),
            "old_value": np.zeros(0, dtype=np.float64),
            "temperature": np.ones(0, dtype=np.float64),
            "reward": np.zeros(0, dtype=np.float64),
        }
    return {
        "x": np.stack([row["x"] for row in rows]).astype(np.float64),
        "mask": np.stack([row["mask"] for row in rows]).astype(bool),
        "action": np.asarray([row["action"] for row in rows], dtype=np.int64),
        "old_logprob": np.asarray([row["old_logprob"] for row in rows], dtype=np.float64),
        "old_value": np.asarray([row["old_value"] for row in rows], dtype=np.float64),
        "temperature": np.asarray([row["temperature"] for row in rows], dtype=np.float64),
        "reward": np.asarray([row["reward"] for row in rows], dtype=np.float64),
    }


def _concat_data(batches: list[dict[str, np.ndarray]], max_rows: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    valid = [batch for batch in batches if batch["reward"].shape[0] > 0]
    if not valid:
        return _flatten([])
    out = {key: np.concatenate([batch[key] for batch in valid], axis=0) for key in valid[0]}
    if max_rows > 0 and out["reward"].shape[0] > max_rows:
        idx = rng.choice(out["reward"].shape[0], size=max_rows, replace=False)
        idx.sort()
        out = {key: value[idx] for key, value in out.items()}
    return out


def _snapshot(params: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray, temperature: float) -> dict:
    return {
        "params": {key: value.copy() for key, value in params.items()},
        "mean": mean.copy(),
        "scale": scale.copy(),
        "temperature": max(0.45, temperature * 0.90),
    }


def _export_policy(output: Path, params: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": np.asarray(["deep_ppo_mlp"]),
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "w_policy": params["w_policy"].astype(np.float32),
        "b_policy": params["b_policy"].astype(np.float32),
        "w_value": params["w_value"].astype(np.float32),
        "b_value": params["b_value"].astype(np.float32),
        "feature_names": np.asarray(DEEP_FEATURE_NAMES),
        "action_labels": np.asarray(ACTION_LABELS),
        "hidden_sizes": np.asarray(_param_hidden_sizes(params), dtype=np.int32),
        "temperature": np.asarray([report.get("temperature", 0.55)], dtype=np.float32),
        "inference_mode": np.asarray(["sampled_softmax"]),
        "generations": np.asarray([report.get("generations", 0)], dtype=np.int32),
        "best_selection_score": np.asarray([report.get("best_selection_score", 0.0)], dtype=np.float32),
    }
    index = 0
    while f"w{index}" in params:
        payload[f"w{index}"] = params[f"w{index}"].astype(np.float32)
        payload[f"b{index}"] = params[f"b{index}"].astype(np.float32)
        index += 1
    np.savez_compressed(output, **payload)


def train(args) -> dict:
    output = Path(args.output or DEFAULT_OUTPUT)
    hidden = _parse_hidden(args.hidden)
    params, mean, scale = _load_policy(output, hidden, args.seed, args.resume)
    rng = np.random.default_rng(args.seed)
    bootstrap = _supervised_bootstrap(
        params,
        mean,
        scale,
        args.seed,
        args.bootstrap_samples,
        args.bootstrap_holdout,
        args.bootstrap_epochs,
        args.batch_size,
        args.bootstrap_learning_rate,
        args.max_grad_norm,
        args.bootstrap_style,
    )
    snapshots: list[dict] = []
    replay_batches: list[dict[str, np.ndarray]] = []
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
                "hands": args.hands,
                "players": args.players,
                "train_seats": args.train_seats,
                "reward_scale": args.reward_scale,
                "reward_clip": args.reward_clip,
                "opponent_pool": args.opponent_pool,
                "snapshot_prob": args.snapshot_prob,
                "snapshots": snapshots,
                "seed": args.seed * 1_000_000 + generation * 10_000 + match_index,
                "match_id": f"deep_ppo_g{generation:04d}_{match_index:04d}",
            })
        matches = map_parallel(_run_training_match, tasks, workers=args.workers, backend=args.parallel_backend)
        data = _flatten(matches)
        train_bust_count = int(sum(match.get("train_bust_count", 0) for match in matches))
        train_seat_count = int(sum(match.get("train_seat_count", 0) for match in matches))
        mean_delta = float(np.mean([match["mean_train_delta"] for match in matches])) if matches else 0.0
        train_bust_rate = train_bust_count / max(1, train_seat_count)
        selection_score = mean_delta - args.selection_bust_penalty * train_bust_rate
        row = {
            "generation": generation,
            "kind": "deep_ppo",
            "matches": len(matches),
            "hands": int(sum(match["hands"] for match in matches)),
            "decisions": int(data["reward"].shape[0]),
            "mean_train_delta": round(mean_delta, 3),
            "selection_score": round(selection_score, 3),
            "train_bust_count": train_bust_count,
            "train_bust_rate": round(train_bust_rate, 4),
            "mean_decision_reward": round(float(np.mean(data["reward"])) if data["reward"].size else 0.0, 5),
            "bot_error_count": int(sum(match.get("bot_error_count", 0) for match in matches)),
        }
        if data["action"].size:
            counts = np.bincount(data["action"], minlength=len(ACTION_LABELS))
            row["action_counts"] = {
                ACTION_LABELS[index]: int(counts[index])
                for index in range(len(ACTION_LABELS))
                if int(counts[index]) > 0
            }
        if generation >= args.selection_warmup and selection_score > best_metric:
            best_metric = selection_score
            best_state = pre_update_state
            best_row = dict(row)
        replay_batches.append(data)
        replay_batches = replay_batches[-max(1, args.replay_generations):]
        update_data = _concat_data(replay_batches, args.replay_max_decisions, rng)
        loss = _update_ppo(
            params,
            mean,
            scale,
            update_data,
            rng,
            args.learning_rate,
            args.entropy_coef,
            args.epochs,
            args.batch_size,
            args.ppo_clip_ratio,
            args.ppo_value_coef,
            args.max_grad_norm,
        )
        row["update_loss"] = round(float(loss), 6)
        reports.append(row)
        if args.progress:
            print(json.dumps(row), file=sys.stderr)
        if args.snapshot_interval > 0 and (generation + 1) % args.snapshot_interval == 0:
            snapshots.append(_snapshot(params, mean, scale, args.temperature))
            snapshots = snapshots[-args.max_snapshots:]

    selected_row = best_row if best_row is not None else (reports[-1] if reports else None)
    if best_row is None:
        best_state = {key: value.copy() for key, value in params.items()}
    report = {
        "mode": "deep_fullhouse_self_play",
        "kind": "deep_ppo",
        "output": str(output),
        "seed": args.seed,
        "hidden": list(hidden),
        "actions": len(ACTION_LABELS),
        "generations": len(reports),
        "matches_per_generation": args.matches_per_generation,
        "hands": args.hands,
        "players": args.players,
        "train_seats": args.train_seats,
        "temperature": args.temperature,
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
    parser = argparse.ArgumentParser(description="Train the independent deep PPO strong mock")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--matches-per-generation", type=int, default=32)
    parser.add_argument("--hands", type=int, default=120)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--train-seats", type=int, default=2)
    parser.add_argument("--hidden", default="96,64,32")
    parser.add_argument("--learning-rate", type=float, default=0.0035)
    parser.add_argument("--bootstrap-learning-rate", type=float, default=0.006)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.18)
    parser.add_argument("--ppo-value-coef", type=float, default=0.40)
    parser.add_argument("--max-grad-norm", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--replay-generations", type=int, default=4)
    parser.add_argument("--replay-max-decisions", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument("--selection-warmup", type=int, default=0)
    parser.add_argument("--selection-bust-penalty", type=float, default=9000.0)
    parser.add_argument("--opponent-pool", choices=["oracle", "fast", "adversarial"], default="fast")
    parser.add_argument("--snapshot-interval", type=int, default=2)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument("--snapshot-prob", type=float, default=0.30)
    parser.add_argument("--bootstrap-samples", type=int, default=12000)
    parser.add_argument("--bootstrap-holdout", type=int, default=2000)
    parser.add_argument("--bootstrap-epochs", type=int, default=4)
    parser.add_argument("--bootstrap-style", choices=["pressure", "value", "conservative", "explore"], default="pressure")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--seed", type=int, default=7171)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
