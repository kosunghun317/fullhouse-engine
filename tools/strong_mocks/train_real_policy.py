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
from tools.strong_mocks.actions import ACTION_LABELS, action_index_to_action, legal_mask, strategic_mask
from tools.strong_mocks.abstractions import abstract_bucket_id
from tools.strong_mocks.dataset import oracle_logits
from tools.strong_mocks.features import FEATURE_NAMES, extract_features
from tools.strong_mocks.policies import decide_rollout, logits_from_model
from training.fast_match import FastBot


DEFAULT_PPO_OUTPUT = ROOT / "bots" / "strong_mocks" / "ppo_policy" / "data" / "policy.npz"
DEFAULT_BUCKET_OUTPUT = ROOT / "bots" / "strong_mocks" / "cfr_bucket" / "data" / "policy.npz"

ORACLE_STYLES = ("balanced", "value", "bluff", "station", "folder", "pressure")
MODEL_DIRS = {
    "oracle_model": ROOT / "bots" / "strong_mocks" / "oracle_imitation" / "data",
    "ppo_model": ROOT / "bots" / "strong_mocks" / "ppo_policy" / "data",
    "cfr_model": ROOT / "bots" / "strong_mocks" / "cfr_bucket" / "data",
}

BOT_PATH_OPPONENTS = (
    ROOT / "bots" / "heuristic",
    ROOT / "bots" / "shark",
    ROOT / "bots" / "mathematician",
    ROOT / "bots" / "benchmarks" / "threshold_caller",
    ROOT / "bots" / "mock_competitors" / "equity_pressure",
    ROOT / "bots" / "mock_competitors" / "bucket_overbet",
)


@dataclass
class Decision:
    bot_id: str
    x: np.ndarray
    mask: np.ndarray
    action: int
    bucket: int
    old_logprob: float = 0.0
    old_value: float = 0.0
    temperature: float = 1.0
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
        "wv": rng.normal(0.0, 0.04, size=(hidden,)).astype(np.float64),
        "bv": np.zeros(1, dtype=np.float64),
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
                if params["w2"].shape != (hidden, len(ACTION_LABELS)):
                    classes = data["classes"].astype(int) if "classes" in data.files else np.arange(params["w2"].shape[1])
                    full_w2 = np.zeros((hidden, len(ACTION_LABELS)), dtype=np.float64)
                    full_b2 = np.full(len(ACTION_LABELS), -4.0, dtype=np.float64)
                    for offset, cls in enumerate(classes):
                        if 0 <= int(cls) < len(ACTION_LABELS) and offset < params["w2"].shape[1]:
                            full_w2[:, int(cls)] = params["w2"][:, offset]
                            full_b2[int(cls)] = params["b2"][offset]
                    params["w2"] = full_w2
                    params["b2"] = full_b2
                if "wv" in data.files and data["wv"].shape == (hidden,):
                    params["wv"] = data["wv"].astype(np.float64)
                else:
                    params["wv"] = np.zeros(hidden, dtype=np.float64)
                if "bv" in data.files:
                    params["bv"] = np.asarray(data["bv"], dtype=np.float64).reshape(1)
                else:
                    params["bv"] = np.zeros(1, dtype=np.float64)
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


def _value_from_hidden(params: dict[str, np.ndarray], hidden: np.ndarray) -> np.ndarray:
    wv = params.get("wv")
    bv = params.get("bv")
    if wv is None:
        return np.zeros(hidden.shape[0] if hidden.ndim > 1 else 1, dtype=np.float64)
    value = hidden @ wv + (float(np.asarray(bv).reshape(-1)[0]) if bv is not None else 0.0)
    return np.asarray(value, dtype=np.float64)


def _policy_action(task: dict, state: dict, rng: random.Random, record: bool) -> tuple[dict, Decision | None]:
    kind = task["kind"]
    x = extract_features(state)
    mask = strategic_mask(state) if kind == "ppo" else legal_mask(state)
    value = 0.0
    if kind == "ppo":
        hidden, logits = _forward(task["params"], x, task["mean"], task["scale"])
        value = float(_value_from_hidden(task["params"], hidden).reshape(-1)[0])
    else:
        b = abstract_bucket_id(state, int(task["bucket_count"]), task.get("abstraction", "feature"))
        logits = np.maximum(task["prefs"][b], 0.0) if task.get("bucket_update") == "cfr-plus" else task["prefs"][b]
    temperature = float(task["temperature"])
    probs = _softmax_masked(logits, mask, temperature)
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
        old_logprob=float(np.log(max(1e-8, probs[action_index]))),
        old_value=value,
        temperature=temperature,
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


def _opponent_action(
    spec: dict,
    state: dict,
    rng: random.Random,
    fast_bots: dict[str, FastBot] | None = None,
    bot_id: str | None = None,
) -> dict:
    kind = spec["kind"]
    if kind == "bot_path":
        if fast_bots is None or bot_id is None or bot_id not in fast_bots:
            return _oracle_action("balanced", state, rng)
        return fast_bots[bot_id].act(state)
    if kind == "oracle":
        return _oracle_action(spec["style"], state, rng)
    if kind == "rollout":
        return decide_rollout(state, style=spec["style"])
    if kind == "model":
        return _model_action(spec["name"], state)
    if kind == "snapshot":
        return _snapshot_action(spec["snapshot"], state, rng)
    return _oracle_action("balanced", state, rng)


def _opponent_pool(mode: str, extra_bot_paths: list[str] | None = None) -> list[dict]:
    base = [{"kind": "oracle", "style": style} for style in ORACLE_STYLES]
    if mode == "oracle":
        return base
    if mode == "fast":
        return base + [{"kind": "model", "name": "oracle_model"}]
    bot_path_specs = [{"kind": "bot_path", "path": str(path)} for path in BOT_PATH_OPPONENTS]
    bot_path_specs.extend({"kind": "bot_path", "path": str(path)} for path in (extra_bot_paths or []))
    if mode == "adversarial":
        return base + [
            {"kind": "rollout", "style": "rollout_pressure"},
            {"kind": "rollout", "style": "rollout_deep"},
            {"kind": "model", "name": "oracle_model"},
        ] + bot_path_specs
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
    pool = _opponent_pool(task["opponent_pool"], task.get("extra_bot_paths") or [])
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
                    action = _opponent_action(spec, state, rng, fast_bots=fast_bots, bot_id=bot_id)
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
    train_busts = sum(1 for bot_id in train_ids if stacks.get(bot_id, 0) <= 0)
    bot_error_count = sum(len(bot.errors) for bot in fast_bots.values())
    return {
        "decisions": [
            {
                "x": decision.x,
                "mask": decision.mask,
                "action": decision.action,
                "bucket": decision.bucket,
                "old_logprob": decision.old_logprob,
                "old_value": decision.old_value,
                "temperature": decision.temperature,
                "reward": decision.reward,
            }
            for decision in decisions
        ],
        "hands": hand_count,
        "train_deltas": train_deltas,
        "mean_train_delta": float(sum(train_deltas.values()) / max(1, len(train_deltas))),
        "train_bust_count": train_busts,
        "train_seat_count": len(train_ids),
        "bot_error_count": bot_error_count,
    }


def _flatten_decisions(matches: list[dict]) -> dict[str, np.ndarray]:
    rows = [item for match in matches for item in match["decisions"]]
    if not rows:
        return {
            "x": np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64),
            "mask": np.zeros((0, len(ACTION_LABELS)), dtype=bool),
            "action": np.zeros(0, dtype=np.int64),
            "bucket": np.zeros(0, dtype=np.int64),
            "old_logprob": np.zeros(0, dtype=np.float64),
            "old_value": np.zeros(0, dtype=np.float64),
            "temperature": np.ones(0, dtype=np.float64),
            "reward": np.zeros(0, dtype=np.float64),
        }
    return {
        "x": np.stack([row["x"] for row in rows]).astype(np.float64),
        "mask": np.stack([row["mask"] for row in rows]).astype(bool),
        "action": np.asarray([row["action"] for row in rows], dtype=np.int64),
        "bucket": np.asarray([row["bucket"] for row in rows], dtype=np.int64),
        "old_logprob": np.asarray([row.get("old_logprob", 0.0) for row in rows], dtype=np.float64),
        "old_value": np.asarray([row.get("old_value", 0.0) for row in rows], dtype=np.float64),
        "temperature": np.asarray([row.get("temperature", 1.0) for row in rows], dtype=np.float64),
        "reward": np.asarray([row["reward"] for row in rows], dtype=np.float64),
    }


def _concat_data(batches: list[dict[str, np.ndarray]], max_rows: int = 0, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    valid = [batch for batch in batches if batch["reward"].shape[0] > 0]
    if not valid:
        return _flatten_decisions([])
    out = {
        key: np.concatenate([batch[key] for batch in valid], axis=0)
        for key in valid[0]
    }
    if max_rows > 0 and out["reward"].shape[0] > max_rows:
        chooser = rng or np.random.default_rng(0)
        idx = chooser.choice(out["reward"].shape[0], size=max_rows, replace=False)
        idx.sort()
        out = {key: value[idx] for key, value in out.items()}
    return out


def _update_feature_norm(
    mean: np.ndarray,
    scale: np.ndarray,
    x: np.ndarray,
    generation: int,
    momentum: float,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] == 0 or momentum <= 0:
        return mean, scale
    batch_mean = x.mean(axis=0)
    batch_scale = x.std(axis=0)
    batch_scale[batch_scale < 1e-6] = 1.0
    if generation == 0 and not resume:
        return batch_mean, batch_scale
    alpha = max(0.0, min(1.0, momentum))
    new_mean = (1.0 - alpha) * mean + alpha * batch_mean
    new_scale = (1.0 - alpha) * scale + alpha * batch_scale
    new_scale[new_scale < 1e-6] = 1.0
    return new_mean, new_scale


def _normalized_advantage(rewards: np.ndarray) -> np.ndarray:
    if rewards.size == 0:
        return rewards
    clipped = np.asarray(rewards, dtype=np.float64)
    std = float(np.std(clipped))
    if std < 1e-6:
        return clipped - float(np.mean(clipped))
    return (clipped - float(np.mean(clipped))) / std


def _clip_grads(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    if max_norm <= 0:
        return grads
    total = 0.0
    for grad in grads:
        total += float(np.sum(grad * grad))
    norm = total ** 0.5
    if norm <= max_norm or norm <= 1e-12:
        return grads
    scale = max_norm / norm
    return [grad * scale for grad in grads]


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
    old_logprob = data.get("old_logprob", np.zeros_like(targets))
    old_value = data.get("old_value", np.zeros_like(targets))
    temperatures = np.maximum(1e-6, data.get("temperature", np.ones_like(targets)))
    adv = _normalized_advantage(targets - old_value)
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
            targetb = targets[idx]
            old_logprob_b = old_logprob[idx]
            tempb = temperatures[idx]
            hidden, logits = _forward(params, xb, mean, scale)
            probs = np.zeros_like(logits, dtype=np.float64)
            for row in range(logits.shape[0]):
                probs[row] = _softmax_masked(logits[row], maskb[row], tempb[row])
            selected = np.maximum(1e-8, probs[np.arange(len(idx)), actionb])
            log_selected = np.log(selected)
            ratio = np.exp(np.clip(log_selected - old_logprob_b, -6.0, 6.0))
            clipped_ratio = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            value_pred = _value_from_hidden(params, hidden)
            policy_loss = -np.mean(np.minimum(ratio * advb, clipped_ratio * advb))
            value_loss = np.mean((value_pred - targetb) ** 2)
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

            z = (xb - mean) / np.maximum(1e-6, scale)
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ params["w2"].T
            grad_value = (2.0 * value_coef / max(1, len(idx))) * (value_pred - targetb)
            grad_wv = hidden.T @ grad_value
            grad_bv = np.asarray([grad_value.sum()], dtype=np.float64)
            grad_hidden += grad_value[:, None] * params["wv"][None, :]
            grad_pre = grad_hidden * (1.0 - hidden * hidden)
            grad_w1 = z.T @ grad_pre
            grad_b1 = grad_pre.sum(axis=0)

            grad_w1, grad_b1, grad_w2, grad_b2, grad_wv, grad_bv = _clip_grads(
                [grad_w1, grad_b1, grad_w2, grad_b2, grad_wv, grad_bv],
                max_grad_norm,
            )
            params["w2"] -= learning_rate * grad_w2
            params["b2"] -= learning_rate * grad_b2
            params["wv"] -= learning_rate * grad_wv
            params["bv"] -= learning_rate * grad_bv
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
        wv=params.get("wv", np.zeros(params["w1"].shape[1], dtype=np.float64)).astype(np.float32),
        bv=params.get("bv", np.zeros(1, dtype=np.float64)).astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        classes=np.arange(len(ACTION_LABELS), dtype=np.int8),
        feature_names=np.asarray(FEATURE_NAMES),
        action_labels=np.asarray(ACTION_LABELS),
        temperature=np.asarray([report.get("temperature", 0.62)], dtype=np.float32),
        inference_mode=np.asarray(["sampled_softmax"]),
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


def _export_state(
    kind: str,
    params: dict | None,
    mean: np.ndarray,
    scale: np.ndarray,
    prefs: np.ndarray | None,
    strategy_sum: np.ndarray | None,
) -> dict:
    if kind == "ppo":
        return {
            "params": {key: value.copy() for key, value in params.items()},
            "mean": mean.copy(),
            "scale": scale.copy(),
        }
    return {
        "prefs": prefs.copy(),
        "strategy_sum": strategy_sum.copy(),
    }


def _append_jsonl(path: str | None, row: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def should_export_policy(selected_mean_delta: float | None, output_exists: bool, min_export_mean_delta: float) -> bool:
    """Return whether a trained checkpoint may overwrite an existing artifact."""
    if selected_mean_delta is None:
        return not output_exists
    return (not output_exists) or float(selected_mean_delta) >= float(min_export_mean_delta)


def validate_training_budget(args, output: Path) -> None:
    canonical = DEFAULT_PPO_OUTPUT if args.kind == "ppo" else DEFAULT_BUCKET_OUTPUT
    allow_smoke = bool(getattr(args, "allow_smoke", False))
    min_training_hands = int(getattr(args, "min_training_hands", 1_000_000))
    if allow_smoke:
        if output.resolve() == canonical.resolve():
            raise SystemExit("--allow-smoke requires --output outside the canonical strong-mock artifact path")
        return
    total_hands = int(args.generations) * int(args.matches_per_generation) * int(args.hands)
    if int(args.hands) < 400:
        raise SystemExit("--hands must be at least 400 unless --allow-smoke is set")
    if total_hands < min_training_hands:
        raise SystemExit(
            f"training budget {total_hands} hands is below --min-training-hands "
            f"{min_training_hands}; use --allow-smoke only with a non-canonical --output"
        )


def train(args) -> dict:
    output = Path(args.output or (DEFAULT_PPO_OUTPUT if args.kind == "ppo" else DEFAULT_BUCKET_OUTPUT))
    validate_training_budget(args, output)
    output_exists_before_training = output.is_file()
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
    best_metric = -float("inf")
    best_state: dict | None = None
    best_row: dict | None = None
    stale_generations = 0
    early_stopped = False
    stop_generation = None
    replay_batches: list[dict[str, np.ndarray]] = []
    for generation in range(args.generations):
        pre_update_state = _export_state(args.kind, params, mean, scale, prefs, strategy_sum)
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
                "extra_bot_paths": args.extra_bot_path or [],
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
        mean_delta = float(np.mean([match["mean_train_delta"] for match in matches])) if matches else 0.0
        train_bust_count = int(sum(match.get("train_bust_count", 0) for match in matches))
        train_seat_count = int(sum(match.get("train_seat_count", 0) for match in matches))
        train_bust_rate = train_bust_count / max(1, train_seat_count)
        selection_score = mean_delta - float(args.selection_bust_penalty) * train_bust_rate
        bot_error_count = int(sum(match.get("bot_error_count", 0) for match in matches))
        row = {
            "generation": generation,
            "kind": args.kind,
            "matches": len(matches),
            "hands": int(sum(match["hands"] for match in matches)),
            "decisions": int(data["reward"].shape[0]),
            "mean_train_delta": round(mean_delta, 3),
            "selection_score": round(selection_score, 3),
            "train_bust_count": train_bust_count,
            "train_bust_rate": round(train_bust_rate, 4),
            "mean_decision_reward": round(float(np.mean(data["reward"])) if data["reward"].size else 0.0, 5),
            "bot_error_count": bot_error_count,
        }
        if data["action"].size:
            counts = np.bincount(data["action"], minlength=len(ACTION_LABELS))
            row["action_counts"] = {
                ACTION_LABELS[index]: int(counts[index])
                for index in range(len(ACTION_LABELS))
                if int(counts[index]) > 0
            }
        if generation >= args.selection_warmup:
            previous_best = best_metric
            if selection_score > best_metric:
                best_metric = selection_score
                best_state = pre_update_state
                best_row = dict(row)
            if previous_best == -float("inf") or selection_score > previous_best + args.early_stop_min_delta:
                stale_generations = 0
            elif args.early_stop_patience > 0:
                stale_generations += 1

        if args.kind == "ppo":
            replay_batches.append(data)
            if args.replay_generations > 0:
                replay_batches = replay_batches[-args.replay_generations:]
                update_data = _concat_data(
                    replay_batches,
                    max_rows=args.replay_max_decisions,
                    rng=rng,
                )
            else:
                replay_batches = []
                update_data = data
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
            mean, scale = _update_feature_norm(
                mean,
                scale,
                data["x"],
                generation,
                args.feature_norm_momentum,
                args.resume,
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
        row["update_loss"] = round(float(loss), 6)
        reports.append(row)
        _append_jsonl(args.log_jsonl, row)
        if args.progress:
            print(json.dumps(row), file=sys.stderr)
        if args.snapshot_interval > 0 and (generation + 1) % args.snapshot_interval == 0:
            snapshots.append(_snapshot(args.kind, params, mean, scale, prefs, args))
            snapshots = snapshots[-args.max_snapshots:]
        if (
            args.early_stop_patience > 0
            and generation >= args.selection_warmup
            and stale_generations >= args.early_stop_patience
        ):
            early_stopped = True
            stop_generation = generation
            break

    selected_mean_delta = (
        best_row["mean_train_delta"]
        if args.export_best and best_row is not None
        else reports[-1]["mean_train_delta"] if reports else None
    )
    selected_generation = (
        best_row["generation"]
        if args.export_best and best_row is not None
        else reports[-1]["generation"] if reports else None
    )
    export_allowed = should_export_policy(
        selected_mean_delta,
        output_exists_before_training,
        args.min_export_mean_delta,
    )
    export_skipped_reason = None
    if not export_allowed:
        export_skipped_reason = (
            f"selected mean_train_delta {selected_mean_delta} is below "
            f"min_export_mean_delta {args.min_export_mean_delta}; kept existing artifact"
        )
    report = {
        "mode": "real_fullhouse_self_play",
        "kind": args.kind,
        "output": str(output),
        "seed": args.seed,
        "generations": len(reports),
        "generations_requested": args.generations,
        "matches_per_generation": args.matches_per_generation,
        "hands": args.hands,
        "players": args.players,
        "train_seats": args.train_seats,
        "final_mean_train_delta": reports[-1]["mean_train_delta"] if reports else 0.0,
        "mean_train_delta": selected_mean_delta if selected_mean_delta is not None else 0.0,
        "exported_generation": selected_generation,
        "exported": bool(export_allowed),
        "output_existed_before_training": bool(output_exists_before_training),
        "min_export_mean_delta": args.min_export_mean_delta,
        "export_skipped_reason": export_skipped_reason,
        "early_stopped": bool(early_stopped),
        "stop_generation": stop_generation,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "selection_bust_penalty": getattr(args, "selection_bust_penalty", None),
        "best_generation": best_row["generation"] if best_row is not None else None,
        "best_selection_score": best_row["selection_score"] if best_row is not None else None,
        "best_mean_train_delta": best_row["mean_train_delta"] if best_row is not None else None,
        "abstraction": args.abstraction,
        "bucket_update": args.bucket_update,
        "ppo_clip_ratio": getattr(args, "ppo_clip_ratio", None),
        "ppo_value_coef": getattr(args, "ppo_value_coef", None),
        "feature_norm_momentum": getattr(args, "feature_norm_momentum", None),
        "replay_generations": getattr(args, "replay_generations", None),
        "replay_max_decisions": getattr(args, "replay_max_decisions", None),
        "temperature": getattr(args, "temperature", None),
        "reports": reports[-5:],
    }
    export_state = best_state if args.export_best and best_state is not None else _export_state(
        args.kind,
        params,
        mean,
        scale,
        prefs,
        strategy_sum,
    )
    if export_allowed:
        if args.kind == "ppo":
            _export_ppo(output, export_state["params"], export_state["mean"], export_state["scale"], report)
        else:
            _export_bucket(output, export_state["prefs"], export_state["strategy_sum"], report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strong mock policies from actual Fullhouse self-play")
    parser.add_argument("--kind", choices=["ppo", "bucket"], default="ppo")
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--matches-per-generation", type=int, default=128)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--train-seats", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--bucket-count", type=int, default=32768)
    parser.add_argument("--abstraction", choices=["feature", "cfr-pokerbot"], default="feature")
    parser.add_argument("--bucket-update", choices=["policy-gradient", "cfr-plus"], default="policy-gradient")
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.20)
    parser.add_argument("--ppo-value-coef", type=float, default=0.35)
    parser.add_argument("--max-grad-norm", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--feature-norm-momentum", type=float, default=0.0)
    parser.add_argument("--replay-generations", type=int, default=4)
    parser.add_argument("--replay-max-decisions", type=int, default=24000)
    parser.add_argument("--temperature", type=float, default=0.62)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument("--opponent-pool", choices=["oracle", "fast", "mixed", "adversarial"], default="adversarial")
    parser.add_argument("--extra-bot-path", action="append", default=[])
    parser.add_argument("--snapshot-interval", type=int, default=2)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument("--snapshot-prob", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--seed", type=int, default=6161)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-best", action="store_true")
    parser.add_argument("--selection-warmup", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--selection-bust-penalty", type=float, default=8000.0)
    parser.add_argument("--min-export-mean-delta", type=float, default=-1_000_000_000.0)
    parser.add_argument("--min-training-hands", type=int, default=1_000_000)
    parser.add_argument("--allow-smoke", action="store_true")
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
