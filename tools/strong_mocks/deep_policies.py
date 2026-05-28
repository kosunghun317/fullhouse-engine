"""Inference helpers for the benchmark-only deep PPO mock."""

from __future__ import annotations

import os
import random
from functools import lru_cache

import numpy as np

from tools.strong_mocks.deep_actions import ACTION_LABELS, action_index_to_action, strategic_mask
from tools.strong_mocks.features import FEATURE_NAMES, extract_features


DERIVED_FEATURE_NAMES = (
    "strength_x_position",
    "strength_x_can_check",
    "strength_minus_owed",
    "made_x_wet",
    "draw_x_wet",
    "draw_x_position",
    "risk_signal",
    "value_signal",
    "bluff_signal",
    "multiway",
    "short_stack",
    "large_owed",
    "spr_x_strength",
    "stack_share_x_risk",
    "pressure_spot",
    "showdown_spot",
)

DEEP_FEATURE_NAMES = FEATURE_NAMES + DERIVED_FEATURE_NAMES


def augment_features(base: np.ndarray) -> np.ndarray:
    """Append stable interaction features to the shared 32-feature vector."""
    x = np.asarray(base, dtype=np.float32)
    if x.ndim == 1:
        x2 = x.reshape(1, -1)
        squeeze = True
    else:
        x2 = x
        squeeze = False
    owed = x2[:, 5]
    spr = x2[:, 8]
    can_check = x2[:, 9]
    active = x2[:, 10]
    position = x2[:, 11]
    strength = x2[:, 17]
    wet = x2[:, 18]
    made = x2[:, 22]
    draw = np.maximum(x2[:, 23], x2[:, 24])
    raises = x2[:, 29]
    allins = x2[:, 30]
    stack_share = x2[:, 31]
    risk = np.clip(owed + 0.25 * raises + 0.35 * allins + 0.12 * active, 0.0, 2.0)
    value = np.clip(0.65 * strength + 0.22 * made + 0.10 * draw - 0.08 * wet, 0.0, 1.3)
    bluff = np.clip(0.20 * position + 0.18 * (1.0 - wet) + 0.15 * draw - 0.10 * active, -0.5, 1.0)
    derived = np.column_stack([
        strength * position,
        strength * can_check,
        strength - owed,
        made * wet,
        draw * wet,
        draw * position,
        risk,
        value,
        bluff,
        (active > 0.45).astype(np.float32),
        (spr < 0.18).astype(np.float32),
        (owed > 0.35).astype(np.float32),
        spr * strength,
        stack_share * risk,
        ((can_check > 0.5) & (bluff > 0.10)).astype(np.float32),
        ((owed > 0.0) & (value > risk)).astype(np.float32),
    ]).astype(np.float32)
    out = np.concatenate([x2, derived], axis=1)
    return out[0] if squeeze else out


def extract_deep_features(state: dict) -> np.ndarray:
    return augment_features(extract_features(state))


def deep_oracle_logits(x: np.ndarray, style: str = "pressure") -> np.ndarray:
    """Teacher policy over the expanded action arms used for bootstrap only."""
    values = np.asarray(x, dtype=np.float32)
    owed = values[:, 5]
    can_check = values[:, 9]
    active = values[:, 10]
    position = values[:, 11]
    strength = values[:, 17]
    wet = values[:, 18]
    made = values[:, 22]
    draw = np.maximum(values[:, 23], values[:, 24])
    raises = values[:, 29]
    allins = values[:, 30]
    risk = values[:, 38]
    value = values[:, 39]
    bluff = values[:, 40]
    short = values[:, 42]

    if style == "conservative":
        risk = risk + 0.10
        bluff = bluff - 0.05
    elif style == "explore":
        bluff = bluff + 0.12
    elif style == "value":
        value = value + 0.10 * made
        bluff = bluff - 0.04
    elif style == "pressure":
        bluff = bluff + 0.10 + 0.06 * can_check

    logits = np.full((values.shape[0], len(ACTION_LABELS)), -0.25, dtype=np.float32)
    logits[:, 0] = 0.55 + risk - value - 1.50 * can_check
    logits[:, 1] = 0.28 + value - 0.50 * risk + 0.25 * can_check
    bet_drive = np.maximum(0.0, value + bluff - 0.33)
    logits[:, 2] = value + 0.70 * bluff - 0.16 - 0.22 * owed
    logits[:, 3] = value + 1.10 * bluff - 0.20 - 0.28 * owed
    logits[:, 4] = value + 1.06 * bluff - 0.18 - 0.30 * owed
    logits[:, 5] = value + 0.98 * bluff - 0.16 - 0.35 * owed
    logits[:, 6] = value + 0.90 * bluff - 0.15 - 0.40 * owed
    logits[:, 7] = value + 0.78 * bluff + 0.03 * draw - 0.18 - 0.45 * owed
    logits[:, 8] = value + 0.62 * bluff + 0.08 * made - 0.24 - 0.52 * owed
    logits[:, 9] = value + 0.45 * bluff + 0.13 * made - 0.32 - 0.58 * owed
    logits[:, 10] = value + 0.30 * bluff + 0.20 * made - 0.43 - 0.65 * owed
    logits[:, 11] = value + 0.20 * bluff + 0.28 * made - 0.58 - 0.75 * owed
    logits[:, 12] = value + 0.10 * bluff + 0.36 * made - 0.80 - 0.92 * owed
    logits[:, 13] = value + 0.50 * made + 0.25 * short + 0.30 * allins - 0.95 - 0.72 * owed

    logits[:, 2:13] += (0.18 * can_check + 0.05 * position + 0.04 * draw)[:, None]
    logits[:, 10:13] += (0.20 * made + 0.10 * strength - 0.10 * wet - 0.08 * active)[:, None]
    logits[:, 13] += 0.30 * (strength > 0.86) + 0.25 * (made > 0.5) - 0.18 * raises
    logits[:, 1] -= 0.22 * can_check * bet_drive
    return logits


def deep_oracle_labels(x: np.ndarray, style: str = "pressure") -> np.ndarray:
    return np.argmax(deep_oracle_logits(x, style), axis=1).astype(np.int64)


def _softmax_masked(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=float).copy()
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


def _state_rng(state: dict, salt: str = "") -> random.Random:
    parts = [
        salt,
        str(state.get("hand_id", "")),
        str(state.get("seat_to_act", "")),
        str(len(state.get("action_log", []))),
        str(state.get("street", "")),
        str(state.get("pot", "")),
        str(state.get("amount_owed", "")),
        ",".join(map(str, state.get("your_cards", []) or [])),
        ",".join(map(str, state.get("community_cards", []) or [])),
    ]
    seed = 0
    for ch in "|".join(parts):
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    return random.Random(seed)


def _sample_index(probs: np.ndarray, rng: random.Random) -> int:
    draw = rng.random()
    total = 0.0
    for index, prob in enumerate(probs):
        total += float(prob)
        if draw <= total:
            return int(index)
    legal = np.flatnonzero(probs > 0)
    return int(legal[-1]) if legal.size else 0


@lru_cache(maxsize=32)
def _load_npz(path: str):
    if not path or not os.path.isfile(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    except Exception:
        return None


def _policy_path(data_dir: str) -> str:
    return os.path.join(data_dir, "policy.npz")


def layer_sizes(model: dict) -> list[int]:
    sizes = []
    index = 0
    while f"w{index}" in model:
        sizes.append(int(model[f"w{index}"].shape[1]))
        index += 1
    return sizes


def _forward_model(model: dict, x: np.ndarray) -> np.ndarray:
    mean = model.get("mean")
    scale = model.get("scale")
    values = np.asarray(x, dtype=float)
    if mean is not None and scale is not None:
        values = (values - mean.astype(float)) / np.maximum(1e-6, scale.astype(float))
    index = 0
    hidden = values
    while f"w{index}" in model:
        hidden = np.tanh(hidden @ model[f"w{index}"].astype(float) + model[f"b{index}"].astype(float))
        index += 1
    return hidden @ model["w_policy"].astype(float) + model["b_policy"].astype(float)


def logits_from_deep_model(state: dict, data_dir: str, fallback_style: str = "pressure") -> np.ndarray:
    x = extract_deep_features(state)
    model = _load_npz(_policy_path(data_dir))
    if model is None:
        return deep_oracle_logits(x.reshape(1, -1), fallback_style)[0]
    model_type = str(model.get("model_type", np.asarray(["deep_ppo_mlp"]))[0])
    if model_type != "deep_ppo_mlp":
        return deep_oracle_logits(x.reshape(1, -1), fallback_style)[0]
    return _forward_model(model, x)


def decide_deep_model_sampled(
    state: dict,
    data_dir: str,
    fallback_style: str = "pressure",
    default_temperature: float = 0.55,
) -> dict:
    logits = logits_from_deep_model(state, data_dir, fallback_style)
    model = _load_npz(_policy_path(data_dir))
    temperature = default_temperature
    if model is not None and "temperature" in model:
        try:
            temperature = float(np.asarray(model["temperature"]).reshape(-1)[0])
        except Exception:
            temperature = default_temperature
    mask = strategic_mask(state)
    probs = _softmax_masked(logits, mask, temperature)
    rng = _state_rng(state, salt=str(data_dir))
    return action_index_to_action(state, _sample_index(probs, rng))
