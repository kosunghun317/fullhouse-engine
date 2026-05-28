"""Contextual selector over human-heuristic poker expert arms."""

from __future__ import annotations

import os
import random
from functools import lru_cache

import numpy as np

from tools.strong_mocks.arm_selector_params import coerce_params, load_params
from tools.strong_mocks.expert_arms import (
    ARM_NAMES,
    INTENTS,
    TARGET_TYPES,
    Candidate,
    default_candidate_score,
    generate_candidates,
)
from tools.strong_mocks.features import FEATURE_NAMES, extract_features


ARM_INDEX = {name: index for index, name in enumerate(ARM_NAMES)}
INTENT_INDEX = {name: index for index, name in enumerate(INTENTS)}
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_TYPES)}

CANDIDATE_META_NAMES = (
    "candidate_risk",
    "candidate_confidence",
    "candidate_equity",
    "candidate_score_hint",
    "candidate_raise_fraction",
    "candidate_is_passive",
    "candidate_is_raise",
    "candidate_is_all_in",
    "candidate_value_minus_risk",
    "candidate_bluff_pressure_proxy",
)

SELECTOR_FEATURE_NAMES = (
    *FEATURE_NAMES,
    *CANDIDATE_META_NAMES,
    *(f"intent_{name}" for name in INTENTS),
    *(f"target_{name}" for name in TARGET_TYPES),
    *(f"arm_{name}" for name in ARM_NAMES),
)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _state_rng(state: dict, salt: str = "") -> random.Random:
    parts = [
        salt,
        str(state.get("hand_id", "")),
        str(state.get("seat_to_act", "")),
        str(len(state.get("action_log", []) or [])),
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


def candidate_feature_vector(state: dict, candidate: Candidate, base_features: np.ndarray | None = None) -> np.ndarray:
    base = extract_features(state).astype(np.float32) if base_features is None else np.asarray(base_features, dtype=np.float32)
    action = candidate.action
    act = action.get("action")
    pot = max(1.0, float(state.get("pot", 0) or 0))
    amount = float(action.get("amount", 0) or 0)
    raise_fraction = _clip(amount / max(1.0, pot), 0.0, 3.0) / 3.0 if act == "raise" else 0.0
    passive = float(act in ("fold", "check", "call"))
    meta = np.asarray(
        [
            candidate.risk,
            candidate.confidence,
            candidate.equity_estimate,
            _clip((candidate.score_hint + 0.2) / 0.5),
            raise_fraction,
            passive,
            float(act == "raise"),
            float(act == "all_in"),
            _clip(candidate.equity_estimate - candidate.risk + 0.5),
            _clip(candidate.confidence * (1.0 - candidate.risk)),
        ],
        dtype=np.float32,
    )
    intents = np.zeros(len(INTENTS), dtype=np.float32)
    if candidate.intent in INTENT_INDEX:
        intents[INTENT_INDEX[candidate.intent]] = 1.0
    targets = np.zeros(len(TARGET_TYPES), dtype=np.float32)
    targets[TARGET_INDEX.get(candidate.target_type, 0)] = 1.0
    arms = np.zeros(len(ARM_NAMES), dtype=np.float32)
    arms[ARM_INDEX.get(candidate.arm, 0)] = 1.0
    return np.concatenate([base, meta, intents, targets, arms]).astype(np.float32)


def candidate_matrix(
    state: dict,
    candidates: list[Candidate] | None = None,
    params: dict[str, float] | None = None,
) -> tuple[list[Candidate], np.ndarray]:
    params = coerce_params(params)
    rows = candidates if candidates is not None else generate_candidates(state, params)
    base = extract_features(state).astype(np.float32) if state.get("type") != "warmup" else np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if not rows:
        rows = generate_candidates({"type": "warmup"})
    matrix = np.stack([candidate_feature_vector(state, candidate, base) for candidate in rows]).astype(np.float32)
    return rows, matrix


def fallback_scores(candidates: list[Candidate], params: dict[str, float] | None = None) -> np.ndarray:
    params = coerce_params(params)
    return np.asarray([default_candidate_score(candidate, params) for candidate in candidates], dtype=np.float64)


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


def _model_scores(model: dict | None, x: np.ndarray, candidates: list[Candidate], params: dict[str, float]) -> np.ndarray:
    scores = fallback_scores(candidates, params)
    if model is None or "weights" not in model:
        return scores
    weights = model["weights"].astype(np.float64)
    bias = float(np.asarray(model.get("bias", np.asarray([0.0]))).reshape(-1)[0])
    mean = model.get("mean")
    scale = model.get("scale")
    values = x.astype(np.float64)
    if mean is not None and scale is not None and len(mean) == values.shape[1]:
        values = (values - mean.astype(np.float64)) / np.maximum(1e-6, scale.astype(np.float64))
    learned = values @ weights + bias
    blend = float(np.asarray(model.get("learned_blend", np.asarray([0.72]))).reshape(-1)[0])
    return (1.0 - blend) * scores + blend * learned


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64) / max(1e-6, float(temperature))
    z -= float(np.max(z))
    exp = np.exp(z)
    total = float(np.sum(exp))
    if total <= 0:
        return np.ones_like(exp) / max(1, exp.size)
    return exp / total


def choose_candidate(
    state: dict,
    data_dir: str,
    sample: bool = False,
    candidates: list[Candidate] | None = None,
    params: dict[str, float] | None = None,
) -> tuple[Candidate, dict]:
    params = load_params(data_dir) if params is None else coerce_params(params)
    rows, matrix = candidate_matrix(state, candidates, params)
    model = _load_npz(_policy_path(data_dir))
    scores = _model_scores(model, matrix, rows, params)
    temperature = 0.42
    if model is not None and "temperature" in model:
        temperature = float(np.asarray(model["temperature"]).reshape(-1)[0])
    if sample:
        probs = _softmax(scores, temperature)
        rng = _state_rng(state, salt=data_dir)
        draw = rng.random()
        running = 0.0
        index = 0
        for i, prob in enumerate(probs):
            running += float(prob)
            if draw <= running:
                index = i
                break
    else:
        probs = _softmax(scores, temperature)
        index = int(np.argmax(scores))
    return rows[index], {
        "index": index,
        "scores": scores,
        "probs": probs,
        "features": matrix,
        "model_loaded": model is not None,
    }


def decide_arm_selector(state: dict, data_dir: str, sample: bool = False) -> dict:
    if state.get("type") == "warmup":
        return {"action": "check"}
    candidate, _info = choose_candidate(state, data_dir, sample=sample)
    return dict(candidate.action)
