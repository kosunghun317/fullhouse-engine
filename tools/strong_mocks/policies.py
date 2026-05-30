"""Inference policies for benchmark-only strong mock bots."""

from __future__ import annotations

import os
import random
from functools import lru_cache

import eval7
import numpy as np

from tools.strong_mocks.actions import ACTION_LABELS, action_index_to_action, masked_argmax
from tools.strong_mocks.actions import raise_to_fraction
from tools.strong_mocks.abstractions import abstract_bucket_id
from tools.strong_mocks.dataset import oracle_logits
from tools.strong_mocks.features import extract_features


RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = [eval7.Card(rank + suit) for rank in RANKS for suit in SUITS]
ROLLOUT_DEFAULT_PARAMS = {
    "samples": 1024.0,
    "deep_samples": 1024.0,
    "f_min_equity": 0.40,
    "f_center": 0.61,
    "f_scale": 0.075,
    "f_min_raise": 0.48,
    "f_max_raise": 1.05,
    "g_call_edge_base": 0.06,
    "g_call_edge_active": 0.035,
    "g_pressure_discount": 0.02,
    "g_thin_call_edge": 0.01,
    "g_raise_edge": 0.16,
    "g_raise_equity": 0.86,
    "g_raise_max_owed_pot": 0.22,
    "g_raise_center": 0.87,
    "g_raise_scale": 0.045,
    "g_min_raise": 0.80,
    "g_max_raise": 1.20,
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / max(1e-9, float(np.sum(exp)))


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + float(np.exp(-value)))


def rollout_params(overrides: dict[str, float] | None = None) -> dict[str, float]:
    params = dict(ROLLOUT_DEFAULT_PARAMS)
    if overrides:
        for key, value in overrides.items():
            if key in params:
                params[key] = float(value)
    params["samples"] = max(1.0, params["samples"])
    params["deep_samples"] = max(1.0, params["deep_samples"])
    params["f_scale"] = max(0.005, params["f_scale"])
    params["g_raise_scale"] = max(0.005, params["g_raise_scale"])
    params["f_min_raise"] = max(0.10, min(2.50, params["f_min_raise"]))
    params["f_max_raise"] = max(params["f_min_raise"], min(2.50, params["f_max_raise"]))
    params["g_min_raise"] = max(0.10, min(2.50, params["g_min_raise"]))
    params["g_max_raise"] = max(params["g_min_raise"], min(2.50, params["g_max_raise"]))
    return params


def rollout_check_raise_fraction(equity: float, params: dict[str, float] | None = None) -> float:
    """f(equity): raise size when checking is available, or 0.0 to check."""
    params = rollout_params(params)
    equity = float(equity)
    if equity < params["f_min_equity"]:
        return 0.0
    weight = _sigmoid((equity - params["f_center"]) / params["f_scale"])
    return params["f_min_raise"] + weight * (params["f_max_raise"] - params["f_min_raise"])


def rollout_facing_bet_raise_fraction(
    equity: float,
    pot_odds: float,
    active_players: int,
    owed_pot_ratio: float,
    style: str = "balanced",
    params: dict[str, float] | None = None,
) -> float:
    """g(equity, odds, context): -1 fold, 0 call, positive raise fraction."""
    params = rollout_params(params)
    edge = float(equity) - float(pot_odds)
    active_extra = max(0, int(active_players) - 2)
    call_edge = params["g_call_edge_base"] + params["g_call_edge_active"] * active_extra
    if style == "rollout_pressure":
        call_edge -= params["g_pressure_discount"]
    if edge < call_edge:
        if style == "rollout_pressure" and edge > params["g_thin_call_edge"]:
            return 0.0
        return -1.0
    if (
        edge < params["g_raise_edge"]
        or float(equity) < params["g_raise_equity"]
        or float(owed_pot_ratio) > params["g_raise_max_owed_pot"]
    ):
        return 0.0
    weight = _sigmoid((float(equity) - params["g_raise_center"]) / params["g_raise_scale"])
    return params["g_min_raise"] + weight * (params["g_max_raise"] - params["g_min_raise"])


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
    action_log = state.get("action_log", [])
    parts = [
        salt,
        str(state.get("hand_id", "")),
        str(state.get("seat_to_act", "")),
        str(len(action_log)),
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


def _mlp_logits(model: dict, x: np.ndarray) -> np.ndarray:
    mean = model.get("mean")
    scale = model.get("scale")
    if mean is not None and scale is not None:
        x = (x - mean.astype(float)) / np.maximum(1e-6, scale.astype(float))
    hidden = np.tanh(x @ model["w1"].astype(float) + model["b1"].astype(float))
    logits = hidden @ model["w2"].astype(float) + model["b2"].astype(float)
    classes = model.get("classes")
    if classes is None or len(classes) == len(ACTION_LABELS):
        return np.asarray(logits, dtype=float)
    full = np.full(len(ACTION_LABELS), -4.0, dtype=float)
    for offset, cls in enumerate(classes.astype(int)):
        if 0 <= cls < len(full):
            full[cls] = logits[offset]
    return full


def logits_from_model(state: dict, data_dir: str, fallback_style: str = "balanced") -> np.ndarray:
    x = extract_features(state)
    model = _load_npz(_policy_path(data_dir))
    if model is None:
        return oracle_logits(x.reshape(1, -1), fallback_style)[0]

    model_type = str(model.get("model_type", np.asarray(["mlp"]))[0])
    if model_type == "cfr_table":
        table = model["policy_table"].astype(float)
        abstraction = str(model.get("abstraction", np.asarray(["feature"]))[0])
        idx = abstract_bucket_id(state, int(table.shape[0]), abstraction)
        probs = table[idx]
        return np.log(np.maximum(1e-6, probs))
    if model_type in ("mlp", "ppo_mlp"):
        return _mlp_logits(model, x)
    return oracle_logits(x.reshape(1, -1), fallback_style)[0]


def decide_model(state: dict, data_dir: str, fallback_style: str = "balanced") -> dict:
    logits = logits_from_model(state, data_dir, fallback_style)
    return action_index_to_action(state, masked_argmax(logits, state))


def decide_model_sampled(
    state: dict,
    data_dir: str,
    fallback_style: str = "balanced",
    default_temperature: float = 0.62,
) -> dict:
    logits = logits_from_model(state, data_dir, fallback_style)
    model = _load_npz(_policy_path(data_dir))
    temperature = default_temperature
    if model is not None and "temperature" in model:
        try:
            temperature = float(np.asarray(model["temperature"]).reshape(-1)[0])
        except Exception:
            temperature = default_temperature
    from tools.strong_mocks.actions import strategic_mask

    mask = strategic_mask(state)
    probs = _softmax_masked(logits, mask, temperature)
    rng = _state_rng(state, salt=str(data_dir))
    return action_index_to_action(state, _sample_index(probs, rng))


def decide_cfr(state: dict, data_dir: str, fallback_style: str = "pressure") -> dict:
    return decide_model(state, data_dir, fallback_style=fallback_style)


def _rank_value(card: str) -> int:
    return RANKS.index(card[0]) + 2 if card and card[0] in RANKS else 2


def _preflop_strength(cards: list[str]) -> float:
    if len(cards) < 2:
        return 0.15
    vals = sorted((_rank_value(card) for card in cards[:2]), reverse=True)
    pair = vals[0] == vals[1]
    suited = len(cards[0]) > 1 and len(cards[1]) > 1 and cards[0][1] == cards[1][1]
    score = 0.42 * vals[0] / 14 + 0.20 * vals[1] / 14
    if pair:
        score += 0.30 + vals[0] / 120
    if suited:
        score += 0.06
    if vals[0] == 14:
        score += 0.06
    return max(0.03, min(0.96, score))


def _rollout_equity(
    state: dict,
    samples: int = 360,
    budget_opponents: int | None = None,
    rng: random.Random | None = None,
) -> float:
    board = state.get("community_cards", [])
    cards = state.get("your_cards", [])
    try:
        hero = [eval7.Card(card) for card in cards]
        community = [eval7.Card(card) for card in board]
    except Exception:
        return 0.0
    if len(hero) < 2:
        active = sum(1 for player in state.get("players", []) if not player.get("is_folded"))
        return max(0.03, _preflop_strength(cards) - 0.055 * max(0, active - 2))
    active = sum(1 for player in state.get("players", []) if not player.get("is_folded"))
    opponents = max(1, min(5, active - 1 if budget_opponents is None else budget_opponents))
    dead = set(hero + community)
    deck = [card for card in FULL_DECK if card not in dead]
    need = opponents * 2 + max(0, 5 - len(community))
    if len(deck) < need:
        return 0.0
    wins = 0.0
    rng = rng or _state_rng(state, salt="rollout_equity")
    for _ in range(max(1, samples)):
        rng.shuffle(deck)
        idx = 0
        opp_hands = []
        for _opp in range(opponents):
            opp_hands.append([deck[idx], deck[idx + 1]])
            idx += 2
        runout = list(community)
        while len(runout) < 5:
            runout.append(deck[idx])
            idx += 1
        hero_score = eval7.evaluate(hero + runout)
        opp_scores = [eval7.evaluate(hand + runout) for hand in opp_hands]
        best = max([hero_score] + opp_scores)
        if hero_score == best:
            wins += 1.0 / (1 + sum(1 for score in opp_scores if score == best))
    return wins / max(1, samples)


def decide_rollout(state: dict, style: str = "balanced", params: dict[str, float] | None = None) -> dict:
    params = rollout_params(params)
    samples = int(params["deep_samples"] if style == "rollout_deep" else params["samples"])
    equity = _rollout_equity(state, samples=samples, rng=_state_rng(state, salt=style))
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    odds = owed / max(1, pot + owed)
    can_check = bool(state.get("can_check"))
    active = sum(1 for player in state.get("players", []) if not player.get("is_folded"))
    if can_check:
        fraction = rollout_check_raise_fraction(equity, params)
        if fraction > 0.0:
            return raise_to_fraction(state, fraction)
        return action_index_to_action(state, 1)
    owed_pot_ratio = owed / max(1.0, float(pot))
    fraction = rollout_facing_bet_raise_fraction(equity, odds, active, owed_pot_ratio, style, params)
    if fraction > 0.0:
        return raise_to_fraction(state, fraction)
    if fraction == 0.0:
        return action_index_to_action(state, 1)
    return action_index_to_action(state, 0)


def decide_ensemble(state: dict, data_dirs: dict[str, str]) -> dict:
    key = str(state.get("hand_id", "")) + ":" + str(state.get("seat_to_act", ""))
    selector = sum(ord(ch) for ch in key) % 5
    if selector == 0:
        return decide_model(state, data_dirs.get("oracle", ""), fallback_style="value")
    if selector == 1:
        return decide_model(state, data_dirs.get("ppo", ""), fallback_style="pressure")
    if selector == 2:
        return decide_cfr(state, data_dirs.get("cfr", ""), fallback_style="bluff")
    if selector == 3:
        return decide_rollout(state, style="rollout_pressure")
    logits = (
        logits_from_model(state, data_dirs.get("oracle", ""), "balanced")
        + logits_from_model(state, data_dirs.get("ppo", ""), "pressure")
    )
    return action_index_to_action(state, masked_argmax(logits, state))
