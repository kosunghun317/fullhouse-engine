"""Inference policies for benchmark-only strong mock bots."""

from __future__ import annotations

import os
import random
from functools import lru_cache

import eval7
import numpy as np

from tools.strong_mocks.actions import ACTION_LABELS, action_index_to_action, masked_argmax
from tools.strong_mocks.abstractions import abstract_bucket_id
from tools.strong_mocks.dataset import oracle_logits
from tools.strong_mocks.features import extract_features


RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = [eval7.Card(rank + suit) for rank in RANKS for suit in SUITS]


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / max(1e-9, float(np.sum(exp)))


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


def _rollout_equity(state: dict, samples: int = 360, budget_opponents: int | None = None) -> float:
    board = state.get("community_cards", [])
    cards = state.get("your_cards", [])
    if not board:
        active = sum(1 for player in state.get("players", []) if not player.get("is_folded"))
        return max(0.03, _preflop_strength(cards) - 0.055 * max(0, active - 2))
    try:
        hero = [eval7.Card(card) for card in cards]
        community = [eval7.Card(card) for card in board]
    except Exception:
        return 0.0
    active = sum(1 for player in state.get("players", []) if not player.get("is_folded"))
    opponents = max(1, min(5, active - 1 if budget_opponents is None else budget_opponents))
    dead = set(hero + community)
    deck = [card for card in FULL_DECK if card not in dead]
    need = opponents * 2 + max(0, 5 - len(community))
    if len(deck) < need:
        return 0.0
    wins = 0.0
    for _ in range(max(1, samples)):
        random.shuffle(deck)
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


def decide_rollout(state: dict, style: str = "balanced") -> dict:
    samples = 520 if style == "rollout_deep" else 300
    equity = _rollout_equity(state, samples=samples)
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    odds = owed / max(1, pot + owed)
    can_check = bool(state.get("can_check"))
    active = sum(1 for player in state.get("players", []) if not player.get("is_folded"))
    margin = 0.06 + 0.035 * max(0, active - 2)
    if style == "rollout_pressure":
        margin -= 0.02
    if can_check:
        if equity > 0.72:
            return action_index_to_action(state, 5)
        if equity > 0.58:
            return action_index_to_action(state, 4)
        if equity > 0.40 and style == "rollout_pressure":
            return action_index_to_action(state, 3)
        return action_index_to_action(state, 1)
    if equity > odds + margin:
        if equity > 0.86 and owed < pot * 0.22:
            return action_index_to_action(state, 5)
        return action_index_to_action(state, 1)
    if equity > odds + 0.01 and style == "rollout_pressure":
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
