"""Mock competitor: trained numpy MLP policy.

The model is trained offline by tools/train_mock_numpy_policy.py and stored as
data/policy.npz. At runtime this bot only performs a small numpy forward pass.
"""

import os
import numpy as np


DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))


def _load_policy():
    path = os.path.join(DATA_DIR, "policy.npz")
    try:
        data = np.load(path, allow_pickle=False)
        return {
            "w1": data["w1"].astype(float),
            "b1": data["b1"].astype(float),
            "w2": data["w2"].astype(float),
            "b2": data["b2"].astype(float),
            "mean": data["mean"].astype(float),
            "scale": data["scale"].astype(float),
        }
    except Exception:
        return None


POLICY = _load_policy()
FALLBACK_WEIGHTS = np.array([0.65, -0.45, -0.30, 0.25, 0.20, -0.55, 0.35, 0.12, -0.08, 0.16], dtype=float)


def _features(state):
    cards = state.get("your_cards", [])
    high = 0.0
    pair = 0.0
    suited = 0.0
    if len(cards) >= 2:
        vals = ["23456789TJQKA".index(c[0]) + 2 for c in cards]
        high = max(vals) / 14
        pair = 1.0 if vals[0] == vals[1] else 0.0
        suited = 1.0 if cards[0][1] == cards[1][1] else 0.0
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    stack = max(1, int(state.get("your_stack", 0) or 0))
    active = sum(1 for p in state.get("players", []) if not p.get("is_folded"))
    street = state.get("street")
    return np.array([
        1.0,
        owed / max(1, pot + owed),
        min(1.0, pot / 20000),
        high,
        pair,
        min(1.0, owed / stack),
        suited,
        min(1.0, active / 6),
        1.0 if street in ("turn", "river") else 0.0,
        1.0 if state.get("can_check") else 0.0,
    ], dtype=float)


def _probs(x):
    if POLICY is None:
        score = float(FALLBACK_WEIGHTS @ x)
        raise_score = score - 0.92
        call_score = score - 0.55
        fold_score = 0.25 - score
        logits = np.array([fold_score, call_score, raise_score], dtype=float)
    else:
        z = (x - POLICY["mean"]) / POLICY["scale"]
        hidden = np.tanh(z @ POLICY["w1"] + POLICY["b1"])
        logits = hidden @ POLICY["w2"] + POLICY["b2"]
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / max(1e-9, float(np.sum(exp)))


def _raise_to(state, frac):
    pot = max(1, int(state.get("pot", 0) or 0))
    total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
    amount = max(int(state.get("min_raise_to", 0) or 0), int(state.get("current_bet", 0) or 0) + int(pot * frac))
    return {"action": "raise", "amount": min(total, amount)}


def decide(state):
    probs = _probs(_features(state))
    fold_p, call_p, raise_p = [float(v) for v in probs]
    if state.get("can_check"):
        if raise_p > 0.62:
            return _raise_to(state, 0.80)
        if raise_p > 0.42:
            return _raise_to(state, 0.50)
        return {"action": "check"}
    if raise_p > 0.72:
        return _raise_to(state, 1.0)
    if call_p >= fold_p:
        return {"action": "call"}
    return {"action": "fold"}
