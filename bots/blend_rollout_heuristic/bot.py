"""Benchmark-only 50/50 blend of standalone rollout and current heuristic.

This wrapper is for local comparison only. It imports the checked-out heuristic
bot and the standalone rollout submission candidate, then chooses one policy
per decision with a deterministic state-derived coin flip.
"""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.heuristic import bot as heuristic_bot
from bots.rollout_search_submission import bot as rollout_bot


def _coin_seed(state):
    action_log = state.get("action_log", []) if isinstance(state, dict) else []
    parts = [
        str(state.get("hand_id", "")),
        str(state.get("seat_to_act", "")),
        str(len(action_log)),
        str(state.get("street", "")),
        str(state.get("pot", "")),
        str(state.get("amount_owed", "")),
        ",".join(str(card) for card in (state.get("your_cards", []) or [])),
        ",".join(str(card) for card in (state.get("community_cards", []) or [])),
    ]
    seed = 0
    for ch in "|".join(parts):
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    return seed


def decide(state):
    if not isinstance(state, dict) or state.get("type") == "warmup":
        return {"action": "check"}
    if _coin_seed(state) % 2 == 0:
        return rollout_bot.decide(state)
    return heuristic_bot.decide(state)
