"""Mock competitor: tight range-aware equity caller."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import equity_decide


CONFIG = {
    "preflop_bias": -0.03,
    "postflop_bias": -0.02,
    "opponent_penalty": 0.070,
    "samples": 150,
    "value_threshold": 0.76,
    "semi_threshold": 0.55,
    "semi_prob": 0.18,
    "call_margin": 0.12,
    "value_frac": 0.70,
    "semi_frac": 0.50,
    "raise_threshold": 0.88,
}


def decide(state):
    return equity_decide(state, CONFIG)
