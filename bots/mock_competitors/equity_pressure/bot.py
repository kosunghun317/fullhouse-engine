"""Mock competitor: equity bot that pressures fold-prone states."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import equity_decide


CONFIG = {
    "preflop_bias": 0.01,
    "postflop_bias": 0.00,
    "opponent_penalty": 0.055,
    "samples": 150,
    "value_threshold": 0.68,
    "semi_threshold": 0.38,
    "semi_prob": 0.55,
    "call_margin": 0.04,
    "value_frac": 0.86,
    "semi_frac": 0.72,
    "raise_threshold": 0.78,
    "raise_frac": 1.05,
}


def decide(state):
    return equity_decide(state, CONFIG)
