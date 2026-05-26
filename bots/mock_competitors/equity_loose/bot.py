"""Mock competitor: loose equity caller / station-like simulator."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import equity_decide


CONFIG = {
    "preflop_bias": 0.04,
    "postflop_bias": 0.02,
    "opponent_penalty": 0.040,
    "samples": 130,
    "value_threshold": 0.66,
    "semi_threshold": 0.43,
    "semi_prob": 0.28,
    "call_margin": 0.02,
    "value_frac": 0.52,
    "semi_frac": 0.42,
    "raise_threshold": 0.78,
}


def decide(state):
    return equity_decide(state, CONFIG)
