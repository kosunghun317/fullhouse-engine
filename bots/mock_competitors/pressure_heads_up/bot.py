"""Mock competitor: heads-up pressure policy."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import pressure_heads_up_decide


CONFIG = {
    "samples": 120,
    "preflop_bias": 0.03,
    "opponent_penalty": 0.035,
    "pressure_prob": 0.48,
    "pressure_frac": 0.90,
    "call_margin": 0.00,
}


def decide(state):
    return pressure_heads_up_decide(state, CONFIG)
