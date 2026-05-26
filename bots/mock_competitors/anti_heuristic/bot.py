"""Mock competitor: intentionally targets known heuristic leaks."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import anti_heuristic_decide


CONFIG = {
    "samples": 130,
    "preflop_bias": 0.01,
    "opponent_penalty": 0.050,
}


def decide(state):
    return anti_heuristic_decide(state, CONFIG)
