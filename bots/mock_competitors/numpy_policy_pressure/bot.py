"""Mock competitor: pressure-oriented trained numpy policy."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import policy_decide


CONFIG = {
    "fold_bias": -0.10,
    "call_bias": 0.04,
    "raise_bias": 0.18,
    "bet_threshold": 0.53,
    "small_bet_threshold": 0.34,
    "large_frac": 0.95,
    "small_frac": 0.56,
    "raise_threshold": 0.70,
    "raise_frac": 1.10,
}


def decide(state):
    return policy_decide(state, CONFIG)
