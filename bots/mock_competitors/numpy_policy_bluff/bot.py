"""Mock competitor: bluff-heavy trained numpy policy."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import policy_decide


CONFIG = {
    "fold_bias": -0.05,
    "raise_bias": 0.22,
    "bet_threshold": 0.50,
    "small_bet_threshold": 0.31,
    "large_frac": 0.72,
    "small_frac": 0.47,
    "raise_threshold": 0.67,
    "raise_frac": 0.85,
}


def decide(state):
    return policy_decide(state, CONFIG)
