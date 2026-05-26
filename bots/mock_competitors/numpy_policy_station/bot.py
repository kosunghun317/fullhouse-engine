"""Mock competitor: call-heavy trained numpy policy."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import policy_decide


CONFIG = {
    "fold_bias": -0.35,
    "call_bias": 0.32,
    "raise_bias": -0.12,
    "bet_threshold": 0.72,
    "small_bet_threshold": 0.55,
    "large_frac": 0.62,
    "small_frac": 0.38,
    "raise_threshold": 0.84,
    "raise_frac": 0.72,
}


def decide(state):
    return policy_decide(state, CONFIG)
