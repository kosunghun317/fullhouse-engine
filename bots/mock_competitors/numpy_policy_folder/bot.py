"""Mock competitor: overfolding trained numpy policy."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import policy_decide


CONFIG = {
    "fold_bias": 0.34,
    "call_bias": -0.08,
    "raise_bias": -0.10,
    "bet_threshold": 0.69,
    "small_bet_threshold": 0.52,
    "large_frac": 0.68,
    "small_frac": 0.42,
    "raise_threshold": 0.80,
    "raise_frac": 0.78,
}


def decide(state):
    return policy_decide(state, CONFIG)
