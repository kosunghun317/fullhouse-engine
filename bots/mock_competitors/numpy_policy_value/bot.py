"""Mock competitor: value-heavy trained numpy policy."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import policy_decide


CONFIG = {
    "call_bias": 0.12,
    "raise_bias": 0.08,
    "bet_threshold": 0.58,
    "small_bet_threshold": 0.44,
    "large_frac": 0.92,
    "small_frac": 0.58,
    "raise_threshold": 0.76,
    "raise_frac": 1.05,
}


def decide(state):
    return policy_decide(state, CONFIG)
