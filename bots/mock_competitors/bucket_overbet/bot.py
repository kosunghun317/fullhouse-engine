"""Mock competitor: overbet/polarized action-bucket policy."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import bucket_decide


CONFIG = {
    "mode": "overbet",
    "premium_frac": 1.15,
    "medium_frac": 0.67,
    "trash_frac": 0.95,
    "medium_bet_prob": 0.45,
    "trash_bluff_prob": 0.24,
    "premium_call": 0.42,
    "medium_call": 0.19,
    "trash_call": 0.05,
}


def decide(state):
    return bucket_decide(state, CONFIG)
