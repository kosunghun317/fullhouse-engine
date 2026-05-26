"""Mock competitor: half-pot action-bucket policy."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import bucket_decide


CONFIG = {
    "mode": "halfpot",
    "premium_frac": 0.55,
    "medium_frac": 0.50,
    "trash_frac": 0.55,
    "medium_bet_prob": 0.62,
    "trash_bluff_prob": 0.20,
    "premium_call": 0.38,
    "medium_call": 0.25,
    "trash_call": 0.08,
}


def decide(state):
    return bucket_decide(state, CONFIG)
