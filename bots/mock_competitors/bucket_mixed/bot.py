"""Mock competitor: mixed coarse action-bucket policy."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.mock_competitors.common import bucket_decide


CONFIG = {
    "mode": "mixed",
    "premium_frac": 0.82,
    "small_frac": 0.34,
    "medium_frac": 0.57,
    "trash_frac": 0.74,
    "medium_bet_prob": 0.52,
    "trash_bluff_prob": 0.16,
    "premium_call": 0.36,
    "medium_call": 0.23,
    "trash_call": 0.09,
}


def decide(state):
    return bucket_decide(state, CONFIG)
