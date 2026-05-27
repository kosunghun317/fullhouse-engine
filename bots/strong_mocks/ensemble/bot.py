"""Strong mock: deterministic ensemble of exported strong mock policies."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.strong_mocks.policies import decide_ensemble


DATA_DIRS = {
    "oracle": os.path.join(ROOT, "bots", "strong_mocks", "oracle_imitation", "data"),
    "ppo": os.path.join(ROOT, "bots", "strong_mocks", "ppo_policy", "data"),
    "cfr": os.path.join(ROOT, "bots", "strong_mocks", "cfr_bucket", "data"),
}


def decide(state):
    return decide_ensemble(state, DATA_DIRS)
