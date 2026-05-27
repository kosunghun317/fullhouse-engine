"""Strong mock: heavier eval7 rollout/search opponent."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.strong_mocks.policies import decide_rollout


def decide(state):
    return decide_rollout(state, style="rollout_pressure")
