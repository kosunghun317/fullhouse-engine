"""Strong mock: heuristic expert arms with a learned contextual selector."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.strong_mocks.arm_selector_policy import decide_arm_selector


DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))


def decide(state):
    return decide_arm_selector(state, DATA_DIR, sample=False)

