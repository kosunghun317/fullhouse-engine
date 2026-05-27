"""Strong mock: coarse CFR-like bucket table policy."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.strong_mocks.policies import decide_cfr


DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))


def decide(state):
    return decide_cfr(state, DATA_DIR, fallback_style="bluff")
