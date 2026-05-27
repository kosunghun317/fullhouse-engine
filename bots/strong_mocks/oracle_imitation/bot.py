"""Strong mock: exported oracle-imitation policy with oracle fallback."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.strong_mocks.policies import decide_model


DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))


def decide(state):
    return decide_model(state, DATA_DIR, fallback_style="value")
