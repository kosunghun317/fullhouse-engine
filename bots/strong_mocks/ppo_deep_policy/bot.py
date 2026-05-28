"""Strong mock: independent deep PPO-style policy with expanded action arms."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.strong_mocks.deep_policies import decide_deep_model_sampled


DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))


def decide(state):
    return decide_deep_model_sampled(state, DATA_DIR, fallback_style="pressure")
