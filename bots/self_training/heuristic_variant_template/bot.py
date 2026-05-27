"""Template wrapper for self-training heuristic variants.

Generated variants copy this file and provide data/config.json. The wrapper
sets env overrides before importing the submitted heuristic bot so each variant
can run in its own bot process with independent constants.
"""

import json
import os
import sys


def _find_repo_root(start):
    current = os.path.abspath(start)
    for _ in range(10):
        if (
            os.path.isdir(os.path.join(current, "bots"))
            and os.path.isdir(os.path.join(current, "engine"))
            and os.path.isdir(os.path.join(current, "sandbox"))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.abspath(os.path.join(start, "../../.."))


DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        CONFIG = json.load(handle)
except Exception:
    CONFIG = {}

ROOT = os.environ.get("FULLHOUSE_REPO_ROOT") or CONFIG.get("repo_root") or _find_repo_root(os.path.dirname(__file__))
ROOT = os.path.abspath(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

for key, value in CONFIG.get("env", {}).items():
    os.environ[str(key)] = str(value)

from bots.heuristic.bot import decide as _heuristic_decide


def decide(state):
    return _heuristic_decide(state)
