"""Build optional read-only data tables for bots/heuristic.

The submitted bot has hard-coded fallbacks. These tables demonstrate how to use
Fullhouse's data allowance safely with numpy-only import-time loading.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bots.heuristic.bot import PREFLOP_SCORE_TABLE


DEFAULT_OUTPUT = ROOT / "bots" / "heuristic" / "data" / "tables.npz"


def build(output):
    classes = np.array(sorted(PREFLOP_SCORE_TABLE), dtype="U3")
    scores = np.array([PREFLOP_SCORE_TABLE[str(cls)] for cls in classes], dtype=np.uint8)
    bet_size_arms = np.array([0.34, 0.42, 0.49, 0.55, 0.67, 0.75, 0.90, 1.10], dtype=np.float32)
    cluster_prior_bias = np.zeros((64, 8), dtype=np.float16)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        preflop_classes=classes,
        preflop_scores=scores,
        bet_size_arms=bet_size_arms,
        cluster_prior_bias=cluster_prior_bias,
        version=np.array(["2026-05-27"], dtype="U10"),
    )
    return {
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "preflop_classes": int(len(classes)),
        "bet_size_arms": int(len(bet_size_arms)),
        "cluster_prior_shape": list(cluster_prior_bias.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Build heuristic optional data tables")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build(Path(args.output))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
