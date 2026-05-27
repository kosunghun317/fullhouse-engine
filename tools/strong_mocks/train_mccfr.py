"""Train a coarse CFR-like bucket policy table.

This is a lightweight abstraction trainer for benchmark opponents. It is not a
full poker solver. It creates regret-matched action tables over the same
feature buckets used by the strong mock wrappers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.strong_mocks.actions import ACTION_LABELS
from tools.strong_mocks.dataset import oracle_logits, sample_feature_matrix


DEFAULT_OUTPUT = ROOT / "bots" / "strong_mocks" / "cfr_bucket" / "data" / "policy.npz"


def _bucket_ids_from_features(x: np.ndarray, bucket_count: int) -> np.ndarray:
    street = np.argmax(x[:, 1:5], axis=1)
    strength = np.minimum(7, np.maximum(0, (x[:, 17] * 8).astype(int)))
    wet = (x[:, 18] > 0.5).astype(int)
    paired = (x[:, 19] > 0.5).astype(int)
    owed = np.minimum(3, np.maximum(0, (x[:, 5] * 4).astype(int)))
    active = np.minimum(5, np.maximum(0, (x[:, 10] * 6).astype(int)))
    position = np.minimum(3, np.maximum(0, (x[:, 11] * 4).astype(int)))
    raises = np.minimum(3, np.maximum(0, (x[:, 29] * 4).astype(int)))
    raw = (((((((street * 8 + strength) * 2 + wet) * 2 + paired) * 4 + owed) * 6 + active) * 4 + position) * 4 + raises)
    return (raw % bucket_count).astype(int)


def train(iterations: int, batch_size: int, bucket_count: int, seed: int, output: Path) -> dict:
    rng = np.random.default_rng(seed)
    regrets = np.zeros((bucket_count, len(ACTION_LABELS)), dtype=np.float64)
    strategy_sum = np.zeros_like(regrets)
    styles = ["balanced", "value", "bluff", "station", "folder", "pressure"]

    for step in range(iterations):
        x = sample_feature_matrix(batch_size, int(rng.integers(0, 2**31 - 1)))
        style = styles[step % len(styles)]
        logits = oracle_logits(x, style=style).astype(np.float64)
        buckets = _bucket_ids_from_features(x, bucket_count)

        positive = np.maximum(regrets[buckets], 0.0)
        denom = positive.sum(axis=1, keepdims=True)
        strategy = np.full_like(positive, 1.0 / len(ACTION_LABELS))
        np.divide(positive, denom, out=strategy, where=denom > 1e-9)
        values = logits - logits.mean(axis=1, keepdims=True)
        chosen_value = (strategy * values).sum(axis=1, keepdims=True)
        instant_regret = values - chosen_value

        np.add.at(regrets, buckets, instant_regret)
        np.add.at(strategy_sum, buckets, strategy)

    positive = np.maximum(strategy_sum, 0.0)
    denom = positive.sum(axis=1, keepdims=True)
    policy = np.full_like(positive, 1.0 / len(ACTION_LABELS))
    np.divide(positive, denom, out=policy, where=denom > 1e-9)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_type=np.asarray(["cfr_table"]),
        policy_table=policy.astype(np.float32),
        regrets=regrets.astype(np.float32),
        action_labels=np.asarray(ACTION_LABELS),
        bucket_count=np.asarray([bucket_count], dtype=np.int32),
        iterations=np.asarray([iterations], dtype=np.int32),
        batch_size=np.asarray([batch_size], dtype=np.int32),
        seed=np.asarray([seed], dtype=np.int64),
    )
    return {
        "output": str(output),
        "iterations": iterations,
        "batch_size": batch_size,
        "bucket_count": bucket_count,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Train coarse CFR-like strong mock policy")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bucket-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=5151)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = train(args.iterations, args.batch_size, args.bucket_count, args.seed, Path(args.output))
    print(json.dumps(result, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
