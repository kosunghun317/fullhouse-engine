"""Train strong mock imitation policies from synthetic oracle labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.strong_mocks.actions import ACTION_LABELS
from tools.strong_mocks.dataset import oracle_labels, sample_feature_matrix
from tools.strong_mocks.features import FEATURE_NAMES


DEFAULT_OUTPUTS = {
    "balanced": ROOT / "bots" / "strong_mocks" / "oracle_imitation" / "data" / "policy.npz",
    "value": ROOT / "bots" / "strong_mocks" / "oracle_imitation" / "data" / "policy_value.npz",
    "pressure": ROOT / "bots" / "strong_mocks" / "ppo_policy" / "data" / "policy.npz",
}


def train(samples: int, seed: int, style: str, output: Path, hidden: int = 48) -> dict:
    x = sample_feature_matrix(samples, seed)
    y = oracle_labels(x, style)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale

    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="tanh",
        solver="adam",
        alpha=0.0005,
        batch_size=512,
        learning_rate_init=0.004,
        max_iter=220,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=12,
    )
    clf.fit(z, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_type=np.asarray(["mlp"]),
        w1=clf.coefs_[0].astype(np.float32),
        b1=clf.intercepts_[0].astype(np.float32),
        w2=clf.coefs_[1].astype(np.float32),
        b2=clf.intercepts_[1].astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        classes=clf.classes_.astype(np.int8),
        feature_names=np.asarray(FEATURE_NAMES),
        action_labels=np.asarray(ACTION_LABELS),
        style=np.asarray([style]),
        train_accuracy=np.asarray([clf.score(z, y)], dtype=np.float32),
    )
    counts = np.bincount(y, minlength=len(ACTION_LABELS))
    return {
        "output": str(output),
        "samples": samples,
        "seed": seed,
        "style": style,
        "iterations": int(clf.n_iter_),
        "train_accuracy": round(float(clf.score(z, y)), 4),
        "label_counts": {ACTION_LABELS[i]: int(counts[i]) for i in range(len(ACTION_LABELS))},
    }


def main():
    parser = argparse.ArgumentParser(description="Train strong mock imitation policy")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--style", choices=["balanced", "value", "bluff", "station", "folder", "pressure"], default="balanced")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--output", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output = Path(args.output) if args.output else DEFAULT_OUTPUTS.get(
        args.style,
        ROOT / "bots" / "strong_mocks" / "oracle_imitation" / "data" / f"policy_{args.style}.npz",
    )
    result = train(args.samples, args.seed, args.style, output, hidden=args.hidden)
    print(json.dumps(result, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
