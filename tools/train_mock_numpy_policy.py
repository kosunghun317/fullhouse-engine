"""Train the mock numpy-policy competitor.

This is not part of the submitted heuristic bot. It creates a tiny MLP policy
that imitates a synthetic poker heuristic over Fullhouse-like states, then
saves numpy weights under bots/mock_competitors/numpy_policy/data/policy.npz.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "bots" / "mock_competitors" / "numpy_policy" / "data" / "policy.npz"


def _sample_states(n, rng):
    street = rng.integers(0, 4, size=n)
    can_check = rng.random(n) < np.where(street == 0, 0.35, 0.55)
    high = rng.beta(2.4, 1.9, size=n)
    pair = rng.random(n) < 0.065
    suited = rng.random(n) < 0.24
    active = rng.integers(2, 7, size=n)
    pot = np.exp(rng.uniform(np.log(150), np.log(35000), size=n))
    stack = np.exp(rng.uniform(np.log(800), np.log(28000), size=n))
    owed_ratio = np.where(can_check, 0.0, rng.beta(1.2, 4.0, size=n))
    risk = np.minimum(1.0, owed_ratio * pot / np.maximum(1.0, stack))
    late_street = (street >= 2).astype(float)
    x = np.column_stack([
        np.ones(n),
        owed_ratio,
        np.minimum(1.0, pot / 20000),
        high,
        pair.astype(float),
        risk,
        suited.astype(float),
        np.minimum(1.0, active / 6),
        late_street,
        can_check.astype(float),
    ])
    return x


def _oracle_labels(x):
    owed_ratio = x[:, 1]
    pot_size = x[:, 2]
    high = x[:, 3]
    pair = x[:, 4]
    risk = x[:, 5]
    suited = x[:, 6]
    active = x[:, 7]
    late_street = x[:, 8]
    can_check = x[:, 9]

    strength = 0.55 * high + 0.25 * pair + 0.06 * suited - 0.12 * (active > 0.55)
    strength -= 0.10 * late_street * (1.0 - pair)
    pressure_cost = owed_ratio + 0.70 * risk + 0.10 * pot_size

    labels = np.zeros(len(x), dtype=int)  # fold/check
    call = (strength > pressure_cost + 0.18) | ((strength > 0.58) & (owed_ratio < 0.25))
    raise_ = (strength > 0.72) & ((can_check > 0.5) | (owed_ratio < 0.16))
    semi_bluff = (can_check > 0.5) & (strength > 0.45) & (strength < 0.68) & (active < 0.55)
    labels[call] = 1
    labels[raise_ | semi_bluff] = 2
    return labels


def train(n, seed, output):
    rng = np.random.default_rng(seed)
    x = _sample_states(n, rng)
    y = _oracle_labels(x)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale

    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="tanh",
        solver="adam",
        alpha=0.0008,
        batch_size=256,
        learning_rate_init=0.006,
        max_iter=250,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=12,
    )
    clf.fit(z, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        w1=clf.coefs_[0].astype(np.float32),
        b1=clf.intercepts_[0].astype(np.float32),
        w2=clf.coefs_[1].astype(np.float32),
        b2=clf.intercepts_[1].astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        classes=clf.classes_.astype(np.int8),
        train_accuracy=np.array([clf.score(z, y)], dtype=np.float32),
    )
    return {
        "output": str(output),
        "samples": n,
        "seed": seed,
        "iterations": int(clf.n_iter_),
        "train_accuracy": round(float(clf.score(z, y)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train mock numpy-policy competitor")
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = train(args.samples, args.seed, Path(args.output))
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
