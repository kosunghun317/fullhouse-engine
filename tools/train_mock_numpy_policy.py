"""Train the mock numpy-policy competitors.

This is not part of the submitted heuristic bot. It creates tiny MLP policies
that imitate synthetic poker heuristics over Fullhouse-like states, then saves
numpy weights under bots/mock_competitors/<variant>/data/policy.npz.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "bots" / "mock_competitors" / "numpy_policy" / "data" / "policy.npz"

VARIANT_OUTPUTS = {
    "balanced": DEFAULT_OUTPUT,
    "value": ROOT / "bots" / "mock_competitors" / "numpy_policy_value" / "data" / "policy.npz",
    "bluff": ROOT / "bots" / "mock_competitors" / "numpy_policy_bluff" / "data" / "policy.npz",
    "station": ROOT / "bots" / "mock_competitors" / "numpy_policy_station" / "data" / "policy.npz",
    "folder": ROOT / "bots" / "mock_competitors" / "numpy_policy_folder" / "data" / "policy.npz",
    "pressure": ROOT / "bots" / "mock_competitors" / "numpy_policy_pressure" / "data" / "policy.npz",
}


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


def _oracle_labels(x, variant="balanced"):
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

    profile = {
        "balanced": {
            "call_margin": 0.18,
            "call_strength": 0.58,
            "call_owed": 0.25,
            "raise_strength": 0.72,
            "raise_owed": 0.16,
            "semi_low": 0.45,
            "semi_high": 0.68,
            "semi_active": 0.55,
        },
        "value": {
            "call_margin": 0.12,
            "call_strength": 0.55,
            "call_owed": 0.32,
            "raise_strength": 0.80,
            "raise_owed": 0.18,
            "semi_low": 0.54,
            "semi_high": 0.70,
            "semi_active": 0.45,
        },
        "bluff": {
            "call_margin": 0.24,
            "call_strength": 0.62,
            "call_owed": 0.18,
            "raise_strength": 0.67,
            "raise_owed": 0.18,
            "semi_low": 0.34,
            "semi_high": 0.70,
            "semi_active": 0.70,
        },
        "station": {
            "call_margin": 0.02,
            "call_strength": 0.42,
            "call_owed": 0.55,
            "raise_strength": 0.84,
            "raise_owed": 0.12,
            "semi_low": 0.62,
            "semi_high": 0.76,
            "semi_active": 0.35,
        },
        "folder": {
            "call_margin": 0.31,
            "call_strength": 0.68,
            "call_owed": 0.14,
            "raise_strength": 0.83,
            "raise_owed": 0.10,
            "semi_low": 0.58,
            "semi_high": 0.70,
            "semi_active": 0.38,
        },
        "pressure": {
            "call_margin": 0.16,
            "call_strength": 0.55,
            "call_owed": 0.28,
            "raise_strength": 0.66,
            "raise_owed": 0.22,
            "semi_low": 0.36,
            "semi_high": 0.72,
            "semi_active": 0.78,
        },
    }.get(variant)
    if profile is None:
        raise ValueError(f"unknown variant: {variant}")

    if variant == "value":
        strength += 0.08 * pair + 0.03 * high
        pressure_cost += 0.07 * (1.0 - pair)
    elif variant == "bluff":
        strength += 0.06 * suited + 0.04 * can_check - 0.04 * pair
        pressure_cost -= 0.07 * can_check
    elif variant == "station":
        pressure_cost -= 0.13
        strength += 0.03 * high
    elif variant == "folder":
        pressure_cost += 0.16 + 0.04 * late_street
    elif variant == "pressure":
        strength += 0.08 * can_check + 0.04 * suited - 0.03 * late_street
        pressure_cost -= 0.10 * can_check

    labels = np.zeros(len(x), dtype=int)  # fold/check
    call = (strength > pressure_cost + profile["call_margin"]) | (
        (strength > profile["call_strength"]) & (owed_ratio < profile["call_owed"])
    )
    raise_ = (strength > profile["raise_strength"]) & (
        (can_check > 0.5) | (owed_ratio < profile["raise_owed"])
    )
    semi_bluff = (
        (can_check > 0.5)
        & (strength > profile["semi_low"])
        & (strength < profile["semi_high"])
        & (active < profile["semi_active"])
    )
    labels[call] = 1
    labels[raise_ | semi_bluff] = 2
    return labels


def train(n, seed, output, variant="balanced"):
    rng = np.random.default_rng(seed)
    x = _sample_states(n, rng)
    y = _oracle_labels(x, variant=variant)
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
        variant=np.array([variant]),
    )
    return {
        "variant": variant,
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
    parser.add_argument("--variant", choices=sorted(VARIANT_OUTPUTS), default="balanced")
    parser.add_argument("--all", action="store_true", help="Train every built-in policy variant")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    variants = sorted(VARIANT_OUTPUTS) if args.all else [args.variant]
    for index, variant in enumerate(variants):
        output = Path(args.output) if args.output and not args.all else VARIANT_OUTPUTS[variant]
        result = train(args.samples, args.seed + index * 97, output, variant=variant)
        for key, value in result.items():
            print(f"{key}: {value}")
        if len(variants) > 1:
            print("")


if __name__ == "__main__":
    main()
