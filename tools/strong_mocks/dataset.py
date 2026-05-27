"""Synthetic state generation and oracle labels for strong mock training."""

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
from tools.strong_mocks.features import FEATURE_NAMES


def sample_feature_matrix(n: int, seed: int = 1337) -> np.ndarray:
    rng = np.random.default_rng(seed)
    street = rng.integers(0, 4, size=n)
    can_check = rng.random(n) < np.where(street == 0, 0.35, 0.55)
    owed_ratio = np.where(can_check, 0.0, rng.beta(1.2, 4.0, size=n))
    pot_norm = rng.beta(1.4, 4.0, size=n)
    stack_norm = rng.beta(2.2, 2.6, size=n)
    spr_norm = rng.beta(1.6, 3.0, size=n)
    active = rng.integers(2, 7, size=n) / 6.0
    position = rng.random(n)
    high = rng.beta(2.4, 1.8, size=n)
    low = np.minimum(high, rng.beta(1.8, 2.6, size=n))
    pair = (rng.random(n) < 0.07).astype(float)
    suited = (rng.random(n) < 0.24).astype(float)
    gap = rng.beta(1.4, 3.0, size=n)
    strength = np.clip(0.42 * high + 0.20 * low + 0.26 * pair + 0.06 * suited - 0.08 * gap, 0.02, 0.98)
    wet = (rng.random(n) < np.where(street >= 1, 0.42, 0.0)).astype(float)
    paired = (rng.random(n) < np.where(street >= 1, 0.18, 0.0)).astype(float)
    monotone = (rng.random(n) < np.where(street >= 1, 0.12, 0.0)).astype(float)
    broadway = (rng.random(n) < np.where(street >= 1, 0.52, 0.0)).astype(float)
    made_pairish = np.clip(pair + (rng.random(n) < (0.16 + 0.38 * street / 3)).astype(float), 0, 1)
    flush_draw = ((street >= 1) & (street <= 2) & (rng.random(n) < 0.18)).astype(float)
    straight_draw = ((street >= 1) & (street <= 2) & (rng.random(n) < 0.18)).astype(float)
    last_action = rng.integers(0, 4, size=n)
    raises = rng.integers(0, 4, size=n) / 4.0
    allins = rng.integers(0, 3, size=n) / 3.0
    stack_share = rng.beta(2.0, 5.0, size=n)

    x = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = 1.0
    for index in range(4):
        x[:, 1 + index] = street == index
    x[:, 5] = owed_ratio
    x[:, 6] = pot_norm
    x[:, 7] = stack_norm
    x[:, 8] = spr_norm
    x[:, 9] = can_check.astype(float)
    x[:, 10] = active
    x[:, 11] = position
    x[:, 12] = high
    x[:, 13] = low
    x[:, 14] = pair
    x[:, 15] = suited
    x[:, 16] = gap
    x[:, 17] = strength
    x[:, 18] = wet
    x[:, 19] = paired
    x[:, 20] = monotone
    x[:, 21] = broadway
    x[:, 22] = made_pairish
    x[:, 23] = flush_draw
    x[:, 24] = straight_draw
    for offset in range(4):
        x[:, 25 + offset] = last_action == offset
    x[:, 29] = raises
    x[:, 30] = allins
    x[:, 31] = stack_share
    return x


def oracle_logits(x: np.ndarray, style: str = "balanced") -> np.ndarray:
    owed = x[:, 5]
    can_check = x[:, 9]
    active = x[:, 10]
    pos = x[:, 11]
    strength = x[:, 17]
    wet = x[:, 18]
    made = x[:, 22]
    draw = np.maximum(x[:, 23], x[:, 24])
    raises = x[:, 29]
    allins = x[:, 30]
    risk_cost = owed + 0.22 * raises + 0.35 * allins + 0.14 * active
    pressure = 0.16 * pos + 0.10 * (1.0 - wet) + 0.10 * draw - 0.06 * active
    value = 0.72 * strength + 0.22 * made + 0.06 * pos - 0.12 * wet

    if style == "value":
        value += 0.12 * made
        pressure -= 0.05
    elif style == "bluff":
        pressure += 0.18 * can_check + 0.08 * pos
        value -= 0.04 * made
    elif style == "station":
        risk_cost -= 0.16
        pressure -= 0.12
    elif style == "folder":
        risk_cost += 0.18 + 0.08 * allins
        pressure -= 0.08
    elif style == "pressure":
        pressure += 0.15 + 0.08 * can_check
        risk_cost -= 0.04

    logits = np.zeros((len(x), len(ACTION_LABELS)), dtype=np.float32)
    logits[:, 0] = 0.45 + risk_cost - value
    logits[:, 1] = 0.20 + value - 0.55 * risk_cost
    bet_drive = np.maximum(0.0, value + pressure - 0.38)
    logits[:, 1] -= 0.30 * can_check * bet_drive
    logits[:, 2] = value + 1.05 * pressure - 0.22 - 0.30 * owed
    logits[:, 3] = value + 0.95 * pressure - 0.18 - 0.35 * owed
    logits[:, 4] = value + 0.75 * pressure + 0.08 * made - 0.22 - 0.45 * owed
    logits[:, 5] = value + 0.50 * pressure + 0.12 * made - 0.30 - 0.52 * owed
    logits[:, 6] = value + 0.35 * pressure + 0.18 * made - 0.44 - 0.65 * owed
    logits[:, 7] = value + 0.55 * made + 0.45 * allins - 0.75 - 0.75 * owed

    # Can-check states should prefer check over fold unless applying pressure.
    logits[:, 0] -= 1.4 * can_check
    logits[:, 1] += 0.30 * can_check
    logits[:, 2:7] += (0.22 * can_check)[:, None]
    return logits


def oracle_labels(x: np.ndarray, style: str = "balanced") -> np.ndarray:
    return np.argmax(oracle_logits(x, style), axis=1).astype(np.int8)


def save_dataset(path: Path, samples: int, seed: int, style: str) -> dict:
    x = sample_feature_matrix(samples, seed)
    y = oracle_labels(x, style)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x.astype(np.float32),
        y=y.astype(np.int8),
        feature_names=np.asarray(FEATURE_NAMES),
        action_labels=np.asarray(ACTION_LABELS),
        style=np.asarray([style]),
        seed=np.asarray([seed], dtype=np.int64),
    )
    counts = np.bincount(y, minlength=len(ACTION_LABELS))
    return {
        "output": str(path),
        "samples": samples,
        "seed": seed,
        "style": style,
        "label_counts": {ACTION_LABELS[i]: int(counts[i]) for i in range(len(ACTION_LABELS))},
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic strong-mock training data")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--style", default="balanced")
    parser.add_argument("--output", default="data/strong_mocks/synthetic_oracle.npz")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = save_dataset(Path(args.output), args.samples, args.seed, args.style)
    print(json.dumps(result, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
