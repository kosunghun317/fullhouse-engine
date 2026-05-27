"""Train a PPO-style neural policy for strong mock opponents.

This is a lightweight one-step contextual policy-gradient trainer. It is not a
full poker RL solver, but it gives us a runnable neural benchmark that learns
from synthetic poker-state contexts and oracle reward logits without requiring
PyTorch or runtime-heavy dependencies.
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
from tools.strong_mocks.dataset import oracle_labels, oracle_logits, sample_feature_matrix
from tools.strong_mocks.features import FEATURE_NAMES
from tools.strong_mocks.train_imitation import train as train_imitation


DEFAULT_OUTPUT = ROOT / "bots" / "strong_mocks" / "ppo_policy" / "data" / "policy.npz"


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64).copy()
    values[~mask] = -30.0
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    exp[~mask] = 0.0
    return exp / np.maximum(1e-9, exp.sum(axis=1, keepdims=True))


def _legal_mask_from_features(x: np.ndarray) -> np.ndarray:
    mask = np.ones((x.shape[0], len(ACTION_LABELS)), dtype=bool)
    can_check = x[:, 9] > 0.5
    mask[can_check, 0] = False
    return mask


def _sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    draws = rng.random(probs.shape[0])
    cumulative = np.cumsum(probs, axis=1)
    return (cumulative < draws[:, None]).sum(axis=1)


def _forward(z: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(z @ params["w1"] + params["b1"])
    logits = hidden @ params["w2"] + params["b2"]
    return hidden, logits


def _export(
    output: Path,
    params: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    report: dict,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_type=np.asarray(["ppo_mlp"]),
        w1=params["w1"].astype(np.float32),
        b1=params["b1"].astype(np.float32),
        w2=params["w2"].astype(np.float32),
        b2=params["b2"].astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        classes=np.arange(len(ACTION_LABELS), dtype=np.int8),
        feature_names=np.asarray(FEATURE_NAMES),
        action_labels=np.asarray(ACTION_LABELS),
        train_accuracy=np.asarray([report["oracle_agreement"]], dtype=np.float32),
        average_reward=np.asarray([report["average_reward"]], dtype=np.float32),
    )


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx.nn  # noqa: F401
        import mlx.optimizers  # noqa: F401
        return True
    except Exception:
        return False


def _require_mlx():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    return mx, nn, optim


def train_policy_gradient(
    output: Path,
    seed: int,
    iterations: int,
    batch_size: int,
    hidden: int,
    learning_rate: float,
    clip_ratio: float,
    entropy_coef: float,
) -> dict:
    rng = np.random.default_rng(seed)
    warmup = sample_feature_matrix(max(4096, batch_size), seed + 17)
    mean = warmup.mean(axis=0)
    scale = warmup.std(axis=0)
    scale[scale < 1e-6] = 1.0

    input_dim = len(FEATURE_NAMES)
    output_dim = len(ACTION_LABELS)
    params = {
        "w1": rng.normal(0.0, 0.10, size=(input_dim, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0.0, 0.08, size=(hidden, output_dim)),
        "b2": np.zeros(output_dim),
    }

    last_reward = 0.0
    for step in range(iterations):
        x = sample_feature_matrix(batch_size, int(rng.integers(0, 2**31 - 1)))
        z = (x - mean) / scale
        mask = _legal_mask_from_features(x)
        rewards = oracle_logits(x, style=["balanced", "pressure", "bluff", "value"][step % 4])

        hidden_values, logits = _forward(z, params)
        old_probs = _masked_softmax(logits, mask)
        actions = _sample_actions(old_probs, rng)
        old_selected = np.maximum(1e-8, old_probs[np.arange(batch_size), actions])
        selected_reward = rewards[np.arange(batch_size), actions]
        baseline = (old_probs * rewards).sum(axis=1)
        advantage = selected_reward - baseline
        advantage = (advantage - advantage.mean()) / max(1e-6, advantage.std())

        for _epoch in range(3):
            hidden_values, logits = _forward(z, params)
            probs = _masked_softmax(logits, mask)
            selected = np.maximum(1e-8, probs[np.arange(batch_size), actions])
            ratio = selected / old_selected
            active = np.ones(batch_size, dtype=bool)
            active[(advantage > 0) & (ratio > 1.0 + clip_ratio)] = False
            active[(advantage < 0) & (ratio < 1.0 - clip_ratio)] = False

            coeff = np.zeros(batch_size)
            coeff[active] = advantage[active] * ratio[active]
            grad_logits = probs * coeff[:, None]
            grad_logits[np.arange(batch_size), actions] -= coeff
            grad_logits /= batch_size

            if entropy_coef:
                entropy_grad = probs * (np.log(np.maximum(1e-8, probs)) + 1.0)
                entropy_grad[~mask] = 0.0
                grad_logits += entropy_coef * entropy_grad / batch_size

            grad_w2 = hidden_values.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ params["w2"].T
            grad_pre = grad_hidden * (1.0 - hidden_values * hidden_values)
            grad_w1 = z.T @ grad_pre
            grad_b1 = grad_pre.sum(axis=0)

            params["w2"] -= learning_rate * grad_w2
            params["b2"] -= learning_rate * grad_b2
            params["w1"] -= learning_rate * grad_w1
            params["b1"] -= learning_rate * grad_b1

        last_reward = float(selected_reward.mean())

    eval_x = sample_feature_matrix(12000, seed + 991)
    eval_z = (eval_x - mean) / scale
    eval_mask = _legal_mask_from_features(eval_x)
    _hidden, eval_logits = _forward(eval_z, params)
    eval_probs = _masked_softmax(eval_logits, eval_mask)
    pred = np.argmax(eval_probs, axis=1)
    labels = oracle_labels(eval_x, style="pressure")
    oracle_reward = oracle_logits(eval_x, style="pressure")
    report = {
        "mode": "policy_gradient",
        "output": str(output),
        "seed": seed,
        "iterations": iterations,
        "batch_size": batch_size,
        "hidden": hidden,
        "oracle_agreement": round(float(np.mean(pred == labels)), 4),
        "average_reward": round(float(np.mean(oracle_reward[np.arange(eval_x.shape[0]), pred])), 4),
        "last_batch_reward": round(last_reward, 4),
    }
    _export(output, params, mean, scale, report)
    return report


def train_mlx_policy(
    output: Path,
    seed: int,
    iterations: int,
    batch_size: int,
    hidden: int,
    learning_rate: float,
    entropy_coef: float,
) -> dict:
    mx, nn, optim = _require_mlx()
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)

    warmup = sample_feature_matrix(max(4096, batch_size), seed + 17)
    mean = warmup.mean(axis=0).astype(np.float32)
    scale = warmup.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    class PolicyMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, output_dim)

        def __call__(self, x):
            return self.l2(mx.tanh(self.l1(x)))

    model = PolicyMLP(len(FEATURE_NAMES), hidden, len(ACTION_LABELS))
    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(model, z, rewards, mask):
        logits = model(z)
        logits = mx.where(mask, logits, mx.array(-30.0, dtype=logits.dtype))
        probs = mx.softmax(logits, axis=1)
        expected_reward = mx.sum(probs * rewards, axis=1)
        entropy = -mx.sum(probs * mx.log(mx.maximum(probs, 1e-8)), axis=1)
        return -mx.mean(expected_reward + entropy_coef * entropy)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    last_loss = 0.0
    for step in range(iterations):
        x = sample_feature_matrix(batch_size, int(rng.integers(0, 2**31 - 1)))
        z = ((x - mean) / scale).astype(np.float32)
        rewards = oracle_logits(x, style=["balanced", "pressure", "bluff", "value"][step % 4]).astype(np.float32)
        mask = _legal_mask_from_features(x)
        z_mx = mx.array(z)
        rewards_mx = mx.array(rewards)
        mask_mx = mx.array(mask)
        loss, grads = loss_and_grad(model, z_mx, rewards_mx, mask_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        last_loss = float(loss)

    eval_x = sample_feature_matrix(12000, seed + 991)
    eval_z = mx.array(((eval_x - mean) / scale).astype(np.float32))
    eval_mask = mx.array(_legal_mask_from_features(eval_x))
    logits = model(eval_z)
    logits = mx.where(eval_mask, logits, mx.array(-30.0, dtype=logits.dtype))
    probs = mx.softmax(logits, axis=1)
    pred = np.array(mx.argmax(probs, axis=1)).astype(int)
    labels = oracle_labels(eval_x, style="pressure")
    oracle_reward = oracle_logits(eval_x, style="pressure")

    mx.eval(model.parameters())
    params = {
        "w1": np.array(model.l1.weight).astype(np.float32).T,
        "b1": np.array(model.l1.bias).astype(np.float32),
        "w2": np.array(model.l2.weight).astype(np.float32).T,
        "b2": np.array(model.l2.bias).astype(np.float32),
    }
    report = {
        "mode": "mlx_expected_reward",
        "backend": "mlx",
        "device": str(mx.default_device()),
        "output": str(output),
        "seed": seed,
        "iterations": iterations,
        "batch_size": batch_size,
        "hidden": hidden,
        "oracle_agreement": round(float(np.mean(pred == labels)), 4),
        "average_reward": round(float(np.mean(oracle_reward[np.arange(eval_x.shape[0]), pred])), 4),
        "last_loss": round(last_loss, 4),
    }
    _export(output, params, mean, scale, report)
    return report


def init_from_imitation(samples: int, seed: int, output: Path) -> dict:
    result = train_imitation(samples=samples, seed=seed, style="pressure", output=output, hidden=64)
    result["mode"] = "init_from_imitation"
    return result


def main():
    parser = argparse.ArgumentParser(description="Train PPO-style strong mock policy")
    parser.add_argument("--init-from-imitation", action="store_true")
    parser.add_argument("--samples", type=int, default=250000)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.018)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--backend", choices=["auto", "numpy", "mlx"], default="auto")
    parser.add_argument("--seed", type=int, default=6161)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.init_from_imitation:
        result = init_from_imitation(args.samples, args.seed, Path(args.output))
    elif args.backend == "mlx" or (args.backend == "auto" and _mlx_available()):
        result = train_mlx_policy(
            output=Path(args.output),
            seed=args.seed,
            iterations=args.iterations,
            batch_size=args.batch_size,
            hidden=args.hidden,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
        )
    else:
        result = train_policy_gradient(
            output=Path(args.output),
            seed=args.seed,
            iterations=args.iterations,
            batch_size=args.batch_size,
            hidden=args.hidden,
            learning_rate=args.learning_rate,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coef,
        )
    print(json.dumps(result, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
