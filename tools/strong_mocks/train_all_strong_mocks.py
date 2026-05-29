"""Train, tune, and gate every benchmark-only strong mock opponent.

The serious defaults are intentionally large enough to produce meaningful
training signals. Use --allow-smoke only for wiring checks; smoke artifacts are
written under the run directory instead of replacing canonical bot artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.check_strong_mocks as strong_gate  # noqa: E402
from tools.strong_mocks import train_arm_selector  # noqa: E402
from tools.strong_mocks import train_deep_ppo  # noqa: E402
from tools.strong_mocks import train_imitation  # noqa: E402
from tools.strong_mocks import train_real_policy  # noqa: E402
from tools.strong_mocks import tune_arm_selector_params  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "fullhouse_strong_mocks"
DEFAULT_MIN_ORACLE_SAMPLES = 200_000
DEFAULT_MIN_TRAINING_HANDS = 1_000_000

STRONG_MOCK_NAMES = (
    "oracle_imitation",
    "ppo_policy",
    "cfr_bucket",
    "ppo_deep_policy",
    "heuristic_rl_selector",
    "rollout_search",
    "ensemble",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _total_hands(generations: int, matches_per_generation: int, hands: int) -> int:
    return int(generations) * int(matches_per_generation) * int(hands)


def _gate_tasks_per_candidate(seed_count: int, baseline_count: int, modes: list[str]) -> int:
    per_seed = 0
    if "sixmax" in modes:
        per_seed += 1
    if "heads-up" in modes:
        per_seed += baseline_count
    return int(seed_count) * per_seed


def validate_orchestration_budget(args) -> None:
    if bool(getattr(args, "allow_smoke", False)):
        return
    if int(args.oracle_samples) < int(args.min_oracle_samples):
        raise SystemExit(
            f"oracle imitation uses {args.oracle_samples} samples; require >= {args.min_oracle_samples}. "
            "Use --allow-smoke only for wiring checks."
        )

    training_budgets = {
        "ppo_policy": (args.ppo_generations, args.ppo_matches_per_generation, args.ppo_hands),
        "cfr_bucket": (args.bucket_generations, args.bucket_matches_per_generation, args.bucket_hands),
        "ppo_deep_policy": (args.deep_generations, args.deep_matches_per_generation, args.deep_hands),
        "heuristic_rl_selector": (args.arm_generations, args.arm_matches_per_generation, args.arm_hands),
    }
    for name, (generations, matches_per_generation, hands) in training_budgets.items():
        total_hands = _total_hands(generations, matches_per_generation, hands)
        if int(hands) < 400:
            raise SystemExit(f"{name} uses {hands} hands/match; require >= 400 unless --allow-smoke is set")
        if total_hands < int(args.min_training_hands):
            raise SystemExit(
                f"{name} training budget {total_hands} hands is below --min-training-hands "
                f"{args.min_training_hands}; use --allow-smoke only for wiring checks."
            )

    final_stage = tune_arm_selector_params._parse_stages(args.arm_tune_stages)[-1]
    final_stage_hands = final_stage.matches * final_stage.hands
    if final_stage_hands < int(args.arm_tune_min_hands_per_candidate):
        raise SystemExit(
            "arm-selector parameter tuning final stage has "
            f"{final_stage_hands} hands/candidate; require >= {args.arm_tune_min_hands_per_candidate}."
        )

    gate_modes = args.gate_mode or ["sixmax", "heads-up"]
    gate_baselines = args.gate_baseline or sorted(strong_gate.BASELINES)
    tasks_per_candidate = _gate_tasks_per_candidate(args.gate_seed_count, len(gate_baselines), gate_modes)
    if int(args.gate_hands) < 400:
        raise SystemExit("--gate-hands must be at least 400 unless --allow-smoke is set")
    if tasks_per_candidate < int(args.gate_min_tasks_per_candidate):
        raise SystemExit(
            f"strong-mock gate has {tasks_per_candidate} tasks/candidate; "
            f"require >= {args.gate_min_tasks_per_candidate}."
        )


def _artifact_root(args, run_dir: Path) -> Path | None:
    if args.artifact_root:
        return Path(args.artifact_root)
    if args.allow_smoke:
        return run_dir / "artifacts"
    return None


def _model_artifact_paths(artifact_root: Path | None) -> dict[str, Path]:
    if artifact_root is None:
        return {
            "oracle_imitation": train_imitation.DEFAULT_OUTPUTS["balanced"],
            "ppo_policy": train_real_policy.DEFAULT_PPO_OUTPUT,
            "cfr_bucket": train_real_policy.DEFAULT_BUCKET_OUTPUT,
            "ppo_deep_policy": train_deep_ppo.DEFAULT_OUTPUT,
            "heuristic_rl_selector": train_arm_selector.DEFAULT_OUTPUT,
            "heuristic_rl_selector_params": ROOT / "bots" / "strong_mocks" / "heuristic_rl_selector" / "data" / "params.npz",
        }
    return {
        "oracle_imitation": artifact_root / "oracle_imitation" / "data" / "policy.npz",
        "ppo_policy": artifact_root / "ppo_policy" / "data" / "policy.npz",
        "cfr_bucket": artifact_root / "cfr_bucket" / "data" / "policy.npz",
        "ppo_deep_policy": artifact_root / "ppo_deep_policy" / "data" / "policy.npz",
        "heuristic_rl_selector": artifact_root / "heuristic_rl_selector" / "data" / "policy.npz",
        "heuristic_rl_selector_params": artifact_root / "heuristic_rl_selector" / "data" / "params.npz",
    }


def _write_bot(path: Path, body: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "bot.py").write_text(body, encoding="utf-8")


def _generated_header() -> str:
    return (
        "import os\n"
        "import sys\n\n"
        f"ROOT = {str(ROOT)!r}\n"
        "if ROOT not in sys.path:\n"
        "    sys.path.insert(0, ROOT)\n\n"
    )


def _prepare_run_local_bots(artifact_root: Path) -> dict[str, str]:
    header = _generated_header()
    _write_bot(
        artifact_root / "oracle_imitation",
        header
        + "from tools.strong_mocks.policies import decide_model\n\n"
        + "DATA_DIR = os.environ.get('BOT_DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))\n\n"
        + "def decide(state):\n"
        + "    return decide_model(state, DATA_DIR, fallback_style='value')\n",
    )
    _write_bot(
        artifact_root / "ppo_policy",
        header
        + "from tools.strong_mocks.policies import decide_model_sampled\n\n"
        + "DATA_DIR = os.environ.get('BOT_DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))\n\n"
        + "def decide(state):\n"
        + "    return decide_model_sampled(state, DATA_DIR, fallback_style='pressure')\n",
    )
    _write_bot(
        artifact_root / "cfr_bucket",
        header
        + "from tools.strong_mocks.policies import decide_cfr\n\n"
        + "DATA_DIR = os.environ.get('BOT_DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))\n\n"
        + "def decide(state):\n"
        + "    return decide_cfr(state, DATA_DIR, fallback_style='bluff')\n",
    )
    _write_bot(
        artifact_root / "ppo_deep_policy",
        header
        + "from tools.strong_mocks.deep_policies import decide_deep_ppo\n\n"
        + "DATA_DIR = os.environ.get('BOT_DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))\n\n"
        + "def decide(state):\n"
        + "    return decide_deep_ppo(state, DATA_DIR, sample=False)\n",
    )
    _write_bot(
        artifact_root / "heuristic_rl_selector",
        header
        + "from tools.strong_mocks.arm_selector_policy import decide_arm_selector\n\n"
        + "DATA_DIR = os.environ.get('BOT_DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))\n\n"
        + "def decide(state):\n"
        + "    return decide_arm_selector(state, DATA_DIR, sample=False)\n",
    )
    _write_bot(
        artifact_root / "rollout_search",
        header
        + "from tools.strong_mocks.policies import decide_rollout\n\n"
        + "def decide(state):\n"
        + "    return decide_rollout(state, style='rollout_pressure')\n",
    )
    _write_bot(
        artifact_root / "ensemble",
        header
        + "from tools.strong_mocks.policies import decide_ensemble\n\n"
        + "BASE_DIR = os.path.dirname(os.path.dirname(__file__))\n"
        + "DATA_DIRS = {\n"
        + "    'oracle': os.path.join(BASE_DIR, 'oracle_imitation', 'data'),\n"
        + "    'ppo': os.path.join(BASE_DIR, 'ppo_policy', 'data'),\n"
        + "    'cfr': os.path.join(BASE_DIR, 'cfr_bucket', 'data'),\n"
        + "}\n\n"
        + "def decide(state):\n"
        + "    return decide_ensemble(state, DATA_DIRS)\n",
    )
    return {name: str(artifact_root / name) for name in STRONG_MOCK_NAMES}


def _candidate_paths(artifact_root: Path | None) -> dict[str, str] | None:
    if artifact_root is None:
        return None
    return _prepare_run_local_bots(artifact_root)


def _stage_record(stage: str, report: dict) -> dict:
    return {
        "stage": stage,
        "report_path": report.get("report_path"),
        "summary": {
            key: value
            for key, value in report.items()
            if key
            in {
                "output",
                "samples",
                "kind",
                "generations",
                "matches_per_generation",
                "hands",
                "best_mean_train_delta",
                "final_mean_train_delta",
                "train_accuracy",
                "best_params",
                "passed",
                "tasks",
            }
        },
    }


def _real_policy_args(args, kind: str, output: Path, log_jsonl: Path):
    if kind == "ppo":
        return SimpleNamespace(
            kind="ppo",
            generations=args.ppo_generations,
            matches_per_generation=args.ppo_matches_per_generation,
            hands=args.ppo_hands,
            players=6,
            train_seats=2,
            hidden=args.ppo_hidden,
            bucket_count=32768,
            abstraction="feature",
            bucket_update="policy-gradient",
            learning_rate=args.ppo_learning_rate,
            entropy_coef=args.ppo_entropy_coef,
            ppo_clip_ratio=args.ppo_clip_ratio,
            ppo_value_coef=args.ppo_value_coef,
            max_grad_norm=args.ppo_max_grad_norm,
            epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            feature_norm_momentum=args.ppo_feature_norm_momentum,
            replay_generations=args.ppo_replay_generations,
            replay_max_decisions=args.ppo_replay_max_decisions,
            temperature=args.ppo_temperature,
            reward_scale=1000.0,
            reward_clip=args.ppo_reward_clip,
            opponent_pool=args.opponent_pool,
            extra_bot_path=[],
            snapshot_interval=2,
            max_snapshots=8,
            snapshot_prob=0.35,
            workers=args.workers,
            parallel_backend=args.parallel_backend,
            seed=args.ppo_seed,
            output=str(output),
            log_jsonl=str(log_jsonl),
            resume=args.resume,
            export_best=args.export_best,
            selection_warmup=args.selection_warmup,
            early_stop_patience=args.ppo_early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            selection_bust_penalty=args.selection_bust_penalty,
            min_export_mean_delta=args.min_export_mean_delta,
            min_training_hands=args.min_training_hands,
            allow_smoke=args.allow_smoke,
            progress=args.progress,
            json=True,
        )
    return SimpleNamespace(
        kind="bucket",
        generations=args.bucket_generations,
        matches_per_generation=args.bucket_matches_per_generation,
        hands=args.bucket_hands,
        players=6,
        train_seats=2,
        hidden=128,
        bucket_count=args.bucket_count,
        abstraction="cfr-pokerbot",
        bucket_update="cfr-plus",
        learning_rate=args.bucket_learning_rate,
        entropy_coef=0.004,
        ppo_clip_ratio=0.20,
        ppo_value_coef=0.35,
        max_grad_norm=0.75,
        epochs=3,
        batch_size=2048,
        feature_norm_momentum=0.0,
        replay_generations=4,
        replay_max_decisions=24000,
        temperature=args.bucket_temperature,
        reward_scale=1000.0,
        reward_clip=args.bucket_reward_clip,
        opponent_pool=args.opponent_pool,
        extra_bot_path=[],
        snapshot_interval=2,
        max_snapshots=8,
        snapshot_prob=0.35,
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        seed=args.bucket_seed,
        output=str(output),
        log_jsonl=str(log_jsonl),
        resume=args.resume,
        export_best=args.export_best,
        selection_warmup=args.selection_warmup,
        early_stop_patience=args.bucket_early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        selection_bust_penalty=args.selection_bust_penalty,
        min_export_mean_delta=args.min_export_mean_delta,
        min_training_hands=args.min_training_hands,
        allow_smoke=args.allow_smoke,
        progress=args.progress,
        json=True,
    )


def _deep_args(args, output: Path):
    return SimpleNamespace(
        generations=args.deep_generations,
        matches_per_generation=args.deep_matches_per_generation,
        hands=args.deep_hands,
        players=6,
        train_seats=2,
        hidden=args.deep_hidden,
        learning_rate=args.deep_learning_rate,
        bootstrap_learning_rate=args.deep_bootstrap_learning_rate,
        entropy_coef=args.deep_entropy_coef,
        ppo_clip_ratio=args.deep_clip_ratio,
        ppo_value_coef=args.deep_value_coef,
        max_grad_norm=args.deep_max_grad_norm,
        epochs=args.deep_epochs,
        batch_size=args.deep_batch_size,
        replay_generations=args.deep_replay_generations,
        replay_max_decisions=args.deep_replay_max_decisions,
        temperature=args.deep_temperature,
        reward_scale=1000.0,
        reward_clip=args.deep_reward_clip,
        selection_warmup=args.selection_warmup,
        selection_bust_penalty=args.selection_bust_penalty,
        opponent_pool=args.deep_opponent_pool,
        snapshot_interval=2,
        max_snapshots=8,
        snapshot_prob=0.30,
        bootstrap_samples=args.deep_bootstrap_samples,
        bootstrap_holdout=args.deep_bootstrap_holdout,
        bootstrap_epochs=args.deep_bootstrap_epochs,
        bootstrap_style=args.deep_bootstrap_style,
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        seed=args.deep_seed,
        output=str(output),
        resume=args.resume,
        min_training_hands=args.min_training_hands,
        allow_smoke=args.allow_smoke,
        progress=args.progress,
        json=True,
    )


def _arm_args(args, output: Path):
    return SimpleNamespace(
        generations=args.arm_generations,
        matches_per_generation=args.arm_matches_per_generation,
        hands=args.arm_hands,
        players=6,
        train_seats=2,
        learning_rate=args.arm_learning_rate,
        bootstrap_learning_rate=args.arm_bootstrap_learning_rate,
        epochs=args.arm_epochs,
        max_grad_norm=args.arm_max_grad_norm,
        temperature=args.arm_temperature,
        learned_blend=args.arm_learned_blend,
        reward_scale=1000.0,
        reward_clip=args.arm_reward_clip,
        selection_warmup=args.selection_warmup,
        selection_bust_penalty=args.selection_bust_penalty,
        opponent_pool=args.arm_opponent_pool,
        snapshot_interval=2,
        max_snapshots=8,
        snapshot_prob=0.25,
        replay_max_decisions=args.arm_replay_max_decisions,
        bootstrap_samples=args.arm_bootstrap_samples,
        bootstrap_holdout=args.arm_bootstrap_holdout,
        bootstrap_epochs=args.arm_bootstrap_epochs,
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        seed=args.arm_seed,
        output=str(output),
        resume=args.resume,
        min_training_hands=args.min_training_hands,
        allow_smoke=args.allow_smoke,
        progress=args.progress,
        json=True,
    )


def _arm_tune_args(args, output_root: Path, policy: Path, promote_output: Path | None):
    return SimpleNamespace(
        run_id="arm-selector-params",
        output_root=str(output_root),
        policy=str(policy),
        promote_output=str(promote_output) if promote_output else None,
        generations=args.arm_tune_generations,
        population=args.arm_tune_population,
        elite_frac=args.arm_tune_elite_frac,
        stages=args.arm_tune_stages,
        init_sigma=args.arm_tune_init_sigma,
        min_sigma=args.arm_tune_min_sigma,
        max_sigma=args.arm_tune_max_sigma,
        smoothing=args.arm_tune_smoothing,
        default_pull=args.arm_tune_default_pull,
        min_hands_per_candidate=args.arm_tune_min_hands_per_candidate,
        allow_smoke=args.allow_smoke,
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        seed=args.arm_tune_seed,
        progress=args.progress,
        json=True,
    )


def _gate_args(args, candidate_paths: dict[str, str] | None):
    return SimpleNamespace(
        candidate=args.gate_candidate,
        baseline=args.gate_baseline,
        mode=args.gate_mode,
        seed_count=args.gate_seed_count,
        seed_start=args.gate_seed_start,
        hands=args.gate_hands,
        min_mean_delta=args.gate_min_mean_delta,
        min_win_rate=args.gate_min_win_rate,
        min_tasks_per_candidate=args.gate_min_tasks_per_candidate,
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        allow_smoke=args.allow_smoke,
        json=True,
        candidate_paths=candidate_paths,
    )


def _save_stage(run_dir: Path, name: str, report: dict) -> dict:
    report_path = run_dir / f"{name}.json"
    report["report_path"] = str(report_path)
    _write_json(report_path, _json_safe(report))
    return report


def run(args) -> dict:
    validate_orchestration_budget(args)
    run_id = args.run_id or "strong-mocks-" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_root = _artifact_root(args, run_dir)
    artifact_paths = _model_artifact_paths(artifact_root)
    candidate_paths = _candidate_paths(artifact_root)

    reports: dict[str, dict] = {}

    if args.progress:
        print("==> Train oracle-imitation strong mock", file=sys.stderr)
    reports["oracle_imitation"] = _save_stage(
        run_dir,
        "oracle_imitation",
        train_imitation.train(
            args.oracle_samples,
            args.oracle_seed,
            args.oracle_style,
            artifact_paths["oracle_imitation"],
            hidden=args.oracle_hidden,
        ),
    )

    if args.progress:
        print("==> Train PPO-style strong mock", file=sys.stderr)
    reports["ppo_policy"] = _save_stage(
        run_dir,
        "ppo_policy",
        train_real_policy.train(
            _real_policy_args(
                args,
                "ppo",
                artifact_paths["ppo_policy"],
                run_dir / "ppo_policy_training.jsonl",
            )
        ),
    )

    if args.progress:
        print("==> Train bucket/CFR-like strong mock", file=sys.stderr)
    reports["cfr_bucket"] = _save_stage(
        run_dir,
        "cfr_bucket",
        train_real_policy.train(
            _real_policy_args(
                args,
                "bucket",
                artifact_paths["cfr_bucket"],
                run_dir / "cfr_bucket_training.jsonl",
            )
        ),
    )

    if args.progress:
        print("==> Train deep PPO strong mock", file=sys.stderr)
    reports["ppo_deep_policy"] = _save_stage(
        run_dir,
        "ppo_deep_policy",
        train_deep_ppo.train(_deep_args(args, artifact_paths["ppo_deep_policy"])),
    )

    if args.progress:
        print("==> Train heuristic RL selector strong mock", file=sys.stderr)
    reports["heuristic_rl_selector"] = _save_stage(
        run_dir,
        "heuristic_rl_selector",
        train_arm_selector.train(_arm_args(args, artifact_paths["heuristic_rl_selector"])),
    )

    if args.progress:
        print("==> Tune heuristic RL selector parameters", file=sys.stderr)
    reports["arm_selector_params"] = _save_stage(
        run_dir,
        "arm_selector_params",
        tune_arm_selector_params.tune(
            _arm_tune_args(
                args,
                run_dir / "arm_selector_param_tuning",
                artifact_paths["heuristic_rl_selector"],
                artifact_paths["heuristic_rl_selector_params"],
            )
        ),
    )

    gate_report = None
    if not args.skip_gate:
        if args.progress:
            print("==> Gate all strong mocks against default/reference bots", file=sys.stderr)
        gate_report = strong_gate.run(_gate_args(args, candidate_paths))
        reports["strong_mock_gate"] = _save_stage(run_dir, "strong_mock_gate", gate_report)

    summary = {
        "mode": "all_strong_mock_training",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "allow_smoke": bool(args.allow_smoke),
        "artifact_root": str(artifact_root) if artifact_root else "canonical",
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
        "candidate_paths": candidate_paths or strong_gate.CANDIDATES,
        "trainable": [
            "oracle_imitation",
            "ppo_policy",
            "cfr_bucket",
            "ppo_deep_policy",
            "heuristic_rl_selector",
            "heuristic_rl_selector_params",
        ],
        "algorithmic": ["rollout_search"],
        "composite": ["ensemble"],
        "stages": [_stage_record(name, report) for name, report in reports.items()],
        "reports": reports,
        "passed": bool(gate_report["passed"]) if gate_report is not None else None,
    }
    _write_json(run_dir / "summary.json", _json_safe(summary))
    if gate_report is not None and args.require_gate and not gate_report["passed"]:
        raise SystemExit(f"strong-mock gate failed; inspect {run_dir / 'strong_mock_gate.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, tune, and gate all strong mock opponents")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--min-oracle-samples", type=int, default=DEFAULT_MIN_ORACLE_SAMPLES)
    parser.add_argument("--min-training-hands", type=int, default=DEFAULT_MIN_TRAINING_HANDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selection-warmup", type=int, default=4)
    parser.add_argument("--early-stop-min-delta", type=float, default=250.0)
    parser.add_argument("--selection-bust-penalty", type=float, default=9000.0)
    parser.add_argument("--min-export-mean-delta", type=float, default=1500.0)
    parser.add_argument("--opponent-pool", choices=["oracle", "fast", "mixed", "adversarial"], default="adversarial")

    parser.add_argument("--oracle-samples", type=int, default=200_000)
    parser.add_argument("--oracle-style", choices=["balanced", "value", "bluff", "station", "folder", "pressure"], default="balanced")
    parser.add_argument("--oracle-hidden", type=int, default=48)
    parser.add_argument("--oracle-seed", type=int, default=4242)

    parser.add_argument("--ppo-generations", type=int, default=48)
    parser.add_argument("--ppo-matches-per-generation", type=int, default=128)
    parser.add_argument("--ppo-hands", type=int, default=400)
    parser.add_argument("--ppo-hidden", type=int, default=128)
    parser.add_argument("--ppo-batch-size", type=int, default=4096)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-learning-rate", type=float, default=0.002)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.002)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.20)
    parser.add_argument("--ppo-value-coef", type=float, default=0.35)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=0.75)
    parser.add_argument("--ppo-feature-norm-momentum", type=float, default=0.0)
    parser.add_argument("--ppo-replay-generations", type=int, default=4)
    parser.add_argument("--ppo-replay-max-decisions", type=int, default=24_000)
    parser.add_argument("--ppo-temperature", type=float, default=0.62)
    parser.add_argument("--ppo-reward-clip", type=float, default=5.0)
    parser.add_argument("--ppo-early-stop-patience", type=int, default=14)
    parser.add_argument("--ppo-seed", type=int, default=6161)

    parser.add_argument("--bucket-generations", type=int, default=40)
    parser.add_argument("--bucket-matches-per-generation", type=int, default=128)
    parser.add_argument("--bucket-hands", type=int, default=400)
    parser.add_argument("--bucket-count", type=int, default=32768)
    parser.add_argument("--bucket-learning-rate", type=float, default=0.004)
    parser.add_argument("--bucket-temperature", type=float, default=0.80)
    parser.add_argument("--bucket-reward-clip", type=float, default=5.0)
    parser.add_argument("--bucket-early-stop-patience", type=int, default=24)
    parser.add_argument("--bucket-seed", type=int, default=5151)

    parser.add_argument("--deep-generations", type=int, default=24)
    parser.add_argument("--deep-matches-per-generation", type=int, default=128)
    parser.add_argument("--deep-hands", type=int, default=400)
    parser.add_argument("--deep-hidden", default="96,64,32")
    parser.add_argument("--deep-learning-rate", type=float, default=0.0035)
    parser.add_argument("--deep-bootstrap-learning-rate", type=float, default=0.006)
    parser.add_argument("--deep-entropy-coef", type=float, default=0.003)
    parser.add_argument("--deep-clip-ratio", type=float, default=0.18)
    parser.add_argument("--deep-value-coef", type=float, default=0.40)
    parser.add_argument("--deep-max-grad-norm", type=float, default=0.70)
    parser.add_argument("--deep-epochs", type=int, default=3)
    parser.add_argument("--deep-batch-size", type=int, default=2048)
    parser.add_argument("--deep-replay-generations", type=int, default=4)
    parser.add_argument("--deep-replay-max-decisions", type=int, default=32_000)
    parser.add_argument("--deep-temperature", type=float, default=0.55)
    parser.add_argument("--deep-reward-clip", type=float, default=10.0)
    parser.add_argument("--deep-opponent-pool", choices=["oracle", "fast", "adversarial"], default="adversarial")
    parser.add_argument("--deep-bootstrap-samples", type=int, default=60_000)
    parser.add_argument("--deep-bootstrap-holdout", type=int, default=12_000)
    parser.add_argument("--deep-bootstrap-epochs", type=int, default=4)
    parser.add_argument("--deep-bootstrap-style", choices=["pressure", "value", "conservative", "explore"], default="pressure")
    parser.add_argument("--deep-seed", type=int, default=7171)

    parser.add_argument("--arm-generations", type=int, default=24)
    parser.add_argument("--arm-matches-per-generation", type=int, default=128)
    parser.add_argument("--arm-hands", type=int, default=400)
    parser.add_argument("--arm-learning-rate", type=float, default=0.0025)
    parser.add_argument("--arm-bootstrap-learning-rate", type=float, default=0.020)
    parser.add_argument("--arm-epochs", type=int, default=2)
    parser.add_argument("--arm-max-grad-norm", type=float, default=0.75)
    parser.add_argument("--arm-temperature", type=float, default=0.46)
    parser.add_argument("--arm-learned-blend", type=float, default=0.72)
    parser.add_argument("--arm-reward-clip", type=float, default=8.0)
    parser.add_argument("--arm-opponent-pool", choices=["rollout", "fast", "adversarial"], default="adversarial")
    parser.add_argument("--arm-replay-max-decisions", type=int, default=36_000)
    parser.add_argument("--arm-bootstrap-samples", type=int, default=60_000)
    parser.add_argument("--arm-bootstrap-holdout", type=int, default=12_000)
    parser.add_argument("--arm-bootstrap-epochs", type=int, default=5)
    parser.add_argument("--arm-seed", type=int, default=8181)

    parser.add_argument("--arm-tune-generations", type=int, default=6)
    parser.add_argument("--arm-tune-population", type=int, default=32)
    parser.add_argument("--arm-tune-elite-frac", type=float, default=0.20)
    parser.add_argument("--arm-tune-stages", default=tune_arm_selector_params.DEFAULT_STAGE_SPEC)
    parser.add_argument("--arm-tune-init-sigma", type=float, default=0.18)
    parser.add_argument("--arm-tune-min-sigma", type=float, default=0.025)
    parser.add_argument("--arm-tune-max-sigma", type=float, default=0.30)
    parser.add_argument("--arm-tune-smoothing", type=float, default=0.35)
    parser.add_argument("--arm-tune-default-pull", type=float, default=0.04)
    parser.add_argument("--arm-tune-min-hands-per-candidate", type=int, default=6400)
    parser.add_argument("--arm-tune-seed", type=int, default=8383)

    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--require-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate-candidate", choices=sorted(strong_gate.CANDIDATES), action="append")
    parser.add_argument("--gate-baseline", choices=sorted(strong_gate.BASELINES), action="append")
    parser.add_argument("--gate-mode", choices=["sixmax", "heads-up"], action="append")
    parser.add_argument("--gate-seed-count", type=int, default=128)
    parser.add_argument("--gate-seed-start", type=int, default=42001)
    parser.add_argument("--gate-hands", type=int, default=400)
    parser.add_argument("--gate-min-mean-delta", type=float, default=500.0)
    parser.add_argument("--gate-min-win-rate", type=float, default=0.55)
    parser.add_argument("--gate-min-tasks-per-candidate", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
