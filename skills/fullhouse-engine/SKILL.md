---
name: fullhouse-engine
description: Use when working in the fullhouse-engine repo for the Fullhouse poker bot hackathon, including setup, bot validation, sandbox restrictions, reference bot changes, or documentation updates.
---

# Fullhouse Engine

## First Steps

- Read `docs/project-overview.md` for repo purpose, file roles, tournament format, and frozen areas.
- Read `docs/restrictions.md` before changing or writing any bot logic.
- Read `docs/bot-state-and-memory.md` when reasoning about `decide()` inputs, action history, opponent modeling, or in-memory bot state.
- Read `docs/heuristic-bot-plan.md` before implementing or tuning the heuristic competition bot.
- Read `docs/heuristic-bot-logic.md` before modifying `bots/heuristic/bot.py`; it explains the current policy and improvement backlog.
- Read `docs/heuristic-improvement-backlog.md` before selecting heuristic bot improvements; it ranks tasks by ease and defines success criteria.
- Read `docs/heuristic-parameter-audit.md` before tuning heuristic thresholds or bet sizes.
- Read `docs/four-day-execution-plan.md` before starting larger heuristic implementation work.
- Read `docs/external-poker-ai-benchmarks.md` before adding any external poker AI benchmark opponent.
- Read `docs/training-pipelines.md` before changing real training, league
  training, fast matches, self-training, or coevolution scripts.
- Read `docs/heuristic-rl-arm-selector.md` before adding any new RL-assisted
  strategy architecture.
- Read `docs/ppo-bot-logic.md` before changing `bots/strong_mocks/ppo_policy`
  `bots/strong_mocks/ppo_deep_policy`, or PPO training logic.
- Read `docs/heuristic-benchmark-results.md` before comparing new heuristic
  changes or promoting a default.
- Read `docs/setup-poetry.md` before changing dependencies or environment setup.

## Environment

- Use Python 3.10. The repo-local version is `3.10.20` in `.python-version`.
- Prefer Poetry commands:
  - `make poetry-install`
  - `poetry run pytest -q`
  - `poetry run python sandbox/validator.py bots/template/bot.py`
  - `poetry run python sandbox/match.py bots/template/bot.py bots/shark/bot.py --hands 20 --seed 7`
- `eval7==0.1.7` needs the documented no-build-isolation install path. Do not replace it with a normal install command unless the package issue is verified fixed.

## Change Policy

- Treat `engine/game.py`, `sandbox/runner.py`, and `db/schema.sql` as frozen unless the user explicitly asks for a change there.
- Keep bot submissions compatible with the validator restrictions: no network, subprocesses, threading, async, file writes during gameplay, dynamic imports, or reflection escapes.
- Keep `bots/heuristic/bot.py` submission-shaped: one production file with internal sections, one final action sanitizer, and no helper package required for validation.
- Use `data/` only for read-only assets loaded at module-import time.
- When changing setup, restrictions, or bot behavior assumptions, update the matching file under `docs/` and this skill if the workflow changes.

## Validation

For setup or dependency changes, run:

```bash
poetry run pytest -q
poetry run python sandbox/validator.py bots/template/bot.py
poetry run python sandbox/match.py bots/template/bot.py bots/shark/bot.py --hands 5 --seed 7 --json
```

For Mermaid documentation changes, keep diagrams in the conservative
`graph TD` form and run:

```bash
poetry run pytest -q tests/test_mermaid_docs.py
```

For bot changes, validate the edited bot directly and run at least one seeded local match against `bots/shark/bot.py`.

For heuristic bot benchmark passes, run:

```bash
poetry run python tools/evaluate_heuristic.py --json
```

For full benchmark summaries, run:

```bash
poetry run python tools/select_heuristic_config.py --preset final --config baseline --progress --json
```

For default-promotion decisions, prefer paired incumbent-vs-candidate gates:

```bash
poetry run python tools/paired_heuristic_gate.py --incumbent baseline --candidate CANDIDATE_CONFIG --preset promotion --workers 0 --parallel-backend process --progress --json
```

Use this before promoting or reverting submitted-bot defaults because it
compares identical suite/seed pairs and exposes paired mean, median, p10, win
rate, bust, and error changes.

For expanded mock-family screens, run:

```bash
poetry run python tools/select_heuristic_config.py --preset mock-family --config baseline --progress
```

For training and coevolution pipelines, read `docs/training-pipelines.md`.
Normal large entrypoints are:

```bash
WORKERS=0 PARALLEL_BACKEND=process scripts/train_opponents_real.sh
WORKERS=0 PARALLEL_BACKEND=process scripts/train_e2e_coevolution.sh
WORKERS=0 PARALLEL_BACKEND=process scripts/train_heuristics_selfplay.sh
```

For independent deep PPO experiments that must not touch the existing
`ppo_policy`, use:

```bash
poetry run python tools/strong_mocks/train_deep_ppo.py --output bots/strong_mocks/ppo_deep_policy/data/policy.npz --progress --json
```

For the preferred RL-assisted architecture, train the heuristic expert-arm
selector. It learns to choose among named poker heuristic candidates rather
than raw actions:

```bash
poetry run python tools/strong_mocks/train_arm_selector.py --output bots/strong_mocks/heuristic_rl_selector/data/policy.npz --progress --json
poetry run pytest -q tests/test_arm_selector.py
poetry run python sandbox/validator.py bots/strong_mocks/heuristic_rl_selector --json
```

Do not manually tune selector thresholds or heuristic constants from smoke-run
chip deltas. Smoke runs check wiring only; promotion needs fixed-seed held-out
sample sizes from `docs/training-pipelines.md`.

For script-driven selector parameter tuning, use CEM/racing:

```bash
poetry run python tools/strong_mocks/tune_arm_selector_params.py --generations 6 --population 32 --stages 16:400:0.35,64:400:1.0 --workers 0 --parallel-backend process --progress --json
```

This writes to `runs/arm_selector_param_tuning/`. Do not pass
`--promote-output` unless the run uses statistically meaningful stages and the
held-out report has been reviewed. `--allow-smoke` is for wiring checks only.

For a serious but bounded real-opponent run on a laptop, prefer:

```bash
RUN_ID=adversarial-opponents-$(date +%Y%m%d-%H%M%S) \
WORKERS=0 \
PARALLEL_BACKEND=process \
OPPONENT_POOL=adversarial \
EXPORT_BEST=1 \
SELECTION_WARMUP=4 \
EARLY_STOP_MIN_DELTA=250 \
MIN_EXPORT_MEAN_DELTA=1500 \
PPO_EARLY_STOP_PATIENCE=14 \
PPO_GENERATIONS=36 \
PPO_MATCHES_PER_GENERATION=128 \
PPO_HANDS=400 \
BUCKET_EARLY_STOP_PATIENCE=24 \
BUCKET_GENERATIONS=32 \
BUCKET_MATCHES_PER_GENERATION=128 \
BUCKET_HANDS=400 \
scripts/train_opponents_real.sh
```

If the final report says `"exported": false`, the run completed but kept the
existing repo artifact because the selected checkpoint did not clear
`MIN_EXPORT_MEAN_DELTA`.

For league-style staged opponent training with held-out promotion gates:

```bash
RUN_ID=league-opponents-$(date +%Y%m%d-%H%M%S) \
WORKERS=0 \
PARALLEL_BACKEND=process \
GENERATION_SCALE=1.0 \
MATCH_SCALE=1.0 \
HAND_SCALE=1.0 \
EVAL_SEEDS=128 \
EVAL_HANDS=400 \
PROMOTE=1 \
scripts/train_opponents_league.sh
```

For E2E alternating PPO/heuristic coevolution:

```bash
WORKERS=0 \
PARALLEL_BACKEND=process \
CYCLES=4 \
PPO_ARMS=stable,explore,conservative \
PPO_INIT=auto \
PPO_INIT_SAMPLES=80000 \
PPO_GENERATIONS=8 \
PPO_MATCHES_PER_GENERATION=128 \
PPO_HANDS=400 \
HEURISTIC_POPULATION=14 \
HEURISTIC_ELITE=4 \
HEURISTIC_MATCHES_PER_GENERATION=64 \
HEURISTIC_HANDS=400 \
EVAL_SEEDS=128 \
EVAL_HANDS=400 \
scripts/train_e2e_coevolution.sh
```

By default, the run ID is `coevolve-current` and artifacts are written under
`runs/fullhouse_coevolution/coevolve-current/`, which is git-ignored and
resumable. Re-running the same command continues from `state.json`; use
`RESET=1` for a clean run. Read `summary.json` and `ev_progress.svg` before
promoting any artifact. PPO promotion requires `PROMOTE_PPO=1`; heuristic
configs should be benchmark-gated manually before changing submitted defaults.

For a league smoke that cannot overwrite repo artifacts:

```bash
RUN_ID=league-smoke \
GENERATION_SCALE=0.05 \
MATCH_SCALE=0.05 \
HAND_SCALE=0.10 \
EVAL_SEEDS=1 \
EVAL_HANDS=12 \
PROMOTE=0 \
WORKERS=1 \
scripts/train_opponents_league.sh
```

For unrestricted fast local bot-directory batches:

```bash
poetry run python -m training.fast_match bots/heuristic bots/strong_mocks/ensemble bots/shark --hands 400 --repeat 32 --workers 0 --parallel-backend process --json
```

Validate fast-runner parity after changes:

```bash
poetry run pytest -q tests/test_fast_match.py
```

For multi-heuristic self-training, run a smoke first:

```bash
poetry run python tools/strong_mocks/self_train_heuristic.py --run-id smoke --generations 1 --population 4 --elite 2 --matches-per-generation 2 --hands 12 --seed 22 --json
```

For larger parallel heuristic self-training, use:

```bash
WORKERS=0 PARALLEL_BACKEND=process scripts/train_heuristics_selfplay.sh
```

For faster offline validations, prefer `--workers 0 --parallel-backend process`.
Use `--parallel-backend thread` only for short local checks where process
startup overhead dominates. Do not use MLX, threading, or multiprocessing from
the submitted `bots/heuristic/bot.py`.

For retraining benchmark-only numpy-policy mock competitors, run:

```bash
poetry run python tools/train_mock_numpy_policy.py --all --samples 60000 --seed 7331
```

For focused sizing/pressure benchmark checks, run:

```bash
poetry run python tools/evaluate_heuristic.py --suite sizing_6max --suite pressure_6max --suite mixed_stress_6max --seed-count 128 --hands 400 --summary-only --json
```

For heuristic submission packaging, run:

```bash
poetry run python tools/harden_submission.py --json
```

For threshold tuning comparisons, run:

```bash
poetry run python tools/tune_heuristic_thresholds.py --json
```

For threshold tuning summaries, run:

```bash
poetry run python tools/tune_heuristic_thresholds.py --seed-count 128 --summary-only --json
```

For risk-aware config ranking, run:

```bash
poetry run python tools/select_heuristic_config.py --preset candidate --progress
```

For weak-spot candidate checks, use at least a 64-seed candidate screen and a
128-seed gate before changing defaults. The latest tested weak-spot configs are
`pressure-control`, `equity-control`, `trap-control`, and `weakspot-control`;
none are promoted as defaults.

For postflop feature candidate checks, the latest tested configs are
`line-aware`, `blocker-probe`, and `pair-danger`. They exercise richer hand
features, blocker bluffs, delayed probes, paired-board caution, and
pot-odds-like sizing suspicion. The latest focused screen kept `baseline` as
the default, so treat these configs as diagnostics until a promotion gate says
otherwise.

For regenerating the explicit 169-class preflop table, run:

```bash
poetry run python tools/generate_preflop_table.py --iterations 20000 --seed 31337
```

For exported qualifier hand-history analysis, run:

```bash
poetry run python tools/analyze_hand_history.py path/to/history.json --json
```
