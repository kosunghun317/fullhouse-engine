---
name: fullhouse-engine
description: Use when working in the fullhouse-engine repo for the Fullhouse poker bot hackathon, including setup, bot validation, sandbox restrictions, reference bot changes, or documentation updates.
---

# Fullhouse Engine

## First Steps

- Read `docs/strategy-and-training.md` for repo purpose, file roles, strategy,
  likely competitor logic, training flow, and optimization flow.
- Read `docs/qualifier-2-strategy.md` before changing portal replay tooling,
  replay-derived opponent profiles, or second-chance qualifier strategy.
- Read `docs/restrictions.md` before changing or writing any bot logic.
- Read `docs/heuristic-full-space-optimization.md` before running or changing
  the full-space heuristic optimizer.
- Read `docs/training-pipelines.md` before changing real training, league
  training, fast matches, self-training, or coevolution scripts.
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

For public portal replay analysis, keep all repo Python commands under Poetry:

```bash
poetry run python tools/download_portal_targeted_history.py --rank-lte 64 --bot-name slop3 --output runs/portal_history/top64_plus_slop3
poetry run python tools/analyze_portal_strategy.py runs/portal_history/top64_plus_slop3 --output-json runs/portal_history/top64_plus_slop3/strategy_report.json
poetry run python tools/build_portal_profiles.py runs/portal_history/top64_plus_slop3
poetry run python tools/build_portal_profile_mocks.py runs/portal_history/top64_plus_slop3 --output runs/portal_profile_mocks/top64_plus_slop3
poetry run python tools/evaluate_portal_profile_mocks.py runs/portal_profile_mocks/top64_plus_slop3 --mode sixmax --hands 400 --seed-count 64 --json
poetry run python tools/evaluate_heuristic.py --portal-mocks-dir runs/portal_profile_mocks/top64_plus_slop3 --portal-only --hands 400 --seed-count 64 --summary-only --json
poetry run python tools/tune_heuristic_full_space.py --portal-mocks-dir runs/portal_profile_mocks/top64_plus_slop3 --generations 4 --population 20 --elite 5 --stages 128:400:0.5,256:400:0.25 --workers 0 --parallel-backend process --progress --json
```

Replay outputs under `runs/portal_history/` can be large and should not be
committed. Generated profile mocks under `runs/portal_profile_mocks/` are also
local-only. Commit code, docs, and small fixtures/manifests only.

For full benchmark summaries, run:

```bash
poetry run python tools/select_heuristic_config.py --preset final --config baseline --progress --json
```

For a one-command unattended submission pass, run:

```bash
TRAIN_STRONG_MOCKS=0 WORKERS=0 PARALLEL_BACKEND=process scripts/build_tuned_submission.sh
```

This uses the default deadline profile: 3 generations, 16 candidates, 4 elites,
stages `128:400:0.5,256:400:0.25`, and a final rollout-search gate. It
reuses existing strong mocks, runs full-space heuristic tuning, bakes the
selected `best_env.json` into the packaged `bot.py`, runs submission validation,
and rebuilds `dist/heuristic_bot.zip`. Omit `TRAIN_STRONG_MOCKS=0` only when
there is enough time to retrain and gate the strong-mock pool first.

For the older full-strength profile, set:

```bash
TUNE_PROFILE=full SUBMISSION_PROFILE=full WORKERS=0 PARALLEL_BACKEND=process scripts/build_tuned_submission.sh
```

For a packaging-only rerun from an existing full-space tune:

```bash
TRAIN_STRONG_MOCKS=0 RUN_TUNING=0 HEURISTIC_ENV_FILE=runs/heuristic_full_space_tuning/<run-id>/best_env.json WORKERS=0 PARALLEL_BACKEND=process scripts/build_tuned_submission.sh
```

For default-promotion decisions, prefer paired incumbent-vs-candidate gates:

```bash
poetry run python tools/paired_heuristic_gate.py --incumbent baseline --candidate CANDIDATE_CONFIG --preset promotion --workers 0 --parallel-backend process --progress --json
```

Use this before promoting or reverting submitted-bot defaults because it
compares identical suite/seed pairs and exposes paired mean, mean confidence
interval, median, p10, win rate, bust, and error changes. Promotion requires
the paired mean-difference lower CI bound to be positive. The submission
pipeline defaults run well above the tool's hard minimum of 100 paired tasks.

For expanded mock-family screens, run:

```bash
poetry run python tools/select_heuristic_config.py --preset mock-family --config baseline --progress
```

For training and coevolution pipelines, read `docs/training-pipelines.md`.
Normal large entrypoints are:

```bash
WORKERS=0 PARALLEL_BACKEND=process scripts/train_all_strong_mocks.sh
WORKERS=0 PARALLEL_BACKEND=process scripts/train_opponents_real.sh
WORKERS=0 PARALLEL_BACKEND=process scripts/train_e2e_coevolution.sh
WORKERS=0 PARALLEL_BACKEND=process scripts/train_heuristics_selfplay.sh
```

Use `scripts/train_all_strong_mocks.sh` as the primary opponent-building
command. It trains oracle imitation, PPO, bucket/CFR, deep PPO, and the
heuristic selector, tunes selector parameters, then gates all seven strong
mocks. `scripts/train_opponents_real.sh` is only the lower-level PPO/bucket
subset.

For independent deep PPO experiments that must not touch the existing
`ppo_policy`, use:

```bash
poetry run python tools/strong_mocks/train_deep_ppo.py --output bots/strong_mocks/ppo_deep_policy/data/policy.npz --bootstrap-samples 60000 --bootstrap-holdout 12000 --generations 24 --matches-per-generation 128 --hands 400 --opponent-pool adversarial --progress --json
```

For the preferred RL-assisted architecture, train the heuristic expert-arm
selector. It learns to choose among named poker heuristic candidates rather
than raw actions:

```bash
poetry run python tools/strong_mocks/train_arm_selector.py --output bots/strong_mocks/heuristic_rl_selector/data/policy.npz --bootstrap-samples 60000 --bootstrap-holdout 12000 --generations 24 --matches-per-generation 128 --hands 400 --opponent-pool adversarial --progress --json
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

For rollout-search f/g sizing experiments, use the dedicated CEM/racing tuner:

```bash
poetry run python tools/strong_mocks/tune_rollout_search.py --generations 4 --population 12 --elite 3 --stages 4:120:0.5,12:240:0.5 --workers 0 --parallel-backend process --progress --json
```

It writes generated wrappers and `best_params.json` under
`runs/rollout_search_tuning/`, uses 1024 eval7 samples for preflop and postflop
equity, and performs a representative 2-second decision-latency check before
trusting the result.

Read `docs/rollout-search.md` for the exact rollout-search equity sampler,
f/g formulas, tunable parameters, and the final rollout gate used by deadline
submission runs.

For adaptive threshold-rollout candidate screens, use the tabulated benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 poetry run python tools/benchmark_adaptive_threshold_rollout.py --hands 200 --seeds 16
```

It prints streaming per-suite completion lines and a final `tabulate` summary.
Use `--json` only when another tool should consume the raw rows.

To run the packaged submission heuristic on that same suite matrix, use:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 poetry run python tools/benchmark_submission_heuristic.py --hands 512 --seeds 128
```

To build a candidate that replaces the packaged heuristic's preflop score with
Monte Carlo, use:

```bash
PYTHONDONTWRITEBYTECODE=1 poetry run python tools/build_preflop_mc_heuristic.py --input dist/heuristic_bot.zip --output runs/preflop_mc_heuristic/preflop-mc-4096/bot --samples 4096 --budget 0.35 --min-samples 512 --json
```

Read `docs/preflop-monte-carlo-heuristic.md` for latency results and benchmark
commands before promoting this variant.

To tune rollout search first, then tune/package the heuristic and compare
against the tuned rollout wrapper last, run:

```bash
ROLLOUT_RUN_ID=rollout-final-$(date +%Y%m%d-%H%M%S) && BUILD_RUN_ID=tuned-final-$(date +%Y%m%d-%H%M%S) && poetry run python tools/strong_mocks/tune_rollout_search.py --run-id "$ROLLOUT_RUN_ID" --generations 4 --population 12 --elite 3 --stages 4:120:0.5,12:240:0.5 --workers 0 --parallel-backend process --progress --json && TRAIN_STRONG_MOCKS=0 RUN_ID="$BUILD_RUN_ID" TUNE_PROFILE=deadline-24h SUBMISSION_PROFILE=deadline-24h ROLLOUT_FINAL_BOT_PATH="runs/rollout_search_tuning/${ROLLOUT_RUN_ID}/best_bot" WORKERS=0 PARALLEL_BACKEND=process scripts/build_tuned_submission.sh
```

Inspect `runs/submission_pipeline/${BUILD_RUN_ID}-submission/final_rollout_gate.json`;
the packaged zip remains `dist/heuristic_bot.zip`.

For full-space heuristic parameter search, use the staged CEM/racing tuner
instead of ad hoc one-parameter sweeps:

```bash
poetry run python tools/tune_heuristic_full_space.py --preset strong-screen --generations 6 --population 24 --elite 6 --stages 128:400:0.5,256:400:0.25 --workers 0 --parallel-backend process --progress --json
```

This writes to `runs/heuristic_full_space_tuning/` and refuses serious runs
below 512 suite/seed tasks per candidate stage and 1024 final-stage tasks per
candidate. It writes both `best_env.json` and `best_env.sh`; use
`HEURISTIC_ENV_FILE=<run>/best_env.json` with packaging/pipeline commands when
the tuned env should be baked into the submission zip. `--allow-smoke` is only
for wiring checks.

For a serious but bounded full strong-mock run on a laptop, prefer:

```bash
RUN_ID=adversarial-strong-mocks-$(date +%Y%m%d-%H%M%S) \
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
DEEP_GENERATIONS=24 \
DEEP_MATCHES_PER_GENERATION=128 \
DEEP_HANDS=400 \
ARM_GENERATIONS=24 \
ARM_MATCHES_PER_GENERATION=128 \
ARM_HANDS=400 \
GATE_SEED_COUNT=128 \
GATE_HANDS=400 \
scripts/train_all_strong_mocks.sh
```

If the final report says `"exported": false`, the run completed but kept the
existing repo artifact because the selected checkpoint did not clear
`MIN_EXPORT_MEAN_DELTA`.

After any targeted strong-mock artifact change outside
`scripts/train_all_strong_mocks.sh`, run the statistical strength gate before
using the artifacts as promotion opponents:

```bash
poetry run python tools/check_strong_mocks.py --seed-count 128 --hands 400 --workers 0 --parallel-backend process --json
```

The gate covers every `bots/strong_mocks/*` wrapper against default/reference
bots in six-max and heads-up tasks. Do not call a strong-mock artifact trained
unless it passes this gate or the failure is intentionally documented.

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
PPO_GENERATIONS=20 \
PPO_MATCHES_PER_GENERATION=128 \
PPO_HANDS=400 \
HEURISTIC_POPULATION=16 \
HEURISTIC_ELITE=4 \
HEURISTIC_MATCHES_PER_GENERATION=384 \
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
ALLOW_SMOKE=1 \
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
poetry run python tools/strong_mocks/self_train_heuristic.py --run-id smoke --generations 1 --population 4 --elite 2 --matches-per-generation 2 --hands 12 --seed 22 --allow-smoke --json
```

This writes to `runs/fullhouse_self_training/<run-id>/` by default, including
`summary.json`, `metrics.jsonl`, `ev_progress.svg`, and `generated/`.

For larger parallel heuristic self-training, use:

```bash
WORKERS=0 PARALLEL_BACKEND=process scripts/train_heuristics_selfplay.sh
```

The wrapper defaults to 16 generations, 384 matches/generation, 400 hands per
match, and 256 held-out strong-screen validation seeds. It also writes
`baseline_validation.json` in the same run directory.

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

To package tuned defaults without editing `bots/heuristic/bot.py`, pass:

```bash
poetry run python tools/harden_submission.py --env-file runs/heuristic_full_space_tuning/<run-id>/best_env.json --json
```

For threshold tuning comparisons, run:

```bash
poetry run python tools/tune_heuristic_thresholds.py --workers 0 --parallel-backend process --summary-only --json
```

For threshold tuning summaries, run:

```bash
poetry run python tools/tune_heuristic_thresholds.py --seed-count 128 --workers 0 --parallel-backend process --summary-only --json
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
