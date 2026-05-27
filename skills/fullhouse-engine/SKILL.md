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
- Read `docs/strong-mock-self-training-pipeline.md` before changing strong mock opponents, mock training scripts, or self-training heuristic variants.
- Read `docs/heuristic-benchmark-results.md` before comparing new heuristic changes against the latest recorded benchmark run.
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

For bot changes, validate the edited bot directly and run at least one seeded local match against `bots/shark/bot.py`.

For heuristic bot benchmark passes, run:

```bash
poetry run python tools/evaluate_heuristic.py --json
```

For full benchmark summaries, run:

```bash
poetry run python tools/select_heuristic_config.py --preset final --config baseline --progress --json
```

For expanded mock-family screens, run:

```bash
poetry run python tools/select_heuristic_config.py --preset mock-family --config baseline --progress
```

For strong mock training and screens, use:

```bash
poetry run python tools/strong_mocks/train_imitation.py --samples 6000 --hidden 32 --seed 4242 --style balanced --output bots/strong_mocks/oracle_imitation/data/policy.npz --json
poetry run python tools/strong_mocks/train_mccfr.py --iterations 180 --batch-size 2048 --bucket-count 4096 --seed 5151 --output bots/strong_mocks/cfr_bucket/data/policy.npz --json
poetry run python tools/strong_mocks/train_ppo.py --backend auto --iterations 64 --batch-size 512 --hidden 32 --seed 6161 --output bots/strong_mocks/ppo_policy/data/policy.npz --json
poetry run python tools/select_heuristic_config.py --preset strong-screen --config baseline --workers 0 --parallel-backend process --progress
```

For multi-heuristic self-training, run a smoke first:

```bash
poetry run python tools/strong_mocks/self_train_heuristic.py --run-id smoke --generations 1 --population 4 --elite 2 --matches-per-generation 2 --hands 12 --seed 22 --generated-root /private/tmp/fullhouse_self_training/generated --result-root /private/tmp/fullhouse_self_training/results --json
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
poetry run python tools/evaluate_heuristic.py --suite sizing_6max --suite pressure_6max --suite mixed_stress_6max --seed-count 10 --summary-only --json
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
poetry run python tools/tune_heuristic_thresholds.py --seed-count 100 --summary-only --json
```

For risk-aware config ranking, run:

```bash
poetry run python tools/select_heuristic_config.py --preset candidate --progress
```

For weak-spot candidate checks, use at least a 10-seed candidate screen and a
30-seed gate before changing defaults. The latest tested weak-spot configs are
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
