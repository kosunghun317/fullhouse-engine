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

For heuristic submission packaging, run:

```bash
poetry run python tools/package_heuristic.py --json
```
