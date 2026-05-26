# Fullhouse Engine Project Overview

Reviewed: 2026-05-26.

## What This Repo Is

`fullhouse-engine` is the public local development repo for the Fullhouse Hackathon, a No-Limit Texas Hold'em poker bot competition scheduled for 2026-06-01 through 2026-06-05 in London.

Participants submit a Python bot that exposes exactly one decision function:

```python
def decide(game_state: dict) -> dict:
    return {"action": "call"}
```

The engine calls `decide()` once whenever the bot must act. The bot receives public table state, its own hole cards, board cards, pot and betting state, stack state, and action history. It returns one action dict.

## Main Components

- `engine/game.py`: No-Limit Hold'em hand engine, betting rules, side pots, showdown, and action-state serialization.
- `engine/tournament.py`: Swiss pairing, standings, and finalist selection helpers.
- `sandbox/validator.py`: Submission validator for static checks, package limits, action shape, edge cases, and timeout behavior.
- `sandbox/runner.py`: Runtime process that loads a submitted bot and exchanges newline-delimited JSON actions.
- `sandbox/match.py`: Local match orchestrator for 2-9 bots, with optional Docker sandboxing.
- `sandbox/Dockerfile`: Production-like bot container using Python 3.10 and sandbox-approved libraries.
- `bots/`: Reference bots and starter template.
- `demo.py`: Flask demo UI showing local reference-bot matches.
- `tests/`: Engine unit tests.

## Bot Contract

Valid returned actions:

```python
{"action": "fold"}
{"action": "check"}
{"action": "call"}
{"action": "raise", "amount": 1200}
{"action": "all_in"}
```

Important action semantics:

- `amount` on a raise is the total bet amount, not the incremental raise size.
- Invalid or missing actions default to fold.
- Checking while facing a bet is converted to a call by the engine.
- Raises below the legal minimum are snapped to the minimum legal raise.
- If the snapped raise would commit the stack, the engine treats it as all-in.

## Submission Formats

Supported submissions:

- `bot.py`: single-file bot.
- Bot directory: root `bot.py` plus optional `data/`.
- `bot.zip`: archive with root `bot.py` plus optional `data/`.

Package limits enforced by `sandbox/validator.py`:

- Total submission: 250 MB.
- `data/`: 200 MB.
- `bot.py`: 5 MB.
- `data/` must not contain `.py` files.
- Symlinks are rejected.
- Zip paths must not be absolute or path-traversing.

## Competition Format

From the repo README:

- 2026-06-01: online qualifier.
- Swiss-system tournament, 400 hands per match, usually 6-bot tables.
- Ranking is by cumulative chip delta.
- Top 64 advance.
- 2026-06-02: patch window after qualifier hand histories are available.
- 2026-06-05: finals night at UCL East, single-elimination bracket.
- Submission deadline: 2026-05-31 23:59 UTC.

The library-request deadline in `CONTRIBUTING.md` is 2026-05-25 23:59 UTC, which is already past as of this review date.

## Frozen Areas

`CONTRIBUTING.md` says these files are frozen before the event and should not be changed casually:

- `engine/game.py`
- `sandbox/runner.py`
- `db/schema.sql`

Acceptable contribution areas include bug reports, local demo UI improvements, additional reference bots, and documentation.
