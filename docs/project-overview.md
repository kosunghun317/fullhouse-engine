# Fullhouse Engine Project Overview

Reviewed: 2026-05-28.

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
- `training/fast_match.py`: Local-only no-limit in-process match runner for
  fast training batches and unrestricted benchmark opponents.
- `bots/`: Reference bots, starter template, heuristic bot, simple benchmark bots, and mock competitor bots.
- `bots/heuristic/`: Current competition bot, optional read-only `data/tables.npz`, and single-file submission entrypoint.
- `bots/mock_competitors/`: Local-only trained/handwritten benchmark opponents that approximate likely RL/NN/CFR/equity submissions.
- `bots/strong_mocks/`: Benchmark-only stronger opponents with trained MLP,
  original PPO, independent deep PPO, CFR-like table, rollout, and ensemble
  policies.
- `bots/self_training/`: Wrapper template for generated heuristic config variants used in local self-training matches.
- `tools/evaluate_heuristic.py`: Seeded benchmark harness for core, stress, and mock suites.
- `tools/select_heuristic_config.py`: Risk-aware config ranking and promotion-screen harness.
- `tools/strong_mocks/`: Shared feature/action abstractions, strong mock trainers, and evolutionary self-training pipeline.
- `tools/strong_mocks/train_real_policy.py`: Fullhouse in-process real
  self-play trainer for PPO-style and CFR+/bucket strong mock policies.
- `tools/strong_mocks/train_deep_ppo.py`: Independent deep PPO trainer with
  more hidden layers and a 14-arm action abstraction.
- `tools/strong_mocks/league_train.py`: League-style staged trainer with
  train/eval pool separation and promotion gates for strong mock policies.
- `tools/coevolve_training.py`: End-to-end alternating PPO/heuristic
  coevolution trainer with candidate gates and progress plotting.
- `tools/plot_training_progress.py`: JSONL-to-SVG plotter for training EV
  progress.
- `tools/parallel.py`: Offline-only process/thread worker helper for seeded benchmark and self-training runs.
- `scripts/train_opponents_real.sh`: Large real-training entrypoint for mock
  opponents.
- `scripts/train_e2e_coevolution.sh`: Single-command alternating PPO and
  heuristic fine-tuning loop.
- `scripts/train_heuristics_selfplay.sh`: Large self-play entrypoint for
  heuristic env-config evolution.
- `tools/package_heuristic.py`: Submission zip builder and validator wrapper for the heuristic bot.
- `tools/build_heuristic_tables.py`: Optional read-only data table builder.
- `tools/harden_submission.py`: Full heuristic submission hardening command for table rebuild, packaging, validation, zip inspection, and sanity matches.
- `tools/train_mock_numpy_policy.py`: Offline trainer for benchmark-only
  numpy-policy mock competitors.
- `demo.py`: Flask demo UI showing local reference-bot matches.
- `tests/`: Engine unit tests.

## Architecture Diagram

```mermaid
graph TD
    User["Developer / participant"] --> Tools["Local tooling"]
    User --> Submission["Submitted bot package"]

    subgraph Core["Fullhouse engine"]
        Game["engine/game.py - hand engine"]
        Tournament["engine/tournament.py - Swiss / standings"]
    end

    subgraph Sandbox["Sandbox runtime"]
        Validator["sandbox/validator.py - static + runtime checks"]
        Match["sandbox/match.py - multi-hand match"]
        Runner["sandbox/runner.py - bot subprocess protocol"]
        Docker["sandbox/Dockerfile - production-like container"]
    end

    subgraph TrainingRuntime["Offline training runtime"]
        FastMatch["training/fast_match.py - no-limit local runner"]
    end

    subgraph Bots["Bot directories"]
        Heuristic["bots/heuristic - competition bot"]
        Data["bots/heuristic/data/tables.npz - read-only lookup data"]
        Mock["bots/mock_competitors - local benchmark families"]
        Strong["bots/strong_mocks - trained / rollout opponents"]
        SelfTrain["bots/self_training - generated heuristic wrappers"]
    end

    subgraph Tooling["Tools and scripts"]
        Eval["tools/evaluate_heuristic.py - suite runner"]
        Select["tools/select_heuristic_config.py - risk-aware ranking"]
        Package["tools/package_heuristic.py - zip builder"]
        Harden["tools/harden_submission.py - submission hardening"]
        RealTrain["tools/strong_mocks/train_real_policy.py - real self-play trainer"]
        LeagueTrain["tools/strong_mocks/league_train.py - league trainer"]
        Scripts["scripts/train_*.sh - large training entrypoints"]
    end

    Tools --> Eval
    Tools --> Select
    Tools --> Package
    Tools --> Harden
    Tools --> RealTrain
    Tools --> LeagueTrain
    Tools --> Scripts

    Eval --> Match
    Select --> Match
    RealTrain --> FastMatch
    LeagueTrain --> FastMatch
    LeagueTrain --> RealTrain
    Scripts --> FastMatch
    Match --> Game
    Match --> Runner
    FastMatch --> Game
    FastMatch --> Heuristic
    FastMatch --> Mock
    FastMatch --> Strong
    Runner --> Heuristic
    Runner --> Mock
    Runner --> Strong
    Validator --> Heuristic
    Package --> Heuristic
    Package --> Data
    Harden --> Validator
    RealTrain --> Game
    RealTrain --> Strong
    LeagueTrain --> Strong
    Scripts --> RealTrain
    Scripts --> SelfTrain
    Submission --> Validator
    Submission --> Runner
```

## Runtime Flow

```mermaid
graph TD
    Match["sandbox/match.py"] --> StartBot["Start bot process"]
    StartBot --> Warmup["Send warmup state"]
    Warmup --> Runner["sandbox/runner.py"]
    Runner --> Bot["bot.py decide"]
    Bot --> Ignore["Warmup action ignored"]
    Ignore --> StartHand["PokerEngine start_hand"]
    StartHand --> State["action_request with match_action_log"]
    State --> RunnerState["Runner sends JSON state"]
    RunnerState --> BotDecision["bot.py returns action dict"]
    BotDecision --> Apply["PokerEngine apply_action"]
    Apply --> Continue{"More actions?"}
    Continue -->|yes| State
    Continue -->|no| Complete["hand_complete"]
    Complete --> Update["Update stacks and rolling log"]
    Update --> MoreHands{"More hands and at least two live stacks?"}
    MoreHands -->|yes| StartHand
    MoreHands -->|no| Result["chip_delta, final_stacks, bot_errors"]
```

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

## Current Heuristic Bot Status

The active competition branch is `heuristic`. The current promoted default is
documented in `docs/heuristic-bot-logic.md`. Use
`docs/heuristic-benchmark-results.md` for benchmark protocol before changing
bot defaults.

Training and benchmark docs are consolidated:

- `docs/training-pipelines.md`: canonical guide for real training, league
  training, self-training, fast matches, and coevolution.
- `docs/ppo-bot-logic.md`: runtime and training details for the PPO strong
  mock.
- `docs/heuristic-benchmark-results.md`: active benchmark protocol and result
  index. Exact numeric run history belongs in git-ignored `runs/` artifacts.

Strong mocks and generated heuristic variants are benchmark-only. Do not move
their helper imports into `bots/heuristic/bot.py`.
