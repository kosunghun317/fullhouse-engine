# Heuristic Benchmark Results

Reviewed: 2026-05-28.

This document is now the active benchmark protocol and result index. Historical
performance tables were removed from active guidance because they were easy to
misread as current performance.

Repo-local run artifacts live under `runs/`, which is git-ignored. Use those
JSON summaries and plots for exact numeric history.

## Current Source Of Truth

| Question | Source |
| --- | --- |
| What command should I run for a benchmark? | This file. |
| How does training and coevolution work? | `docs/training-pipelines.md`. |
| How does the heuristic bot decide? | `docs/heuristic-bot-logic.md`. |
| How does the PPO strong mock decide? | `docs/ppo-bot-logic.md`. |
| What exact numbers came from a run? | The matching `runs/**/summary.json`, `metrics.jsonl`, or saved JSON output. |

## Benchmark Principle

Use enough samples to make the result actionable. Tiny smoke tests only prove
that the harness runs.

For promotion decisions, prefer:

```text
400 hands per match
64-128 held-out seeds for focused screens
100+ runs per major configuration when time allows
```

## Metrics

Track at least:

| Metric | Why it matters |
| --- | --- |
| `mean_delta` | Primary chip-EV signal. |
| `median_delta` | Robustness against one huge outlier. |
| `stdev_delta` | Volatility. |
| `min_delta` | Worst-case risk. |
| `bust_count` / `bust_rate` | Tournament survival risk. |
| `positive_runs` | How often the config wins chips. |
| `error_count` | Reliability and invalid-action regressions. |

Risk-adjusted ranking should penalize variance, worst-case losses, busts, and
bot errors. Raw mean delta alone is not enough.

## Promotion Flow

```mermaid
graph TD
    Candidate["candidate config or logic change"] --> Smoke["smoke benchmark"]
    Smoke --> Focused["focused held-out screen"]
    Focused --> Paired["paired incumbent comparison"]
    Paired --> Final["final promotion matrix"]
    Final --> Gate{"better risk-adjusted result?"}
    Gate -->|yes| Promote["promote intentionally"]
    Gate -->|no| Reject["keep incumbent"]
```

## Core Commands

Run-and-forget submission pipeline:

```bash
WORKERS=0 PARALLEL_BACKEND=process scripts/run_submission_pipeline.sh
```

This writes all reports to `runs/submission_pipeline/<run-id>/` and rebuilds
the upload zip at `dist/heuristic_bot.zip`.

Focused selector run:

```bash
poetry run python tools/select_heuristic_config.py \
  --preset promotion \
  --config baseline \
  --progress \
  --json
```

Final selector run:

```bash
poetry run python tools/select_heuristic_config.py \
  --preset final \
  --config baseline \
  --progress \
  --json
```

Direct evaluator:

```bash
poetry run python tools/evaluate_heuristic.py \
  --preset final \
  --hands 400 \
  --seeds 100 \
  --json
```

Strong-mock focused screen:

```bash
poetry run python tools/select_heuristic_config.py \
  --preset strong-screen \
  --config baseline \
  --progress \
  --json
```

Paired default-promotion gate:

```bash
poetry run python tools/paired_heuristic_gate.py \
  --incumbent baseline \
  --candidate CANDIDATE_CONFIG \
  --preset promotion \
  --workers 0 \
  --parallel-backend process \
  --progress \
  --json
```

Use the paired gate when deciding whether a candidate should replace or
disable an active default. It compares identical suite/seed pairs and reports
mean, median, 10th-percentile, win rate, bust, and error deltas.

## What To Record For New Runs

When a run matters, write a short dated note in this file with:

- command,
- run artifact path under `runs/`,
- candidate/config names,
- hand count and seed count,
- mean/median/stdev/min delta,
- bust and error counts,
- decision: promote, reject, or rerun larger.

Do not paste large historical tables here. Keep exact bulk results in the run
artifact directory.

## Current Active Notes

- The active competition bot is `bots/heuristic`.
- 2026-05-28 paired smoke for targeted profile selection kept the active
  default enabled. Command:
  `poetry run python tools/paired_heuristic_gate.py --incumbent baseline --candidate profile-targeting-off --suite reference_6max --suite mixed_stress_6max --suite mock_adaptive_6max --seeds 9101,9102,9103 --hands 120 --workers 0 --parallel-backend process --json`.
  Result over 9 paired runs: disabling targeting had `mean_diff=-7530.22`,
  `median_diff=-5019`, `p10_diff=-28943.8`, `win_rate=0.3333`, and 2 extra
  busts, so it was not promotable.
- Strong mocks are benchmark opponents only.
- PPO mock training previously had unstable learning due rollout/inference and
  call-off issues; see `docs/ppo-bot-logic.md` for the corrected runtime and
  training assumptions.
- Any future default change should be judged against a fresh final matrix, not
  against historical tables.
