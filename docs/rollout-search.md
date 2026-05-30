# Rollout Search Strong Mock

Reviewed: 2026-05-31.

`bots/strong_mocks/rollout_search` is a benchmark-only opponent. It is not a
learned policy and it is not submitted. The wrapper calls
`decide_rollout(state, style="rollout_pressure")` in
`tools/strong_mocks/policies.py`.

## Equity Rollout

The bot estimates hero showdown equity with `eval7`:

1. Parse hero cards and current community cards.
2. Count active non-folded opponents, capped at 5 opponents.
3. Build a deck excluding hero and known board cards.
4. For each sample, shuffle the deck, deal each opponent two cards, complete
   the board to five cards, evaluate all hands, and give split-pot credit for
   ties.
5. Return `wins / samples`.

The default `rollout_pressure` sample count is 1024. `rollout_deep` also uses
1024 samples. Preflop is now real Monte Carlo too: with an empty board, the
same sampler deals all five community cards. The old preflop formula is only a
fallback if the state has missing or malformed hero cards.

The rollout RNG is deterministic for a public state. The seed is derived from
the style, hand id, seat, action-log length, street, pot, owed amount, hero
cards, and board cards. Repeated calls on the same state produce the same
decision.

## Default Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `samples` | `1024` | Rollouts for `rollout_pressure`. |
| `deep_samples` | `1024` | Rollouts for `rollout_deep`. |
| `f_min_equity` | `0.40` | Minimum equity before raising when checking is available. |
| `f_center` | `0.61` | Equity midpoint of the check-spot sizing curve. |
| `f_scale` | `0.075` | Smoothness of the check-spot sizing curve. |
| `f_min_raise` | `0.48` | Smallest pot fraction used by `f`. |
| `f_max_raise` | `1.05` | Largest pot fraction used by `f`. |
| `g_call_edge_base` | `0.06` | Base required edge over pot odds when facing a bet. |
| `g_call_edge_active` | `0.035` | Extra required edge per active player beyond heads-up. |
| `g_pressure_discount` | `0.02` | Edge discount in `rollout_pressure`. |
| `g_thin_call_edge` | `0.01` | Thin-call edge floor in `rollout_pressure`. |
| `g_raise_edge` | `0.16` | Minimum edge over pot odds before raising over a bet. |
| `g_raise_equity` | `0.86` | Minimum absolute equity before raising over a bet. |
| `g_raise_max_owed_pot` | `0.22` | Maximum `owed / pot` ratio that still permits raising. |
| `g_raise_center` | `0.87` | Equity midpoint of the facing-bet raise sizing curve. |
| `g_raise_scale` | `0.045` | Smoothness of the facing-bet raise sizing curve. |
| `g_min_raise` | `0.80` | Smallest pot fraction used by `g` when it raises. |
| `g_max_raise` | `1.20` | Largest pot fraction used by `g` when it raises. |

Parameter coercion clamps sample counts to at least 1, sizing scales to at
least `0.005`, and raise fractions into `[0.10, 2.50]` while preserving
`max_raise >= min_raise`.

## Function f: Checking Is Available

When the bot may check, it computes:

```text
if equity < f_min_equity:
    check
else:
    w = sigmoid((equity - f_center) / f_scale)
    raise_fraction = f_min_raise + w * (f_max_raise - f_min_raise)
```

`raise_fraction` is passed to `raise_to_fraction()`, which targets
`current_bet + int(pot * raise_fraction)`, then clamps to the legal minimum
raise and the bot's available stack. If no raise is selected, the bot checks.

## Function g: Facing A Bet

When facing a bet, the bot computes:

```text
pot_odds = owed / (pot + owed)
edge = equity - pot_odds
call_edge = g_call_edge_base + g_call_edge_active * max(0, active_players - 2)
if style == "rollout_pressure":
    call_edge -= g_pressure_discount
```

Then:

```text
if edge < call_edge:
    if style == "rollout_pressure" and edge > g_thin_call_edge:
        call
    else:
        fold
elif edge < g_raise_edge or equity < g_raise_equity or owed / pot > g_raise_max_owed_pot:
    call
else:
    w = sigmoid((equity - g_raise_center) / g_raise_scale)
    raise_fraction = g_min_raise + w * (g_max_raise - g_min_raise)
    raise
```

The same legal action sanitizer applies to raises. A large raise can become
all-in if the stack is short relative to the requested size.

## Tuning f And g

Use the dedicated CEM/racing tuner:

```bash
poetry run python tools/strong_mocks/tune_rollout_search.py \
  --generations 4 \
  --population 12 \
  --elite 3 \
  --stages 4:120:0.5,12:240:0.5 \
  --workers 0 \
  --parallel-backend process \
  --progress \
  --json
```

The tuner searches the 16 f/g parameters above with a diagonal Gaussian. It
uses common suite/seed blocks inside each stage, keeps only the top fraction of
candidates for later stages, and writes generated candidate wrappers plus
`best_params.json` under `runs/rollout_search_tuning/<run-id>/`.

Before trusting a result, the tuner measures representative preflop, flop,
turn, and river decisions and exits if the maximum 1024-sample decision time is
above 2 seconds. A local smoke run on 2026-05-31 measured a worst default
decision of `0.199055s`; treat this as machine-specific, not a portable
guarantee.

## 24-Hour Submission Workflow

The full 6-generation, 24-candidate full-space heuristic tune is estimated at
roughly three days on the current 8-core laptop. For a 24-hour deadline, use:

```bash
TRAIN_STRONG_MOCKS=0 \
TUNE_PROFILE=deadline-24h \
SUBMISSION_PROFILE=deadline-24h \
WORKERS=0 \
PARALLEL_BACKEND=process \
scripts/build_tuned_submission.sh
```

This keeps the full-space tuner above its serious-run minimums but reduces it
to 3 generations, 16 candidates, and 4 elites. The stage sizes remain
`128:400:0.5,256:400:0.25`, so each candidate still gets 512 suite/seed tasks
in the first stage and finalists still get 1024 tasks in the final stage.

The deadline submission profile skips the broad post-tuning gates and runs a
final direct rollout-search gate at the end:

```text
tools/select_heuristic_config.py \
  --suite heads_up_strong_rollout \
  --bot-override rollout_search=<rollout-run>/best_bot \
  --seed-count 512
```

When using `scripts/build_tuned_submission.sh`, inspect
`runs/submission_pipeline/<build-run-id>-submission/final_rollout_gate.json` to
decide whether the tuned heuristic is good enough against rollout search before
submitting the generated zip.

Use `TRAIN_STRONG_MOCKS=0` when the strong mocks already exist locally. If all
strong mocks must be retrained first, budget additional time outside this
24-hour tuning estimate.

Full command order for optimizing both rollout search and heuristics:

```bash
ROLLOUT_RUN_ID=rollout-final-$(date +%Y%m%d-%H%M%S) && \
BUILD_RUN_ID=tuned-final-$(date +%Y%m%d-%H%M%S) && \
poetry run python tools/strong_mocks/tune_rollout_search.py \
  --run-id "$ROLLOUT_RUN_ID" \
  --generations 4 \
  --population 12 \
  --elite 3 \
  --stages 4:120:0.5,12:240:0.5 \
  --workers 0 \
  --parallel-backend process \
  --progress \
  --json && \
TRAIN_STRONG_MOCKS=0 \
RUN_ID="$BUILD_RUN_ID" \
TUNE_PROFILE=deadline-24h \
SUBMISSION_PROFILE=deadline-24h \
ROLLOUT_FINAL_BOT_PATH="runs/rollout_search_tuning/${ROLLOUT_RUN_ID}/best_bot" \
WORKERS=0 \
PARALLEL_BACKEND=process \
scripts/build_tuned_submission.sh
```

This writes the tuned rollout wrapper first, then tunes and packages the
heuristic, then compares the final tuned heuristic against the tuned rollout
wrapper in
`runs/submission_pipeline/${BUILD_RUN_ID}-submission/final_rollout_gate.json`,
and finally leaves the submission zip at `dist/heuristic_bot.zip`.
