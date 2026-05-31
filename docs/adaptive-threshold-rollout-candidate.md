# Adaptive Threshold Rollout Candidate

This document describes `bots/adaptive_threshold_rollout_submission/bot.py`.
It is an add-only experimental submission candidate and does not modify the
current heuristic optimizer, heuristic bot, tuned parameter files, or training
artifacts.

## Purpose

The candidate applies the threshold-machine exploit suggested in the external
analysis:

- estimate our current equity;
- estimate whether opponents behave like pot-odds threshold callers;
- generate bet sizes that sit just below target call thresholds for value;
- generate bluff sizes just above fold thresholds only when the EV clears a
  break-even margin;
- avoid paying large bets too lightly because equity-driven aggression is often
  under-bluffed.

This is intentionally separate from `bots/rollout_search_submission` and
`bots/threshold_rollout_submission` so those previous candidates remain
available for comparison.

## Equity Core

The bot embeds the tuned rollout-search parameters from
`runs/rollout_search_tuning/rollout-final-20260531-004635/best_params.json`.
It uses `eval7` Monte Carlo equity with:

- `samples = 1024`;
- a `1.50s` internal deadline;
- at least `128` rollout samples before time checks can stop the loop;
- deterministic state-derived RNG seeds.

Preflop also uses the Monte Carlo rollout when cards are valid. The fixed
preflop hand-strength formula is only a fallback for malformed or missing card
states.

## Mechanical Opponent Weight

The runner exposes `match_action_log` as actions only, not full decision states
or revealed cards at prior showdowns. Because of that, the bot cannot literally
fit `q - T(B)` from past hands inside `decide()`.

Instead it uses an action-only proxy:

- hero raise followed by opponent fold: increases threshold-mechanical weight;
- hero raise followed by opponent call: modestly increases it;
- hero raise followed by opponent raise/all-in: decreases it;
- heads-up and river spots slightly increase exploit weight;
- multiway spots decrease exploit weight.

The resulting weight is clamped to `[0.18, 0.86]`. Low weight falls back toward
the tuned rollout policy; high weight allows threshold-derived bet sizing.

## Bet Sizing

For a bet fraction `b = B / P`, the call threshold is:

```text
T(b) = b / (1 + 2b)
```

For a target perceived opponent equity `q`, the indifference bet fraction is:

```text
b*(q) = q / (1 - 2q)
```

The bot builds candidate sizes from:

- off-tree probes: `0.27, 0.41, 0.63, 0.88, 1.27, 1.75, 2.60` pot;
- the tuned rollout open-size formula;
- threshold sizes from estimated opponent equity quantiles;
- a Monte Carlo noise buffer of roughly `2.5 * sqrt(q * (1 - q) / samples)`.

For value, target thresholds are moved below the opponent-equity quantile.
For bluffs, target thresholds are moved above the quantile. Candidate sizes are
capped by effective stack divided by pot.

## EV Selection

For each candidate bet size, the bot estimates a soft call probability by
blending:

- a mechanical threshold model using `q - T(b)`;
- a generic size-sensitive call model.

The blend weight is the mechanical opponent weight. The bet EV approximation is:

```text
EV = fold_prob * pot
     + call_prob * (equity_when_called * (pot + 2 * bet) - bet)
```

`equity_when_called` is discounted because calls are range-strengthening. Value
bets must beat checking by a hurdle that grows when the mechanical weight is
low. Bluffs must clear both direct bluff break-even and an additional safety
margin.

## Facing Bets

Facing a bet, the bot starts from the tuned rollout `g` function and then adds
threshold-exploit caution:

- larger bets require extra equity;
- river bets require extra equity;
- multiway spots require extra equity;
- raises are reserved for very high-equity hands facing small bets.

The call threshold uses the runner state convention:

```text
pot_odds = amount_owed / (state["pot"] + amount_owed)
```

The state pot already includes the opponent's bet, so this is equivalent to
`B / (P + 2B)` for a normal bet into a prior pot `P`.

## Current Caveat

This candidate is designed to test whether threshold exploitation helps enough
without destabilizing the broad rollout baseline. The previous static threshold
variant improved some mechanical-style opponents but performed badly against
strong heads-up rollout opponents. This adaptive version therefore uses the
threshold layer as a gated exploit, not as the default policy.

## Benchmark Command

Use the benchmark helper for larger screens. Non-JSON output streams one
completion line per suite and prints a final `tabulate` summary table sorted by
suite and mean result:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
poetry run python tools/benchmark_adaptive_threshold_rollout.py \
  --hands 200 \
  --seeds 16
```

Change the final table style with `--table-format`, for example
`--table-format simple_grid`. Use `--json` when another tool should consume the
raw result rows.

To run the current submission-ready heuristic zip on the exact same suite
matrix, use:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
poetry run python tools/benchmark_submission_heuristic.py \
  --hands 512 \
  --seeds 128
```

For a side-by-side comparison with rollout candidates:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
poetry run python tools/benchmark_submission_heuristic.py \
  --hands 512 \
  --seeds 128 \
  --candidate rollout=bots/rollout_search_submission \
  --candidate adaptive=bots/adaptive_threshold_rollout_submission \
  --candidate blend_rollout=bots/blend_rollout_heuristic
```
