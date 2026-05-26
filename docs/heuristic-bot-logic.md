# Heuristic Bot Logic

Reviewed: 2026-05-26.

This document explains the current logic in `bots/heuristic/bot.py` and records improvement directions for future tuning.

## High-Level Design

The bot is an interpretable population-exploit bot. It is not a solver and does not try to approximate a full GTO strategy.

Its core idea:

1. Play a conservative tight-aggressive baseline.
2. Estimate hand strength fast enough to stay far below the 2-second action limit.
3. Track public opponent actions in memory.
4. Classify opponents into simple behavioral types.
5. Adjust call thresholds, bluff frequency, and bet sizing by table profile.
6. Route every final decision through one action sanitizer so illegal outputs are avoided.

The submitted bot is intentionally a single file:

```text
bots/heuristic/bot.py
```

That keeps the production submission close to the tournament's simplest valid format: one root `bot.py`.

## Runtime Contract

The only public entrypoint is:

```python
def decide(game_state):
    ...
```

The current implementation:

- Records `started_at = time.perf_counter()` for internal budget control.
- Returns `{"action": "check"}` for warmup states.
- Updates in-memory opponent statistics from `match_action_log`.
- Uses `_preflop_policy()` before the flop.
- Uses `_estimate_equity()` plus `_postflop_policy()` after the flop.
- Sanitizes every policy intent through `_sanitize_action()`.
- Catches all internal exceptions and returns a conservative fallback action.

This makes reliability a first-class strategy feature: crashes, malformed returns, and timeouts are often worse than a slightly conservative fold.

## Constants And Tunables

Important constants:

| Name | Current Value | Purpose |
| --- | ---: | --- |
| `BIG_BLIND` | `100` | Fullhouse blind size. |
| `DECIDE_BUDGET_S` | `1.65` | Internal soft limit below the engine's 2-second timeout. |
| `EQUITY_BUDGET_S` | `0.20` | Max time allocated to one Monte Carlo equity estimate. |
| `EQUITY_CACHE_MAX` | `4096` | Equity cache reset threshold. |
| `SEEN_ACTIONS_MAX` | `1200` | Action de-duplication reset threshold. |

Local tuning env vars:

| Env Var | Default | Purpose |
| --- | ---: | --- |
| `HEURISTIC_RNG_SEED` | unset | Optional deterministic bot-local RNG seed for evaluation. |
| `HEURISTIC_CALL_MARGIN_BASE` | `0.075` | Base call margin over raw pot odds. |
| `HEURISTIC_CALL_MARGIN_MULTIWAY` | `0.035` | Extra call margin per extra opponent. |
| `HEURISTIC_RISK_REQ_LOW` | `0.76` | Required equity when risking at least 28% of stack. |
| `HEURISTIC_RISK_REQ_MID` | `0.86` | Required equity when risking at least 45% of stack. |
| `HEURISTIC_RISK_REQ_HIGH` | `0.92` | Required equity when risking at least 70% of stack. |
| `HEURISTIC_VALUE_THRESHOLD_BASE` | `0.66` | Base postflop value-bet threshold. |
| `HEURISTIC_THIN_VALUE_BASE` | `0.59` | Base thin-value threshold. |
| `HEURISTIC_DRY_BLUFF_PROB` | `0.45` | Mixed bluff probability on dry boards. |
| `HEURISTIC_WET_BLUFF_PROB` | `0.25` | Mixed semi-bluff probability on wet boards. |

Hand-class groups:

- `ULTRA_PREMIUM_CLASSES`: `AA`, `KK`
- `PREMIUM_CLASSES`: `AA`, `KK`, `QQ`, `JJ`, `AKs`, `AKo`
- `STRONG_CLASSES`: medium-high pairs and broadways like `TT`, `99`, `AQ`, `AJs`, `KQs`
- `SPECULATIVE_CLASSES`: small pairs, suited aces, and suited connectors

These classes drive preflop decisions and are deliberately easier to reason about than a learned model.

## In-Memory State

The bot uses module-level memory:

```python
OPPONENTS = {}
SEEN_ACTIONS = set()
EQUITY_CACHE = {}
```

`OPPONENTS` stores per-`bot_id` public action statistics. It persists across hands in one match because the bot process stays alive.

`SEEN_ACTIONS` prevents double-counting entries from the rolling `match_action_log`.

`EQUITY_CACHE` memoizes Monte Carlo equity estimates for repeated states.

No file writes, databases, network calls, subprocesses, threads, async tasks, pickle, or dynamic imports are used.

For local evaluation only, the bot supports deterministic randomness through:

```bash
HEURISTIC_RNG_SEED=123 poetry run python tools/evaluate_heuristic.py --json
```

If the env var is unset, the bot uses normal process-local randomness.

## Card And Hand Helpers

`_hand_class(cards)` converts two hole cards into a standard 169-class preflop label:

- Pair: `AA`, `77`, `22`
- Suited non-pair: `AKs`, `T9s`
- Offsuit non-pair: `AKo`, `QJo`

`_preflop_score(cards)` gives a rough numeric hand score:

- Premium classes return `92`.
- Strong classes return `76`.
- Speculative classes start at `55`.
- Other hands are scored from high-card value, low-card value, suitedness, pair status, connectedness, ace-high bonus, and broadway-card bonus.

The score is not a true equity number. It is a compact ordering for preflop action selection.

## State Parsing

The bot derives several helper facts from the `game_state`:

- `_players_in_hand(state)`: players not folded.
- `_active_opponent_count(state)`: non-hero players still in the hand.
- `_pot_odds(state)`: `amount_owed / (pot + amount_owed)`.
- `_facing_raise_preflop(state)`: whether current preflop bet exceeds the big blind.
- `_effective_stack(state)`: `your_stack + your_bet_this_street`.

`_position_bucket(state)` estimates position as `early`, `middle`, or `late`.

For 3+ player hands, it finds the small blind in `action_log`, infers the dealer as the seat before the small blind, then measures hero's relative seat. This is only an approximation, but it is better than treating raw seat number as position.

Heads-up is simplified:

- Preflop is treated as early.
- Postflop is treated as late.

## Opponent Model

The bot only uses public action history from `match_action_log`.

For each opponent, it tracks:

- `actions`
- `raises`
- `calls`
- `folds`
- `checks`
- `all_ins`
- `raise_total`
- `raise_count`

Actions ignored:

- `small_blind`
- `big_blind`
- missing actions

Rates use smoothing:

```python
(count + prior) / (actions + prior_mass)
```

Current behavior profiles:

| Profile | Condition |
| --- | --- |
| `unknown` | Fewer than 10 recorded actions or no clear pattern. |
| `maniac` | Raise rate above `0.33` or all-in rate above `0.10`. |
| `station` | Call rate above `0.42` and fold rate below `0.25`. |
| `nit` | Fold rate above `0.42` and raise rate below `0.18`. |
| `abc` | Raise rate below `0.20` and call rate below `0.34`. |

`_table_profile(state)` summarizes the remaining table. It returns a profile when that profile appears in at least half of visible non-hero, non-folded opponents. Otherwise it returns `mixed`.

`_fold_pressure(state)` maps table profile to an approximate fold likelihood:

| Table Profile | Fold Pressure |
| --- | ---: |
| `nit` | `0.72` |
| `abc` | `0.58` |
| `station` | `0.22` |
| `maniac` | `0.30` |
| other/mixed | `0.45` |

This value controls selective bluffing and semi-bluffing.

## Equity Engine

Preflop:

- The bot does not run Monte Carlo preflop.
- It turns `_preflop_score()` into a rough equity-like number.
- It applies a multiway penalty of `0.055` per extra opponent.
- It clamps the final value to `[0.05, 0.86]`.

Postflop:

- Uses `eval7`.
- Samples unknown opponent hands and future board runouts.
- Compares hero's evaluated hand against sampled opponent hands.
- Splits ties as fractional wins.
- Caches by `(sorted_hole_cards, board_cards, opponent_count)`.

Sampling schedule:

| Board Cards | Samples |
| ---: | ---: |
| Flop, 3 cards | `520` |
| Turn, 4 cards | `700` |
| River, 5 cards | `900` |

If there are 4+ opponents, sample count is reduced to 75%.

The estimate is also limited by `EQUITY_BUDGET_S = 0.20`, and the remaining `DECIDE_BUDGET_S` is checked before sampling. This keeps the bot well below the engine timeout in normal operation.

## Board Texture

`_board_texture(state)` classifies postflop boards:

- `wet`: three or more cards of a suit, or multiple close rank connections.
- `dry`: paired board or low suit/connection coordination.
- `medium`: everything in between.
- `none`: fewer than three board cards.

Texture adjusts value betting and bluffing:

- Wet boards require slightly stronger value thresholds.
- Dry boards are better candidates for continuation bets and small bluffs.

## Bet Sizing

There are two sizing helpers:

`_raise_to_preflop(state, big_blinds)`:

- Used for preflop open raises and selected premium reraises.
- Targets a big-blind multiple, while respecting `min_raise_to`.
- Also ensures the target is at least `2.4x` the current bet when reraising.

`_raise_to_fraction(state, fraction)`:

- Used for postflop betting.
- Adds a fraction of the current pot to the current bet.
- Respects `min_raise_to`.
- Caps the amount at the hero's stack plus current street investment.

Current sizing is intentionally simple:

- Preflop opens are usually around `3.0x` to `3.4x`.
- Postflop value bets are usually around half-pot to three-quarter-pot.
- Bluff/semi-bluff bets are smaller and only used against fold-prone tables.

## Action Sanitizer

`_sanitize_action(state, intent)` is the only place where policy intent becomes a final action dict.

It enforces:

- `check` becomes `call` if checking is not legal.
- `fold` becomes `check` if checking is free.
- `raise` amounts are converted to legal totals.
- Raises that cannot meet `min_raise_to` become all-in or call/check as appropriate.
- Unknown intents fall back to check, cheap call, or fold.

This is important because the policy functions can stay expressive without each branch duplicating legality rules.

## Preflop Policy

The preflop policy computes:

- Hand class.
- Preflop score.
- Position bucket.
- Table profile.
- Pot odds.
- Effective stack.
- Whether the bot is facing a raise.
- Whether the game is heads-up.

Position bonus:

- Late: `+8`
- Middle: `+3`
- Early: `+0`
- Extra `+5` against nits.
- Penalty `-8` when facing a raise from a maniac.

Facing a raise:

- Heads-up vs maniac: calls wider with medium-plus hands if stack risk is controlled.
- `AA` and `KK`: may reraise when the current bet is not too large and risk is low.
- Premium/very strong hands: prefer calls over stack-off reraises when risk is reasonable.
- Marginal late-position hands: call only at good odds.
- Most other hands fold.

Unopened or cheap pots:

- Premium and very high adjusted scores open-raise to about `3.4 BB`.
- Heads-up vs maniac: medium-plus hands open to about `2.8 BB`.
- Strong hands open to about `3.0 BB`.
- Medium late/middle hands may check or cheap-call.
- Weak hands check if free, otherwise fold unless the call is tiny and the score is acceptable.

The most important deliberate choice: the bot avoids frequent preflop all-in confrontations. It gives up some theoretical EV with premiums to reduce tournament-damaging variance.

## Postflop Policy

Postflop decisions use:

- Estimated equity.
- Pot odds.
- Opponent count.
- Table profile.
- Board texture.
- Whether checking is legal.
- Risk guard.

Value thresholds:

```text
value_threshold = 0.66 + 0.055 * extra_opponents
thin_value      = 0.59 + 0.045 * extra_opponents
```

Adjustments:

- Stations reduce value thresholds because they call too much.
- Maniacs slightly reduce value thresholds.
- Wet boards increase value threshold by `0.025`.

When checking is legal:

- Strong equity value-bets.
- Thin value-bets against stations and maniacs.
- Medium equity may bluff dry boards if fold pressure is high.
- Wet-board semi-bluffs are rare and require high fold pressure.
- Otherwise the bot checks.

When facing a bet:

1. Compute a call margin.
2. Apply the risk guard.
3. Call if `equity >= pot_odds + margin`.
4. Rarely raise only with very high equity and low current risk.
5. Make very cheap defensive calls with at least minimal equity.
6. Otherwise fold.

## Call Margin

`_call_margin(state, profile)` adds conservatism to raw pot odds.

Baseline:

```text
0.075 + 0.035 * extra_opponents
```

Additional margin:

- Early position: `+0.025`
- Owed amount larger than 70% of pot: `+0.04`
- Nit or ABC profile: `+0.03`

Reduced margin:

- Maniac: `-0.055`
- Station: `-0.015`

The result is clamped to at least `0.015`.

## Risk Guard

`_passes_risk_guard(state, equity, profile)` protects the bot from overcommitting based on noisy Monte Carlo estimates.

It computes:

```text
risk = amount_owed / effective_stack
```

Required equity:

| Risk | Required Equity |
| ---: | ---: |
| `>= 0.70` | `0.92` |
| `>= 0.45` | `0.86` |
| `>= 0.28` | `0.76` |

Adjustments:

- 3+ opponents: `+0.04`
- Maniac: `-0.02`

This is intentionally conservative. The tournament ranking is chip delta over finite hands, and busting can be much worse than folding a marginally profitable spot.

## Fallback Policy

If anything unexpected happens inside `decide()`, the bot returns:

- `check` if free.
- `call` only if the price is tiny: max of one big blind or 8% of pot.
- otherwise `fold`.

This fallback is deliberately simple and legal.

## Current Strengths

- Validator-clean.
- No forbidden imports or runtime behaviors.
- One-file submission shape.
- Bounded equity calculation.
- In-memory opponent model.
- Conservative stack-risk handling.
- Strong benchmark performance against current reference and mutant 6-max suites.
- Clear structure for future tuning without splitting the bot into packages.

## Current Weaknesses

1. Opponent modeling is action-only.

   The bot does not infer whether opponents showed down weak or strong hands because normal action requests do not include a clean hand-complete callback.

2. Preflop strategy is hand-authored.

   The 169-class policy is a rough chart, not a solved range. Some hands are likely overplayed or underplayed by position.

3. Equity estimates assume random-ish opponent hole cards.

   Opponent ranges are not narrowed from preflop action, bet sizing, or street progression.

4. Heads-up anti-maniac behavior is still high variance.

   Benchmark results were positive on average but had one busted run against the aggressor profile.

5. Board texture is crude.

   It recognizes suited and connected boards, but not made-hand categories, blockers, nut advantage, pair blockers, or draw quality.

6. Bet sizing is coarse.

   It uses fixed fractions and big-blind multiples. It does not yet optimize sizes against specific call/fold thresholds.

7. Benchmark randomness must be interpreted carefully.

   The bot can use `HEURISTIC_RNG_SEED` for reproducible local evaluation, but some benchmark opponents also use their own random choices. Large benchmark samples are still more reliable than one seed.

## Future Improvements

### 1. Preflop Table Upgrade

Replace the current score heuristic with an explicit 169-hand matrix by:

- Position bucket.
- Unopened/limped/raised pot.
- Heads-up vs 6-max.
- Stack depth.
- Known opponent profile.

This would make the bot easier to tune and reduce accidental over/under-play.

### 2. Range-Aware Equity

Current Monte Carlo samples opponents from all unknown cards. A better version would weight opponent hands by action:

- Tight raisers get stronger ranges.
- Limp/call stations get wider ranges.
- Maniacs get very wide raising ranges.
- Nits get narrow continuing ranges.

This can still be done with simple weighted sampling, without building a neural net.

### 3. Better Hand Category Detection

Add deterministic made-hand and draw features:

- Pair, two pair, trips, straight, flush.
- Overpair and top pair.
- Flush draws.
- Open-ended straight draws.
- Gutshots.
- Board-pair danger.
- Nut blockers.

Then use these features to adjust equity thresholds and bluff choices.

### 4. Smarter Opponent Model

Improve public-action stats:

- Track actions by street.
- Track aggression after checking.
- Track fold-to-flop-bet and fold-to-turn-bet.
- Track limp/fold and raise/fold behavior.
- Normalize raise sizes by pot and stack.
- Track table-level looseness separately from individual profile.

### 5. Hand-History Patch Workflow

After Day 1 hand histories are available, compute population leaks:

- Showdown hand strength after calls.
- Bluff frequency after large bets.
- Fold rate by bet size.
- Overcalling patterns.
- All-in thresholds.

Then patch priors and thresholds rather than rewriting the whole bot.

### 6. Safer Heads-Up Maniac Mode

The current heads-up anti-maniac branch widened preflop defense, but one benchmark run still busted.

Future changes:

- Cap postflop call risk even more in heads-up.
- Add a "protect lead" mode when ahead in stack.
- Prefer smaller value lines over large confrontations unless equity is extremely high.
- Identify aggressor's deterministic raise-size patterns and counter those specifically.

### 7. Deterministic Evaluation Mode

Implemented through `HEURISTIC_RNG_SEED`. Future work should use this in benchmark tooling when exact reproducibility is more important than sampling varied bot randomness.

### 8. Threshold Search Harness

Implemented as `tools/tune_heuristic_thresholds.py` with named env configurations. It currently compares baseline, conservative, value-heavy, and pressure settings across selected benchmark suites.

Future work can expand the named configurations or replace them with random/grid search across:

- Preflop open thresholds.
- Call margins.
- Risk guard required equities.
- Bluff frequencies.
- Bet fractions.

Score with mean chip delta, worst-run result, bust rate, and bot errors.

### 9. Submission Packaging Check

Before final upload:

- Validate `bots/heuristic/bot.py`.
- Run `poetry run python tools/package_heuristic.py --json`.
- Confirm no extra `.py` files are inside `data/`.
- Run the exact validator against the final submission path.

## Latest Benchmark Reference

See `docs/heuristic-benchmark-results.md` for the latest recorded test run. Any future logic change should be compared against that baseline before it is considered an improvement.
