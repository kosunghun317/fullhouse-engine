# Heuristic Bot Implementation Plan

Reviewed: 2026-05-26.

## Project Framing

Frame this bot as an interpretable, population-exploit poker bot for a resource-constrained 6-max No-Limit Hold'em tournament.

Do not frame it as a GTO poker AI. The practical edge is:

- Legal, crash-free, timeout-safe execution.
- Tight-aggressive default poker behavior.
- Fast equity estimates.
- Simple online opponent modeling from public action history.
- Bet sizing that exploits common weak bot patterns.

The strategy target is "good enough by default, ruthless against obvious mistakes."

## Implementation Shape

Primary submission path:

```text
bots/heuristic/
└── bot.py
```

Keep the production bot in one file because the simplest valid submission is `bot.py`. Use clear internal sections instead of introducing a package that later has to be flattened for submission.

Planned `bot.py` sections:

1. Constants and tunables.
2. Card and hand-class helpers.
3. State parsing helpers.
4. In-memory opponent model.
5. Preflop policy.
6. Equity engine.
7. Board texture helpers.
8. Postflop policy.
9. Bet sizing policy.
10. Action sanitizer.
11. `decide(game_state)`.

Design rule: every policy function returns an intent, and only `sanitize_action()` produces the final action dict. This prevents legal-action fixes from being scattered through the bot.

## State And Memory Design

Use module-level memory only:

```python
OPPONENTS = {}
SEEN_ACTIONS = set()
HAND_SNAPSHOTS = {}
```

Track only public information from `action_log`, `match_action_log`, and visible stack/pot state.

Per-opponent smoothed stats:

- `actions`
- `raises`
- `calls`
- `folds`
- `all_ins`
- normalized raise sizes
- aggression score
- looseness score

Classify opponents conservatively:

- `unknown`: not enough data.
- `maniac`: high raise/all-in rate.
- `station`: high call rate, low fold rate.
- `nit`: high fold rate, low raise rate.
- `abc`: low aggression, mostly honest large bets.

Use smoothing such as `(count + prior) / (total + prior_mass)` so three actions do not dominate strategy.

## Preflop Plan

Use a 169-class hand table:

- Premium: `AA`, `KK`, `QQ`, `JJ`, `AK`.
- Strong: `TT-88`, `AQ`, `AJs`, `KQs`.
- Speculative: small pairs, suited aces, suited connectors.
- Trash: everything else.

Position should be inferred from blinds and action order, not from raw seat number alone.

Preflop outputs:

- Open-raise strong enough hands.
- Tighten sharply against raises.
- Widen slightly in late position against nits.
- Avoid speculative calls when short-stacked or facing large raises.

## Equity Engine Plan

Use `eval7` Monte Carlo postflop with strict budgets:

- Cache by `(hole_class, exact_hole_cards, board, active_opponent_count)`.
- Use fewer samples when the action is obvious.
- Use more samples on close call/fold or value-bet decisions.
- Hard cap runtime per equity estimate so `decide()` remains well below 2 seconds.

Preflop should start from lookup values, not live Monte Carlo.

Do not start with a neural net. A small neural approximation can be added later only if profiling shows equity estimation is the bottleneck. If used, store weights as `numpy` arrays in read-only `data/` or hard-code plain arrays; avoid pickle.

## Decision Policy

Use a layered decision:

1. Parse state and update memory.
2. Compute pot odds and active opponent count.
3. Infer opponent/table profile.
4. Choose baseline action from preflop or postflop policy.
5. Adjust thresholds and sizing by opponent type.
6. Sanitize final action.

Default call rule:

```text
call if equity > pot_odds + margin
```

Margin increases when:

- Multiway.
- Out of position.
- Facing large aggression.
- Opponent is nit/abc and suddenly bets big.

Margin decreases when:

- Opponent is maniac.
- Pot odds are very favorable.
- Stack is short enough that folding forfeits too much equity.

Default betting rule:

```text
value bet strong equity
semi-bluff good draws against foldy opponents
bluff weak hands only against nits/folders
check/fold weak hands against stations and maniacs
```

## Exploit Rules

Against maniacs:

- Tighten preflop.
- Call/raise wider with real equity.
- Trap stronger hands.
- Bluff rarely.

Against stations:

- Value bet larger and thinner.
- Do not pure bluff.
- Avoid fancy check-raise lines unless holding strong equity.

Against nits:

- Steal blinds and small pots more often.
- Continuation-bet dry boards.
- Fold more when they suddenly raise large.

Against pot-odds bots:

- Bluff sizes should deny simple odds.
- Value sizes should invite dominated calls.
- Avoid giving cheap river calls when strong.

## Bet Sizing

Use a small fixed size menu:

- `min_raise_to`
- `1/3 pot`
- `1/2 pot`
- `2/3 pot`
- `pot`
- `all_in`

Convert all size intents through one function:

```python
size_raise_to(state, fraction_of_pot)
```

Then pass through `sanitize_action()`.

This avoids future refactors where each strategy branch calculates raise legality differently.

## Reliability Rules

- `decide()` must never print to stdout.
- Catch internal exceptions and return a legal fallback.
- Never import forbidden modules.
- No file writes.
- No subprocess, threading, async, network, dynamic import, `eval`, `exec`, or pickle.
- Use only allowed sandbox libraries in `bot.py`.
- Treat `state.get("type") == "warmup"` as a no-op safe response.

Fallback policy:

```python
if can_check: check
elif amount_owed is tiny relative to pot: call
else: fold
```

## Local Evaluation Plan

After a validating first bot exists, add a local harness outside the submitted bot:

```text
tools/evaluate_heuristic.py
```

The harness should run seeded 400-hand matches against:

- Fullhouse reference bots.
- Mutated always-call / overfold / overraise / minraise variants.
- Self-play copies of the heuristic bot.

Metrics:

- Mean chip delta.
- Median chip delta.
- Standard deviation.
- Bust rate.
- Worst table result.
- Validator pass/fail.
- Timeout/error count.

Only tune thresholds against aggregate results, not one lucky seed.

## Public Baselines To Study

Use these as references, not direct dependencies:

- Fullhouse reference bots: exact contest interface and the first benchmark.
- MIT Pokerbots resources: useful examples around `eval7` and poker-bot structure.
- `dberweger2017/deepcfr-texas-no-limit-holdem-6-players`: closest public 6-player NLHE Deep CFR style reference.
- `Nemandza82/g5-poker-bot`: ACPC six-player no-limit winner; useful for opponent-modeling/search ideas.
- `AI-Decision/DecisionHoldem`: heads-up NLHE depth-limited solving reference.
- Slumbot: public heads-up benchmark/source/API; useful conceptually, not as an in-game dependency.
- OpenSpiel: CFR and imperfect-information game concepts.
- RLCard and PokerRL: RL/CFR environment ideas.

Do not spend the first week trying to port any of these systems. The contest bot should be a small, legal, inspectable policy tailored to Fullhouse.

## Small-Commit Build Sequence

1. Scaffold `bots/heuristic/bot.py` with safe `decide()`, action sanitizer, and validator pass.
2. Add preflop hand classification and position inference.
3. Add in-memory opponent model from `match_action_log`.
4. Add bounded `eval7` postflop equity estimates and caching.
5. Add postflop value/call/bluff policy.
6. Add exploit adjustments by opponent type.
7. Add local evaluation harness and mutated benchmark bots if needed.
8. Tune thresholds from seeded aggregate results.

Each step should update docs or this plan if assumptions change, then commit.
