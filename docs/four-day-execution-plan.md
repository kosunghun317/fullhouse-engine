# Four-Day Heuristic Execution Plan

Reviewed: 2026-05-27.

This plan classifies the attached strategy suggestions by what is realistically implementable before the hackathon, then orders the implementation work.

## Constraints

- Keep `bots/heuristic/bot.py` validator-shaped and submission-safe.
- Do not import runtime-heavy ML libraries into the submitted bot.
- Use `data/` only for small read-only tables loaded at module import.
- Optimize against Fullhouse rules: 400-hand matches, 6-max tables, 2 seconds/action, no network, no runtime file writes, 768 MB RAM, 0.5 CPU.
- Prefer changes that improve existing-opponent benchmarks and reduce bust frequency.

## Classification

| Idea | 4-Day Status | Why | Execution Target |
| --- | --- | --- | --- |
| Mock competitor bot zoo | Implement now | Low risk, directly improves tests against likely field strategies. | Add `bots/mock_competitors/` and benchmark suites. |
| Preflop lookup table | Done | Explicit 169-class table is already embedded. | Keep generator and tune thresholds around it. |
| Data-package lookup tables | Implement now | Useful, simple, and allowed by rules. | Add optional `data/tables.npz` support and package script inclusion. |
| Pot-odds threshold detector | Implement soon | Concrete exploit, easy to validate. | Track small/large pressure responses and adjust sizing. |
| SPR-aware risk/value logic | Implement soon | Likely improves bust control and pressure suites. | Add `_spr()` and risk/value threshold adjustments. |
| Off-bucket sizing mix | Implement soon | Direct counter to bucketed CFR/NN policies. | Low-frequency nearby size variation with guardrails. |
| Low-frequency trap checks | Implement soon | Simple anti-maniac/anti-modeling improvement. | Strong-hand checks vs aggressive profiles only. |
| Delayed c-bet / probe logic | Prototype if time | Needs action-line state parsing, but manageable. | Turn bet after flop checks through and opponent checks again. |
| Blocker-based bluffs | Prototype if time | Requires careful board/card feature extraction. | Only rare bluffs with nut blockers, never vs stations. |
| Online contextual bandit | Prototype if time | Reward attribution in 400 hands is noisy. | Start benchmark-only or cluster Thompson over safe modes. |
| LinUCB over expert arms | Defer unless time remains | More moving parts and high regression risk. | Document feature vector; avoid raw-action bandit. |
| Postflop bucket equity table | Defer | Generation quality/time tradeoff is uncertain. | Keep selective `eval7` Monte Carlo for now. |
| CFR/NN/RL training | Defer | Not aligned with time limit. | Use mock competitors instead. |

## Execution Order

### Stage 1: Mock Competitor Zoo

Create `bots/mock_competitors/` with opponents that imitate likely submissions:

- `equity_mc`: eval7 Monte Carlo equity + pot odds.
- `cbet_reg`: preflop raiser and over-c-bettor.
- `bucket_policy`: CFR-like coarse board/hand buckets and fixed action sizes.
- `numpy_policy`: deterministic lightweight policy over fixed action buckets.
- `opponent_modeler`: classifies hero from action history and counters.
- `copy_shark_plus`: modified public Shark with tighter postflop discipline.
- `pot_odds_plus`: pot-odds threshold bot with multiple thresholds.

Success:

- Each validates through `sandbox/validator.py`.
- Add mock suites to `tools/evaluate_heuristic.py`.
- Run one smoke suite without bot errors.
- If a mock is NN-like, train it offline and save weights under its own `data/` directory before using it.

### Stage 2: Lookup Data Support

Use the data-size allowance without making the bot dependent on data:

- Generate `bots/heuristic/data/tables.npz`.
- Store small metadata/tables:
  - preflop scores for all 169 classes.
  - bet-size arms.
  - context cluster priors for future bandit/config selection.
- Load with `np.load(..., allow_pickle=False)` at import time.
- Fallback to hard-coded constants if data is absent.
- Update `tools/package_heuristic.py` to include optional `data/`.

Implemented builder:

```bash
poetry run python tools/build_heuristic_tables.py --json
```

Success:

- Package zip contains `bot.py` and `data/tables.npz`.
- Validator passes the zip.

### Stage 3: Safe Strategy Improvements

Implement low-risk tactical changes:

- SPR-aware risk/value adjustment.
- Off-bucket size variation with small probability and same strategic intent.
- Trap checks only with very strong hands vs aggressive profiles.
- Optional delayed-cbet/probe if action parsing stays clean.

Success:

- Existing core benchmark does not regress materially.
- `pressure_6max` and `mixed_stress_6max` bust count does not increase.

### Stage 4: Bandit / Statistical Selection

Start outside the submitted bot:

- Add an offline config-selector tool that ranks named configs by lower-confidence-bound score.
- If stable, add a tiny runtime clustered Thompson selector only over safe expert modes.

Initial context features:

- street bucket.
- preflop score/equity bucket.
- made-hand rank.
- draw flag.
- SPR bucket.
- position bucket.
- active opponent count.
- table profile.
- target fold/call/raise/all-in rates.
- facing-bet and bet-size bucket.

Success:

- Bandit/config selector identifies baseline-or-better configs on held-out seeds.
- No raw-action exploration.

## Current Default Policy

The current default remains `baseline`. Prior tuning showed `pressure` and `small-ball` help selected suites but are not robust enough to promote.

Smoke tests are validity checks only. Do not promote or reject a default from a smoke result. A config decision needs a large enough sample:

- Candidate screen: at least `10` seeds at 400 hands on core suites plus relevant stress suites.
- Default promotion: at least `30` seeds at 400 hands on core suites and no obvious stress-suite regression.
- Final acceptance: `100` seeds at 400 hands.

Decision metrics:

- mean chip delta.
- median chip delta.
- bust count.
- positive-run count.
- worst seed/min delta.
- targeted suite result if the change is meant to fix one weakness.

Default changes must pass:

```bash
poetry run pytest -q
poetry run python sandbox/validator.py bots/heuristic/bot.py
poetry run python tools/package_heuristic.py --json
poetry run python tools/evaluate_heuristic.py --seed-start 1001 --seed-count 100 --hands 400 --summary-only --json
```
