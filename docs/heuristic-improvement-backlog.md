# Heuristic Improvement Backlog

Reviewed: 2026-05-26.

This backlog turns the future-improvement ideas from `docs/heuristic-bot-logic.md` into ranked, measurable tasks.

Effort scale:

- `1`: very easy, mostly tooling or constants.
- `2`: easy, small isolated bot change.
- `3`: medium, strategy behavior can shift and needs benchmarks.
- `4`: hard, needs careful tuning.
- `5`: blocked or research-heavy.

## Ranked Improvements

| Rank | Improvement | Effort | Quantifiable Target | Action Plan | Status |
| ---: | --- | ---: | --- | --- | --- |
| 1 | Deterministic evaluation mode | 1 | Same benchmark command can be rerun with fixed bot-local RNG; validator still passes without env vars. | Add optional env-controlled RNG seed in `bots/heuristic/bot.py`; document env var. | Done |
| 2 | Submission packaging check | 1 | One command creates a zip and validates it with `sandbox/validator.py`. | Add `tools/package_heuristic.py`; default output outside committed source or under ignored build output. | Done |
| 3 | Threshold search harness | 2 | One command evaluates named parameter sets and reports mean/min delta and errors. | Add local tuner that sets env vars before `run_match()`; keep submitted bot defaults unchanged. | Done |
| 4 | Code hygiene for submission bot | 1 | Remove unused import/helper while keeping validator green and benchmark command runnable. | Remove unused `math` import and unused `_players_left_to_act()` unless needed by a new feature. | Done |
| 5 | Explicit 169-class preflop table | 3 | `_preflop_score()` has table coverage for all 169 canonical classes. | Generate an internal preflop score table at import from transparent rules; policy reads the table. | Done |
| 6 | Smarter opponent model | 3 | Track pressure folds, normalized raise size, and expose them in profile/fold-pressure decisions. | Extend per-opponent stats from public action stream; update classification and fold pressure. | Done |
| 7 | Better hand-category detection | 3 | Postflop policy can identify made-hand class, flush draw, and straight draw from hero+board. | Add feature extractor using `eval7.handtype()` plus deterministic draw checks; feed thresholds. | Done |
| 8 | Safer heads-up maniac mode | 3 | Heads-up aggressor benchmark should avoid stack-off variance when hero has a large lead. | Add stack-lead protection and tighter high-risk calls against maniacs. | Done |
| 9 | Range-aware equity adjustment | 4 | Equity used by policy changes by opponent profile and betting context; benchmarks remain positive. | Apply transparent profile/risk adjustments to raw Monte Carlo equity before policy thresholds. | Done |
| 10 | Hand-history patch workflow | 4 | One command can summarize local/exported hand histories with showdown/action leak metrics. | Add analyzer that accepts JSON hand logs/results; make it tolerant of unknown Day 1 schema. | Done |
| 11 | SPR-aware candidate configs | 3 | Low-SPR/high-SPR knobs can be benchmarked without changing defaults. | Add SPR threshold/call-margin/value-sizing env knobs and named configs. | Candidate implemented |
| 12 | Off-bucket sizing candidate configs | 3 | Bucket/threshold stress suites can test occasional nonstandard legal bet sizes. | Add no-op-by-default sizing perturbation helper and named configs. | Candidate implemented |
| 13 | Full preflop matrix tuning | 5 | Separate matrix by position, pot state, heads-up/6-max, stack depth, and opponent profile. | Use benchmark-driven tuning after the explicit 169-class table exists. | Backlog |

## Pro-Heuristic Gaps

The current bot has basic position, pot odds, equity, board texture, and opponent profile logic. It does not yet encode several common high-level tactics used by strong human players and modern solver-informed strategy.

Sources checked:

- Stack-to-pot ratio planning: https://www.pokerology.com/poker/strategy/spr/
- Bet sizing, polarized and merged betting ranges: https://www.pokerstars.com/poker/learn/lesson/bet-sizing/
- Continuation betting, range advantage, and nut advantage: https://visionpoker.net/en/articles/continuation-bet-strategy-advanced
- Range construction and board-dependent sizing: https://legal.ggpoker.com/blog/range-construction-101
- Blockers/card removal: https://www.pokerskill.com/poker-glossary/blockers/
- Minimum defense frequency: https://www.888poker.com/magazine/strategy/practical-guide-using-mdf-poker
- ICM and tournament risk pressure: https://www.pokerstars.com/poker/learn/lesson/icm-and-its-impact-on-tournaments/

| Priority | Gap | What Strong Players Do | Lightweight Bot Plan | Measurable Target |
| ---: | --- | --- | --- | --- |
| 1 | SPR-aware commitment | Use stack-to-pot ratio to decide when one-pair/value hands are stack-off candidates and when pot control is better. | Add `_spr(state)` and adjust risk guard/value sizing by SPR bands: `<2`, `2-5`, `>5`. | Reduce busts in `pressure_6max` and `mixed_stress_6max` without lowering `reference_6max` mean. |
| 2 | Nut advantage/range advantage | C-bet more on boards favorable to opener/range, slow down on boards favoring caller/defender. | Add board-class features: high-card dry, low-connected, monotone, paired, broadway-heavy; combine with preflop aggressor flag from action log. | Improve `tight_6max` and `sizing_6max` positive-run rate. |
| 3 | Blockers/card removal | Choose bluffs with cards that block opponent nut/value combos and avoid bluffing cards that block opponent folds. | Add cheap blocker flags: ace of flush suit, king of flush suit, pair/blocker to board-paired full houses, straight blockers on four-liner boards. | Improve bluff EV in `heads_up_threshold` and reduce failed bluffs into stations. |
| 4 | Polarized vs merged sizing | Use larger sizes with polarized value/bluff ranges; smaller sizes with merged thin-value/protection ranges. | Split postflop action into `polarized_value`, `merged_value`, `semi_bluff`, `pure_bluff` buckets and map to env-tunable size families. | Beat baseline on `sizing_6max` over at least 30 seeds. |
| 5 | Delayed c-bet/probe logic | Sometimes check back flop with medium strength, then bet turn after opponent checks twice. | Track previous-street check sequences from `action_log`; add turn probe/delayed-cbet branch. | Improve mean delta in `tight_6max`; no increase in errors/time. |
| 6 | MDF-inspired anti-bluff defense | Continue enough versus aggressive bet sizes to avoid being exploited by any-two-card bluffs, but overfold versus honest populations. | Approximate MDF from `pot/(pot+bet)` and blend it with equity/pot-odds only against maniac/high-bluff profiles. | Improve `heads_up_aggressor` without harming 6-max acceptance run. |
| 7 | Tournament stack pressure | Humans tighten stack-off decisions when chip survival has nonlinear value and loosen when short. | Add stack percentile/table-chip share features; tighten calls when healthy, widen value jams when short. | Reduce 100-run bust count in core 6-max suites. |
| 8 | Slowplay/trap frequency | Strong players sometimes check strong hands against aggressive opponents to induce bluffs. | Add very low-frequency trap with strong made hands vs maniac when can check and SPR is low/medium. | Improve `pressure_6max` while keeping station value extraction positive. |

## Lightweight Statistical Improvements

These methods are intentionally not neural-net-like. They fit the sandbox because the submitted bot can ship plain coefficients in code or read small arrays from `data/` at import time. Training/tuning should happen offline only.

| Priority | Component | Method | Features | Output | Acceptance Test |
| ---: | --- | --- | --- | --- | --- |
| 1 | Fold-to-size model | Beta-binomial MLE per opponent/profile/bin | Bet fraction bin, street, profile, heads-up vs multiway, previous pressure events. | Smoothed probability that opponent folds to a given size. | Use model to choose between `0.34/0.50/0.67/0.90` pot; beat fixed sizing on `sizing_6max`. |
| 2 | Bet-size contextual bandit | Offline epsilon-greedy/UCB simulation over fixed size arms | Board texture, hand bucket, profile, fold pressure, SPR, position. | Best size arm for value/bluff class. | Improve 30-seed stress-suite aggregate without raising bust count. |
| 3 | Equity correction regression | Linear/logistic regression or isotonic calibration | Raw equity, pot odds, bet size, profile, street, opponent count, board class. | Calibrated showdown/win probability or call/fold score. | Better call/fold decisions in `pressure_6max`; validator still under 2s. |
| 4 | Opponent profile classifier | Multinomial logistic regression or Naive Bayes | Raise/call/fold/check/all-in rates, pressure fold/call, average raise BB, street-specific stats. | Probability over `maniac/station/nit/abc/unknown`. | More stable profile labels after 20-50 actions; fewer misclassified stations. |
| 5 | Preflop action matrix tuning | Coordinate search or Thompson sampling over threshold arms | Position, profile, pot state, stack depth, hand class score. | Open/call/fold/reraise threshold table. | Improve core 6-max mean over current explicit 169-class score policy. |
| 6 | Risk guard calibration | Logistic/linear model for bust-risk penalty | Risk fraction, effective stack, SPR, opponent count, profile, current chip lead. | Extra equity margin needed before calling. | Lower bust count in 100-run acceptance with mean delta not materially lower. |
| 7 | Showdown leak model after Day 1 | MLE from exported histories | Action line, bet sizes, reached showdown, revealed hand class. | Population priors for overbluffing/overcalling by line. | Patch priors before finals; compare against Day 1 hand-history holdout. |
| 8 | Config selection bandit | Multi-armed bandit over named env configs | Suite/config result summaries. | Which config to promote to default. | Avoid overfitting to one suite; select config by lower-confidence-bound score. |

For any statistical upgrade, keep the runtime implementation simple: a few coefficient arrays, if/else feature extraction, and no training inside `decide()`.

Implemented candidate status:

- SPR-aware commitment is available through the `spr-aware` and `spr-anti-bucket` env configs; defaults are unchanged until the selector promotes them.
- Off-bucket sizing is available through the `anti-bucket` and `spr-anti-bucket` env configs; it only changes bet sizes within the same strategic action class.
- Config selection is implemented as a risk-aware ranking tool in `tools/select_heuristic_config.py`; it is an offline selector, not runtime learning.

## Expected Competitor Strategies

The sandbox design makes it likely that many serious teams will train or prepare something offline and ship a compressed runtime artifact. The allowed `numpy`, `scipy`, `scikit-learn`, `eval7`, optional read-only `data/`, and 200 MB data limit are enough for lookup tables, small neural-network weights, sklearn models, or coarse CFR-style blueprints, even though PyTorch/TensorFlow are not allowed at runtime.

Likely field types:

| Type | What They May Build | Likely Weakness | Counter-Plan |
| --- | --- | --- | --- |
| Reference-bot clones | Small edits to template/aggressor/shark/pot-odds bots. | Predictable thresholds and static leaks. | Keep current opponent profiles and sizing exploits. |
| Eval7 equity bots | Monte Carlo equity plus pot odds and fixed bet sizes. | Treat opponent ranges as random; overcall/overfold at thresholds. | Use profile-aware pressure, exploit size thresholds, and avoid donating into stations. |
| Preflop-chart TAG bots | Strong hand chart plus simple postflop rules. | Too tight versus steals, too honest after checking. | Steal blinds, c-bet dry boards, respect sudden large raises. |
| Offline lookup-table bots | Precomputed preflop/equity tables in `data/`. | Tables may ignore opponent type, action history, or unusual sizes. | Vary sizes and lines in close spots; classify whether they are threshold-driven. |
| Sklearn/regression bots | Logistic/MLP/random-forest policy trained offline and shipped as coefficients or arrays. | Distribution shift; fragile to state encodings and rare bet sizes. | Use controlled mixed lines and stress suites with threshold/pressure bots. |
| CFR/blueprint-inspired bots | Coarse abstraction with fixed action buckets. | Off-tree actions, weird stack states, multiway simplifications. | Use legal off-bucket sizing occasionally; punish overfolding/overcalling by observed response. |
| Bandit/adaptive bots | Try to learn which strategy works during 400 hands. | Early samples are noisy; may overreact to our first lines. | Avoid revealing one deterministic pattern early; mix value/check/bluff lines at low frequency. |
| Always-aggressive high-variance bots | Jam/raise often to exploit finite-match upside. | Overcommit weak ranges and bust often. | Keep risk guard, call wider only with real equity, trap more with strong hands. |

## Anti-RL/Anti-NN Plan

The bot should not become random. Random bad actions are just chip leakage. The useful version is a small "anti-brittle-policy" layer that only fires when the EV cost is low or when the hand is strong enough to tolerate deception.

Candidate improvements:

| Priority | Tactic | Implementation | Guardrail | Target |
| ---: | --- | --- | --- | --- |
| 1 | Low-frequency trap checks | With strong made hands vs maniac/high-pressure table, sometimes check instead of value betting. | Only when `can_check`, SPR is low/medium, and equity is clearly above value threshold. | Improve `pressure_6max`; no station regression. |
| 2 | Off-bucket bet sizes | Randomly choose among nearby legal fractions such as `0.34`, `0.42`, `0.55`, `0.67`, `0.90` instead of one fixed size. | Only among sizes with same strategic purpose; never increase stack-risk calls. | Break fixed-action-bucket bots and threshold callers. |
| 3 | Blocker-based rare bluffs | Bluff only with cards blocking nut flush/straight/value combos. | No blocker bluff into station profile; keep probability below a tuned cap. | Improve `heads_up_threshold` and `tight_6max`. |
| 4 | Delayed c-bets | Check back some medium/strong hands on flop, bet turn after repeated weakness. | Need action-log detection of check-check lines; avoid wet multiway pots. | Reduce predictability against regression/CFR-style bots. |
| 5 | Early-match ambiguity | During first 20-40 hands, avoid always mapping hand strength to the same line. | Small frequency only; no trash stack-off. | Prevent adaptive/bandit bots from learning a clean exploit quickly. |
| 6 | Threshold probing | Use small test bets against unknowns; update pressure-fold stats. | Stop probing if opponent calls too much. | Faster opponent classification in 400-hand matches. |
| 7 | Stack-pressure deviation | When we are table chip leader, pressure medium stacks more; when healthy but not dominant, avoid marginal all-ins. | Keep chip-delta scoring in mind; do not optimize for survival alone. | Lower bust count while preserving upside. |

Practical rule: cap all "unlikely action" mixing to a small explicit budget, probably `2-6%` of eligible close decisions, and make it env-tunable. The action still has to be a recognizable poker line: trap, delayed c-bet, blocker bluff, probe, or off-bucket size. Do not add pure noise.

## Implementation Policy

Implement in rank order until a benchmark regression appears. For every bot logic change:

1. Run `poetry run python sandbox/validator.py bots/heuristic/bot.py`.
2. Run at least one short smoke match.
3. Run the benchmark matrix before calling the change complete.
4. Update this backlog status and any affected logic docs.
5. Commit the slice.

Final acceptance run:

- Run 100 match simulations per benchmark configuration.
- Report mean, median, standard deviation, min, max, positive-run count, nonnegative-run count, bust count, and bot-error count.
- Use the committed benchmark harness, not an ad hoc script.

Ranks 1-10 are implemented. Rank 11 is intentionally left as a later tuning pass because it is likely to overfit without a larger seed set and should use the 100-run benchmark output as its input signal.

The acceptance run is recorded in `docs/heuristic-benchmark-results.md`. Its main actionable finding is that heads-up aggressor remains high variance and should be optimized only if doing so does not reduce 6-max performance.
