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
| 5 | Explicit 169-class preflop table | 3 | `_preflop_score()` has table coverage for all 169 canonical classes. | Generate an internal preflop score table at import from transparent rules; policy reads the table. | Planned |
| 6 | Smarter opponent model | 3 | Track pressure folds, normalized raise size, and expose them in profile/fold-pressure decisions. | Extend per-opponent stats from public action stream; update classification and fold pressure. | Planned |
| 7 | Better hand-category detection | 3 | Postflop policy can identify made-hand class, flush draw, and straight draw from hero+board. | Add feature extractor using `eval7.handtype()` plus deterministic draw checks; feed thresholds. | Planned |
| 8 | Safer heads-up maniac mode | 3 | Heads-up aggressor benchmark should avoid stack-off variance when hero has a large lead. | Add stack-lead protection and tighter high-risk calls against maniacs. | Planned |
| 9 | Range-aware equity adjustment | 4 | Equity used by policy changes by opponent profile and betting context; benchmarks remain positive. | Apply transparent profile/risk adjustments to raw Monte Carlo equity before policy thresholds. | Planned |
| 10 | Hand-history patch workflow | 4 | One command can summarize local/exported hand histories with showdown/action leak metrics. | Add analyzer that accepts JSON hand logs/results; make it tolerant of unknown Day 1 schema. | Planned |
| 11 | Full preflop matrix tuning | 5 | Separate matrix by position, pot state, heads-up/6-max, stack depth, and opponent profile. | Use benchmark-driven tuning after the explicit 169-class table exists. | Backlog |

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

The current focus is ranks 1-10. Rank 11 is intentionally left as a later tuning pass because it is likely to overfit without a larger seed set.
