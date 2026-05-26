# Heuristic Parameter Audit

Reviewed: 2026-05-27.

This document lists the current bot choices that are not fully justified by data yet. These are the knobs to tune before changing architecture.

The bot now exposes env vars for the major audited knobs. This makes benchmark sweeps possible without refactoring the submission bot or changing the Fullhouse interface.

## Highest Priority Unknowns

| Area | Current Choice | Why It Is Weakly Justified | Quantifiable Test |
| --- | --- | --- | --- |
| Postflop value sizing | `0.52`, `0.75`, `0.45`, `0.85` pot fractions | Chosen as simple poker-like sizes, not optimized against Fullhouse field behavior. | Sweep value/semi-bluff fractions over 400-hand 6-max suites; keep configs that improve mean without increasing bust count. |
| Dry bluff frequency | `HEURISTIC_DRY_BLUFF_PROB=0.45` | Directionally reasonable against nits/ABC bots, but not calibrated to actual fold-to-bet rates. | Sweep `0.15, 0.30, 0.45, 0.60`; compare mean delta and bust count in `reference_6max`. |
| Wet semi-bluff frequency | `HEURISTIC_WET_BLUFF_PROB=0.25` | Draw detection improved, but frequency is still hand-authored. | Sweep `0.05, 0.15, 0.25, 0.35`; require no regression vs station/maniac suites. |
| Call margin | `0.075 + 0.035 * extra_opponents` | Pot-odds safety margin is plausible but not derived from opponent range modeling. | Sweep base `0.055-0.105` and multiway `0.025-0.055`; optimize for 6-max chip delta and fewer busts. |
| Risk guard thresholds | `0.76`, `0.86`, `0.92` at risk cutoffs `0.28`, `0.45`, `0.70` | Protects stack, but the thresholds are not fit to tournament scoring. | Sweep required equities and risk cutoffs separately; track bust count as a first-class metric. |
| SPR-aware commitment | Promoted `spr-anti-bucket` defaults | Low stack-to-pot ratio should change value/call thresholds, but the right amount is matchup-dependent. | Keep monitoring against `legacy-baseline`; retune only if the 100-seed final matrix shows a regression. |
| Off-bucket sizing | Promoted `12%` eligible sizing perturbation | May exploit threshold/bucket bots, but can also overpay for folds or reduce value. | Keep monitoring `sizing_6max`, `mock_bucket_6max`, and final 100-seed run. |
| Preflop score cutoffs | `88`, `72`, `62`, `58`, and raise-facing cutoffs | Generated 169 table exists, but action boundaries are still rough. | Convert to env-tunable thresholds or small matrix; tune by position/profile with 100-run 6-max benchmark. |
| Opponent profile thresholds | Raise/call/fold rates such as `0.33`, `0.42`, `0.62` | Based on intuitive behavior classes, not calibrated from hand histories. | After Day 1 histories, compare classifications against actual showdown/action leaks. |
| Equity context correction | Adjustments such as `-0.045` vs nit, `+0.025` vs maniac | Transparent but hand-authored replacement for true range-weighted equity. | Sweep corrections; reject if heads-up gain harms 6-max acceptance run. |
| Monte Carlo samples | `520/700/900`, reduced 25% for 4+ opponents | Designed for speed, not profiled against accuracy/error in Fullhouse states. | Sample random decision states; compare action stability at 200/500/1000/2000 samples under 2s budget. |

## Bet Sizing Clarification

The current bot does not use an explicit "1/3 step size" grid. It uses a small number of fixed pot fractions:

- `0.42` pot for dry-board bluffs and nit steals.
- `0.45` pot for thin value against stations/maniacs.
- `0.52` pot for normal value on safer boards.
- `0.55` pot for wet-board semi-bluffs.
- `0.75` pot for value on wet boards or against loose profiles.
- `0.85` pot for rare high-equity raises facing a small bet.

These are practical abstractions, not solved sizes. The reason for fixed sizes is that a discrete policy is easier to keep legal, fast, and interpretable under the 2-second limit. The cost is obvious: opponents with threshold-based calling rules may be exploitable by more precise sizes, and our bot may overpay for folds.

## Current Tuning Harness

Named threshold configs live in `tools/tune_heuristic_thresholds.py`:

- `baseline`
- `anti-bucket`
- `conservative`
- `legacy-baseline`
- `spr-anti-bucket`
- `spr-aware`
- `value-heavy`
- `pressure`

The default harness now uses 400 hands and all benchmark configurations to match the hackathon setup more closely.

`baseline` now includes the promoted `spr-anti-bucket` behavior. Use
`legacy-baseline` only when comparing against the pre-promotion default.

Preflop table generation:

```bash
poetry run python tools/generate_preflop_table.py --iterations 20000 --seed 31337
```

The committed bot table is explicit and generated from deterministic sampled heads-up equity. It should be regenerated only when we intentionally change the score definition.

Latest tuning decision:

- `baseline` now means the promoted SPR/off-bucket profile.
- `legacy-baseline` preserves the pre-promotion defaults.
- The promotion was accepted after a 30-seed screen and a 100-seed final matrix across core, stress, and mock suites.
- `pressure`, `small-ball`, `spr-aware`, and `anti-bucket` remain comparison configs, not production defaults.

Quick smoke:

```bash
poetry run python tools/tune_heuristic_thresholds.py \
  --config baseline \
  --suite reference_6max \
  --seed-count 2 \
  --hands 20 \
  --summary-only \
  --json
```

Rule-matched comparison:

```bash
poetry run python tools/select_heuristic_config.py \
  --preset final \
  --config baseline \
  --progress \
  --json
```

## Refactor Boundary

Do not split the submission bot just to tune these parameters.

Acceptable changes:

- Add env tunables for constants that need sweeps.
- Add benchmark-only config dictionaries.
- Add documentation for why a default survived the benchmark.

Avoid:

- Importing benchmark dependencies into `bots/heuristic/bot.py`.
- Adding files that the submitted bot must import.
- Refactoring the engine or sandbox to support non-Fullhouse poker engines.
- Optimizing against heads-up-only external models if it weakens 6-max results.

## Next Parameter Work

1. Keep `baseline` as the promoted SPR/off-bucket default unless a future promotion screen beats it.
2. Compare against `legacy-baseline` only when checking whether a new idea is better than the pre-promotion policy.
3. Prioritize high-variance watch items from the 100-seed matrix: `pressure_6max`, `heads_up_equity_mc`, and `heads_up_aggressor`.
4. Treat external neural/RL baselines as diagnostic opponents, not final acceptance criteria.
5. Promote any new default only after candidate, promotion, and final acceptance screens.
