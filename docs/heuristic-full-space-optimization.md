# Heuristic Full-Space Optimization Method

Reviewed: 2026-05-29.

This document explains the optimization method used by
`tools/tune_heuristic_full_space.py`. It is local-only infrastructure for
searching `HEURISTIC_*` environment parameters in `bots/heuristic/bot.py`.

## Problem Shape

The heuristic bot exposes about 90 numeric knobs. The objective is noisy
400-hand poker EV across multiple opponent suites, not a differentiable loss.
The goal is a stable local maximum that generalizes across opponent families,
not a single lucky global-looking maximum on one seed block.

Brute force is not viable: even 3 values per 90 parameters is `3^90`
combinations. One-parameter sweeps also miss interactions between thresholds,
bet sizes, preflop cutoffs, profile classifiers, and risk guards.

## Chosen Method

The implemented method is a diagonal cross-entropy method (CEM) with staged
successive-halving style racing and common random numbers.

In each generation:

1. Encode every tunable parameter into `[0, 1]`.
2. Sample a population from a diagonal Gaussian around the current mean.
3. Include the current mean, the best prior candidate, and named configs as
   anchors.
4. Evaluate candidates on the same suite/seed block within each stage.
5. Keep only the top fraction for the next, larger stage.
6. Update the search mean and per-parameter sigma from the elite set using
   soft score weights, smoothing, sigma decay, and a sigma floor.
7. Run a final independent held-out validation on the selected best env.

This is intentionally closer to CEM/CMA-ES-style local black-box optimization
than to genetic crossover. The diagonal covariance is a pragmatic compromise:
it scales cleanly to this parameter count and avoids overfitting a full
covariance estimate from a small, noisy elite set.

## Objective

Each suite summary is scored as:

```text
mean_delta
- risk_weight * stdev_delta
+ min_weight * min_delta
- bust_penalty * bust_rate
- error_penalty * error_rate
```

The candidate score is the mean of suite scores. This makes a candidate less
attractive if it wins only by taking unstable bust-heavy lines, collapses on
one suite, or causes bot errors.

## Statistical Controls

Serious runs refuse weak budgets unless `--allow-smoke` is explicit:

- at least 16 candidates per generation,
- at least 3 generations,
- at least 400 hands per match,
- at least 512 suite/seed tasks per candidate stage,
- at least 1024 suite/seed tasks in the final racing stage,
- at least 1024 held-out final-validation suite/seed tasks for the selected
  candidate.

`--allow-smoke` is only for wiring checks. Smoke output must not be treated as
evidence for promotion.

## Common Random Numbers

Within a stage, all candidates use the same seed block. This reduces comparison
noise because candidates face the same shuffled scenarios and opponent seats
for a given suite/seed task. Different generations and later stages use fresh
seed blocks to avoid fitting one fixed sample.

## Promotion Policy

The tuner writes candidates and evidence, but it does not edit
`bots/heuristic/bot.py`.

Important outputs:

- `param_space.json`: parsed knobs and bounds.
- `generation_*.json`: staged rankings.
- `best_config.json`: best candidate from racing.
- `best_env.sh`: environment exports for reproducing the best candidate.
- `final_validation.json`: independent held-out validation of the selected env.
- `metrics.jsonl` and `ev_progress.svg`: progress trace.

Promote only after reviewing `best_config.json`, `final_validation.json`, and a
fresh submission pipeline run. If the final candidate is strong, copy the
specific env defaults into source intentionally; do not make generated wrapper
bots part of the submitted bot.

## Strong-Mock Requirement

Optimization is only useful if the opponent pool is credible. Strong mocks are
therefore gated by `tools/check_strong_mocks.py`, which checks every
`bots/strong_mocks/*` wrapper against default/reference bots in six-max and
heads-up tasks with at least 512 tasks per candidate by default.

## References

- Nikolaus Hansen and Andreas Ostermeier, "Completely Derandomized
  Self-Adaptation in Evolution Strategies", 2001:
  https://cse-lab.seas.harvard.edu/publications/completely-derandomized-self-adaptation-evolution-strategies-0
- CMA-ES project overview:
  https://cma-es.github.io/
- Reuven Rubinstein, "The Cross-Entropy Method for Combinatorial and
  Continuous Optimization", 1999:
  https://www.ovid.com/journals/mcap/fulltext/00126527-199901020-00001~the-cross-entropy-method-for-combinatorial-and-continuous
- Pieter-Tjerk de Boer, Dirk Kroese, Shie Mannor, and Reuven Rubinstein,
  "A Tutorial on the Cross-Entropy Method", 2005:
  https://research.utwente.nl/en/publications/a-tutorial-on-the-cross-entropy-method/
- Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet
  Talwalkar, "Hyperband: A Novel Bandit-Based Approach to Hyperparameter
  Optimization", JMLR 2018:
  https://jmlr.csail.mit.edu/papers/volume18/16-558/16-558.pdf
- Russell Heikes, Douglas Montgomery, and Ronald Rardin, "Using Common Random
  Numbers in Simulation Experiments", 1976:
  https://journals.sagepub.com/doi/10.1177/003754977602700301
