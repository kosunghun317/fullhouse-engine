# Preflop Monte Carlo Heuristic Candidate

`tools/build_preflop_mc_heuristic.py` builds a separate candidate from the
current submission zip and replaces only the preflop hand-score function with a
Monte Carlo estimate.

The generated candidate keeps the tuned submission bot's postflop logic,
environment-baked constants, and `data/tables.npz`. It appends a new
`_preflop_score(cards)` implementation to `bot.py`, so the existing preflop
policy thresholds and hand-class guards still run, but the numeric score comes
from heads-up all-in equity against one random hand.

## Build

```bash
PYTHONDONTWRITEBYTECODE=1 \
poetry run python tools/build_preflop_mc_heuristic.py \
  --input dist/heuristic_bot.zip \
  --output runs/preflop_mc_heuristic/preflop-mc-4096/bot \
  --samples 4096 \
  --budget 0.35 \
  --min-samples 512 \
  --zip-output runs/preflop_mc_heuristic/preflop-mc-4096/preflop_mc_heuristic.zip \
  --json
```

The output directory must not already exist. This avoids overwriting previous
benchmark candidates.

## Latency Check

The 4096-sample candidate built on June 1, 2026 validated successfully:

- official validator preflop decision: `0.043s`;
- cold-cache mean across all 169 hand classes: `0.031s`;
- cold-cache p95: `0.065s`;
- cold-cache max: `0.300s`;
- no decisions over `1s` or `2s`.

This means preflop Monte Carlo is not a resource-limit problem at 4096 samples
with the current implementation.

## Focused Screen

Small focused screen, `16 seeds x 200 hands`, comparing the generated
preflop-MC candidate to `dist/heuristic_bot.zip`:

| suite | heuristic mean | preflop-MC mean | read |
|---|---:|---:|---|
| `heads_up_aggressor` | `+3750` | `+6250` | improved mean and bust |
| `reference_6max` | `+12181` | `+15631` | improved mean and bust |
| `strong_hu_rollout` | `+5000` | `+5000` | unchanged in this screen |
| `strong_hybrid_6max` | `+5637` | `+6333` | slightly higher mean, higher bust |

The sample is too small to promote by itself. Use a larger matrix before
submitting the preflop-MC variant.

## Benchmark

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
poetry run python tools/benchmark_submission_heuristic.py \
  --hands 512 \
  --seeds 128 \
  --candidate preflop_mc=runs/preflop_mc_heuristic/preflop-mc-4096/bot
```
