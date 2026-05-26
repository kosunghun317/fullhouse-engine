# Heuristic Bot Benchmark Results

Reviewed: 2026-05-26.

## Validation Commands

```bash
poetry run pytest -q
poetry run python sandbox/validator.py bots/heuristic/bot.py
poetry run python sandbox/match.py bots/heuristic/bot.py bots/shark/bot.py --hands 100 --seed 77 --json
poetry run python tools/evaluate_heuristic.py \
  --suite reference_6max \
  --suite mutant_6max \
  --suite heads_up_shark \
  --suite heads_up_aggressor \
  --suite heads_up_station \
  --seeds 101,202,303 \
  --hands 400 \
  --json
```

## Results Summary

Core validation:

- `pytest`: 8 passed, with existing `eval7`/`pyparsing` deprecation warnings.
- Validator: `bots/heuristic/bot.py` passed.
- Direct shark smoke match: heuristic `+2,834` over 100 hands, no bot errors.

Benchmark matrix:

| Suite | Hands x Runs | Mean Delta | Median Delta | Min Delta | Max Delta | Positive Runs | Heuristic Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reference_6max` | 400 x 3 | 20,910 | 22,871 | 13,304 | 26,555 | 3/3 | 0 |
| `mutant_6max` | 400 x 3 | 35,548.67 | 44,021 | 18,200 | 44,425 | 3/3 | 0 |
| `heads_up_shark` | 400 x 3 | 9,806 | 10,000 | 9,418 | 10,000 | 3/3 | 0 |
| `heads_up_aggressor` | 400 x 3 | 3,333.33 | 10,000 | -10,000 | 10,000 | 2/3 | 0 |
| `heads_up_station` | 400 x 3 | 10,000 | 10,000 | 10,000 | 10,000 | 3/3 | 0 |

## Notes

The deck seed is deterministic, but some benchmark opponents use their own process-local randomness. Treat these results as a smoke benchmark plus directional signal, not a statistically final estimate.

The weakest suite is `heads_up_aggressor`, where one run busted. This is acceptable for the first heuristic implementation because the target format is 6-max, but future tuning should reduce heads-up maniac variance without weakening 6-max risk controls.
