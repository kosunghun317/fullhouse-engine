# Portal Replay Artifacts

Reviewed: 2026-06-02.

This document records local replay-data artifacts generated from the public
Fullhouse portal. Large replay files live under `runs/portal_history/` and are
not committed.

## All Qualifier Public

- Local path: `runs/portal_history/all_qualifier_public/`
- Mode: all complete public internal qualifier matches
- Selected bots: 298
- Selected matches: 675
- Downloaded hands: 310,799
- Strategy report bots: 298
- Strategy report parser errors: none
- Total artifact size: about 546 MB
- `hands.ndjson`: about 208 MB
- per-match JSON directory: about 332 MB
- `strategy_report.json`: about 208 KB

Download command:

```bash
poetry run python tools/download_portal_targeted_history.py \
  --all-complete-source-matches \
  --output runs/portal_history/all_qualifier_public \
  --page-size 1000 \
  --timeout 45 \
  --retries 4
```

Analysis command:

```bash
poetry run python tools/analyze_portal_strategy.py \
  runs/portal_history/all_qualifier_public \
  --top 320 \
  --output-json runs/portal_history/all_qualifier_public/strategy_report.json
```

## High-Signal Snapshot

Top official-rank rows in the all-public report:

| Rank | Bot | Delta | Hands alive | VPIP | PFR | Pressure fold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gems_VC2` | 338,184 | 4,804 | 0.3801 | 0.2871 | 0.7064 |
| 2 | `TheCrystalline` | 303,427 | 3,972 | 0.2238 | 0.1770 | 0.8072 |
| 3 | `AniBot` | 279,226 | 5,482 | 0.2364 | 0.1915 | 0.7530 |
| 4 | `Hyperion` | 269,961 | 3,676 | 0.4301 | 0.3077 | 0.5039 |
| 5 | `v16ship2` | 246,935 | 3,609 | 0.2763 | 0.2350 | 0.7790 |

`slop3` official row in the same report:

| Rank | Delta | Hands alive | VPIP | PFR | Call rate | Fold rate | Showdown | Pressure fold |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 158 | -22,449 | 3,055 | 0.3221 | 0.1555 | 0.2063 | 0.4925 | 0.1391 | 0.7804 |

The immediate exploit direction remains the same as the smaller sample:
increase initiative against the field, reduce passive calls, pressure high-fold
profiles, and switch to value-heavy lines against sticky callers.
