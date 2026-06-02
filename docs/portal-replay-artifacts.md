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
- Profile artifacts: `runs/portal_history/all_qualifier_public/profiles/`
- Profile count: 8
- Profiled bots: 297 of 298; one bot was below the 200-hand default cutoff
- Generated profile mocks: `runs/portal_profile_mocks/all_qualifier_public/`
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

Profile command:

```bash
poetry run python tools/build_portal_profiles.py \
  runs/portal_history/all_qualifier_public
```

Mock generation command:

```bash
poetry run python tools/build_portal_profile_mocks.py \
  runs/portal_history/all_qualifier_public \
  --output runs/portal_profile_mocks/all_qualifier_public
```

Mock evaluation smoke command:

```bash
poetry run python tools/evaluate_portal_profile_mocks.py \
  runs/portal_profile_mocks/all_qualifier_public \
  --mode heads-up \
  --profile tight_overfolder \
  --hands 20 \
  --seeds 7
```

Smoke result: the generated `tight_overfolder` mock directory passed
`sandbox/validator.py`, and the 20-hand heads-up evaluation completed with
zero candidate errors. The candidate lost that one tiny smoke sample by 3,325
chips; this is a wiring check only, not a strategy conclusion.

Portal-suite tuner smoke command:

```bash
poetry run python tools/tune_heuristic_full_space.py \
  --run-id portal-profile-smoke \
  --portal-mocks-dir runs/portal_profile_mocks/all_qualifier_public \
  --portal-only \
  --generations 1 \
  --population 2 \
  --elite 1 \
  --stages 1:10:0.5 \
  --allow-smoke \
  --skip-final-validation \
  --workers 1 \
  --parallel-backend process \
  --json
```

Smoke result: the tuner registered `portal_profiles_1` and
`portal_profiles_2`, completed the tiny CEM pass, wrote
`runs/heuristic_full_space_tuning/portal-profile-smoke/`, and reported zero
heuristic errors. This was only a suite-wiring check.

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

Generated profile mix from `profile_summary.json`:

| Profile | Bots | Top official rank | Mock VPIP | Mock PFR | Mock pressure fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tight_overfolder` | 87 | 3 | 0.1701 | 0.1294 | 0.8343 |
| `large_size_jammer` | 67 | 2 | 0.2571 | 0.1872 | 0.7301 |
| `sticky_station` | 44 | 4 | 0.3176 | 0.1977 | 0.6147 |
| `pressure_overfolder` | 41 | 1 | 0.2941 | 0.2463 | 0.7844 |
| `unknown_mixed` | 39 | 24 | 0.2601 | 0.1846 | 0.7323 |
| `balanced_aggressor` | 10 | 20 | 0.3712 | 0.2833 | 0.6783 |
| `passive_caller` | 6 | 45 | 0.2033 | 0.1099 | 0.7101 |
| `loose_passive_overfolder` | 3 | 134 | 0.4096 | 0.1518 | 0.8129 |

## Top-64 Profile Mocks

The same all-public report can be filtered by official rank before profile
construction:

```bash
poetry run python tools/build_portal_profiles.py \
  runs/portal_history/all_qualifier_public \
  --rank-lte 64 \
  --output runs/portal_history/all_qualifier_public/profiles_top64

poetry run python tools/build_portal_profile_mocks.py \
  runs/portal_history/all_qualifier_public/profiles_top64/profile_summary.json \
  --output runs/portal_profile_mocks/all_qualifier_top64
```

Generated top-64 profile mix:

| Profile | Bots | Mock VPIP | Mock PFR | Mock pressure fold |
| --- | ---: | ---: | ---: | ---: |
| `large_size_jammer` | 14 | 0.2767 | 0.2058 | 0.7574 |
| `pressure_overfolder` | 14 | 0.3008 | 0.2599 | 0.7884 |
| `sticky_station` | 13 | 0.3453 | 0.2402 | 0.6198 |
| `tight_overfolder` | 10 | 0.1949 | 0.1406 | 0.8323 |
| `unknown_mixed` | 6 | 0.2745 | 0.1899 | 0.7507 |
| `balanced_aggressor` | 5 | 0.4074 | 0.2981 | 0.6756 |
| `passive_caller` | 1 | 0.2096 | 0.1330 | 0.7267 |

The top-64 slice is a stronger tuning target than the full-field average:
initiative is higher, but pressure-fold rates remain high enough to justify
selective re-pressure. It should be used together with the full-field mocks so
the bot does not overfit only to winner profiles.

The immediate exploit direction remains the same as the smaller sample:
increase initiative against the field, reduce passive calls, pressure high-fold
profiles, and switch to value-heavy lines against sticky callers.
