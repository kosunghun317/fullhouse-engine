# Public Repo Strategy Research

Reviewed: 2026-06-02.

This is a separate research lane from the replay-exploit candidate work. Its
purpose is to inspect publicly visible Fullhouse Engine forks, compare any
public bot code against portal replay records, and only then derive
counter-strategy changes. Do not use this document to copy a competitor bot or
to overfit from one public file.

## Execution Blueprint

Evidence thresholds:

1. Public source discovery: identify repos through the official upstream fork
   graph and GitHub search.
2. Source inspection: inspect only public bot logic and summarize behavior at
   the strategy level.
3. Portal comparison: require a direct or highly plausible match between public
   source naming and a portal bot, then compare source behavior to replay
   metrics.
4. Counter adoption: only promote counter-specific logic when the portal
   metrics support it. If the source and replay disagree, prefer replay
   metrics.
5. Guardrail: public-source findings can add profile-specific tests and
   tunable parameters, but they cannot justify a hard-coded line by themselves.

Commands used:

```bash
curl -L -o /private/tmp/fullhouse_forks.json \
  'https://api.github.com/repos/uzlez/fullhouse-engine/forks?per_page=100'

poetry run python -c "import json; from pathlib import Path; data=json.loads(Path('/private/tmp/fullhouse_forks.json').read_text()); print(len(data))"

poetry run python -c "import json; from pathlib import Path; report=json.loads(Path('runs/portal_history/all_qualifier_public/strategy_report.json').read_text()); print([b for b in report['bots'] if b.get('bot_name')=='skantbot8'])"
```

Primary public sources:

- Official upstream: <https://github.com/uzlez/fullhouse-engine>
- GitHub forks API:
  <https://api.github.com/repos/uzlez/fullhouse-engine/forks?per_page=100>
- Public aggregate fork:
  <https://github.com/vladimirfilip/fullhouse-engine>
- SkantBot public file family:
  <https://raw.githubusercontent.com/vladimirfilip/fullhouse-engine/main/bots/Pav1602_skantbot/bot.py>,
  <https://raw.githubusercontent.com/vladimirfilip/fullhouse-engine/main/bots/Pav1602_skantbot2/bot.py>,
  <https://raw.githubusercontent.com/vladimirfilip/fullhouse-engine/main/bots/Pav1602_skantbot4/bot.py>
- Other checked public forks:
  <https://github.com/ds726/fullhouse-engine>,
  <https://github.com/Con-TI/fullhouse-engine>,
  <https://github.com/Benjamin-Yu-Sheng-Chang/fullhouse-hackathon>,
  <https://github.com/Mehedi-dev-2404/fullhouse-engine>,
  <https://github.com/jickzx/fullhouse-engine>

## Fork Snapshot

The fork API returned 34 public forks. Most small forks looked like untouched
copies or reference-bot changes. The highest-signal repos by recent activity,
size, and visible bot files were:

| Repo | Last push | Size | Read |
| --- | --- | ---: | --- |
| `vladimirfilip/fullhouse-engine` | 2026-06-02 | 4,066 | Aggregates many participant-style bot directories; highest signal. |
| `ds726/fullhouse-engine` | 2026-06-01 | 75 | Public `BOT_NAME = "ds726"` appears in portal, but not a qualifier. |
| `Benjamin-Yu-Sheng-Chang/fullhouse-hackathon` | 2026-05-31 | 312 | Many named strategy bots, but no direct qualifier match found yet. |
| `Mehedi-dev-2404/fullhouse-engine` | 2026-05-31 | 2,981 | Has `bots/mybot`; no direct qualifier match found yet. |
| `Con-TI/fullhouse-engine` | 2026-05-25 | 4,311 | Has `bots/mybot` and `mccfr.py`; likely not a qualifier by portal name. |
| `jickzx/fullhouse-engine` | 2026-05-19 | 32,560 | Large repo, but tree inspection did not expose a clear unique qualifier bot. |

## Portal Matches

Direct or plausible public-source matches against
`runs/portal_history/all_qualifier_public/strategy_report.json`:

| Public source signal | Portal bot | Official rank | Delta | Replay profile | Decision |
| --- | --- | ---: | ---: | --- | --- |
| `Pav1602_skantbot*`, `BOT_NAME = "SkantBot"` family | `skantbot8` | 52 | 102,988 | `sticky_station` | Qualifier target worth countering. |
| `BOT_NAME = "ds726"` | `ds726` | 186 | -53,808 | `tight_overfolder` by metrics | Not a qualifier target. |
| `Con-TI` fork name and conservative/equity theme | `Conservative Equitist` | 213 | -80,000 | non-qualifier metrics | Not enough evidence. |
| `Monte` name appears in public bot lists | `Monte` | 129 | 2,683 | `large_size_jammer` by metrics | Not a qualifier target from current evidence. |

`skantbot8` replay metrics from the selected all-public qualifier report:

| Metric | Value |
| --- | ---: |
| Official matches | 13 |
| Hands alive | 4,444 |
| VPIP | 0.3877 |
| PFR | 0.2966 |
| Raise rate | 0.2719 |
| Call rate | 0.1541 |
| Fold rate | 0.3822 |
| Pressure fold rate | 0.6117 |
| Pressure call rate | 0.3883 |
| All-in rate | 0.0044 |
| Average raise-to-pot | 1.707 |
| Showdown rate | 0.0810 |
| Selected-match busts | 5 of 13 |
| Selected-match wins of 50,000+ chips | 2 of 13 |

The portal profile builder classifies `skantbot8` as `sticky_station`. That is
the key correction to the source-code read: the public code family contains
many aggressive and overfolder-exploit controls, but the observed qualifier bot
calls enough under pressure that a pure anti-overfolder bluff plan is wrong.

## SkantBot Source Read

The public SkantBot family is not a simple threshold bot. It appears to use:

- position-aware preflop opening, 3-bet, call, 4-bet, and jam/call charts;
- mixed frequencies driven by deterministic per-hand randomness;
- opponent tracking for VPIP/PFR, fold-to-3bet, fold-to-cbet, and broad
  station/maniac labels;
- `eval7` equity, including range-weighted estimates when an aggressor range
  can be inferred;
- stack-risk guards that become cautious when a call or raise risks a large
  fraction of stack;
- low out-of-position bluff frequency and high equity thresholds for raises.

This source profile lines up with replay metrics on high VPIP/PFR and moderate
raise rate. It does not line up with an overfolder profile: replay pressure
fold is only 0.6117 and pressure call is 0.3883, so the real submitted version
behaved more like a sticky aggressive caller than a target for repeated bluffs.

## Counter Plan

Counter `skantbot8` and SkantBot-like public strategies as a
`sticky_aggressive_chart` subtype, not as a generic overfolder.

Implementation direction:

1. Add a replay profile guard that treats high-VPIP, high-PFR, moderate
   pressure-fold opponents as sticky-aggressive. This can be represented by
   the existing `sticky_station` path initially, with a future subtype only if
   tests show value.
2. Suppress low-equity bluffs and river blocker bluffs against this profile.
   Its source tracks pressure stats and its replay pressure-call rate is high
   enough to punish careless pressure.
3. Value bet larger than default with strong made hands, but keep raise-commit
   caps active because the public code has premium-heavy re-pressure and
   jam-or-fold branches.
4. Respect postflop raises and all-ins more than normal. The source uses high
   equity thresholds for raises and replay all-in frequency is low, so large
   re-pressure should be treated as value-skewed unless our equity is strong.
5. Avoid building a table image that folds too much to 3-bets. The source has
   fold-to-3bet exploitation, so the style-policy layer should defend enough
   medium-strong hands or 4-bet blockers within strict stack caps.
6. In early hands, apply small and medium value-weighted pressure before its
   public opponent tracker has much data, but do not use large speculative
   bluffs. The observed selected-match bust rate shows this family can lose
   stacks, but it also has large upside wins, so stack-trap counters need
   equity discipline.
7. Keep all SkantBot-specific implications parameterized. The robust
   optimizer should decide bluff suppression, value sizing, and call-off
   margins through held-out replay/profile suites rather than small samples.

Mock/test direction:

- Build a SkantBot-like mock profile from the portal row instead of copying
  public code: VPIP about 0.39, PFR about 0.30, raise rate about 0.27,
  pressure-call about 0.39, all-in about 0.004, average raise-to-pot about
  1.7.
- Include it in the portal-profile distribution as a sticky-aggressive variant.
- In candidate gates, require that anti-SkantBot changes improve or preserve
  performance against all sticky-station variants, not only this one public
  match.

## Adoption Status

Evidence-backed adoption:

- Use `sticky_station` and future sticky-aggressive guardrails against
  SkantBot-like public strategies.
- Suppress speculative pressure into this profile.
- Value-heavy pressure and tighter call-off response are justified by both
  source and replay metrics.

Implemented in `bots/replay_exploit_heuristic`:

- `skantbot` and `sticky_aggressive` ids map to `sticky_aggressive`.
- Online stats can classify sticky-aggressive opponents from high raise rate,
  enough call rate, bounded fold rate, and moderate pressure-fold rate.
- Style selection has tunable weights, and sticky-aggressive opponents bias
  style toward value while suppressing pressure.
- Sticky-aggressive call-off, bluff, opening, and re-pressure adjustments are
  exposed as `REPLAY_EXPLOIT_*` parameters for CEM instead of hard-coded
  adoption.

Not adopted:

- Do not copy preflop charts or source code.
- Do not assume all SkantBot variants fold to 3-bets or cbets.
- Do not tune directly on 13 selected matches or one public bot file.
