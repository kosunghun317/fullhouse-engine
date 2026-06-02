# Qualifier 2 Strategy And Replay-Derived Training Plan

Reviewed: 2026-06-02.

This document is the working plan for the second-chance qualifier. It is based
on the public portal replay data, the existing heuristic/rollout submission
code, and the local training infrastructure. The goal is to stop guessing from
local mock competitors alone and instead build a replay-derived opponent pool
from actual qualifier behavior.

## Current Evidence

The portal data is public through Supabase REST with the anon key exposed by
the portal client. No participant auth token was needed for the public replay
tables.

The official `Fullhouse 2026 Qualifier` leaderboard is a materialized sum over
multiple internal qualifier tournaments. The public `matches`, `match_bots`,
`hands`, and nested `hand_winners` rows are sufficient to reconstruct match
outcomes and action tendencies.

Initial downloaded cohort:

- Output: `runs/portal_history/top16_plus_slop3/`
- Scope: official top 16 plus `slop3`
- Matches: 217
- Hands: 104,430
- Report: `runs/portal_history/top16_plus_slop3/strategy_report.json`

The broad all-public qualifier pull has also been completed locally:

- Output: `runs/portal_history/all_qualifier_public/`
- Matches: 675
- Hands: 310,799
- Bots: 298
- Report: `runs/portal_history/all_qualifier_public/strategy_report.json`
- Tracked manifest: `docs/portal-replay-artifacts.md`

Observed official top-16 tendencies:

| Group | Tendency | Strategic implication |
| --- | --- | --- |
| Top aggressive winners | VPIP roughly 0.28-0.46, PFR roughly 0.23-0.42, high pressure fold rates. | They win uncontested pots and fold disciplinedly when re-pressured. |
| Tight pressure bots | VPIP roughly 0.15-0.24, PFR roughly 0.11-0.21, high fold-to-pressure. | They can be attacked with selective steals, especially after passive checks. |
| Sticky/passive bots | Higher call rate, lower PFR, higher showdown rate. | Do not bluff them blindly; value size against them. |
| `slop3` | VPIP 0.3221, PFR 0.1555, call rate 0.2063, fold rate 0.4925, showdown 0.1391. | Too much passive calling and too little initiative relative to the top field. |

The main leak to attack is not a single size or threshold. The field appears to
contain many bots with mechanical pressure responses: they raise or fold well,
but call too narrowly when the bet is large enough. A strong Qualifier 2 bot
should therefore be an adaptive pressure/value bot rather than just a tuned
equity caller.

## Strategic Hypotheses

### H1: Initiative Is Underweighted

`slop3` had materially lower PFR and raise rate than most top finishers. The
new bot should move more EV from passive call/check lines into open raises,
isolating 3-bets, continuation pressure, and delayed probes.

Concrete target ranges:

- Overall VPIP: 0.25-0.38, not simply looser than current.
- PFR: 0.22-0.32 against mixed tables.
- Call rate: reduce from the current high passive-call profile unless pot odds
  and equity are clearly favorable.
- Pressure fold rate: keep high against large bets, but not so high that the
  bot folds exploitable medium-equity spots to frequent small pressure.

### H2: Field Overfolds To Re-Pressure

Many top bots have pressure-fold rates above 0.70. Use larger pressure sizes
against profiles that fold after facing raises, but only where the board and
range story are credible.

Candidate actions:

- 3-bet or squeeze preflop with premium, blocker, and high-card suited hands.
- Semi-bluff draws at pot sizes that cross common threshold-bot breakpoints.
- On river, convert low-showdown-value blocker hands into pressure against
  bots with high fold-to-pressure and low showdown rate.

### H3: Sticky Bots Need Opposite Treatment

Bots with high call rate and high showdown rate punish bluff-heavy plans. These
must be exploited with thinner value and smaller bluff frequency.

Candidate actions:

- Increase value size with strong one-pair+ hands only when board texture is
  not highly dangerous.
- Keep bluff sizes smaller or skip bluffs against stations.
- Widen value calls slightly against very aggressive sticky opponents only when
  owed-to-pot is small.

### H4: Threshold Sizes Are Exploitable

The adaptive threshold-rollout code already models mechanical thresholds.
Replay data should fit per-profile response curves instead of relying on
synthetic assumptions.

Candidate actions:

- Learn fold/call response as a function of bet fraction, street, board
  drawiness, previous aggression, and bot profile.
- Choose value/bluff sizes from expected value over inferred response curves.
- Avoid overfitting to bot names: train profile clusters and use online
  match-action evidence to classify opponents.

## Data Pipeline

### Public Table Flow

1. Download metadata:
   - `tournaments`
   - `matches`
   - `match_bots`
   - `bots`
   - `leaderboard`
2. Identify official leaderboard tournament:
   - default name: `Fullhouse 2026 Qualifier`
3. Identify source tournaments:
   - default name substring: `Qualifier`
4. Select bots:
   - by rank range, bot name, bot id, or explicit match id.
5. Download hand rows:
   - `hands`
   - embedded `hand_winners(bot_id,amount)`
6. Write:
   - small metadata JSON/NDJSON files,
   - streamed `hands.ndjson`,
   - per-match replay JSON under `matches/`,
   - strategy reports and derived profile artifacts.

All large replay outputs live under `runs/portal_history/`, which is not meant
to be committed. Code, docs, and small manifests may be committed.

### Immediate Data Targets

Use increasingly broad pulls:

1. `top16_plus_slop3`: fast high-signal field sample.
2. `top64_plus_slop3`: finalist-level field sample.
3. `all_qualifier_public`: every complete public internal qualifier match.

The all-match pull is useful if storage and time are acceptable. It should use
streaming writes only; do not hold all hand rows in memory.

### Commands

Use Poetry for every repo Python command:

```bash
poetry run python tools/download_portal_targeted_history.py \
  --rank-lte 64 \
  --bot-name slop3 \
  --output runs/portal_history/top64_plus_slop3

poetry run python tools/analyze_portal_strategy.py \
  runs/portal_history/top64_plus_slop3 \
  --output-json runs/portal_history/top64_plus_slop3/strategy_report.json
```

For all complete public qualifier matches, use the dedicated all-match mode
instead of simulating it through `--rank-lte 9999`:

```bash
poetry run python tools/download_portal_targeted_history.py \
  --all-complete-source-matches \
  --output runs/portal_history/all_qualifier_public
```

## Modular Tooling Plan

The portal workflow should be separated into reusable modules:

| Module | Responsibility |
| --- | --- |
| `tools/portal/__init__.py` | Package marker and public helper exports. |
| `tools/portal/client.py` | Supabase REST URL building, request retry, pagination, page-size cap, JWT/anon-key handling. |
| `tools/portal/schema.py` | Table specs, select columns, official/default tournament naming. |
| `tools/portal/download.py` | Metadata fetch, source-tournament resolution, selected hand streaming, per-match writers. |
| `tools/portal/analysis.py` | Action-log parsing, seat-to-bot mapping, bot-level strategy statistics. |
| `tools/portal/profiles.py` | Cluster/profile construction and artifact serialization. |
| `tools/download_portal_match_history.py` | Thin CLI wrapper for full table dumps. |
| `tools/download_portal_targeted_history.py` | Thin CLI wrapper for targeted/all-match replay downloads. |
| `tools/analyze_portal_strategy.py` | Thin CLI wrapper for strategy reports. |

This gives future training tools a clean import path instead of copying query
logic across scripts.

## Opponent Modeling Plan

### Replay Features

For each bot, infer:

- VPIP and PFR.
- Open raise frequency by position proxy.
- 3-bet / re-raise frequency where detectable.
- Call, fold, check, raise, all-in rates.
- Fold-to-pressure and call-to-pressure.
- Showdown rate and hand-win rate.
- Average raise-to-pot and distribution quantiles.
- Street-specific bet/check/fold patterns.
- Bet-size response curves: probability of fold/call/raise given bet fraction.

The portal action log is flat and lacks street markers, so street inference is
approximate. The parser should avoid overclaiming exact state reconstruction.
For robust training, prefer aggregated profile tendencies over brittle exact
hand-state labels.

### Profile Clusters

Build replay-derived clusters such as:

- `pressure_overfolder`: high PFR/raise, high fold-to-pressure, low showdown.
- `tight_overfolder`: low VPIP/PFR, high fold-to-pressure.
- `sticky_station`: high call rate, high showdown, low PFR.
- `large_size_jammer`: high all-in or high raise-size quantiles.
- `balanced_aggressor`: medium/high PFR with less extreme pressure fold.
- `passive_caller`: VPIP high, PFR low, call high.

Each profile should serialize to JSON with:

- selected bot ids and names,
- summary statistics,
- response curves,
- a mock-generation config,
- confidence/hand-count fields.

### Replay-Derived Mocks

Add mock competitors that sample actions from profile distributions rather than
from a single hand-coded strategy.

Runtime requirements:

- self-contained `bot.py` submission-like shape,
- no network,
- no writes,
- optional read-only `data/profile.json` or `profile.npz`,
- deterministic state-hash RNG where needed,
- action sanitizer identical in spirit to existing mocks.

Mock behavior should combine:

- preflop VPIP/PFR/3-bet policy from profile statistics,
- equity estimate from simple preflop score or eval7 rollout,
- facing-pressure response from learned fold/call/raise curves,
- street-specific aggression and sizing distributions,
- profile-specific value/bluff bias.

## Parameter Learning Plan

### Candidate Strategy Surface

The current heuristic exposes many `HEURISTIC_*` env knobs. The new replay
work should add or tune parameters around:

- preflop open score and open size,
- preflop 3-bet score, squeeze score, and 3-bet size,
- call margin against small/medium/large bets,
- bluff and semi-bluff frequencies by profile,
- value sizing by profile,
- river blocker bluff threshold,
- delayed probe frequency,
- fold-to-pressure exploit sizing,
- station value-size bonus and bluff suppression,
- max call-off risk for large bets,
- low-SPR commit threshold.

### Search Method

Use staged CEM/racing rather than manual one-off tuning:

1. Generate candidate env configs.
2. Evaluate against replay-derived mock profile pool.
3. Include current local strong mocks to avoid overfitting.
4. Score by risk-adjusted chip delta:
   - mean delta,
   - lower confidence bound,
   - p10 / downside tail,
   - bust frequency,
   - timeout/error count.
5. Promote only through held-out profile clusters and seed blocks.

The existing `tools/tune_heuristic_full_space.py` is the right optimizer shape.
Add a replay-derived suite registry rather than creating a separate optimizer
unless the suite wiring cannot support generated profile mocks cleanly.

### Training/Evaluation Splits

Split replay-derived profiles by bot id and source tournament:

- Training: most top/mid/low profiles across internal qualifier source runs.
- Validation: held-out source tournaments and held-out bot ids.
- Final gate: official top-64 profiles plus current local strong mocks.

Avoid training and validating on the exact same bot IDs whenever possible. If
the sample is small, use profile-level holdout instead of exact-name holdout.

### Promotion Gates

For a candidate to replace the incumbent:

- paired mean delta improves against replay-derived mocks,
- lower CI bound is positive or materially less negative with tail-risk gains,
- p10 does not collapse,
- bust/error rate does not regress,
- performance against existing strong mocks is not materially worse,
- validator passes,
- one official-style sandbox match passes against reference bots.

## Execution Roadmap

### Subtask 1: Document And Commit Plan

- Add this document.
- Update `docs/strategy-and-training.md`, `docs/training-pipelines.md`, and
  `skills/fullhouse-engine/SKILL.md` once the modular workflow exists.
- Commit before code restructuring.

### Subtask 2: Modularize Portal Tooling

- Create `tools/portal/`.
- Move REST and schema logic into modules.
- Keep current CLI behavior stable.
- Add tests for URL/pagination-free selection logic and action parser basics.
- Commit.

### Subtask 3: Broaden Data Pull

- Download top-64 plus `slop3`.
- If time/storage is acceptable, download all complete public qualifier
  matches using streaming mode.
- Commit code/docs only, not large replay artifacts.

### Subtask 4: Build Profile Artifacts

- Generate bot-level and cluster-level profile JSON from replay data.
- Store generated profiles under `runs/portal_history/<run>/profiles/`.
- Add a small checked-in example fixture only if tests need one.
- Commit tooling and tests.

### Subtask 5: Build Replay-Derived Mocks

- Add a profile mock template under `bots/mock_competitors/portal_profile/`.
- Generate concrete profile mock directories for local training/evaluation
  under `runs/portal_profile_mocks/` or a non-committed run directory.
- Add tests for action legality and profile loading.
- Commit.

### Subtask 6: Wire Tuning/Evaluation

- Add replay-derived suite selection to benchmark/tuning tools.
- Run smoke tuning, then a bounded serious run.
- Use paired gates before promoting a submission config.
- Update docs and skill commands.
- Commit.

## Near-Term Strategic Recommendation

The first candidate should not be a completely new RL model. It should be a
profile-aware pressure/value heuristic:

- Raise more preflop to reach top-field initiative levels.
- Reduce passive calls with mediocre hands.
- Add selective 3-bet/squeeze pressure against overfolders.
- Add river/blocker pressure against low-showdown, high-fold profiles.
- Suppress bluffs and widen value against sticky/passive callers.
- Use replay-fitted bet fractions instead of only local-mock hand-tuned sizes.

This is different from parameter-only tuning because the opponent pool and
profile response curves come from actual qualifier replays. It is still
compatible with the validator and the existing one-file submission shape.
