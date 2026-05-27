#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-real-opponents-$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-/private/tmp/fullhouse_real_training/${RUN_ID}}"
mkdir -p "$RESULT_ROOT"

WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
OPPONENT_POOL="${OPPONENT_POOL:-adversarial}"
EXPORT_BEST="${EXPORT_BEST:-1}"
SELECTION_WARMUP="${SELECTION_WARMUP:-4}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-250}"
MIN_EXPORT_MEAN_DELTA="${MIN_EXPORT_MEAN_DELTA:-1500}"
EXPORT_BEST_FLAG=()
if [[ "$EXPORT_BEST" != "0" && "$EXPORT_BEST" != "false" ]]; then
  EXPORT_BEST_FLAG=(--export-best)
fi

PPO_GENERATIONS="${PPO_GENERATIONS:-48}"
PPO_EARLY_STOP_PATIENCE="${PPO_EARLY_STOP_PATIENCE:-14}"
PPO_MATCHES_PER_GENERATION="${PPO_MATCHES_PER_GENERATION:-128}"
PPO_HANDS="${PPO_HANDS:-400}"
PPO_HIDDEN="${PPO_HIDDEN:-128}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4096}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
PPO_LR="${PPO_LR:-0.002}"
PPO_ENTROPY="${PPO_ENTROPY:-0.002}"
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.20}"
PPO_VALUE_COEF="${PPO_VALUE_COEF:-0.35}"
PPO_MAX_GRAD_NORM="${PPO_MAX_GRAD_NORM:-0.75}"
PPO_FEATURE_NORM_MOMENTUM="${PPO_FEATURE_NORM_MOMENTUM:-0.0}"
PPO_REPLAY_GENERATIONS="${PPO_REPLAY_GENERATIONS:-4}"
PPO_REPLAY_MAX_DECISIONS="${PPO_REPLAY_MAX_DECISIONS:-24000}"
PPO_TEMPERATURE="${PPO_TEMPERATURE:-0.62}"
PPO_REWARD_CLIP="${PPO_REWARD_CLIP:-5.0}"

BUCKET_GENERATIONS="${BUCKET_GENERATIONS:-40}"
BUCKET_EARLY_STOP_PATIENCE="${BUCKET_EARLY_STOP_PATIENCE:-24}"
BUCKET_MATCHES_PER_GENERATION="${BUCKET_MATCHES_PER_GENERATION:-128}"
BUCKET_HANDS="${BUCKET_HANDS:-400}"
BUCKET_COUNT="${BUCKET_COUNT:-32768}"
BUCKET_LR="${BUCKET_LR:-0.004}"
BUCKET_TEMPERATURE="${BUCKET_TEMPERATURE:-0.80}"
BUCKET_REWARD_CLIP="${BUCKET_REWARD_CLIP:-5.0}"

FAST_SMOKE_REPEAT="${FAST_SMOKE_REPEAT:-8}"
FAST_SMOKE_HANDS="${FAST_SMOKE_HANDS:-160}"
STRONG_SCREEN_SEEDS="${STRONG_SCREEN_SEEDS:-128}"
STRONG_SCREEN_HANDS="${STRONG_SCREEN_HANDS:-400}"
PPO_OUTPUT="${PPO_OUTPUT:-bots/strong_mocks/ppo_policy/data/policy.npz}"
BUCKET_OUTPUT="${BUCKET_OUTPUT:-bots/strong_mocks/cfr_bucket/data/policy.npz}"

echo "==> Real PPO-style opponent training: ${RUN_ID}"
poetry run python tools/strong_mocks/train_real_policy.py \
  --kind ppo \
  --generations "$PPO_GENERATIONS" \
  --matches-per-generation "$PPO_MATCHES_PER_GENERATION" \
  --hands "$PPO_HANDS" \
  --players 6 \
  --train-seats 2 \
  --hidden "$PPO_HIDDEN" \
  --learning-rate "$PPO_LR" \
  --entropy-coef "$PPO_ENTROPY" \
  --ppo-clip-ratio "$PPO_CLIP_RATIO" \
  --ppo-value-coef "$PPO_VALUE_COEF" \
  --max-grad-norm "$PPO_MAX_GRAD_NORM" \
  --feature-norm-momentum "$PPO_FEATURE_NORM_MOMENTUM" \
  --replay-generations "$PPO_REPLAY_GENERATIONS" \
  --replay-max-decisions "$PPO_REPLAY_MAX_DECISIONS" \
  --temperature "$PPO_TEMPERATURE" \
  --reward-clip "$PPO_REWARD_CLIP" \
  --epochs "$PPO_EPOCHS" \
  --batch-size "$PPO_BATCH_SIZE" \
  --opponent-pool "$OPPONENT_POOL" \
  --snapshot-interval 2 \
  --max-snapshots 8 \
  --snapshot-prob 0.35 \
  --selection-warmup "$SELECTION_WARMUP" \
  --early-stop-patience "$PPO_EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --min-export-mean-delta "$MIN_EXPORT_MEAN_DELTA" \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --seed 6161 \
  --output "$PPO_OUTPUT" \
  --log-jsonl "$RESULT_ROOT/ppo_real_training.jsonl" \
  --progress \
  "${EXPORT_BEST_FLAG[@]}" \
  --json

echo "==> Real bucket/CFR-like opponent training: ${RUN_ID}"
poetry run python tools/strong_mocks/train_real_policy.py \
  --kind bucket \
  --generations "$BUCKET_GENERATIONS" \
  --matches-per-generation "$BUCKET_MATCHES_PER_GENERATION" \
  --hands "$BUCKET_HANDS" \
  --players 6 \
  --train-seats 2 \
  --bucket-count "$BUCKET_COUNT" \
  --abstraction cfr-pokerbot \
  --bucket-update cfr-plus \
  --learning-rate "$BUCKET_LR" \
  --temperature "$BUCKET_TEMPERATURE" \
  --reward-clip "$BUCKET_REWARD_CLIP" \
  --opponent-pool "$OPPONENT_POOL" \
  --snapshot-interval 2 \
  --max-snapshots 8 \
  --snapshot-prob 0.35 \
  --selection-warmup "$SELECTION_WARMUP" \
  --early-stop-patience "$BUCKET_EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --min-export-mean-delta "$MIN_EXPORT_MEAN_DELTA" \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --seed 5151 \
  --output "$BUCKET_OUTPUT" \
  --log-jsonl "$RESULT_ROOT/bucket_real_training.jsonl" \
  --progress \
  "${EXPORT_BEST_FLAG[@]}" \
  --json

echo "==> Validate real-trained strong mock opponents"
poetry run python sandbox/validator.py bots/strong_mocks/ppo_policy --json
poetry run python sandbox/validator.py bots/strong_mocks/cfr_bucket --json
poetry run python sandbox/validator.py bots/strong_mocks/ensemble --json

echo "==> Fast unrestricted strong-mock smoke"
poetry run python -m training.fast_match \
  bots/heuristic \
  bots/strong_mocks/ppo_policy \
  bots/strong_mocks/cfr_bucket \
  bots/strong_mocks/ensemble \
  bots/strong_mocks/rollout_search \
  bots/shark \
  --hands "$FAST_SMOKE_HANDS" \
  --repeat "$FAST_SMOKE_REPEAT" \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --seed-start 9001 \
  --json > "$RESULT_ROOT/fast_strong_mock_smoke.json"

echo "==> Strong-screen validation after real opponent training"
poetry run python tools/select_heuristic_config.py \
  --preset strong-screen \
  --config baseline \
  --seed-count "$STRONG_SCREEN_SEEDS" \
  --hands "$STRONG_SCREEN_HANDS" \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --progress \
  --json > "$RESULT_ROOT/strong_screen_validation.json"

echo "Training logs: $RESULT_ROOT"
