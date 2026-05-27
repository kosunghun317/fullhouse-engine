#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-real-opponents-$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-/private/tmp/fullhouse_real_training/${RUN_ID}}"
mkdir -p "$RESULT_ROOT"

WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
OPPONENT_POOL="${OPPONENT_POOL:-mixed}"

PPO_GENERATIONS="${PPO_GENERATIONS:-48}"
PPO_MATCHES_PER_GENERATION="${PPO_MATCHES_PER_GENERATION:-72}"
PPO_HANDS="${PPO_HANDS:-120}"
PPO_HIDDEN="${PPO_HIDDEN:-128}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4096}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
PPO_LR="${PPO_LR:-0.006}"
PPO_ENTROPY="${PPO_ENTROPY:-0.006}"

BUCKET_GENERATIONS="${BUCKET_GENERATIONS:-40}"
BUCKET_MATCHES_PER_GENERATION="${BUCKET_MATCHES_PER_GENERATION:-96}"
BUCKET_HANDS="${BUCKET_HANDS:-120}"
BUCKET_COUNT="${BUCKET_COUNT:-32768}"
BUCKET_LR="${BUCKET_LR:-0.010}"

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
  --epochs "$PPO_EPOCHS" \
  --batch-size "$PPO_BATCH_SIZE" \
  --opponent-pool "$OPPONENT_POOL" \
  --snapshot-interval 2 \
  --max-snapshots 8 \
  --snapshot-prob 0.35 \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --seed 6161 \
  --output bots/strong_mocks/ppo_policy/data/policy.npz \
  --log-jsonl "$RESULT_ROOT/ppo_real_training.jsonl" \
  --progress \
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
  --opponent-pool "$OPPONENT_POOL" \
  --snapshot-interval 2 \
  --max-snapshots 8 \
  --snapshot-prob 0.35 \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --seed 5151 \
  --output bots/strong_mocks/cfr_bucket/data/policy.npz \
  --log-jsonl "$RESULT_ROOT/bucket_real_training.jsonl" \
  --progress \
  --json

echo "==> Validate real-trained strong mock opponents"
poetry run python sandbox/validator.py bots/strong_mocks/ppo_policy --json
poetry run python sandbox/validator.py bots/strong_mocks/cfr_bucket --json
poetry run python sandbox/validator.py bots/strong_mocks/ensemble --json

echo "==> Strong-screen smoke after real opponent training"
poetry run python tools/select_heuristic_config.py \
  --preset strong-screen \
  --config baseline \
  --seed-count 3 \
  --hands 120 \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --progress \
  --json

echo "Training logs: $RESULT_ROOT"
