#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-heuristic-selfplay-$(date +%Y%m%d-%H%M%S)}"
GENERATED_ROOT="${GENERATED_ROOT:-/private/tmp/fullhouse_self_training/generated}"
RESULT_ROOT="${RESULT_ROOT:-/private/tmp/fullhouse_self_training/results}"

GENERATIONS="${GENERATIONS:-16}"
POPULATION="${POPULATION:-30}"
ELITE="${ELITE:-8}"
MATCHES_PER_GENERATION="${MATCHES_PER_GENERATION:-128}"
HANDS="${HANDS:-400}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-128}"
VALIDATION_HANDS="${VALIDATION_HANDS:-400}"
MIN_HEURISTICS="${MIN_HEURISTICS:-2}"
MAX_HEURISTICS="${MAX_HEURISTICS:-3}"
WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"

echo "==> Multi-heuristic self-training: ${RUN_ID}"
poetry run python tools/strong_mocks/self_train_heuristic.py \
  --run-id "$RUN_ID" \
  --generations "$GENERATIONS" \
  --population "$POPULATION" \
  --elite "$ELITE" \
  --matches-per-generation "$MATCHES_PER_GENERATION" \
  --hands "$HANDS" \
  --min-heuristics "$MIN_HEURISTICS" \
  --max-heuristics "$MAX_HEURISTICS" \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --seed 8080 \
  --generated-root "$GENERATED_ROOT" \
  --result-root "$RESULT_ROOT" \
  --json

echo "==> Focused post-self-training validation of current baseline"
poetry run python tools/select_heuristic_config.py \
  --preset strong-screen \
  --config baseline \
  --seed-count "$VALIDATION_SEEDS" \
  --hands "$VALIDATION_HANDS" \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --progress \
  --json > "$RESULT_ROOT/${RUN_ID}_baseline_validation.json"

echo "Generated variants: ${GENERATED_ROOT}/${RUN_ID}"
echo "Training results: ${RESULT_ROOT}/${RUN_ID}.json"
