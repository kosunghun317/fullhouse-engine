#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-heuristic-selfplay-$(date +%Y%m%d-%H%M%S)}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  if [[ -n "${RESULT_ROOT:-}" ]]; then
    RUN_ROOT="${RESULT_ROOT%/}/${RUN_ID}"
  else
    RUN_ROOT="runs/fullhouse_self_training/${RUN_ID}"
  fi
fi
if [[ -z "${GENERATED_DIR:-}" && -n "${GENERATED_ROOT:-}" ]]; then
  GENERATED_DIR="${GENERATED_ROOT%/}/${RUN_ID}"
fi
BASELINE_VALIDATION_PATH="${BASELINE_VALIDATION_PATH:-${RUN_ROOT}/baseline_validation.json}"

GENERATIONS="${GENERATIONS:-16}"
POPULATION="${POPULATION:-30}"
ELITE="${ELITE:-8}"
MATCHES_PER_GENERATION="${MATCHES_PER_GENERATION:-384}"
HANDS="${HANDS:-400}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-256}"
VALIDATION_HANDS="${VALIDATION_HANDS:-400}"
MIN_HEURISTICS="${MIN_HEURISTICS:-2}"
MAX_HEURISTICS="${MAX_HEURISTICS:-3}"
WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"

mkdir -p "$RUN_ROOT"
mkdir -p "$(dirname "$BASELINE_VALIDATION_PATH")"

GENERATED_ARGS=()
if [[ -n "${GENERATED_DIR:-}" ]]; then
  GENERATED_ARGS=(--generated-root "$GENERATED_DIR")
fi

echo "==> Multi-heuristic self-training: ${RUN_ID}"
poetry run python tools/strong_mocks/self_train_heuristic.py \
  --run-id "$RUN_ID" \
  --run-root "$RUN_ROOT" \
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
  "${GENERATED_ARGS[@]}" \
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
  --json > "$BASELINE_VALIDATION_PATH"

echo "Run root: ${RUN_ROOT}"
echo "Generated variants: ${GENERATED_DIR:-${RUN_ROOT}/generated}"
echo "Training summary: ${RUN_ROOT}/summary.json"
echo "Training metrics: ${RUN_ROOT}/metrics.jsonl"
echo "Progress plot: ${RUN_ROOT}/ev_progress.svg"
echo "Baseline validation: ${BASELINE_VALIDATION_PATH}"
