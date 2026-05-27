#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-league-opponents-$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-/private/tmp/fullhouse_league_training}"
WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"

GENERATION_SCALE="${GENERATION_SCALE:-1.0}"
MATCH_SCALE="${MATCH_SCALE:-1.0}"
HAND_SCALE="${HAND_SCALE:-1.0}"
EVAL_SEEDS="${EVAL_SEEDS:-8}"
EVAL_HANDS="${EVAL_HANDS:-160}"
PROMOTE_MARGIN="${PROMOTE_MARGIN:-250}"
PROMOTE="${PROMOTE:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0}"
MIN_EXPORT_MEAN_DELTA="${MIN_EXPORT_MEAN_DELTA:--1000000000}"

PROMOTE_FLAG=()
if [[ "$PROMOTE" != "0" && "$PROMOTE" != "false" ]]; then
  PROMOTE_FLAG=(--promote)
fi

echo "==> League-style strong mock training: ${RUN_ID}"
poetry run python tools/strong_mocks/league_train.py \
  --run-id "$RUN_ID" \
  --result-root "$RESULT_ROOT" \
  --kind ppo \
  --kind bucket \
  --stage oracle_bootstrap \
  --stage public_adversarial \
  --stage league_mixed \
  --generation-scale "$GENERATION_SCALE" \
  --match-scale "$MATCH_SCALE" \
  --hand-scale "$HAND_SCALE" \
  --eval-seeds "$EVAL_SEEDS" \
  --eval-hands "$EVAL_HANDS" \
  --promote-margin "$PROMOTE_MARGIN" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --min-export-mean-delta "$MIN_EXPORT_MEAN_DELTA" \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --progress \
  "${PROMOTE_FLAG[@]}" \
  --json

echo "League logs: ${RESULT_ROOT}/${RUN_ID}"
