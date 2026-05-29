#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-coevolve-current}"
RESULT_ROOT="${RESULT_ROOT:-runs/fullhouse_coevolution}"
WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"

CYCLES="${CYCLES:-4}"
PPO_ARMS="${PPO_ARMS:-stable,explore,conservative}"
PPO_GENERATIONS="${PPO_GENERATIONS:-20}"
PPO_MATCHES_PER_GENERATION="${PPO_MATCHES_PER_GENERATION:-128}"
PPO_HANDS="${PPO_HANDS:-400}"
PPO_HIDDEN="${PPO_HIDDEN:-128}"
PPO_INIT="${PPO_INIT:-auto}"
PPO_INIT_SAMPLES="${PPO_INIT_SAMPLES:-80000}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4096}"
PPO_REPLAY_GENERATIONS="${PPO_REPLAY_GENERATIONS:-4}"
PPO_REPLAY_MAX_DECISIONS="${PPO_REPLAY_MAX_DECISIONS:-24000}"
PPO_MIN_TRAINING_HANDS="${PPO_MIN_TRAINING_HANDS:-1000000}"
HEURISTIC_POPULATION="${HEURISTIC_POPULATION:-16}"
HEURISTIC_ELITE="${HEURISTIC_ELITE:-4}"
HEURISTIC_MATCHES_PER_GENERATION="${HEURISTIC_MATCHES_PER_GENERATION:-384}"
HEURISTIC_HANDS="${HEURISTIC_HANDS:-400}"
EVAL_SEEDS="${EVAL_SEEDS:-128}"
EVAL_HANDS="${EVAL_HANDS:-400}"
OPPONENT_POOL="${OPPONENT_POOL:-adversarial}"
SELECTION_WARMUP="${SELECTION_WARMUP:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0}"
PROMOTE_PPO="${PROMOTE_PPO:-0}"
RESET="${RESET:-0}"
ALLOW_SMOKE="${ALLOW_SMOKE:-0}"

PPO_ARM_FLAGS=()
IFS=',' read -ra ARM_ITEMS <<< "$PPO_ARMS"
for arm in "${ARM_ITEMS[@]}"; do
  if [[ -n "$arm" ]]; then
    PPO_ARM_FLAGS+=(--ppo-arm "$arm")
  fi
done

PROMOTE_FLAG=()
if [[ "$PROMOTE_PPO" != "0" && "$PROMOTE_PPO" != "false" ]]; then
  PROMOTE_FLAG=(--promote-ppo)
fi

RESET_FLAG=()
if [[ "$RESET" != "0" && "$RESET" != "false" ]]; then
  RESET_FLAG=(--reset)
fi

SMOKE_FLAG=()
if [[ "$ALLOW_SMOKE" != "0" && "$ALLOW_SMOKE" != "false" ]]; then
  SMOKE_FLAG=(--allow-smoke)
fi

echo "==> E2E PPO/heuristic coevolution: ${RUN_ID}"
echo "==> Output directory: ${RESULT_ROOT}/${RUN_ID}"
poetry run python tools/coevolve_training.py \
  --run-id "$RUN_ID" \
  --result-root "$RESULT_ROOT" \
  --cycles "$CYCLES" \
  "${PPO_ARM_FLAGS[@]}" \
  --ppo-generations "$PPO_GENERATIONS" \
  --ppo-matches-per-generation "$PPO_MATCHES_PER_GENERATION" \
  --ppo-hands "$PPO_HANDS" \
  --ppo-hidden "$PPO_HIDDEN" \
  --ppo-init "$PPO_INIT" \
  --ppo-init-samples "$PPO_INIT_SAMPLES" \
  --ppo-epochs "$PPO_EPOCHS" \
  --ppo-batch-size "$PPO_BATCH_SIZE" \
  --ppo-replay-generations "$PPO_REPLAY_GENERATIONS" \
  --ppo-replay-max-decisions "$PPO_REPLAY_MAX_DECISIONS" \
  --ppo-min-training-hands "$PPO_MIN_TRAINING_HANDS" \
  --heuristic-population "$HEURISTIC_POPULATION" \
  --heuristic-elite "$HEURISTIC_ELITE" \
  --heuristic-matches-per-generation "$HEURISTIC_MATCHES_PER_GENERATION" \
  --heuristic-hands "$HEURISTIC_HANDS" \
  --eval-seeds "$EVAL_SEEDS" \
  --eval-hands "$EVAL_HANDS" \
  --opponent-pool "$OPPONENT_POOL" \
  --selection-warmup "$SELECTION_WARMUP" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --workers "$WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --progress \
  "${PROMOTE_FLAG[@]}" \
  "${RESET_FLAG[@]}" \
  "${SMOKE_FLAG[@]}" \
  --json

echo "Coevolution logs: ${RESULT_ROOT}/${RUN_ID}"
echo "Progress plot: ${RESULT_ROOT}/${RUN_ID}/ev_progress.svg"
