#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-tuned-submission-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/tuned_submission}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/build.log"
export RUN_ID RUN_ROOT
if [[ "${TUNED_SUBMISSION_LOGGING:-0}" != "1" ]]; then
  export TUNED_SUBMISSION_LOGGING=1
  "$0" "$@" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi

WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-dist/heuristic_bot.zip}"

TRAIN_STRONG_MOCKS="${TRAIN_STRONG_MOCKS:-1}"
RUN_TUNING="${RUN_TUNING:-1}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"

STRONG_RUN_ID="${STRONG_RUN_ID:-${RUN_ID}-strong-mocks}"
TUNE_RUN_ID="${TUNE_RUN_ID:-${RUN_ID}-heuristic-full-space}"
TUNE_RUN_ROOT="${TUNE_RUN_ROOT:-runs/heuristic_full_space_tuning/${TUNE_RUN_ID}}"
SUBMISSION_RUN_ID="${SUBMISSION_RUN_ID:-${RUN_ID}-submission}"

TUNE_PRESET="${TUNE_PRESET:-strong-screen}"
TUNE_GENERATIONS="${TUNE_GENERATIONS:-6}"
TUNE_POPULATION="${TUNE_POPULATION:-24}"
TUNE_ELITE="${TUNE_ELITE:-6}"
TUNE_STAGES="${TUNE_STAGES:-128:400:0.5,256:400:0.25}"
HEURISTIC_ENV_FILE="${HEURISTIC_ENV_FILE:-}"

enabled() {
  [[ "$1" != "0" && "$1" != "false" && "$1" != "False" ]]
}

finish() {
  local status=$?
  {
    echo "# Tuned Submission Build"
    echo
    echo "- run_id: \`${RUN_ID}\`"
    echo "- status: \`${status}\`"
    echo "- log: \`${LOG_FILE}\`"
    echo "- strong_mocks_run: \`${STRONG_RUN_ID}\`"
    echo "- tune_run_root: \`${TUNE_RUN_ROOT}\`"
    echo "- tuned_env: \`${HEURISTIC_ENV_FILE:-unset}\`"
    echo "- submission_run: \`${SUBMISSION_RUN_ID}\`"
    echo "- submission_zip: \`${SUBMISSION_ZIP}\`"
  } > "${RUN_DIR}/SUMMARY.md"
  echo
  if [[ "$status" -eq 0 ]]; then
    echo "==> Tuned submission build completed: ${RUN_DIR}"
    echo "==> Summary: ${RUN_DIR}/SUMMARY.md"
    echo "==> Upload candidate: ${SUBMISSION_ZIP}"
  else
    echo "==> Tuned submission build failed with status ${status}: ${RUN_DIR}"
    echo "==> Check log: ${LOG_FILE}"
  fi
}
trap finish EXIT

echo "==> Tuned submission build"
echo "==> run_id=${RUN_ID}"
echo "==> workers=${WORKERS} backend=${PARALLEL_BACKEND}"

if enabled "$TRAIN_STRONG_MOCKS"; then
  echo
  echo "==> Train, tune, and gate strong mocks"
  RUN_ID="$STRONG_RUN_ID" \
    WORKERS="$WORKERS" \
    PARALLEL_BACKEND="$PARALLEL_BACKEND" \
    scripts/train_all_strong_mocks.sh
fi

if enabled "$RUN_TUNING"; then
  echo
  echo "==> Tune heuristic full parameter space"
  poetry run python tools/tune_heuristic_full_space.py \
    --run-id "$TUNE_RUN_ID" \
    --run-root "$TUNE_RUN_ROOT" \
    --preset "$TUNE_PRESET" \
    --generations "$TUNE_GENERATIONS" \
    --population "$TUNE_POPULATION" \
    --elite "$TUNE_ELITE" \
    --stages "$TUNE_STAGES" \
    --workers "$WORKERS" \
    --parallel-backend "$PARALLEL_BACKEND" \
    --progress \
    --json | tee "${RUN_DIR}/heuristic_tuning.json"
fi

if [[ -z "$HEURISTIC_ENV_FILE" ]]; then
  if [[ -f "${TUNE_RUN_ROOT}/best_env.json" ]]; then
    HEURISTIC_ENV_FILE="${TUNE_RUN_ROOT}/best_env.json"
  elif [[ -f "${TUNE_RUN_ROOT}/best_env.sh" ]]; then
    HEURISTIC_ENV_FILE="${TUNE_RUN_ROOT}/best_env.sh"
  else
    echo "missing tuned env file under ${TUNE_RUN_ROOT}" >&2
    exit 2
  fi
fi
export HEURISTIC_ENV_FILE

if enabled "$RUN_PIPELINE"; then
  echo
  echo "==> Run submission pipeline with baked tuned env"
  RUN_ID="$SUBMISSION_RUN_ID" \
    WORKERS="$WORKERS" \
    PARALLEL_BACKEND="$PARALLEL_BACKEND" \
    SUBMISSION_ZIP="$SUBMISSION_ZIP" \
    HEURISTIC_ENV_FILE="$HEURISTIC_ENV_FILE" \
    scripts/run_submission_pipeline.sh
fi
