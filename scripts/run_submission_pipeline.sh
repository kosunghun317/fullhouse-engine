#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-submission-$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-runs/submission_pipeline}"
RUN_DIR="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/pipeline.log"
export RUN_ID RESULT_ROOT
if [[ "${SUBMISSION_PIPELINE_LOGGING:-0}" != "1" ]]; then
  export SUBMISSION_PIPELINE_LOGGING=1
  "$0" "$@" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi

WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
HANDS="${HANDS:-400}"
INCUMBENT="${INCUMBENT:-baseline}"
CANDIDATES="${CANDIDATES:-profile-targeting-off line-aware blocker-probe pair-danger weakspot-control spr-anti-bucket}"

RUN_TESTS="${RUN_TESTS:-1}"
RUN_HARDEN_START="${RUN_HARDEN_START:-1}"
RUN_PAIRED_GATE="${RUN_PAIRED_GATE:-1}"
RUN_STRONG_SCREEN="${RUN_STRONG_SCREEN:-1}"
RUN_MOCK_FAMILY="${RUN_MOCK_FAMILY:-1}"
RUN_FINAL="${RUN_FINAL:-1}"
RUN_HARDEN_END="${RUN_HARDEN_END:-1}"

PROMOTION_PRESET="${PROMOTION_PRESET:-promotion}"
PROMOTION_SEED_COUNT="${PROMOTION_SEED_COUNT:-128}"
STRONG_SEED_COUNT="${STRONG_SEED_COUNT:-128}"
MOCK_SEED_COUNT="${MOCK_SEED_COUNT:-128}"
FINAL_SEED_COUNT="${FINAL_SEED_COUNT:-200}"
HARDEN_HANDS="${HARDEN_HANDS:-80}"
HARDEN_SEED="${HARDEN_SEED:-9901}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-dist/heuristic_bot.zip}"
HEURISTIC_ENV_FILE="${HEURISTIC_ENV_FILE:-}"

enabled() {
  [[ "$1" != "0" && "$1" != "false" && "$1" != "False" ]]
}

run_json() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  echo "==> command: $*"
  "$@" | tee "${RUN_DIR}/${label}.json"
}

finish() {
  local status=$?
  {
    echo "# Fullhouse Submission Pipeline"
    echo
    echo "- run_id: \`${RUN_ID}\`"
    echo "- status: \`${status}\`"
    echo "- log: \`${LOG_FILE}\`"
    echo "- submission_zip: \`${SUBMISSION_ZIP}\`"
    echo "- tuned_env: \`${HEURISTIC_ENV_FILE:-unset}\`"
    echo "- reports:"
    for report in "$RUN_DIR"/*.json; do
      [[ -e "$report" ]] || continue
      echo "  - \`${report}\`"
    done
  } > "${RUN_DIR}/SUMMARY.md"
  echo
  if [[ "$status" -eq 0 ]]; then
    echo "==> Submission pipeline completed: ${RUN_DIR}"
    echo "==> Summary: ${RUN_DIR}/SUMMARY.md"
    echo "==> Upload candidate: ${SUBMISSION_ZIP}"
  else
    echo "==> Submission pipeline failed with status ${status}: ${RUN_DIR}"
    echo "==> Check log: ${LOG_FILE}"
  fi
}
trap finish EXIT

echo "==> Fullhouse submission pipeline"
echo "==> run_id=${RUN_ID}"
echo "==> output=${RUN_DIR}"
echo "==> workers=${WORKERS} backend=${PARALLEL_BACKEND} hands=${HANDS}"
echo "==> incumbent=${INCUMBENT}"
echo "==> candidates=${CANDIDATES}"
if [[ -n "$HEURISTIC_ENV_FILE" ]]; then
  if [[ ! -f "$HEURISTIC_ENV_FILE" ]]; then
    echo "missing HEURISTIC_ENV_FILE: ${HEURISTIC_ENV_FILE}" >&2
    exit 2
  fi
  echo "==> tuned_env=${HEURISTIC_ENV_FILE}"
fi

harden_args=()
if [[ -n "$HEURISTIC_ENV_FILE" ]]; then
  harden_args=(--env-file "$HEURISTIC_ENV_FILE")
fi

if enabled "$RUN_TESTS"; then
  echo
  echo "==> pytest"
  poetry run pytest -q
fi

if enabled "$RUN_HARDEN_START"; then
  run_json "harden_start" \
    poetry run python tools/harden_submission.py \
      --output "$SUBMISSION_ZIP" \
      "${harden_args[@]}" \
      --hands "$HARDEN_HANDS" \
      --seed "$HARDEN_SEED" \
      --json
fi

candidate_args=()
for candidate in $CANDIDATES; do
  candidate_args+=(--candidate "$candidate")
done
selector_config_args=(--config "$INCUMBENT")
paired_candidate_args=("${candidate_args[@]}")
if [[ -n "$HEURISTIC_ENV_FILE" ]]; then
  selector_config_args=(--env-file "$HEURISTIC_ENV_FILE" --env-config-name tuned-env)
  paired_candidate_args=(--candidate-env-file "$HEURISTIC_ENV_FILE" --candidate-env-name tuned-env)
fi

if enabled "$RUN_PAIRED_GATE"; then
  run_json "paired_gate" \
    poetry run python tools/paired_heuristic_gate.py \
      --incumbent "$INCUMBENT" \
      "${paired_candidate_args[@]}" \
      --preset "$PROMOTION_PRESET" \
      --seed-count "$PROMOTION_SEED_COUNT" \
      --hands "$HANDS" \
      --workers "$WORKERS" \
      --parallel-backend "$PARALLEL_BACKEND" \
      --progress \
      --json
fi

if enabled "$RUN_STRONG_SCREEN"; then
  run_json "strong_screen" \
    poetry run python tools/select_heuristic_config.py \
      --preset strong-screen \
      "${selector_config_args[@]}" \
      --seed-count "$STRONG_SEED_COUNT" \
      --hands "$HANDS" \
      --workers "$WORKERS" \
      --parallel-backend "$PARALLEL_BACKEND" \
      --progress \
      --json
fi

if enabled "$RUN_MOCK_FAMILY"; then
  run_json "mock_family" \
    poetry run python tools/select_heuristic_config.py \
      --preset mock-family \
      "${selector_config_args[@]}" \
      --seed-count "$MOCK_SEED_COUNT" \
      --hands "$HANDS" \
      --workers "$WORKERS" \
      --parallel-backend "$PARALLEL_BACKEND" \
      --progress \
      --json
fi

if enabled "$RUN_FINAL"; then
  run_json "final_matrix" \
    poetry run python tools/select_heuristic_config.py \
      --preset final \
      "${selector_config_args[@]}" \
      --seed-count "$FINAL_SEED_COUNT" \
      --hands "$HANDS" \
      --workers "$WORKERS" \
      --parallel-backend "$PARALLEL_BACKEND" \
      --progress \
      --json
fi

if enabled "$RUN_HARDEN_END"; then
  run_json "harden_end" \
    poetry run python tools/harden_submission.py \
      --output "$SUBMISSION_ZIP" \
      "${harden_args[@]}" \
      --hands "$HARDEN_HANDS" \
      --seed "$((HARDEN_SEED + 100))" \
      --json
fi
