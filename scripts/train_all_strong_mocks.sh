#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-strong-mocks-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/fullhouse_strong_mocks}"
WORKERS="${WORKERS:-0}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
OPPONENT_POOL="${OPPONENT_POOL:-adversarial}"

ALLOW_SMOKE="${ALLOW_SMOKE:-0}"
SKIP_GATE="${SKIP_GATE:-0}"
REQUIRE_GATE="${REQUIRE_GATE:-1}"
EXPORT_BEST="${EXPORT_BEST:-1}"
RESUME="${RESUME:-0}"
PROGRESS="${PROGRESS:-1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"

MIN_ORACLE_SAMPLES="${MIN_ORACLE_SAMPLES:-200000}"
MIN_TRAINING_HANDS="${MIN_TRAINING_HANDS:-1000000}"
SELECTION_WARMUP="${SELECTION_WARMUP:-4}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-250}"
SELECTION_BUST_PENALTY="${SELECTION_BUST_PENALTY:-9000}"
MIN_EXPORT_MEAN_DELTA="${MIN_EXPORT_MEAN_DELTA:-1500}"

ORACLE_SAMPLES="${ORACLE_SAMPLES:-200000}"
ORACLE_STYLE="${ORACLE_STYLE:-pressure}"
ORACLE_HIDDEN="${ORACLE_HIDDEN:-48}"

PPO_GENERATIONS="${PPO_GENERATIONS:-48}"
PPO_MATCHES_PER_GENERATION="${PPO_MATCHES_PER_GENERATION:-128}"
PPO_HANDS="${PPO_HANDS:-400}"
PPO_HIDDEN="${PPO_HIDDEN:-128}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4096}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-0.002}"
PPO_ENTROPY_COEF="${PPO_ENTROPY_COEF:-0.002}"
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.20}"
PPO_VALUE_COEF="${PPO_VALUE_COEF:-0.35}"
PPO_MAX_GRAD_NORM="${PPO_MAX_GRAD_NORM:-0.75}"
PPO_REPLAY_GENERATIONS="${PPO_REPLAY_GENERATIONS:-4}"
PPO_REPLAY_MAX_DECISIONS="${PPO_REPLAY_MAX_DECISIONS:-24000}"
PPO_TEMPERATURE="${PPO_TEMPERATURE:-0.62}"
PPO_REWARD_CLIP="${PPO_REWARD_CLIP:-5.0}"
PPO_EARLY_STOP_PATIENCE="${PPO_EARLY_STOP_PATIENCE:-14}"

BUCKET_GENERATIONS="${BUCKET_GENERATIONS:-40}"
BUCKET_MATCHES_PER_GENERATION="${BUCKET_MATCHES_PER_GENERATION:-128}"
BUCKET_HANDS="${BUCKET_HANDS:-400}"
BUCKET_COUNT="${BUCKET_COUNT:-32768}"
BUCKET_LEARNING_RATE="${BUCKET_LEARNING_RATE:-0.004}"
BUCKET_TEMPERATURE="${BUCKET_TEMPERATURE:-0.80}"
BUCKET_REWARD_CLIP="${BUCKET_REWARD_CLIP:-5.0}"
BUCKET_EARLY_STOP_PATIENCE="${BUCKET_EARLY_STOP_PATIENCE:-24}"

DEEP_GENERATIONS="${DEEP_GENERATIONS:-24}"
DEEP_MATCHES_PER_GENERATION="${DEEP_MATCHES_PER_GENERATION:-128}"
DEEP_HANDS="${DEEP_HANDS:-400}"
DEEP_BOOTSTRAP_SAMPLES="${DEEP_BOOTSTRAP_SAMPLES:-60000}"
DEEP_BOOTSTRAP_HOLDOUT="${DEEP_BOOTSTRAP_HOLDOUT:-12000}"
DEEP_BOOTSTRAP_EPOCHS="${DEEP_BOOTSTRAP_EPOCHS:-4}"

ARM_GENERATIONS="${ARM_GENERATIONS:-24}"
ARM_MATCHES_PER_GENERATION="${ARM_MATCHES_PER_GENERATION:-128}"
ARM_HANDS="${ARM_HANDS:-400}"
ARM_BOOTSTRAP_SAMPLES="${ARM_BOOTSTRAP_SAMPLES:-60000}"
ARM_BOOTSTRAP_HOLDOUT="${ARM_BOOTSTRAP_HOLDOUT:-12000}"
ARM_BOOTSTRAP_EPOCHS="${ARM_BOOTSTRAP_EPOCHS:-5}"

ARM_TUNE_GENERATIONS="${ARM_TUNE_GENERATIONS:-6}"
ARM_TUNE_POPULATION="${ARM_TUNE_POPULATION:-32}"
ARM_TUNE_STAGES="${ARM_TUNE_STAGES:-16:400:0.35,64:400:1.0}"
ARM_TUNE_MIN_HANDS_PER_CANDIDATE="${ARM_TUNE_MIN_HANDS_PER_CANDIDATE:-6400}"

GATE_SEED_COUNT="${GATE_SEED_COUNT:-128}"
GATE_HANDS="${GATE_HANDS:-400}"
GATE_MIN_TASKS_PER_CANDIDATE="${GATE_MIN_TASKS_PER_CANDIDATE:-512}"
GATE_MIN_MEAN_DELTA="${GATE_MIN_MEAN_DELTA:-500}"
GATE_MIN_WIN_RATE="${GATE_MIN_WIN_RATE:-0.55}"

CMD=(
  poetry run python tools/strong_mocks/train_all_strong_mocks.py
  --run-id "$RUN_ID"
  --output-root "$OUTPUT_ROOT"
  --min-oracle-samples "$MIN_ORACLE_SAMPLES"
  --min-training-hands "$MIN_TRAINING_HANDS"
  --workers "$WORKERS"
  --parallel-backend "$PARALLEL_BACKEND"
  --opponent-pool "$OPPONENT_POOL"
  --selection-warmup "$SELECTION_WARMUP"
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA"
  --selection-bust-penalty "$SELECTION_BUST_PENALTY"
  --min-export-mean-delta "$MIN_EXPORT_MEAN_DELTA"
  --oracle-samples "$ORACLE_SAMPLES"
  --oracle-style "$ORACLE_STYLE"
  --oracle-hidden "$ORACLE_HIDDEN"
  --ppo-generations "$PPO_GENERATIONS"
  --ppo-matches-per-generation "$PPO_MATCHES_PER_GENERATION"
  --ppo-hands "$PPO_HANDS"
  --ppo-hidden "$PPO_HIDDEN"
  --ppo-batch-size "$PPO_BATCH_SIZE"
  --ppo-epochs "$PPO_EPOCHS"
  --ppo-learning-rate "$PPO_LEARNING_RATE"
  --ppo-entropy-coef "$PPO_ENTROPY_COEF"
  --ppo-clip-ratio "$PPO_CLIP_RATIO"
  --ppo-value-coef "$PPO_VALUE_COEF"
  --ppo-max-grad-norm "$PPO_MAX_GRAD_NORM"
  --ppo-replay-generations "$PPO_REPLAY_GENERATIONS"
  --ppo-replay-max-decisions "$PPO_REPLAY_MAX_DECISIONS"
  --ppo-temperature "$PPO_TEMPERATURE"
  --ppo-reward-clip "$PPO_REWARD_CLIP"
  --ppo-early-stop-patience "$PPO_EARLY_STOP_PATIENCE"
  --bucket-generations "$BUCKET_GENERATIONS"
  --bucket-matches-per-generation "$BUCKET_MATCHES_PER_GENERATION"
  --bucket-hands "$BUCKET_HANDS"
  --bucket-count "$BUCKET_COUNT"
  --bucket-learning-rate "$BUCKET_LEARNING_RATE"
  --bucket-temperature "$BUCKET_TEMPERATURE"
  --bucket-reward-clip "$BUCKET_REWARD_CLIP"
  --bucket-early-stop-patience "$BUCKET_EARLY_STOP_PATIENCE"
  --deep-generations "$DEEP_GENERATIONS"
  --deep-matches-per-generation "$DEEP_MATCHES_PER_GENERATION"
  --deep-hands "$DEEP_HANDS"
  --deep-bootstrap-samples "$DEEP_BOOTSTRAP_SAMPLES"
  --deep-bootstrap-holdout "$DEEP_BOOTSTRAP_HOLDOUT"
  --deep-bootstrap-epochs "$DEEP_BOOTSTRAP_EPOCHS"
  --arm-generations "$ARM_GENERATIONS"
  --arm-matches-per-generation "$ARM_MATCHES_PER_GENERATION"
  --arm-hands "$ARM_HANDS"
  --arm-bootstrap-samples "$ARM_BOOTSTRAP_SAMPLES"
  --arm-bootstrap-holdout "$ARM_BOOTSTRAP_HOLDOUT"
  --arm-bootstrap-epochs "$ARM_BOOTSTRAP_EPOCHS"
  --arm-tune-generations "$ARM_TUNE_GENERATIONS"
  --arm-tune-population "$ARM_TUNE_POPULATION"
  --arm-tune-stages "$ARM_TUNE_STAGES"
  --arm-tune-min-hands-per-candidate "$ARM_TUNE_MIN_HANDS_PER_CANDIDATE"
  --gate-seed-count "$GATE_SEED_COUNT"
  --gate-hands "$GATE_HANDS"
  --gate-min-tasks-per-candidate "$GATE_MIN_TASKS_PER_CANDIDATE"
  --gate-min-mean-delta "$GATE_MIN_MEAN_DELTA"
  --gate-min-win-rate "$GATE_MIN_WIN_RATE"
  --json
)

if [[ -n "$ARTIFACT_ROOT" ]]; then
  CMD+=(--artifact-root "$ARTIFACT_ROOT")
fi
if [[ "$ALLOW_SMOKE" != "0" && "$ALLOW_SMOKE" != "false" ]]; then
  CMD+=(--allow-smoke)
fi
if [[ "$SKIP_GATE" != "0" && "$SKIP_GATE" != "false" ]]; then
  CMD+=(--skip-gate)
fi
if [[ "$REQUIRE_GATE" == "0" || "$REQUIRE_GATE" == "false" ]]; then
  CMD+=(--no-require-gate)
fi
if [[ "$EXPORT_BEST" == "0" || "$EXPORT_BEST" == "false" ]]; then
  CMD+=(--no-export-best)
fi
if [[ "$RESUME" != "0" && "$RESUME" != "false" ]]; then
  CMD+=(--resume)
fi
if [[ "$PROGRESS" != "0" && "$PROGRESS" != "false" ]]; then
  CMD+=(--progress)
fi

echo "==> Train, tune, and gate all strong mocks: ${RUN_ID}"
"${CMD[@]}"
echo "Run directory: ${OUTPUT_ROOT}/${RUN_ID}"
