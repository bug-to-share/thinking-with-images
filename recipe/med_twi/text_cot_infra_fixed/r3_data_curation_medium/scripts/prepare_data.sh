#!/usr/bin/env bash
# Regenerate English n=8 candidates with the fixed SGLang visual loader.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
INPUT_DATA="${TRAIN_DATA_FULL:-$PROJECT_DIR/data/med_twi/text_cot_infra_fixed/r1_text_cot_em/train.parquet}"
VALIDATION_DATA="${VAL_DATA_FIXED:-$PROJECT_DIR/eval/benchmarks/SLAKE/views/slake_validation_en/validation.parquet}"
ARTIFACT_DIR="${FIXED_CURATION_DIR:-$PROJECT_DIR/data/med_twi/text_cot_infra_fixed/base_rollout_qwen3_vl_4b_n8}"
ROLLOUT_FILE="$ARTIFACT_DIR/rollouts.jsonl"
FILTERED_DATA="$ARTIFACT_DIR/train_correct_1_to_7.parquet"
SELECTION_FILE="$ARTIFACT_DIR/selection.jsonl"
SUMMARY_FILE="$ARTIFACT_DIR/summary.json"
BASE_URL="${BASE_URL:-http://127.0.0.1:30003}"
MODEL_PATH="${MODEL_PATH:-Qwen3-VL-4B-Instruct-patched}"
mkdir -p "$ARTIFACT_DIR"
python3 "$PROJECT_DIR/data/med_twi/scripts/filter_text_cot_by_base_rollout.py" rollout \
  --input "$INPUT_DATA" --output "$ROLLOUT_FILE" --base-url "$BASE_URL" --model "$MODEL_PATH" "$@"
python3 "$PROJECT_DIR/data/med_twi/scripts/filter_text_cot_by_base_rollout.py" filter \
  --input "$INPUT_DATA" --rollouts "$ROLLOUT_FILE" --output "$FILTERED_DATA" \
  --selection-output "$SELECTION_FILE" --summary-output "$SUMMARY_FILE" \
  --validation "$VALIDATION_DATA" --reward-path "$SCRIPT_DIR/../rewards/text_cot_reward_fmtgate_em07_fmt03.py" \
  --samples 8 --min-correct 1 --max-correct 7
