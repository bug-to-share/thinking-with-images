#!/usr/bin/env bash
# Build the R4 medium+hard subset from the same fixed-infra English rollouts.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
INPUT_DATA="${TRAIN_DATA_FULL:-$PROJECT_DIR/data/med_twi/text_cot_infra_fixed/r1_text_cot_em/train.parquet}"
VALIDATION_DATA="${VAL_DATA_FIXED:-$PROJECT_DIR/eval/benchmarks/SLAKE/views/slake_validation_en/validation.parquet}"
ARTIFACT_DIR="${FIXED_CURATION_DIR:-$PROJECT_DIR/data/med_twi/text_cot_infra_fixed/base_rollout_qwen3_vl_4b_n8}"
ROLLOUT_FILE="$ARTIFACT_DIR/rollouts.jsonl"
OUTPUT_DATA="$ARTIFACT_DIR/train_correct_0_to_7.parquet"
SELECTION_FILE="$ARTIFACT_DIR/selection_0_to_7.jsonl"
SUMMARY_FILE="$ARTIFACT_DIR/summary_0_to_7.json"
REWARD_PATH="$PROJECT_DIR/recipe/med_twi/text_cot_infra_fixed/r4_data_curation_medium_hard/rewards/text_cot_reward_fmtgate_em07_fmt03.py"

[[ -f "$ROLLOUT_FILE" ]] || {
    echo "Missing rollout file: $ROLLOUT_FILE. Run the R3 data-preparation script first." >&2
    exit 1
}

cd "$PROJECT_DIR"
python3 data/med_twi/scripts/filter_text_cot_by_base_rollout.py filter \
    --input "$INPUT_DATA" \
    --rollouts "$ROLLOUT_FILE" \
    --output "$OUTPUT_DATA" \
    --selection-output "$SELECTION_FILE" \
    --summary-output "$SUMMARY_FILE" \
    --validation "$VALIDATION_DATA" \
    --reward-path "$REWARD_PATH" \
    --samples 8 \
    --min-correct 0 \
    --max-correct 7
