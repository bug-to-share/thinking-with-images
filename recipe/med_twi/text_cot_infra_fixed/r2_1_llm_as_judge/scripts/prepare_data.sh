#!/usr/bin/env bash
# Re-score newly regenerated fixed-infra candidates with Qwen3-14B.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Judge every row in the full English training set, then retain rows with
# 1--7 judge-correct samples.
INPUT_DATA="${TRAIN_DATA_FULL:-$PROJECT_DIR/data/med_twi/text_cot_infra_fixed/r1_text_cot_em/train.parquet}"
VAL_DATA="${VAL_DATA_FIXED:-$PROJECT_DIR/eval/benchmarks/SLAKE/views/slake_validation_en/validation.parquet}"
ARTIFACT_DIR="${FIXED_CURATION_DIR:-$PROJECT_DIR/data/med_twi/text_cot_infra_fixed/base_rollout_qwen3_vl_4b_n8}"
ROLLOUTS="$ARTIFACT_DIR/rollouts.jsonl"
OUTPUT_DATA="$ARTIFACT_DIR/llm_as_judge_qwen3_14b/train_correct_1_to_7.parquet"
SELECTION_OUTPUT="$ARTIFACT_DIR/llm_as_judge_qwen3_14b/selection.jsonl"
SUMMARY_OUTPUT="$ARTIFACT_DIR/llm_as_judge_qwen3_14b/summary.json"
PROGRESS_OUTPUT="$ARTIFACT_DIR/llm_as_judge_qwen3_14b/judge_progress_full.jsonl"
mkdir -p "$(dirname "$OUTPUT_DATA")"
python3 "$PROJECT_DIR/data/med_twi/scripts/filter_text_cot_rollouts_with_llm_judge.py" \
    --input "$INPUT_DATA" --rollouts "$ROLLOUTS" --validation "$VAL_DATA" \
    --reward-path "$SCRIPT_DIR/../rewards/text_cot_reward_llm_as_judge.py" \
    --output "$OUTPUT_DATA" --selection-output "$SELECTION_OUTPUT" \
    --summary-output "$SUMMARY_OUTPUT" --progress-output "$PROGRESS_OUTPUT" "$@"
