#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EVAL_ROOT="${MEDEVALKIT_DIR:-$PROJECT_ROOT/eval/MedEvalKit}"
if [[ ! -f "$EVAL_ROOT/eval.py" ]]; then
    echo "MedEvalKit is required. See docs/reproduction.md." >&2
    exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export EVAL_DATASETS="${EVAL_DATASETS:-SLAKE}"
export DATASETS_PATH="${DATASETS_PATH:-$PROJECT_ROOT/eval/benchmarks}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-VL}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
export CUSTOM_PROMPT_CONFIG="${CUSTOM_PROMPT_CONFIG:-${PROJECT_ROOT}/recipe/med_twi/text_cot_infra_fixed/r1_text_cot_em/config/medevalkit_text_cot_prompt.json}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-1}"
export TEMPERATURE="${TEMPERATURE:-0.5}"
export TOP_P="${TOP_P:-0.0001}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1}"
export REASONING="${REASONING:-False}"
export RUN_LLM_JUDGE="${RUN_LLM_JUDGE:-False}"
export OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/eval/results/med_twi/text_cot_infra_fixed/r1_text_cot_em}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export USE_VLLM="${USE_VLLM:-True}"
export SEED="${SEED:-42}"
export TEST_TIMES="${TEST_TIMES:-1}"
export JUDGE_MODEL_TYPE="${JUDGE_MODEL_TYPE:-openai}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-14B}"
export API_KEY="${API_KEY:-EMPTY}"
export BASE_URL="${BASE_URL:-http://127.0.0.1:30000/v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$EVAL_ROOT"
exec "$PYTHON_BIN" eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --custom_prompt_config "$CUSTOM_PROMPT_CONFIG" \
    --seed "$SEED" \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$RUN_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$JUDGE_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES" \
    "$@"
