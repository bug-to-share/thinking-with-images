#!/usr/bin/env bash
# Instruct-model ZoomIn plus Retrieve agentic GRPO recipe.
# Run from any location; paths are resolved from the repository root.

set -euo pipefail

# Keep the guard in this entrypoint so wrapper recipes cannot bypass it.
if [[ "${MED_TWI_MEMORY_GUARD_ACTIVE:-0}" != "1" ]]; then
    AVAILABLE_MEMORY_THRESHOLD_GIB=${AVAILABLE_MEMORY_THRESHOLD_GIB:-60}
    CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-10}
    GUARD_EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_agentic_zoomin_retrieve}

    available_gib() {
        awk '/MemAvailable:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo
    }

    initial_available=$(available_gib)
    if (( initial_available < AVAILABLE_MEMORY_THRESHOLD_GIB )); then
        echo "[$(date '+%F %T')] MemAvailable=${initial_available} GiB < ${AVAILABLE_MEMORY_THRESHOLD_GIB} GiB; refusing to start ${GUARD_EXPERIMENT_NAME}" >&2
        exit 137
    fi

    MED_TWI_MEMORY_GUARD_ACTIVE=1 setsid bash "${BASH_SOURCE[0]}" "$@" &
    train_pid=$!

    cleanup_guarded_training() {
        kill -- -"$train_pid" 2>/dev/null || true
    }
    trap cleanup_guarded_training INT TERM

    while kill -0 "$train_pid" 2>/dev/null; do
        available=$(available_gib)
        if (( available < AVAILABLE_MEMORY_THRESHOLD_GIB )); then
            echo "[$(date '+%F %T')] MemAvailable=${available} GiB < ${AVAILABLE_MEMORY_THRESHOLD_GIB} GiB; stopping ${GUARD_EXPERIMENT_NAME}" >&2
            cleanup_guarded_training
            wait "$train_pid" || true
            exit 137
        fi
        sleep "$CHECK_INTERVAL_SECONDS"
    done

    wait "$train_pid"
    exit $?
fi

ulimit -n 65535

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export RAY_COLOR_PREFIX=0
export TOKENIZERS_PARALLELISM=false
export SLAKE_LANGUAGE=en
# Keep the judge API contract aligned with r2_1_llm_as_judge.
export LLM_AS_A_JUDGE_BASE=${LLM_AS_A_JUDGE_BASE:-http://127.0.0.1:30000/v1}
export LLM_AS_A_JUDGE_API_KEY=${LLM_AS_A_JUDGE_API_KEY:-EMPTY}
export LLM_AS_A_JUDGE_MODEL=${LLM_AS_A_JUDGE_MODEL:-Qwen/Qwen3-14B}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

PROJECT_NAME=${PROJECT_NAME:-med_twi}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_agentic_zoomin_retrieve}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export SLAKE_KG_DIR=${SLAKE_KG_DIR:-"$PROJECT_DIR/eval/benchmarks/SLAKE/KG"}

# TRAIN_FILE and TEST_FILE remain supported for compatibility with older launches.
TRAIN_DATA=${TRAIN_DATA:-${TRAIN_FILE:-"$PROJECT_DIR/data/med_twi/agentic_zoomin_retrieve/qwen3_vl_4b_instruct/train_en.parquet"}}
VAL_DATA=${VAL_DATA:-${TEST_FILE:-"$PROJECT_DIR/eval/benchmarks/SLAKE/views/slake_validation_en/validation.parquet"}}

CONFIG_DIR=${CONFIG_DIR:-"$PROJECT_DIR/recipe/med_twi/agentic_zoomin_retrieve/qwen3_vl_4b_instruct/config"}
CONFIG_NAME=${CONFIG_NAME:-zoomin_retrieve_grpo}
CONFIG_FILE="$CONFIG_DIR/$CONFIG_NAME.yaml"
TOOL_CONFIG=${TOOL_CONFIG:-"$CONFIG_DIR/tools.yaml"}
REWARD_FN=${REWARD_FN:-"$PROJECT_DIR/recipe/med_twi/agentic_zoomin_retrieve/qwen3_vl_4b_instruct/reward.py"}
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME"
LOG_DIR="$PROJECT_DIR/logs/med_twi/agentic_zoomin_retrieve/qwen3_vl_4b_instruct"
TENSORBOARD_DIR="$PROJECT_DIR/tensorboard_log/$PROJECT_NAME/$EXPERIMENT_NAME"

if [[ ! -f "$TRAIN_DATA" ]]; then
    python3 "$SCRIPT_DIR/prepare_slake_en_data.py" \
        --dataset-root "$PROJECT_DIR/eval/benchmarks/SLAKE" \
        --output "$TRAIN_DATA"
fi

for required_file in \
    "$TRAIN_DATA" \
    "$VAL_DATA" \
    "$CONFIG_FILE" \
    "$TOOL_CONFIG" \
    "$REWARD_FN"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$TENSORBOARD_DIR"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/verl:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATA=(
    data.seed=42
    data.train_files="$TRAIN_DATA"
    data.val_files="$VAL_DATA"
    data.train_batch_size=96
    data.val_batch_size=128
    data.dataloader_num_workers=0
    data.validation_shuffle=False
    data.max_prompt_length=16384
    data.max_response_length=8192
    data.return_raw_chat=True
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=64
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size=48
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.0
    'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra"]'
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=6
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.max_num_batched_tokens=32768
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.multi_stage_wake_up=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6
    actor_rollout_ref.rollout.multi_turn.max_user_turns=6
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
    actor_rollout_ref.rollout.multi_turn.format=hermes
    actor_rollout_ref.rollout.trace.token2text=True
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.ref.fsdp_config.param_offload=False
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.kl_ctrl.kl_coef=0.0
    reward.custom_reward_function.path="$REWARD_FN"
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.critic_warmup=0
    'trainer.logger=["console","tensorboard"]'
    trainer.log_val_generations=5
    trainer.val_before_train=False
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.default_local_dir="$CHECKPOINT_DIR"
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
    trainer.nnodes=1
    trainer.save_freq=10
    trainer.test_freq=5
    trainer.total_epochs=3
    +trainer.tensorboard_dir="$TENSORBOARD_DIR"
    +trainer.rl_logging_board_dir="$CHECKPOINT_DIR/rl_logging_board"
)

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${ALGORITHM[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee -a "$LOG_DIR/${EXPERIMENT_NAME}.log"
