#!/usr/bin/env bash
# Single-turn text-CoT GRPO baseline for SLAKE on Qwen3-VL-4B.

set -euo pipefail

# Keep the guard in this entrypoint so direct launches cannot bypass it.
if [[ "${MED_TWI_MEMORY_GUARD_ACTIVE:-0}" != "1" ]]; then
    AVAILABLE_MEMORY_THRESHOLD_GIB=${AVAILABLE_MEMORY_THRESHOLD_GIB:-80}
    CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-10}
    GUARD_EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_grpo_text_cot_infra_fixed_r3}

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
export VERL_QWEN3_VL_DEDUP_VISION_INPUTS="${VERL_QWEN3_VL_DEDUP_VISION_INPUTS:-1}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export RAY_COLOR_PREFIX=0
export TOKENIZERS_PARALLELISM=false
export SLAKE_LANGUAGE=en

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
USE_KL_LOSS=true
ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}

ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

# Match the optimizer-step budget of the 4919-sample R1/R2 runs:
# 1150 samples yield about 8 steps/epoch, so 15 epochs is about 120 steps.
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}

PROJECT_NAME=${PROJECT_NAME:-med_twi}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_grpo_text_cot_infra_fixed_r3}
########################### end user-adjustable ###########################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

TRAIN_DATA=${TRAIN_DATA:-"$PROJECT_DIR/data/med_twi/text_cot_infra_fixed/r3_data_curation_medium/train_correct_1_to_7.parquet"}
VAL_DATA=${VAL_DATA:-"$PROJECT_DIR/eval/benchmarks/SLAKE/views/slake_validation_en/validation.parquet"}
CONFIG_DIR="$PROJECT_DIR/recipe/med_twi/text_cot_infra_fixed/r3_data_curation_medium/config"
CONFIG_FILE="$CONFIG_DIR/text_cot_grpo.yaml"
REWARD_FN=${REWARD_FN:-"$PROJECT_DIR/recipe/med_twi/text_cot_infra_fixed/r3_data_curation_medium/rewards/text_cot_reward_fmtgate_em07_fmt03.py"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$PROJECT_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME"}
LOG_DIR=${LOG_DIR:-"$PROJECT_DIR/logs/med_twi/text_cot_infra_fixed/r3_data_curation_medium"}

for required_file in "$TRAIN_DATA" "$VAL_DATA" "$CONFIG_FILE" "$REWARD_FN"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        echo "Build the dataset with: python -m data.build_text_cot_dataset" >&2
        exit 1
    fi
done

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/verl:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=false
    data.seed=42
    data.train_files="$TRAIN_DATA"
    data.val_files="$VAL_DATA"
    data.image_key=images
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.val_batch_size="$VAL_BATCH_SIZE"
    data.max_prompt_length="$MAX_PROMPT_LENGTH"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=true
    data.filter_overlong_prompts_workers=32
    data.truncation=error
    data.return_raw_chat=true
    data.return_multi_modal_inputs=true
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.model.use_fused_kernels=false
    actor_rollout_ref.model.enable_gradient_checkpointing=true
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.use_dynamic_bsz=true
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS"
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF"
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL"
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.rollout.enable_chunked_prefill=true
    actor_rollout_ref.rollout.enforce_eager=false
    actor_rollout_ref.rollout.free_cache_engine=true
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.rollout.multi_turn.enable=false
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.ref.fsdp_config.param_offload=true
)

REWARD=(
    reward.custom_reward_function.path="$REWARD_FN"
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.balance_batch=true
    trainer.critic_warmup=0
    'trainer.logger=["console","tensorboard"]'
    trainer.log_val_generations=10
    trainer.val_before_train=true
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.default_local_dir="$CHECKPOINT_DIR"
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
    trainer.nnodes="$NNODES"
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.total_epochs="$TOTAL_EPOCHS"
    +trainer.tensorboard_dir="$CHECKPOINT_DIR/tensorboard"
    +trainer.rl_logging_board_dir="$CHECKPOINT_DIR/rl_logging_board"
)

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name=text_cot_grpo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee -a "$LOG_DIR/${EXPERIMENT_NAME}.log"
