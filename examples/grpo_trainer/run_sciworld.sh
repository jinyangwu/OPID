set -x

ENGINE=${1:-vllm}
if [ "$#" -gt 0 ]; then
    shift
fi

ulimit -u 65536
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

MODELS_ROOT=${MODELS_ROOT:?Please set MODELS_ROOT}
MODEL_PATH=${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-1.5B-Instruct}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.2}

MAX_STEPS=${MAX_STEPS:-30}
HISTORY_LENGTH=${HISTORY_LENGTH:-5}

# ScienceWorld-specific settings:
# GENERALIZATION_LEVEL: 0 / 1 / 2
# SIMPLIFICATIONS_PRESET: easy / medium / hard / empty string
# SCIWORLD_JAR_PATH: set to null to use the builtin ScienceWorld jar
GENERALIZATION_LEVEL=${GENERALIZATION_LEVEL:-0}
SIMPLIFICATIONS_PRESET=${SIMPLIFICATIONS_PRESET-easy}
SCIWORLD_ENV_STEP_LIMIT=${SCIWORLD_ENV_STEP_LIMIT:-$MAX_STEPS}
SCIWORLD_JAR_PATH=${SCIWORLD_JAR_PATH:-null}

PROJECT_NAME=${PROJECT_NAME:-agentic_sciworld}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_qwen2.5_1.5b_sciworld}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TP_SIZE=${TP_SIZE:-1}

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gigpo.step_advantage_w=0.0 \
    algorithm.gigpo.mode=mean_std_norm \
    env.env_name=sciworld \
    env.seed=0 \
    env.max_steps=$MAX_STEPS \
    env.rollout.n=$GROUP_SIZE \
    env.history_length=$HISTORY_LENGTH \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
    +env.sciworld.generalization_level=$GENERALIZATION_LEVEL \
    +env.sciworld.simplifications_preset=$SIMPLIFICATIONS_PRESET \
    +env.sciworld.env_step_limit=$SCIWORLD_ENV_STEP_LIMIT \
    +env.sciworld.jar_path=$SCIWORLD_JAR_PATH \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir=$DEFAULT_LOCAL_DIR \
    $@
