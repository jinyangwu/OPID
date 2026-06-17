set -x

# Runtime backend.
ENGINE=vllm

ulimit -u 65536
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Model, data, and rollout scale.
MODELS_ROOT=${MODELS_ROOT:?Please set MODELS_ROOT}
MODEL_PATH=${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-1.5B-Instruct}
TRAIN_DATA_SIZE=16
VAL_DATA_SIZE=128
GROUP_SIZE=8
NUM_CPUS_PER_ENV_WORKER=0.1

# SOPD advantage and teacher/OPD signal schedule.
SOPD_MODE=mean_norm
SOPD_STEP_ADV_W=${SOPD_STEP_ADV_W:-1.0}
SOPD_STEP_HINT_TEACHER_ADV_W=${SOPD_STEP_HINT_TEACHER_ADV_W:-0.001}
SOPD_OPD_START_AFTER_STEPS=${SOPD_OPD_START_AFTER_STEPS:-null}
SOPD_OPD_STOP_AFTER_STEPS=${SOPD_OPD_STOP_AFTER_STEPS:-null}

# SOPD singleton grouping and analysis.
SOPD_ENABLE_ANALYSIS=${SOPD_ENABLE_ANALYSIS:-True}
SOPD_SELECTOR=${SOPD_SELECTOR:-llm}
SOPD_ANALYSIS_BACKEND=${SOPD_ANALYSIS_BACKEND:-openai}
SOPD_ANALYSIS_NUM_WORKERS=${SOPD_ANALYSIS_NUM_WORKERS:-128}
SOPD_ANALYSIS_MAX_STEP_HINTS_PER_TRAJ=${SOPD_ANALYSIS_MAX_STEP_HINTS_PER_TRAJ:-64}
SOPD_ENABLE_SIMILARITY=${SOPD_ENABLE_SIMILARITY:-False}
SOPD_SIMILARITY_THRESH=${SOPD_SIMILARITY_THRESH:-0.95}
SOPD_SAVE_STATE_GROUP_METRICS=${SOPD_SAVE_STATE_GROUP_METRICS:-True}
SOPD_STATE_GROUP_DUMP_DIR=${SOPD_STATE_GROUP_DUMP_DIR:-null}

# Experiment naming and output location.
PROJECT_NAME=agentic_webshop
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sopd_qwen2.5_1.5b_webshop_singleton_opd}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}

# Prompt observation history.
history_length=2

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=copd \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=6000 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.opd_loss_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.copd.singleton_only=True \
    algorithm.copd.step_advantage_w=$SOPD_STEP_ADV_W \
    algorithm.copd.episode_hint_teacher_advantage_w=0.0 \
    algorithm.copd.step_hint_teacher_advantage_w=$SOPD_STEP_HINT_TEACHER_ADV_W \
    algorithm.copd.opd_start_after_steps=$SOPD_OPD_START_AFTER_STEPS \
    algorithm.copd.opd_stop_after_steps=$SOPD_OPD_STOP_AFTER_STEPS \
    algorithm.copd.failed_only=False \
    algorithm.copd.failed_only_after_steps=null \
    algorithm.copd.mode=$SOPD_MODE \
    algorithm.copd.enable_analysis=$SOPD_ENABLE_ANALYSIS \
    algorithm.copd.selector=$SOPD_SELECTOR \
    algorithm.copd.analysis_backend=$SOPD_ANALYSIS_BACKEND \
    algorithm.copd.analysis_num_workers=$SOPD_ANALYSIS_NUM_WORKERS \
    algorithm.copd.analysis_max_completion_tokens=4096 \
    algorithm.copd.analysis_max_step_hints_per_traj=$SOPD_ANALYSIS_MAX_STEP_HINTS_PER_TRAJ \
    algorithm.copd.enable_similarity=$SOPD_ENABLE_SIMILARITY \
    algorithm.copd.similarity_thresh=$SOPD_SIMILARITY_THRESH \
    algorithm.copd.save_state_group_metrics=$SOPD_SAVE_STATE_GROUP_METRICS \
    algorithm.copd.state_group_dump_dir=$SOPD_STATE_GROUP_DUMP_DIR \
    algorithm.copd.normalize_teacher_adv=False \
    env.history_length=$history_length \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir=$DEFAULT_LOCAL_DIR \
    "$@"
