set -x

# Runtime backend.
ENGINE=vllm

ulimit -u 65536
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Model, data, and rollout scale.
MODELS_ROOT=${MODELS_ROOT:?Please set MODELS_ROOT}
MODEL_PATH=${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct}
TRAIN_DATA_SIZE=128
VAL_DATA_SIZE=512
GROUP_SIZE=8

TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"

# OPID advantage and teacher/OPD signal schedule.
OPID_MODE=mean_std_norm
OPID_STEP_ADV_W=0.0
OPID_EPISODE_SKILL_TEACHER_ADV_W=${OPID_EPISODE_SKILL_TEACHER_ADV_W:-0.001}
OPID_STEP_SKILL_TEACHER_ADV_W=${OPID_STEP_SKILL_TEACHER_ADV_W:-0.001}
OPID_OPD_START_AFTER_STEPS=${OPID_OPD_START_AFTER_STEPS:-null}
OPID_OPD_STOP_AFTER_STEPS=${OPID_OPD_STOP_AFTER_STEPS:-null}

# OPID episode filtering and teacher prompt construction.
OPID_FAILED_ONLY=${OPID_FAILED_ONLY:-False}
OPID_FAILED_ONLY_AFTER_STEPS=${OPID_FAILED_ONLY_AFTER_STEPS:-null}
OPID_FAILURE_SUCCESS_THRESHOLD=${OPID_FAILURE_SUCCESS_THRESHOLD:-1.0}

# OPID episode + critical-step skill analysis.
OPID_ENABLE_ANALYSIS=${OPID_ENABLE_ANALYSIS:-True}
OPID_SELECTOR=${OPID_SELECTOR:-llm}
OPID_ANALYSIS_BACKEND=openai
OPID_ANALYSIS_NUM_WORKERS=128
OPID_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ=${OPID_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ:-2}

# Experiment naming and output location.
PROJECT_NAME=agentic_search
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opid-grpo_qwen2.5_3b_search}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}

# Prompt observation history.
history_length=4

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opid \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.opd_loss_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.opid.step_advantage_w=$OPID_STEP_ADV_W \
    algorithm.opid.episode_skill_teacher_advantage_w=$OPID_EPISODE_SKILL_TEACHER_ADV_W \
    algorithm.opid.step_skill_teacher_advantage_w=$OPID_STEP_SKILL_TEACHER_ADV_W \
    algorithm.opid.opd_start_after_steps=$OPID_OPD_START_AFTER_STEPS \
    algorithm.opid.opd_stop_after_steps=$OPID_OPD_STOP_AFTER_STEPS \
    algorithm.opid.failed_only=$OPID_FAILED_ONLY \
    algorithm.opid.failed_only_after_steps=$OPID_FAILED_ONLY_AFTER_STEPS \
    algorithm.opid.failure_success_threshold=$OPID_FAILURE_SUCCESS_THRESHOLD \
    algorithm.opid.mode=$OPID_MODE \
    algorithm.opid.enable_analysis=$OPID_ENABLE_ANALYSIS \
    algorithm.opid.selector=$OPID_SELECTOR \
    algorithm.opid.analysis_backend=$OPID_ANALYSIS_BACKEND \
    algorithm.opid.analysis_num_workers=$OPID_ANALYSIS_NUM_WORKERS \
    algorithm.opid.analysis_max_completion_tokens=4096 \
    algorithm.opid.analysis_max_step_skills_per_traj=$OPID_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ \
    algorithm.opid.normalize_teacher_adv=False \
    env.history_length=$history_length \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$GROUP_SIZE \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=150 \
    trainer.total_training_steps=150 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir=$DEFAULT_LOCAL_DIR \
    $@
