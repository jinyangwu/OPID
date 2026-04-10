set -x

ENGINE=vllm

ulimit -u 65536
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODELS_ROOT=${MODELS_ROOT:?Please set MODELS_ROOT}
MODEL_PATH=${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-1.5B-Instruct}
TRAIN_DATA_SIZE=16
VAL_DATA_SIZE=128
GROUP_SIZE=8
NUM_CPUS_PER_ENV_WORKER=0.1

COPD_MODE=mean_norm
COPD_SELECTOR=llm
COPD_ANALYSIS_BACKEND=openai
COPD_ANALYSIS_NUM_WORKERS=128
COPD_STEP_ADV_W=1
COPD_TEACHER_ADV_W=0.1
COPD_TEACHER_ADV_DISABLE_AFTER_STEPS=${COPD_TEACHER_ADV_DISABLE_AFTER_STEPS:-null}
COPD_STATS_MIN_GROUP_SIZE=2
COPD_STATS_VAR_QUANTILE=0.75
COPD_STATS_TOPK_PER_TRAJ=3
COPD_SIMILARITY_THRESH=0.95

PROJECT_NAME=agentic_alfworld
EXPERIMENT_NAME=copd_qwen2.5_1.5b_alfworld
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}

history_length=5
# We only use data preparation to indicate the modality and the data size.
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
    data.max_prompt_length=2048 \
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
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.copd.step_advantage_w=$COPD_STEP_ADV_W \
    algorithm.copd.teacher_advantage_w=$COPD_TEACHER_ADV_W \
    algorithm.copd.teacher_adv_disable_after_steps=$COPD_TEACHER_ADV_DISABLE_AFTER_STEPS \
    algorithm.copd.mode=$COPD_MODE \
    algorithm.copd.selector=$COPD_SELECTOR \
    algorithm.copd.enable_similarity=False \
    algorithm.copd.similarity_thresh=$COPD_SIMILARITY_THRESH \
    algorithm.copd.stats_min_group_size=$COPD_STATS_MIN_GROUP_SIZE \
    algorithm.copd.stats_var_quantile=$COPD_STATS_VAR_QUANTILE \
    algorithm.copd.stats_topk_per_traj=$COPD_STATS_TOPK_PER_TRAJ \
    algorithm.copd.stats_below_group_mean_only=True \
    algorithm.copd.analysis_backend=$COPD_ANALYSIS_BACKEND \
    algorithm.copd.analysis_num_workers=$COPD_ANALYSIS_NUM_WORKERS \
    algorithm.copd.analysis_max_history_steps=15 \
    algorithm.copd.analysis_max_completion_tokens=1024 \
    algorithm.copd.normalize_teacher_adv=False \
    env.history_length=$history_length \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir=$DEFAULT_LOCAL_DIR
