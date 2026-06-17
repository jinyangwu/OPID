python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /raid3/data/GTPO/MODELS/ckpt/copd-grpo_qwen2.5_3b_alfworld_llm-5_episode-step-hint-v3_opd-adv-0.001_exp1-qwen3.5-27b/global_step_150/actor \
    --target_dir /raid3/data/GTPO/MODELS/release/opid_3b_alfworld_step150-qwen3.5-27b

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /raid3/data/GTPO/MODELS/ckpt/copd-grpo_qwen2.5_3b_sciworld_llm-5_episode-step-hint-v3_opd-adv-0.001/global_step_80/actor \
    --target_dir /raid3/data/GTPO/MODELS/release/opid_3b_sciworld_step80 