export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export VERL_AUTO_PADDING=1

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
set -ex

bash examples/grpo_trainer/run_webshop_skills.sh


python scripts/test_openai_api.py --env-file .env --show-reasoning --dump-response --json-output

conda activate retriever
local_dir=$MODELS_ROOT/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir