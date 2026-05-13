#!/usr/bin/env bash
set -euo pipefail

# 默认在物理 GPU 4,5,6 上持续训练。需要调整时可以覆盖环境变量：
#   GPU_IDS=0,1 RESERVE_MEM_FRACTION=0.80 bash occupy_gpus_456.sh
GPU_IDS="${GPU_IDS:-4,5,6}"
PYTHON_BIN="${PYTHON_BIN:-/raid3/data/GTPO/conda_envs/skillrl/bin/python}"
RESERVE_MEM_FRACTION="${RESERVE_MEM_FRACTION:-${MEM_FRACTION:-0.90}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
LAYERS="${LAYERS:-4}"
STEPS="${STEPS:-0}"

"${PYTHON_BIN}" train.py \
  --gpu-ids "${GPU_IDS}" \
  --reserve-mem-fraction "${RESERVE_MEM_FRACTION}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --layers "${LAYERS}" \
  --steps "${STEPS}"
