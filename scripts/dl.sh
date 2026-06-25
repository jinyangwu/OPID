#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B-FP8}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-${MODEL_NAME##*/}}"
MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT, e.g. /path/to/models}"
LOCAL_DIR="${LOCAL_DIR:-${MODELS_ROOT}/${MODEL_DIR_NAME}}"

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

huggingface-cli download --resume-download "$MODEL_NAME" --local-dir "$LOCAL_DIR"
