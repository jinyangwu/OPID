#!/usr/bin/env bash

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT, e.g. /path/to/models}"
RELEASE_ROOT="${RELEASE_ROOT:-${MODELS_ROOT}/release}"

ALFWORLD_CKPT="${ALFWORLD_CKPT:?Please set ALFWORLD_CKPT to the FSDP actor checkpoint directory}"
ALFWORLD_TARGET="${ALFWORLD_TARGET:-${RELEASE_ROOT}/opid_3b_alfworld_step150}"
SCIWORLD_CKPT="${SCIWORLD_CKPT:?Please set SCIWORLD_CKPT to the FSDP actor checkpoint directory}"
SCIWORLD_TARGET="${SCIWORLD_TARGET:-${RELEASE_ROOT}/opid_3b_sciworld_step80}"

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "$ALFWORLD_CKPT" \
    --target_dir "$ALFWORLD_TARGET"

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "$SCIWORLD_CKPT" \
    --target_dir "$SCIWORLD_TARGET"
