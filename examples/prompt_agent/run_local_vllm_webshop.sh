#!/usr/bin/env bash

set -euo pipefail

ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH to a local HF model/checkpoint path.}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}/v1}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-EMPTY}}"

VLLM_BIN="${VLLM_BIN:-vllm}"
START_VLLM="${START_VLLM:-1}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-logs/vllm}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-${VLLM_LOG_DIR}/webshop_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"

ENV_NUM="${ENV_NUM:-128}"
TEST_TIMES="${TEST_TIMES:-1}"
MAX_STEPS="${MAX_STEPS:-30}"
HISTORY_LENGTH="${HISTORY_LENGTH:-2}"
WEBSHOP_HUMAN_GOALS="${WEBSHOP_HUMAN_GOALS:-0}"
WEBSHOP_USE_SMALL="${WEBSHOP_USE_SMALL:-0}"
TEMPERATURE="${TEMPERATURE:-0.4}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-512}"
REQUEST_WORKERS="${REQUEST_WORKERS:-$ENV_NUM}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-logs/webshop_local_vllm}"

server_pid=""

is_server_ready() {
  curl -fsS "${BASE_URL}/models" >/dev/null 2>&1
}

cleanup() {
  if [[ -n "$server_pid" && "$KEEP_VLLM_ALIVE" != "1" ]]; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "$START_VLLM" == "1" ]]; then
  if is_server_ready; then
    :
  else
    mkdir -p "$VLLM_LOG_DIR"
    read -r -a VLLM_EXTRA_ARGS_ARRAY <<< "${VLLM_EXTRA_ARGS:-}"

    "$VLLM_BIN" serve "$MODEL_PATH" \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --data-parallel-size "$DATA_PARALLEL_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --dtype "$DTYPE" \
      "${VLLM_EXTRA_ARGS_ARRAY[@]}" \
      >"$VLLM_LOG_FILE" 2>&1 &
    server_pid=$!

    deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
    until is_server_ready; do
      if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        echo "vLLM server exited before becoming ready. See ${VLLM_LOG_FILE}" >&2
        exit 1
      fi
      if (( SECONDS >= deadline )); then
        echo "Timed out waiting for vLLM at ${BASE_URL}. See ${VLLM_LOG_FILE}" >&2
        exit 1
      fi
      sleep 2
    done
  fi
fi

WEBSHOP_EXTRA_ARGS=()
if [[ "$WEBSHOP_USE_SMALL" == "1" || "$WEBSHOP_USE_SMALL" == "true" || "$WEBSHOP_USE_SMALL" == "True" ]]; then
  WEBSHOP_EXTRA_ARGS+=(--use-small)
fi

python3 -m examples.prompt_agent.local_vllm_webshop \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --model-name "$MODEL_NAME" \
  --env-num "$ENV_NUM" \
  --test-times "$TEST_TIMES" \
  --max-steps "$MAX_STEPS" \
  --history-length "$HISTORY_LENGTH" \
  --human-goals "$WEBSHOP_HUMAN_GOALS" \
  --temperature "$TEMPERATURE" \
  --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
  --request-workers "$REQUEST_WORKERS" \
  --log-dir "$EVAL_LOG_DIR" \
  "${WEBSHOP_EXTRA_ARGS[@]}" \
  "$@"
