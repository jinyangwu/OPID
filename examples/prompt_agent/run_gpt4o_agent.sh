#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-alfworld}"

if [[ "$ENV_NAME" == "alfworld" ]]; then
  echo "Launching AlfWorld agent..."
  python3 -m examples.prompt_agent.gpt4o_alfworld "$@"
elif [[ "$ENV_NAME" == "webshop" ]]; then
  echo "Launching WebShop agent..."
  python3 -m examples.prompt_agent.gpt4o_webshop "$@"
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfworld' or 'webshop'." >&2
  exit 1
fi
