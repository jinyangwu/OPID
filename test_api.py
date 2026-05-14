#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from utils import chat_completion_with_retry, create_openai_client, extract_message_text


REPO_ROOT = Path(__file__).resolve().parent


def load_env_file(env_file: str) -> Dict[str, str]:
    """Load a shell-style .env file into os.environ.

    This supports the common lines used by `source .env`, including:
      export KEY=value
      KEY="value"
      KEY='value'
    Values from the file overwrite the current process environment, matching
    the behavior of running `source .env` in a shell.
    """
    loaded: Dict[str, str] = {}
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = REPO_ROOT / env_path
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value
        loaded[key] = value
    return loaded


def mask_secret(value: Optional[str], keep: int = 4) -> str:
    if not value:
        return "<missing>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep * 2) + value[-keep:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test an OpenAI-compatible API after loading .env.")
    parser.add_argument("--env-file", default=".env", help="Env file to load first. Default: .env")
    parser.add_argument("--api-key", default=None, help="Override OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL.")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL.")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: okasdasdas",
        help="Prompt used for the connectivity test.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Max completion tokens for the test request. Default: 64",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Client timeout in seconds. Default: 60",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Retry attempts. Defaults to OPENAI_API_RETRIES or 3 after loading .env.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=None,
        help="Initial retry delay. Defaults to OPENAI_API_RETRY_DELAY or 1.0 after loading .env.",
    )
    parser.add_argument(
        "--dump-response",
        action="store_true",
        help="Print raw response JSON after a successful request.",
    )
    return parser.parse_args()


def get_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
        "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL"),
        "model": args.model or os.environ.get("OPENAI_MODEL"),
        "prompt": args.prompt,
        "max_completion_tokens": args.max_completion_tokens
        if args.max_completion_tokens is not None
        else int(os.environ.get("OPENAI_API_MAX_COMPLETION_TOKENS", "64")),
        "temperature": args.temperature,
        "timeout": args.timeout
        if args.timeout is not None
        else float(os.environ.get("OPENAI_API_TIMEOUT", "60")),
        "retries": args.retries
        if args.retries is not None
        else int(os.environ.get("OPENAI_API_RETRIES", "3")),
        "retry_delay": args.retry_delay
        if args.retry_delay is not None
        else float(os.environ.get("OPENAI_API_RETRY_DELAY", "1.0")),
        "dump_response": args.dump_response,
    }


def print_error(exc: BaseException) -> None:
    print("\nRequest failed.", file=sys.stderr)
    print(f"Exception type: {type(exc).__name__}", file=sys.stderr)
    print(f"Exception: {exc}", file=sys.stderr)

    cause = exc.__cause__ or exc
    if cause is not exc:
        print(f"caused_by: {type(cause).__name__}: {cause}", file=sys.stderr)
    status_code = getattr(cause, "status_code", None)
    if status_code is not None:
        print(f"status_code: {status_code}", file=sys.stderr)
    body = getattr(cause, "body", None)
    if body is not None:
        print("error_body:", file=sys.stderr)
        try:
            print(json.dumps(body, ensure_ascii=False, indent=2), file=sys.stderr)
        except TypeError:
            print(body, file=sys.stderr)


def main() -> int:
    args = parse_args()
    loaded_env = load_env_file(args.env_file)
    config = get_config(args)

    print("API test configuration:")
    print(f"  env_file: {args.env_file}")
    print(f"  env_loaded: {'yes' if loaded_env else 'no'}")
    print(f"  base_url: {config['base_url'] or '<default>'}")
    print(f"  model: {config['model'] or '<missing>'}")
    print(f"  api_key: {mask_secret(config['api_key'])}")
    print(f"  retries: {config['retries']}")
    print(f"  timeout: {config['timeout']}")

    missing = [name for name in ("api_key", "model") if not config[name]]
    if missing:
        print(f"\nMissing required configuration: {', '.join(missing)}", file=sys.stderr)
        print("Set them in .env or pass --api-key/--model.", file=sys.stderr)
        return 2

    try:
        client = create_openai_client(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
        )
        response = chat_completion_with_retry(
            client=client,
            model=config["model"],
            messages=[{"role": "user", "content": config["prompt"]}],
            retries=config["retries"],
            retry_delay=config["retry_delay"],
            temperature=config["temperature"],
            max_completion_tokens=config["max_completion_tokens"],
            return_response=True,
        )
    except Exception as exc:
        print_error(exc)
        return 1

    text = extract_message_text(response)
    print("\nRequest succeeded.")
    print(f"response_id: {getattr(response, 'id', '<unknown>')}")
    print(f"response_model: {getattr(response, 'model', '<unknown>')}")
    print(f"content: {text!r}")

    if config["dump_response"]:
        print("\nRaw response:")
        try:
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
        except Exception:
            print(repr(response))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
