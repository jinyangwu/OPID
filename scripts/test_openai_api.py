#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import build_prompt_dict, chat_completion_with_retry, create_openai_client, extract_message_text


def mask_secret(value: Optional[str], keep: int = 4) -> str:
    if not value:
        return "<missing>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep * 2) + value[-keep:]


def load_env_file(env_file: str) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    path = Path(env_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
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
        if key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal OpenAI-compatible API connectivity test using utils/openai_api.py."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file to load before resolving config. Defaults to .env in repo root.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to OPENAI_API_KEY after loading env file.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible endpoint. Defaults to OPENAI_BASE_URL after loading env file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to OPENAI_MODEL after loading env file.",
    )
    parser.add_argument(
        "--prompt",
        default="中国最长的河流是什么？",
        help="User prompt for the test request.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="Max completion tokens. Defaults to 32.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Defaults to 0.0.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Retry attempts. Defaults to OPENAI_API_RETRIES or 5.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=None,
        help="Initial retry delay in seconds. Defaults to OPENAI_API_RETRY_DELAY or 1.0.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Client timeout in seconds. Defaults to 60.",
    )
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="Print the exact messages payload before sending.",
    )
    parser.add_argument(
        "--dump-response",
        action="store_true",
        help="Print the raw response payload after the request returns.",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Print message.reasoning_content separately when available.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Ask the model to return only valid JSON and validate the response.",
    )
    parser.add_argument(
        "--json-schema",
        default='{"status": "ok", "answer": "string"}',
        help="Schema/example text appended to the JSON instruction when --json-output is enabled.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    model = args.model or os.environ.get("OPENAI_MODEL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    retries = args.retries if args.retries is not None else int(os.environ.get("OPENAI_API_RETRIES", "5"))
    retry_delay = (
        args.retry_delay
        if args.retry_delay is not None
        else float(os.environ.get("OPENAI_API_RETRY_DELAY", "1.0"))
    )
    max_completion_tokens = args.max_completion_tokens if args.max_completion_tokens is not None else 32

    return {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "retries": retries,
        "retry_delay": retry_delay,
        "max_completion_tokens": max_completion_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "prompt": args.prompt,
        "system_prompt": args.system_prompt,
        "show_messages": args.show_messages,
        "dump_response": args.dump_response,
        "show_reasoning": args.show_reasoning,
        "json_output": args.json_output,
        "json_schema": args.json_schema,
    }


def build_test_prompt(config: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = config["system_prompt"]
    if config["json_output"]:
        json_instruction = (
            "Return ONLY valid JSON. Do not include markdown fences, explanations, or extra text.\n"
            f"Use this schema/example:\n{config['json_schema']}"
        )
        system_prompt = f"{system_prompt}\n\n{json_instruction}" if system_prompt else json_instruction

    return build_prompt_dict(
        system_prompt=system_prompt,
        user_prompt=config["prompt"],
    )


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def main() -> int:
    args = parse_args()
    loaded_env = load_env_file(args.env_file)
    config = resolve_config(args)

    print("OpenAI-compatible API test configuration:")
    print(f"  env_file: {args.env_file}")
    print(f"  env_loaded: {'yes' if loaded_env else 'no'}")
    print(f"  base_url: {config['base_url'] or '<default>'}")
    print(f"  model: {config['model'] or '<missing>'}")
    print(f"  retries: {config['retries']}")
    print(f"  retry_delay: {config['retry_delay']}")
    print(f"  timeout: {config['timeout']}")
    print(f"  api_key: {mask_secret(config['api_key'])}")

    missing = [
        name
        for name, value in (
            ("api_key", config["api_key"]),
            ("model", config["model"]),
        )
        if not value
    ]
    if missing:
        print(f"Missing required configuration: {', '.join(missing)}", file=sys.stderr)
        return 2

    prompt = build_test_prompt(config)
    if config["show_messages"]:
        print("\nMessages payload:")
        print(json.dumps(prompt["messages"], ensure_ascii=False, indent=2))

    try:
        client = create_openai_client(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
        )
    except Exception as exc:
        print("\nClient initialization failed.")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception: {exc}")
        return 1

    print("\nSending request...")
    try:
        response = chat_completion_with_retry(
            client=client,
            model=config["model"],
            prompt=prompt,
            retries=config["retries"],
            retry_delay=config["retry_delay"],
            temperature=config["temperature"],
            max_completion_tokens=config["max_completion_tokens"],
            return_response=True,
        )
    except Exception as exc:
        print("\nRequest failed.")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception: {exc}")

        cause = exc.__cause__
        if cause is not None:
            print(f"caused_by: {type(cause).__name__}: {cause}")
            status_code = getattr(cause, "status_code", None)
            if status_code is not None:
                print(f"status_code: {status_code}")
            body = getattr(cause, "body", None)
            if body is not None:
                try:
                    print("error_body:")
                    print(json.dumps(body, ensure_ascii=False, indent=2))
                except Exception:
                    print(f"error_body: {body}")
        else:
            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                print(f"status_code: {status_code}")
            body = getattr(exc, "body", None)
            if body is not None:
                try:
                    print("error_body:")
                    print(json.dumps(body, ensure_ascii=False, indent=2))
                except Exception:
                    print(f"error_body: {body}")
        return 1

    print("\nRequest succeeded.")
    print(f"response_id: {getattr(response, 'id', '<unknown>')}")
    print(f"model: {getattr(response, 'model', '<unknown>')}")
    message = response.choices[0].message if getattr(response, "choices", None) else None
    if config["show_reasoning"] and message is not None:
        print(f"reasoning: {getattr(message, 'reasoning_content', None)!r}")
        print(f"message_content: {getattr(message, 'content', None)!r}")
    extracted_text = extract_message_text(response)
    print(f"content: {extracted_text!r}")
    if config["json_output"] and extracted_text:
        try:
            parsed_json = json.loads(extract_json_text(extracted_text))
        except Exception as exc:
            print("\nJSON parse failed.")
            print(f"Exception type: {type(exc).__name__}")
            print(f"Exception: {exc}")
            if not config["dump_response"]:
                print("\nTip: rerun with --dump-response to inspect the raw payload.")
            return 1

        print("\nParsed JSON:")
        print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
    if config["dump_response"] or not extracted_text:
        print("\nRaw response:")
        try:
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
        except Exception:
            print(repr(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
