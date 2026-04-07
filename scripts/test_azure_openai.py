#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Optional

try:
    from openai import AzureOpenAI
except ImportError as exc:  # pragma: no cover
    print("Failed to import openai.AzureOpenAI. Please install the openai package first.", file=sys.stderr)
    raise


def mask_secret(value: Optional[str], keep: int = 4) -> str:
    if not value:
        return "<missing>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep * 2) + value[-keep:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Azure OpenAI connectivity/deployment test."
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        help="Azure OpenAI endpoint. Defaults to AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AZURE_OPENAI_API_KEY"),
        help="Azure OpenAI API key. Defaults to AZURE_OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--api-version",
        default=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        help="Azure OpenAI API version. Defaults to AZURE_OPENAI_API_VERSION or 2024-12-01-preview.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_MODEL", "gpt5"),
        help="Azure deployment/model name. Defaults to AZURE_OPENAI_MODEL or gpt5.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: ok",
        help="User prompt for the test request.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=32,
        help="Max completion tokens for the test request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the test request.",
    )
    return parser.parse_args()


def extract_content(response: Any) -> str:
    try:
        return response.choices[0].message.content
    except Exception:
        return "<unable to extract message content>"


def main() -> int:
    args = parse_args()

    print("Azure OpenAI test configuration:")
    print(f"  endpoint: {args.endpoint or '<missing>'}")
    print(f"  api_version: {args.api_version}")
    print(f"  model/deployment: {args.model}")
    print(f"  api_key: {mask_secret(args.api_key)}")

    missing = [
        name
        for name, value in (
            ("endpoint", args.endpoint),
            ("api_key", args.api_key),
            ("model", args.model),
        )
        if not value
    ]
    if missing:
        print(f"Missing required configuration: {', '.join(missing)}", file=sys.stderr)
        return 2

    client = AzureOpenAI(
        api_key=args.api_key,
        azure_endpoint=args.endpoint,
        api_version=args.api_version,
    )

    print("\nSending request...")
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )
    except Exception as exc:
        print("\nRequest failed.")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception: {exc}")

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
    print(f"content: {extract_content(response)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
