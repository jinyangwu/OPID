#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Google Gemini/Vertex AI connectivity test via gigpo.api."
    )
    parser.add_argument(
        "--service-account-file",
        default=os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE"),
        help="Service account json path. Defaults to GOOGLE_SERVICE_ACCOUNT_FILE.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GOOGLE_GENAI_MODEL", "gemini-2.5-pro"),
        help="Gemini model name. Defaults to GOOGLE_GENAI_MODEL or gemini-2.5-pro.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: ok with mememem",
        help="User prompt for the test request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the test request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max output tokens for the test request.",
    )
    return parser.parse_args()


def mask_path(value: Optional[str]) -> str:
    if not value:
        return "<missing>"
    directory, filename = os.path.split(value)
    if not filename:
        return value
    if len(filename) <= 8:
        return os.path.join(directory, "*" * len(filename)) if directory else "*" * len(filename)
    masked = filename[:4] + "*" * (len(filename) - 8) + filename[-4:]
    return os.path.join(directory, masked) if directory else masked


def main() -> int:
    args = parse_args()

    if args.service_account_file:
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"] = args.service_account_file

    print("Google Gemini test configuration:")
    print(f"  model: {args.model}")
    print(f"  service_account_file: {mask_path(args.service_account_file)}")
    print(f"  service_account_exists: {bool(args.service_account_file and os.path.exists(args.service_account_file))}")
    print(f"  project(default in gigpo.api): {os.environ.get('GOOGLE_CLOUD_PROJECT', 'decision-agent-gemini')}")
    print(f"  location(default in gigpo.api): {os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')}")

    if not args.service_account_file:
        print("Missing GOOGLE_SERVICE_ACCOUNT_FILE or --service-account-file.", file=sys.stderr)
        return 2
    if not os.path.exists(args.service_account_file):
        print(f"Service account file not found: {args.service_account_file}", file=sys.stderr)
        return 2

    try:
        from gigpo.api import init_client, unified_infer
    except ImportError as exc:
        print("Failed to import gigpo.api or its dependencies.", file=sys.stderr)
        print(f"ImportError: {exc}", file=sys.stderr)
        return 2

    print("\nInitializing Google client...")
    try:
        client, model = init_client("google", args.model)
    except Exception as exc:
        print("Client initialization failed.", file=sys.stderr)
        print(f"Exception type: {type(exc).__name__}", file=sys.stderr)
        print(f"Exception: {exc}", file=sys.stderr)
        return 1

    print("Sending request...")
    try:
        response_text = unified_infer(
            prompt=args.prompt,
            client=client,
            model=model,
            provider="google",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=1,
        )
    except Exception as exc:
        print("Request failed.", file=sys.stderr)
        print(f"Exception type: {type(exc).__name__}", file=sys.stderr)
        print(f"Exception: {exc}", file=sys.stderr)
        return 1

    print("\nRequest succeeded.")
    print(f"content: {response_text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
