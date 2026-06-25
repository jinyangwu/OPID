#!/usr/bin/env python3
"""Rebuild an OPID episode-analysis prompt from a saved analysis jsonl row."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import chat_completion_with_retry, create_openai_client


def import_opid_episode_analyzer():
    try:
        from opid.analysis import OPIDEpisodeAnalyzer

        return OPIDEpisodeAnalyzer
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise

        # This demo only needs the prompt-construction methods in opid.analysis.
        # Some lightweight Python envs do not have torch installed, so provide
        # tiny import-time stubs for annotations and unused module imports.
        torch_stub = types.ModuleType("torch")

        class Tensor:
            pass

        torch_stub.Tensor = Tensor
        sys.modules["torch"] = torch_stub
        if "verl" not in sys.modules:
            verl_stub = types.ModuleType("verl")

            class DataProto:
                pass

            verl_stub.DataProto = DataProto
            sys.modules["verl"] = verl_stub
        sys.modules.pop("opid.analysis", None)
        core_gigpo_stub = types.ModuleType("gigpo.core_gigpo")
        sys.modules["gigpo.core_gigpo"] = core_gigpo_stub
        try:
            import gigpo

            gigpo.core_gigpo = core_gigpo_stub
        except Exception:
            pass

        from opid.analysis import OPIDEpisodeAnalyzer

        return OPIDEpisodeAnalyzer


DEFAULT_JSONL = os.environ.get(
    "OPID_ANALYSIS_JSONL",
    str(REPO_ROOT / "outputs" / "opid_analysis" / "step_00000001.jsonl"),
)


def load_env_file(env_file: Optional[str]) -> None:
    if not env_file:
        return
    path = Path(env_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return

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
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_jsonl_record(path: Path, index: int, traj_uid: Optional[str]) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    selected_idx = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if traj_uid is not None:
                if str(record.get("traj_uid")) == traj_uid:
                    return record
                continue
            if selected_idx == index:
                return record
            selected_idx += 1

    if traj_uid is not None:
        raise ValueError(f"traj_uid not found in {path}: {traj_uid}")
    raise IndexError(f"record index {index} not found in {path}")


def get_saved_user_prompt(record: Dict[str, Any]) -> str:
    prompt = record.get("llm_prompt")
    messages = prompt.get("messages") if isinstance(prompt, dict) else None
    if not isinstance(messages, list):
        raise ValueError("record does not contain llm_prompt.messages")

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("record llm_prompt.messages does not contain a user message")


def extract_task_description(saved_prompt: str) -> str:
    match = re.search(
        r"^- Task description:\s*(.*?)\n- episode_success:",
        saved_prompt,
        flags=re.MULTILINE | re.DOTALL,
    )
    return " ".join(match.group(1).split()) if match else ""


def split_response_and_reward(text: str) -> Tuple[str, Optional[float]]:
    response = text.strip()
    marker = "\nReturn: "
    if marker not in response:
        return response, None

    before, after = response.rsplit(marker, 1)
    after = after.strip()
    if "\n" in after:
        return response, None
    try:
        return before.rstrip(), float(after)
    except ValueError:
        return response, None


def parse_steps_from_saved_prompt(saved_prompt: str) -> List[Dict[str, Any]]:
    marker = "- Interaction trajectory:"
    marker_idx = saved_prompt.find(marker)
    if marker_idx < 0:
        raise ValueError("saved prompt does not contain an Interaction trajectory section")

    trajectory = saved_prompt[marker_idx + len(marker) :].lstrip()
    step_matches = list(re.finditer(r"^Step\s+(\d+)\s*$", trajectory, flags=re.MULTILINE))
    if not step_matches:
        raise ValueError("no Step blocks found in saved prompt")

    steps: List[Dict[str, Any]] = []
    for match_idx, match in enumerate(step_matches):
        step_index = int(match.group(1))
        block_start = match.end()
        block_end = step_matches[match_idx + 1].start() if match_idx + 1 < len(step_matches) else len(trajectory)
        block = trajectory[block_start:block_end].lstrip("\n")

        observation_prefix = "Observation: "
        response_marker = "\nResponse: "
        if not block.startswith(observation_prefix) or response_marker not in block:
            raise ValueError(f"could not parse Step {step_index} block")

        observation_text, response_text = block[len(observation_prefix) :].split(response_marker, 1)
        response_text, step_reward = split_response_and_reward(response_text)
        step: Dict[str, Any] = {
            "step_index": step_index,
            "observation": observation_text.rstrip("\n"),
            "response": response_text,
        }
        if step_reward is not None:
            step["step_reward"] = step_reward
        steps.append(step)

    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one row from a saved OPID analysis jsonl file, reconstruct the episode steps, "
            "call the current OPIDEpisodeAnalyzer._build_episode_analysis_prompt, and optionally call the LLM."
        )
    )
    parser.add_argument("--jsonl", default=DEFAULT_JSONL, help="Saved OPID analysis jsonl file.")
    parser.add_argument("--index", type=int, default=0, help="0-based row index to load when --traj-uid is unset.")
    parser.add_argument("--traj-uid", default=None, help="Specific trajectory uid to load.")
    parser.add_argument("--env-file", default=".env", help="Env file for OPENAI_* variables. Use '' to skip.")
    parser.add_argument("--backend", default="openai", help="OPID analysis backend. Usually openai.")
    parser.add_argument("--analysis-mode", default=None, help="Override analysis mode from the saved row.")
    parser.add_argument("--max-step-skills-per-traj", type=int, default=5)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL for the demo request.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="Override OPENAI_API_KEY.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--retry-delay", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--call-llm", action="store_true", help="Actually send the rebuilt prompt to the LLM.")
    parser.add_argument("--parse-response", action="store_true", help="Parse the LLM response with _parse_analysis_response.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the rebuilt prompt and LLM response as JSON.",
    )
    parser.add_argument("--no-print-prompt", action="store_true", help="Do not print the rebuilt user prompt.")
    parser.add_argument("--print-messages-json", action="store_true", help="Print the whole messages payload as JSON.")
    return parser.parse_args()


def save_prompt_response_json(
    *,
    output_path: str,
    args: argparse.Namespace,
    record: Dict[str, Any],
    prompt: Dict[str, Any],
    response: Optional[str],
    parsed_response: Optional[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    candidate_step_indices: List[int],
    task_description: str,
    analysis_mode: str,
) -> None:
    path = Path(output_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": {
            "jsonl": args.jsonl,
            "index": args.index,
            "traj_uid": record.get("traj_uid"),
            "global_step": record.get("global_step"),
        },
        "config": {
            "backend": args.backend,
            "analysis_mode": analysis_mode,
            "max_step_skills_per_traj": args.max_step_skills_per_traj,
            "max_completion_tokens": args.max_completion_tokens,
            "model": args.model or os.environ.get("OPENAI_MODEL"),
            "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL"),
            "temperature": args.temperature,
        },
        "episode": {
            "episode_success": record.get("episode_success"),
            "task_description": task_description,
            "candidate_step_indices": candidate_step_indices,
            "num_steps": len(steps),
        },
        "prompt": {
            "messages": prompt.get("messages", []),
            "user": prompt.get("messages", [{}])[0].get("content", ""),
        },
        "response": {
            "raw": response,
            "parsed": parsed_response,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved prompt/response JSON to: {path}")


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)

    record = load_jsonl_record(Path(args.jsonl), index=args.index, traj_uid=args.traj_uid)
    saved_prompt = get_saved_user_prompt(record)
    task_description = extract_task_description(saved_prompt)
    steps = parse_steps_from_saved_prompt(saved_prompt)
    candidate_step_indices = record.get("candidate_step_indices") or [step["step_index"] for step in steps]
    analysis_mode = args.analysis_mode or record.get("analysis_mode") or "teacher_bootstrap"

    analyzer_cls = import_opid_episode_analyzer()
    analyzer = analyzer_cls(
        backend=args.backend,
        max_completion_tokens=args.max_completion_tokens,
        max_step_skills_per_traj=args.max_step_skills_per_traj,
    )
    prompt = analyzer._build_episode_analysis_prompt(
        steps=steps,
        candidate_step_indices=candidate_step_indices,
        analysis_mode=analysis_mode,
        episode_success=record.get("episode_success"),
        task_description=task_description,
    )

    print("Loaded OPID analysis row:")
    print(f"  jsonl: {args.jsonl}")
    print(f"  traj_uid: {record.get('traj_uid')}")
    print(f"  analysis_mode: {analysis_mode}")
    print(f"  episode_success: {record.get('episode_success')}")
    print(f"  steps: {len(steps)}")
    print(f"  candidate_step_indices: {candidate_step_indices}")
    print(f"  task_description: {task_description or '<empty>'}")

    if args.print_messages_json:
        print("\nRebuilt messages JSON:")
        print(json.dumps(prompt["messages"], ensure_ascii=False, indent=2))
    elif not args.no_print_prompt:
        print("\nRebuilt user prompt:")
        print(prompt["messages"][0]["content"])

    if not args.call_llm:
        if args.output_json:
            save_prompt_response_json(
                output_path=args.output_json,
                args=args,
                record=record,
                prompt=prompt,
                response=None,
                parsed_response=None,
                steps=steps,
                candidate_step_indices=candidate_step_indices,
                task_description=task_description,
                analysis_mode=analysis_mode,
            )
        print("\nDry run only. Rerun with --call-llm to send this prompt.")
        return 0

    model = args.model or os.environ.get("OPENAI_MODEL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    retries = args.retries if args.retries is not None else int(os.environ.get("OPENAI_API_RETRIES", "5"))
    retry_delay = (
        args.retry_delay
        if args.retry_delay is not None
        else float(os.environ.get("OPENAI_API_RETRY_DELAY", "1.0"))
    )
    if not model:
        raise ValueError("Missing model. Set OPENAI_MODEL or pass --model.")
    if not api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    print("\nSending rebuilt prompt to LLM...")
    print(f"  model: {model}")
    print(f"  base_url: {base_url or '<default>'}")
    client = create_openai_client(api_key=api_key, base_url=base_url, timeout=args.timeout)
    content = chat_completion_with_retry(
        client=client,
        model=model,
        prompt=prompt,
        retries=retries,
        retry_delay=retry_delay,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    print("\nLLM raw reply:")
    print(content)

    parsed = None
    if args.parse_response:
        parsed = analyzer._parse_analysis_response(content)
        print("\nParsed reply:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

    if args.output_json:
        save_prompt_response_json(
            output_path=args.output_json,
            args=args,
            record=record,
            prompt=prompt,
            response=content,
            parsed_response=parsed,
            steps=steps,
            candidate_step_indices=candidate_step_indices,
            task_description=task_description,
            analysis_mode=analysis_mode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
