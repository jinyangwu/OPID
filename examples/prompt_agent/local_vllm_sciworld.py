import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.prompt_agent.local_vllm_alfworld import (  # noqa: E402
    AttrDict,
    LocalOpenAIAgent,
    build_run_name,
    collect_actions_concurrently,
    has_action_format,
    json_safe,
    load_env_file,
    metric_stats,
    resolve_extra_body,
    resolve_result_path,
    resolve_request_workers,
    resolve_run_config_path,
    resolve_trajectory_path,
    setup_logging,
    write_json,
    write_run_config,
    write_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local OpenAI-compatible/vLLM model on ScienceWorld."
    )
    parser.add_argument("--env-file", default=".env", help="Optional env file to load first.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL or http://127.0.0.1:8000/v1.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the endpoint. Defaults to OPENAI_API_KEY or EMPTY for local vLLM.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Served model name. Defaults to OPENAI_MODEL or local-sciworld-model.",
    )
    parser.add_argument("--env-num", type=int, default=128, help="Number of parallel ScienceWorld envs.")
    parser.add_argument("--test-times", type=int, default=1, help="Evaluation rounds.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode.")
    parser.add_argument("--history-length", type=int, default=5, help="Prompt history length.")
    parser.add_argument("--seed", type=int, default=1000, help="Environment seed.")
    parser.add_argument("--generalization-level", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--simplifications-preset", default="easy")
    parser.add_argument(
        "--env-step-limit",
        type=int,
        default=None,
        help="ScienceWorld internal step limit. Defaults to --max-steps.",
    )
    parser.add_argument(
        "--jar-path",
        default=None,
        help="ScienceWorld jar path. Use null/none/empty for the builtin jar.",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--request-workers",
        type=int,
        default=None,
        help="Concurrent API requests per environment step. Defaults to env-num.",
    )
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--fallback-action", default="look around")
    parser.add_argument("--extra-body-json", default=None)
    parser.add_argument("--log-dir", default="logs/sciworld_local_vllm")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory for per-run config JSON files. Defaults to <log-dir>/configs.",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Directory for per-run result JSON files. Defaults to <log-dir>/results.",
    )
    parser.add_argument(
        "--trajectory-dir",
        default=None,
        help="Directory for per-episode trajectory JSONL files. Defaults to <log-dir>/trajectories.",
    )
    parser.add_argument("--no-save-trajectories", action="store_true")
    parser.add_argument("--log-responses", action="store_true")
    return parser.parse_args()


def resolve_client_config(args: argparse.Namespace) -> Tuple[str, str, str]:
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    model_name = args.model_name or os.environ.get("OPENAI_MODEL") or "local-sciworld-model"
    return base_url, api_key, model_name


def none_if_null(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped or stripped.lower() in {"none", "null"}:
        return None
    return stripped


def build_env(args: argparse.Namespace):
    from agent_system.environments.env_manager import SciWorldEnvironmentManager
    from agent_system.environments.env_package.sciworld import (
        build_sciworld_envs,
        sciworld_projection,
    )

    variation_path = (
        PROJECT_ROOT
        / "agent_system/environments/env_package/sciworld/variations_idx"
        / f"L{args.generalization_level}_idx.json"
    )
    with variation_path.open("r", encoding="utf-8") as file:
        variations_idx = json.load(file)

    envs = build_sciworld_envs(
        seed=args.seed,
        env_num=args.env_num,
        group_n=1,
        split="test",
        simplifications_preset=args.simplifications_preset,
        env_step_limit=args.env_step_limit or args.max_steps,
        jar_path=none_if_null(args.jar_path),
        variations_idx=variations_idx["test"],
    )
    config = AttrDict(
        {
            "env": AttrDict(
                {
                    "env_name": "sciworld",
                    "history_length": args.history_length,
                    "use_skills_only_memory": False,
                    "use_retrieval_memory": False,
                }
            )
        }
    )
    return SciWorldEnvironmentManager(envs, sciworld_projection, config)


def task_key(info: Dict[str, Any], fallback: str = "unknown") -> str:
    task_num = info.get("task_num")
    if task_num is not None:
        return f"task_{task_num}"
    task_description = str(info.get("task_description", "")).strip()
    if task_description:
        return task_description[:80]
    return fallback


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    base_url, api_key, model_name = resolve_client_config(args)
    extra_body = resolve_extra_body(args.extra_body_json)
    dataset_name = "sciworld"
    split_name = f"test_L{args.generalization_level}"
    run_name = build_run_name(dataset_name, model_name, split_name)
    log_path = setup_logging(args.log_dir, run_name)
    request_workers = resolve_request_workers(args.request_workers, args.env_num)
    config_path = resolve_run_config_path(args.log_dir, args.config_dir, run_name)
    result_path = resolve_result_path(args.log_dir, args.result_dir, run_name)
    trajectory_path = None
    if not args.no_save_trajectories:
        trajectory_path = resolve_trajectory_path(args.log_dir, args.trajectory_dir, run_name)
    write_run_config(
        config_path,
        args=args,
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        request_workers=request_workers,
        log_path=log_path,
        result_path=result_path,
        trajectory_path=trajectory_path,
        extra_body=extra_body,
        script_path=__file__,
        run_name=run_name,
        dataset=dataset_name,
        split=split_name,
    )

    logging.info("Local ScienceWorld evaluation log: %s", log_path)
    logging.info("ScienceWorld run name: %s", run_name)
    logging.info("ScienceWorld run config written to: %s", config_path)
    logging.info("ScienceWorld results will be written to: %s", result_path)
    if trajectory_path:
        logging.info("ScienceWorld trajectories will be written to: %s", trajectory_path)
    logging.info(
        "env_num=%d test_times=%d max_steps=%d env_step_limit=%d generalization_level=%d simplifications=%s request_workers=%d",
        args.env_num,
        args.test_times,
        args.max_steps,
        args.env_step_limit or args.max_steps,
        args.generalization_level,
        args.simplifications_preset,
        request_workers,
    )

    env_manager = None
    overall_success_rates: List[float] = []
    overall_scores: List[float] = []
    action_format_rates: List[float] = []
    api_error_rates: List[float] = []
    task_success_history = defaultdict(list)
    task_score_history = defaultdict(list)
    round_results: List[Dict[str, Any]] = []

    try:
        env_manager = build_env(args)
        agent = LocalOpenAIAgent(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
            fallback_action=args.fallback_action,
            extra_body=extra_body,
        )

        for test_idx in range(args.test_times):
            logging.info("========== Start test %d ==========", test_idx)
            start_time = time.time()
            obs, infos = env_manager.reset({})
            env_dones = [False] * args.env_num
            success_flags = np.zeros(args.env_num, dtype=bool)
            final_scores = np.zeros(args.env_num, dtype=float)
            task_keys = [task_key(info, str(idx)) for idx, info in enumerate(infos)]
            response_format_flags: List[float] = []
            api_error_flags: List[float] = []
            trajectories = [
                {
                    "test_idx": test_idx,
                    "env_idx": env_idx,
                    "task": env_manager.tasks[env_idx] if hasattr(env_manager, "tasks") else "",
                    "task_key": task_keys[env_idx],
                    "initial_info": json_safe(infos[env_idx]),
                    "steps": [],
                }
                for env_idx in range(args.env_num)
            ]

            for step_idx in range(args.max_steps):
                logging.info(
                    "Step %d | Dones %d/%d | Current SR %.4f | Current score %.4f",
                    step_idx,
                    int(np.sum(env_dones)),
                    args.env_num,
                    float(success_flags.mean()),
                    float(final_scores.mean()),
                )

                current_observations = list(obs["text"])
                actions, model_responses, step_format_flags, step_error_flags = collect_actions_concurrently(
                    agent=agent,
                    observations=obs["text"],
                    env_dones=env_dones,
                    done_action="<think>The episode is done.</think><action>look around</action>",
                    request_workers=request_workers,
                    log_responses=args.log_responses,
                    format_checker=has_action_format,
                )
                response_format_flags.extend(step_format_flags)
                api_error_flags.extend(step_error_flags)

                actions_for_env = list(actions)
                next_obs, rewards, dones, infos = env_manager.step(actions_for_env)

                for env_idx in range(args.env_num):
                    if env_dones[env_idx]:
                        continue
                    trajectories[env_idx]["steps"].append(
                        {
                            "step_idx": step_idx,
                            "observation": current_observations[env_idx],
                            "model_response": model_responses[env_idx],
                            "raw_action_text": actions[env_idx],
                            "executed_action": actions_for_env[env_idx],
                            "reward": json_safe(rewards[env_idx]),
                            "done": bool(dones[env_idx]),
                            "info": json_safe(infos[env_idx]),
                            "next_observation": next_obs["text"][env_idx],
                        }
                    )

                obs = next_obs

                for env_idx in range(args.env_num):
                    if env_dones[env_idx]:
                        continue

                    final_scores[env_idx] = float(
                        infos[env_idx].get("task_score", infos[env_idx].get("score", final_scores[env_idx]))
                    )
                    if task_keys[env_idx] == "unknown":
                        task_keys[env_idx] = task_key(infos[env_idx], str(env_idx))
                        trajectories[env_idx]["task_key"] = task_keys[env_idx]
                    if dones[env_idx]:
                        env_dones[env_idx] = True
                        success_flags[env_idx] = bool(infos[env_idx].get("won", False))

                if all(env_dones):
                    logging.info("All environments finished early.")
                    break

            task_success = defaultdict(list)
            task_scores = defaultdict(list)
            for env_idx, key in enumerate(task_keys):
                task_success[key].append(float(success_flags[env_idx]))
                task_scores[key].append(float(final_scores[env_idx]))
                trajectories[env_idx]["task_key"] = key
                trajectories[env_idx]["success"] = bool(success_flags[env_idx])
                trajectories[env_idx]["completed"] = bool(env_dones[env_idx])
                trajectories[env_idx]["final_score"] = float(final_scores[env_idx])
                trajectories[env_idx]["num_steps"] = len(trajectories[env_idx]["steps"])

            round_success_rate = float(success_flags.mean())
            round_score = float(final_scores.mean())
            round_format_rate = float(np.mean(response_format_flags)) if response_format_flags else 0.0
            round_api_error_rate = float(np.mean(api_error_flags)) if api_error_flags else 0.0
            overall_success_rates.append(round_success_rate)
            overall_scores.append(round_score)
            action_format_rates.append(round_format_rate)
            api_error_rates.append(round_api_error_rate)

            logging.info("Test %d success rate: %.4f", test_idx, round_success_rate)
            logging.info("Test %d average task score: %.4f", test_idx, round_score)
            logging.info("Test %d action format valid rate: %.4f", test_idx, round_format_rate)
            logging.info("Test %d API error fallback rate: %.4f", test_idx, round_api_error_rate)
            round_tasks = []
            for key in sorted(task_success):
                success_mean = float(np.mean(task_success[key]))
                score_mean = float(np.mean(task_scores[key]))
                task_success_history[key].append(success_mean)
                task_score_history[key].append(score_mean)
                round_tasks.append(
                    {
                        "task": key,
                        "success_rate": success_mean,
                        "score": score_mean,
                        "total": len(task_success[key]),
                    }
                )
                logging.info(
                    "    %-35s: success %.4f | score %.4f | n=%d",
                    key,
                    success_mean,
                    score_mean,
                    len(task_success[key]),
                )
            round_results.append(
                {
                    "test_idx": test_idx,
                    "success_rate": round_success_rate,
                    "average_task_score": round_score,
                    "action_format_valid": round_format_rate,
                    "api_error_fallback": round_api_error_rate,
                    "tasks": round_tasks,
                    "time_elapsed_sec": time.time() - start_time,
                }
            )
            logging.info("Test %d time elapsed: %.2fs", test_idx, time.time() - start_time)
            if trajectory_path:
                write_trajectories(trajectory_path, trajectories)

        logging.info("=============== Final Summary ===============")
        logging.info(
            "Success rate avg +/- std: %.4f +/- %.4f",
            float(np.mean(overall_success_rates)) if overall_success_rates else 0.0,
            float(np.std(overall_success_rates)) if overall_success_rates else 0.0,
        )
        logging.info(
            "Task score avg +/- std: %.4f +/- %.4f",
            float(np.mean(overall_scores)) if overall_scores else 0.0,
            float(np.std(overall_scores)) if overall_scores else 0.0,
        )
        logging.info(
            "Action format valid avg +/- std: %.4f +/- %.4f",
            float(np.mean(action_format_rates)) if action_format_rates else 0.0,
            float(np.std(action_format_rates)) if action_format_rates else 0.0,
        )
        logging.info(
            "API error fallback avg +/- std: %.4f +/- %.4f",
            float(np.mean(api_error_rates)) if api_error_rates else 0.0,
            float(np.std(api_error_rates)) if api_error_rates else 0.0,
        )
        task_summary = []
        for key in sorted(task_success_history):
            task_summary.append(
                {
                    "task": key,
                    "success": metric_stats(task_success_history[key]),
                    "score": metric_stats(task_score_history[key]),
                }
            )
        result_payload = {
            "run_name": run_name,
            "dataset": dataset_name,
            "model_name": model_name,
            "split": split_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "paths": {
                "log": log_path,
                "config": config_path,
                "results": result_path,
                "trajectories": trajectory_path,
            },
            "total_tests": args.test_times,
            "envs_per_test": args.env_num,
            "total_envs": args.env_num * args.test_times,
            "summary": {
                "success_rate": metric_stats(overall_success_rates),
                "average_task_score": metric_stats(overall_scores),
                "action_format_valid": metric_stats(action_format_rates),
                "api_error_fallback": metric_stats(api_error_rates),
            },
            "tasks": task_summary,
            "rounds": round_results,
        }
        write_json(result_path, result_payload)
        logging.info("ScienceWorld results written to: %s", result_path)
    finally:
        if env_manager is not None:
            env_manager.close()


if __name__ == "__main__":
    main()
