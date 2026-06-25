import argparse
import logging
import os
import sys
import time
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
        description="Evaluate a local OpenAI-compatible/vLLM model on WebShop."
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
        help="Served model name. Defaults to OPENAI_MODEL or local-webshop-model.",
    )
    parser.add_argument("--env-num", type=int, default=128, help="Number of parallel WebShop envs.")
    parser.add_argument("--test-times", type=int, default=1, help="Evaluation rounds.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode.")
    parser.add_argument("--history-length", type=int, default=2, help="Prompt history length.")
    parser.add_argument("--seed", type=int, default=1000, help="Environment seed.")
    parser.add_argument("--human-goals", type=int, default=0)
    parser.add_argument("--use-small", action="store_true", help="Use the smaller 1k-item WebShop data.")
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
    parser.add_argument("--fallback-action", default="click[back to search]")
    parser.add_argument("--extra-body-json", default=None)
    parser.add_argument("--log-dir", default="logs/webshop_local_vllm")
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
    parser.add_argument(
        "--startup-wait-seconds",
        type=float,
        default=1.0,
        help="How long to wait after spawning WebShop workers.",
    )
    return parser.parse_args()


def resolve_client_config(args: argparse.Namespace) -> Tuple[str, str, str]:
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    model_name = args.model_name or os.environ.get("OPENAI_MODEL") or "local-webshop-model"
    return base_url, api_key, model_name


def has_action_tag(text: str) -> bool:
    lowered = text.lower()
    return "<action>" in lowered and "</action>" in lowered


def build_env(args: argparse.Namespace):
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import (
        build_webshop_envs,
        webshop_projection,
    )

    base_dir = PROJECT_ROOT / "agent_system/environments/env_package/webshop/webshop/data"
    if args.use_small:
        file_path = base_dir / "items_shuffle_1000.json"
        attr_path = base_dir / "items_ins_v2_1000.json"
    else:
        file_path = base_dir / "items_shuffle.json"
        attr_path = base_dir / "items_ins_v2.json"

    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": args.human_goals,
        "file_path": str(file_path),
        "attr_path": str(attr_path),
    }
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
    envs = build_webshop_envs(
        seed=args.seed,
        env_num=args.env_num,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    config = AttrDict(
        {
            "env": AttrDict(
                {
                    "env_name": "webshop",
                    "history_length": args.history_length,
                    "use_skills_only_memory": False,
                }
            )
        }
    )
    time.sleep(max(args.startup_wait_seconds, args.env_num * 0.1))
    return WebshopEnvironmentManager(envs, webshop_projection, config)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    base_url, api_key, model_name = resolve_client_config(args)
    extra_body = resolve_extra_body(args.extra_body_json)
    dataset_name = "webshop"
    split_name = "eval_small" if args.use_small else "eval"
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

    logging.info("Local WebShop evaluation log: %s", log_path)
    logging.info("WebShop run name: %s", run_name)
    logging.info("WebShop run config written to: %s", config_path)
    logging.info("WebShop results will be written to: %s", result_path)
    if trajectory_path:
        logging.info("WebShop trajectories will be written to: %s", trajectory_path)
    logging.info(
        "env_num=%d test_times=%d max_steps=%d history_length=%d human_goals=%d use_small=%s request_workers=%d",
        args.env_num,
        args.test_times,
        args.max_steps,
        args.history_length,
        args.human_goals,
        args.use_small,
        request_workers,
    )

    env_manager = None
    overall_success_rates: List[float] = []
    overall_task_scores: List[float] = []
    action_tag_rates: List[float] = []
    projection_valid_rates: List[float] = []
    api_error_rates: List[float] = []
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
            final_task_scores = np.zeros(args.env_num, dtype=float)
            response_action_tag_flags: List[float] = []
            projection_valid_flags: List[float] = []
            api_error_flags: List[float] = []
            trajectories = [
                {
                    "test_idx": test_idx,
                    "env_idx": env_idx,
                    "task": env_manager.tasks[env_idx] if hasattr(env_manager, "tasks") else "",
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
                    float(final_task_scores.mean()),
                )

                current_observations = list(obs["text"])
                actions, model_responses, step_action_tag_flags, step_error_flags = collect_actions_concurrently(
                    agent=agent,
                    observations=obs["text"],
                    env_dones=env_dones,
                    done_action="<think>The episode is done.</think><action>click[back to search]</action>",
                    request_workers=request_workers,
                    log_responses=args.log_responses,
                    format_checker=has_action_tag,
                )
                response_action_tag_flags.extend(step_action_tag_flags)
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

                    projection_valid_flags.append(float(infos[env_idx].get("is_action_valid", 0.0)))
                    final_task_scores[env_idx] = float(
                        infos[env_idx].get("task_score", final_task_scores[env_idx])
                    )
                    if dones[env_idx]:
                        env_dones[env_idx] = True
                        success_flags[env_idx] = bool(infos[env_idx].get("won", False))

                if all(env_dones):
                    logging.info("All environments finished early.")
                    break

            round_success_rate = float(success_flags.mean())
            round_task_score = float(final_task_scores.mean())
            round_action_tag_rate = (
                float(np.mean(response_action_tag_flags)) if response_action_tag_flags else 0.0
            )
            round_projection_valid_rate = (
                float(np.mean(projection_valid_flags)) if projection_valid_flags else 0.0
            )
            round_api_error_rate = float(np.mean(api_error_flags)) if api_error_flags else 0.0

            overall_success_rates.append(round_success_rate)
            overall_task_scores.append(round_task_score)
            action_tag_rates.append(round_action_tag_rate)
            projection_valid_rates.append(round_projection_valid_rate)
            api_error_rates.append(round_api_error_rate)

            for env_idx in range(args.env_num):
                trajectories[env_idx]["success"] = bool(success_flags[env_idx])
                trajectories[env_idx]["completed"] = bool(env_dones[env_idx])
                trajectories[env_idx]["final_task_score"] = float(final_task_scores[env_idx])
                trajectories[env_idx]["num_steps"] = len(trajectories[env_idx]["steps"])

            logging.info("Test %d success rate: %.4f", test_idx, round_success_rate)
            logging.info("Test %d average task score: %.4f", test_idx, round_task_score)
            logging.info("Test %d action tag rate: %.4f", test_idx, round_action_tag_rate)
            logging.info(
                "Test %d projection valid rate: %.4f",
                test_idx,
                round_projection_valid_rate,
            )
            logging.info("Test %d API error fallback rate: %.4f", test_idx, round_api_error_rate)
            round_results.append(
                {
                    "test_idx": test_idx,
                    "success_rate": round_success_rate,
                    "average_task_score": round_task_score,
                    "action_tag": round_action_tag_rate,
                    "projection_valid": round_projection_valid_rate,
                    "api_error_fallback": round_api_error_rate,
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
            float(np.mean(overall_task_scores)) if overall_task_scores else 0.0,
            float(np.std(overall_task_scores)) if overall_task_scores else 0.0,
        )
        logging.info(
            "Action tag avg +/- std: %.4f +/- %.4f",
            float(np.mean(action_tag_rates)) if action_tag_rates else 0.0,
            float(np.std(action_tag_rates)) if action_tag_rates else 0.0,
        )
        logging.info(
            "Projection valid avg +/- std: %.4f +/- %.4f",
            float(np.mean(projection_valid_rates)) if projection_valid_rates else 0.0,
            float(np.std(projection_valid_rates)) if projection_valid_rates else 0.0,
        )
        logging.info(
            "API error fallback avg +/- std: %.4f +/- %.4f",
            float(np.mean(api_error_rates)) if api_error_rates else 0.0,
            float(np.std(api_error_rates)) if api_error_rates else 0.0,
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
                "average_task_score": metric_stats(overall_task_scores),
                "action_tag": metric_stats(action_tag_rates),
                "projection_valid": metric_stats(projection_valid_rates),
                "api_error_fallback": metric_stats(api_error_rates),
            },
            "rounds": round_results,
        }
        write_json(result_path, result_payload)
        logging.info("WebShop results written to: %s", result_path)
    finally:
        if env_manager is not None:
            env_manager.close()


if __name__ == "__main__":
    main()
