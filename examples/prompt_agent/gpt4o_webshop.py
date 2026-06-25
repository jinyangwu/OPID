import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import AzureOpenAI, OpenAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SYSTEM_PROMPT = (
    "You are a careful WebShop agent. "
    "Always answer in English. "
    "Choose exactly one legal action for the current page. "
    "Reply with the format <think>...</think><action>...</action>."
)

ACTION_TEMPLATE = (
    "<think>I will choose one admissible action that best advances the "
    "shopping goal.</think><action>{action}</action>"
)

DEFAULT_AZURE_API_VERSION = "2025-01-01-preview"
NOOP_ACTION = "click[back to search]"


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a prompt-based WebShop agent with OpenAI or Azure OpenAI."
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "azure"],
        default="auto",
        help="LLM provider. 'auto' prefers Azure when Azure env vars are set.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Model name for OpenAI, or deployment name for Azure OpenAI. "
            "Defaults to OPENAI_MODEL / AZURE_OPENAI_DEPLOYMENT / gpt-4o."
        ),
    )
    parser.add_argument("--env-num", type=int, default=1, help="Number of parallel WebShop environments.")
    parser.add_argument("--test-times", type=int, default=1, help="How many evaluation rounds to run.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=1, help="Environment seed.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. If omitted, the client default is used.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Maximum completion tokens per model call.",
    )
    parser.add_argument(
        "--human-goals",
        type=int,
        default=1,
        help="Whether to use human-authored goals in WebShop.",
    )
    parser.add_argument(
        "--use-small",
        action="store_true",
        help="Use the smaller 1k-item WebShop dataset.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/webshop",
        help="Directory for evaluation logs.",
    )
    parser.add_argument(
        "--startup-wait-seconds",
        type=float,
        default=1.0,
        help="How long to wait after spawning WebShop workers.",
    )
    return parser.parse_args()


def resolve_provider(provider: str) -> str:
    if provider != "auto":
        return provider

    has_azure = bool(os.environ.get("AZURE_OPENAI_API_KEY")) and bool(
        os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    if has_azure:
        return "azure"

    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    raise EnvironmentError(
        "No provider credentials found. Set either OPENAI_API_KEY, or "
        "AZURE_OPENAI_API_KEY together with AZURE_OPENAI_ENDPOINT."
    )


def resolve_model_name(provider: str, model_name: Optional[str]) -> str:
    if model_name:
        return model_name
    if provider == "azure":
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    return os.environ.get("OPENAI_MODEL", "gpt-4o")


def build_env(args: argparse.Namespace):
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import (
        build_webshop_envs,
        webshop_projection,
    )

    base_dir = os.path.join(
        os.path.dirname(__file__),
        "../../agent_system/environments/env_package/webshop/webshop/data",
    )
    if args.use_small:
        file_path = os.path.join(base_dir, "items_shuffle_1000.json")
        attr_path = os.path.join(base_dir, "items_ins_v2_1000.json")
    else:
        file_path = os.path.join(base_dir, "items_shuffle.json")
        attr_path = os.path.join(base_dir, "items_ins_v2.json")

    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": args.human_goals,
        "file_path": file_path,
        "attr_path": attr_path,
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
                    "history_length": 5,
                    "use_skills_only_memory": False,
                }
            )
        }
    )
    time.sleep(max(args.startup_wait_seconds, args.env_num * 0.1))
    return WebshopEnvironmentManager(envs, webshop_projection, config)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def build_search_query(task_description: str) -> str:
    query = normalize_text(task_description).strip(".")
    query = re.sub(r"^(please\s+)?(help me\s+)?(find|buy|get)\s+", "", query)
    query = re.sub(r"^(i need|i want|i would like)\s+", "", query)
    return query or "shopping item"


def extract_action(raw_response: str) -> Optional[str]:
    if not raw_response:
        return None

    action_match = re.search(
        r"<action>\s*(.*?)\s*</action>",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if action_match:
        action = action_match.group(1).strip()
        return action or None

    fallback_match = re.search(
        r"(search\[[^\]]+\]|click\[[^\]]+\])",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fallback_match:
        return fallback_match.group(1).strip()

    return None


def choose_fallback_action(available_actions: Dict, task_description: str) -> str:
    if available_actions.get("has_search_bar"):
        return f"search[{build_search_query(task_description)}]"

    clickables = available_actions.get("clickables", [])
    navigation_buttons = {
        "search",
        "< prev",
        "next >",
        "back to search",
        "description",
        "features",
        "reviews",
    }
    preferred_clickables = [item for item in clickables if item not in navigation_buttons]
    if preferred_clickables:
        return f"click[{preferred_clickables[0]}]"
    if clickables:
        return f"click[{clickables[0]}]"
    return NOOP_ACTION


def sanitize_action(
    raw_action: Optional[str],
    available_actions: Dict,
    task_description: str,
) -> Tuple[str, bool]:
    has_search_bar = bool(available_actions.get("has_search_bar"))
    clickables = available_actions.get("clickables", [])
    clickable_map = {normalize_text(item): item for item in clickables}

    if raw_action:
        candidate = raw_action.strip()

        search_match = re.fullmatch(r"search\[(.*)\]", candidate, flags=re.IGNORECASE | re.DOTALL)
        if search_match and has_search_bar:
            query = normalize_text(search_match.group(1))
            if query:
                return f"search[{query}]", True

        click_match = re.fullmatch(r"click\[(.*)\]", candidate, flags=re.IGNORECASE | re.DOTALL)
        if click_match:
            target = normalize_text(click_match.group(1)).strip("'\"")
            if target in clickable_map:
                return f"click[{clickable_map[target]}]", True

            for normalized_target, original_target in clickable_map.items():
                if target == normalized_target.strip("'\""):
                    return f"click[{original_target}]", True

    return choose_fallback_action(available_actions, task_description), False


def format_action_for_env(action: str) -> str:
    return ACTION_TEMPLATE.format(action=action)


class Agent:
    def __init__(
        self,
        provider: str,
        model_name: str,
        max_completion_tokens: int = 512,
        temperature: Optional[float] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature

        if provider == "azure":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = os.environ.get(
                "AZURE_OPENAI_API_VERSION",
                DEFAULT_AZURE_API_VERSION,
            )
            if not api_key or not endpoint:
                raise EnvironmentError(
                    "Azure OpenAI requires AZURE_OPENAI_API_KEY and "
                    "AZURE_OPENAI_ENDPOINT."
                )
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OpenAI requires OPENAI_API_KEY.")
            client_kwargs = {"api_key": api_key}
            base_url = os.environ.get("OPENAI_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)

    def _chat_completion(self, prompt: str) -> str:
        request_kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "".join(text_parts).strip()
        return ""

    def get_action(
        self,
        prompt: str,
        available_actions: Dict,
        task_description: str,
    ) -> Tuple[str, str, bool]:
        try:
            raw_response = self._chat_completion(prompt)
            raw_action = extract_action(raw_response)
            action, used_model_action = sanitize_action(
                raw_action,
                available_actions,
                task_description,
            )
            return action, raw_response, used_model_action
        except Exception as exc:
            fallback_action = choose_fallback_action(available_actions, task_description)
            return fallback_action, f"[llm_error] {exc}", False


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def main() -> None:
    args = parse_args()
    provider = resolve_provider(args.provider)
    model_name = resolve_model_name(provider, args.model_name)
    log_path = setup_logging(args.log_dir)

    logging.info("Provider: %s | Model/Deployment: %s", provider, model_name)
    logging.info("Logs will be written to %s", log_path)

    env_manager = None

    overall_success_rates: List[float] = []
    overall_task_scores: List[float] = []
    overall_action_valid_rates: List[float] = []
    overall_model_action_rates: List[float] = []

    try:
        env_manager = build_env(args)
        agent = Agent(
            provider=provider,
            model_name=model_name,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )

        for test_idx in range(args.test_times):
            logging.info("========== Start test %d ==========", test_idx)
            start_time = time.time()

            obs, infos = env_manager.reset({})
            env_dones = [False] * args.env_num
            success_flags = np.zeros(args.env_num, dtype=bool)
            task_scores = np.zeros(args.env_num, dtype=float)
            action_valid_flags: List[float] = []
            used_model_action_flags: List[float] = []

            for step_idx in range(args.max_steps):
                logging.info(
                    "Step %d | Dones %d/%d | Current SR %.4f",
                    step_idx,
                    int(np.sum(env_dones)),
                    args.env_num,
                    float(success_flags.mean()),
                )

                responses_for_env: List[str] = []
                for env_idx in range(args.env_num):
                    if env_dones[env_idx]:
                        responses_for_env.append(format_action_for_env(NOOP_ACTION))
                        continue

                    action, raw_response, used_model_action = agent.get_action(
                        prompt=obs["text"][env_idx],
                        available_actions=infos[env_idx]["available_actions"],
                        task_description=env_manager.tasks[env_idx],
                    )
                    responses_for_env.append(format_action_for_env(action))
                    used_model_action_flags.append(float(used_model_action))

                    if not used_model_action:
                        logging.info(
                            "Env %d fallback action used. Raw model output: %s",
                            env_idx,
                            raw_response.replace("\n", "\\n"),
                        )

                obs, rewards, dones, infos = env_manager.step(responses_for_env)

                for env_idx in range(args.env_num):
                    if env_dones[env_idx]:
                        continue

                    action_valid_flags.append(float(infos[env_idx]["is_action_valid"]))

                    if dones[env_idx]:
                        env_dones[env_idx] = True
                        success_flags[env_idx] = bool(infos[env_idx].get("won", False))
                        task_scores[env_idx] = float(infos[env_idx].get("task_score", 0.0))

                if all(env_dones):
                    logging.info("All environments finished early.")
                    break

            round_success_rate = float(success_flags.mean())
            round_task_score = float(task_scores.mean())
            round_action_valid_rate = (
                float(np.mean(action_valid_flags)) if action_valid_flags else 0.0
            )
            round_model_action_rate = (
                float(np.mean(used_model_action_flags))
                if used_model_action_flags
                else 0.0
            )

            overall_success_rates.append(round_success_rate)
            overall_task_scores.append(round_task_score)
            overall_action_valid_rates.append(round_action_valid_rate)
            overall_model_action_rates.append(round_model_action_rate)

            logging.info("Test %d success rate: %.4f", test_idx, round_success_rate)
            logging.info("Test %d average task score: %.4f", test_idx, round_task_score)
            logging.info(
                "Test %d action format valid rate: %.4f",
                test_idx,
                round_action_valid_rate,
            )
            logging.info(
                "Test %d direct model action rate: %.4f",
                test_idx,
                round_model_action_rate,
            )
            logging.info(
                "Test %d time elapsed: %.2fs",
                test_idx,
                time.time() - start_time,
            )

        logging.info("=============== Final Summary ===============")
        logging.info(
            "Total tests: %d | Envs / test: %d | Total envs: %d",
            args.test_times,
            args.env_num,
            args.env_num * args.test_times,
        )
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
            "Action format valid avg +/- std: %.4f +/- %.4f",
            float(np.mean(overall_action_valid_rates))
            if overall_action_valid_rates
            else 0.0,
            float(np.std(overall_action_valid_rates))
            if overall_action_valid_rates
            else 0.0,
        )
        logging.info(
            "Direct model action avg +/- std: %.4f +/- %.4f",
            float(np.mean(overall_model_action_rates))
            if overall_model_action_rates
            else 0.0,
            float(np.std(overall_model_action_rates))
            if overall_model_action_rates
            else 0.0,
        )
    finally:
        if env_manager is not None:
            env_manager.close()


if __name__ == "__main__":
    main()
