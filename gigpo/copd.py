import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gigpo import core_gigpo
from gigpo.prompts import SUMMARY_SYSTEM_PROMPT
from utils import build_prompt_dict, chat_completion_with_retry, create_openai_client


logger = logging.getLogger(__name__)


def build_traj_step_indices(traj_index: Sequence) -> np.ndarray:
    """Return per-sample step indices in trajectory order."""
    counters: Dict[object, int] = defaultdict(int)
    step_indices = []
    for traj_uid in traj_index:
        step_indices.append(counters[traj_uid])
        counters[traj_uid] += 1
    return np.asarray(step_indices, dtype=np.int64)


def select_critical_steps_by_stats(
    step_rewards: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    min_group_size: int = 2,
    var_quantile: float = 0.75,
    topk_per_traj: int = 1,
    below_group_mean_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Select critical steps using reward statistics of each observation group.

    A step becomes a candidate when its observation group has sufficiently high
    return variance. Candidate steps are ranked by how much they underperform
    the group mean (or deviate from it), and we keep the top-k per trajectory.
    """
    if topk_per_traj <= 0 or len(step_rewards) == 0:
        return np.zeros(len(step_rewards), dtype=bool), {
            "num_groups": 0.0,
            "eligible_groups": 0.0,
            "selected_steps": 0.0,
        }

    step_group_uids = core_gigpo.build_step_group(
        anchor_obs=anchor_obs,
        index=index,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    step_rewards_np = step_rewards.detach().cpu().numpy().astype(np.float32)
    step_indices = build_traj_step_indices(traj_index)

    group_to_indices: Dict[object, List[int]] = defaultdict(list)
    for sample_idx, group_uid in enumerate(step_group_uids):
        group_to_indices[group_uid].append(sample_idx)

    group_stats = {}
    eligible_group_vars: List[float] = []
    for group_uid, sample_indices in group_to_indices.items():
        rewards = step_rewards_np[sample_indices]
        group_mean = float(np.mean(rewards))
        group_var = float(np.var(rewards))
        group_stats[group_uid] = {
            "mean": group_mean,
            "var": group_var,
            "size": float(len(sample_indices)),
        }
        if len(sample_indices) >= min_group_size:
            eligible_group_vars.append(group_var)

    if eligible_group_vars:
        clipped_quantile = min(max(var_quantile, 0.0), 1.0)
        var_cutoff = float(np.quantile(np.asarray(eligible_group_vars, dtype=np.float32), clipped_quantile))
    else:
        var_cutoff = float("inf")

    sample_scores = np.zeros(len(step_rewards_np), dtype=np.float32)
    candidate_mask = np.zeros(len(step_rewards_np), dtype=bool)
    for sample_idx, group_uid in enumerate(step_group_uids):
        group_stat = group_stats[group_uid]
        group_mean = group_stat["mean"]
        group_var = group_stat["var"]
        group_size = int(group_stat["size"])
        if group_size < min_group_size or group_var < var_cutoff or group_var <= 0.0:
            continue

        reward = step_rewards_np[sample_idx]
        if below_group_mean_only:
            deviation = max(group_mean - reward, 0.0)
        else:
            deviation = abs(reward - group_mean)
        if deviation <= 0.0 and below_group_mean_only:
            continue

        candidate_mask[sample_idx] = True
        sample_scores[sample_idx] = group_var * (deviation + 1e-6)

    critical_mask = np.zeros(len(step_rewards_np), dtype=bool)
    traj_to_candidates: Dict[object, List[int]] = defaultdict(list)
    for sample_idx, is_candidate in enumerate(candidate_mask):
        if is_candidate:
            traj_to_candidates[traj_index[sample_idx]].append(sample_idx)

    for traj_uid, sample_indices in traj_to_candidates.items():
        ranked = sorted(
            sample_indices,
            key=lambda i: (sample_scores[i], step_indices[i]),
            reverse=True,
        )
        for sample_idx in ranked[:topk_per_traj]:
            critical_mask[sample_idx] = True

    metrics = {
        "num_groups": float(len(group_to_indices)),
        "eligible_groups": float(sum(
            1
            for group_stat in group_stats.values()
            if int(group_stat["size"]) >= min_group_size and group_stat["var"] >= var_cutoff and group_stat["var"] > 0.0
        )),
        "selected_steps": float(critical_mask.sum()),
        "variance_cutoff": 0.0 if not np.isfinite(var_cutoff) else float(var_cutoff),
    }
    return critical_mask, metrics


def build_episode_records(
    tokenizer,
    obs_texts: Sequence,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    traj_index: Sequence,
    step_indices: Sequence,
    step_rewards: Optional[torch.Tensor] = None,
    obs_raws: Optional[Sequence] = None,
) -> Dict[object, List[Dict[str, object]]]:
    """Decode trajectories into step records for analysis.

    `observation` prefers raw text feedback from the environment when
    available (for example `anchor_obs` in rollout batches). The prompt-form
    observation is preserved in `observation_prompt` for debugging.
    """
    if step_rewards is not None:
        reward_np = step_rewards.detach().cpu().numpy().astype(np.float32)
    else:
        reward_np = None

    episodes: Dict[object, List[Dict[str, object]]] = defaultdict(list)
    responses_cpu = responses.detach().cpu()
    response_mask_cpu = response_mask.detach().cpu()
    for sample_idx, traj_uid in enumerate(traj_index):
        valid_len = int(response_mask_cpu[sample_idx].sum().item())
        response_text = tokenizer.decode(
            responses_cpu[sample_idx][:valid_len],
            skip_special_tokens=True,
        )
        prompt_observation = str(obs_texts[sample_idx])
        raw_observation = prompt_observation
        if obs_raws is not None:
            candidate_raw_observation = obs_raws[sample_idx]
            if isinstance(candidate_raw_observation, (str, np.str_)):
                raw_observation = str(candidate_raw_observation)
        step_record = {
            "step_index": int(step_indices[sample_idx]),
            "observation": raw_observation,
            "observation_prompt": prompt_observation,
            "response": response_text,
        }
        if reward_np is not None:
            step_record["step_reward"] = float(reward_np[sample_idx])
        episodes[traj_uid].append(step_record)

    for traj_uid in episodes:
        episodes[traj_uid].sort(key=lambda step: step["step_index"])
    return episodes


def build_enhanced_observation_text(observation: str, episode_summary: str, hindsight_hint: str) -> str:
    sections = [observation.strip()]
    if episode_summary:
        sections.append("[Episode Summary]\n" + episode_summary.strip())
    if hindsight_hint:
        sections.append("[Hindsight Hint For Current Step]\n" + hindsight_hint.strip())
    return "\n\n".join(section for section in sections if section)


def _truncate_text(text: str, max_chars: int = 160) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = str(text).lower()
    return any(keyword in lowered for keyword in keywords)


def _generate_heuristic_step_hint(
    current_step: Dict[str, object],
    previous_step: Optional[Dict[str, object]],
    next_step: Optional[Dict[str, object]],
) -> str:
    current_obs = str(current_step.get("observation", ""))
    current_resp = str(current_step.get("response", ""))
    current_reward = float(current_step.get("step_reward", 0.0))

    if next_step is not None:
        next_obs = str(next_step.get("observation", ""))
        next_resp = str(next_step.get("response", ""))
        next_reward = float(next_step.get("step_reward", 0.0))

        if _contains_any(next_obs, ("invalid", "fail", "not available", "nothing happens", "error", "cannot")):
            return (
                "The next observation suggests the current response did not work. "
                "Avoid repeating it and choose a different action grounded in the visible state."
            )

        if current_obs and next_obs and _truncate_text(current_obs, 120) == _truncate_text(next_obs, 120):
            return (
                "The environment barely changed after this step, so progress likely stalled. "
                "Pick a meaningfully different next move instead of paraphrasing the same plan."
            )

        if current_resp and next_resp and _truncate_text(current_resp, 120) == _truncate_text(next_resp, 120):
            return (
                "The trajectory is repeating nearly the same response. "
                "Use the current observation to break the loop and commit to a different action."
            )

        if next_reward < current_reward:
            return (
                "The return gets worse after this step. "
                "Re-evaluate the current observation and prefer a safer next action that directly addresses the visible constraint."
            )

    if previous_step is not None:
        previous_reward = float(previous_step.get("step_reward", 0.0))
        if current_reward < previous_reward:
            return (
                "This step underperforms the previous one. "
                "Double-check whether the response matches the current observation before continuing."
            )

    if _contains_any(current_obs, ("option", "available", "price", "button", "search results", "click", "buy now")):
        return (
            "The observation exposes concrete options or constraints. "
            "Base the next response on those visible details instead of staying generic."
        )

    if _contains_any(current_obs, ("look", "visible", "found", "location", "inside", "on the")):
        return (
            "Use the concrete state information in the observation to choose the next step. "
            "Focus on the object, location, or interaction that is explicitly visible now."
        )

    return (
        "Re-read the current observation, verify whether the last response actually changed the state, "
        "and choose the next action that most directly advances the task."
    )


class COPDEpisodeAnalyzer:
    """
    Analyze trajectories for hindsight summaries and step hints.

    Backend choices:
    - ``heuristic``: local fallback, no external dependency.
    - ``openai``: OpenAI-compatible JSON analysis through ``utils.openai_api``.
    - ``azure``: legacy alias for ``openai``.
    """

    def __init__(
        self,
        backend: str = "heuristic",
        max_history_steps: int = 6,
        max_completion_tokens: int = 4096,
        max_selected_steps_per_traj: int = 1,
    ):
        self.requested_backend = backend
        if backend == "azure":
            logger.warning("COPD backend='azure' is deprecated; using the OpenAI-compatible backend instead.")
            backend = "openai"
        elif backend == "google":
            logger.warning("COPD backend='google' is no longer supported here; falling back to heuristic.")
            backend = "heuristic"
        elif backend not in {"heuristic", "openai"}:
            raise ValueError(f"Unsupported COPD backend: {backend}")

        self.backend = backend
        self.max_history_steps = max_history_steps
        self.max_completion_tokens = max_completion_tokens
        self.max_selected_steps_per_traj = max_selected_steps_per_traj
        self.client = None
        self.model = None

        if self.backend == "openai":
            self.model = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash")
            self.client = create_openai_client(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )

        logger.info(
            "Initialized COPDEpisodeAnalyzer with requested_backend=%s, resolved_backend=%s, model=%s, max_history_steps=%s, max_completion_tokens=%s, max_selected_steps_per_traj=%s",
            self.requested_backend,
            self.backend,
            self.model,
            self.max_history_steps,
            self.max_completion_tokens,
            self.max_selected_steps_per_traj,
        )

    def analyze_episode(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        select_steps: bool = False,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
    ) -> Dict[str, object]:
        if self.backend == "openai":
            max_retries = 3
            last_error: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return self._analyze_episode_with_openai(
                        steps=steps,
                        candidate_step_indices=candidate_step_indices,
                        select_steps=select_steps,
                        analysis_mode=analysis_mode,
                        episode_success=episode_success,
                    )
                except Exception as exc:  # pragma: no cover - runtime fallback
                    last_error = exc
                    if attempt < max_retries:
                        logger.warning(
                            "COPD OpenAI analysis attempt %s/%s failed, retrying: %s",
                            attempt + 1,
                            max_retries + 1,
                            exc,
                        )
                    else:
                        logger.warning("COPD OpenAI analysis failed, falling back to heuristic: %s", exc)

            fallback = self._analyze_episode_with_heuristic(
                steps=steps,
                candidate_step_indices=candidate_step_indices,
                select_steps=select_steps,
                analysis_mode=analysis_mode,
                episode_success=episode_success,
            )
            fallback["analysis_backend_requested"] = self.requested_backend
            fallback["analysis_backend_used"] = "heuristic"
            fallback["analysis_error"] = str(last_error) if last_error is not None else None
            return fallback
        return self._analyze_episode_with_heuristic(
            steps=steps,
            candidate_step_indices=candidate_step_indices,
            select_steps=select_steps,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
        )

    def _analyze_episode_with_heuristic(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        select_steps: bool = False,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
    ) -> Dict[str, object]:
        if not steps:
            return {
                "episode_summary": "",
                "overall_hint": "",
                "selected_steps": [],
                "step_hints": {},
                "analysis_backend_requested": self.requested_backend,
                "analysis_backend_used": "heuristic",
                "analysis_error": None,
                "llm_prompt": None,
                "llm_raw_output": None,
            }

        tail_steps = steps[-self.max_history_steps :]
        summary_lines = []
        for step in tail_steps:
            reward_str = ""
            if "step_reward" in step:
                reward_str = f" | return={step['step_reward']:.3f}"
            summary_lines.append(
                f"Step {step['step_index']}: obs={step['observation'][:120]} | resp={step['response'][:120]}{reward_str}"
            )
        episode_summary = "Recent trajectory context:\n" + "\n".join(summary_lines)

        if candidate_step_indices is None:
            candidate_step_indices = [int(step["step_index"]) for step in steps]

        selected_steps: List[int] = []
        if select_steps:
            scored = []
            for step in steps:
                step_idx = int(step["step_index"])
                if step_idx not in candidate_step_indices:
                    continue
                step_reward = float(step.get("step_reward", 0.0))
                scored.append((step_reward, step_idx))
            scored.sort(key=lambda item: (item[0], item[1]))
            selected_steps = [step_idx for _, step_idx in scored[: self.max_selected_steps_per_traj]]
        else:
            selected_steps = [int(step_idx) for step_idx in candidate_step_indices]

        step_hints = {}
        for step_pos, step in enumerate(steps):
            step_idx = int(step["step_index"])
            if step_idx not in selected_steps:
                continue

            previous_step = steps[step_pos - 1] if step_pos > 0 else None
            next_step = steps[step_pos + 1] if step_pos + 1 < len(steps) else None
            hint = _generate_heuristic_step_hint(
                current_step=step,
                previous_step=previous_step,
                next_step=next_step,
            )
            step_hints[step_idx] = hint

        overall_hint = "Summarize the useful trajectory pattern and reuse it when the same task structure appears again."

        return {
            "episode_summary": episode_summary,
            "overall_hint": overall_hint,
            "selected_steps": selected_steps,
            "step_hints": step_hints,
            "analysis_backend_requested": self.requested_backend,
            "analysis_backend_used": "heuristic",
            "analysis_error": None,
            "llm_prompt": None,
            "llm_raw_output": None,
        }

    def _analyze_episode_with_openai(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        select_steps: bool = False,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
    ) -> Dict[str, object]:
        candidate_list = [] if candidate_step_indices is None else [int(idx) for idx in candidate_step_indices]
        prompt = self._build_episode_analysis_prompt(
            steps=steps,
            candidate_step_indices=candidate_list,
            select_steps=select_steps,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
        )
        content = chat_completion_with_retry(
            client=self.client,
            model=self.model,
            prompt=prompt,
            retries=max(1, int(os.environ.get("OPENAI_API_RETRIES", "5"))),
            retry_delay=float(os.environ.get("OPENAI_API_RETRY_DELAY", "1.0")),
            max_completion_tokens=self.max_completion_tokens,
        )
        parsed = self._parse_analysis_response(content)
        if select_steps:
            parsed["selected_steps"] = parsed.get("selected_steps", [])[: self.max_selected_steps_per_traj]
        parsed["analysis_backend_requested"] = self.requested_backend
        parsed["analysis_backend_used"] = "openai"
        parsed["analysis_error"] = None
        parsed["llm_prompt"] = prompt
        parsed["llm_raw_output"] = content
        return parsed

    def _build_episode_analysis_prompt(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Sequence[int],
        select_steps: bool,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
    ) -> Dict[str, Any]:
        selection_instruction = (
            "Select at most "
            f"{self.max_selected_steps_per_traj} critical steps from the candidate set and provide one hindsight hint for each selected step."
            if select_steps
            else "Provide one hindsight hint for every candidate step."
        )
        summary_instruction = "Write a concise episode_summary."
        overall_hint_instruction = "Write one reusable overall_hint distilled from the trajectory."
        prompt_text = f"""Analyze the following agent episode and return ONLY valid JSON.

Tasks:
1. {summary_instruction}
2. {overall_hint_instruction}
3. {selection_instruction}

Important constraints:
- Step indexing is 0-based: step 0 is the first step of the trajectory.
- episode_success: {episode_success}

Candidate step indices: {candidate_step_indices}

Return format:
{{
  "episode_summary": "string",
  "overall_hint": "string",
  "selected_steps": [0, 2],
  "step_hints": {{
    "0": "hint for step 0",
    "2": "hint for step 2"
  }}
}}

Episode:
{self._format_episode_steps(steps)}"""
        return build_prompt_dict(user_prompt=prompt_text)

    def _build_episode_analysis_prompt_v2(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Sequence[int],
        select_steps: bool,
    ) -> Dict[str, Any]:
        selection_instruction = (
            "Select at most "
            f"{self.max_selected_steps_per_traj} important steps from the candidate set."
            if select_steps
            else "Return one important_steps entry for every candidate step in the candidate set."
        )
        prompt_text = f"""{SUMMARY_SYSTEM_PROMPT}

Rules:
- Step indexing is 0-based: step 0 is the first step of the trajectory.
- {selection_instruction}
- Only choose step_index values from this candidate set: {candidate_step_indices}

Episode:
{self._format_episode_steps(steps)}"""
        return build_prompt_dict(user_prompt=prompt_text)

    def _parse_analysis_response(self, response: str) -> Dict[str, object]:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            raise ValueError("No JSON object found in COPD analyzer response.")
        parsed = json.loads(response[json_start:json_end])
        step_hints_raw = parsed.get("step_hints", {})
        step_hints = {int(step_idx): str(hint) for step_idx, hint in step_hints_raw.items()}
        selected_steps = [int(step_idx) for step_idx in parsed.get("selected_steps", step_hints.keys())]
        return {
            "episode_summary": str(parsed.get("episode_summary", "")),
            "overall_hint": str(parsed.get("overall_hint", "")),
            "selected_steps": selected_steps,
            "step_hints": step_hints,
        }

    def _format_episode_steps(self, steps: List[Dict[str, object]]) -> str:
        step_lines = []
        for step in steps:
            reward_str = ""
            if "step_reward" in step:
                reward_str = f"\nReturn: {step['step_reward']:.6f}"
            step_lines.append(
                f"Step {step['step_index']}\n"
                f"Observation: {step['observation']}\n"
                f"Response: {step['response']}{reward_str}\n"
            )
        return "".join(step_lines)
