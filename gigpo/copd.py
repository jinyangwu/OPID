import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gigpo import core_gigpo
from utils import build_prompt_dict, chat_completion_with_retry, create_openai_client


logger = logging.getLogger(__name__)


_TASK_DESCRIPTION_PATTERNS = (
    re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your task is:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your current task is:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your question:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
)


def _clean_task_description(task_description: object) -> str:
    task = " ".join(str(task_description or "").split())
    return task.strip()


def _extract_task_description_from_text(text: object) -> str:
    text = str(text or "")
    for pattern in _TASK_DESCRIPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            task_description = _clean_task_description(match.group(1))
            if task_description:
                return task_description
    return ""


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
        task_description = (
            _extract_task_description_from_text(prompt_observation)
            or _extract_task_description_from_text(raw_observation)
        )
        if task_description:
            step_record["task_description"] = task_description
        if reward_np is not None:
            step_record["step_reward"] = float(reward_np[sample_idx])
        episodes[traj_uid].append(step_record)

    for traj_uid in episodes:
        episodes[traj_uid].sort(key=lambda step: step["step_index"])
    return episodes


class COPDEpisodeAnalyzer:
    """
    Analyze trajectories for reusable episode-level teacher guidance.

    The analyzer uses an OpenAI-compatible JSON analysis backend.
    ``azure`` remains a legacy alias for ``openai``.
    """

    def __init__(
        self,
        backend: str = "openai",
        max_history_steps: int = 6,
        max_completion_tokens: int = 4096,
    ):
        self.requested_backend = backend
        if backend == "azure":
            logger.warning("COPD backend='azure' is deprecated; using the OpenAI-compatible backend instead.")
            backend = "openai"
        elif backend != "openai":
            raise ValueError(f"Unsupported COPD backend: {backend}")

        self.backend = backend
        self.max_history_steps = max_history_steps
        self.max_completion_tokens = max_completion_tokens
        self.client = None
        self.model = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash")

        logger.info(
            "Initialized COPDEpisodeAnalyzer with requested_backend=%s, resolved_backend=%s, model=%s, max_history_steps=%s, max_completion_tokens=%s",
            self.requested_backend,
            self.backend,
            self.model,
            self.max_history_steps,
            self.max_completion_tokens,
        )

    def _get_openai_client(self):
        if self.client is None:
            self.client = create_openai_client(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        return self.client

    def analyze_episode(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, object]:
        return self._analyze_episode_with_openai(
            steps=steps,
            candidate_step_indices=candidate_step_indices,
            task_description=task_description,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
        )

    def _analyze_episode_with_openai(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, object]:
        candidate_list = (
            [int(step["step_index"]) for step in steps]
            if candidate_step_indices is None
            else [int(idx) for idx in candidate_step_indices]
        )
        prompt = self._build_episode_analysis_prompt(
            steps=steps,
            candidate_step_indices=candidate_list,
            task_description=task_description,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
        )
        content = chat_completion_with_retry(
            client=self._get_openai_client(),
            model=self.model,
            prompt=prompt,
            retries=max(1, int(os.environ.get("OPENAI_API_RETRIES", "5"))),
            retry_delay=float(os.environ.get("OPENAI_API_RETRY_DELAY", "1.0")),
            max_completion_tokens=self.max_completion_tokens,
        )
        parsed = self._parse_analysis_response(content)
        parsed["step_hints"] = {}
        parsed["analysis_backend_requested"] = self.requested_backend
        parsed["analysis_backend_used"] = "openai"
        parsed["analysis_error"] = None
        parsed["analysis_mode"] = analysis_mode
        parsed["task_description"] = task_description or self._infer_task_description(steps)
        parsed["llm_prompt"] = prompt
        parsed["llm_raw_output"] = content
        return parsed

    def _infer_task_description(self, steps: List[Dict[str, object]]) -> str:
        for step in steps:
            task_description = _clean_task_description(step.get("task_description", ""))
            if task_description:
                return task_description

        for step in steps:
            for field_name in ("observation_prompt", "observation"):
                task_description = _extract_task_description_from_text(step.get(field_name, ""))
                if task_description:
                    return task_description
        return ""

    def _build_episode_analysis_prompt(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Sequence[int],
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        del candidate_step_indices
        task_description = _clean_task_description(task_description) or self._infer_task_description(steps)

        outcome_label = "unknown"
        if episode_success is not None:
            try:
                outcome_label = "success" if float(episode_success) >= 1.0 else "failure"
            except (TypeError, ValueError):
                outcome_label = "unknown"

        if outcome_label == "success":
            episode_hint_instruction = (
                "Write one reusable episode_hint that extracts the successful workflow: "
                "the core decision rule and action ordering that made this trajectory work. "
                "Phrase it as a general skill pattern, not as instructions for this exact task."
            )
        elif outcome_label == "failure":
            episode_hint_instruction = (
                "Write one reusable episode_hint that explains why the trajectory failed and how to avoid "
                "that failure pattern next time. Focus on the core mistake and the safer general workflow. "
                "Phrase it as a general avoidance skill pattern, not as a one-off lesson tied to this exact task."
            )
        else:
            episode_hint_instruction = (
                "Write one reusable episode_hint distilled from the trajectory. If it succeeded, extract "
                "the successful workflow; if it failed, explain the failure cause and how to avoid it. "
                "Make the hint a short, broadly reusable strategy for similar task patterns."
            )
        prompt_text = f"""Analyze the following agent episode and return ONLY valid JSON.

{episode_hint_instruction}

Important constraints:
- Step indexing is 0-based: step 0 is the first step of the trajectory.
- Use the full episode context.
- Write episode_hint as sequence-level policy-facing guidance, not as a current-step instruction.
- The episode_hint is the only teacher guidance used for every OPD-scored step in this episode.
- Keep episode_hint concise: 1-2 sentences, ideally under 45 words.
- Generalize away instance details: do not mention specific product names, colors, sizes, prices, brands, or quoted search terms from the task.
- Do not list many warning signs or examples; include only the single reusable rule that should guide future behavior.

Return format:
{{
  "episode_hint": "string"
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Interaction trajectory: {self._format_episode_steps(steps)}
"""
        return build_prompt_dict(user_prompt=prompt_text)

    def _parse_analysis_response(self, response: str) -> Dict[str, object]:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            raise ValueError("No JSON object found in COPD analyzer response.")
        parsed = json.loads(response[json_start:json_end])
        if "episode_hint" not in parsed:
            raise ValueError("COPD analyzer response missing required field: episode_hint")
        return {
            "episode_summary": str(parsed.get("episode_summary", "")),
            "episode_hint": str(parsed["episode_hint"]),
            "step_hints": {},
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
