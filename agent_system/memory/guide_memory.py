import hashlib
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


_TASK_PATTERNS = (
    re.compile(
        r"Your task is to:\s*(.+?)(?=\n\n## |\nPrior to this step|\nYou are now at step|\nYour current observation is|\nYour admissible actions|\nNow it's your turn|$)",
        re.S,
    ),
    re.compile(
        r"Instruction:\s*(.+?)(?=\s*\[SEP\]|\n|$)",
        re.S,
    ),
)


def _normalize_text(text: Any, *, lowercase: bool = True, max_chars: Optional[int] = None) -> str:
    normalized = " ".join(str(text or "").split())
    if lowercase:
        normalized = normalized.lower()
    if max_chars is not None and max_chars > 0:
        normalized = normalized[:max_chars]
    return normalized


def extract_task_query(observation: Any) -> str:
    text = str(observation or "").strip()
    if not text:
        return ""

    for pattern in _TASK_PATTERNS:
        match = pattern.search(text)
        if match:
            task = " ".join(match.group(1).split()).strip(" .'\"")
            if task:
                return task
    return text


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _stable_hash(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_augmented_observation_text(
    *,
    observation: str,
    episode_guides: Sequence[str],
    step_guides: Sequence[str],
    episode_hint: str = "",
    step_hint: str = "",
) -> str:
    def _format_guidance_section(title: str, instruction: str, body: Any) -> str:
        body_text = str(body).strip()
        if not body_text:
            return ""
        return f"**{title}**\n{instruction}:\n[{body_text}]"

    def _insert_before_anchor(prompt_text: str, sections: Sequence[str], anchors: Sequence[str]) -> str:
        merged_sections = [str(section).strip() for section in sections if str(section or "").strip()]
        if not merged_sections:
            return prompt_text

        insertion_block = "\n\n".join(merged_sections)
        for anchor in anchors:
            anchor_idx = prompt_text.find(anchor)
            if anchor_idx == -1:
                continue

            prefix = prompt_text[:anchor_idx].rstrip()
            suffix = prompt_text[anchor_idx:].lstrip()
            if prefix and suffix:
                return f"{prefix}\n\n{insertion_block}\n\n{suffix}"
            if prefix:
                return f"{prefix}\n\n{insertion_block}"
            return f"{insertion_block}\n\n{suffix}"

        return f"{prompt_text}\n\n{insertion_block}" if prompt_text else insertion_block

    augmented_observation = str(observation).strip()
    episode_sections = []
    detail_sections = []

    if episode_guides:
        episode_sections.append(
            _format_guidance_section(
                "Guide Memory: Episode-Level Strategy",
                "Use the retrieved episode-level guidance to keep the overall task strategy in mind.",
                "\n".join(f"- {guide}" for guide in episode_guides),
            )
        )
    if step_guides:
        detail_sections.append(
            _format_guidance_section(
                "Guide Memory: Similar-Step Action Hints",
                "Use these retrieved hints as references for choosing the next admissible action in the current state.",
                "\n".join(f"- {guide}" for guide in step_guides),
            )
        )
    if episode_hint:
        detail_sections.append(
            _format_guidance_section(
                "Episode-Level Hint",
                "This hint summarizes the intended strategy for the whole task; use it to guide planning across steps.",
                episode_hint,
            )
        )
    if step_hint:
        detail_sections.append(
            _format_guidance_section(
                "Current-Step Decision Guidance",
                (
                    "This is policy-facing advice for the current decision point. "
                    "Use it to guide your reasoning and select the next admissible action."
                ),
                step_hint,
            )
        )

    augmented_observation = _insert_before_anchor(
        augmented_observation,
        episode_sections + detail_sections,
        (
            "Now it's your turn to",
            "Now it's your turn",
        ),
    )
    return augmented_observation


@dataclass
class GuideRecord:
    guide_id: str
    guide_text: str
    task_signature: str
    source: str
    status: str
    support_count: int
    created_step: int
    last_updated_step: int
    last_used_step: int = -1
    state_signature: Optional[str] = None
    state_preview: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class COPDGuideMemory:
    """
    Online guide cache for COPD.

    This module is intentionally parallel to ``skills_only_memory``: it stores
    self-evolving episode-level and step-level guidance gathered from COPD
    trajectory analysis, and does not depend on any external skill bank.
    """

    def __init__(self, config, default_dump_dir: Optional[str] = None):
        self.config = config
        self.enabled = bool(config.get("enable", False))
        self.episode_enable = bool(config.get("episode_enable", True))
        self.step_enable = bool(config.get("step_enable", True))
        self.episode_top_k = int(config.get("episode_top_k", 1))
        self.step_top_k = int(config.get("step_top_k", 1))
        self.promote_min_support = int(config.get("promote_min_support", 2))
        self.merge_similarity_thresh = float(config.get("merge_similarity_thresh", 0.9))
        self.state_similarity_thresh = float(config.get("state_similarity_thresh", 0.92))
        self.max_episode_guides_per_task = int(config.get("max_episode_guides_per_task", 8))
        self.max_step_guides_per_task = int(config.get("max_step_guides_per_task", 24))
        self.max_task_chars = int(config.get("max_task_chars", 256))
        self.max_state_chars = int(config.get("max_state_chars", 384))
        self.max_episode_guide_chars = int(config.get("max_episode_guide_chars", 256))
        self.max_step_guide_chars = int(config.get("max_step_guide_chars", 224))
        self.dump_freq_steps = int(config.get("dump_freq_steps", 0))

        dump_dir = config.get("dump_dir", None)
        if dump_dir:
            self.dump_dir = str(dump_dir)
        elif default_dump_dir:
            self.dump_dir = os.path.join(default_dump_dir, "guide_memory")
        else:
            self.dump_dir = None

        self._episode_guides: Dict[str, List[GuideRecord]] = defaultdict(list)
        self._step_guides: Dict[str, List[GuideRecord]] = defaultdict(list)

    def _task_signature(self, observation: Any) -> Tuple[str, str]:
        task_query = extract_task_query(observation)
        normalized = _normalize_text(task_query, max_chars=self.max_task_chars)
        return task_query, normalized

    def _task_signature_from_query(self, task_query: Any) -> Tuple[str, str]:
        normalized = _normalize_text(task_query, max_chars=self.max_task_chars)
        return str(task_query or "").strip(), normalized

    def _state_signature(self, task_signature: str, observation: Any) -> Tuple[str, str]:
        state_preview = _normalize_text(observation, max_chars=self.max_state_chars)
        state_signature = _stable_hash([task_signature, state_preview])
        return state_signature, state_preview

    def _is_promoted(self, record: GuideRecord) -> bool:
        return record.support_count >= self.promote_min_support

    def _refresh_status(self, record: GuideRecord) -> None:
        record.status = "active" if self._is_promoted(record) else "pending"

    def _rank_record(self, record: GuideRecord) -> Tuple[int, int]:
        return (
            record.support_count,
            record.last_updated_step,
        )

    def _match_episode_record(self, records: Sequence[GuideRecord], guide_text: str) -> Optional[GuideRecord]:
        for record in records:
            if _text_similarity(record.guide_text, guide_text) >= self.merge_similarity_thresh:
                return record
        return None

    def _match_step_record(
        self,
        records: Sequence[GuideRecord],
        guide_text: str,
        state_signature: str,
        state_preview: str,
    ) -> Optional[GuideRecord]:
        for record in records:
            if record.state_signature == state_signature and _text_similarity(record.guide_text, guide_text) >= self.merge_similarity_thresh:
                return record
            if (
                record.state_preview
                and _text_similarity(record.state_preview, state_preview) >= self.state_similarity_thresh
                and _text_similarity(record.guide_text, guide_text) >= self.merge_similarity_thresh
            ):
                return record
        return None

    def _prune_episode_records(self, task_signature: str) -> None:
        records = self._episode_guides[task_signature]
        records.sort(
            key=lambda record: (
                1 if record.status == "active" else 0,
                *self._rank_record(record),
            ),
            reverse=True,
        )
        self._episode_guides[task_signature] = records[: self.max_episode_guides_per_task]

    def _prune_step_records(self, task_signature: str) -> None:
        records = self._step_guides[task_signature]
        records.sort(
            key=lambda record: (
                1 if record.status == "active" else 0,
                *self._rank_record(record),
            ),
            reverse=True,
        )
        self._step_guides[task_signature] = records[: self.max_step_guides_per_task]

    def _add_episode_guide(
        self,
        *,
        task_signature: str,
        guide_text: str,
        global_step: int,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.episode_enable:
            return False
        normalized_guide = _normalize_text(guide_text, lowercase=False, max_chars=self.max_episode_guide_chars)
        if not normalized_guide:
            return False

        records = self._episode_guides[task_signature]
        record = self._match_episode_record(records, normalized_guide)
        if record is None:
            record = GuideRecord(
                guide_id=f"episode_{uuid.uuid4().hex[:12]}",
                guide_text=normalized_guide,
                task_signature=task_signature,
                source=source,
                status="pending",
                support_count=0,
                created_step=global_step,
                last_updated_step=global_step,
                metadata=metadata or {},
            )
            records.append(record)

        record.support_count += 1
        record.last_updated_step = global_step
        if metadata:
            record.metadata.update(metadata)
        self._refresh_status(record)
        self._prune_episode_records(task_signature)
        return True

    def _add_step_guide(
        self,
        *,
        task_signature: str,
        state_signature: str,
        state_preview: str,
        guide_text: str,
        global_step: int,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.step_enable:
            return False
        normalized_guide = _normalize_text(guide_text, lowercase=False, max_chars=self.max_step_guide_chars)
        if not normalized_guide:
            return False

        records = self._step_guides[task_signature]
        record = self._match_step_record(
            records,
            normalized_guide,
            state_signature=state_signature,
            state_preview=state_preview,
        )
        if record is None:
            record = GuideRecord(
                guide_id=f"step_{uuid.uuid4().hex[:12]}",
                guide_text=normalized_guide,
                task_signature=task_signature,
                source=source,
                status="pending",
                support_count=0,
                created_step=global_step,
                last_updated_step=global_step,
                state_signature=state_signature,
                state_preview=state_preview,
                metadata=metadata or {},
            )
            records.append(record)

        record.support_count += 1
        record.last_updated_step = global_step
        if metadata:
            record.metadata.update(metadata)
        self._refresh_status(record)
        self._prune_step_records(task_signature)
        return True

    def _select_episode_guides(
        self,
        task_signature: str,
        *,
        top_k: int,
        global_step: Optional[int] = None,
    ) -> List[str]:
        active_episode = [
            record for record in self._episode_guides.get(task_signature, [])
            if record.status == "active"
        ]
        active_episode.sort(key=self._rank_record, reverse=True)
        selected_episode = active_episode[: max(int(top_k), 0)]
        for record in selected_episode:
            if global_step is not None:
                record.last_used_step = global_step
        return [record.guide_text for record in selected_episode]

    def _select_step_guides(
        self,
        task_signature: str,
        *,
        observation: Any,
        top_k: int,
        global_step: Optional[int] = None,
    ) -> List[str]:
        _, current_state_preview = self._state_signature(task_signature, observation)
        scored_records = []
        for record in self._step_guides.get(task_signature, []):
            if record.status != "active" or not record.state_preview:
                continue
            similarity = _text_similarity(current_state_preview, record.state_preview)
            if similarity < self.state_similarity_thresh:
                continue
            scored_records.append((similarity, record))

        scored_records.sort(key=lambda item: (item[0], *self._rank_record(item[1])), reverse=True)
        selected_step = [record for _, record in scored_records[: max(int(top_k), 0)]]
        for record in selected_step:
            if global_step is not None:
                record.last_used_step = global_step
        return [record.guide_text for record in selected_step]

    def retrieve_guides(
        self,
        *,
        task_query: Any,
        observation: Any = None,
        global_step: Optional[int] = None,
        include_episode: bool = True,
        include_step: bool = True,
        episode_top_k: Optional[int] = None,
        step_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        task_query_text, task_signature = self._task_signature_from_query(task_query)
        result = {
            "task_query": task_query_text,
            "task_signature": task_signature,
            "episode_guides": [],
            "step_guides": [],
        }
        if not self.enabled:
            return result

        if include_episode and self.episode_enable:
            result["episode_guides"] = self._select_episode_guides(
                task_signature,
                top_k=self.episode_top_k if episode_top_k is None else episode_top_k,
                global_step=global_step,
            )

        if include_step and self.step_enable and observation is not None:
            result["step_guides"] = self._select_step_guides(
                task_signature,
                observation=observation,
                top_k=self.step_top_k if step_top_k is None else step_top_k,
                global_step=global_step,
            )

        return result

    def retrieve_for_observation(
        self,
        *,
        observation: Any,
        anchor_observation: Any = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        task_query, _ = self._task_signature(observation)
        state_source = anchor_observation if anchor_observation is not None else observation
        return self.retrieve_guides(
            task_query=task_query,
            observation=state_source,
            global_step=global_step,
            include_episode=True,
            include_step=True,
        )

    def build_augmented_observation(
        self,
        *,
        observation: str,
        episode_guides: Sequence[str],
        step_guides: Sequence[str],
        episode_hint: str = "",
        episode_summary: str = "",
        hindsight_hint: str = "",
    ) -> str:
        return build_augmented_observation_text(
            observation=observation,
            episode_guides=episode_guides,
            step_guides=step_guides,
            episode_hint=episode_hint or episode_summary,
            step_hint=hindsight_hint,
        )

    def format_for_rollout_prompt(
        self,
        *,
        episode_guides: Sequence[str],
        step_guides: Sequence[str],
    ) -> str:
        sections = []
        if episode_guides:
            sections.append(
                "## Guide Memory: Episode-Level Strategy\n"
                "Use the retrieved episode-level guidance to keep the overall task strategy in mind.\n"
                + "\n".join(f"- {guide}" for guide in episode_guides)
            )
        if step_guides:
            sections.append(
                "## Guide Memory: Similar-Step Action Hints\n"
                "Use these retrieved hints as references for choosing the next admissible action in the current state.\n"
                + "\n".join(f"- {guide}" for guide in step_guides)
            )
        return "\n\n".join(section for section in sections if section)

    def update_from_episode_analysis(
        self,
        *,
        obs_texts: Sequence[Any],
        anchor_obs: Optional[Sequence[Any]],
        traj_uids: Sequence[Any],
        step_indices: Sequence[int],
        critical_mask: Sequence[bool],
        episode_analysis: Dict[Any, Dict[str, Any]],
        global_step: int,
        episode_success: Optional[Sequence[float]] = None,
        analysis_mode: str = "teacher_bootstrap",
    ) -> Dict[str, float]:
        metrics = {
            "copd/guide_memory/enabled": 1.0 if self.enabled else 0.0,
            "copd/guide_memory/episode_candidates_added": 0.0,
            "copd/guide_memory/step_candidates_added": 0.0,
        }
        if not self.enabled:
            return metrics

        critical_mask_list = [bool(value) for value in critical_mask]
        step_indices_list = [int(value) for value in step_indices]

        traj_to_indices: Dict[Any, List[int]] = defaultdict(list)
        for sample_idx, traj_uid in enumerate(traj_uids):
            traj_to_indices[traj_uid].append(sample_idx)

        traj_success: Dict[Any, float] = {}
        if episode_success is not None:
            success_list = [float(value) for value in episode_success]
            for traj_uid, sample_indices in traj_to_indices.items():
                first_idx = min(sample_indices, key=lambda idx: step_indices_list[idx])
                traj_success[traj_uid] = success_list[first_idx]

        for traj_uid, sample_indices in traj_to_indices.items():
            if not sample_indices:
                continue
            representative_idx = min(sample_indices, key=lambda idx: step_indices_list[idx])
            task_query, task_signature = self._task_signature(obs_texts[representative_idx])
            task_source_text = str(obs_texts[representative_idx] or "").strip()
            if (
                anchor_obs is not None
                and representative_idx < len(anchor_obs)
                and task_query == task_source_text
            ):
                anchor_task_query, anchor_task_signature = self._task_signature(anchor_obs[representative_idx])
                if anchor_task_signature and anchor_task_query != str(anchor_obs[representative_idx] or "").strip():
                    task_signature = anchor_task_signature
            analysis = episode_analysis.get(traj_uid, {})
            traj_success_value = traj_success.get(traj_uid)
            episode_guide_text = str(
                analysis.get("episode_hint")
                or analysis.get("overall_hint")
                or analysis.get("episode_summary")
                or ""
            ).strip()
            if episode_guide_text:
                added = self._add_episode_guide(
                    task_signature=task_signature,
                    guide_text=episode_guide_text,
                    global_step=global_step,
                    source=f"{analysis_mode}/episode_analysis",
                    metadata={
                        "traj_uid": str(traj_uid),
                        "analysis_mode": analysis_mode,
                        "episode_success": traj_success_value,
                    },
                )
                metrics["copd/guide_memory/episode_candidates_added"] += float(added)

            step_hints = analysis.get("step_hints", {})
            for sample_idx in sample_indices:
                if not critical_mask_list[sample_idx]:
                    continue
                step_idx = step_indices_list[sample_idx]
                step_hint = str(step_hints.get(step_idx, "")).strip()
                if not step_hint:
                    continue
                state_source = (
                    anchor_obs[sample_idx]
                    if anchor_obs is not None and sample_idx < len(anchor_obs)
                    else obs_texts[sample_idx]
                )
                state_signature, state_preview = self._state_signature(task_signature, state_source)
                added = self._add_step_guide(
                    task_signature=task_signature,
                    state_signature=state_signature,
                    state_preview=state_preview,
                    guide_text=step_hint,
                    global_step=global_step,
                    source=f"{analysis_mode}/critical_step_hint",
                    metadata={
                        "traj_uid": str(traj_uid),
                        "step_idx": step_idx,
                        "analysis_mode": analysis_mode,
                        "episode_success": traj_success_value,
                    },
                )
                metrics["copd/guide_memory/step_candidates_added"] += float(added)

        metrics.update(self.snapshot_metrics(prefix="copd/guide_memory"))
        self._dump_if_needed(global_step=global_step)
        return metrics

    def snapshot_metrics(self, prefix: str = "guide_memory") -> Dict[str, float]:
        episode_records = [record for records in self._episode_guides.values() for record in records]
        step_records = [record for records in self._step_guides.values() for record in records]

        def _count(records: Sequence[GuideRecord], status: str) -> float:
            return float(sum(1 for record in records if record.status == status))

        return {
            f"{prefix}/episode_tasks": float(len(self._episode_guides)),
            f"{prefix}/step_tasks": float(len(self._step_guides)),
            f"{prefix}/episode_total": float(len(episode_records)),
            f"{prefix}/episode_active": _count(episode_records, "active"),
            f"{prefix}/episode_pending": _count(episode_records, "pending"),
            f"{prefix}/step_total": float(len(step_records)),
            f"{prefix}/step_active": _count(step_records, "active"),
            f"{prefix}/step_pending": _count(step_records, "pending"),
        }

    def dump_snapshot(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "episode_guides": {
                task_signature: [record.to_json() for record in records]
                for task_signature, records in self._episode_guides.items()
            },
            "step_guides": {
                task_signature: [record.to_json() for record in records]
                for task_signature, records in self._step_guides.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _dump_if_needed(self, global_step: int) -> None:
        if not self.dump_dir or self.dump_freq_steps <= 0:
            return
        if global_step % self.dump_freq_steps != 0:
            return
        filename = os.path.join(self.dump_dir, f"step_{global_step:08d}.json")
        try:
            self.dump_snapshot(filename)
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            logger.warning("Failed to dump guide memory snapshot to %s: %s", filename, exc)
