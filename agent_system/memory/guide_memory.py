import hashlib
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


_TASK_PATTERNS = (
    re.compile(
        r"Your task is to:\s*(.+?)(?=\n\n## |\nPrior to this step|\nYou are now at step|\nYour current observation is|\nYour admissible actions|\nNow it's your turn|$)",
        re.S,
    ),
    re.compile(
        r"Your current task is:\s*(.+?)(?=\n\n## |\nPrior to this step|\nYou are now at step|\nYour current observation is|\nYour admissible actions|\nNow it's your turn|$)",
        re.S,
    ),
    re.compile(
        r"Instruction:\s*(.+?)(?=\s*\[SEP\]|\n|$)",
        re.S,
    ),
)


_HASHING_EMBEDDING_ALIASES = {"", "none", "null", "hash", "hashing", "hashing-fallback"}
_EMBEDDING_DEVICE_AUTO_ALIASES = {"", "auto", "none", "null"}
DEFAULT_EMBEDDING_MODEL_PATH = "/raid3/data/GTPO/MODELS/Qwen3-Embedding-0.6B"
SUCCESS_WORKFLOW = "success_workflow"
FAILURE_AVOIDANCE = "failure_avoidance"
UNKNOWN_SKILL_TYPE = "unknown"
_SKILL_TYPE_LABELS = {
    SUCCESS_WORKFLOW: "Success workflow",
    FAILURE_AVOIDANCE: "Failure avoidance",
    UNKNOWN_SKILL_TYPE: "Reusable skill",
}


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


def _slugify_text(text: Any, *, max_chars: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return slug[:max_chars].strip("_") or "general_task"


def _infer_task_pattern(task_text: Any) -> str:
    task = _normalize_text(task_text, lowercase=True)
    if not task:
        return "general_task_execution"

    if "clean" in task:
        return "clean_object_workflow"
    if "heat" in task:
        return "heat_object_workflow"
    if "cool" in task:
        return "cool_object_workflow"
    if "look at" in task and "under" in task:
        return "look_at_object_in_light"
    webshop_strong_terms = (
        "dollar", "$", "price", "rating", "reviews", "buy", "purchase",
        "amazon", "product", "shopping",
    )
    webshop_attribute_terms = (
        "under", "over", "size", "color", "material", "brand",
    )
    webshop_category_terms = (
        "shirt", "shoe", "mug", "laptop", "dress", "jacket", "cream",
        "bottle", "bag", "book", "watch", "case",
    )
    if (
        any(term in task for term in webshop_strong_terms)
        or (
            any(term in task for term in webshop_category_terms)
            and any(term in task for term in webshop_attribute_terms)
        )
    ):
        return "constrained_product_search"
    if any(term in task for term in ("examine", "inspect", "find", "look")):
        return "object_location_and_examination"
    if any(term in task for term in ("search", "answer", "question", "respond")):
        return "information_search"
    return "general_task_execution"


def _title_from_skill_text(skill_text: Any, skill_type: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(skill_text or ""))
    if words:
        title = " ".join(words[:5]).strip()
    else:
        title = _SKILL_TYPE_LABELS.get(skill_type, _SKILL_TYPE_LABELS[UNKNOWN_SKILL_TYPE])
    return title[:80]


def _hashing_embedding(text: str, dim: int) -> List[float]:
    """Deterministic lexical fallback used when a sentence embedding model is unavailable."""
    import math

    vector = [0.0] * dim
    tokens = re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(a, b):
        dot += float(left) * float(right)
        norm_a += float(left) * float(left)
        norm_b += float(right) * float(right)
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return float(dot / ((norm_a ** 0.5) * (norm_b ** 0.5)))


def build_augmented_observation_text(
    *,
    observation: str,
    skills: Optional[Sequence[str]] = None,
    step_hint: str = "",
) -> str:
    """Insert OPD teacher-only context while preserving the existing step-hint path."""

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

    sections = []
    if skills:
        sections.append(
            _format_guidance_section(
                "Retrieved Reusable Skills",
                (
                    "Use these sequence-level skills as reusable guidance for completing "
                    "tasks with similar intent"
                ),
                "\n".join(f"- {skill}" for skill in skills),
            )
        )

    if step_hint:
        sections.append(
            _format_guidance_section(
                "Current-Step Decision Guidance",
                (
                    "This is policy-facing advice for the current decision point. "
                    "Use it to guide your reasoning and select the next admissible action"
                ),
                step_hint,
            )
        )

    return _insert_before_anchor(
        str(observation).strip(),
        sections,
        (
            "Now it's your turn to",
            "Now it's your turn",
        ),
    )


@dataclass
class GuideSkillRecord:
    skill_id: str
    family_id: str
    task_text: str
    skill_text: str
    skill_type: str
    retrieval_text: str
    support_count: int
    status: str
    created_step: int
    last_updated_step: int
    title: str = ""
    task_pattern: str = ""
    applicability_text: str = ""
    success_count: int = 0
    failure_count: int = 0
    evidence_count: int = 0
    last_used_step: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class COPDGuideMemory:
    """
    Online sequence-level skill memory for COPD teacher prompts.

    ``episode_hint`` becomes a reusable skill candidate keyed by the trajectory
    task. ``step_hints`` remain direct current-step teacher guidance and are not
    stored in this memory.
    """

    _embedding_model_cache: Dict[str, Any] = {}

    def __init__(self, config, default_dump_dir: Optional[str] = None):
        self.config = config
        self.enabled = bool(config.get("enable", False))
        self.top_k = int(config.get("top_k", 2))
        self.max_per_skill_type = int(config.get("max_per_skill_type", 1))
        self.similarity_threshold = float(config.get("similarity_threshold", 0.3))
        self.dedupe_skill_similarity_thresh = float(config.get("dedupe_skill_similarity_thresh", 0.88))
        self.enable_batch_task_aggregation = bool(config.get("enable_batch_task_aggregation", True))
        self.embedding_model_path = str(
            config.get("embedding_model_path", DEFAULT_EMBEDDING_MODEL_PATH)
            or DEFAULT_EMBEDDING_MODEL_PATH
        )
        self.embedding_device = str(config.get("embedding_device", "") or "").strip()
        self.embedding_batch_size = max(int(config.get("embedding_batch_size", 64)), 1)
        self.hash_embedding_dim = int(config.get("hash_embedding_dim", 384))
        self.promote_min_support = int(config.get("promote_min_support", 2))
        self.merge_task_similarity_thresh = float(
            config.get(
                "merge_task_similarity_thresh",
                0.85,
            )
        )
        self.merge_skill_similarity_thresh = float(
            config.get(
                "merge_skill_similarity_thresh",
                0.9,
            )
        )
        self.max_skills = int(config.get("max_skills", 128))
        self.max_task_chars = int(config.get("max_task_chars", 256))
        self.max_skill_chars = int(config.get("max_skill_chars", 256))
        self.max_retrieval_chars = int(config.get("max_retrieval_chars", 768))
        self.max_evidence_examples = int(config.get("max_evidence_examples", 8))
        self.max_embedding_cache_entries = int(config.get("max_embedding_cache_entries", 4096))
        self.aggregate_with_llm = bool(config.get("aggregate_with_llm", False))
        self.aggregate_min_group_size = max(int(config.get("aggregate_min_group_size", 2)), 1)
        self.aggregate_max_candidates = max(int(config.get("aggregate_max_candidates", 8)), 1)
        self.dump_freq_steps = int(config.get("dump_freq_steps", 0))

        dump_dir = config.get("dump_dir", None)
        if dump_dir:
            self.dump_dir = str(dump_dir)
        elif default_dump_dir:
            self.dump_dir = os.path.join(default_dump_dir, "guide_memory")
        else:
            self.dump_dir = None

        self._skills: List[GuideSkillRecord] = []
        self._text_embedding_cache: Dict[str, List[float]] = {}
        self._embedding_model = None
        self._embedding_backend = "hashing" if self.embedding_model_path.lower() in _HASHING_EMBEDDING_ALIASES else "model"
        self._embedding_fallback_warned = False

    def _task_text_from_observation(self, observation: Any) -> str:
        return _normalize_text(extract_task_query(observation), lowercase=False, max_chars=self.max_task_chars)

    def _normalize_skill_text(self, skill_text: Any) -> str:
        return _normalize_text(skill_text, lowercase=False, max_chars=self.max_skill_chars)

    def _skill_family_id(self, task_pattern: str) -> str:
        return _slugify_text(task_pattern, max_chars=64)

    def _build_applicability_text(
        self,
        *,
        task_text: str,
        task_pattern: str,
        skill_type: str,
        explicit_applicability: Any = "",
    ) -> str:
        explicit = _normalize_text(explicit_applicability, lowercase=False, max_chars=240)
        if explicit:
            return explicit

        label = _SKILL_TYPE_LABELS.get(skill_type, _SKILL_TYPE_LABELS[UNKNOWN_SKILL_TYPE]).lower()
        if task_pattern == "constrained_product_search":
            return (
                f"Use this {label} for shopping tasks that require finding a product while "
                "satisfying hard constraints such as category, price, color, size, material, "
                "brand, ratings, or other product attributes."
            )
        if task_pattern.endswith("_object_workflow") or task_pattern in {
            "look_at_object_in_light",
            "object_location_and_examination",
        }:
            return (
                f"Use this {label} for embodied tasks that require locating objects, checking "
                "the current room state, and sequencing navigation or object interactions safely."
            )
        if task_pattern == "information_search":
            return (
                f"Use this {label} for search or question-answering tasks where the agent must "
                "gather evidence before responding."
            )
        return f"Use this {label} for tasks with a similar workflow or failure pattern."

    def _build_retrieval_text(
        self,
        *,
        family_id: str,
        task_pattern: str,
        skill_type: str,
        title: str,
        applicability_text: str,
        skill_text: str,
        task_text: str = "",
    ) -> str:
        parts = [
            family_id,
            task_pattern,
            _SKILL_TYPE_LABELS.get(skill_type, _SKILL_TYPE_LABELS[UNKNOWN_SKILL_TYPE]),
            title,
            applicability_text,
            skill_text,
        ]
        if task_text:
            parts.append(task_text)
        return _normalize_text(" ".join(parts), lowercase=False, max_chars=self.max_retrieval_chars)

    def _build_query_retrieval_text(self, task_text: str) -> str:
        task_pattern = _infer_task_pattern(task_text)
        family_id = self._skill_family_id(task_pattern)
        return _normalize_text(
            f"{family_id} {task_pattern} {task_text}",
            lowercase=False,
            max_chars=self.max_retrieval_chars,
        )

    def _build_skill_fields(
        self,
        *,
        task_text: str,
        skill_text: str,
        skill_type: str,
        title: Any = "",
        task_pattern: Any = "",
        applicability_text: Any = "",
    ) -> Dict[str, str]:
        normalized_task = _normalize_text(task_text, lowercase=False, max_chars=self.max_task_chars)
        normalized_skill = self._normalize_skill_text(skill_text)
        pattern = _normalize_text(task_pattern, lowercase=True, max_chars=96) or _infer_task_pattern(normalized_task)
        family_id = self._skill_family_id(pattern)
        normalized_title = _normalize_text(title, lowercase=False, max_chars=80) or _title_from_skill_text(
            normalized_skill,
            skill_type,
        )
        applicability = self._build_applicability_text(
            task_text=normalized_task,
            task_pattern=pattern,
            skill_type=skill_type,
            explicit_applicability=applicability_text,
        )
        retrieval_text = self._build_retrieval_text(
            family_id=family_id,
            task_pattern=pattern,
            skill_type=skill_type,
            title=normalized_title,
            applicability_text=applicability,
            skill_text=normalized_skill,
            task_text=normalized_task,
        )
        return {
            "family_id": family_id,
            "task_pattern": pattern,
            "title": normalized_title,
            "applicability_text": applicability,
            "retrieval_text": retrieval_text,
        }

    def _skill_type_from_success(self, episode_success: Optional[float]) -> str:
        if episode_success is None:
            return UNKNOWN_SKILL_TYPE
        try:
            return SUCCESS_WORKFLOW if float(episode_success) >= 1.0 else FAILURE_AVOIDANCE
        except (TypeError, ValueError):
            return UNKNOWN_SKILL_TYPE

    def _format_skill_for_prompt(self, record: GuideSkillRecord) -> str:
        label = _SKILL_TYPE_LABELS.get(record.skill_type, _SKILL_TYPE_LABELS[UNKNOWN_SKILL_TYPE])
        if record.applicability_text:
            return f"{label}: {record.skill_text} Applies when: {record.applicability_text}"
        return f"{label}: {record.skill_text}"

    def _get_embedding_model(self):
        if self._embedding_backend == "hashing":
            return None
        device_key = self.embedding_device.lower()
        cache_key = f"{self.embedding_model_path}::{device_key or 'auto'}"
        if cache_key in self._embedding_model_cache:
            return self._embedding_model_cache[cache_key]
        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {}
            if device_key not in _EMBEDDING_DEVICE_AUTO_ALIASES:
                model_kwargs["device"] = self.embedding_device
            model = SentenceTransformer(self.embedding_model_path, **model_kwargs)
            self._embedding_model_cache[cache_key] = model
            return model
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            if not self._embedding_fallback_warned:
                logger.warning(
                    "Failed to load guide memory embedding model %s on device %s; falling back to hashing embeddings: %s",
                    self.embedding_model_path,
                    self.embedding_device or "auto",
                    exc,
                )
                self._embedding_fallback_warned = True
            self._embedding_backend = "hashing"
            return None

    def _embedding_key(self, text: Any) -> str:
        return _normalize_text(text, lowercase=True, max_chars=self.max_retrieval_chars)

    def _remember_embedding(self, key: str, embedding: Sequence[float]) -> None:
        if not key or self.max_embedding_cache_entries <= 0:
            return
        if key in self._text_embedding_cache:
            return
        if len(self._text_embedding_cache) >= self.max_embedding_cache_entries:
            oldest_key = next(iter(self._text_embedding_cache))
            self._text_embedding_cache.pop(oldest_key, None)
        self._text_embedding_cache[key] = [float(value) for value in embedding]

    def _embed_tasks(self, task_texts: Sequence[Any]) -> List[List[float]]:
        normalized_tasks = [
            self._embedding_key(task_text)
            for task_text in task_texts
        ]
        if not normalized_tasks:
            return []
        model = self._get_embedding_model()
        if model is None:
            return [
                _hashing_embedding(normalized_task, self.hash_embedding_dim)
                for normalized_task in normalized_tasks
            ]

        embeddings = model.encode(
            normalized_tasks,
            batch_size=self.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [
            [float(value) for value in embedding.tolist()]
            for embedding in embeddings
        ]

    def _embed_tasks_deduped(self, task_texts: Sequence[Any]) -> List[List[float]]:
        ordered_keys = [self._embedding_key(task_text) for task_text in task_texts]
        empty_embedding = _hashing_embedding("", self.hash_embedding_dim)
        embeddings_by_key = {
            key: self._text_embedding_cache[key]
            for key in ordered_keys
            if key in self._text_embedding_cache
        }
        unique_keys = []
        seen_keys = set()
        for key in ordered_keys:
            if not key or key in seen_keys or key in embeddings_by_key:
                continue
            seen_keys.add(key)
            unique_keys.append(key)

        unique_embeddings = self._embed_tasks(unique_keys) if unique_keys else []
        for key, embedding in zip(unique_keys, unique_embeddings):
            embeddings_by_key[key] = embedding
            self._remember_embedding(key, embedding)
        return [
            embeddings_by_key.get(key, empty_embedding)
            for key in ordered_keys
        ]

    def _embed_task(self, task_text: str) -> List[float]:
        embeddings = self._embed_tasks_deduped([task_text])
        return embeddings[0] if embeddings else _hashing_embedding("", self.hash_embedding_dim)

    def _is_promoted(self, record: GuideSkillRecord) -> bool:
        return record.support_count >= self.promote_min_support

    def _refresh_status(self, record: GuideSkillRecord) -> None:
        record.status = "active" if self._is_promoted(record) else "pending"

    def _rank_record(self, record: GuideSkillRecord) -> Tuple[int, int]:
        return (
            record.support_count,
            record.last_updated_step,
        )

    def _match_skill_record(
        self,
        *,
        retrieval_embedding: Sequence[float],
        skill_text: str,
        skill_type: str,
        family_id: str,
        task_text: str,
    ) -> Optional[GuideSkillRecord]:
        record_embeddings = self._embed_tasks_deduped(
            [record.retrieval_text for record in self._skills]
        )
        task_key = _normalize_text(task_text, lowercase=True, max_chars=self.max_task_chars)
        for record, record_embedding in zip(self._skills, record_embeddings):
            if record.skill_type != skill_type:
                continue
            if (
                task_key
                and task_key == _normalize_text(record.task_text, lowercase=True, max_chars=self.max_task_chars)
                and record.family_id == family_id
            ):
                return record
            retrieval_similarity = _cosine_similarity(retrieval_embedding, record_embedding)
            skill_similarity = _text_similarity(record.skill_text, skill_text)
            if (
                retrieval_similarity >= self.merge_task_similarity_thresh
                and (
                    record.family_id == family_id
                    or skill_similarity >= self.merge_skill_similarity_thresh
                )
            ):
                return record
        return None

    def _prune_records(self) -> None:
        if self.max_skills <= 0:
            self._skills = []
            return
        self._skills.sort(
            key=lambda record: (
                1 if record.status == "active" else 0,
                record.support_count,
                record.last_updated_step,
            ),
            reverse=True,
        )
        self._skills = self._skills[: self.max_skills]

    def _add_skill(
        self,
        *,
        task_text: str,
        skill_text: str,
        skill_type: str,
        family_id: str,
        title: str,
        task_pattern: str,
        applicability_text: str,
        retrieval_text: str,
        global_step: int,
        source: str,
        support_increment: int = 1,
        retrieval_embedding: Optional[Sequence[float]] = None,
        evidence_examples: Optional[Sequence[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, bool]:
        normalized_task = _normalize_text(task_text, lowercase=False, max_chars=self.max_task_chars)
        normalized_skill = self._normalize_skill_text(skill_text)
        if not normalized_task or not normalized_skill:
            return False, False

        if retrieval_embedding is None:
            retrieval_embedding = self._embed_task(retrieval_text)
        else:
            retrieval_embedding = [float(value) for value in retrieval_embedding]
        record = self._match_skill_record(
            retrieval_embedding=retrieval_embedding,
            skill_text=normalized_skill,
            skill_type=skill_type,
            family_id=family_id,
            task_text=normalized_task,
        )
        merged = record is not None
        if record is None:
            record = GuideSkillRecord(
                skill_id=f"skill_{uuid.uuid4().hex[:12]}",
                family_id=family_id,
                task_text=normalized_task,
                skill_text=normalized_skill,
                skill_type=skill_type,
                retrieval_text=retrieval_text,
                support_count=0,
                status="pending",
                created_step=global_step,
                last_updated_step=global_step,
                title=title,
                task_pattern=task_pattern,
                applicability_text=applicability_text,
                metadata={},
            )
            self._skills.append(record)
        else:
            if len(normalized_skill) > len(record.skill_text):
                record.skill_text = normalized_skill
                record.title = title or record.title
                record.task_text = normalized_task
            if applicability_text and len(applicability_text) > len(record.applicability_text):
                record.applicability_text = applicability_text
            record.retrieval_text = self._build_retrieval_text(
                family_id=record.family_id,
                task_pattern=record.task_pattern or task_pattern,
                skill_type=record.skill_type,
                title=record.title or title,
                applicability_text=record.applicability_text or applicability_text,
                skill_text=record.skill_text,
                task_text=record.task_text,
            )

        record.support_count += max(int(support_increment), 1)
        if skill_type == SUCCESS_WORKFLOW:
            record.success_count += max(int(support_increment), 1)
        elif skill_type == FAILURE_AVOIDANCE:
            record.failure_count += max(int(support_increment), 1)
        record.last_updated_step = global_step
        incoming_evidence = list(evidence_examples or [])
        if incoming_evidence:
            record.evidence_examples.extend(incoming_evidence)
            record.evidence_examples = record.evidence_examples[-self.max_evidence_examples:]
            record.evidence_count += len(incoming_evidence)
        record.metadata.update(
            {
                "source": source,
                "last_task_hash": _stable_hash([normalized_task])[:12],
                "skill_type": skill_type,
                "family_id": record.family_id,
                "task_pattern": record.task_pattern,
            }
        )
        if metadata:
            record.metadata.update(metadata)
        self._refresh_status(record)
        self._prune_records()
        return True, merged

    def _cluster_batch_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        skill_aggregator: Optional[Callable[..., Dict[str, Any]]] = None,
        global_step: Optional[int] = None,
        analysis_mode: str = "teacher_bootstrap",
    ) -> List[Dict[str, Any]]:
        if not self.enable_batch_task_aggregation:
            return [
                {
                    "task_text": candidate["task_text"],
                    "skill_text": candidate["skill_text"],
                    "skill_type": candidate["skill_type"],
                    "title": candidate.get("title", ""),
                    "task_pattern": candidate.get("task_pattern", ""),
                    "applicability_text": candidate.get("applicability_text", ""),
                    "support_increment": 1,
                    "traj_uids": [str(candidate["traj_uid"])],
                    "episode_success_values": (
                        [candidate["episode_success"]]
                        if candidate["episode_success"] is not None
                        else []
                    ),
                    "step_hint_count": int(candidate["step_hint_count"]),
                    "candidate_hints": [candidate["skill_text"]],
                }
                for candidate in candidates
            ]

        grouped_candidates: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            task_key = _normalize_text(candidate["task_text"], lowercase=True, max_chars=self.max_task_chars)
            grouped_candidates[(task_key, candidate["skill_type"])].append(candidate)

        clustered: List[Dict[str, Any]] = []
        for (_, skill_type), group in grouped_candidates.items():
            representative = max(
                group,
                key=lambda item: (len(item["skill_text"]), item["traj_uid"]),
            )
            candidate_hints = [
                item["skill_text"]
                for item in group[: min(len(group), self.aggregate_max_candidates)]
            ]
            aggregated_fields: Dict[str, Any] = {}
            aggregation_used = False
            aggregation_error = ""
            if (
                self.aggregate_with_llm
                and skill_aggregator is not None
                and len(group) >= self.aggregate_min_group_size
                and candidate_hints
            ):
                try:
                    aggregated_fields = skill_aggregator(
                        task_text=representative["task_text"],
                        skill_type=skill_type,
                        candidate_hints=candidate_hints,
                        episode_success_values=[
                            item["episode_success"]
                            for item in group
                            if item["episode_success"] is not None
                        ],
                        global_step=global_step,
                        analysis_mode=analysis_mode,
                    ) or {}
                    aggregation_used = bool(
                        str(aggregated_fields.get("skill_text") or "").strip()
                    )
                    if not aggregation_used:
                        aggregated_fields = {}
                except Exception as exc:  # pragma: no cover - runtime backend dependent
                    aggregation_error = str(exc)
                    logger.warning(
                        "Guide memory LLM skill aggregation failed for task=%s type=%s: %s",
                        representative["task_text"],
                        skill_type,
                        exc,
                    )

            aggregated_skill_text = self._normalize_skill_text(
                aggregated_fields.get("skill_text")
                or representative["skill_text"]
            )
            aggregated_title = aggregated_fields.get("title") or representative.get("title", "")
            aggregated_task_pattern = aggregated_fields.get("task_pattern") or representative.get("task_pattern", "")
            aggregated_applicability = (
                aggregated_fields.get("applicability_text")
                or aggregated_fields.get("when_to_apply")
                or representative.get("applicability_text", "")
            )
            clustered.append(
                {
                    "task_text": representative["task_text"],
                    "skill_text": aggregated_skill_text,
                    "skill_type": skill_type,
                    "title": aggregated_title,
                    "task_pattern": aggregated_task_pattern,
                    "applicability_text": aggregated_applicability,
                    "support_increment": len(group),
                    "traj_uids": [str(item["traj_uid"]) for item in group],
                    "episode_success_values": [
                        item["episode_success"]
                        for item in group
                        if item["episode_success"] is not None
                    ],
                    "step_hint_count": sum(int(item["step_hint_count"]) for item in group),
                    "candidate_hints": candidate_hints,
                    "aggregation_used": aggregation_used,
                    "aggregation_error": aggregation_error,
                    "aggregation_raw": aggregated_fields.get("raw_output", ""),
                }
            )

        return clustered

    def _retrieve_guides_from_embedding(
        self,
        *,
        task_text: str,
        query_embedding: Sequence[float],
        global_step: Optional[int] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        result = {
            "task_query": task_text,
            "skills": [],
            "skill_records": [],
        }
        if not self.enabled or not task_text:
            return result

        threshold = self.similarity_threshold if similarity_threshold is None else float(similarity_threshold)
        scored_records = []
        active_records = [record for record in self._skills if record.status == "active"]
        active_embeddings = self._embed_tasks_deduped(
            [record.retrieval_text for record in active_records]
        )
        for record, record_embedding in zip(active_records, active_embeddings):
            if record.status != "active":
                continue
            similarity = _cosine_similarity(query_embedding, record_embedding)
            if similarity < threshold:
                continue
            scored_records.append((similarity, record))

        scored_records.sort(
            key=lambda item: (
                item[0],
                item[1].support_count,
                item[1].last_updated_step,
            ),
            reverse=True,
        )
        selected = self._select_diverse_records(
            scored_records=scored_records,
            top_k=max(int(self.top_k if top_k is None else top_k), 0),
        )
        skills = []
        skill_records = []
        for similarity, record in selected:
            if global_step is not None:
                record.last_used_step = global_step
            skills.append(self._format_skill_for_prompt(record))
            skill_records.append(
                {
                    "skill_id": record.skill_id,
                    "family_id": record.family_id,
                    "task_text": record.task_text,
                    "skill_text": record.skill_text,
                    "skill_type": record.skill_type,
                    "title": record.title,
                    "task_pattern": record.task_pattern,
                    "applicability_text": record.applicability_text,
                    "similarity": float(similarity),
                    "support_count": int(record.support_count),
                }
            )

        result["skills"] = skills
        result["skill_records"] = skill_records
        return result

    def retrieve_guides(
        self,
        *,
        task_query: Any,
        global_step: Optional[int] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        task_text = _normalize_text(task_query, lowercase=False, max_chars=self.max_task_chars)
        if not self.enabled or not task_text:
            return {
                "task_query": task_text,
                "skills": [],
                "skill_records": [],
            }
        query_retrieval_text = self._build_query_retrieval_text(task_text)
        query_embedding = self._embed_task(query_retrieval_text)
        return self._retrieve_guides_from_embedding(
            task_text=task_text,
            query_embedding=query_embedding,
            global_step=global_step,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def retrieve_guides_batch(
        self,
        *,
        task_queries: Sequence[Any],
        global_step: Optional[int] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        task_texts = [
            _normalize_text(task_query, lowercase=False, max_chars=self.max_task_chars)
            for task_query in task_queries
        ]
        results = [
            {
                "task_query": task_text,
                "skills": [],
                "skill_records": [],
            }
            for task_text in task_texts
        ]
        if not self.enabled:
            return results

        valid_positions = [
            idx
            for idx, task_text in enumerate(task_texts)
            if task_text
        ]
        if not valid_positions:
            return results

        valid_query_texts = [
            self._build_query_retrieval_text(task_texts[idx])
            for idx in valid_positions
        ]
        embeddings = self._embed_tasks_deduped(valid_query_texts)
        for idx, embedding in zip(valid_positions, embeddings):
            results[idx] = self._retrieve_guides_from_embedding(
                task_text=task_texts[idx],
                query_embedding=embedding,
                global_step=global_step,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        return results

    def _select_diverse_records(
        self,
        *,
        scored_records: Sequence[Tuple[float, GuideSkillRecord]],
        top_k: int,
    ) -> List[Tuple[float, GuideSkillRecord]]:
        if top_k <= 0:
            return []

        selected: List[Tuple[float, GuideSkillRecord]] = []
        per_type_counts: Dict[str, int] = defaultdict(int)
        max_per_type = max(int(self.max_per_skill_type), 1)

        for similarity, record in scored_records:
            if len(selected) >= top_k:
                break
            if per_type_counts[record.skill_type] >= max_per_type:
                continue
            if any(
                _text_similarity(record.skill_text, selected_record.skill_text) >= self.dedupe_skill_similarity_thresh
                for _, selected_record in selected
            ):
                continue
            selected.append((similarity, record))
            per_type_counts[record.skill_type] += 1

        return selected

    def retrieve_for_observation(
        self,
        *,
        observation: Any,
        anchor_observation: Any = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        del anchor_observation
        return self.retrieve_guides(
            task_query=self._task_text_from_observation(observation),
            global_step=global_step,
        )

    def retrieve_for_observations(
        self,
        *,
        observations: Sequence[Any],
        anchor_observations: Optional[Sequence[Any]] = None,
        global_step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        del anchor_observations
        return self.retrieve_guides_batch(
            task_queries=[
                self._task_text_from_observation(observation)
                for observation in observations
            ],
            global_step=global_step,
        )

    def build_augmented_observation(
        self,
        *,
        observation: str,
        skills: Optional[Sequence[str]] = None,
        hindsight_hint: str = "",
        step_hint: str = "",
    ) -> str:
        return build_augmented_observation_text(
            observation=observation,
            skills=skills or [],
            step_hint=step_hint or hindsight_hint,
        )

    def format_for_rollout_prompt(self, *, skills: Sequence[str], **_: Any) -> str:
        if not skills:
            return ""
        return (
            "## Retrieved Reusable Skills\n"
            "Use these sequence-level skills as reusable guidance for completing tasks with similar intent.\n"
            + "\n".join(f"- {skill}" for skill in skills)
        )

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
        skill_aggregator: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        del anchor_obs, critical_mask
        metrics = {
            "copd/guide_memory/enabled": 1.0 if self.enabled else 0.0,
            "copd/guide_memory/skill_candidates_added": 0.0,
            "copd/guide_memory/skill_candidates_merged": 0.0,
            "copd/guide_memory/batch_candidate_count": 0.0,
            "copd/guide_memory/batch_cluster_count": 0.0,
            "copd/guide_memory/batch_llm_aggregation_count": 0.0,
            "copd/guide_memory/batch_llm_aggregation_error_count": 0.0,
        }
        if not self.enabled:
            return metrics

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

        candidates: List[Dict[str, Any]] = []
        for traj_uid, sample_indices in traj_to_indices.items():
            if not sample_indices:
                continue
            representative_idx = min(sample_indices, key=lambda idx: step_indices_list[idx])
            analysis = episode_analysis.get(traj_uid, {})
            task_text = _normalize_text(
                analysis.get("task_description") or self._task_text_from_observation(obs_texts[representative_idx]),
                lowercase=False,
                max_chars=self.max_task_chars,
            )
            skill_text = str(
                analysis.get("episode_hint")
                or analysis.get("overall_hint")
                or ""
            ).strip()
            if not skill_text:
                continue

            episode_success_value = traj_success.get(traj_uid)
            candidates.append(
                {
                    "traj_uid": str(traj_uid),
                    "task_text": task_text,
                    "skill_text": self._normalize_skill_text(skill_text),
                    "skill_type": self._skill_type_from_success(episode_success_value),
                    "title": analysis.get("skill_title") or analysis.get("title") or "",
                    "task_pattern": analysis.get("task_pattern") or "",
                    "applicability_text": analysis.get("applicability_text") or analysis.get("when_to_apply") or "",
                    "episode_success": episode_success_value,
                    "episode_summary": str(analysis.get("episode_summary", "")),
                    "step_hint_count": len(analysis.get("step_hints", {}) or {}),
                }
            )

        clustered_candidates = self._cluster_batch_candidates(
            candidates,
            skill_aggregator=skill_aggregator,
            global_step=global_step,
            analysis_mode=analysis_mode,
        )
        metrics["copd/guide_memory/batch_candidate_count"] = float(len(candidates))
        metrics["copd/guide_memory/batch_cluster_count"] = float(len(clustered_candidates))
        metrics["copd/guide_memory/batch_llm_aggregation_count"] = float(
            sum(1 for candidate in clustered_candidates if candidate.get("aggregation_used"))
        )
        metrics["copd/guide_memory/batch_llm_aggregation_error_count"] = float(
            sum(1 for candidate in clustered_candidates if candidate.get("aggregation_error"))
        )

        skill_fields = [
            self._build_skill_fields(
                task_text=candidate["task_text"],
                skill_text=candidate["skill_text"],
                skill_type=candidate["skill_type"],
                title=candidate.get("title", ""),
                task_pattern=candidate.get("task_pattern", ""),
                applicability_text=candidate.get("applicability_text", ""),
            )
            for candidate in clustered_candidates
        ]
        candidate_embeddings = self._embed_tasks_deduped(
            [fields["retrieval_text"] for fields in skill_fields]
        )
        for candidate, fields, retrieval_embedding in zip(clustered_candidates, skill_fields, candidate_embeddings):
            evidence_examples = [
                {
                    "task_text": candidate["task_text"],
                    "traj_uids": candidate.get("traj_uids", []),
                    "episode_success_values": candidate.get("episode_success_values", []),
                    "skill_text": candidate["skill_text"],
                    "global_step": int(global_step),
                }
            ]
            added, merged = self._add_skill(
                task_text=candidate["task_text"],
                skill_text=candidate["skill_text"],
                skill_type=candidate["skill_type"],
                family_id=fields["family_id"],
                title=fields["title"],
                task_pattern=fields["task_pattern"],
                applicability_text=fields["applicability_text"],
                retrieval_text=fields["retrieval_text"],
                global_step=global_step,
                source=f"{analysis_mode}/episode_hint",
                support_increment=int(candidate.get("support_increment", 1)),
                retrieval_embedding=retrieval_embedding,
                evidence_examples=evidence_examples,
                metadata={
                    "traj_uids": candidate.get("traj_uids", []),
                    "analysis_mode": analysis_mode,
                    "episode_success_values": candidate.get("episode_success_values", []),
                    "step_hint_count": int(candidate.get("step_hint_count", 0)),
                    "batch_support_increment": int(candidate.get("support_increment", 1)),
                    "candidate_hints": candidate.get("candidate_hints", []),
                    "aggregation_used": bool(candidate.get("aggregation_used", False)),
                    "aggregation_error": str(candidate.get("aggregation_error", "")),
                    "aggregation_raw": str(candidate.get("aggregation_raw", "")),
                },
            )
            metrics["copd/guide_memory/skill_candidates_added"] += float(added)
            metrics["copd/guide_memory/skill_candidates_merged"] += float(merged)

        metrics.update(self.snapshot_metrics(prefix="copd/guide_memory"))
        self._dump_if_needed(global_step=global_step)
        return metrics

    def snapshot_metrics(self, prefix: str = "guide_memory") -> Dict[str, float]:
        def _count_status(status: str) -> float:
            return float(sum(1 for record in self._skills if record.status == status))

        def _count_type(skill_type: str) -> float:
            return float(sum(1 for record in self._skills if record.skill_type == skill_type))

        support_values = [record.support_count for record in self._skills]
        mean_support = float(sum(support_values) / len(support_values)) if support_values else 0.0
        family_count = len({record.family_id for record in self._skills if record.family_id})
        evidence_total = sum(int(record.evidence_count) for record in self._skills)
        return {
            f"{prefix}/skill_total": float(len(self._skills)),
            f"{prefix}/skill_active": _count_status("active"),
            f"{prefix}/skill_pending": _count_status("pending"),
            f"{prefix}/skill_family_total": float(family_count),
            f"{prefix}/skill_success_workflow": _count_type(SUCCESS_WORKFLOW),
            f"{prefix}/skill_failure_avoidance": _count_type(FAILURE_AVOIDANCE),
            f"{prefix}/skill_unknown": _count_type(UNKNOWN_SKILL_TYPE),
            f"{prefix}/skill_mean_support": mean_support,
            f"{prefix}/skill_evidence_total": float(evidence_total),
            f"{prefix}/embedding_cache_entries": float(len(self._text_embedding_cache)),
            f"{prefix}/aggregate_with_llm": 1.0 if self.aggregate_with_llm else 0.0,
        }

    def dump_snapshot(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "skills": [record.to_json() for record in self._skills],
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
