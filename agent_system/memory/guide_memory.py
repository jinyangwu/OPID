import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
DEFAULT_EMBEDDING_MODEL_NAME = "Qwen3-Embedding-0.6B"
SUCCESS_WORKFLOW = "success_workflow"
FAILURE_AVOIDANCE = "failure_avoidance"
UNKNOWN_SKILL_TYPE = "unknown"
_SKILL_TYPE_LABELS = {
    SUCCESS_WORKFLOW: "Success workflow",
    FAILURE_AVOIDANCE: "Failure avoidance",
    UNKNOWN_SKILL_TYPE: "Reusable skill",
}
_TORCH_MODULE = None
_TORCH_IMPORT_FAILED = False


def _get_torch_module():
    global _TORCH_IMPORT_FAILED, _TORCH_MODULE
    if _TORCH_IMPORT_FAILED:
        return None
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    try:
        import torch

        _TORCH_MODULE = torch
        return _TORCH_MODULE
    except Exception:
        _TORCH_IMPORT_FAILED = True
        return None


def _embedding_tensor_device(torch, preferred_device: Any = None, model: Any = None):
    def usable(device):
        if device.type != "cuda":
            return True
        return torch.cuda.is_available() and (
            device.index is None or device.index < torch.cuda.device_count()
        )

    if preferred_device is not None:
        preferred_device_text = str(preferred_device).strip()
        if preferred_device_text and preferred_device_text.lower() not in _EMBEDDING_DEVICE_AUTO_ALIASES:
            device = torch.device(preferred_device_text)
            if usable(device):
                return device

    model_device = getattr(model, "device", None)
    if model_device is not None:
        device = torch.device(model_device)
        if usable(device):
            return device

    try:
        parameters = model.parameters() if model is not None else []
        for parameter in parameters:
            device = parameter.device
            if usable(device):
                return device
    except Exception:
        pass

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


def _coerce_embedding(embedding: Any, device: Any = None):
    torch = _get_torch_module()
    if torch is None:
        return [float(value) for value in embedding]
    device = _embedding_tensor_device(torch, preferred_device=device)
    if torch.is_tensor(embedding):
        return embedding.detach().to(dtype=torch.float32, device=device).flatten().clone()
    return torch.as_tensor(embedding, dtype=torch.float32, device=device).flatten().clone()


def _empty_embedding(dim: int, device: Any = None):
    torch = _get_torch_module()
    if torch is None:
        return _hashing_embedding("", dim)
    return torch.zeros(dim, dtype=torch.float32, device=_embedding_tensor_device(torch, preferred_device=device))


def _embedding_matrix(embeddings: Sequence[Any], device: Any = None):
    torch = _get_torch_module()
    if torch is None or len(embeddings) <= 0:
        return None

    target_device = device
    if target_device is None:
        for embedding in embeddings:
            if torch.is_tensor(embedding):
                target_device = embedding.device
                break
    if target_device is None:
        target_device = _embedding_tensor_device(torch)

    vectors = []
    width = None
    for embedding in embeddings:
        if torch.is_tensor(embedding):
            vector = embedding.detach().to(dtype=torch.float32, device=target_device).flatten()
        else:
            vector = torch.as_tensor(embedding, dtype=torch.float32, device=target_device).flatten()
        if vector.numel() <= 0:
            return None
        if width is None:
            width = vector.numel()
        elif vector.numel() != width:
            return None
        vectors.append(vector)
    return torch.stack(vectors, dim=0)


def _cosine_similarity(a: Any, b: Any) -> float:
    torch = _get_torch_module()
    if torch is not None and (torch.is_tensor(a) or torch.is_tensor(b)):
        if torch.is_tensor(a):
            device = a.device
        elif torch.is_tensor(b):
            device = b.device
        else:
            device = _embedding_tensor_device(torch)
        left = a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32, device=device)
        right = b if torch.is_tensor(b) else torch.as_tensor(b, dtype=torch.float32, device=device)
        left = left.detach().to(dtype=torch.float32, device=device).flatten()
        right = right.detach().to(dtype=torch.float32, device=device).flatten()
        if left.numel() <= 0 or right.numel() <= 0 or left.numel() != right.numel():
            return 0.0
        norm = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if float(norm) <= 0.0:
            return 0.0
        return float(torch.dot(left, right) / norm)

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


def _cosine_similarity_matrix(queries: Sequence[Any], candidates: Sequence[Any]) -> List[List[float]]:
    if len(queries) <= 0:
        return []
    if len(candidates) <= 0:
        return [[] for _ in queries]

    torch = _get_torch_module()
    if torch is not None:
        query_matrix = _embedding_matrix(queries)
        candidate_matrix = _embedding_matrix(
            candidates,
            device=query_matrix.device if query_matrix is not None else None,
        )
        if (
            query_matrix is not None
            and candidate_matrix is not None
            and query_matrix.shape[1] == candidate_matrix.shape[1]
        ):
            query_norms = torch.linalg.vector_norm(query_matrix, dim=1, keepdim=True).clamp_min(1e-12)
            candidate_norms = torch.linalg.vector_norm(candidate_matrix, dim=1, keepdim=True).clamp_min(1e-12)
            similarities = (query_matrix / query_norms) @ (candidate_matrix / candidate_norms).T
            return similarities.detach().cpu().tolist()

    return [
        [
            _cosine_similarity(query_embedding, candidate_embedding)
            for candidate_embedding in candidates
        ]
        for query_embedding in queries
    ]


def _cosine_similarities(query: Any, candidates: Sequence[Any]) -> List[float]:
    rows = _cosine_similarity_matrix([query], candidates)
    return rows[0] if rows else []


def build_augmented_observation_text(
    *,
    observation: str,
    skills: Optional[Sequence[str]] = None,
    episode_hint: str = "",
) -> str:
    """Insert episode-level OPD teacher context into an observation prompt."""

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

    if episode_hint:
        sections.append(
            _format_guidance_section(
                "Episode-Level Guidance",
                (
                    "Use this sequence-level lesson as reusable guidance for every "
                    "decision in the current episode"
                ),
                episode_hint,
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
    task_text: str
    skill_text: str
    skill_type: str
    retrieval_text: str
    support_count: int
    status: str
    created_step: int
    last_updated_step: int

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class COPDGuideMemory:
    """
    Online sequence-level skill memory for COPD teacher prompts.

    ``episode_hint`` becomes a reusable skill candidate keyed by the trajectory
    task.
    """

    _embedding_model_cache: Dict[str, Any] = {}

    def __init__(self, config, default_dump_dir: Optional[str] = None):
        self.config = config
        self.enabled = bool(config.get("enable", False))
        self.top_k = int(config.get("top_k", 2))
        self.max_per_skill_type = int(config.get("max_per_skill_type", 1))
        self.dedupe_skill_similarity_thresh = float(config.get("dedupe_skill_similarity_thresh", 0.88))
        self.enable_batch_task_aggregation = bool(config.get("enable_batch_task_aggregation", True))
        embedding_model_path = config.get("embedding_model_path")
        if not embedding_model_path:
            embedding_model_path = os.path.join(os.environ["MODELS_ROOT"], DEFAULT_EMBEDDING_MODEL_NAME)
        self.embedding_model_path = str(embedding_model_path)
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
        self.max_skills = int(config.get("max_skills", 128))
        self.max_embedding_cache_entries = int(config.get("max_embedding_cache_entries", 4096))
        self.batch_cluster_similarity_thresh = float(
            config.get(
                "batch_cluster_similarity_thresh",
                config.get("merge_task_similarity_thresh", 0.85),
            )
        )
        self.dump_freq_steps = int(config.get("dump_freq_steps", 0))

        dump_dir = config.get("dump_dir", None)
        if dump_dir:
            self.dump_dir = str(dump_dir)
        elif default_dump_dir:
            self.dump_dir = os.path.join(default_dump_dir, "guide_memory")
        else:
            self.dump_dir = None

        self._skills: List[GuideSkillRecord] = []
        self._text_embedding_cache: Dict[str, Any] = {}
        self._embedding_model = None
        self._embedding_backend = "hashing" if self.embedding_model_path.lower() in _HASHING_EMBEDDING_ALIASES else "model"
        self._embedding_fallback_warned = False
        self._merge_time_last_sec = 0.0
        self._merge_time_total_sec = 0.0
        self._merge_time_count = 0
        self._retrieval_time_last_sec = 0.0
        self._retrieval_time_total_sec = 0.0
        self._retrieval_time_count = 0

    def _record_merge_time(self, duration_sec: float) -> None:
        duration = max(float(duration_sec), 0.0)
        self._merge_time_last_sec = duration
        self._merge_time_total_sec += duration
        self._merge_time_count += 1

    def _record_retrieval_time(self, duration_sec: float) -> None:
        duration = max(float(duration_sec), 0.0)
        self._retrieval_time_last_sec = duration
        self._retrieval_time_total_sec += duration
        self._retrieval_time_count += 1

    def _task_text_from_observation(self, observation: Any) -> str:
        return _normalize_text(extract_task_query(observation), lowercase=False)

    def _normalize_skill_text(self, skill_text: Any) -> str:
        return _normalize_text(skill_text, lowercase=False)

    def _build_retrieval_text(
        self,
        *,
        skill_text: str,
        task_text: str,
    ) -> str:
        return _normalize_text(
            f"TASK: {task_text}\nSKILL: {skill_text}",
            lowercase=False,
        )

    def _build_query_retrieval_text(self, task_text: str) -> str:
        return _normalize_text(task_text, lowercase=False)

    def _record_retrieval_text(self, record: GuideSkillRecord) -> str:
        return _normalize_text(record.retrieval_text, lowercase=False)

    def _skill_type_from_success(self, episode_success: Optional[float]) -> str:
        if episode_success is None:
            return UNKNOWN_SKILL_TYPE
        try:
            return SUCCESS_WORKFLOW if float(episode_success) >= 1.0 else FAILURE_AVOIDANCE
        except (TypeError, ValueError):
            return UNKNOWN_SKILL_TYPE

    def _format_skill_for_prompt(self, record: GuideSkillRecord) -> str:
        label = _SKILL_TYPE_LABELS.get(record.skill_type, _SKILL_TYPE_LABELS[UNKNOWN_SKILL_TYPE])
        return f"{label}: {record.skill_text}"

    def _get_embedding_model(self):
        if self._embedding_backend == "hashing":
            return None
        device_key = self.embedding_device.lower()
        cache_key = f"{self.embedding_model_path}::{device_key or 'auto'}"
        if cache_key in self._embedding_model_cache:
            self._embedding_model = self._embedding_model_cache[cache_key]
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {}
            if device_key not in _EMBEDDING_DEVICE_AUTO_ALIASES:
                model_kwargs["device"] = self.embedding_device
            model = SentenceTransformer(self.embedding_model_path, **model_kwargs)
            self._embedding_model_cache[cache_key] = model
            self._embedding_model = model
            return self._embedding_model
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

    def _embedding_device(self, model: Any = None):
        torch = _get_torch_module()
        if torch is None:
            return None
        return _embedding_tensor_device(
            torch,
            preferred_device=self.embedding_device,
            model=model or self._embedding_model,
        )

    def _embedding_key(self, text: Any) -> str:
        return _normalize_text(text, lowercase=True)

    def _remember_embedding(self, key: str, embedding: Any) -> None:
        if not key or self.max_embedding_cache_entries <= 0:
            return
        if key in self._text_embedding_cache:
            return
        if len(self._text_embedding_cache) >= self.max_embedding_cache_entries:
            oldest_key = next(iter(self._text_embedding_cache))
            self._text_embedding_cache.pop(oldest_key, None)
        self._text_embedding_cache[key] = _coerce_embedding(embedding, device=self._embedding_device())

    def _embed_tasks(self, task_texts: Sequence[Any]) -> List[Any]:
        normalized_tasks = [
            self._embedding_key(task_text)
            for task_text in task_texts
        ]
        if not normalized_tasks:
            return []
        model = self._get_embedding_model()
        embedding_device = self._embedding_device(model)
        if model is None:
            return [
                _coerce_embedding(
                    _hashing_embedding(normalized_task, self.hash_embedding_dim),
                    device=embedding_device,
                )
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
            _coerce_embedding(embedding, device=embedding_device)
            for embedding in embeddings
        ]

    def _embed_tasks_deduped(self, task_texts: Sequence[Any]) -> List[Any]:
        ordered_keys = [self._embedding_key(task_text) for task_text in task_texts]
        empty_embedding = _empty_embedding(self.hash_embedding_dim, device=self._embedding_device())
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

    def _embed_task(self, task_text: str):
        embeddings = self._embed_tasks_deduped([task_text])
        return embeddings[0] if embeddings else _empty_embedding(self.hash_embedding_dim, device=self._embedding_device())

    def _is_promoted(self, record: GuideSkillRecord) -> bool:
        return record.support_count >= self.promote_min_support

    def _refresh_status(self, record: GuideSkillRecord) -> None:
        record.status = "active" if self._is_promoted(record) else "pending"

    def _match_skill_record(
        self,
        *,
        retrieval_embedding: Any,
        skill_type: str,
        task_text: str,
    ) -> Optional[GuideSkillRecord]:
        del task_text
        return self._matching_skill_records(
            retrieval_embeddings=[retrieval_embedding],
            skill_types=[skill_type],
        )[0]

    def _matching_skill_records(
        self,
        *,
        retrieval_embeddings: Sequence[Any],
        skill_types: Sequence[str],
        records: Optional[Sequence[GuideSkillRecord]] = None,
        record_embeddings: Optional[Sequence[Any]] = None,
    ) -> List[Optional[GuideSkillRecord]]:
        if len(retrieval_embeddings) <= 0:
            return []

        candidate_records = list(self._skills if records is None else records)
        if not candidate_records:
            return [None for _ in retrieval_embeddings]

        embeddings = list(record_embeddings) if record_embeddings is not None else self._embed_tasks_deduped(
            [self._record_retrieval_text(record) for record in candidate_records]
        )
        similarity_rows = _cosine_similarity_matrix(retrieval_embeddings, embeddings)
        matches: List[Optional[GuideSkillRecord]] = []
        for skill_type, similarities in zip(skill_types, similarity_rows):
            matched_record = None
            for record, similarity in zip(candidate_records, similarities):
                if record.skill_type != skill_type:
                    continue
                if float(similarity) >= self.merge_task_similarity_thresh:
                    matched_record = record
                    break
            matches.append(matched_record)

        while len(matches) < len(retrieval_embeddings):
            matches.append(None)
        return matches

    def _merge_candidate_records(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        retrieval_embeddings: Sequence[Any],
    ) -> List[Optional[GuideSkillRecord]]:
        skill_records = list(self._skills)
        skill_embeddings = self._embed_tasks_deduped(
            [self._record_retrieval_text(record) for record in skill_records]
        )
        return self._matching_skill_records(
            retrieval_embeddings=retrieval_embeddings,
            skill_types=[str(candidate["skill_type"]) for candidate in candidates],
            records=skill_records,
            record_embeddings=skill_embeddings,
        )

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
        global_step: int,
        support_increment: int = 1,
        retrieval_embedding: Optional[Any] = None,
        matched_record: Optional[GuideSkillRecord] = None,
        match_precomputed: bool = False,
    ) -> Tuple[bool, bool]:
        normalized_task = _normalize_text(task_text, lowercase=False)
        normalized_skill = self._normalize_skill_text(skill_text)
        if not normalized_task or not normalized_skill:
            return False, False

        retrieval_text = self._build_retrieval_text(
            task_text=normalized_task,
            skill_text=normalized_skill,
        )
        if retrieval_embedding is None:
            retrieval_embedding = self._embed_task(retrieval_text)
        else:
            retrieval_embedding = _coerce_embedding(retrieval_embedding, device=self._embedding_device())
        if match_precomputed:
            record = matched_record if matched_record in self._skills else None
            if matched_record is not None and record is None:
                record = self._match_skill_record(
                    retrieval_embedding=retrieval_embedding,
                    skill_type=skill_type,
                    task_text=normalized_task,
                )
        else:
            record = self._match_skill_record(
                retrieval_embedding=retrieval_embedding,
                skill_type=skill_type,
                task_text=normalized_task,
            )
        merged = record is not None
        if record is None:
            record = GuideSkillRecord(
                skill_id=f"skill_{uuid.uuid4().hex[:12]}",
                task_text=normalized_task,
                skill_text=normalized_skill,
                skill_type=skill_type,
                retrieval_text=retrieval_text,
                support_count=0,
                status="pending",
                created_step=global_step,
                last_updated_step=global_step,
            )
            self._skills.append(record)
        else:
            if len(normalized_skill) > len(record.skill_text):
                record.skill_text = normalized_skill
                record.task_text = normalized_task
                record.retrieval_text = self._build_retrieval_text(
                    task_text=record.task_text,
                    skill_text=record.skill_text,
                )

        record.support_count += max(int(support_increment), 1)
        record.last_updated_step = global_step
        self._refresh_status(record)
        self._prune_records()
        return True, merged

    def _build_candidate_cluster_text(self, candidate: Dict[str, Any]) -> str:
        return _normalize_text(
            f"TASK: {candidate.get('task_text', '')}\nSKILL: {candidate.get('skill_text', '')}",
            lowercase=False,
        )

    def _build_clustered_candidate(
        self,
        group: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        representative = max(
            group,
            key=lambda item: (len(item["skill_text"]), item["traj_uid"]),
        )
        return {
            "task_text": representative["task_text"],
            "skill_text": self._normalize_skill_text(representative["skill_text"]),
            "skill_type": representative["skill_type"],
            "support_increment": len(group),
        }

    def _cluster_batch_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.enable_batch_task_aggregation:
            return [
                self._build_clustered_candidate([candidate])
                for candidate in candidates
            ]

        cluster_texts = [
            self._build_candidate_cluster_text(candidate)
            for candidate in candidates
        ]
        candidate_embeddings = self._embed_tasks_deduped(cluster_texts)
        clusters: List[Dict[str, Any]] = []
        for candidate, embedding in zip(candidates, candidate_embeddings):
            best_cluster_idx = None
            best_similarity = -1.0
            for cluster_idx, cluster in enumerate(clusters):
                if cluster["skill_type"] != candidate["skill_type"]:
                    continue
                cluster_similarity = max(
                    _cosine_similarity(embedding, existing_embedding)
                    for existing_embedding in cluster["embeddings"]
                )
                if cluster_similarity > best_similarity:
                    best_similarity = cluster_similarity
                    best_cluster_idx = cluster_idx

            if (
                best_cluster_idx is not None
                and best_similarity >= self.batch_cluster_similarity_thresh
            ):
                clusters[best_cluster_idx]["candidates"].append(candidate)
                clusters[best_cluster_idx]["embeddings"].append(embedding)
            else:
                clusters.append(
                    {
                        "skill_type": candidate["skill_type"],
                        "candidates": [candidate],
                        "embeddings": [embedding],
                    }
                )

        return [
            self._build_clustered_candidate(cluster["candidates"])
            for cluster in clusters
        ]

    def _active_records_and_embeddings(self) -> Tuple[List[GuideSkillRecord], List[Any]]:
        active_records = [record for record in self._skills if record.status == "active"]
        active_embeddings = self._embed_tasks_deduped(
            [self._record_retrieval_text(record) for record in active_records]
        )
        return active_records, active_embeddings

    def _retrieval_result_from_similarities(
        self,
        *,
        task_text: str,
        records: Sequence[GuideSkillRecord],
        similarities: Sequence[float],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = {
            "task_query": task_text,
            "skills": [],
            "skill_records": [],
        }
        scored_records = [
            (float(similarity), record)
            for record, similarity in zip(records, similarities)
            if record.status == "active"
        ]
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
            skills.append(self._format_skill_for_prompt(record))
            skill_records.append(
                {
                    "skill_id": record.skill_id,
                    "task_text": record.task_text,
                    "skill_text": record.skill_text,
                    "skill_type": record.skill_type,
                    "similarity": float(similarity),
                    "support_count": int(record.support_count),
                }
            )

        result["skills"] = skills
        result["skill_records"] = skill_records
        return result

    def _retrieve_guides_from_embedding(
        self,
        *,
        task_text: str,
        query_embedding: Any,
        global_step: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = {
            "task_query": task_text,
            "skills": [],
            "skill_records": [],
        }
        if not self.enabled or not task_text:
            return result

        active_records, active_embeddings = self._active_records_and_embeddings()
        similarities = _cosine_similarities(query_embedding, active_embeddings)
        return self._retrieval_result_from_similarities(
            task_text=task_text,
            records=active_records,
            similarities=similarities,
            top_k=top_k,
        )

    def retrieve_guides(
        self,
        *,
        task_query: Any,
        global_step: Optional[int] = None,
        top_k: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            task_text = _normalize_text(task_query, lowercase=False)
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
            )
        finally:
            self._record_retrieval_time(time.perf_counter() - started)

    def retrieve_guides_batch(
        self,
        *,
        task_queries: Sequence[Any],
        global_step: Optional[int] = None,
        top_k: Optional[int] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        try:
            task_texts = [
                _normalize_text(task_query, lowercase=False)
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
            active_records, active_embeddings = self._active_records_and_embeddings()
            similarity_rows = _cosine_similarity_matrix(embeddings, active_embeddings)
            if len(similarity_rows) != len(valid_positions):
                similarity_rows = [[] for _ in valid_positions]
            for idx, similarities in zip(valid_positions, similarity_rows):
                results[idx] = self._retrieval_result_from_similarities(
                    task_text=task_texts[idx],
                    records=active_records,
                    similarities=similarities,
                    top_k=top_k,
                )
            return results
        finally:
            self._record_retrieval_time(time.perf_counter() - started)

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
        episode_hint: str = "",
    ) -> str:
        return build_augmented_observation_text(
            observation=observation,
            skills=skills or [],
            episode_hint=episode_hint,
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
    ) -> Dict[str, float]:
        del anchor_obs, critical_mask, analysis_mode
        metrics = {
            "copd/guide_memory/enabled": 1.0 if self.enabled else 0.0,
            "copd/guide_memory/skill_candidates_added": 0.0,
            "copd/guide_memory/skill_candidates_merged": 0.0,
            "copd/guide_memory/batch_candidate_count": 0.0,
            "copd/guide_memory/batch_cluster_count": 0.0,
            "copd/guide_memory/batch_embedding_aggregation_count": 0.0,
            "copd/guide_memory/skill_merge_time_sec": 0.0,
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
            if traj_uid not in episode_analysis:
                continue
            representative_idx = min(sample_indices, key=lambda idx: step_indices_list[idx])
            analysis = episode_analysis[traj_uid]
            task_text = _normalize_text(
                analysis.get("task_description") or self._task_text_from_observation(obs_texts[representative_idx]),
                lowercase=False,
            )
            skill_text = str(analysis["episode_hint"]).strip()
            if not skill_text:
                continue

            episode_success_value = traj_success.get(traj_uid)
            candidates.append(
                {
                    "traj_uid": str(traj_uid),
                    "task_text": task_text,
                    "skill_text": self._normalize_skill_text(skill_text),
                    "skill_type": self._skill_type_from_success(episode_success_value),
                }
            )

        clustered_candidates = self._cluster_batch_candidates(candidates)
        metrics["copd/guide_memory/batch_candidate_count"] = float(len(candidates))
        metrics["copd/guide_memory/batch_cluster_count"] = float(len(clustered_candidates))
        metrics["copd/guide_memory/batch_embedding_aggregation_count"] = float(
            sum(
                1
                for candidate in clustered_candidates
                if int(candidate.get("support_increment", 1)) > 1
            )
        )

        merge_started = time.perf_counter()
        try:
            retrieval_texts = [
                self._build_retrieval_text(
                    task_text=candidate["task_text"],
                    skill_text=candidate["skill_text"],
                )
                for candidate in clustered_candidates
            ]
            candidate_embeddings = self._embed_tasks_deduped(
                retrieval_texts
            )
            matched_records = self._merge_candidate_records(
                candidates=clustered_candidates,
                retrieval_embeddings=candidate_embeddings,
            )
            for candidate, retrieval_embedding, matched_record in zip(
                clustered_candidates,
                candidate_embeddings,
                matched_records,
            ):
                added, merged = self._add_skill(
                    task_text=candidate["task_text"],
                    skill_text=candidate["skill_text"],
                    skill_type=candidate["skill_type"],
                    global_step=global_step,
                    support_increment=int(candidate.get("support_increment", 1)),
                    retrieval_embedding=retrieval_embedding,
                    matched_record=matched_record,
                    match_precomputed=True,
                )
                metrics["copd/guide_memory/skill_candidates_added"] += float(added)
                metrics["copd/guide_memory/skill_candidates_merged"] += float(merged)
        finally:
            merge_time_sec = time.perf_counter() - merge_started
            self._record_merge_time(merge_time_sec)
            metrics["copd/guide_memory/skill_merge_time_sec"] = float(merge_time_sec)

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
        mean_merge_time = (
            self._merge_time_total_sec / self._merge_time_count
            if self._merge_time_count > 0
            else 0.0
        )
        mean_retrieval_time = (
            self._retrieval_time_total_sec / self._retrieval_time_count
            if self._retrieval_time_count > 0
            else 0.0
        )
        return {
            f"{prefix}/skill_total": float(len(self._skills)),
            f"{prefix}/skill_active": _count_status("active"),
            f"{prefix}/skill_pending": _count_status("pending"),
            f"{prefix}/skill_success_workflow": _count_type(SUCCESS_WORKFLOW),
            f"{prefix}/skill_failure_avoidance": _count_type(FAILURE_AVOIDANCE),
            f"{prefix}/skill_unknown": _count_type(UNKNOWN_SKILL_TYPE),
            f"{prefix}/skill_mean_support": mean_support,
            f"{prefix}/embedding_cache_entries": float(len(self._text_embedding_cache)),
            f"{prefix}/batch_cluster_similarity_thresh": float(self.batch_cluster_similarity_thresh),
            f"{prefix}/skill_merge_time_sec_last": float(self._merge_time_last_sec),
            f"{prefix}/skill_merge_time_sec_total": float(self._merge_time_total_sec),
            f"{prefix}/skill_merge_time_sec_mean": float(mean_merge_time),
            f"{prefix}/skill_merge_time_count": float(self._merge_time_count),
            f"{prefix}/skill_retrieval_time_sec_last": float(self._retrieval_time_last_sec),
            f"{prefix}/skill_retrieval_time_sec_total": float(self._retrieval_time_total_sec),
            f"{prefix}/skill_retrieval_time_sec_mean": float(mean_retrieval_time),
            f"{prefix}/skill_retrieval_time_count": float(self._retrieval_time_count),
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
