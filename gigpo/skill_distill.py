import logging
import re
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


_TASK_PATTERNS = (
    re.compile(
        r"Your task is to:\s*(.+?)(?=\n\n## Retrieved Relevant Experience|\n\n## Current Progress|\nPrior to this step|\nYou are now at step|\nYour current observation is|\nYour admissible actions|\nNow it's your turn|$)",
        re.S,
    ),
    re.compile(
        r"Your task is to:\s*(.+?)(?=\.\s*(?:Prior to this step|You are now at step|Your current observation is|Your admissible actions|Now it's your turn)|$)",
        re.S,
    ),
)

_SKILL_SECTION_HEADER = "## Retrieved Relevant Experience"
_INSERTION_MARKERS = (
    "\n\n## Current Progress",
    "\nPrior to this step",
    "\nYou are now at step",
    "\nYour current observation is",
    "\nNow it's your turn",
)


def extract_task_query(observation: str) -> str:
    """Extract the task description from a formatted environment prompt."""
    text = str(observation or "").strip()
    if not text:
        return ""

    for pattern in _TASK_PATTERNS:
        match = pattern.search(text)
        if match:
            task = " ".join(match.group(1).split()).strip(" .")
            if task:
                return task

    return text


def inject_retrieved_skills(observation: str, retrieved_skills_text: str) -> Tuple[str, bool]:
    """Insert formatted skills into an existing prompt without rebuilding it from scratch."""
    base_text = str(observation or "").strip()
    skills_text = str(retrieved_skills_text or "").strip()

    if not base_text or not skills_text:
        return base_text, False

    if _SKILL_SECTION_HEADER in base_text:
        return base_text, False

    skill_block = f"{_SKILL_SECTION_HEADER}\n\n{skills_text}"
    for marker in _INSERTION_MARKERS:
        marker_idx = base_text.find(marker)
        if marker_idx != -1:
            injected_text = (
                f"{base_text[:marker_idx].rstrip()}\n\n"
                f"{skill_block}"
                f"{base_text[marker_idx:]}"
            )
            return injected_text, True

    return f"{base_text}\n\n{skill_block}", True


class SkillEnhancedTeacherContext:
    """Build teacher-side skill-enhanced prompts from the existing skills-only memory bank."""

    def __init__(self, config):
        from omegaconf import OmegaConf

        from agent_system.memory import SkillsOnlyMemory

        skills_json_path = OmegaConf.select(config, "env.skills_only_memory.skills_json_path")
        if not skills_json_path:
            raise ValueError("env.skills_only_memory.skills_json_path must be provided for skill distillation.")

        memory_cfg = OmegaConf.select(config, "env.skills_only_memory") or {}
        self.memory = SkillsOnlyMemory(
            skills_json_path=skills_json_path,
            retrieval_mode=memory_cfg.get("retrieval_mode", "template"),
            embedding_model_path=memory_cfg.get("embedding_model_path", None),
            task_specific_top_k=memory_cfg.get("task_specific_top_k", None),
        )
        self.top_k = int(memory_cfg.get("top_k", 6))
        self.similarity_threshold = float(memory_cfg.get("similarity_threshold", 0.7))
        self.max_tokens = int(memory_cfg.get("max_tokens", 2000))
        self.include_examples = bool(memory_cfg.get("include_examples", False))
        self._formatted_skill_cache: Dict[str, str] = {}

    def _retrieve_formatted_skills(self, query: str) -> str:
        query = str(query or "").strip()
        if not query:
            return ""

        if query not in self._formatted_skill_cache:
            retrieved_skills = self.memory.retrieve(
                task_description=query,
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold,
                max_tokens=self.max_tokens,
                include_examples=self.include_examples,
            )
            self._formatted_skill_cache[query] = self.memory.format_for_prompt(retrieved_skills)
        return self._formatted_skill_cache[query]

    def build_enhanced_observation(self, observation: str) -> Tuple[str, bool]:
        query = extract_task_query(observation)
        retrieved_skills_text = self._retrieve_formatted_skills(query)
        return inject_retrieved_skills(observation=observation, retrieved_skills_text=retrieved_skills_text)

    def build_enhanced_observations(self, observations: List[str]) -> Tuple[List[str], Dict[str, float]]:
        enhanced_observations: List[str] = []
        injected_count = 0
        unique_queries = set()

        for observation in observations:
            query = extract_task_query(observation)
            if query:
                unique_queries.add(query)
            enhanced_observation, injected = self.build_enhanced_observation(observation)
            enhanced_observations.append(enhanced_observation)
            injected_count += int(injected)

        total_count = len(observations)
        stats = {
            "batch_size": float(total_count),
            "num_unique_queries": float(len(unique_queries)),
            "num_injected": float(injected_count),
            "injected_ratio": float(injected_count / total_count) if total_count > 0 else 0.0,
        }
        return enhanced_observations, stats
