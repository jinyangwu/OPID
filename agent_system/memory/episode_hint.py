import re
from typing import Any, Sequence


def build_augmented_observation_text(
    *,
    observation: str,
    episode_hint: str = "",
    step_hint: str = "",
) -> str:
    """Insert episode-level and optional critical-step teacher context into a prompt."""

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

    def _insert_after_task_description(prompt_text: str, sections: Sequence[str]) -> str:
        merged_sections = [str(section).strip() for section in sections if str(section or "").strip()]
        if not merged_sections:
            return prompt_text

        insertion_block = "\n\n".join(merged_sections)
        task_line_patterns = (
            r"^.*\bYour task is to:\s*.*(?:\n|$)",
            r"^.*\bYour current task is:\s*.*(?:\n|$)",
            r"^.*\bYour task is:\s*.*(?:\n|$)",
            r"^.*\bYour question:\s*.*(?:\n|$)",
        )
        for pattern in task_line_patterns:
            match = re.search(pattern, prompt_text, flags=re.MULTILINE)
            if match is None:
                continue

            prefix = prompt_text[: match.end()].rstrip()
            suffix = prompt_text[match.end() :].lstrip()
            if prefix and suffix:
                return f"{prefix}\n\n{insertion_block}\n\n{suffix}"
            if prefix:
                return f"{prefix}\n\n{insertion_block}"
            return f"{insertion_block}\n\n{suffix}"

        return _insert_before_anchor(
            prompt_text,
            merged_sections,
            (
                "Now it's your turn to",
                "Now it's your turn",
            ),
        )

    episode_section = _format_guidance_section(
        "Episode-Level Guidance",
        (
            "Refer to this episode-level guidance when deciding what action "
            "to take in the current episode"
        ),
        episode_hint,
    )
    step_section = _format_guidance_section(
        "Critical-Step Guidance",
        "Use this current-step guidance for this decision only",
        step_hint,
    )
    prompt_text = _insert_after_task_description(str(observation).strip(), [episode_section])
    return _insert_before_anchor(
        prompt_text,
        [step_section],
        (
            "Now it's your turn to",
            "Now it's your turn",
        ),
    )
