import pytest

from opid.prompting import build_augmented_observation_text, select_skill_teacher_sources


PROMPT = """You are an expert autonomous agent.
Your task is to: find a red mug under 20 dollars.

Now it's your turn to choose the next action."""


def test_augmented_observation_injects_episode_skill_before_turn_anchor():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        episode_skill="Check price and color before selecting the product.",
    )

    assert "Episode-Level Skill" in augmented
    assert "Check price and color before selecting the product." in augmented
    assert augmented.index("Episode-Level Skill") < augmented.index("Now it's your turn")
    assert "Retrieved Reusable Skills" not in augmented


def test_augmented_observation_injects_step_skill_after_episode_skill():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        episode_skill="Check constraints before selecting the product.",
        step_skill="Search with the required attributes before clicking.",
    )

    assert "Episode-Level Skill" in augmented
    assert "Critical-Step Skill" in augmented
    assert "Search with the required attributes before clicking." in augmented
    assert augmented.index("Episode-Level Skill") < augmented.index("Critical-Step Skill")
    assert augmented.index("Critical-Step Skill") < augmented.index("Now it's your turn")


def test_augmented_observation_without_episode_skill_is_unchanged():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        episode_skill="",
        step_skill="",
    )

    assert augmented == PROMPT


def test_step_priority_uses_only_step_skill_when_available():
    assert select_skill_teacher_sources(
        step_skill="Check the receptacle first.",
        episode_skill_enabled=True,
        step_skill_enabled=True,
        mode="step_priority",
    ) == (False, True)


def test_additive_uses_episode_and_step_skills_when_available():
    assert select_skill_teacher_sources(
        step_skill="Check the receptacle first.",
        episode_skill_enabled=True,
        step_skill_enabled=True,
        mode="additive",
    ) == (True, True)


def test_additive_uses_only_episode_skill_without_step_skill():
    assert select_skill_teacher_sources(
        step_skill="",
        episode_skill_enabled=True,
        step_skill_enabled=True,
        mode="additive",
    ) == (True, False)


def test_skill_teacher_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported OPID skill_teacher_mode"):
        select_skill_teacher_sources(
            step_skill="skill",
            episode_skill_enabled=True,
            step_skill_enabled=True,
            mode="unknown",
        )
