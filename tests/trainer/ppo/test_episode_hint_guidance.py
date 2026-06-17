import pytest

from agent_system.memory.episode_hint import build_augmented_observation_text, select_hint_teacher_sources


PROMPT = """You are an expert autonomous agent.
Your task is to: find a red mug under 20 dollars.

Now it's your turn to choose the next action."""


def test_augmented_observation_injects_episode_hint_before_turn_anchor():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        episode_hint="Check price and color before selecting the product.",
    )

    assert "Episode-Level Guidance" in augmented
    assert "Check price and color before selecting the product." in augmented
    assert augmented.index("Episode-Level Guidance") < augmented.index("Now it's your turn")
    assert "Retrieved Reusable Skills" not in augmented


def test_augmented_observation_injects_step_hint_after_episode_hint():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        episode_hint="Check constraints before selecting the product.",
        step_hint="Search with the required attributes before clicking.",
    )

    assert "Episode-Level Guidance" in augmented
    assert "Critical-Step Guidance" in augmented
    assert "Search with the required attributes before clicking." in augmented
    assert augmented.index("Episode-Level Guidance") < augmented.index("Critical-Step Guidance")
    assert augmented.index("Critical-Step Guidance") < augmented.index("Now it's your turn")


def test_augmented_observation_without_episode_hint_is_unchanged():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        episode_hint="",
        step_hint="",
    )

    assert augmented == PROMPT


def test_step_priority_uses_only_step_hint_when_available():
    assert select_hint_teacher_sources(
        step_hint="Check the receptacle first.",
        episode_hint_enabled=True,
        step_hint_enabled=True,
        mode="step_priority",
    ) == (False, True)


def test_additive_uses_episode_and_step_hints_when_available():
    assert select_hint_teacher_sources(
        step_hint="Check the receptacle first.",
        episode_hint_enabled=True,
        step_hint_enabled=True,
        mode="additive",
    ) == (True, True)


def test_additive_uses_only_episode_hint_without_step_hint():
    assert select_hint_teacher_sources(
        step_hint="",
        episode_hint_enabled=True,
        step_hint_enabled=True,
        mode="additive",
    ) == (True, False)


def test_hint_teacher_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported COPD hint_teacher_mode"):
        select_hint_teacher_sources(
            step_hint="hint",
            episode_hint_enabled=True,
            step_hint_enabled=True,
            mode="unknown",
        )
