from gigpo.skill_distill import extract_task_query, inject_retrieved_skills


ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".strip()

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".strip()


def test_extract_task_query_from_alfworld_prompt():
    observation = ALFWORLD_TEMPLATE.format(
        task_description="put the hot apple in the fridge",
        step_count=2,
        history_length=2,
        action_history="[]",
        current_step=3,
        current_observation="You are in the kitchen.",
        admissible_actions="'open fridge'",
    )

    assert extract_task_query(observation) == "put the hot apple in the fridge"


def test_extract_task_query_from_webshop_prompt():
    observation = WEBSHOP_TEMPLATE.format(
        task_description="buy a red phone case",
        step_count=1,
        history_length=1,
        action_history="[]",
        current_step=2,
        current_observation="search results page",
        available_actions="'click[item]'",
    )

    assert extract_task_query(observation) == "buy a red phone case"


def test_inject_retrieved_skills_before_current_progress():
    observation = ALFWORLD_TEMPLATE.format(
        task_description="clean the mug and place it in the cabinet",
        step_count=1,
        history_length=1,
        action_history="[]",
        current_step=2,
        current_observation="You see a sink and a cabinet.",
        admissible_actions="'go to sink'",
    )

    injected, did_inject = inject_retrieved_skills(
        observation,
        "### General Principles\n- **Plan Ahead**: Think before acting.",
    )

    assert did_inject is True
    assert "## Retrieved Relevant Experience" in injected
    assert injected.index("## Retrieved Relevant Experience") < injected.index("Prior to this step")


def test_inject_retrieved_skills_is_idempotent():
    observation = """
You are an expert agent operating in the ALFRED Embodied Environment.

## Retrieved Relevant Experience

### General Principles
- **Plan Ahead**: Think before acting.

## Current Progress

You are now at step 2.
""".strip()

    injected, did_inject = inject_retrieved_skills(
        observation,
        "### General Principles\n- **Plan Ahead**: Think before acting.",
    )

    assert did_inject is False
    assert injected == observation
