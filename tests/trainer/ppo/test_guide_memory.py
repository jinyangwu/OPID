from agent_system.memory.guide_memory import COPDGuideMemory, build_augmented_observation_text


PROMPT = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: find a red mug under 20 dollars.
Your current observation is: Search page.

Now it's your turn to take one action for the current step.
""".strip()


def _make_memory(promote_min_support=1):
    return COPDGuideMemory(
        {
            "enable": True,
            "top_k": 2,
            "max_per_skill_type": 1,
            "similarity_threshold": 0.2,
            "dedupe_skill_similarity_thresh": 0.88,
            "enable_batch_task_aggregation": True,
            "embedding_model_path": "hashing",
            "embedding_batch_size": 4,
            "promote_min_support": promote_min_support,
            "merge_task_similarity_thresh": 0.5,
            "merge_skill_similarity_thresh": 0.8,
            "max_skills": 8,
            "max_skill_chars": 256,
        }
    )


def _update_once(memory, *, task_prompt=PROMPT, global_step=1):
    return memory.update_from_episode_analysis(
        obs_texts=[task_prompt, task_prompt],
        anchor_obs=["Search page.", "Results page."],
        traj_uids=["traj-1", "traj-1"],
        step_indices=[0, 1],
        critical_mask=[True, True],
        episode_analysis={
            "traj-1": {
                "episode_hint": "Search with the key product attributes before clicking an item.",
                "step_hints": {
                    0: "Do not click before checking the visible product constraints.",
                },
            }
        },
        global_step=global_step,
        episode_success=[0.0, 0.0],
        analysis_mode="teacher_bootstrap",
    )


def test_episode_hint_becomes_sequence_skill_but_step_hint_does_not():
    memory = _make_memory()

    metrics = _update_once(memory)

    assert metrics["copd/guide_memory/skill_candidates_added"] == 1.0
    assert len(memory._skills) == 1
    assert memory._skills[0].skill_text == "Search with the key product attributes before clicking an item."
    assert "Do not click before checking" not in memory._skills[0].skill_text


def test_pending_skill_is_not_retrieved_until_promoted():
    memory = _make_memory(promote_min_support=2)

    _update_once(memory, global_step=1)
    first_retrieval = memory.retrieve_for_observation(observation=PROMPT, global_step=1)

    assert first_retrieval["skills"] == []
    assert memory._skills[0].status == "pending"

    metrics = _update_once(memory, global_step=2)
    second_retrieval = memory.retrieve_for_observation(observation=PROMPT, global_step=2)

    assert metrics["copd/guide_memory/skill_candidates_merged"] == 1.0
    assert len(memory._skills) == 1
    assert memory._skills[0].status == "active"
    assert second_retrieval["skills"] == [
        "Failure avoidance: Search with the key product attributes before clicking an item."
    ]


def test_semantic_task_retrieval_filters_by_similarity():
    memory = _make_memory()
    _update_once(memory)

    similar = memory.retrieve_for_observation(
        observation="Your task is to: find a red mug below 20 dollars.\nNow it's your turn",
        global_step=3,
    )
    unrelated = memory.retrieve_for_observation(
        observation="Your task is to: clean the kitchen sink.\nNow it's your turn",
        global_step=3,
    )

    assert similar["skills"] == [
        "Failure avoidance: Search with the key product attributes before clicking an item."
    ]
    assert unrelated["skills"] == []


def test_same_batch_same_task_candidates_are_aggregated_before_storage():
    memory = _make_memory(promote_min_support=2)

    metrics = memory.update_from_episode_analysis(
        obs_texts=[PROMPT, PROMPT],
        anchor_obs=["Search page.", "Search page."],
        traj_uids=["traj-1", "traj-2"],
        step_indices=[0, 0],
        critical_mask=[True, True],
        episode_analysis={
            "traj-1": {
                "episode_hint": "Avoid clicking a product before checking color, price, and product type.",
                "step_hints": {0: "Check the constraints first."},
            },
            "traj-2": {
                "episode_hint": "Avoid clicking a product before checking color, price, and product type.",
                "step_hints": {0: "Check the constraints first."},
            },
        },
        global_step=1,
        episode_success=[0.0, 0.0],
        analysis_mode="teacher_bootstrap",
    )

    assert metrics["copd/guide_memory/batch_candidate_count"] == 2.0
    assert metrics["copd/guide_memory/batch_cluster_count"] == 1.0
    assert metrics["copd/guide_memory/skill_candidates_added"] == 1.0
    assert len(memory._skills) == 1
    assert memory._skills[0].support_count == 2
    assert memory._skills[0].status == "active"


def test_retrieval_returns_one_success_and_one_failure_skill_for_same_task():
    memory = _make_memory()

    memory.update_from_episode_analysis(
        obs_texts=[PROMPT, PROMPT],
        anchor_obs=["Search page.", "Search page."],
        traj_uids=["traj-success", "traj-failure"],
        step_indices=[0, 0],
        critical_mask=[True, True],
        episode_analysis={
            "traj-success": {
                "episode_hint": "Search by required attributes, compare the result cards, then select the matching item.",
            },
            "traj-failure": {
                "episode_hint": "Avoid selecting the first result before verifying price, color, and product category.",
            },
        },
        global_step=1,
        episode_success=[1.0, 0.0],
        analysis_mode="teacher_bootstrap",
    )

    retrieval = memory.retrieve_for_observation(observation=PROMPT, global_step=2)

    assert len(retrieval["skills"]) == 2
    assert any(skill.startswith("Success workflow:") for skill in retrieval["skills"])
    assert any(skill.startswith("Failure avoidance:") for skill in retrieval["skills"])
    assert {record["skill_type"] for record in retrieval["skill_records"]} == {
        "success_workflow",
        "failure_avoidance",
    }


def test_retrieval_caps_redundant_same_type_skills_for_same_task():
    memory = _make_memory()

    memory.update_from_episode_analysis(
        obs_texts=[PROMPT, PROMPT],
        anchor_obs=["Search page.", "Search page."],
        traj_uids=["traj-1", "traj-2"],
        step_indices=[0, 0],
        critical_mask=[True, True],
        episode_analysis={
            "traj-1": {
                "episode_hint": "Avoid clicking before checking that the item is red and below the price limit.",
            },
            "traj-2": {
                "episode_hint": "Avoid buying until the product page confirms color, price, and item type.",
            },
        },
        global_step=1,
        episode_success=[0.0, 0.0],
        analysis_mode="teacher_bootstrap",
    )

    retrieval = memory.retrieve_for_observation(observation=PROMPT, global_step=2)

    assert len(memory._skills) == 2
    assert len(retrieval["skills"]) == 1
    assert retrieval["skills"][0].startswith("Failure avoidance:")


def test_batch_retrieval_deduplicates_identical_task_queries():
    memory = _make_memory()
    _update_once(memory)

    embed_calls = []
    original_embed_tasks = memory._embed_tasks

    def counted_embed_tasks(task_texts):
        embed_calls.append(list(task_texts))
        return original_embed_tasks(task_texts)

    memory._embed_tasks = counted_embed_tasks

    retrievals = memory.retrieve_for_observations(
        observations=[PROMPT, PROMPT],
        global_step=3,
    )

    assert len(retrievals) == 2
    assert all(retrieval["skills"] for retrieval in retrievals)
    assert embed_calls == [["find a red mug under 20 dollars"]]


def test_augmented_observation_injects_skills_and_step_hint():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        skills=["Search with the key product attributes before clicking an item."],
        step_hint="Check price and color before selecting the product.",
    )

    assert "Retrieved Reusable Skills" in augmented
    assert "Current-Step Decision Guidance" in augmented
    assert augmented.index("Retrieved Reusable Skills") < augmented.index("Now it's your turn")
    assert "Check price and color before selecting the product." in augmented


def test_augmented_observation_step_hint_only_keeps_step_hint_path():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        skills=[],
        step_hint="Check price and color before selecting the product.",
    )

    assert "Retrieved Reusable Skills" not in augmented
    assert "Current-Step Decision Guidance" in augmented
    assert "Check price and color before selecting the product." in augmented
