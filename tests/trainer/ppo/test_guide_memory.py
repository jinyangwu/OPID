from agent_system.memory.guide_memory import (
    COPDGuideMemory,
    _cosine_similarity_matrix,
    build_augmented_observation_text,
)


PROMPT = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: find a red mug under 20 dollars.
Your current observation is: Search page.

Now it's your turn to take one action for the current step.
""".strip()


def _make_memory(promote_min_support=1, **overrides):
    config = {
        "enable": True,
        "top_k": 2,
        "max_per_skill_type": 1,
        "dedupe_skill_similarity_thresh": 0.88,
        "enable_batch_task_aggregation": True,
        "embedding_model_path": "hashing",
        "embedding_batch_size": 4,
        "promote_min_support": promote_min_support,
        "merge_task_similarity_thresh": 0.5,
        "max_skills": 8,
    }
    config.update(overrides)
    return COPDGuideMemory(config)


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


def test_cosine_similarity_matrix_scores_multiple_queries():
    scores = _cosine_similarity_matrix(
        queries=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        candidates=[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
    )

    assert len(scores) == 2
    assert len(scores[0]) == 3
    assert abs(scores[0][0] - 1.0) < 1e-6
    assert abs(scores[0][1]) < 1e-6
    assert abs(scores[1][0]) < 1e-6
    assert abs(scores[1][1] - 1.0) < 1e-6
    assert scores[0][2] > 0.7
    assert scores[1][2] > 0.7


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
    assert len(second_retrieval["skills"]) == 1
    assert second_retrieval["skills"][0].startswith(
        "Failure avoidance: Search with the key product attributes before clicking an item."
    )


def test_same_task_text_does_not_bypass_embedding_merge_threshold():
    memory = _make_memory(
        merge_task_similarity_thresh=1.1,
    )

    memory.update_from_episode_analysis(
        obs_texts=[PROMPT],
        anchor_obs=["Search page."],
        traj_uids=["traj-1"],
        step_indices=[0],
        critical_mask=[True],
        episode_analysis={
            "traj-1": {
                "episode_hint": "Search broadly using the product noun first.",
            },
        },
        global_step=1,
        episode_success=[0.0],
        analysis_mode="teacher_bootstrap",
    )
    metrics = memory.update_from_episode_analysis(
        obs_texts=[PROMPT],
        anchor_obs=["Search page."],
        traj_uids=["traj-2"],
        step_indices=[0],
        critical_mask=[True],
        episode_analysis={
            "traj-2": {
                "episode_hint": "Avoid buying until the product page confirms every constraint.",
            },
        },
        global_step=2,
        episode_success=[0.0],
        analysis_mode="teacher_bootstrap",
    )

    assert metrics["copd/guide_memory/skill_candidates_merged"] == 0.0
    assert metrics["copd/guide_memory/skill_candidates_added"] == 1.0
    assert len(memory._skills) == 2


def test_semantic_task_retrieval_ranks_by_similarity_without_threshold():
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

    assert len(similar["skills"]) == 1
    assert similar["skills"][0].startswith(
        "Failure avoidance: Search with the key product attributes before clicking an item."
    )
    assert len(unrelated["skills"]) == 1


def test_guide_memory_records_merge_and_retrieval_timing_metrics():
    memory = _make_memory()

    update_metrics = _update_once(memory)
    assert update_metrics["copd/guide_memory/skill_merge_time_sec"] >= 0.0
    assert update_metrics["copd/guide_memory/skill_merge_time_count"] == 1.0
    assert update_metrics["copd/guide_memory/skill_retrieval_time_count"] == 0.0

    memory.retrieve_for_observation(observation=PROMPT, global_step=2)
    snapshot_metrics = memory.snapshot_metrics(prefix="copd/guide_memory")
    assert snapshot_metrics["copd/guide_memory/skill_retrieval_time_sec_last"] >= 0.0
    assert snapshot_metrics["copd/guide_memory/skill_retrieval_time_count"] == 1.0


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


def test_batch_history_merge_uses_similarity_matrix():
    from agent_system.memory import guide_memory as guide_memory_module

    memory = _make_memory(merge_task_similarity_thresh=0.0)
    memory.update_from_episode_analysis(
        obs_texts=[PROMPT, PROMPT],
        anchor_obs=["Search page.", "Search page."],
        traj_uids=["traj-success", "traj-failure"],
        step_indices=[0, 0],
        critical_mask=[True, True],
        episode_analysis={
            "traj-success": {
                "episode_hint": "Search by required attributes before selecting the item.",
            },
            "traj-failure": {
                "episode_hint": "Avoid selecting a result before checking every constraint.",
            },
        },
        global_step=1,
        episode_success=[1.0, 0.0],
        analysis_mode="teacher_bootstrap",
    )

    matrix_calls = []
    original_matrix = guide_memory_module._cosine_similarity_matrix

    def counted_matrix(queries, candidates):
        matrix_calls.append((len(queries), len(candidates)))
        return original_matrix(queries, candidates)

    guide_memory_module._cosine_similarity_matrix = counted_matrix
    try:
        metrics = memory.update_from_episode_analysis(
            obs_texts=[PROMPT, PROMPT],
            anchor_obs=["Search page.", "Search page."],
            traj_uids=["traj-success-2", "traj-failure-2"],
            step_indices=[0, 0],
            critical_mask=[True, True],
            episode_analysis={
                "traj-success-2": {
                    "episode_hint": "Compare required attributes before selecting the item.",
                },
                "traj-failure-2": {
                    "episode_hint": "Avoid buying before confirming every product constraint.",
                },
            },
            global_step=2,
            episode_success=[1.0, 0.0],
            analysis_mode="teacher_bootstrap",
        )
    finally:
        guide_memory_module._cosine_similarity_matrix = original_matrix

    assert metrics["copd/guide_memory/skill_candidates_merged"] == 2.0
    assert any(query_count == 2 and candidate_count >= 2 for query_count, candidate_count in matrix_calls)


def test_batch_candidates_are_clustered_by_task_skill_embedding():
    memory = _make_memory(
        promote_min_support=2,
        batch_cluster_similarity_thresh=0.2,
    )
    related_prompt = PROMPT.replace(
        "find a red mug under 20 dollars",
        "find a blue cup under 25 dollars",
    )

    metrics = memory.update_from_episode_analysis(
        obs_texts=[PROMPT, related_prompt],
        anchor_obs=["Search page.", "Search page."],
        traj_uids=["traj-1", "traj-2"],
        step_indices=[0, 0],
        critical_mask=[True, True],
        episode_analysis={
            "traj-1": {
                "episode_hint": "Avoid clicking before checking product constraints.",
            },
            "traj-2": {
                "episode_hint": "Avoid buying before checking product constraints.",
            },
        },
        global_step=1,
        episode_success=[0.0, 0.0],
        analysis_mode="teacher_bootstrap",
    )

    assert metrics["copd/guide_memory/batch_embedding_aggregation_count"] == 1.0
    assert len(memory._skills) == 1
    assert memory._skills[0].support_count == 2
    assert memory._skills[0].skill_text in {
        "Avoid clicking before checking product constraints.",
        "Avoid buying before checking product constraints.",
    }


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

    assert len(memory._skills) == 1
    assert memory._skills[0].support_count == 2
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
    assert len(embed_calls) == 1
    assert "find a red mug under 20 dollars" in embed_calls[0][0]


def test_embedding_cache_uses_configured_device_when_torch_available():
    from agent_system.memory import guide_memory as guide_memory_module

    torch = guide_memory_module._get_torch_module()
    if torch is None:
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    memory = _make_memory(embedding_device=device)
    embedding = memory._embed_task("find a red mug under 20 dollars")

    assert torch.is_tensor(embedding)
    assert embedding.device == torch.device(device)
    assert memory._text_embedding_cache
    assert all(
        torch.is_tensor(cached_embedding) and cached_embedding.device == torch.device(device)
        for cached_embedding in memory._text_embedding_cache.values()
    )


def test_skill_snapshot_does_not_persist_embeddings():
    import json
    import os
    import tempfile

    memory = _make_memory()
    _update_once(memory)

    with tempfile.TemporaryDirectory() as tmp_dir:
        snapshot_path = os.path.join(tmp_dir, "guide_memory.json")
        memory.dump_snapshot(snapshot_path)

        with open(snapshot_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    assert payload["skills"]
    serialized_skill = payload["skills"][0]
    assert "task_embedding" not in serialized_skill
    assert "retrieval_embedding" not in serialized_skill
    assert serialized_skill["retrieval_text"] == (
        "TASK: find a red mug under 20 dollars SKILL: Search with the key product attributes before clicking an item."
    )
    assert set(serialized_skill) == {
        "skill_id",
        "task_text",
        "skill_text",
        "skill_type",
        "retrieval_text",
        "support_count",
        "status",
        "created_step",
        "last_updated_step",
    }


def test_augmented_observation_injects_skills_and_episode_hint():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        skills=["Search with the key product attributes before clicking an item."],
        episode_hint="Check price and color before selecting the product.",
    )

    assert "Retrieved Reusable Skills" in augmented
    assert "Episode-Level Guidance" in augmented
    assert augmented.index("Retrieved Reusable Skills") < augmented.index("Now it's your turn")
    assert "Check price and color before selecting the product." in augmented


def test_augmented_observation_episode_hint_only_keeps_episode_guidance_path():
    augmented = build_augmented_observation_text(
        observation=PROMPT,
        skills=[],
        episode_hint="Check price and color before selecting the product.",
    )

    assert "Retrieved Reusable Skills" not in augmented
    assert "Episode-Level Guidance" in augmented
    assert "Check price and color before selecting the product." in augmented
