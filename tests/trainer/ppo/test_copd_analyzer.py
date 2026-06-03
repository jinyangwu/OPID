from gigpo.copd import COPDEpisodeAnalyzer


def _sample_steps():
    return [
        {
            "step_index": 0,
            "observation": "Search page with several options.",
            "observation_prompt": (
                "You are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
                "Your task is to: find a red mug under 20 dollars.\n"
                "Your current observation is: Search page with several options."
            ),
            "response": "<action>search[red mug]</action>",
            "step_reward": 0.1,
        },
        {
            "step_index": 1,
            "observation": "Results include a matching red mug.",
            "observation_prompt": (
                "You are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
                "Your task is to: find a red mug under 20 dollars.\n"
                "Your current observation is: Results include a matching red mug."
            ),
            "response": "<action>click[item 2]</action>",
            "step_reward": 0.5,
        },
    ]


def test_copd_openai_prompt_requests_episode_hint_and_step_hints():
    analyzer = COPDEpisodeAnalyzer()

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "selected_steps" not in user_prompt
    assert "overall_hint" not in user_prompt
    assert "episode_hint" in user_prompt
    assert "step_hints" in user_prompt
    assert "Return format:" in user_prompt
    assert "Task description:" in user_prompt
    assert "find a red mug under 20 dollars" in user_prompt
    assert "successful workflow" in user_prompt
    assert "critical step" in user_prompt
    assert "step_hints keys must come from Candidate step indices" in user_prompt
    assert "under 45 words" in user_prompt
    assert "do not mention specific product names, colors, sizes, prices, brands" in user_prompt
    assert "Do not list many warning signs or examples" in user_prompt
    assert "episode_success: success" in user_prompt
    assert "interpreted_outcome" not in user_prompt
    assert "Because this is a successful episode" not in user_prompt


def test_copd_prompt_for_failed_episode_emphasizes_avoidance():
    analyzer = COPDEpisodeAnalyzer()

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=0.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "why the trajectory failed" in user_prompt
    assert "avoid that failure pattern" in user_prompt
    assert "core mistake and the safer general workflow" in user_prompt
    assert "episode_success: failure" in user_prompt
    assert "interpreted_outcome" not in user_prompt
    assert "Because this is a failed episode" not in user_prompt


def test_sopd_prompt_treats_all_singleton_steps_as_key_candidates():
    analyzer = COPDEpisodeAnalyzer(max_step_hints_per_traj=1)

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        analysis_mode="singleton_opd",
        episode_success=0.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "SOPD (Singleton OPD)" in user_prompt
    assert "Every candidate singleton step is treated as a key training step" in user_prompt
    assert "Do not select a subset" in user_prompt
    assert "reinforce the observed decision or correct it" in user_prompt
    assert "exactly two top-level fields" in user_prompt
    assert "every candidate singleton step index exactly once" in user_prompt
    assert "Candidate singleton step indices: [0, 1]" in user_prompt
    assert "correct on-policy decision" not in user_prompt
    assert "episode_hint" not in user_prompt
    assert "episode_success: failure" not in user_prompt


def test_copd_parse_uses_episode_hint_and_step_hints():
    analyzer = COPDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        {
          "episode_summary": "summary",
          "episode_hint": "hint",
          "selected_steps": [99],
          "step_hints": {
            "1": "click the matching item"
          }
        }
        """
    )

    assert "selected_steps" not in parsed
    assert parsed["episode_hint"] == "hint"
    assert parsed["step_hints"] == {1: "click the matching item"}


def test_copd_parse_extracts_json_from_markdown_fence():
    analyzer = COPDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        Here is the result:
        ```json
        {
          "episode_hint": "check the constraints before choosing",
          "step_hints": {"0": "verify the key constraint first"}
        }
        ```
        """
    )

    assert parsed["episode_hint"] == "check the constraints before choosing"
    assert parsed["step_hints"] == {0: "verify the key constraint first"}


def test_copd_parse_prefers_object_with_episode_hint():
    analyzer = COPDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        Ignore this helper object: {"note": "not the answer"}
        Final answer:
        {
          "episode_hint": "use the final matching evidence",
          "step_hints": {"1": "click only after matching the task"}
        }
        """
    )

    assert parsed["episode_hint"] == "use the final matching evidence"
    assert parsed["step_hints"] == {1: "click only after matching the task"}


def test_copd_parse_repairs_common_json_noise():
    analyzer = COPDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        {
          'episode_hint': 'compare all required attributes before acting',
          'step_hints': {'1': 'check the attributes before selecting'},
        }
        """
    )

    assert parsed["episode_hint"] == "compare all required attributes before acting"
    assert parsed["step_hints"] == {1: "check the attributes before selecting"}
