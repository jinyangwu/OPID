from opid.analysis import OPIDEpisodeAnalyzer


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


def test_opid_openai_prompt_requests_episode_skill_and_step_skills():
    analyzer = OPIDEpisodeAnalyzer()

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "selected_steps" not in user_prompt
    assert "overall_skill" not in user_prompt
    assert "episode_skill" in user_prompt
    assert "step_skills" in user_prompt
    assert "Return format:" in user_prompt
    assert "Task description:" in user_prompt
    assert "find a red mug under 20 dollars" in user_prompt
    assert "successful trajectory into workflow" in user_prompt
    assert "critical step" in user_prompt
    assert "Candidate step indices" in user_prompt
    assert "Return only these top-level fields: episode_summary, episode_skill, step_skills" in user_prompt
    assert "episode_success: success" in user_prompt
    assert "interpreted_outcome" not in user_prompt
    assert "Because this is a successful episode" not in user_prompt


def test_opid_prompt_for_failed_episode_emphasizes_avoidance():
    analyzer = OPIDEpisodeAnalyzer()

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=0.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "failed trajectory into avoidance rules" in user_prompt
    assert "core mistake and warning signs" in user_prompt
    assert "episode_success: failure" in user_prompt
    assert "interpreted_outcome" not in user_prompt
    assert "Because this is a failed episode" not in user_prompt


def test_opid_parse_uses_episode_skill_and_step_skills():
    analyzer = OPIDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        {
          "episode_summary": "summary",
          "episode_skill": "skill",
          "selected_steps": [99],
          "step_skills": {
            "1": "click the matching item"
          }
        }
        """
    )

    assert "selected_steps" not in parsed
    assert parsed["episode_skill"] == "skill"
    assert parsed["step_skills"] == {1: "click the matching item"}


def test_opid_parse_extracts_json_from_markdown_fence():
    analyzer = OPIDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        Here is the result:
        ```json
        {
          "episode_skill": "check the constraints before choosing",
          "step_skills": {"0": "verify the key constraint first"}
        }
        ```
        """
    )

    assert parsed["episode_skill"] == "check the constraints before choosing"
    assert parsed["step_skills"] == {0: "verify the key constraint first"}


def test_opid_parse_prefers_object_with_episode_skill():
    analyzer = OPIDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        Ignore this helper object: {"note": "not the answer"}
        Final answer:
        {
          "episode_skill": "use the final matching evidence",
          "step_skills": {"1": "click only after matching the task"}
        }
        """
    )

    assert parsed["episode_skill"] == "use the final matching evidence"
    assert parsed["step_skills"] == {1: "click only after matching the task"}


def test_opid_parse_repairs_common_json_noise():
    analyzer = OPIDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        {
          'episode_skill': 'compare all required attributes before acting',
          'step_skills': {'1': 'check the attributes before selecting'},
        }
        """
    )

    assert parsed["episode_skill"] == "compare all required attributes before acting"
    assert parsed["step_skills"] == {1: "check the attributes before selecting"}
