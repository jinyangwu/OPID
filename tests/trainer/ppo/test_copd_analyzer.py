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


def test_copd_openai_prompt_uses_episode_hint_only():
    analyzer = COPDEpisodeAnalyzer()

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "selected_steps" not in user_prompt
    assert "overall_hint" not in user_prompt
    assert "episode_summary" not in user_prompt
    assert "episode_hint" in user_prompt
    assert "step_hints" not in user_prompt
    assert "Return format:" in user_prompt
    assert "Task description:" in user_prompt
    assert "find a red mug under 20 dollars" in user_prompt
    assert "successful workflow" in user_prompt
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


def test_copd_parse_uses_episode_hint_only():
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
    assert parsed["step_hints"] == {}
