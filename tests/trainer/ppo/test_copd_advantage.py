import numpy as np
import torch

from gigpo.core_gigpo import compute_copd_advantage_components


def test_copd_uses_independent_episode_and_step_hint_teacher_weights():
    response_mask = torch.ones((2, 2))
    old_log_prob = torch.zeros((2, 2))

    _, _, teacher_advantages, scores, metrics = compute_copd_advantage_components(
        token_level_rewards=torch.zeros((2, 2)),
        step_rewards=torch.zeros((2, 2)),
        response_mask=response_mask,
        anchor_obs=np.asarray(["obs-a", "obs-b"], dtype=object),
        index=np.asarray(["task", "task"], dtype=object),
        traj_index=np.asarray(["traj-a", "traj-b"], dtype=object),
        episode_teacher_log_prob=torch.ones((2, 2)),
        step_teacher_log_prob=torch.full((2, 2), 2.0),
        old_log_prob=old_log_prob,
        critical_step_mask=torch.tensor([1.0, 0.0]),
        step_hint_mask=torch.tensor([0.0, 1.0]),
        episode_hint_teacher_advantage_w=0.25,
        step_hint_teacher_advantage_w=0.5,
    )

    expected = torch.tensor([[0.25, 0.25], [1.0, 1.0]])
    torch.testing.assert_close(teacher_advantages, expected)
    torch.testing.assert_close(scores, expected)
    assert metrics["copd/adv/episode_hint_teacher_weight"] == 0.25
    assert metrics["copd/adv/step_hint_teacher_weight"] == 0.5


def test_copd_adds_episode_and_step_hint_teacher_advantages_on_same_step():
    response_mask = torch.ones((1, 2))
    old_log_prob = torch.zeros((1, 2))

    _, _, teacher_advantages, scores, _ = compute_copd_advantage_components(
        token_level_rewards=torch.zeros((1, 2)),
        step_rewards=torch.zeros((1, 2)),
        response_mask=response_mask,
        anchor_obs=np.asarray(["obs"], dtype=object),
        index=np.asarray(["task"], dtype=object),
        traj_index=np.asarray(["traj"], dtype=object),
        episode_teacher_log_prob=torch.ones((1, 2)),
        step_teacher_log_prob=torch.full((1, 2), 2.0),
        old_log_prob=old_log_prob,
        critical_step_mask=torch.tensor([1.0]),
        step_hint_mask=torch.tensor([1.0]),
        episode_hint_teacher_advantage_w=0.001,
        step_hint_teacher_advantage_w=0.001,
    )

    expected = torch.full((1, 2), 0.003)
    torch.testing.assert_close(teacher_advantages, expected)
    torch.testing.assert_close(scores, expected)
