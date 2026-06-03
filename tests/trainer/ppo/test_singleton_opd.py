import numpy as np

from gigpo.core_gigpo import compute_singleton_step_mask


def test_compute_singleton_step_mask_groups_by_prompt_uid():
    anchor_obs = np.asarray(["same", "branch", "branch", "same", "other"], dtype=object)
    prompt_uids = np.asarray(["task-a", "task-a", "task-a", "task-b", "task-b"], dtype=object)

    singleton_mask, metrics = compute_singleton_step_mask(anchor_obs, prompt_uids)

    assert singleton_mask.tolist() == [True, False, False, True, True]
    assert metrics["copd/state_group/size_1_group_count"] == 3.0
    assert metrics["copd/state_group/singleton_sample_count"] == 3.0
    assert metrics["copd/state_group/singleton_sample_prop"] == 0.6
