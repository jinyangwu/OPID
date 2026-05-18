# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo
from gigpo import copd as core_copd

from agent_system.memory.episode_hint import build_augmented_observation_text
from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]
module_logger = logging.getLogger(__name__)
COPD_STATE_GROUP_METRIC_PREFIX = "copd/state_group/"


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'
    COPD = "copd"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    step_advantage_w=1.0,
    gigpo_mode="mean_std_norm",
    gigpo_enable_similarity=False,
    gigpo_similarity_thresh=0.95,
    teacher_advantage_w=1.0,
    episode_hint_teacher_advantage_w=None,
    step_hint_teacher_advantage_w=0.0,
    copd_mode="mean_norm",
    copd_enable_similarity=False,
    copd_similarity_thresh=0.95,
    copd_normalize_teacher_adv=False,
    copd_clip_teacher_adv=None,
    **kwargs,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns, gigpo_adv_metrics = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            return_metrics=True,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        if data.meta_info is None:
            data.meta_info = {}
        data.meta_info["gigpo_adv_metrics"] = gigpo_adv_metrics
    elif adv_estimator == AdvantageEstimator.COPD:
        teacher_log_prob = data.batch['teacher_log_prob'] if 'teacher_log_prob' in data.batch.keys() else None
        episode_teacher_log_prob = (
            data.batch['episode_teacher_log_prob']
            if 'episode_teacher_log_prob' in data.batch.keys()
            else None
        )
        step_teacher_log_prob = (
            data.batch['step_teacher_log_prob']
            if 'step_teacher_log_prob' in data.batch.keys()
            else None
        )
        old_log_prob = data.batch['old_log_probs'] if 'old_log_probs' in data.batch.keys() else None
        if 'critical_step_mask' in data.batch.keys():
            critical_step_mask = data.batch['critical_step_mask']
        elif 'critical_step_mask' in data.non_tensor_batch.keys():
            critical_step_mask = data.non_tensor_batch['critical_step_mask']
        else:
            critical_step_mask = None
        if 'step_hint_mask' in data.batch.keys():
            step_hint_mask = data.batch['step_hint_mask']
        elif 'step_hint_mask' in data.non_tensor_batch.keys():
            step_hint_mask = data.non_tensor_batch['step_hint_mask']
        else:
            step_hint_mask = None

        advantages, returns, copd_adv_metrics = core_gigpo.compute_copd_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            step_rewards=data.batch['step_rewards'],
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            teacher_log_prob=teacher_log_prob,
            episode_teacher_log_prob=episode_teacher_log_prob,
            step_teacher_log_prob=step_teacher_log_prob,
            old_log_prob=old_log_prob,
            critical_step_mask=critical_step_mask,
            step_hint_mask=step_hint_mask,
            step_advantage_w=step_advantage_w,
            teacher_advantage_w=teacher_advantage_w,
            episode_hint_teacher_advantage_w=episode_hint_teacher_advantage_w,
            step_hint_teacher_advantage_w=step_hint_teacher_advantage_w,
            mode=copd_mode,
            enable_similarity=copd_enable_similarity,
            similarity_thresh=copd_similarity_thresh,
            normalize_teacher_adv=copd_normalize_teacher_adv,
            clip_teacher_adv=copd_clip_teacher_adv,
            return_metrics=True,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        if data.meta_info is None:
            data.meta_info = {}
        data.meta_info["copd_adv_metrics"] = copd_adv_metrics
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self._copd_analyzer = None
        self._copd_teacher_adv_last_enabled_state = None
        self._copd_teacher_adv_last_weight = None
        self._copd_failed_only_last_enabled_state = None
        self._copd_analysis_last_enabled_state = None
        self._copd_teacher_signal_executor = None
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO,
            AdvantageEstimator.COPD,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _get_copd_phase_switch_after_steps(self) -> Optional[int]:
        phase_switch_after_steps = OmegaConf.select(self.config, "algorithm.copd.phase_switch_after_steps")
        if phase_switch_after_steps is None:
            return None
        return int(phase_switch_after_steps)

    def _get_copd_opd_start_after_steps(self) -> Optional[int]:
        start_after_steps = OmegaConf.select(self.config, "algorithm.copd.opd_start_after_steps")
        if start_after_steps is None:
            # Backward compatibility for older launch commands.
            start_after_steps = OmegaConf.select(self.config, "algorithm.copd.teacher_advantage_start_after_steps")
        if start_after_steps is None:
            return None
        return int(start_after_steps)

    def _get_copd_failed_only_after_steps(self) -> Optional[int]:
        failed_only_after_steps = OmegaConf.select(self.config, "algorithm.copd.failed_only_after_steps")
        if failed_only_after_steps is None:
            return None
        return int(failed_only_after_steps)

    def _should_copd_analyze_failed_only(self) -> bool:
        failed_only_after_steps = self._get_copd_failed_only_after_steps()
        if failed_only_after_steps is None:
            enabled = bool(OmegaConf.select(self.config, "algorithm.copd.failed_only"))
            schedule_text = "static config"
        else:
            enabled = self.global_steps > failed_only_after_steps
            schedule_text = f"after step {failed_only_after_steps}"

        if self._copd_failed_only_last_enabled_state != enabled:
            module_logger.info(
                "COPD failed-only episode analysis is %s at global_step=%s (%s).",
                "enabled" if enabled else "disabled",
                self.global_steps,
                schedule_text,
            )
            self._copd_failed_only_last_enabled_state = enabled
        return enabled

    def _get_copd_teacher_advantage_weight(self, teacher_enabled: bool) -> float:
        if not teacher_enabled:
            return 0.0

        target_weight = float(OmegaConf.select(self.config, "algorithm.copd.teacher_advantage_w") or 0.0)
        start_weight = OmegaConf.select(self.config, "algorithm.copd.teacher_advantage_w_start")
        if start_weight is None:
            weight = target_weight
        else:
            start_weight = float(start_weight)
            ramp_steps = OmegaConf.select(self.config, "algorithm.copd.teacher_advantage_w_ramp_steps")
            start_after_steps = self._get_copd_opd_start_after_steps() or 0
            if ramp_steps is None:
                ramp_steps = max(int(self.total_training_steps) - start_after_steps, 1)
            else:
                ramp_steps = int(ramp_steps)

            if ramp_steps <= 1:
                weight = target_weight
            else:
                teacher_enabled_step = self.global_steps - start_after_steps
                progress = min(max(teacher_enabled_step - 1, 0), ramp_steps - 1) / float(ramp_steps - 1)
                weight = start_weight + (target_weight - start_weight) * progress

        if self._copd_teacher_adv_last_weight is None or abs(self._copd_teacher_adv_last_weight - weight) > 1e-8:
            module_logger.info(
                "COPD teacher advantage weight is %.6f at global_step=%s.",
                weight,
                self.global_steps,
            )
            self._copd_teacher_adv_last_weight = weight
        return weight

    def _is_copd_analysis_enabled(self) -> bool:
        configured = OmegaConf.select(self.config, "algorithm.copd.enable_analysis")
        if configured is None:
            enabled = True
        elif isinstance(configured, str):
            enabled = configured.lower() in ("1", "true", "yes", "on")
        else:
            enabled = bool(configured)

        if self._copd_analysis_last_enabled_state != enabled:
            module_logger.info(
                "COPD analysis and teacher signal construction are %s at global_step=%s.",
                "enabled" if enabled else "disabled",
                self.global_steps,
            )
            self._copd_analysis_last_enabled_state = enabled
        return enabled

    def _is_copd_teacher_adv_enabled(self) -> bool:
        start_after_steps = self._get_copd_opd_start_after_steps()
        phase_switch_after_steps = self._get_copd_phase_switch_after_steps()

        enabled = True
        disabled_reason = None
        if start_after_steps is not None and self.global_steps <= start_after_steps:
            enabled = False
            disabled_reason = "before_start"
        elif phase_switch_after_steps is not None and self.global_steps > phase_switch_after_steps:
            enabled = False
            disabled_reason = "after_phase_switch"

        if self._copd_teacher_adv_last_enabled_state != enabled:
            if enabled:
                schedule_parts = []
                if start_after_steps is not None:
                    schedule_parts.append(f"after step {start_after_steps}")
                if phase_switch_after_steps is not None:
                    schedule_parts.append(f"until step {phase_switch_after_steps}")
                schedule_text = f" ({', '.join(schedule_parts)})" if schedule_parts else ""
                module_logger.info(
                    "COPD teacher advantage is enabled at global_step=%s%s.",
                    self.global_steps,
                    schedule_text,
                )
            elif disabled_reason == "before_start":
                module_logger.info(
                    "COPD teacher advantage is disabled at global_step=%s until after step %s.",
                    self.global_steps,
                    start_after_steps,
                )
            else:
                module_logger.info(
                    "COPD teacher advantage is disabled at global_step=%s because phase_switch_after_steps=%s.",
                    self.global_steps,
                    phase_switch_after_steps,
                )
            self._copd_teacher_adv_last_enabled_state = enabled
        return enabled

    def _set_zero_copd_teacher_signals(self, batch: DataProto, metrics: Dict[str, float]) -> DataProto:
        batch_size = len(batch)
        batch.batch["teacher_log_prob"] = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        batch.batch["episode_teacher_log_prob"] = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        batch.batch["step_teacher_log_prob"] = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        batch.batch["critical_step_mask"] = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=batch.batch["responses"].device,
        )
        batch.batch["step_hint_mask"] = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=batch.batch["responses"].device,
        )
        batch.batch["teacher_signal_mask"] = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=batch.batch["responses"].device,
        )
        metrics["copd/critical_step_ratio"] = 0.0
        metrics["copd/teacher_batch_size"] = 0.0
        metrics["copd/teacher_available"] = 0.0
        metrics["copd/episode_hint_guidance/enabled"] = 0.0
        metrics["copd/step_hint_guidance/step_guided_step_ratio"] = 0.0
        metrics["copd/step_hint_guidance/step_hints_applied"] = 0.0
        return batch

    def _lazy_init_copd_teacher_signal_executor(self):
        if self._copd_teacher_signal_executor is None:
            self._copd_teacher_signal_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="copd-teacher-signal",
            )
        return self._copd_teacher_signal_executor

    def _build_copd_teacher_signal_snapshot(self, batch: DataProto) -> DataProto:
        tensors = {
            "responses": batch.batch["responses"].clone(),
            "attention_mask": batch.batch["attention_mask"].clone(),
        }
        if "step_rewards" in batch.batch.keys():
            tensors["step_rewards"] = batch.batch["step_rewards"].clone()

        non_tensor_keys = [
            "obs_text",
            "obs_text_base",
            "anchor_obs",
            "traj_uid",
            "uid",
            "data_source",
            "episode_success",
            "episode_rewards",
            "multi_modal_inputs",
        ]
        non_tensors = {}
        for key in non_tensor_keys:
            if key in batch.non_tensor_batch:
                non_tensors[key] = deepcopy(batch.non_tensor_batch[key])

        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=deepcopy(batch.meta_info),
        )

    def _prepare_copd_teacher_signals_async_task(self, batch: DataProto, teacher_enabled: bool):
        local_metrics: Dict[str, float] = {}
        output_batch = self._prepare_copd_teacher_signals(
            batch=batch,
            metrics=local_metrics,
            teacher_enabled=teacher_enabled,
        )
        return output_batch, local_metrics

    def _merge_async_copd_teacher_signals(
        self,
        batch: DataProto,
        teacher_signal_batch: DataProto,
    ) -> DataProto:
        source_indices = batch.non_tensor_batch.get("_batch_source_idx")
        teacher_log_prob = teacher_signal_batch.batch["teacher_log_prob"]
        episode_teacher_log_prob = (
            teacher_signal_batch.batch["episode_teacher_log_prob"]
            if "episode_teacher_log_prob" in teacher_signal_batch.batch.keys()
            else teacher_log_prob
        )
        step_teacher_log_prob = (
            teacher_signal_batch.batch["step_teacher_log_prob"]
            if "step_teacher_log_prob" in teacher_signal_batch.batch.keys()
            else torch.zeros_like(teacher_log_prob, dtype=torch.float32)
        )
        critical_step_mask = teacher_signal_batch.batch["critical_step_mask"]
        step_hint_mask = (
            teacher_signal_batch.batch["step_hint_mask"]
            if "step_hint_mask" in teacher_signal_batch.batch.keys()
            else torch.zeros_like(critical_step_mask, dtype=torch.bool)
        )
        if "teacher_signal_mask" in teacher_signal_batch.batch.keys():
            teacher_signal_mask = teacher_signal_batch.batch["teacher_signal_mask"]
        else:
            teacher_signal_mask = critical_step_mask

        if source_indices is not None:
            gather_idx = torch.as_tensor(
                np.asarray(source_indices, dtype=np.int64),
                dtype=torch.long,
                device=teacher_log_prob.device,
            )
            teacher_log_prob = teacher_log_prob.index_select(0, gather_idx)
            episode_teacher_log_prob = episode_teacher_log_prob.index_select(0, gather_idx)
            step_teacher_log_prob = step_teacher_log_prob.index_select(0, gather_idx)
            critical_step_mask = critical_step_mask.index_select(0, gather_idx)
            step_hint_mask = step_hint_mask.index_select(0, gather_idx)
            teacher_signal_mask = teacher_signal_mask.index_select(0, gather_idx)

        batch.batch["teacher_log_prob"] = teacher_log_prob
        batch.batch["episode_teacher_log_prob"] = episode_teacher_log_prob
        batch.batch["step_teacher_log_prob"] = step_teacher_log_prob
        batch.batch["critical_step_mask"] = critical_step_mask
        batch.batch["step_hint_mask"] = step_hint_mask
        batch.batch["teacher_signal_mask"] = teacher_signal_mask
        batch.non_tensor_batch.pop("_batch_source_idx", None)
        return batch

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        phase_switch_after_steps = OmegaConf.select(config, "algorithm.copd.phase_switch_after_steps")
        opd_start_after_steps = OmegaConf.select(config, "algorithm.copd.opd_start_after_steps")
        legacy_teacher_advantage_start_after_steps = OmegaConf.select(
            config,
            "algorithm.copd.teacher_advantage_start_after_steps",
        )
        if opd_start_after_steps is None:
            opd_start_after_steps = legacy_teacher_advantage_start_after_steps
        elif legacy_teacher_advantage_start_after_steps is not None and int(legacy_teacher_advantage_start_after_steps) != int(opd_start_after_steps):
            module_logger.warning(
                "Both algorithm.copd.opd_start_after_steps=%s and legacy algorithm.copd.teacher_advantage_start_after_steps=%s are set; using opd_start_after_steps.",
                opd_start_after_steps,
                legacy_teacher_advantage_start_after_steps,
            )
        failed_only_after_steps = OmegaConf.select(config, "algorithm.copd.failed_only_after_steps")
        if phase_switch_after_steps is not None and int(phase_switch_after_steps) < 0:
            raise ValueError("algorithm.copd.phase_switch_after_steps must be null or a non-negative integer.")
        if opd_start_after_steps is not None and int(opd_start_after_steps) < 0:
            raise ValueError("algorithm.copd.opd_start_after_steps must be null or a non-negative integer.")
        if failed_only_after_steps is not None and int(failed_only_after_steps) < 0:
            raise ValueError("algorithm.copd.failed_only_after_steps must be null or a non-negative integer.")
        if failed_only_after_steps is not None and bool(OmegaConf.select(config, "algorithm.copd.failed_only")):
            module_logger.warning(
                "algorithm.copd.failed_only_after_steps=%s is set, so scheduled all-then-failed analysis overrides algorithm.copd.failed_only=True until after that step.",
                failed_only_after_steps,
            )
        teacher_advantage_w = float(OmegaConf.select(config, "algorithm.copd.teacher_advantage_w") or 0.0)
        episode_hint_teacher_advantage_w = OmegaConf.select(config, "algorithm.copd.episode_hint_teacher_advantage_w")
        step_hint_teacher_advantage_w = float(
            OmegaConf.select(config, "algorithm.copd.step_hint_teacher_advantage_w") or 0.0
        )
        teacher_advantage_w_start = OmegaConf.select(config, "algorithm.copd.teacher_advantage_w_start")
        teacher_advantage_w_ramp_steps = OmegaConf.select(config, "algorithm.copd.teacher_advantage_w_ramp_steps")
        if teacher_advantage_w < 0:
            raise ValueError("algorithm.copd.teacher_advantage_w must be non-negative.")
        if episode_hint_teacher_advantage_w is not None and float(episode_hint_teacher_advantage_w) < 0:
            raise ValueError("algorithm.copd.episode_hint_teacher_advantage_w must be null or a non-negative number.")
        if step_hint_teacher_advantage_w < 0:
            raise ValueError("algorithm.copd.step_hint_teacher_advantage_w must be non-negative.")
        if teacher_advantage_w_start is not None and float(teacher_advantage_w_start) < 0:
            raise ValueError("algorithm.copd.teacher_advantage_w_start must be null or a non-negative number.")
        if teacher_advantage_w_start is None and teacher_advantage_w_ramp_steps is not None:
            module_logger.warning(
                "algorithm.copd.teacher_advantage_w_ramp_steps is set but algorithm.copd.teacher_advantage_w_start is null, so teacher_advantage_w will stay constant."
            )
        if teacher_advantage_w_ramp_steps is not None and int(teacher_advantage_w_ramp_steps) <= 0:
            raise ValueError("algorithm.copd.teacher_advantage_w_ramp_steps must be null or a positive integer.")
        if config.algorithm.adv_estimator == AdvantageEstimator.COPD or str(config.algorithm.adv_estimator) == AdvantageEstimator.COPD.value:
            if float(OmegaConf.select(config, "algorithm.copd.step_advantage_w") or 0.0) != 0.0:
                raise ValueError("Episode-level COPD OPD requires algorithm.copd.step_advantage_w=0.0.")
            if str(OmegaConf.select(config, "algorithm.copd.selector")) != "llm":
                raise ValueError("Episode-level COPD OPD requires algorithm.copd.selector=llm.")
        if teacher_advantage_w_start is not None and phase_switch_after_steps is not None:
            teacher_start_after_steps = int(opd_start_after_steps or 0)
            teacher_enabled_steps_before_phase_switch = int(phase_switch_after_steps) - teacher_start_after_steps
            if teacher_enabled_steps_before_phase_switch <= 0:
                module_logger.warning(
                    "algorithm.copd.teacher_advantage_w_start is set, but teacher advantage is never enabled because opd_start_after_steps=%s and phase_switch_after_steps=%s. The teacher weight ramp will therefore never take effect.",
                    teacher_start_after_steps,
                    phase_switch_after_steps,
                )
            elif teacher_advantage_w_ramp_steps is not None and teacher_enabled_steps_before_phase_switch < int(teacher_advantage_w_ramp_steps):
                module_logger.warning(
                    "algorithm.copd.teacher_advantage_w_ramp_steps=%s is longer than the teacher-enabled phase (%s steps). The teacher weight ramp will stop early at the phase switch.",
                    teacher_advantage_w_ramp_steps,
                    teacher_enabled_steps_before_phase_switch,
                )
        elif opd_start_after_steps is not None and phase_switch_after_steps is not None:
            if int(opd_start_after_steps) >= int(phase_switch_after_steps):
                module_logger.warning(
                    "algorithm.copd.opd_start_after_steps=%s is not earlier than phase_switch_after_steps=%s, so teacher advantage will never be enabled.",
                    opd_start_after_steps,
                    phase_switch_after_steps,
                )

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
        rollout_extra_infos_dict=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        def _json_safe(value):
            if isinstance(value, np.generic):
                return value.item()
            if torch.is_tensor(value):
                return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
            return value

        def _sort_key(entry):
            try:
                return (
                    int(entry.get("sample_id", 0)),
                    int(entry.get("rollout_id", 0)),
                    int(entry.get("step_num", 0)),
                )
            except (TypeError, ValueError):
                step_id = str(entry.get("step_id", "0_0_0")).split("_")
                padded = (step_id + ["0", "0", "0"])[:3]
                try:
                    return tuple(int(part) for part in padded)
                except ValueError:
                    return (0, 0, 0)

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        rollout_extra_infos_dict = rollout_extra_infos_dict or {}
        for k, v in rollout_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        if all(key in base_data for key in ("sample_id", "rollout_id", "step_num")) and "step_id" not in base_data:
            base_data["step_id"] = [
                f"{int(base_data['sample_id'][i])}_{int(base_data['rollout_id'][i])}_{int(base_data['step_num'][i])}"
                for i in range(n)
            ]

        entries = []
        for i in range(n):
            entries.append({k: _json_safe(v[i]) for k, v in base_data.items()})
        entries.sort(key=_sort_key)

        with open(filename, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _get_copd_analysis_dump_dir(self) -> Optional[str]:
        save_analysis = OmegaConf.select(self.config, "algorithm.copd.save_analysis")
        if not save_analysis:
            return None

        dump_dir = OmegaConf.select(self.config, "algorithm.copd.analysis_dump_dir")
        if dump_dir:
            return dump_dir

        return os.path.join(self.config.trainer.default_local_dir, "copd_analysis")

    def _dump_copd_analysis(
        self,
        analysis_tasks: Dict[object, Dict[str, object]],
        episode_analysis: Dict[object, Dict[str, object]],
        selector: str,
    ) -> None:
        dump_dir = self._get_copd_analysis_dump_dir()
        if dump_dir is None or not analysis_tasks:
            return

        os.makedirs(dump_dir, exist_ok=True)
        filename = os.path.join(dump_dir, f"step_{self.global_steps:08d}.jsonl")

        with open(filename, "w", encoding="utf-8") as f:
            for traj_uid, task in analysis_tasks.items():
                analysis = episode_analysis.get(traj_uid, {})
                entry = {
                    "global_step": int(self.global_steps),
                    "traj_uid": str(traj_uid),
                    "selector": selector,
                    "analysis_mode": analysis.get("analysis_mode"),
                    "analysis_backend_requested": analysis.get(
                        "analysis_backend_requested",
                        self.config.algorithm.copd.analysis_backend,
                    ),
                    "analysis_backend_used": analysis.get(
                        "analysis_backend_used",
                        self.config.algorithm.copd.analysis_backend,
                    ),
                    "analysis_error": analysis.get("analysis_error"),
                    "episode_success": task.get("episode_success"),
                    "candidate_step_indices": task.get("candidate_step_indices"),
                    "num_steps": len(task.get("steps", [])),
                    "step_indices": [
                        int(step.get("step_index", -1))
                        for step in task.get("steps", [])
                    ],
                    "episode_summary": str(analysis.get("episode_summary", "")),
                    "episode_hint": str(analysis.get("episode_hint", "")),
                    "step_hints": {
                        str(step_idx): str(hint)
                        for step_idx, hint in analysis.get("step_hints", {}).items()
                    },
                    "llm_prompt": analysis.get("llm_prompt"),
                    "llm_raw_output": analysis.get("llm_raw_output"),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        module_logger.info("Dumped COPD analysis results to %s", filename)

    def _get_copd_augmented_observation_dump_dir(self) -> Optional[str]:
        save_augmented_observations = OmegaConf.select(
            self.config,
            "algorithm.copd.save_augmented_observations",
        )
        if not save_augmented_observations:
            return None

        dump_dir = OmegaConf.select(
            self.config,
            "algorithm.copd.augmented_observation_dump_dir",
        )
        if dump_dir:
            return dump_dir

        return os.path.join(
            self.config.trainer.default_local_dir,
            "copd_augmented_observations",
        )

    def _dump_copd_augmented_observations(self, entries: List[Dict[str, object]]) -> None:
        dump_dir = self._get_copd_augmented_observation_dump_dir()
        if dump_dir is None or not entries:
            return

        os.makedirs(dump_dir, exist_ok=True)
        filename = os.path.join(dump_dir, f"step_{self.global_steps:08d}.jsonl")
        with open(filename, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        module_logger.info("Dumped COPD augmented observations to %s", filename)

    def _get_copd_state_group_dump_dir(self) -> Optional[str]:
        save_state_group_metrics = OmegaConf.select(
            self.config,
            "algorithm.copd.save_state_group_metrics",
        )
        if save_state_group_metrics is False:
            return None

        dump_dir = OmegaConf.select(
            self.config,
            "algorithm.copd.state_group_dump_dir",
        )
        if dump_dir:
            return dump_dir

        return os.path.join(
            self.config.trainer.default_local_dir,
            "copd_state_group",
        )

    @staticmethod
    def _metric_value_to_json(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                return tensor.item()
            return tensor.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                str(key): RayPPOTrainer._metric_value_to_json(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [RayPPOTrainer._metric_value_to_json(item) for item in value]
        return str(value)

    @staticmethod
    def _extract_copd_state_group_histogram(
        state_group_metrics: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        histogram = []
        for key, value in state_group_metrics.items():
            if not key.startswith(COPD_STATE_GROUP_METRIC_PREFIX):
                continue
            metric_name = key[len(COPD_STATE_GROUP_METRIC_PREFIX):]
            if not metric_name.startswith("size_") or not metric_name.endswith("_group_count"):
                continue
            label = metric_name[len("size_"):-len("_group_count")]
            count = float(value)
            group_prop = float(
                state_group_metrics.get(
                    f"{COPD_STATE_GROUP_METRIC_PREFIX}size_{label}_group_prop",
                    0.0,
                )
            )
            sample_prop = float(
                state_group_metrics.get(
                    f"{COPD_STATE_GROUP_METRIC_PREFIX}size_{label}_sample_prop",
                    0.0,
                )
            )
            histogram.append(
                {
                    "label": f">{label[3:]}" if label.startswith("gt_") else label,
                    "group_count": count,
                    "group_prop": group_prop,
                    "sample_prop": sample_prop,
                }
            )

        def _sort_key(item: Dict[str, float]) -> float:
            label = str(item["label"])
            if label.startswith(">"):
                return float(label[1:]) + 0.5
            return float(label)

        histogram.sort(key=_sort_key)
        return histogram

    @staticmethod
    def _build_copd_state_group_svg(
        *,
        global_step: int,
        histogram: List[Dict[str, float]],
        summary: Dict[str, Any],
    ) -> str:
        width = 960
        height = 520
        margin_left = 72
        margin_right = 32
        margin_top = 74
        margin_bottom = 82
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        max_count = max((float(item["group_count"]) for item in histogram), default=0.0)
        y_max = max(max_count, 1.0)
        bar_gap = 12
        bar_width = (
            (plot_width - bar_gap * max(len(histogram) - 1, 0)) / max(len(histogram), 1)
        )
        title = f"COPD State Group Size Distribution - Step {global_step}"
        subtitle = (
            f"groups={summary.get('num_groups', 0)}, samples={summary.get('num_samples', 0)}, "
            f"mean={float(summary.get('mean', 0.0)):.2f}, std={float(summary.get('std', 0.0)):.2f}"
        )

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{margin_left}" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">{title}</text>',
            f'<text x="{margin_left}" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{subtitle}</text>',
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1"/>',
        ]

        for tick_idx in range(5):
            ratio = tick_idx / 4
            y = margin_top + plot_height - ratio * plot_height
            value = y_max * ratio
            parts.append(
                f'<line x1="{margin_left - 4}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{value:.0f}</text>'
            )

        for idx, item in enumerate(histogram):
            x = margin_left + idx * (bar_width + bar_gap)
            count = float(item["group_count"])
            bar_height = 0.0 if y_max <= 0 else (count / y_max) * plot_height
            y = margin_top + plot_height - bar_height
            label = str(item["label"])
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{max(y - 6, margin_top + 12):.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#111827">{count:.0f}</text>'
            )
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{margin_top + plot_height + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{label}</text>'
            )

        parts.append(
            f'<text x="{margin_left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">Group size bucket</text>'
        )
        parts.append(
            f'<text x="20" y="{margin_top + plot_height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151" transform="rotate(-90 20 {margin_top + plot_height / 2})">Number of groups</text>'
        )
        parts.append("</svg>")
        return "\n".join(parts)

    def _dump_and_remove_copd_state_group_metrics(self, metrics: Dict[str, Any]) -> None:
        state_group_metrics = {
            key: metrics.pop(key)
            for key in list(metrics.keys())
            if key.startswith(COPD_STATE_GROUP_METRIC_PREFIX)
        }
        if not state_group_metrics:
            return

        dump_dir = self._get_copd_state_group_dump_dir()
        if dump_dir is None:
            return

        os.makedirs(dump_dir, exist_ok=True)
        raw_metrics = {
            key: self._metric_value_to_json(value)
            for key, value in sorted(state_group_metrics.items())
        }
        histogram = self._extract_copd_state_group_histogram(raw_metrics)
        summary = {
            key[len(COPD_STATE_GROUP_METRIC_PREFIX):]: value
            for key, value in raw_metrics.items()
            if key.startswith(COPD_STATE_GROUP_METRIC_PREFIX)
            and not key[len(COPD_STATE_GROUP_METRIC_PREFIX):].startswith("size_")
            and not key[len(COPD_STATE_GROUP_METRIC_PREFIX):].startswith("raw_")
        }
        payload = {
            "global_step": int(self.global_steps),
            "raw_metrics": raw_metrics,
            "raw_group_sizes": raw_metrics.get(
                f"{COPD_STATE_GROUP_METRIC_PREFIX}raw_group_sizes",
                [],
            ),
            "histogram": histogram,
            "summary": summary,
        }

        json_path = os.path.join(dump_dir, f"step_{self.global_steps:08d}.json")
        svg_path = os.path.join(dump_dir, f"step_{self.global_steps:08d}.svg")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(
                self._build_copd_state_group_svg(
                    global_step=int(self.global_steps),
                    histogram=histogram,
                    summary=summary,
                )
            )

        module_logger.info("Dumped COPD state-group metrics to %s and %s", json_path, svg_path)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _lazy_init_copd_analyzer(self):
        if self._copd_analyzer is None:
            max_step_hints_per_traj = OmegaConf.select(
                self.config,
                "algorithm.copd.analysis_max_step_hints_per_traj",
            )
            if max_step_hints_per_traj is None:
                max_step_hints_per_traj = 1
            module_logger.info(
                "Initializing COPD analyzer with backend=%s, max_completion_tokens=%s, max_step_hints_per_traj=%s",
                self.config.algorithm.copd.analysis_backend,
                self.config.algorithm.copd.analysis_max_completion_tokens,
                max_step_hints_per_traj,
            )
            self._copd_analyzer = core_copd.COPDEpisodeAnalyzer(
                backend=self.config.algorithm.copd.analysis_backend,
                max_completion_tokens=self.config.algorithm.copd.analysis_max_completion_tokens,
                max_step_hints_per_traj=max_step_hints_per_traj,
            )
        return self._copd_analyzer

    def _get_copd_failure_success_threshold(self) -> float:
        threshold = OmegaConf.select(self.config, "algorithm.copd.failure_success_threshold")
        return 1.0 if threshold is None else float(threshold)

    def _build_copd_traj_success_map(self, batch: DataProto) -> Dict[object, float]:
        if "episode_success" in batch.non_tensor_batch:
            episode_success_np = np.asarray(batch.non_tensor_batch["episode_success"], dtype=np.float32)
        elif "episode_rewards" in batch.non_tensor_batch:
            threshold = self._get_copd_failure_success_threshold()
            episode_success_np = (
                np.asarray(batch.non_tensor_batch["episode_rewards"], dtype=np.float32) >= threshold
            ).astype(np.float32)
        else:
            return {}

        traj_success: Dict[object, float] = {}
        for sample_idx, traj_uid in enumerate(batch.non_tensor_batch["traj_uid"]):
            if traj_uid not in traj_success:
                traj_success[traj_uid] = float(episode_success_np[sample_idx])
        return traj_success

    def _analyze_copd_episodes(self, analyzer, analysis_tasks: Dict[object, Dict[str, object]]):
        """
        Analyze multiple trajectories for COPD. Azure-backed analysis is run with
        a thread pool to hide per-request latency.
        """
        if not analysis_tasks:
            return {}, 0

        backend = self.config.algorithm.copd.analysis_backend
        configured_workers = int(self.config.algorithm.copd.analysis_num_workers)
        max_workers = max(1, min(configured_workers, len(analysis_tasks)))

        module_logger.info(
            "Running COPD episode analysis for %s trajectories with backend=%s and num_workers=%s",
            len(analysis_tasks),
            backend,
            max_workers,
        )
        print(f"COPD analysis backend: {backend}, configured_workers: {configured_workers}, max_workers: {max_workers}")

        assert backend in ["openai", "heuristic"], f"Unsupported COPD analysis backend: {backend}"

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="copd-analysis") as executor:
            future_to_traj = {
                executor.submit(
                    analyzer.analyze_episode,
                    steps=task["steps"],
                    candidate_step_indices=task["candidate_step_indices"],
                    analysis_mode=task["analysis_mode"],
                    episode_success=task.get("episode_success"),
                ): traj_uid
                for traj_uid, task in analysis_tasks.items()
            }
            for future in as_completed(future_to_traj):
                traj_uid = future_to_traj[future]
                try:
                    results[traj_uid] = future.result()
                except Exception as exc:
                    task = analysis_tasks.get(traj_uid, {})
                    module_logger.warning(
                        "COPD episode analysis failed for traj_uid=%s; skipping teacher signal for this trajectory: %s",
                        traj_uid,
                        exc,
                    )
                    results[traj_uid] = {
                        "episode_summary": "",
                        "episode_hint": "",
                        "step_hints": {},
                        "analysis_backend_requested": backend,
                        "analysis_backend_used": backend,
                        "analysis_error": str(exc),
                        "analysis_mode": task.get("analysis_mode"),
                        "task_description": "",
                        "llm_prompt": None,
                        "llm_raw_output": None,
                    }
        return results, max_workers

    def _prepare_copd_teacher_signals(
        self,
        batch: DataProto,
        metrics: Dict[str, float],
        teacher_enabled: bool,
    ) -> DataProto:
        """
        Run COPD analysis for the current batch and optionally compute teacher
        log-probs during the bootstrap phase.
        """
        batch_size = len(batch)
        response_mask = compute_response_mask(batch)
        zero_teacher_log_prob = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        zero_critical_mask = torch.zeros(batch_size, dtype=torch.bool, device=batch.batch["responses"].device)
        zero_step_hint_mask = torch.zeros(batch_size, dtype=torch.bool, device=batch.batch["responses"].device)
        batch.batch["teacher_signal_mask"] = zero_critical_mask.clone()
        traj_uids = batch.non_tensor_batch.get("traj_uid", [])
        num_trajectories = len(set(traj_uids)) if len(traj_uids) > 0 else 0

        if not self._is_copd_analysis_enabled():
            module_logger.info(
                "Skipping COPD analysis and teacher signal construction for batch_size=%s, num_trajectories=%s.",
                batch_size,
                num_trajectories,
            )
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["episode_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["step_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["critical_step_mask"] = zero_critical_mask
            batch.batch["step_hint_mask"] = zero_step_hint_mask
            metrics["copd/analysis_enabled"] = 0.0
            metrics["copd/analysis_disabled"] = 1.0
            metrics["copd/analysis_num_requests"] = 0.0
            metrics["copd/analysis_num_workers"] = 0.0
            metrics["copd/critical_step_ratio"] = 0.0
            metrics["copd/teacher_batch_size"] = 0.0
            metrics["copd/teacher_available"] = 0.0
            metrics["copd/teacher_skipped_analysis_disabled"] = 1.0
            metrics["copd/episode_hint_guidance/enabled"] = 0.0
            metrics["copd/step_hint_guidance/step_guided_step_ratio"] = 0.0
            metrics["copd/step_hint_guidance/step_hints_applied"] = 0.0
            return batch

        metrics["copd/analysis_enabled"] = 1.0
        metrics["copd/analysis_disabled"] = 0.0
        metrics["copd/episode_hint_guidance/enabled"] = 1.0
        configured_failed_only = bool(OmegaConf.select(self.config, "algorithm.copd.failed_only"))
        failed_only_after_steps = self._get_copd_failed_only_after_steps()
        failed_only = self._should_copd_analyze_failed_only()
        analysis_mode = "failed_episode_opd" if failed_only else "teacher_bootstrap"

        module_logger.info(
            "Preparing COPD analysis for batch_size=%s, num_trajectories=%s, selector=%s, analysis_backend=%s, analysis_mode=%s, failed_only_after_steps=%s",
            batch_size,
            num_trajectories,
            self.config.algorithm.copd.selector,
            self.config.algorithm.copd.analysis_backend,
            analysis_mode,
            failed_only_after_steps,
        )
        metrics["copd/analysis_mode_teacher_bootstrap"] = 0.0 if failed_only else 1.0
        metrics["copd/analysis_mode_failed_episode_opd"] = 1.0 if failed_only else 0.0

        if "obs_text" not in batch.non_tensor_batch:
            module_logger.warning("COPD teacher signal skipped because obs_text is missing from the rollout batch.")
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["episode_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["step_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["critical_step_mask"] = zero_critical_mask
            batch.batch["step_hint_mask"] = zero_step_hint_mask
            metrics["copd/critical_step_ratio"] = 0.0
            metrics["copd/teacher_batch_size"] = 0.0
            metrics["copd/teacher_available"] = 0.0
            metrics["copd/episode_hint_guidance/enabled"] = 0.0
            metrics["copd/step_hint_guidance/step_guided_step_ratio"] = 0.0
            metrics["copd/step_hint_guidance/step_hints_applied"] = 0.0
            return batch

        step_indices = core_copd.build_traj_step_indices(batch.non_tensor_batch["traj_uid"])
        batch.non_tensor_batch["step_idx"] = step_indices

        selector = self.config.algorithm.copd.selector
        analyzer = self._lazy_init_copd_analyzer()
        metrics["copd/failed_only"] = 1.0 if failed_only else 0.0
        metrics["copd/failed_only_config"] = 1.0 if configured_failed_only else 0.0
        metrics["copd/failed_only_after_steps"] = (
            float(failed_only_after_steps) if failed_only_after_steps is not None else -1.0
        )
        metrics["copd/failed_only_schedule_active"] = 1.0 if failed_only_after_steps is not None else 0.0
        metrics["copd/failure_success_threshold"] = self._get_copd_failure_success_threshold()
        critical_mask_np = np.zeros(batch_size, dtype=bool)
        base_obs_texts = batch.non_tensor_batch.get("obs_text_base", batch.non_tensor_batch["obs_text"])
        analysis_obs_texts = base_obs_texts

        episodes = core_copd.build_episode_records(
            tokenizer=self.tokenizer,
            obs_texts=analysis_obs_texts,
            obs_raws=batch.non_tensor_batch.get("anchor_obs"),
            responses=batch.batch["responses"],
            response_mask=response_mask,
            traj_index=batch.non_tensor_batch["traj_uid"],
            step_indices=step_indices,
            step_rewards=batch.batch["step_rewards"] if "step_rewards" in batch.batch.keys() else None,
        )
        if episodes:
            episode_lengths = [len(steps) for steps in episodes.values()]
            module_logger.info(
                "Built COPD episode records for %s trajectories (min_steps=%s, mean_steps=%.2f, max_steps=%s)",
                len(episodes),
                min(episode_lengths),
                float(np.mean(episode_lengths)),
                max(episode_lengths),
            )
        else:
            module_logger.info("No COPD episode records were built for the current batch.")

        episode_analysis: Dict[object, Dict[str, object]] = {}
        analysis_tasks: Dict[object, Dict[str, object]] = {}
        traj_success = self._build_copd_traj_success_map(batch)
        if failed_only and not traj_success:
            module_logger.warning(
                "COPD failed_only is enabled, but episode_success/episode_rewards are missing; analyzing all trajectories."
            )

        def _should_analyze_traj(traj_uid: object) -> bool:
            if not failed_only:
                return True
            success_value = traj_success.get(traj_uid)
            if success_value is None:
                return True
            return success_value < 1.0

        analyzed_traj_count = float(sum(1 for traj_uid in episodes if _should_analyze_traj(traj_uid)))
        metrics["copd/analyzed_traj_count"] = analyzed_traj_count
        metrics["copd/failed_traj_count"] = analyzed_traj_count
        metrics["copd/skipped_success_traj_count"] = float(max(len(episodes) - int(analyzed_traj_count), 0))
        if selector != "llm":
            raise ValueError("Episode-level COPD OPD requires algorithm.copd.selector=llm.")

        module_logger.info(
            "COPD LLM analyzer will build episode-level teacher guidance for %s/%s trajectories.",
            int(analyzed_traj_count),
            len(episodes),
        )
        for traj_uid, steps in episodes.items():
            if not _should_analyze_traj(traj_uid):
                continue
            analysis_tasks[traj_uid] = {
                "steps": steps,
                "candidate_step_indices": [int(step["step_index"]) for step in steps],
                "analysis_mode": analysis_mode,
                "episode_success": traj_success.get(traj_uid),
            }
        analyzed, analysis_workers = self._analyze_copd_episodes(analyzer=analyzer, analysis_tasks=analysis_tasks)
        episode_analysis.update(analyzed)
        successful_episode_analysis = {
            traj_uid: analysis
            for traj_uid, analysis in episode_analysis.items()
            if not analysis.get("analysis_error") and str(analysis.get("episode_hint", "")).strip()
        }
        failed_analysis_count = len(episode_analysis) - len(successful_episode_analysis)

        critical_mask_np = np.zeros(batch_size, dtype=bool)
        analyzed_traj_uids = set(successful_episode_analysis.keys())
        for sample_idx, sample_traj_uid in enumerate(batch.non_tensor_batch["traj_uid"]):
            if sample_traj_uid in analyzed_traj_uids:
                critical_mask_np[sample_idx] = True
        module_logger.info(
            "COPD episode-level OPD selected %s steps from %s successful analyzed trajectories (%s failed/skipped).",
            int(critical_mask_np.sum()),
            len(analyzed_traj_uids),
            failed_analysis_count,
        )
        metrics["copd/analysis_num_requests"] = float(len(analysis_tasks))
        metrics["copd/analysis_num_workers"] = float(analysis_workers)
        metrics["copd/analysis_succeeded_traj_count"] = float(len(successful_episode_analysis))
        metrics["copd/analysis_failed_traj_count"] = float(failed_analysis_count)
        module_logger.info(
            "COPD episode analysis finished: requests=%s, workers=%s, successful_trajectories=%s, failed_trajectories=%s",
            len(analysis_tasks),
            analysis_workers,
            len(successful_episode_analysis),
            failed_analysis_count,
        )
        self._dump_copd_analysis(
            analysis_tasks=analysis_tasks,
            episode_analysis=episode_analysis,
            selector=selector,
        )

        critical_indices = np.where(critical_mask_np)[0]
        critical_mask = torch.as_tensor(
            critical_mask_np,
            device=batch.batch["responses"].device,
            dtype=torch.bool,
        )
        batch.batch["critical_step_mask"] = critical_mask

        metrics["copd/critical_step_ratio"] = float(critical_mask_np.mean()) if batch_size > 0 else 0.0
        metrics["copd/teacher_batch_size"] = float(len(critical_indices))
        metrics["copd/teacher_available"] = 1.0
        metrics["copd/step_hint_guidance/step_guided_step_ratio"] = 0.0
        metrics["copd/step_hint_guidance/step_hints_applied"] = 0.0
        module_logger.info(
            "COPD finalized %s OPD-scored steps (step_ratio=%.4f, analysis_mode=%s).",
            len(critical_indices),
            metrics["copd/critical_step_ratio"],
            analysis_mode,
        )

        if not teacher_enabled:
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["episode_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["step_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["critical_step_mask"] = critical_mask
            batch.batch["step_hint_mask"] = zero_step_hint_mask
            metrics["copd/teacher_batch_size"] = 0.0
            metrics["copd/teacher_available"] = 0.0
            teacher_start_after_steps = self._get_copd_opd_start_after_steps()
            phase_switch_after_steps = self._get_copd_phase_switch_after_steps()
            metrics["copd/teacher_skipped_by_schedule"] = 1.0
            metrics["copd/teacher_skipped_before_start"] = (
                1.0
                if teacher_start_after_steps is not None and self.global_steps <= teacher_start_after_steps
                else 0.0
            )
            metrics["copd/teacher_skipped_after_phase_switch"] = (
                1.0
                if phase_switch_after_steps is not None and self.global_steps > phase_switch_after_steps
                else 0.0
            )
            return batch

        if len(critical_indices) == 0:
            module_logger.info("COPD has no episode-level OPD steps for the current batch.")
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["episode_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["step_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["step_hint_mask"] = zero_step_hint_mask
            metrics["copd/teacher_available"] = 0.0
            return batch

        if "multi_modal_inputs" in batch.non_tensor_batch:
            module_logger.warning("COPD teacher signal skipped for the current batch because multi_modal_inputs are present.")
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["episode_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["step_teacher_log_prob"] = zero_teacher_log_prob.clone()
            batch.batch["critical_step_mask"] = zero_critical_mask
            batch.batch["step_hint_mask"] = zero_step_hint_mask
            batch.batch["teacher_signal_mask"] = zero_critical_mask
            metrics["copd/teacher_skipped_multimodal"] = 1.0
            return batch

        episode_obs_texts = []
        episode_data_sources = []
        step_obs_texts = []
        step_data_sources = []
        step_hint_indices = []
        augmented_observation_dump_entries: List[Dict[str, object]] = []
        critical_response_mask = response_mask[critical_indices]
        critical_responses = batch.batch["responses"][critical_indices]
        critical_preview = []
        step_hint_guided_steps = 0
        step_hint_teacher_enabled = (
            float(OmegaConf.select(self.config, "algorithm.copd.step_hint_teacher_advantage_w") or 0.0) > 0.0
        )

        for sample_idx in critical_indices:
            traj_uid = batch.non_tensor_batch["traj_uid"][sample_idx]
            analysis = episode_analysis[traj_uid]
            step_idx = int(step_indices[sample_idx])
            observation_text = str(base_obs_texts[sample_idx])
            episode_summary = str(analysis.get("episode_summary", ""))
            episode_hint = str(analysis["episode_hint"])
            step_hint = str(analysis.get("step_hints", {}).get(step_idx, ""))
            data_source = (
                batch.non_tensor_batch["data_source"][sample_idx]
                if "data_source" in batch.non_tensor_batch
                else None
            )
            episode_enhanced_obs = build_augmented_observation_text(
                observation=observation_text,
                episode_hint=episode_hint,
            )
            episode_obs_texts.append(episode_enhanced_obs)
            episode_data_sources.append(data_source)
            step_enhanced_obs = ""
            if step_hint.strip() and step_hint_teacher_enabled:
                step_hint_guided_steps += 1
                step_enhanced_obs = build_augmented_observation_text(
                    observation=observation_text,
                    step_hint=step_hint,
                )
                step_obs_texts.append(step_enhanced_obs)
                step_data_sources.append(data_source)
                step_hint_indices.append(int(sample_idx))
            augmented_observation_dump_entries.append(
                {
                    "global_step": int(self.global_steps),
                    "sample_idx": int(sample_idx),
                    "traj_uid": str(traj_uid),
                    "step_idx": step_idx,
                    "analysis_mode": analysis.get("analysis_mode"),
                    "observation": observation_text,
                    "augmented_observation": episode_enhanced_obs,
                    "episode_augmented_observation": episode_enhanced_obs,
                    "step_augmented_observation": step_enhanced_obs,
                    "episode_summary": episode_summary,
                    "episode_hint": episode_hint,
                    "step_hint": step_hint,
                }
            )
            if len(critical_preview) < 3:
                critical_preview.append(
                    {
                        "traj_uid": str(traj_uid),
                        "step_idx": step_idx,
                        "hint_preview": episode_hint.replace("\n", " ")[:160],
                        "step_hint_preview": step_hint.replace("\n", " ")[:160],
                        "obs_preview": str(batch.non_tensor_batch["obs_text"][sample_idx]).replace("\n", " ")[:160],
                    }
                )

        if len(critical_indices) > 0:
            metrics["copd/step_hint_guidance/step_guided_step_ratio"] = float(
                step_hint_guided_steps / len(critical_indices)
            )
        metrics["copd/step_hint_guidance/step_hints_applied"] = float(step_hint_guided_steps)

        module_logger.info(
            "COPD built %s episode-hint and %s step-hint observations for teacher scoring across %s trajectories.",
            len(episode_obs_texts),
            len(step_obs_texts),
            len({batch.non_tensor_batch['traj_uid'][sample_idx] for sample_idx in critical_indices}),
        )
        if critical_preview and module_logger.isEnabledFor(logging.DEBUG):
            module_logger.debug("COPD episode-level OPD preview: %s", critical_preview)
        self._dump_copd_augmented_observations(augmented_observation_dump_entries)

        def _compute_hint_log_probs(
            *,
            label: str,
            obs_texts: List[str],
            responses: torch.Tensor,
            response_masks: torch.Tensor,
            data_sources: List[object],
        ) -> torch.Tensor:
            teacher_prompt_batch = self.traj_collector.build_text_prompt_batch(
                obs_contents=obs_texts,
                data_sources=data_sources,
                meta_info=deepcopy(batch.meta_info),
            )
            prompt_lengths = teacher_prompt_batch.batch["attention_mask"].sum(dim=-1).detach().cpu().numpy()
            module_logger.info(
                "COPD %s teacher prompt lengths: min=%s, mean=%.2f, max=%s",
                label,
                int(prompt_lengths.min()),
                float(prompt_lengths.mean()),
                int(prompt_lengths.max()),
            )

            teacher_input_ids = torch.cat([teacher_prompt_batch.batch["input_ids"], responses], dim=-1)
            teacher_attention_mask = torch.cat(
                [
                    teacher_prompt_batch.batch["attention_mask"],
                    response_masks.to(dtype=teacher_prompt_batch.batch["attention_mask"].dtype),
                ],
                dim=-1,
            )
            teacher_position_ids = torch.clip(torch.cumsum(teacher_attention_mask, dim=-1) - 1, min=0)
            teacher_batch = DataProto.from_dict(
                tensors={
                    "responses": responses,
                    "input_ids": teacher_input_ids,
                    "attention_mask": teacher_attention_mask,
                    "position_ids": teacher_position_ids,
                },
                meta_info=deepcopy(batch.meta_info),
            )
            teacher_batch_padded, teacher_pad_size = pad_dataproto_to_divisor(
                teacher_batch,
                self.actor_rollout_wg.world_size,
            )
            teacher_log_prob_padded = self.actor_rollout_wg.compute_log_prob(teacher_batch_padded)
            teacher_log_prob = unpad_dataproto(teacher_log_prob_padded, pad_size=teacher_pad_size)
            return teacher_log_prob.batch["old_log_probs"]

        episode_teacher_lp = _compute_hint_log_probs(
            label="episode-hint",
            obs_texts=episode_obs_texts,
            responses=critical_responses,
            response_masks=critical_response_mask,
            data_sources=episode_data_sources,
        )
        full_episode_teacher_log_prob = zero_teacher_log_prob.clone()
        full_episode_teacher_log_prob[critical_indices] = episode_teacher_lp
        full_step_teacher_log_prob = zero_teacher_log_prob.clone()
        step_hint_mask_np = np.zeros(batch_size, dtype=bool)
        metrics["copd/step_hint_teacher_log_prob_mean"] = 0.0
        if step_hint_indices:
            step_hint_mask_np[step_hint_indices] = True
            step_hint_tensor_indices = torch.as_tensor(
                step_hint_indices,
                dtype=torch.long,
                device=batch.batch["responses"].device,
            )
            step_teacher_lp = _compute_hint_log_probs(
                label="step-hint",
                obs_texts=step_obs_texts,
                responses=batch.batch["responses"].index_select(0, step_hint_tensor_indices),
                response_masks=response_mask.index_select(0, step_hint_tensor_indices),
                data_sources=step_data_sources,
            )
            full_step_teacher_log_prob[step_hint_indices] = step_teacher_lp
            metrics["copd/step_hint_teacher_log_prob_mean"] = float(
                step_teacher_lp.mean().detach().cpu().item()
            )

        step_hint_mask = torch.as_tensor(
            step_hint_mask_np,
            device=batch.batch["responses"].device,
            dtype=torch.bool,
        )
        batch.batch["teacher_log_prob"] = full_episode_teacher_log_prob
        batch.batch["episode_teacher_log_prob"] = full_episode_teacher_log_prob
        batch.batch["step_teacher_log_prob"] = full_step_teacher_log_prob
        batch.batch["step_hint_mask"] = step_hint_mask
        batch.batch["teacher_signal_mask"] = critical_mask

        module_logger.info(
            "COPD computed episode-hint teacher log-probs for %s OPD-scored steps (token_mean=%.6f, token_min=%.6f, token_max=%.6f).",
            len(critical_indices),
            float(episode_teacher_lp.mean().detach().cpu().item()),
            float(episode_teacher_lp.min().detach().cpu().item()),
            float(episode_teacher_lp.max().detach().cpu().item()),
        )
        if len(critical_indices) > 0:
            metrics["copd/teacher_log_prob_mean"] = float(
                episode_teacher_lp.mean().detach().cpu().item()
            )
            metrics["copd/episode_hint_teacher_log_prob_mean"] = metrics["copd/teacher_log_prob_mean"]
        return batch

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    copd_teacher_future = None
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.COPD:
                        copd_teacher_schedule_enabled = self._is_copd_teacher_adv_enabled()
                        copd_analysis_enabled = self._is_copd_analysis_enabled()
                        copd_teacher_adv_enabled = copd_teacher_schedule_enabled and copd_analysis_enabled
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator in [AdvantageEstimator.GiGPO, AdvantageEstimator.COPD]:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.COPD:
                        metrics["copd/teacher_enabled"] = 1.0 if copd_teacher_adv_enabled else 0.0
                        metrics["copd/teacher_disabled_by_schedule"] = 0.0 if copd_teacher_schedule_enabled else 1.0
                        metrics["copd/teacher_disabled_by_analysis"] = (
                            1.0 if copd_teacher_schedule_enabled and not copd_analysis_enabled else 0.0
                        )
                        metrics["copd/analysis_enabled"] = 1.0 if copd_analysis_enabled else 0.0
                        metrics["copd/analysis_disabled"] = 0.0 if copd_analysis_enabled else 1.0
                        if copd_analysis_enabled:
                            copd_teacher_future = self._lazy_init_copd_teacher_signal_executor().submit(
                                self._prepare_copd_teacher_signals_async_task,
                                self._build_copd_teacher_signal_snapshot(batch),
                                copd_teacher_adv_enabled,
                            )
                    
                    batch = adjust_batch(
                        self.config,
                        batch,
                        track_source_indices=self.config.algorithm.adv_estimator == AdvantageEstimator.COPD,
                    )

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.COPD:
                        with _timer("copd_teacher", timing_raw):
                            if copd_teacher_future is None:
                                module_logger.info(
                                    "COPD analysis is disabled; using zero teacher signals for this batch."
                                )
                                batch = self._set_zero_copd_teacher_signals(batch=batch, metrics=metrics)
                                metrics["copd/analysis_enabled"] = 0.0
                                metrics["copd/analysis_disabled"] = 1.0
                                metrics["copd/analysis_num_requests"] = 0.0
                                metrics["copd/analysis_num_workers"] = 0.0
                                metrics["copd/teacher_skipped_analysis_disabled"] = 1.0
                                batch.non_tensor_batch.pop("_batch_source_idx", None)
                            else:
                                try:
                                    teacher_signal_batch, teacher_signal_metrics = copd_teacher_future.result()
                                    metrics.update(teacher_signal_metrics)
                                    batch = self._merge_async_copd_teacher_signals(
                                        batch=batch,
                                        teacher_signal_batch=teacher_signal_batch,
                                    )
                                except Exception as exc:
                                    module_logger.warning(
                                        "Asynchronous COPD teacher signal preparation failed; falling back to zero teacher signals for this batch: %s",
                                        exc,
                                    )
                                    batch.non_tensor_batch.pop("_batch_source_idx", None)
                                    batch = self._set_zero_copd_teacher_signals(batch=batch, metrics=metrics)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.COPD:
                            copd_teacher_advantage_w = self._get_copd_teacher_advantage_weight(
                                teacher_enabled=copd_teacher_adv_enabled
                            )
                            metrics["copd/teacher_advantage_w_current"] = float(copd_teacher_advantage_w)
                            episode_hint_teacher_advantage_w = OmegaConf.select(
                                self.config,
                                "algorithm.copd.episode_hint_teacher_advantage_w",
                            )
                            if not copd_teacher_adv_enabled:
                                episode_hint_teacher_advantage_w = 0.0
                            elif episode_hint_teacher_advantage_w is not None:
                                episode_hint_teacher_advantage_w = float(episode_hint_teacher_advantage_w)
                            step_hint_teacher_advantage_w = (
                                float(OmegaConf.select(self.config, "algorithm.copd.step_hint_teacher_advantage_w") or 0.0)
                                if copd_teacher_adv_enabled
                                else 0.0
                            )
                            metrics["copd/episode_hint_teacher_advantage_w_current"] = float(
                                copd_teacher_advantage_w
                                if episode_hint_teacher_advantage_w is None
                                else episode_hint_teacher_advantage_w
                            )
                            metrics["copd/step_hint_teacher_advantage_w_current"] = float(step_hint_teacher_advantage_w)
                        else:
                            copd_teacher_advantage_w = 0.0
                            episode_hint_teacher_advantage_w = None
                            step_hint_teacher_advantage_w = 0.0

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=(
                                self.config.algorithm.copd.step_advantage_w
                                if self.config.algorithm.adv_estimator == AdvantageEstimator.COPD
                                else self.config.algorithm.gigpo.step_advantage_w
                            ),
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            teacher_advantage_w=copd_teacher_advantage_w,
                            episode_hint_teacher_advantage_w=episode_hint_teacher_advantage_w,
                            step_hint_teacher_advantage_w=step_hint_teacher_advantage_w,
                            copd_mode=self.config.algorithm.copd.mode,
                            copd_enable_similarity=self.config.algorithm.copd.enable_similarity,
                            copd_similarity_thresh=self.config.algorithm.copd.similarity_thresh,
                            copd_normalize_teacher_adv=self.config.algorithm.copd.normalize_teacher_adv,
                            copd_clip_teacher_adv=self.config.algorithm.copd.clip_teacher_adv,
                        )
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.COPD:
                            metrics.update(batch.meta_info.pop("copd_adv_metrics", {}))
                        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                            metrics.update(batch.meta_info.pop("gigpo_adv_metrics", {}))

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            rollout_extra_infos_dict = {
                                key: batch.non_tensor_batch[key]
                                for key in (
                                    "sample_id",
                                    "rollout_id",
                                    "step_num",
                                    "step_id",
                                    "uid",
                                    "traj_uid",
                                )
                                if key in batch.non_tensor_batch
                            }
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                rollout_extra_infos_dict=rollout_extra_infos_dict,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                self._dump_and_remove_copd_state_group_metrics(metrics)
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
