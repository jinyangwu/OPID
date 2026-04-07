import logging
from copy import deepcopy
from typing import Dict, Optional

import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask

from gigpo.skill_distill import SkillEnhancedTeacherContext


module_logger = logging.getLogger(__name__)


class SkillDistillRayPPOTrainer(RayPPOTrainer):
    """Teacher-enhanced COPD variant where the teacher context is built from retrieved skills."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skill_teacher_builder: Optional[SkillEnhancedTeacherContext] = None
        self._skill_teacher_disabled = False

    def _lazy_init_skill_teacher_builder(self) -> Optional[SkillEnhancedTeacherContext]:
        if self._skill_teacher_disabled:
            return None

        if self._skill_teacher_builder is None:
            try:
                self._skill_teacher_builder = SkillEnhancedTeacherContext(self.config)
            except Exception as exc:
                self._skill_teacher_disabled = True
                module_logger.warning("Skill distillation teacher disabled because initialization failed: %s", exc)
                return None

        return self._skill_teacher_builder

    def _prepare_copd_teacher_signals(self, batch: DataProto, metrics: Dict[str, float]) -> DataProto:
        """
        Prepare teacher log-probs using skill-enhanced prompts built from Claude-style skills.

        Unlike the base COPD implementation, this variant applies teacher scoring to
        every rollout step and does not require critical-step selection or hindsight
        analysis. The student response is re-scored under a skill-enhanced context,
        and that policy-on-enhanced-context log-prob becomes the teacher signal.
        """
        batch_size = len(batch)
        zero_teacher_log_prob = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        zero_critical_mask = torch.zeros(batch_size, dtype=torch.bool, device=batch.batch["responses"].device)

        if "obs_text" not in batch.non_tensor_batch:
            module_logger.warning("Skill distillation teacher skipped because obs_text is missing from the rollout batch.")
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["critical_step_mask"] = zero_critical_mask
            metrics["copd/critical_step_ratio"] = 0.0
            metrics["copd/teacher_batch_size"] = 0.0
            metrics["copd/teacher_available"] = 0.0
            return batch

        if "multi_modal_inputs" in batch.non_tensor_batch:
            module_logger.warning("Skill distillation teacher skipped because multi_modal_inputs are present.")
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["critical_step_mask"] = zero_critical_mask
            metrics["copd/critical_step_ratio"] = 0.0
            metrics["copd/teacher_batch_size"] = 0.0
            metrics["copd/teacher_available"] = 0.0
            metrics["skill_distill/teacher_skipped_multimodal"] = 1.0
            return batch

        teacher_builder = self._lazy_init_skill_teacher_builder()
        if teacher_builder is None:
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["critical_step_mask"] = zero_critical_mask
            metrics["copd/critical_step_ratio"] = 0.0
            metrics["copd/teacher_batch_size"] = 0.0
            metrics["copd/teacher_available"] = 0.0
            return batch

        try:
            response_mask = compute_response_mask(batch)
            obs_texts = [str(obs_text) for obs_text in batch.non_tensor_batch["obs_text"]]
            enhanced_obs_texts, skill_stats = teacher_builder.build_enhanced_observations(obs_texts)

            critical_mask = torch.ones(batch_size, dtype=torch.bool, device=batch.batch["responses"].device)
            batch.batch["critical_step_mask"] = critical_mask
            metrics["copd/critical_step_ratio"] = 1.0 if batch_size > 0 else 0.0
            metrics["copd/teacher_batch_size"] = float(batch_size)
            metrics["copd/teacher_available"] = 1.0
            metrics["skill_distill/num_unique_queries"] = skill_stats["num_unique_queries"]
            metrics["skill_distill/num_injected"] = skill_stats["num_injected"]
            metrics["skill_distill/injected_ratio"] = skill_stats["injected_ratio"]

            data_sources = (
                [
                    batch.non_tensor_batch["data_source"][sample_idx]
                    for sample_idx in range(batch_size)
                ]
                if "data_source" in batch.non_tensor_batch
                else None
            )

            teacher_prompt_batch = self.traj_collector.build_text_prompt_batch(
                obs_contents=enhanced_obs_texts,
                data_sources=data_sources,
                meta_info=deepcopy(batch.meta_info),
            )
            teacher_prompt_lengths = teacher_prompt_batch.batch["attention_mask"].sum(dim=-1).detach().cpu().numpy()
            metrics["skill_distill/teacher_prompt_length_mean"] = float(teacher_prompt_lengths.mean())

            teacher_input_ids = torch.cat([teacher_prompt_batch.batch["input_ids"], batch.batch["responses"]], dim=-1)
            teacher_attention_mask = torch.cat(
                [
                    teacher_prompt_batch.batch["attention_mask"],
                    response_mask.to(dtype=teacher_prompt_batch.batch["attention_mask"].dtype),
                ],
                dim=-1,
            )
            teacher_position_ids = torch.clip(torch.cumsum(teacher_attention_mask, dim=-1) - 1, min=0)

            teacher_batch = DataProto.from_dict(
                tensors={
                    "responses": batch.batch["responses"],
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

            batch.batch["teacher_log_prob"] = teacher_log_prob.batch["old_log_probs"]
            teacher_lp = teacher_log_prob.batch["old_log_probs"]
            metrics["copd/teacher_log_prob_mean"] = float(teacher_lp.mean().detach().cpu().item())
            return batch
        except Exception as exc:
            module_logger.warning("Skill distillation teacher skipped for the current batch due to error: %s", exc)
            batch.batch["teacher_log_prob"] = zero_teacher_log_prob
            batch.batch["critical_step_mask"] = zero_critical_mask
            metrics["copd/critical_step_ratio"] = 0.0
            metrics["copd/teacher_batch_size"] = 0.0
            metrics["copd/teacher_available"] = 0.0
            return batch
