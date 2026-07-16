# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""FlowSpec trainer: FLM initialization, target conditioning, and metrics."""

from argparse import Namespace
from typing import List, Tuple

import torch
import torch.distributed as dist

from torchspec.models.draft.flowspec import FlowSpecConfig, FlowSpecDraftModel
from torchspec.models.flowspec import FlowSpecModel
from torchspec.training import checkpoint
from torchspec.training.dflash_trainer import DFlashTrainer
from torchspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from torchspec.training.optimizer import BF16Optimizer
from torchspec.utils.distributed import get_gloo_group
from torchspec.utils.logging import logger


def _rank_sample_limit(max_samples: int, rank: int, world_size: int) -> int:
    if max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return max_samples // world_size + int(rank < max_samples % world_size)


def _acceptance_objective_active(
    global_step: int,
    total_steps: int,
    start_ratio: float | None,
) -> bool:
    if start_ratio is None:
        return False
    if not 0.0 <= start_ratio <= 1.0:
        raise ValueError(f"acceptance_start_ratio must be in [0, 1], got {start_ratio}")
    return global_step >= int(total_steps * start_ratio)


class FlowSpecTrainer(DFlashTrainer):
    """Train a dense-vocabulary FLM using cached target-model hidden states."""

    _anchor_slot_offset = 1
    _extra_loss_component_keys = [
        "block_match_length",
        "objective_loss",
        "acceptance_stage",
        "uniform_ce_loss",
        "acceptance_loss",
        "soft_acceptance_length",
    ]

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.block_size = getattr(args, "flowspec_block_size", 8)
        self.num_anchors = getattr(args, "flowspec_num_anchors", 512)
        self.num_target_layers = getattr(args, "flowspec_num_target_layers", 5)
        self.use_time_reparameterization = getattr(
            args,
            "flowspec_use_time_reparameterization",
            True,
        )
        self.loss_decay_gamma = getattr(args, "flowspec_loss_decay_gamma", 7.0)
        self.uniform_ce_weight = getattr(args, "flowspec_uniform_ce_weight", 0.5)
        self.acceptance_start_ratio = getattr(
            args,
            "flowspec_acceptance_start_ratio",
            None,
        )
        if self.acceptance_start_ratio is not None and not (
            0.0 <= self.acceptance_start_ratio <= 1.0
        ):
            raise ValueError(
                "flowspec_acceptance_start_ratio must be in [0, 1], got "
                f"{self.acceptance_start_ratio}"
            )
        self.boundary_probability = getattr(
            args,
            "flowspec_boundary_probability",
            0.0,
        )
        self.block_time_max_exponent = getattr(
            args,
            "flowspec_block_time_max_exponent",
            1.0,
        )
        self.use_target_distribution = getattr(
            args,
            "flowspec_use_target_distribution",
            False,
        )

    def init_model(
        self,
        draft_model_config,
        target_model_path: str,
        mooncake_config=None,
    ) -> int:
        if mooncake_config is not None:
            from torchspec.transfer.mooncake.utils import check_mooncake_master_available

            check_mooncake_master_available(
                mooncake_config.master_server_address,
                mooncake_config.metadata_server,
            )

        init_context = self._get_init_weight_context_manager()
        with init_context():
            if isinstance(draft_model_config, str):
                config = FlowSpecConfig.from_pretrained(draft_model_config)
            elif isinstance(draft_model_config, dict):
                config = FlowSpecConfig(**draft_model_config)
            elif isinstance(draft_model_config, FlowSpecConfig):
                config = draft_model_config
            else:
                raise TypeError(
                    "Unsupported draft_model_config type: "
                    f"{type(draft_model_config).__name__}. Expected str, dict, or "
                    "FlowSpecConfig."
                )

            if config.num_target_layers != self.num_target_layers:
                raise ValueError(
                    "training.flowspec_num_target_layers must match the draft config, "
                    f"got {self.num_target_layers} and {config.num_target_layers}."
                )
            draft_model = FlowSpecDraftModel(config)

        draft_model = draft_model.to(torch.bfloat16)
        dist.barrier(group=get_gloo_group())

        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] FlowSpec draft model: {trainable_count:,} trainable parameters"
        )

        flowspec_model = FlowSpecModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            use_time_reparameterization=self.use_time_reparameterization,
            loss_decay_gamma=self.loss_decay_gamma,
            uniform_ce_weight=self.uniform_ce_weight,
            boundary_probability=self.boundary_probability,
            block_time_max_exponent=self.block_time_max_exponent,
            use_target_distribution=self.use_target_distribution,
            use_flex_attention=getattr(self.args, "attention_backend", "sdpa") == "flex_attention",
        )
        full_state = flowspec_model.state_dict() if dist.get_rank() == 0 else {}

        flowspec_model = apply_fsdp2(
            flowspec_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=list(draft_model.layers),
        )
        flowspec_model = fsdp2_load_full_state_dict(
            flowspec_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        if getattr(self.args, "compile_model", False):
            logger.info("Compiling FlowSpec model with torch.compile (inductor backend)")
            flowspec_model = torch.compile(flowspec_model)

        self.model = flowspec_model
        unwrapped = getattr(self.model, "_orig_mod", self.model)
        self.flowspec = getattr(unwrapped, "module", unwrapped)
        self.draft_model = self.flowspec.draft_model

        total_steps = self.args.lr_total_steps
        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.1)
        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
            decay_style=decay_style if decay_style != "WSD" else "cosine",
            min_lr=getattr(self.args, "min_lr", 0.0),
        )

        if decay_style == "WSD" and total_steps:
            from torchspec.training.lr_scheduler import LRSchedulerWithWarmup

            wsd_ratio = getattr(
                self.args,
                "lr_wsd_decay_ratio",
                getattr(self.args, "wsd_decay_ratio", 0.2),
            )
            wsd_decay_style = (
                getattr(self.args, "lr_wsd_decay_style", None)
                or getattr(self.args, "wsd_decay_style", None)
                or "cosine"
            )
            self.optimizer.scheduler = LRSchedulerWithWarmup(
                self.optimizer.optimizer,
                max_lr=self.args.learning_rate,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
                decay_style="WSD",
                min_lr=getattr(self.args, "min_lr", 0.0),
                wsd_decay_steps=int(wsd_ratio * total_steps),
                wsd_decay_style=wsd_decay_style,
            )
        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        self.target_lm_head_weight = None
        if self.use_target_distribution:
            self._init_target_lm_head(target_model_path)
            if self.target_lm_head is None:
                raise ValueError("Failed to initialize the target LM head.")
            self.target_lm_head_weight = self.target_lm_head.lm_head.weight

        self.prof.on_init_end()
        logger.info(f"[Rank {self.dp_rank}] FlowSpec model initialized with FSDP2")
        return 0

    def _split_hidden_states(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        total_dim = hidden_states.shape[-1]
        if total_dim % self.num_target_layers != 0:
            raise ValueError(
                "Concatenated target hidden size must be divisible by "
                f"flowspec_num_target_layers, got {total_dim} and {self.num_target_layers}."
            )
        return list(hidden_states.split(total_dim // self.num_target_layers, dim=-1))

    def _forward(
        self,
        batch: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        last_hidden_states = None
        if self.use_target_distribution:
            last_hidden_states = batch.get("last_hidden_states")
            if last_hidden_states is None:
                raise ValueError(
                    "FlowSpec soft targets require last_hidden_states. "
                    "Set inference.store_last_hidden_states=true."
                )
            last_hidden_states = last_hidden_states.to(device, non_blocking=True)

        hidden_states_list = self._split_hidden_states(hidden_states)
        del hidden_states
        return self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=self.target_lm_head_weight,
            last_hidden_states=last_hidden_states,
            use_acceptance_objective=_acceptance_objective_active(
                self.global_step,
                self.args.lr_total_steps,
                self.acceptance_start_ratio,
            ),
        )

    def _aggregate_ode_eval_metrics(
        self,
        all_step_metrics: list[dict[str, torch.Tensor]],
        num_steps: int,
    ) -> dict:
        device = torch.device("cuda")
        match_sum = torch.zeros(self.block_size, device=device)
        count_sum = torch.zeros(self.block_size, device=device)
        prefix_match_sum = torch.zeros(self.block_size, device=device)
        block_match_length_sum = torch.zeros((), device=device)
        block_count = torch.zeros((), device=device)
        row_count = torch.zeros((), device=device)

        for step_metrics in all_step_metrics:
            match_sum += step_metrics["token_match_per_position"]
            count_sum += step_metrics["count_per_position"]
            prefix_match_sum += step_metrics["prefix_match_per_position"]
            block_match_length_sum += step_metrics["block_match_length_sum"]
            block_count += step_metrics["block_count"]
            row_count += step_metrics["row_count"]

        reduced = torch.cat(
            [
                match_sum,
                count_sum,
                prefix_match_sum,
                block_match_length_sum[None],
                block_count[None],
                row_count[None],
            ]
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

        offset = self.block_size
        match_sum = reduced[:offset]
        count_sum = reduced[offset : 2 * offset]
        prefix_match_sum = reduced[2 * offset : 3 * offset]
        block_match_length_sum, block_count, row_count = reduced[3 * offset :]
        match_sum = match_sum[self._anchor_slot_offset :]
        count_sum = count_sum[self._anchor_slot_offset :]
        prefix_match_sum = prefix_match_sum[self._anchor_slot_offset :]
        accuracy = match_sum / count_sum.clamp(min=1.0)
        prefix_match = prefix_match_sum / block_count.clamp(min=1.0)
        conditional_match = torch.zeros_like(prefix_match)
        conditional_match[0] = prefix_match[0]
        conditional_match[1:] = torch.where(
            prefix_match[:-1] > 0,
            prefix_match[1:] / prefix_match[:-1],
            torch.zeros_like(prefix_match[1:]),
        )

        prefix = "ode_eval/"
        metrics = {
            f"{prefix}avg_acc": (match_sum.sum() / count_sum.sum().clamp(min=1.0)).item(),
            f"{prefix}block_match_length": (
                block_match_length_sum / block_count.clamp(min=1.0)
            ).item(),
            f"{prefix}num_rows": row_count.item(),
            f"{prefix}num_blocks": block_count.item(),
            f"{prefix}num_steps": num_steps,
        }
        for position, value in enumerate(accuracy):
            metrics[f"{prefix}acc_{position}"] = value.item()
            metrics[f"{prefix}prefix_match_{position}"] = prefix_match[position].item()
            metrics[f"{prefix}conditional_match_{position}"] = conditional_match[position].item()
        return metrics

    @torch.inference_mode()
    def ode_eval_from_cache(
        self,
        num_steps: int,
        max_samples: int,
        seed: int,
    ) -> dict:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        if not getattr(self, "_eval_cache", None):
            return {}

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_limit = min(
            len(self._eval_cache),
            _rank_sample_limit(max_samples, rank, world_size),
        )
        eval_mbs = getattr(self.args, "eval_micro_batch_size", None) or self.args.micro_batch_size
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + rank)

        self.model.eval()
        all_metrics: list[dict[str, torch.Tensor]] = []
        try:
            for i in range(0, local_limit, eval_mbs):
                batch = self._eval_collator(self._eval_cache[i : i + eval_mbs])
                input_ids = batch["input_ids"].cuda()
                hidden_states = batch["hidden_states"].cuda()
                loss_mask = batch["loss_mask"]
                if loss_mask.dim() == 3:
                    loss_mask = loss_mask.squeeze(-1)
                loss_mask = loss_mask.cuda()
                hidden_states_list = self._split_hidden_states(hidden_states)
                del hidden_states

                all_metrics.append(
                    self.flowspec.evaluate_integrated(
                        input_ids=input_ids,
                        hidden_states_list=hidden_states_list,
                        loss_mask=loss_mask,
                        num_steps=num_steps,
                        generator=generator,
                    )
                )
        finally:
            self.model.train()

        metrics = self._aggregate_ode_eval_metrics(all_metrics, num_steps)
        metrics["ode_eval/step"] = self.global_step
        if rank == 0:
            logger.info(
                "ODE eval: steps=%d rows=%d acc_0=%.4f block_match_length=%.2f",
                num_steps,
                int(metrics["ode_eval/num_rows"]),
                metrics["ode_eval/acc_0"],
                metrics["ode_eval/block_match_length"],
            )
        return metrics
