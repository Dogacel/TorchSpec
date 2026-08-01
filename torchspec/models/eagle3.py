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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from torchspec.models.ops.loss import (
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,
    compiled_sum_forward_kl_loss_from_hs,
)
from torchspec.utils.distributed import (
    get_draft_sp_group,
    get_sp_ring_rank,
    get_sp_ulysses_group,
)


@dataclass
class PrecomputedTarget:
    """Target probabilities packed to the supervised positions."""

    target_p_padded: torch.Tensor  # (N_loss_candidates, V_draft)
    candidate_valid: Optional[torch.Tensor] = None  # (N_loss_candidates,)
    target_argmax: Optional[torch.Tensor] = None  # (N_loss_candidates,) argmax of the above


@dataclass
class LazyTarget:
    """Deferred target computation to avoid materializing (B, T, V_full)."""

    hidden_states_padded: torch.Tensor  # (B, T + length, D)
    lm_head_weight: torch.Tensor  # (V_full, D)


def _pack_loss_mask_indices(loss_mask: torch.Tensor) -> torch.Tensor:
    """Compatibility path for direct model callers without collator-packed indices.

    Production training supplies ``loss_valid_idx`` from the CPU prefetch path.
    Keeping this fallback avoids an API break for standalone/tests while still
    avoiding a data-dependent GPU ``nonzero`` operation.
    """
    flat_mask = loss_mask.detach().reshape(-1)
    values = flat_mask.cpu().tolist()
    return torch.tensor(
        [i for i, value in enumerate(values) if value],
        dtype=torch.long,
        device=loss_mask.device,
    )


def _build_loss_candidates(
    loss_valid_idx: torch.Tensor,
    *,
    depth_count: int,
    base_seq_length: int,
    step_seq_length: int,
    candidate_valid: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-capacity indices for every TTT depth in one GPU operation."""
    batch_idx = torch.div(loss_valid_idx, base_seq_length, rounding_mode="floor")
    base_position = loss_valid_idx.remainder(base_seq_length)
    depths = torch.arange(depth_count, device=loss_valid_idx.device).unsqueeze(1)
    shifted_position = base_position.unsqueeze(0) - depths
    valid_weights = (shifted_position >= 0) & (shifted_position < step_seq_length)
    valid_idx = batch_idx.unsqueeze(0) * step_seq_length + shifted_position.clamp(
        0, step_seq_length - 1
    )
    if candidate_valid is not None:
        valid_weights = valid_weights & candidate_valid.unsqueeze(0)
    return valid_idx, valid_weights


class Eagle3Model(nn.Module):
    def __init__(
        self,
        draft_model,
        length: int = 7,
        attention_backend="sdpa",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_pruning = draft_model.vocab_size != draft_model.target_vocab_size
        self._usp_sp_group = get_draft_sp_group() if attention_backend == "usp" else None
        self._usp_ulysses_group = get_sp_ulysses_group() if attention_backend == "usp" else None
        self._usp_ulysses_world_size = (
            dist.get_world_size(self._usp_ulysses_group)
            if self._usp_ulysses_group is not None
            else 1
        )

    def _calculate_loss(
        self,
        hidden_states: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget],
        valid_idx: torch.Tensor,
        valid_weights: torch.Tensor,
        idx: int,
        seq_length: int,
        norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        norm_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if valid_idx.numel() == 0:
            # FSDP requires every trainable param to participate in gradient
            # all-reduce/reduce-scatter.
            total = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad)
            zero = total * 0.0
            return zero, zero.detach(), zero.detach()

        torch._dynamo.maybe_mark_dynamic(valid_idx, 0)
        hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

        if isinstance(target, PrecomputedTarget):
            args = (
                hs_flat,
                target.target_p_padded,
                valid_idx,
                norm_weight,
                lm_head_weight,
                norm_eps,
                valid_weights,
                target.target_argmax,
            )
            if self.gradient_checkpointing and self.training:
                return torch_checkpoint(
                    compiled_forward_kl_loss,
                    *args,
                    use_reentrant=False,
                )
            return compiled_forward_kl_loss(*args)
        else:
            ths_flat = target.hidden_states_padded[:, idx : idx + seq_length, :].reshape(
                -1, target.lm_head_weight.shape[-1]
            )
            args = (
                hs_flat,
                ths_flat,
                valid_idx,
                norm_weight,
                lm_head_weight,
                target.lm_head_weight,
                norm_eps,
                valid_weights,
            )
            use_sum_lazy_loss = self.attention_backend == "usp"
            if self.gradient_checkpointing and self.training:
                return torch_checkpoint(
                    compiled_sum_forward_kl_loss_from_hs
                    if use_sum_lazy_loss
                    else compiled_forward_kl_loss_from_hs,
                    *args,
                    use_reentrant=False,
                )
            if use_sum_lazy_loss:
                return compiled_sum_forward_kl_loss_from_hs(*args)
            return compiled_forward_kl_loss_from_hs(*args)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget],
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        loss_valid_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        norm_weight, lm_head_weight, norm_eps = self.draft_model.get_lm_head_params()
        hidden_states = self.draft_model.project_hidden_states(hidden_states)
        if loss_valid_idx is None:
            loss_valid_idx = _pack_loss_mask_indices(loss_mask)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if self.attention_backend == "usp":
            usp_chunk_size = seq_length - self.length
            if usp_chunk_size <= 0:
                raise ValueError(
                    f"USP local seq_length ({seq_length}) must be larger than ttt_length ({self.length})"
                )
            if position_ids is None:
                device = hidden_states.device
                ring_chunk_size = usp_chunk_size * self._usp_ulysses_world_size
                position_start = get_sp_ring_rank() * ring_chunk_size + past_key_values_length
                position_ids = torch.arange(
                    position_start,
                    position_start + ring_chunk_size,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
        elif position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        plosses = []
        vlosses = []
        acces = []
        acc_counts = []
        cache_keys = None
        cache_values = None

        loss_seq_length = usp_chunk_size if self.attention_backend == "usp" else seq_length
        candidate_valid = target.candidate_valid if isinstance(target, PrecomputedTarget) else None
        loss_indices, loss_weights = _build_loss_candidates(
            loss_valid_idx,
            depth_count=self.length,
            base_seq_length=seq_length,
            step_seq_length=loss_seq_length,
            candidate_valid=candidate_valid,
        )

        input_ids = input_ids.clamp(min=0, max=self.draft_model.target_vocab_size - 1)
        # Every TTT step shifts the token ids left by one and fills the tail with
        # token 0. The embedding is frozen, so embed the overlapping extended
        # sequence once and take views instead of launching the same lookup for
        # every depth.
        if self.length > 1:
            tail_ids = input_ids.new_zeros((batch_size, self.length - 1))
            extended_input_ids = torch.cat((input_ids, tail_ids), dim=1)
        else:
            extended_input_ids = input_ids
        all_input_embeds = self.draft_model.embed_input_ids(extended_input_ids)
        all_input_embeds = all_input_embeds.to(hidden_states.dtype)

        for idx in range(self.length):
            step_hidden_states = hidden_states
            step_attention_mask = attention_mask
            step_position_ids = position_ids
            step_seq_length = seq_length

            if self.attention_backend == "usp":
                step_seq_length = usp_chunk_size
                step_hidden_states = hidden_states[:, :step_seq_length, :]
                if attention_mask is not None:
                    step_attention_mask = attention_mask[:, :step_seq_length]
                if position_ids is not None:
                    step_position_ids = position_ids[
                        :, : step_seq_length * self._usp_ulysses_world_size
                    ]

            inputs_embeds = all_input_embeds[:, idx : idx + step_seq_length]

            if self.gradient_checkpointing and self.training:
                hidden_states_out, cache_keys, cache_values = torch_checkpoint(
                    self.draft_model.backbone,
                    inputs_embeds,
                    step_hidden_states,
                    step_attention_mask,
                    step_position_ids,
                    cache_keys,
                    cache_values,
                    True,
                    use_reentrant=False,
                )
            else:
                hidden_states_out, cache_keys, cache_values = self.draft_model.backbone(
                    input_embeds=inputs_embeds,
                    hidden_states=step_hidden_states,
                    attention_mask=step_attention_mask,
                    position_ids=step_position_ids,
                    cache_keys=cache_keys,
                    cache_values=cache_values,
                    use_cache=True,
                )

            hidden_states = hidden_states_out

            local_sum_loss, local_correct, local_count = self._calculate_loss(
                hidden_states=hidden_states,
                target=target,
                valid_idx=loss_indices[idx],
                valid_weights=loss_weights[idx],
                idx=idx,
                seq_length=step_seq_length,
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_eps=norm_eps,
            )

            # Model takes its own normed hidden states as input for the next step, so apply norm here if needed.
            if self.draft_model.norm_output:
                hidden_states = self.draft_model.norm(hidden_states)

            if self.attention_backend == "usp":
                # A shard can have no local loss tokens while its Ulysses peers do.
                # Keep the zero-loss path connected to this layer's activations so
                # autograd still executes the same sequence-parallel collectives.
                local_sum_loss = local_sum_loss + hidden_states.sum() * 0.0

            loss = local_sum_loss / local_count.clamp_min(1.0)
            metric_loss = loss.detach()
            metric_acc = (local_correct / local_count.clamp_min(1.0)).detach()

            if self._usp_sp_group is not None:
                reduced_stats = torch.stack(
                    (
                        local_sum_loss.detach().clone().float(),
                        local_correct.detach().clone().float(),
                        local_count.detach().clone().float(),
                    )
                )
                dist.all_reduce(reduced_stats, op=dist.ReduceOp.SUM, group=self._usp_sp_group)
                reduced_sum_loss, reduced_correct, reduced_count = reduced_stats.unbind()
                denom = reduced_count.clamp_min(1.0)
                loss = (local_sum_loss / denom).to(loss.dtype)
                metric_loss = (reduced_sum_loss / denom).detach()
                metric_acc = (reduced_correct / denom).to(device=loss.device, dtype=torch.float32)
                metric_count = reduced_count.to(device=loss.device, dtype=torch.float32)
            else:
                metric_count = local_count.detach().float().to(device=loss.device)

            plosses.append(loss)
            vlosses.append(metric_loss)
            acces.append(metric_acc)
            acc_counts.append(metric_count)

        return plosses, vlosses, acces, acc_counts


@torch.no_grad()
def compute_target_p_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    t2d: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    length: int = 7,
    loss_valid_idx: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> PrecomputedTarget:
    """Project only supervised target states and reuse their distribution at every depth."""
    del length  # Compact targets do not need depth padding.
    target_lm_head_weight = target_lm_head_weight.detach()
    output_weight = target_lm_head_weight if t2d is None else target_lm_head_weight[t2d]

    bsz, seq_len, hidden_size = target_hidden_states.shape
    if loss_valid_idx is None:
        if loss_mask is None:
            raise ValueError("compute_target_p_padded requires loss_valid_idx or loss_mask")
        loss_valid_idx = _pack_loss_mask_indices(loss_mask)

    valid_hs = target_hidden_states.reshape(-1, hidden_size).index_select(0, loss_valid_idx)
    candidate_valid = None
    if t2d is not None:
        candidate_valid = torch.empty(
            loss_valid_idx.shape[0],
            device=target_hidden_states.device,
            dtype=torch.bool,
        )
        for i in range(0, valid_hs.shape[0], chunk_size):
            chunk_hs = valid_hs[i : i + chunk_size]
            chunk_argmax = F.linear(chunk_hs, target_lm_head_weight).argmax(-1)
            candidate_valid[i : i + chunk_size] = t2d[chunk_argmax]

    if valid_hs.shape[0] <= chunk_size:
        target_logits = F.linear(valid_hs, output_weight)
        target_p = F.softmax(target_logits.float(), dim=-1)
        target_argmax = target_logits.argmax(dim=-1)
    else:
        target_p = torch.empty(
            (valid_hs.shape[0], output_weight.shape[0]),
            device=valid_hs.device,
            dtype=torch.float32,
        )
        target_argmax = torch.empty(
            valid_hs.shape[0],
            device=valid_hs.device,
            dtype=torch.long,
        )
        for i in range(0, valid_hs.shape[0], chunk_size):
            chunk_logits = F.linear(valid_hs[i : i + chunk_size], output_weight)
            target_p[i : i + chunk_size].copy_(F.softmax(chunk_logits.float(), dim=-1))
            target_argmax[i : i + chunk_size].copy_(chunk_logits.argmax(dim=-1))

    return PrecomputedTarget(
        target_p_padded=target_p,
        candidate_valid=candidate_valid,
        target_argmax=target_argmax,
    )


def compute_lazy_target_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    length: int,
) -> LazyTarget:
    return LazyTarget(
        hidden_states_padded=F.pad(target_hidden_states, (0, 0, 0, length), value=0.0),
        lm_head_weight=target_lm_head_weight.detach(),
    )
