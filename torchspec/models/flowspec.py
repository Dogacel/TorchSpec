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

"""Conditional FLM training and sampling for FlowSpec.

An anchor is the target-generated bonus token at slot 0 of a size-``K`` flow
block.  The drafter sees target context strictly before the anchor, keeps the
anchor clean and unscored, and predicts slots 1 through ``K - 1``.  Each block
attends bidirectionally within itself and cannot see other flow blocks.

FLM uses two time coordinates. Reparameterized time ``tau`` is sampled and
given to the denoiser; physical time ``t`` defines the interpolant
``x_t = (1 - t) * noise + t * clean``. The clean state contains target-model
token probabilities. The corresponding error schedule maps between the two
coordinates for training and Euler integration.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.draft.flowspec import FlowSpecDraftModel
from torchspec.models.flowspec_schedule import (
    NOISE_TYPES,
    TIME_SCHEDULES,
    make_time_schedule,
    sample_noise,
)
from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask


def _create_flowspec_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    context_len: int,
    block_size: int,
):
    """Create the target-prefix plus isolated-bidirectional-block attention rule."""

    num_anchors = anchor_positions.shape[1]

    def flowspec_mask_mod(b, h, q_idx, kv_idx):
        query_block = q_idx // block_size
        anchor = anchor_positions[b, query_block]

        is_context = kv_idx < context_len
        context_visible = is_context & (kv_idx < anchor)

        is_draft = kv_idx >= context_len
        key_block = (kv_idx - context_len) // block_size
        own_block_visible = is_draft & (key_block == query_block)

        return (context_visible | own_block_visible) & block_keep_mask[b, query_block]

    flowspec_mask_mod.__name__ = f"flowspec_mask_A{num_anchors}_B{block_size}_C{context_len}"
    return flowspec_mask_mod


def _create_flowspec_dense_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
    """Materialize the FlowSpec mask for SDPA and semantic tests."""

    batch_size, num_anchors = anchor_positions.shape
    draft_len = num_anchors * block_size
    kv_len = context_len + draft_len
    device = anchor_positions.device
    mask_mod = _create_flowspec_mask_mod(
        anchor_positions,
        block_keep_mask,
        context_len,
        block_size,
    )
    batch = torch.arange(batch_size, device=device)[:, None, None]
    queries = torch.arange(draft_len, device=device)[None, :, None]
    keys = torch.arange(kv_len, device=device)[None, None, :]
    return mask_mod(batch, 0, queries, keys).unsqueeze(1)


class FlowSpecModel(nn.Module):
    """Train and sample a target-conditioned FlowSpec denoiser."""

    def __init__(
        self,
        draft_model: FlowSpecDraftModel,
        block_size: int = 8,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        boundary_probability: float = 0.0,
        clean_deadline_jitter: float = 0.0,
        block_schedule_overlap: float = 1.0,
        use_diffusion_forcing_schedule: bool = True,
        time_schedule: str = "simplex_sharp",
        noise_type: str = "simplex",
        self_condition_probability: float = 0.0,
        use_flex_attention: bool = True,
        ode_epsilon: float = 1e-5,
    ):
        super().__init__()
        if block_size < 2:
            raise ValueError(f"block_size must be at least 2, got {block_size}")
        if num_anchors <= 0:
            raise ValueError(f"num_anchors must be positive, got {num_anchors}")
        if not math.isfinite(loss_decay_gamma) or loss_decay_gamma <= 0.0:
            raise ValueError(
                f"loss_decay_gamma must be finite and positive, got {loss_decay_gamma}"
            )
        if not 0.0 <= boundary_probability < 1.0:
            raise ValueError(f"boundary_probability must be in [0, 1), got {boundary_probability}")
        if not math.isfinite(clean_deadline_jitter) or not 0.0 <= clean_deadline_jitter < 1.0:
            raise ValueError(
                f"clean_deadline_jitter must be finite and in [0, 1), got {clean_deadline_jitter}"
            )
        if not math.isfinite(block_schedule_overlap) or block_schedule_overlap < 1.0:
            raise ValueError(
                "block_schedule_overlap must be finite and at least 1, got "
                f"{block_schedule_overlap}"
            )
        if ode_epsilon <= 0:
            raise ValueError(f"ode_epsilon must be positive, got {ode_epsilon}")
        if time_schedule not in TIME_SCHEDULES:
            raise ValueError(
                f"time_schedule must be one of {sorted(TIME_SCHEDULES)}, got {time_schedule!r}"
            )
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {sorted(NOISE_TYPES)}, got {noise_type!r}")
        if not 0.0 <= self_condition_probability <= 1.0:
            raise ValueError(
                f"self_condition_probability must be in [0, 1], got {self_condition_probability}"
            )
        if (self_condition_probability > 0.0) != draft_model.config.self_conditioning:
            raise ValueError(
                "draft-model self_conditioning must match whether "
                "self_condition_probability is positive"
            )
        self.draft_model = draft_model
        self.block_size = block_size
        self.num_predictions = block_size - 1
        self.num_anchors = num_anchors
        self.boundary_probability = boundary_probability
        self.clean_deadline_jitter = clean_deadline_jitter
        self.block_schedule_overlap = block_schedule_overlap
        self.use_diffusion_forcing_schedule = use_diffusion_forcing_schedule
        self.time_schedule = time_schedule
        self.noise_type = noise_type
        self.self_condition_probability = self_condition_probability
        self.use_flex_attention = use_flex_attention
        self.ode_epsilon = ode_epsilon
        self.time_from_tau = make_time_schedule(time_schedule, draft_model.config.vocab_size)
        window_stride = 1.0 / (self.num_predictions - 1 + block_schedule_overlap)
        clean_starts = torch.arange(self.num_predictions, dtype=torch.float32) * window_stride
        clean_deadlines = clean_starts + block_schedule_overlap * window_stride
        clean_deadlines[-1] = 1.0
        self.block_schedule_stride = window_stride
        self.register_buffer("clean_starts", clean_starts, persistent=False)
        self.register_buffer("clean_deadlines", clean_deadlines, persistent=False)
        self.register_buffer(
            "loss_decay_weights",
            torch.exp(
                -(torch.arange(block_size, dtype=torch.float32) - 1.0).clamp_min(0.0)
                / loss_decay_gamma
            ),
            persistent=False,
        )
        if use_diffusion_forcing_schedule:
            expected_deadlines = clean_deadlines - clean_deadline_jitter * window_stride / 2.0
            active_probability = (
                boundary_probability + (1.0 - boundary_probability) * expected_deadlines
            )
            # Correct nested window exposure before applying the position objective.
            schedule_loss_weights = torch.cat([torch.ones(1), active_probability.reciprocal()])
        else:
            schedule_loss_weights = torch.ones(block_size, dtype=torch.float32)
        self.register_buffer(
            "schedule_loss_weights",
            schedule_loss_weights,
            persistent=False,
        )

    def _sample_noise(
        self,
        shape: torch.Size | Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return sample_noise(self.noise_type, shape, device, generator=generator)

    def tau_to_time(self, tau: torch.Tensor) -> torch.Tensor:
        # Uniform tau is mapped through the fitted recognition-time distribution.
        return self.time_from_tau(tau)

    def _valid_anchor_mask(self, loss_mask: torch.Tensor) -> torch.Tensor:
        sequence_len = loss_mask.shape[1]
        if sequence_len < self.block_size:
            return torch.zeros(
                loss_mask.shape[0],
                0,
                dtype=torch.bool,
                device=loss_mask.device,
            )
        target_windows = loss_mask.unfold(1, self.block_size, 1)
        return (target_windows > 0.5).all(dim=-1)

    def _sample_anchor_positions(
        self,
        loss_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_len = loss_mask.shape
        valid = self._valid_anchor_mask(loss_mask)
        candidate_count = valid.shape[1]
        device = loss_mask.device

        if candidate_count == 0:
            anchors = torch.zeros(
                batch_size,
                self.num_anchors,
                dtype=torch.long,
                device=device,
            )
            keep_mask = torch.zeros_like(anchors, dtype=torch.bool)
            return anchors, keep_mask

        candidates = torch.arange(candidate_count, device=device)
        candidates = candidates.unsqueeze(0).expand(batch_size, -1)
        candidate_values = torch.where(valid, candidates, sequence_len + 1)
        scores = torch.rand(
            batch_size,
            candidate_count,
            device=device,
            generator=generator,
        )
        scores = torch.where(valid, scores, torch.inf)
        order = scores.argsort(dim=1)
        randomized = torch.gather(candidate_values, 1, order)

        take = min(self.num_anchors, candidate_count)
        selected = randomized[:, :take].sort(dim=1).values
        if take < self.num_anchors:
            padding = torch.zeros(
                batch_size,
                self.num_anchors - take,
                dtype=torch.long,
                device=device,
            )
            selected = torch.cat([selected, padding], dim=1)

        valid_counts = valid.sum(dim=1).clamp(max=self.num_anchors)
        keep_mask = torch.arange(self.num_anchors, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1)
        anchors = torch.where(keep_mask, selected, 0)
        return anchors, keep_mask

    def _block_token_indices(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(
            self.block_size,
            device=anchor_positions.device,
        )
        return anchor_positions.unsqueeze(-1) + offsets

    def _gather_block_tokens(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_len = input_ids.shape
        indices = self._block_token_indices(anchor_positions).clamp(max=sequence_len - 1)
        expanded_ids = input_ids.unsqueeze(1).expand(-1, anchor_positions.shape[1], -1)
        return torch.gather(expanded_ids, 2, indices)

    def _target_logits(
        self,
        last_hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, hidden_size = last_hidden_states.shape
        # Hidden state anchor + k predicts scored slot k + 1; slot 0 is known.
        hidden_offsets = torch.arange(
            self.num_predictions,
            device=anchor_positions.device,
        )
        hidden_indices = anchor_positions.unsqueeze(-1) + hidden_offsets
        expanded_hidden = last_hidden_states.unsqueeze(1).expand(
            -1,
            anchor_positions.shape[1],
            -1,
            -1,
        )
        gather_indices = hidden_indices.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            hidden_size,
        )
        target_hidden = torch.gather(expanded_hidden, 2, gather_indices)
        return F.linear(
            target_hidden.to(lm_head_weight.dtype),
            lm_head_weight,
        )

    def _create_position_ids(
        self,
        anchor_positions: torch.Tensor,
        context_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = anchor_positions.shape[0]
        context_positions = torch.arange(context_len, device=anchor_positions.device).unsqueeze(0)
        context_positions = context_positions.expand(batch_size, -1)
        draft_positions = self._block_token_indices(anchor_positions).flatten(1)
        return context_positions, draft_positions

    def _create_attention_mask(
        self,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        context_len: int,
    ):
        batch_size, num_anchors = anchor_positions.shape
        draft_len = num_anchors * self.block_size
        if anchor_positions.device.type == "cuda" and self.use_flex_attention:
            mask_mod = _create_flowspec_mask_mod(
                anchor_positions,
                block_keep_mask,
                context_len,
                self.block_size,
            )
            return compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=context_len + draft_len,
                device=anchor_positions.device,
            )
        return _create_flowspec_dense_mask(
            anchor_positions,
            block_keep_mask,
            context_len,
            self.block_size,
        )

    def _corrupt_distribution(
        self,
        target: torch.Tensor,
        physical_time: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = self._sample_noise(target.shape, target.device)
        elif noise.shape != target.shape:
            raise ValueError(
                f"noise must have shape {tuple(target.shape)}, got {tuple(noise.shape)}"
            )
        if physical_time.shape == target.shape[:-2]:
            physical_time = physical_time.unsqueeze(-1)
        elif physical_time.shape != target.shape[:-1]:
            raise ValueError(
                "physical_time must have shape [B, N] or [B, N, K] matching target, got "
                f"{tuple(physical_time.shape)} and {tuple(target.shape)}"
            )
        time = physical_time.unsqueeze(-1).float()
        target = target.float()
        noise = noise.float()
        return (1.0 - time) * noise + time * target

    def _sample_training_times(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        tau = torch.rand(
            batch_size,
            self.num_anchors,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        if self.boundary_probability > 0.0:
            tau = ((tau - self.boundary_probability) / (1.0 - self.boundary_probability)).clamp_min(
                0.0
            )
        return tau

    def _block_time_coordinates(
        self,
        block_tau: torch.Tensor,
        clean_deadlines: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map global tau onto consecutive per-token denoising windows."""

        if clean_deadlines is None:
            clean_deadlines = self.clean_deadlines.expand(*block_tau.shape, -1)
        elif clean_deadlines.shape != (*block_tau.shape, self.num_predictions):
            raise ValueError(
                "clean_deadlines must have shape "
                f"{(*block_tau.shape, self.num_predictions)}, got {tuple(clean_deadlines.shape)}"
            )

        starts = self.clean_starts.expand_as(clean_deadlines)
        # Ordered overlapping windows expose a cleaner prefix and noisier suffix.
        prediction_tau = ((block_tau.unsqueeze(-1) - starts) / (clean_deadlines - starts)).clamp(
            0.0,
            1.0,
        )
        endpoint_tolerance = 8 * torch.finfo(prediction_tau.dtype).eps
        prediction_tau = torch.where(
            prediction_tau >= 1.0 - endpoint_tolerance,
            torch.ones_like(prediction_tau),
            prediction_tau,
        )
        prediction_tau = torch.where(
            prediction_tau <= endpoint_tolerance,
            torch.zeros_like(prediction_tau),
            prediction_tau,
        )
        token_tau = torch.cat([torch.ones_like(block_tau).unsqueeze(-1), prediction_tau], dim=-1)
        token_time = self.tau_to_time(token_tau)
        return token_tau, token_time

    def _regular_time_coordinates(
        self,
        block_tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use one shared denoising time for every prediction in a block."""

        prediction_tau = block_tau.unsqueeze(-1).expand(*block_tau.shape, self.num_predictions)
        token_tau = torch.cat([torch.ones_like(block_tau).unsqueeze(-1), prediction_tau], dim=-1)
        return token_tau, self.tau_to_time(token_tau)

    def _sample_clean_deadlines(
        self,
        block_tau: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        deadlines = self.clean_deadlines.expand(*block_tau.shape, -1)
        if self.clean_deadline_jitter == 0.0:
            return deadlines

        # Move every deadline earlier by less than one window stride. Independent
        # offsets therefore preserve the ordering of the clean deadlines.
        offsets = torch.rand(
            deadlines.shape,
            device=block_tau.device,
            dtype=block_tau.dtype,
            generator=generator,
        )
        return deadlines - offsets * (self.clean_deadline_jitter * self.block_schedule_stride)

    def _sample_training_time_coordinates(
        self,
        block_tau: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        deadlines = self._sample_clean_deadlines(block_tau, generator=generator)
        return self._block_time_coordinates(block_tau, clean_deadlines=deadlines)

    def _time_coordinates(
        self,
        block_tau: torch.Tensor,
        training: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_diffusion_forcing_schedule:
            return self._regular_time_coordinates(block_tau)
        if training:
            return self._sample_training_time_coordinates(block_tau, generator=generator)
        return self._block_time_coordinates(block_tau)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        if input_ids.ndim != 2 or loss_mask.shape != input_ids.shape:
            raise ValueError(
                "input_ids and loss_mask must both have shape [B, S], got "
                f"{tuple(input_ids.shape)} and {tuple(loss_mask.shape)}"
            )
        if lm_head_weight is None or last_hidden_states is None:
            raise ValueError("FlowSpec requires target LM logits for its clean probability state")
        if last_hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(
                "last_hidden_states must have shape [B, S, D] matching input_ids, got "
                f"{tuple(last_hidden_states.shape)} and {tuple(input_ids.shape)}"
            )
        if lm_head_weight.ndim != 2 or lm_head_weight.shape != (
            self.draft_model.config.vocab_size,
            last_hidden_states.shape[-1],
        ):
            raise ValueError(
                "lm_head_weight must have shape [V, D] matching the draft vocabulary and "
                f"target hidden size, got {tuple(lm_head_weight.shape)}"
            )

        batch_size, context_len = input_ids.shape
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(loss_mask)
        dataset_target_ids = self._gather_block_tokens(input_ids, anchor_positions)
        hard_target_distribution = F.one_hot(
            dataset_target_ids,
            self.draft_model.config.vocab_size,
        ).float()
        with torch.no_grad():
            prediction_distribution = (
                self._target_logits(
                    last_hidden_states,
                    lm_head_weight,
                    anchor_positions,
                )
                .float()
                .softmax(dim=-1)
            )
            target_distribution = torch.cat(
                [hard_target_distribution[:, :, :1], prediction_distribution],
                dim=2,
            )

        tau = self._sample_training_times(batch_size, input_ids.device)
        token_tau, physical_time = self._time_coordinates(
            tau,
            training=self.training,
        )
        flow_state = self._corrupt_distribution(
            target_distribution,
            physical_time,
        )
        flow_state = flow_state.flatten(1, 2)
        model_times = token_tau.flatten(1, 2)

        context_positions, draft_positions = self._create_position_ids(
            anchor_positions, context_len
        )
        attention_mask = self._create_attention_mask(
            anchor_positions,
            block_keep_mask,
            context_len,
        )
        self_condition_state = None
        if self.self_condition_probability > 0.0:
            zero_self_condition = torch.cat(
                [
                    hard_target_distribution[:, :, :1],
                    torch.zeros_like(target_distribution[:, :, 1:]),
                ],
                dim=2,
            )
            with torch.no_grad():
                initial_logits = self.draft_model(
                    flow_state=flow_state,
                    timesteps=model_times,
                    context_feature=context_feature,
                    draft_position_ids=draft_positions,
                    context_position_ids=context_positions,
                    block_mask=attention_mask,
                    self_condition_state=zero_self_condition.flatten(1, 2),
                )
                predicted_clean = (
                    initial_logits.float().softmax(dim=-1).view_as(target_distribution)
                )
                predicted_clean = torch.cat(
                    [hard_target_distribution[:, :, :1], predicted_clean[:, :, 1:]],
                    dim=2,
                )
                use_self_condition = (
                    torch.rand(batch_size, device=input_ids.device)
                    < self.self_condition_probability
                ).view(batch_size, 1, 1, 1)
                self_condition_state = torch.where(
                    use_self_condition,
                    predicted_clean,
                    zero_self_condition,
                ).flatten(1, 2)
        logits = self.draft_model(
            flow_state=flow_state,
            timesteps=model_times,
            context_feature=context_feature,
            draft_position_ids=draft_positions,
            context_position_ids=context_positions,
            block_mask=attention_mask,
            self_condition_state=self_condition_state,
        )

        draft_log_probabilities = logits.float().log_softmax(dim=-1).view_as(target_distribution)
        per_token_loss = -(target_distribution * draft_log_probabilities).sum(dim=-1)
        prediction_positions = (
            torch.arange(self.block_size, device=input_ids.device).view(1, 1, -1) > 0
        )
        token_mask = (
            block_keep_mask.unsqueeze(-1).expand_as(per_token_loss)
            & prediction_positions
            & token_tau.lt(1.0)
        ).float()
        token_count = token_mask.sum().clamp(min=1.0)
        schedule_weights = self.schedule_loss_weights.view(1, 1, -1)
        corrected_token_weights = token_mask * schedule_weights
        # Earlier experiments directly optimized differentiable prefix acceptance;
        # this baseline uses the standard DFlash-style gamma-decayed token CE.
        position_weights = self.loss_decay_weights.view(1, 1, -1)
        objective_weights = corrected_token_weights * position_weights
        weighted_token_count = objective_weights.sum().clamp(min=1.0)
        loss = (per_token_loss * objective_weights).sum() / weighted_token_count

        loss_components = {}
        with torch.no_grad():
            flat_predictions = logits.argmax(dim=-1)
            predictions = flat_predictions.view(batch_size, self.num_anchors, self.block_size)
            matches = predictions == dataset_target_ids
            correct = matches.float() * token_mask
            accuracy = correct.sum() / token_count
            count_per_position = token_mask.sum(dim=(0, 1))
            safe_count = count_per_position.clamp(min=1.0)
            loss_per_position = (per_token_loss * token_mask).sum(dim=(0, 1)) / safe_count
            accuracy_per_position = correct.sum(dim=(0, 1)) / safe_count

            active_predictions = token_mask[..., 1:].bool()
            prefix_survival = (
                torch.where(
                    active_predictions,
                    matches[..., 1:],
                    torch.ones_like(active_predictions),
                ).cumprod(dim=-1)
                & active_predictions
            )
            valid_blocks = block_keep_mask.bool() & active_predictions.any(dim=-1)
            block_match_length = (
                prefix_survival.sum(dim=-1) * valid_blocks
            ).sum() / valid_blocks.sum().clamp(min=1)
            loss_components = {
                "block_match_length": block_match_length.float(),
            }
        return (
            loss,
            accuracy,
            loss_per_position,
            accuracy_per_position,
            count_per_position,
            loss_components,
        )

    @torch.no_grad()
    def integrate(
        self,
        hidden_states_list: List[torch.Tensor],
        anchor_positions: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        num_steps: int,
        noise: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Integrate FLM and return ``[B, N, K, V]`` clean vocabulary states."""

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        batch_size, num_anchors = anchor_positions.shape
        if anchor_token_ids.shape != anchor_positions.shape:
            raise ValueError(
                "anchor_token_ids must match anchor_positions, got "
                f"{tuple(anchor_token_ids.shape)} and {tuple(anchor_positions.shape)}"
            )
        vocab_size = self.draft_model.config.vocab_size
        if bool(((anchor_token_ids < 0) | (anchor_token_ids >= vocab_size)).any()):
            raise ValueError(f"anchor_token_ids must be in [0, {vocab_size})")
        if block_keep_mask is None:
            block_keep_mask = torch.ones_like(anchor_positions, dtype=torch.bool)
        elif block_keep_mask.shape != anchor_positions.shape:
            raise ValueError(
                "block_keep_mask must match anchor_positions, got "
                f"{tuple(block_keep_mask.shape)} and {tuple(anchor_positions.shape)}"
            )

        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        context_len = context_feature.shape[1]
        context_positions, draft_positions = self._create_position_ids(
            anchor_positions, context_len
        )
        attention_mask = self._create_attention_mask(
            anchor_positions,
            block_keep_mask,
            context_len,
        )

        expected_shape = (
            batch_size,
            num_anchors,
            self.block_size,
            vocab_size,
        )
        if noise is None:
            state = self._sample_noise(
                expected_shape,
                anchor_positions.device,
                generator=generator,
            )
        elif noise.shape != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}, got {tuple(noise.shape)}")
        else:
            state = noise.float()
        known_anchor = F.one_hot(anchor_token_ids.long(), vocab_size).float().unsqueeze(2)
        state = torch.cat([known_anchor, state[:, :, 1:]], dim=2).flatten(1, 2)
        self_condition_state = (
            torch.cat(
                [
                    known_anchor,
                    torch.zeros(
                        batch_size,
                        num_anchors,
                        self.num_predictions,
                        vocab_size,
                        device=state.device,
                        dtype=state.dtype,
                    ),
                ],
                dim=2,
            ).flatten(1, 2)
            if self.self_condition_probability > 0.0
            else None
        )

        tau_grid = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            device=anchor_positions.device,
        )
        for step in range(num_steps):
            tau_current = tau_grid[step].expand(batch_size, num_anchors)
            tau_next = tau_grid[step + 1].expand(batch_size, num_anchors)
            token_tau_current, time_current = self._time_coordinates(tau_current)
            token_tau_next, time_next = self._time_coordinates(tau_next)
            logits = self.draft_model(
                flow_state=state,
                timesteps=token_tau_current.flatten(1, 2),
                context_feature=context_feature,
                draft_position_ids=draft_positions,
                context_position_ids=context_positions,
                block_mask=attention_mask,
                self_condition_state=self_condition_state,
            )
            clean_state = logits.float().softmax(dim=-1)
            clean_state = clean_state.view(expected_shape)
            clean_state = torch.cat(
                [known_anchor, clean_state[:, :, 1:]],
                dim=2,
            ).flatten(1, 2)
            if self.self_condition_probability > 0.0:
                self_condition_state = clean_state

            current = time_current.flatten(1, 2).unsqueeze(-1)
            delta = (time_next - time_current).flatten(1, 2).unsqueeze(-1)
            velocity = (clean_state - state) / (1.0 - current + self.ode_epsilon)
            updated_state = state + delta * velocity
            newly_clean = (
                ((token_tau_current < 1.0) & (token_tau_next >= 1.0)).flatten(1, 2).unsqueeze(-1)
            )
            state = torch.where(newly_clean, clean_state, updated_state)

        return state.view(expected_shape)

    @torch.no_grad()
    def generate(
        self,
        hidden_states_list: List[torch.Tensor],
        anchor_positions: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        num_steps: int,
        noise: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        state = self.integrate(
            hidden_states_list=hidden_states_list,
            anchor_positions=anchor_positions,
            anchor_token_ids=anchor_token_ids,
            num_steps=num_steps,
            noise=noise,
            block_keep_mask=block_keep_mask,
            generator=generator,
        )
        return state.argmax(dim=-1)

    @torch.no_grad()
    def evaluate_integrated(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        num_steps: int,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Run the ODE from noise and return additive dataset-match totals."""

        if input_ids.ndim != 2 or loss_mask.shape != input_ids.shape:
            raise ValueError(
                "input_ids and loss_mask must both have shape [B, S], got "
                f"{tuple(input_ids.shape)} and {tuple(loss_mask.shape)}"
            )
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            loss_mask,
            generator=generator,
        )
        target_ids = self._gather_block_tokens(input_ids, anchor_positions)
        predictions = self.generate(
            hidden_states_list=hidden_states_list,
            anchor_positions=anchor_positions,
            anchor_token_ids=target_ids[..., 0],
            num_steps=num_steps,
            block_keep_mask=block_keep_mask,
            generator=generator,
        )
        prediction_positions = (
            torch.arange(self.block_size, device=input_ids.device).view(1, 1, -1) > 0
        )
        token_mask = block_keep_mask.unsqueeze(-1).expand_as(predictions) & prediction_positions
        token_matches = predictions.eq(target_ids) & token_mask
        prediction_prefix_matches = token_matches[..., 1:].cumprod(dim=-1)
        prefix_matches = torch.cat(
            [torch.zeros_like(token_matches[..., :1]), prediction_prefix_matches],
            dim=-1,
        )
        block_match_length = prediction_prefix_matches.sum(dim=-1)
        count_per_position = token_mask.sum(dim=(0, 1)).float()
        block_count = block_keep_mask.sum().float()

        return {
            "token_match_per_position": (token_matches & token_mask).sum(dim=(0, 1)).float(),
            "count_per_position": count_per_position,
            "prefix_match_per_position": (prefix_matches * block_keep_mask.unsqueeze(-1))
            .sum(dim=(0, 1))
            .float(),
            "block_match_length_sum": (block_match_length * block_keep_mask).sum().float(),
            "block_count": block_count,
            "row_count": torch.tensor(
                input_ids.shape[0],
                device=input_ids.device,
                dtype=torch.float32,
            ),
        }
