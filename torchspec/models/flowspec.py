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

An anchor is the last known target-context token.  For block size ``K``, its
flow state represents tokens ``anchor + 1`` through ``anchor + K``.  Each block
attends bidirectionally within itself, sees target context through its anchor,
and cannot see other flow blocks.

FLM uses two time coordinates.  Reparameterized time ``tau`` is sampled and
given to the denoiser; physical time ``t`` defines the interpolant
``x_t = (1 - t) * noise + t * clean``. ``clean`` can be a one-hot dataset
token or the target LM's probability vector. The decoding-error schedule maps
between the two time coordinates for both training and Euler integration.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr

from torchspec.models.draft.flowspec import FlowSpecDraftModel
from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask


def _build_flm_time_schedule(
    vocab_size: int,
    num_points: int = 1_000,
    quadrature_points: int = 100,
) -> Tuple[CubicSpline, CubicSpline]:
    """Build cubic-spline LUTs for ``tau(t)`` and its inverse ``t(tau)``."""

    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be greater than 1, got {vocab_size}")
    if num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {num_points}")
    if quadrature_points < 1:
        raise ValueError(f"quadrature_points must be positive, got {quadrature_points}")

    # Gauss-Hermite nodes approximate the expectation over the clean
    # vocabulary coordinate's Gaussian noise without Monte Carlo sampling.
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    nodes = nodes * math.sqrt(2.0)
    weights = weights / math.sqrt(math.pi)

    physical_time = np.linspace(0.0, 1.0, num_points)
    noise_scale = np.maximum(1.0 - physical_time, 1e-12)
    standardized_signal = physical_time / noise_scale

    # Given clean-coordinate noise z, Phi(z + t/(1-t)) is the probability that
    # one wrong vocabulary coordinate is smaller than the clean coordinate;
    # log space avoids underflow when this probability is raised to V-1.
    log_cdf = log_ndtr(standardized_signal[:, None] + nodes[None, :])

    # All V-1 wrong coordinates must be smaller; integrating Phi(...)**(V-1)
    # over z gives the probability that argmax recovers the clean token.
    correct_probability = np.sum(
        weights[None, :] * np.exp((vocab_size - 1) * log_cdf),
        axis=-1,
    )

    # Normalize chance accuracy (1/V) to tau=0 and perfect accuracy to tau=1.
    tau = vocab_size / (vocab_size - 1.0) * (correct_probability - 1.0 / vocab_size)
    tau = tau + (physical_time - 1.0) * 1e-10
    tau = np.maximum.accumulate(np.clip(tau, 0.0, 1.0))
    tau[0] = 0.0
    tau[-1] = 1.0

    tau_of_time = CubicSpline(physical_time, tau)

    # The inverse spline needs unique, increasing tau knots in flat regions.
    unique_tau, unique_indices = np.unique(tau, return_index=True)
    unique_time = physical_time[unique_indices]
    unique_time[0] = 0.0
    unique_time[-1] = 1.0
    time_of_tau = CubicSpline(unique_tau, unique_time)
    return tau_of_time, time_of_tau


class _TorchCubicSpline(nn.Module):
    """Evaluate a SciPy cubic spline from Torch buffers on the input device."""

    def __init__(self, spline: CubicSpline):
        super().__init__()
        self.register_buffer("knots", torch.from_numpy(spline.x))
        self.register_buffer("coefficients", torch.from_numpy(spline.c))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        input_dtype = values.dtype
        flat_values = values.double().clamp(self.knots[0], self.knots[-1]).reshape(-1)
        intervals = torch.searchsorted(self.knots, flat_values, right=True) - 1
        intervals = intervals.clamp(0, self.knots.numel() - 2)
        offset = flat_values - self.knots[intervals]
        coefficients = self.coefficients[:, intervals]

        # SciPy stores c0*dx^3 + c1*dx^2 + c2*dx + c3 per knot interval.
        result = (
            (coefficients[0] * offset + coefficients[1]) * offset + coefficients[2]
        ) * offset + coefficients[3]
        return result.clamp(0.0, 1.0).reshape(values.shape).to(input_dtype)


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
        context_visible = is_context & (kv_idx <= anchor)

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
        use_time_reparameterization: bool = True,
        loss_decay_gamma: float = 7.0,
        boundary_probability: float = 0.0,
        block_time_max_exponent: float = 1.0,
        use_target_distribution: bool = False,
        use_flex_attention: bool = True,
        ode_epsilon: float = 1e-5,
    ):
        super().__init__()
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if num_anchors <= 0:
            raise ValueError(f"num_anchors must be positive, got {num_anchors}")
        if not math.isfinite(loss_decay_gamma) or loss_decay_gamma <= 0.0:
            raise ValueError(
                f"loss_decay_gamma must be finite and positive, got {loss_decay_gamma}"
            )
        if not 0.0 <= boundary_probability < 1.0:
            raise ValueError(f"boundary_probability must be in [0, 1), got {boundary_probability}")
        if not math.isfinite(block_time_max_exponent) or block_time_max_exponent < 1.0:
            raise ValueError(
                "block_time_max_exponent must be finite and at least 1, got "
                f"{block_time_max_exponent}"
            )
        if ode_epsilon <= 0:
            raise ValueError(f"ode_epsilon must be positive, got {ode_epsilon}")

        self.draft_model = draft_model
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.boundary_probability = boundary_probability
        self.block_time_max_exponent = block_time_max_exponent
        self.use_target_distribution = use_target_distribution
        self.use_flex_attention = use_flex_attention
        self.ode_epsilon = ode_epsilon

        if use_time_reparameterization:
            # This analytic schedule assumes one-hot endpoints. It remains an
            # approximation when target-distribution endpoints are enabled.
            tau_of_time, time_of_tau = _build_flm_time_schedule(draft_model.config.vocab_size)
        else:
            identity = np.array([0.0, 1.0])
            tau_of_time = CubicSpline(identity, identity)
            time_of_tau = CubicSpline(identity, identity)
        self.tau_of_time = _TorchCubicSpline(tau_of_time)
        self.time_of_tau = _TorchCubicSpline(time_of_tau)
        self.register_buffer(
            "block_time_exponents",
            torch.linspace(1.0, block_time_max_exponent, block_size),
            persistent=False,
        )
        self.register_buffer(
            "loss_decay_weights",
            torch.exp(-torch.arange(block_size, dtype=torch.float32) / loss_decay_gamma),
            persistent=False,
        )

    def tau_to_time(self, tau: torch.Tensor) -> torch.Tensor:
        # Evaluate the precomputed inverse LUT instead of the vocabulary CDF.
        return self.time_of_tau(tau)

    def time_to_tau(self, physical_time: torch.Tensor) -> torch.Tensor:
        # The forward LUT is useful for schedule inspection and inverse checks.
        return self.tau_of_time(physical_time)

    def _valid_anchor_mask(self, loss_mask: torch.Tensor) -> torch.Tensor:
        sequence_len = loss_mask.shape[1]
        if sequence_len <= self.block_size:
            return torch.zeros(
                loss_mask.shape[0],
                0,
                dtype=torch.bool,
                device=loss_mask.device,
            )
        target_windows = loss_mask.unfold(1, self.block_size, 1)
        return (target_windows[:, 1:] > 0.5).all(dim=-1)

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
            1,
            self.block_size + 1,
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
        # A causal hidden state at anchor + k predicts token anchor + k + 1.
        hidden_offsets = torch.arange(
            self.block_size,
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

    def _corrupt_targets(
        self,
        target_ids: torch.Tensor,
        physical_time: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = F.one_hot(target_ids, self.draft_model.config.vocab_size).float()
        return self._corrupt_distribution(target, physical_time, noise)

    def _corrupt_distribution(
        self,
        target: torch.Tensor,
        physical_time: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(target, dtype=torch.float32)
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
        return (1.0 - time) * noise.float() + time * target.float()

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

    def _expand_block_times(self, block_times: torch.Tensor) -> torch.Tensor:
        # Every token in a flow block shares its anchor's sampled time.
        return block_times.unsqueeze(-1).expand(-1, -1, self.block_size).flatten(1)

    def _block_time_coordinates(
        self,
        block_tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-position model time and physical interpolation time."""

        if self.block_time_max_exponent == 1.0:
            token_tau = block_tau.unsqueeze(-1).expand(-1, -1, self.block_size)
            token_time = self.tau_to_time(block_tau).unsqueeze(-1).expand_as(token_tau)
            return token_tau, token_time

        # Tau measures denoising difficulty, so a position-dependent power keeps
        # later tokens harder without amplifying the shift through the LUT.
        token_tau = block_tau.unsqueeze(-1).pow(self.block_time_exponents)
        token_time = self.tau_to_time(token_tau)
        return token_tau, token_time

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
        if (lm_head_weight is None) != (last_hidden_states is None):
            raise ValueError("lm_head_weight and last_hidden_states must be provided together")
        if last_hidden_states is not None:
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
        if self.use_target_distribution:
            if lm_head_weight is None or last_hidden_states is None:
                raise ValueError(
                    "use_target_distribution=True requires lm_head_weight and last_hidden_states"
                )
            with torch.no_grad():
                target_distribution = (
                    self._target_logits(
                        last_hidden_states,
                        lm_head_weight,
                        anchor_positions,
                    )
                    .float()
                    .softmax(dim=-1)
                )
        else:
            target_distribution = F.one_hot(
                dataset_target_ids,
                self.draft_model.config.vocab_size,
            ).float()

        tau = self._sample_training_times(batch_size, input_ids.device)
        token_tau, physical_time = self._block_time_coordinates(tau)
        flow_state = self._corrupt_distribution(
            target_distribution,
            physical_time,
        ).flatten(1, 2)
        model_times = token_tau.flatten(1, 2)

        context_positions, draft_positions = self._create_position_ids(
            anchor_positions, context_len
        )
        attention_mask = self._create_attention_mask(
            anchor_positions,
            block_keep_mask,
            context_len,
        )
        logits = self.draft_model(
            flow_state=flow_state,
            timesteps=model_times,
            context_feature=context_feature,
            draft_position_ids=draft_positions,
            context_position_ids=context_positions,
            block_mask=attention_mask,
        )

        draft_log_probabilities = logits.float().log_softmax(dim=-1).view_as(target_distribution)
        per_token_loss = -(target_distribution * draft_log_probabilities).sum(dim=-1)
        token_mask = block_keep_mask.unsqueeze(-1).expand_as(per_token_loss).float()
        token_count = token_mask.sum().clamp(min=1.0)
        position_weights = self.loss_decay_weights.view(1, 1, -1)
        objective_weights = token_mask * position_weights
        weighted_token_count = objective_weights.sum().clamp(min=1.0)

        # The previous objective mixed uniform CE with -log(E[L] / N), where
        # E[L] = sum_j prod_{i<=j} sum_v min(q_i(v), p_i(v)). Gamma-decayed CE
        # keeps the early-token priority without product-conditioned gradients.
        loss = (per_token_loss * objective_weights).sum() / weighted_token_count

        loss_components = {}
        with torch.no_grad():
            predictions = logits.argmax(dim=-1).view(batch_size, self.num_anchors, self.block_size)
            correct = (predictions == dataset_target_ids).float() * token_mask
            accuracy = correct.sum() / token_count
            count_per_position = token_mask.sum(dim=(0, 1))
            safe_count = count_per_position.clamp(min=1.0)
            loss_per_position = (per_token_loss * token_mask).sum(dim=(0, 1)) / safe_count
            accuracy_per_position = correct.sum(dim=(0, 1)) / safe_count

            prefix_survival = correct.bool().cumprod(dim=-1)
            valid_blocks = block_keep_mask.bool()
            block_match_length = (
                prefix_survival.sum(dim=-1) * valid_blocks
            ).sum() / valid_blocks.sum().clamp(min=1)
            loss_components = {
                "block_match_length": block_match_length.float(),
                "objective_loss": loss.detach(),
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
        num_steps: int,
        noise: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Integrate FLM and return ``[B, N, K, V]`` clean-token probabilities."""

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        batch_size, num_anchors = anchor_positions.shape
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
            self.draft_model.config.vocab_size,
        )
        if noise is None:
            state = torch.randn(
                expected_shape,
                device=anchor_positions.device,
                dtype=torch.float32,
                generator=generator,
            )
        elif noise.shape != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}, got {tuple(noise.shape)}")
        else:
            state = noise.float()
        state = state.flatten(1, 2)

        tau_grid = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            device=anchor_positions.device,
        )
        for step in range(num_steps):
            tau_current = tau_grid[step].expand(batch_size, num_anchors)
            tau_next = tau_grid[step + 1].expand(batch_size, num_anchors)
            token_tau_current, time_current = self._block_time_coordinates(tau_current)
            _, time_next = self._block_time_coordinates(tau_next)
            logits = self.draft_model(
                flow_state=state,
                timesteps=token_tau_current.flatten(1, 2),
                context_feature=context_feature,
                draft_position_ids=draft_positions,
                context_position_ids=context_positions,
                block_mask=attention_mask,
            )
            clean_probabilities = logits.float().softmax(dim=-1)

            if step == num_steps - 1:
                state = clean_probabilities
                break

            current = time_current.flatten(1, 2).unsqueeze(-1)
            delta = (time_next - time_current).flatten(1, 2).unsqueeze(-1)
            velocity = (clean_probabilities - state) / (1.0 - current + self.ode_epsilon)
            state = state + delta * velocity

        return state.view(expected_shape)

    @torch.no_grad()
    def generate(
        self,
        hidden_states_list: List[torch.Tensor],
        anchor_positions: torch.Tensor,
        num_steps: int,
        noise: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        probabilities = self.integrate(
            hidden_states_list=hidden_states_list,
            anchor_positions=anchor_positions,
            num_steps=num_steps,
            noise=noise,
            block_keep_mask=block_keep_mask,
            generator=generator,
        )
        return probabilities.argmax(dim=-1)

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
            num_steps=num_steps,
            block_keep_mask=block_keep_mask,
            generator=generator,
        )

        token_mask = block_keep_mask.unsqueeze(-1).expand_as(predictions)
        token_matches = predictions.eq(target_ids)
        prefix_matches = token_matches.cumprod(dim=-1)
        block_match_length = prefix_matches.sum(dim=-1)
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
