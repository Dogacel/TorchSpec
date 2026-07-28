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

"""FlowSpec endpoint-noise distributions and time reparameterizations."""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr

SIMPLEX_NOISE_LOGIT_SCALE = 34.02838408946991
CENTERED_TIME_LOGIT_SCALE = 0.3166523850475464
TARGET_MEAN_SQUARED_L2 = 0.8491979241371155
SIMPLEX_VISIBILITY_MEAN = 0.08306463807821274
SIMPLEX_VISIBILITY_SCALE = 1.0227549076080322
AFFINE_VISIBILITY_MEAN = -4.458788685107652
AFFINE_VISIBILITY_SCALE = 0.33961935034991586
SOFT_TARGET_LOGIT_NORMAL_WEIGHTS = (0.7558941005427922, 0.2441058994572079)
SOFT_TARGET_LOGIT_NORMAL_MEANS = (1.552638264898977, 1.8207033834003472)
SOFT_TARGET_LOGIT_NORMAL_SCALES = (0.2488555344031524, 0.596577512661047)

TIME_SCHEDULES = frozenset(
    {
        "onehot_lut",
        "simplex_sharp",
        "simplex_visibility",
        "affine_balanced",
        "soft_target_logit_normal_mixture",
        "soft_target_argmax_uniform_blend",
        "identity",
    }
)
NOISE_TYPES = frozenset({"standard_gaussian", "simplex", "affine_gaussian"})


def _standard_normal_quantile(probability: torch.Tensor) -> torch.Tensor:
    """Evaluate the standard-normal quantile with Acklam's rational approximation."""
    eps = torch.finfo(probability.dtype).eps
    probability = probability.clamp(eps, 1.0 - eps)
    q = probability - 0.5
    r = q.square()
    central = (
        (
            (
                (
                    ((-39.69683028665376 * r + 220.9460984245205) * r - 275.9285104469687) * r
                    + 138.3577518672690
                )
                * r
                - 30.66479806614716
            )
            * r
            + 2.506628277459239
        )
        * q
    ) / (
        (
            (
                ((-54.47609879822406 * r + 161.5858368580409) * r - 155.6989798598866) * r
                + 66.80131188771972
            )
            * r
            - 13.28068155288572
        )
        * r
        + 1.0
    )
    tail_probability = torch.where(probability < 0.5, probability, 1.0 - probability)
    tail_q = torch.sqrt(-2.0 * torch.log(tail_probability))
    tail = (
        (
            (
                ((-0.007784894002430293 * tail_q - 0.3223964580411365) * tail_q - 2.400758277161838)
                * tail_q
                - 2.549732539343734
            )
            * tail_q
            + 4.374664141464968
        )
        * tail_q
        + 2.938163982698783
    ) / (
        (
            ((0.007784695709041462 * tail_q + 0.3224671290700398) * tail_q + 2.445134137142996)
            * tail_q
            + 3.754408661907416
        )
        * tail_q
        + 1.0
    )
    tail = torch.where(probability < 0.5, tail, -tail)
    return torch.where(q.abs() <= 0.47575, central, tail)


def _standard_normal_cdf(value: torch.Tensor) -> torch.Tensor:
    """Smooth normal-CDF approximation using the GELU tanh polynomial."""
    scaled = math.sqrt(2.0 / math.pi) * (value + 0.044715 * value.pow(3))
    return 0.5 * (1.0 + torch.tanh(scaled))


class RecognitionTimeSchedule(nn.Module):
    """Map uniform tau through a fitted logit-normal physical-time marginal."""

    def __init__(self, mean: float = 0.0, scale: float = CENTERED_TIME_LOGIT_SCALE):
        super().__init__()
        self.mean = mean
        self.scale = scale

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        input_dtype = tau.dtype
        tau = tau.float().clamp(0.0, 1.0)
        normal_quantile = _standard_normal_quantile(tau)
        physical_time = torch.sigmoid(self.mean + self.scale * normal_quantile)
        physical_time = torch.where(tau <= 0.0, torch.zeros_like(physical_time), physical_time)
        return torch.where(tau >= 1.0, torch.ones_like(physical_time), physical_time).to(
            input_dtype
        )


class BlendedRecognitionTimeSchedule(nn.Module):
    """Invert a smooth mixture of uniform-time and recognizability CDFs."""

    def __init__(self, mean: float, scale: float, recognition_weight: float = 0.5):
        super().__init__()
        self.mean = mean
        self.scale = scale
        self.recognition_weight = recognition_weight

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        input_dtype = tau.dtype
        tau = tau.float().clamp(0.0, 1.0)
        low = torch.zeros_like(tau)
        high = torch.ones_like(tau)
        for _ in range(24):
            physical_time = 0.5 * (low + high)
            clipped = physical_time.clamp(1e-7, 1.0 - 1e-7)
            standardized = (torch.logit(clipped) - self.mean) / self.scale
            recognition_cdf = _standard_normal_cdf(standardized)
            cdf = (
                self.recognition_weight * recognition_cdf
                + (1.0 - self.recognition_weight) * physical_time
            )
            low = torch.where(cdf < tau, physical_time, low)
            high = torch.where(cdf >= tau, physical_time, high)
        physical_time = 0.5 * (low + high)
        physical_time = torch.where(tau <= 0.0, torch.zeros_like(physical_time), physical_time)
        return torch.where(tau >= 1.0, torch.ones_like(physical_time), physical_time).to(
            input_dtype
        )


class LogitNormalMixtureTimeSchedule(nn.Module):
    """Invert a smooth mixture fitted to soft-target recognition thresholds."""

    def __init__(
        self,
        weights: Tuple[float, ...] = SOFT_TARGET_LOGIT_NORMAL_WEIGHTS,
        means: Tuple[float, ...] = SOFT_TARGET_LOGIT_NORMAL_MEANS,
        scales: Tuple[float, ...] = SOFT_TARGET_LOGIT_NORMAL_SCALES,
    ):
        super().__init__()
        if not len(weights) == len(means) == len(scales) or not weights:
            raise ValueError("weights, means, and scales must have the same non-zero length")
        if any(weight <= 0.0 for weight in weights) or not math.isclose(sum(weights), 1.0):
            raise ValueError("weights must be positive and sum to one")
        if any(scale <= 0.0 for scale in scales):
            raise ValueError("scales must be positive")
        self.weights = weights
        self.means = means
        self.scales = scales

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        input_dtype = tau.dtype
        tau = tau.float().clamp(0.0, 1.0)
        low = torch.zeros_like(tau)
        high = torch.ones_like(tau)
        for _ in range(24):
            physical_time = 0.5 * (low + high)
            log_odds = torch.logit(physical_time.clamp(1e-7, 1.0 - 1e-7))
            cdf = sum(
                weight * torch.special.ndtr((log_odds - mean) / scale)
                for weight, mean, scale in zip(self.weights, self.means, self.scales)
            )
            low = torch.where(cdf < tau, physical_time, low)
            high = torch.where(cdf >= tau, physical_time, high)
        physical_time = 0.5 * (low + high)
        physical_time = torch.where(tau <= 0.0, torch.zeros_like(physical_time), physical_time)
        return torch.where(tau >= 1.0, torch.ones_like(physical_time), physical_time).to(
            input_dtype
        )


class UniformBlendedTimeSchedule(nn.Module):
    """Blend a data-calibrated time map with uniform physical time."""

    def __init__(self, base_schedule: nn.Module, base_weight: float = 0.9):
        super().__init__()
        if not 0.0 <= base_weight <= 1.0:
            raise ValueError(f"base_weight must be in [0, 1], got {base_weight}")
        self.base_schedule = base_schedule
        self.base_weight = base_weight

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.base_weight * self.base_schedule(tau) + (1.0 - self.base_weight) * tau


def build_onehot_time_schedule(
    vocab_size: int,
    num_points: int = 1_000,
    quadrature_points: int = 100,
) -> CubicSpline:
    """Build the original FLM inverse recognizability spline."""
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    nodes = nodes * math.sqrt(2.0)
    weights = weights / math.sqrt(math.pi)
    physical_time = np.linspace(0.0, 1.0, num_points)
    standardized_signal = physical_time / np.maximum(1.0 - physical_time, 1e-12)
    log_cdf = log_ndtr(standardized_signal[:, None] + nodes[None, :])
    correct_probability = np.sum(weights[None, :] * np.exp((vocab_size - 1) * log_cdf), axis=-1)
    tau = vocab_size / (vocab_size - 1.0) * (correct_probability - 1.0 / vocab_size)
    tau = np.maximum.accumulate(np.clip(tau + (physical_time - 1.0) * 1e-10, 0.0, 1.0))
    tau[0], tau[-1] = 0.0, 1.0
    unique_tau, unique_indices = np.unique(tau, return_index=True)
    unique_time = physical_time[unique_indices]
    unique_time[0], unique_time[-1] = 0.0, 1.0
    return CubicSpline(unique_tau, unique_time)


class TorchCubicSpline(nn.Module):
    def __init__(self, spline: CubicSpline):
        super().__init__()
        self.register_buffer("knots", torch.from_numpy(spline.x))
        self.register_buffer("coefficients", torch.from_numpy(spline.c))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        input_dtype = values.dtype
        flat = values.double().clamp(self.knots[0], self.knots[-1]).reshape(-1)
        intervals = (torch.searchsorted(self.knots, flat, right=True) - 1).clamp(
            0, self.knots.numel() - 2
        )
        offset = flat - self.knots[intervals]
        coefficients = self.coefficients[:, intervals]
        result = (
            (coefficients[0] * offset + coefficients[1]) * offset + coefficients[2]
        ) * offset + coefficients[3]
        return result.clamp(0.0, 1.0).reshape(values.shape).to(input_dtype)


def make_time_schedule(name: str, vocab_size: int) -> nn.Module:
    if name == "onehot_lut":
        return TorchCubicSpline(build_onehot_time_schedule(vocab_size))
    if name == "simplex_sharp":
        return RecognitionTimeSchedule()
    if name == "simplex_visibility":
        return RecognitionTimeSchedule(SIMPLEX_VISIBILITY_MEAN, SIMPLEX_VISIBILITY_SCALE)
    if name == "affine_balanced":
        return BlendedRecognitionTimeSchedule(AFFINE_VISIBILITY_MEAN, AFFINE_VISIBILITY_SCALE)
    if name == "soft_target_logit_normal_mixture":
        return LogitNormalMixtureTimeSchedule()
    if name == "soft_target_argmax_uniform_blend":
        return UniformBlendedTimeSchedule(LogitNormalMixtureTimeSchedule())
    if name == "identity":
        return nn.Identity()
    raise ValueError(
        f"unknown FlowSpec time schedule {name!r}; expected one of {sorted(TIME_SCHEDULES)}"
    )


def sample_noise(
    noise_type: str,
    shape: torch.Size | Tuple[int, ...],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a FlowSpec source state in vocabulary coordinates."""
    if noise_type not in NOISE_TYPES:
        raise ValueError(
            f"unknown FlowSpec noise type {noise_type!r}; expected one of {sorted(NOISE_TYPES)}"
        )
    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    if noise_type == "simplex":
        return (SIMPLEX_NOISE_LOGIT_SCALE * noise).softmax(dim=-1)
    if noise_type == "standard_gaussian":
        return noise

    vocab_size = shape[-1]
    coordinate_mean = 1.0 / vocab_size
    sigma = math.sqrt((TARGET_MEAN_SQUARED_L2 - coordinate_mean) / (vocab_size - 1))
    return sigma * (noise - noise.mean(dim=-1, keepdim=True)) + coordinate_mean
