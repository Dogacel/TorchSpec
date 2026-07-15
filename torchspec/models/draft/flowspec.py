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

"""FlowSpec draft network: a target-conditioned flow language model.

Architecture provenance:

======================  ============  ==============================================
Component               Source        FlowSpec adaptation
======================  ============  ==============================================
Continuous input        FLM           Exact dense vocabulary-to-hidden projection
Time embedding          FLM           Sinusoidal biased MLP for packed draft blocks
Time modulation         FLM DiT       AdaLN-Zero decoder and final-layer modulation
Context conditioning    DFlash        Projected target hidden states as context KV
Attention               DFlash/Qwen   GQA, Q/K norm, RoPE, and context-plus-draft KV
MLP and normalization   DFlash/Qwen   SwiGLU and RMSNorm
Output                  FLM           Time-conditioned, zero-initialized vocab head
======================  ============  ==============================================

The network consumes a continuous state over vocabulary coordinates.
Draft queries attend to target context and their own packed block according to
the caller's block mask. The training/sampling wrapper owns the interpolant,
anchor selection, loss, and probability-flow integration.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class FlowSpecConfig(PretrainedConfig):
    """Configuration for the FlowSpec draft network."""

    model_type = "flowspec"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = None,
        vocab_size: int = 151936,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 40960,
        rope_theta: float = 1_000_000.0,
        time_condition_size: Optional[int] = None,
        time_frequency_size: int = 256,
        time_max_period: float = 10_000.0,
        num_target_layers: int = 5,
        target_hidden_size: int = 4096,
        target_num_hidden_layers: int = 36,
        target_layer_ids: Optional[List[int]] = None,
        output_bias: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {num_key_value_heads}")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads, "
                f"got {num_attention_heads} and {num_key_value_heads}"
            )
        if head_dim is None:
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_attention_heads when "
                    f"head_dim is omitted, got {hidden_size} and {num_attention_heads}"
                )
            head_dim = hidden_size // num_attention_heads
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        if time_condition_size is None:
            time_condition_size = hidden_size
        if time_condition_size <= 0:
            raise ValueError(f"time_condition_size must be positive, got {time_condition_size}")
        if time_frequency_size < 2 or time_frequency_size % 2 != 0:
            raise ValueError(
                "time_frequency_size must be an even integer greater than or equal "
                f"to 2, got {time_frequency_size}"
            )
        if time_max_period <= 0:
            raise ValueError(f"time_max_period must be positive, got {time_max_period}")
        if num_target_layers <= 0:
            raise ValueError(f"num_target_layers must be positive, got {num_target_layers}")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.time_condition_size = time_condition_size
        self.time_frequency_size = time_frequency_size
        self.time_max_period = time_max_period
        self.num_target_layers = num_target_layers
        self.target_hidden_size = target_hidden_size
        self.target_num_hidden_layers = target_num_hidden_layers
        self.target_layer_ids = target_layer_ids
        self.output_bias = output_bias


class FlowSpecRMSNorm(nn.Module):
    """Compute RMSNorm statistics in FP32 for mixed-precision stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FlowSpecTimestepEmbedding(nn.Module):
    """Embed fractional flow times with arbitrary leading dimensions."""

    def __init__(
        self,
        condition_size: int,
        frequency_size: int = 256,
        max_period: float = 10_000.0,
    ):
        super().__init__()
        if frequency_size < 2 or frequency_size % 2 != 0:
            raise ValueError(
                "frequency_size must be an even integer greater than or equal to 2, "
                f"got {frequency_size}"
            )
        self.frequency_size = frequency_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, condition_size, bias=True),
            nn.SiLU(),
            nn.Linear(condition_size, condition_size, bias=True),
        )

    def _frequency_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        arguments = timesteps.float().unsqueeze(-1) * frequencies
        return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        features = self._frequency_embedding(timesteps)
        return self.mlp(features.to(self.mlp[0].weight.dtype))


class FlowSpecVocabularyEmbedding(nn.Module):
    """Project token IDs or exact dense vocabulary-space states to hidden states."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            return F.embedding(inputs, self.weight)
        if inputs.ndim != 3:
            raise ValueError(
                f"vocabulary inputs must have shape [B, Q] or [B, Q, V], got {tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"flow state vocabulary dimension is {inputs.shape[-1]}, expected "
                f"{self.weight.shape[0]}"
            )

        projected = torch.matmul(inputs.float(), self.weight.float())
        return projected.to(inputs.dtype)


class FlowSpecRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings + 20, inv_freq.device, torch.float32)

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_seq_len_cached = seq_len
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        frequencies = torch.einsum("i,j->ij", positions, self.inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer(
            "cos_cached", embedding.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", embedding.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, hidden_states.device, hidden_states.dtype)
        return (
            self.cos_cached[:, :, :seq_len].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ),
            self.sin_cached[:, :, :seq_len].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ),
        )


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first = hidden_states[..., : hidden_states.shape[-1] // 2]
    second = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None].expand(
        batch, num_kv_heads, repetitions, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * repetitions, seq_len, head_dim)


class FlowSpecAttention(nn.Module):
    """Draft-query attention over target context KV and draft KV.

    The same K/V projections are used for both sources.  ``block_mask`` may be
    either a FlexAttention ``BlockMask`` or a dense SDPA-compatible tensor.  A
    missing mask intentionally means fully bidirectional attention and is only
    appropriate for isolated component tests.
    """

    def __init__(self, config: FlowSpecConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = FlowSpecRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = FlowSpecRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = FlowSpecRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        batch_size, draft_len, _ = draft_hidden.shape
        context_len = context_hidden.shape[1]

        query = self.q_proj(draft_hidden)
        query = query.view(batch_size, draft_len, self.num_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)

        context_key = self.k_proj(context_hidden)
        context_value = self.v_proj(context_hidden)
        draft_key = self.k_proj(draft_hidden)
        draft_value = self.v_proj(draft_hidden)

        key = torch.cat([context_key, draft_key], dim=1)
        value = torch.cat([context_value, draft_value], dim=1)
        total_len = context_len + draft_len

        key = key.view(batch_size, total_len, self.num_kv_heads, self.head_dim)
        key = self.k_norm(key).transpose(1, 2)
        value = value.view(batch_size, total_len, self.num_kv_heads, self.head_dim)
        value = value.transpose(1, 2)

        all_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        # A fixed cache length avoids synchronizing position IDs back to the host.
        rotary_len = self.rotary_emb.max_seq_len_cached
        cos, sin = self.rotary_emb(query, seq_len=rotary_len)
        cos = cos.squeeze(1).squeeze(0)
        sin = sin.squeeze(1).squeeze(0)

        query_cos = cos[draft_position_ids].unsqueeze(1)
        query_sin = sin[draft_position_ids].unsqueeze(1)
        query = query * query_cos + _rotate_half(query) * query_sin

        key_cos = cos[all_position_ids].unsqueeze(1)
        key_sin = sin[all_position_ids].unsqueeze(1)
        key = key * key_cos + _rotate_half(key) * key_sin

        if block_mask is not None and not torch.is_tensor(block_mask):
            from torchspec.models.ops.flex_attention import compile_friendly_flex_attention

            attention_output = compile_friendly_flex_attention(
                query=query,
                key=key,
                value=value,
                block_mask=block_mask,
                enable_gqa=True,
            )
        else:
            key = _repeat_kv(key, self.num_kv_groups)
            value = _repeat_kv(value, self.num_kv_groups)
            attention_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=block_mask,
                is_causal=False,
                dropout_p=0.0,
            )

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, draft_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attention_output)


class FlowSpecMLP(nn.Module):
    def __init__(self, config: FlowSpecConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _modulate(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return hidden_states * (1 + scale) + shift


class FlowSpecDecoderLayer(nn.Module):
    """Target-conditioned decoder layer with per-position AdaLN-Zero."""

    def __init__(self, config: FlowSpecConfig):
        super().__init__()
        self.self_attn = FlowSpecAttention(config)
        self.mlp = FlowSpecMLP(config)
        self.input_layernorm = FlowSpecRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FlowSpecRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ada_ln_modulation = nn.Linear(
            config.time_condition_size, 6 * config.hidden_size, bias=True
        )
        nn.init.zeros_(self.ada_ln_modulation.weight)
        nn.init.zeros_(self.ada_ln_modulation.bias)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        time_condition: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.ada_ln_modulation(time_condition).chunk(6, dim=-1)

        attention_input = _modulate(
            self.input_layernorm(draft_hidden), attention_shift, attention_scale
        )
        attention_output = self.self_attn(
            draft_hidden=attention_input,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
        )
        draft_hidden = draft_hidden + attention_gate * attention_output

        mlp_input = _modulate(self.post_attention_layernorm(draft_hidden), mlp_shift, mlp_scale)
        return draft_hidden + mlp_gate * self.mlp(mlp_input)


class FlowSpecFinalLayer(nn.Module):
    """Time-conditioned normalization and clean-token posterior logits."""

    def __init__(self, config: FlowSpecConfig):
        super().__init__()
        self.norm = FlowSpecRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ada_ln_modulation = nn.Linear(
            config.time_condition_size, 2 * config.hidden_size, bias=True
        )
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=config.output_bias)

        nn.init.zeros_(self.ada_ln_modulation.weight)
        nn.init.zeros_(self.ada_ln_modulation.bias)
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.ada_ln_modulation(time_condition).chunk(2, dim=-1)
        hidden_states = _modulate(self.norm(hidden_states), shift, scale)
        return self.linear(hidden_states)


def build_target_layer_ids(num_target_layers: int, num_hidden_layers: int) -> List[int]:
    """Choose uniformly spaced target layers using the DFlash convention."""

    if num_target_layers <= 0:
        raise ValueError(f"num_target_layers must be positive, got {num_target_layers}")
    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
    if num_target_layers == 1:
        return [num_hidden_layers // 2]

    start = 1
    end = num_hidden_layers - 3
    if end < start:
        raise ValueError(
            "target model must have at least four hidden layers when selecting "
            f"multiple target layers, got {num_hidden_layers}"
        )
    span = end - start
    return [
        int(round(start + i * span / (num_target_layers - 1))) for i in range(num_target_layers)
    ]


class FlowSpecDraftModel(PreTrainedModel):
    """Dense-vocabulary FLM denoiser conditioned on target-model features.

    Public forward contract:

    * ``flow_state``: ``[B, Q, V]`` continuous vocabulary-space state.
    * ``timesteps``: scalar, ``[B]``, ``[B, 1]``, or ``[B, Q]`` flow time.
    * ``context_feature``: ``[B, S, D]`` projected target context.
    * return value: ``[B, Q, V]`` clean-token posterior logits.
    """

    config_class = FlowSpecConfig
    base_model_prefix = "flowspec"

    def __init__(self, config: FlowSpecConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_target_layers = config.num_target_layers

        self.target_layer_ids = config.target_layer_ids
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(
                config.num_target_layers, config.target_num_hidden_layers
            )
        elif len(self.target_layer_ids) != self.num_target_layers:
            raise ValueError(
                "target_layer_ids must contain num_target_layers entries, "
                f"got {len(self.target_layer_ids)} and {self.num_target_layers}"
            )

        context_input_size = config.num_target_layers * config.target_hidden_size
        self.context_proj = nn.Linear(context_input_size, config.hidden_size, bias=False)
        self.context_norm = FlowSpecRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vocab_embed = FlowSpecVocabularyEmbedding(config.vocab_size, config.hidden_size)
        self.time_embed = FlowSpecTimestepEmbedding(
            condition_size=config.time_condition_size,
            frequency_size=config.time_frequency_size,
            max_period=config.time_max_period,
        )
        self.layers = nn.ModuleList(
            [FlowSpecDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.output_layer = FlowSpecFinalLayer(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.vocab_embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.vocab_embed = value

    def get_output_embeddings(self) -> nn.Module:
        return self.output_layer.linear

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.output_layer.linear = value

    def extract_context_feature(
        self,
        all_hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(all_hidden_states) != self.num_target_layers:
            raise ValueError(
                f"expected {self.num_target_layers} target hidden states, "
                f"got {len(all_hidden_states)}"
            )
        concatenated = torch.cat(all_hidden_states, dim=-1)
        concatenated = concatenated.to(self.context_proj.weight.dtype)
        return self.context_norm(self.context_proj(concatenated))

    def _prepare_time_condition(
        self,
        timesteps: torch.Tensor,
        batch_size: int,
        draft_len: int,
    ) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size, draft_len)
        elif timesteps.shape == (batch_size,):
            timesteps = timesteps[:, None].expand(-1, draft_len)
        elif timesteps.shape == (batch_size, 1):
            timesteps = timesteps.expand(-1, draft_len)
        elif timesteps.shape != (batch_size, draft_len):
            raise ValueError(
                "timesteps must be scalar or have shape [B], [B, 1], or [B, Q], "
                f"got {tuple(timesteps.shape)} for B={batch_size}, Q={draft_len}"
            )
        return F.silu(self.time_embed(timesteps))

    def forward(
        self,
        flow_state: torch.Tensor,
        timesteps: torch.Tensor,
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        if flow_state.ndim != 3:
            raise ValueError(f"flow_state must have shape [B, Q, V], got {tuple(flow_state.shape)}")
        batch_size, draft_len, vocab_size = flow_state.shape
        if vocab_size != self.config.vocab_size:
            raise ValueError(
                f"flow_state vocabulary dimension is {vocab_size}, expected "
                f"{self.config.vocab_size}"
            )
        if context_feature.shape[:2] != context_position_ids.shape:
            raise ValueError(
                "context_feature and context_position_ids disagree: "
                f"{tuple(context_feature.shape)} vs {tuple(context_position_ids.shape)}"
            )
        if context_feature.shape[0] != batch_size:
            raise ValueError(
                f"context batch size is {context_feature.shape[0]}, expected {batch_size}"
            )
        if context_feature.shape[-1] != self.hidden_size:
            raise ValueError(
                f"context hidden size is {context_feature.shape[-1]}, expected {self.hidden_size}"
            )
        if draft_position_ids.shape != (batch_size, draft_len):
            raise ValueError(
                f"draft_position_ids must have shape {(batch_size, draft_len)}, got "
                f"{tuple(draft_position_ids.shape)}"
            )

        draft_hidden = self.vocab_embed(flow_state).to(context_feature.dtype)
        timesteps = timesteps.to(device=flow_state.device)
        time_condition = self._prepare_time_condition(timesteps, batch_size, draft_len).to(
            draft_hidden.dtype
        )

        for layer in self.layers:
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=context_feature,
                time_condition=time_condition,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
            )

        return self.output_layer(draft_hidden, time_condition)
