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

"""FlowSpec draft model: a flow-matching speculator.

The DFlash transformer skeleton (dual-source KV: context features from the
target model + the draft block itself) is reused as a diffusion-transformer
velocity network:

  - The draft stream carries NOISY target-hidden states (one per future
    position) instead of MASK-token embeddings; slot 0 carries the known
    anchor token's embedding.
  - Every decoder layer is modulated by the flow time t via adaLN-Zero
    (DiT-style): the layer is identity at init and learns how much attention/
    MLP output to inject per timestep.
  - The output is a velocity in target-hidden space: the direction from the
    current noisy state toward the clean target hidden.

Sampling runs this whole backbone K times (context features are computed once
and reused — only the block stream is recomputed), integrating the 15 unknown
positions jointly from Gaussian noise to predicted hidden states, which the
frozen target LM head then decodes into draft tokens.

The flow state is the target's LM-head-ready hidden (full size, standardized
with EMA statistics) — deliberately NOT compressed: LLM hiddens are already a
bottleneck (unlike redundant pixels), and a smaller state would cap token
discrimination through the vocabulary projection.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from torchspec.models.draft.dflash import (
    DFlashAttention,
    DFlashConfig,
    DFlashDraftModel,
    DFlashMLP,
    DFlashRMSNorm,
    build_target_layer_ids,
)


class FlowSpecConfig(DFlashConfig):
    """Configuration for the FlowSpec draft model. Extends :class:`DFlashConfig`."""

    model_type = "flowspec"

    def __init__(
        self,
        fuse_embedding: bool = True,
        time_embed_dim: int = 256,
        flow_t_mean: float = 0.0,
        flow_t_std: float = 1.0,
        flow_time_shift: float = 1.0,
        flow_sample_steps: int = 16,
        flow_corrective_frac: float = 0.5,
        flow_corrupt_alpha_min: float = 0.3,
        flow_source: str = "anchor",
        flow_source_sigma: float = 0.25,
        flow_t_slope: float = 0.35,
        flow_t_jitter: float = 0.05,
        flow_t_per_position: bool = True,
        flow_sample_lead: int = 4,
        flow_state: str = "hidden",
        flow_state_dim: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Include the token embedding as an extra slice in the context fusion
        # (EAGLE-3 style): deep hiddens smear the discrete token identity; the
        # embedding is its unambiguous record.
        self.fuse_embedding = fuse_embedding
        self.time_embed_dim = time_embed_dim
        # Training-time t distribution: t = sigmoid(N(mean, std)) — logit-normal,
        # concentrated at mid-noise where the trajectory bends (SD3 recipe),
        # then shifted by flow_time_shift (t' = s*t / (1 + (s-1)*t); 1.0 = off).
        self.flow_t_mean = flow_t_mean
        self.flow_t_std = flow_t_std
        self.flow_time_shift = flow_time_shift
        # Euler steps used by the eval-time sampler.
        self.flow_sample_steps = flow_sample_steps
        # Corrective-field training: with this probability per block, build x_t
        # from a CORRUPTED clean target (blend with another position's hidden,
        # alpha in [flow_corrupt_alpha_min, 1]) while the velocity target still
        # points at the TRUE target. Teaches the model to repair its own
        # imperfect mid-sampling estimates — without it, multi-step sampling
        # feeds the model off-distribution states it has never learned to fix
        # (exposure bias between denoising steps; cf. Analog Bits
        # self-conditioning). Validated on the toy check: sampled accuracy
        # 0.90 flat across step counts vs 0.54→0.25 degrading without it.
        self.flow_corrective_frac = flow_corrective_frac
        self.flow_corrupt_alpha_min = flow_corrupt_alpha_min
        # Source distribution for the flow (what x_0 is):
        #   "anchor" — the CURRENT hidden state (the anchor position's target
        #     hidden) broadcast to every slot, plus flow_source_sigma * noise:
        #     the flow is the trajectory of the model's state as text continues.
        #     t=0 is then the most informative single vector available, removing
        #     the informationless-bootstrap failure of a pure-noise source.
        #   "noise" — standard N(0, I) (sigma ignored; the v2 formulation).
        self.flow_source = flow_source
        self.flow_source_sigma = flow_source_sigma
        # Depth-graded per-position time (diffusion-forcing / rolling-diffusion
        # style): within a block, slot k trains at t_k = clamp(t_base -
        # flow_t_slope * k/(block-1) + jitter). Nearer future = less noise;
        # matches how uncertainty actually grows with horizon and concentrates
        # learning where acceptance is won. flow_t_per_position=False restores
        # a single shared t per block.
        self.flow_t_slope = flow_t_slope
        self.flow_t_jitter = flow_t_jitter
        self.flow_t_per_position = flow_t_per_position
        # Staggered sampler: front slots run ahead by up to flow_sample_lead
        # steps (reach t=1 early and freeze) while deeper slots keep refining.
        self.flow_sample_lead = flow_sample_lead
        # "hidden": the flow state is the target's last hidden (target_hidden_size).
        # "chunk_latent": the state is a CALM-style chunk-autoencoder latent of
        #   flow_state_dim dims (source = E(last 4 committed tokens), target =
        #   E(next 4 tokens)); decoding goes through the AE decoder.
        self.flow_state = flow_state
        self.flow_state_dim = flow_state_dim


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of flow time t in [0, 1]; t: [N] -> [N, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=t.device)
        * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FlowSpecDecoderLayer(nn.Module):
    """DFlash decoder layer with adaLN-Zero time modulation (DiT-style).

    Reuses DFlashAttention (dual-source KV) and DFlashMLP unchanged; the time
    condition produces per-token shift/scale/gate for both sublayers. Gates are
    zero at init, so the layer starts as identity.
    """

    def __init__(self, config):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ada_mod = nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        nn.init.zeros_(self.ada_mod.weight)
        nn.init.zeros_(self.ada_mod.bias)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        time_cond: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_mod(time_cond).chunk(
            6, dim=-1
        )

        residual = draft_hidden
        h = self.input_layernorm(draft_hidden) * (1.0 + scale_a) + shift_a
        h = self.self_attn(
            draft_hidden=h,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
        )
        draft_hidden = residual + gate_a * h

        residual = draft_hidden
        h = self.post_attention_layernorm(draft_hidden) * (1.0 + scale_m) + shift_m
        draft_hidden = residual + gate_m * self.mlp(h)

        return draft_hidden


class FlowSpecDraftModel(PreTrainedModel):
    """FlowSpec velocity network.

    Trainable parameters:
      - context_proj: fuses captured target hiddens (+ token embeddings) into
        context features (computed once per draft; reused across flow steps)
      - x_in: projects the noisy flow state into the model stream
      - time_mlp: flow-time conditioning vector for adaLN modulation
      - N FlowSpecDecoderLayers (dual-source attention + MLP + adaLN-Zero)
      - v_out: velocity head (zero-init)

    Frozen (from target): embed_tokens. The LM head is external (loaded
    separately in the trainer) and only used to decode finished states.
    """

    config_class = FlowSpecConfig

    def __init__(self, config: FlowSpecConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.target_hidden_size = target_hidden_size
        num_target_layers = getattr(config, "num_target_layers", 5)
        self.num_target_layers = num_target_layers
        self.fuse_embedding = bool(getattr(config, "fuse_embedding", True))

        target_num_hidden = getattr(config, "target_num_hidden_layers", 36)
        self.target_layer_ids = getattr(config, "target_layer_ids", None)
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(num_target_layers, target_num_hidden)

        fusion_in_dim = num_target_layers * target_hidden_size + (
            self.hidden_size if self.fuse_embedding else 0
        )
        self.context_proj = nn.Linear(fusion_in_dim, self.hidden_size, bias=False)
        self.context_norm = DFlashRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Token embedding (shared from target, frozen): anchor slot + context fusion
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Flow state in/out. State dim = target hidden by default; chunk-latent
        # mode uses the autoencoder's latent size instead.
        self.state_dim = int(getattr(config, "flow_state_dim", 0)) or target_hidden_size
        self.x_in = nn.Linear(self.state_dim, self.hidden_size, bias=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(getattr(config, "time_embed_dim", 256), self.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )
        # Input-additive noise-level embedding: with per-position (graded) t,
        # neighbors must SEE each other's noise levels through attention K/V —
        # adaLN alone modulates a token's own computation but does not announce
        # its noise level to others. Added directly to each slot's input.
        self.t_in = nn.Linear(getattr(config, "time_embed_dim", 256), self.hidden_size, bias=False)
        self.layers = nn.ModuleList([FlowSpecDecoderLayer(config) for _ in range(self.num_layers)])
        self.final_norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.v_out = nn.Linear(self.hidden_size, self.state_dim, bias=False)
        nn.init.zeros_(self.v_out.weight)

        # Running standardization stats of the target hidden space (EMA, updated
        # by the training wrapper); the flow state lives in standardized space.
        self.register_buffer("h_mean", torch.zeros(self.state_dim), persistent=True)
        self.register_buffer("h_var", torch.ones(self.state_dim), persistent=True)
        self.register_buffer(
            "stats_initialized", torch.zeros((), dtype=torch.bool), persistent=True
        )
        self.stats_momentum = 0.99

    # Reuse DFlash's target-embedding loading/freezing verbatim.
    load_embedding = DFlashDraftModel.load_embedding
    freeze_embedding = DFlashDraftModel.freeze_embedding

    @torch.no_grad()
    def load_context_proj(self, safetensors_path: str) -> None:
        """Warm-start the context fusion (fc + norm) from a trained DFlash
        checkpoint (e.g. z-lab/Qwen3-8B-DFlash-b16). Requires fuse_embedding
        to be False so the fusion input is the same 5-layer concat DFlash uses.
        """
        from safetensors import safe_open

        if self.fuse_embedding:
            raise ValueError(
                "load_context_proj requires fuse_embedding=False (DFlash's fc "
                "consumes exactly the captured target layers, no embedding slice)."
            )
        with safe_open(safetensors_path, framework="pt") as f:
            keys = set(f.keys())

            def find(*suffixes: str) -> str:
                for suffix in suffixes:
                    matches = [k for k in keys if k.endswith(suffix)]
                    if len(matches) == 1:
                        return matches[0]
                raise KeyError(f"No unique key for any of {suffixes!r} in checkpoint")

            # SpecForge/TorchSpec name: context_proj/context_norm;
            # z-lab release name: fc/hidden_norm.
            proj = f.get_tensor(find("context_proj.weight", "fc.weight"))
            norm = f.get_tensor(find("context_norm.weight", "hidden_norm.weight"))
            if proj.shape != self.context_proj.weight.shape:
                raise ValueError(
                    f"fc shape mismatch: checkpoint {tuple(proj.shape)} vs "
                    f"model {tuple(self.context_proj.weight.shape)}"
                )
            self.context_proj.weight.copy_(proj)
            self.context_norm.weight.copy_(norm)

    def freeze_context_proj(self) -> None:
        """Freeze the context fusion so the flow trains against a stationary,
        pre-organized conditioning basis (like a frozen text encoder in image
        models)."""
        self.context_proj.weight.requires_grad = False
        self.context_norm.weight.requires_grad = False

    @torch.no_grad()
    def update_stats(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        mean = mean.to(torch.float32)
        var = var.to(torch.float32)
        if not bool(self.stats_initialized):
            new_mean, new_var = mean, var
            self.stats_initialized.fill_(True)
        else:
            m = self.stats_momentum
            new_mean = m * self.h_mean.float() + (1.0 - m) * mean
            new_var = m * self.h_var.float() + (1.0 - m) * var
        self.h_mean.copy_(new_mean.to(self.h_mean.dtype))
        self.h_var.copy_(new_var.to(self.h_var.dtype))

    def _std(self) -> torch.Tensor:
        return self.h_var.float().clamp(min=1e-6).sqrt()

    def standardize(self, h: torch.Tensor) -> torch.Tensor:
        return (h.float() - self.h_mean.float()) / self._std()

    def unstandardize(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * self._std() + self.h_mean.float()

    def extract_context_feature(
        self,
        all_hidden_states: List[torch.Tensor],
        token_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse captured target hiddens (+ token embeddings) into context features.

        Args:
            all_hidden_states: list of [B, seq_len, D] tensors from target layers
            token_embeds: [B, seq_len, D] token embeddings (when fuse_embedding)

        Returns:
            context_feature: [B, seq_len, hidden_size]
        """
        sources = list(all_hidden_states)
        if self.fuse_embedding:
            if token_embeds is None:
                raise ValueError("fuse_embedding=True requires token_embeds")
            sources.append(token_embeds)
        concatenated = torch.cat(sources, dim=-1).to(self.context_proj.weight.dtype)
        return self.context_norm(self.context_proj(concatenated))

    def forward(
        self,
        draft_inputs_embeds: torch.Tensor,
        t: torch.Tensor,
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        """One velocity evaluation.

        Args:
            draft_inputs_embeds: [B, draft_len, hidden] — x_in-projected noisy
                states with the anchor-token embedding substituted at slot 0
                of each block (built by the training wrapper / sampler).
            t: [B, draft_len] — flow time per token (equal within a block).
            context_feature: [B, ctx_len, hidden]
            draft_position_ids / context_position_ids: as in DFlash
            block_mask: FlexAttention BlockMask

        Returns:
            velocity: [B, draft_len, target_hidden_size]
        """
        dtype = self.x_in.weight.dtype
        time_dim = getattr(self.config, "time_embed_dim", 256)
        bsz, draft_len = t.shape
        t_sin = timestep_embedding(t.reshape(-1), time_dim).to(dtype)
        time_cond = self.time_mlp(t_sin).view(bsz, draft_len, -1)

        h = draft_inputs_embeds.to(dtype) + self.t_in(t_sin).view(bsz, draft_len, -1)
        for layer in self.layers:
            h = layer(
                draft_hidden=h,
                time_cond=time_cond,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
            )
        return self.v_out(self.final_norm(h))
