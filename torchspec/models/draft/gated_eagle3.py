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

"""Depth-routed MLP variant of the Llama EAGLE-3 draft model."""

import torch
import torch.nn as nn
from transformers import AutoConfig, LlamaConfig

from torchspec.models.draft.llama3_eagle import (
    LlamaDecoderLayer,
    LlamaForCausalLMEagle3,
    LlamaMLP,
)


class LlamaGatedConfig(LlamaConfig):
    """Llama-compatible config for the depth-routed EAGLE-3 architecture."""

    model_type = "llama_gated"

    def __init__(
        self,
        fc_norm: bool = True,
        post_norm: bool | None = None,
        norm_output: bool | None = None,
        **kwargs,
    ):
        if post_norm is None and norm_output is None:
            post_norm = True
            norm_output = True
        elif post_norm is None:
            post_norm = bool(norm_output)
        elif norm_output is None:
            norm_output = bool(post_norm)
        elif bool(post_norm) != bool(norm_output):
            raise ValueError(
                "post_norm and norm_output must agree when both are set "
                f"(got post_norm={post_norm!r}, norm_output={norm_output!r})"
            )

        super().__init__(**kwargs)
        self.fc_norm = bool(fc_norm)
        self.post_norm = bool(post_norm)
        self.norm_output = bool(norm_output)


# AutoConfig can resolve saved ``model_type: llama_gated`` configs after
# TorchSpec is imported, without relying on the ``architectures`` field.
AutoConfig.register(LlamaGatedConfig.model_type, LlamaGatedConfig, exist_ok=True)


class DepthGatedMLP(nn.Module):
    """Depth-selected specialist plus a token-gated shared MLP.

    This follows the Qwen3.5 MoE shared-expert convention:
    a token-dependent scalar sigmoid gate weights the shared MLP,
    while rollout depth is used as top-k gate for the other MLP.
    """

    def __init__(self, config):
        super().__init__()
        self.shared_mlp = LlamaMLP(config)
        self.depth0_mlp = LlamaMLP(config)
        self.depthn_mlp = LlamaMLP(config)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor, depth: int = 0) -> torch.Tensor:
        shared = self.shared_mlp(x)
        if depth == 0:
            specialist = self.depth0_mlp(x)
        else:
            specialist = self.depthn_mlp(x)

        shared_gate = torch.sigmoid(self.shared_expert_gate(x).float()).to(dtype=x.dtype)
        return specialist + shared_gate * shared


class DepthGatedLlamaDecoderLayer(LlamaDecoderLayer):
    """Llama EAGLE-3 layer whose MLP specializes by rollout depth."""

    mlp_class = DepthGatedMLP


class LlamaForCausalLMEagle3Gated(LlamaForCausalLMEagle3):
    """Llama EAGLE-3 with shared, depth-0, and depth-N gated MLPs.

    ``fc_norm`` and post-norm rollout state (the existing ``norm_output``
    behavior) default to enabled. Either default can still be explicitly
    disabled in the draft config.
    """

    config_class = LlamaGatedConfig
    decoder_layer_class = DepthGatedLlamaDecoderLayer
    all_tied_weights_keys = {}

    def __init__(self, config, attention_backend: str = "sdpa") -> None:
        super().__init__(config, attention_backend=attention_backend)
