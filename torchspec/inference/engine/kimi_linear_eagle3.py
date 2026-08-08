"""vLLM Kimi-Linear adapter that exposes EAGLE3 auxiliary hidden states.

vLLM 0.26 has a native Kimi-Linear serving model, but that model does not yet
implement ``SupportsEagle3``. TorchSpec's hidden-state connector relies on
that public interface. This adapter keeps vLLM's kernels, cache management,
and weight loader unchanged and only adds residual-stream capture.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.models.kimi_linear import (
    KimiLinearForCausalLM,
    KimiLinearModel,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class KimiLinearModelEagle3(KimiLinearModel, EagleModelMixin):
    """Kimi-Linear target model with post-layer residual-stream capture."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if intermediate_tensors is not None:
            raise NotImplementedError("KimiLinearForCausalLMEagle3 requires pp_size=1")

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class KimiLinearForCausalLMEagle3(KimiLinearForCausalLM, SupportsEagle3):
    """Native vLLM Kimi-Linear model with the EAGLE3 interface enabled."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model = KimiLinearModelEagle3(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
