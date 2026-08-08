import torch
from vllm.model_executor.models.interfaces import EagleModelMixin, supports_eagle3

from torchspec.data.template import TEMPLATE_REGISTRY
from torchspec.inference.engine.kimi_linear_eagle3 import (
    KimiLinearForCausalLMEagle3,
    KimiLinearModelEagle3,
)
from torchspec.models.target.target_utils import _GenericRMSNorm


def test_kimi_linear_vllm_adapter_exposes_eagle3_contract():
    assert supports_eagle3(KimiLinearForCausalLMEagle3)
    assert issubclass(KimiLinearModelEagle3, EagleModelMixin)


def test_kimi_linear_template_matches_official_headers():
    template = TEMPLATE_REGISTRY.get("kimi-linear")
    assert template.user_header == "<|im_user|>user<|im_middle|>"
    assert template.assistant_header == "<|im_assistant|>assistant<|im_middle|>"
    assert template.end_of_turn_token == "<|im_end|>"


def test_generic_rms_norm_matches_definition():
    norm = _GenericRMSNorm(3, 1e-5)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    expected = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(norm(values), expected)
