import json
from pathlib import Path

import torch
from vllm.model_executor.models.interfaces import EagleModelMixin, supports_eagle3

from torchspec.data.parse import GeneralParser
from torchspec.data.template import TEMPLATE_REGISTRY
from torchspec.inference.engine.kimi_linear_eagle3 import (
    KimiLinearForCausalLMEagle3,
    KimiLinearModelEagle3,
)
from torchspec.models.target.target_utils import _LoadedRMSNorm


def test_kimi_linear_vllm_adapter_exposes_eagle3_contract():
    assert supports_eagle3(KimiLinearForCausalLMEagle3)
    assert issubclass(KimiLinearModelEagle3, EagleModelMixin)


def test_kimi_linear_template_matches_official_headers():
    template = TEMPLATE_REGISTRY.get("kimi-linear")
    assert template.user_header == "<|im_user|>user<|im_middle|>"
    assert template.assistant_header == "<|im_assistant|>assistant<|im_middle|>"
    assert template.end_of_turn_token == "<|im_end|>"


def test_kimi_linear_template_preserves_explicit_system_message():
    class RecordingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "rendered"

    tokenizer = RecordingTokenizer()
    parser = GeneralParser(tokenizer, TEMPLATE_REGISTRY.get("kimi-linear"))
    conversation = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
    ]

    assert parser.format(conversation) == "rendered"
    assert tokenizer.messages == conversation


def test_loaded_rms_norm_matches_definition():
    norm = _LoadedRMSNorm(3, 1e-5)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    expected = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(norm(values), expected)


def test_kimi_linear_eagle3_uses_full_attention_aux_layers():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/draft_models/kimi_linear_48b_eagle3.json"
    config = json.loads(config_path.read_text())
    aux_layers = config["eagle_config"]["eagle_aux_hidden_state_layer_ids"]
    full_attention_layers = {4, 8, 12, 16, 20, 24, 27}

    assert aux_layers == [4, 16, 24]
    assert set(aux_layers).issubset(full_attention_layers)
