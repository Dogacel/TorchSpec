"""Tests for the depth-gated Llama EAGLE-3 draft architecture."""

import torch
import torch.nn as nn
from transformers import AutoConfig

from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from torchspec.models.draft.gated_eagle3 import (
    DepthGatedMLP,
    LlamaForCausalLMEagle3Gated,
    LlamaGatedConfig,
)
from torchspec.models.eagle3 import Eagle3Model, PrecomputedTarget


def _make_config(**overrides):
    values = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
        "vocab_size": 64,
        "pad_token_id": 0,
        "pretraining_tp": 1,
    }
    values.update(overrides)
    config = LlamaGatedConfig(**values)
    config.architectures = ["LlamaForCausalLMEagle3Gated"]
    return config


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class _FirstFeatureLogit(nn.Module):
    def forward(self, x):
        return x[..., :1]


def test_depth_gated_mlp_selects_specialist_and_gates_shared_per_token():
    mlp = DepthGatedMLP(_make_config(hidden_size=4, intermediate_size=8))
    mlp.shared_mlp = _Scale(1.0)
    mlp.depth0_mlp = _Scale(2.0)
    mlp.depthn_mlp = _Scale(4.0)
    mlp.shared_expert_gate = _FirstFeatureLogit()

    x = torch.tensor(
        [
            [[-2.0, 1.0, 2.0, 3.0], [2.0, 4.0, 5.0, 6.0]],
        ]
    )
    shared_gate = torch.sigmoid(x[..., :1])

    torch.testing.assert_close(mlp(x, depth=0), 2.0 * x + shared_gate * x)
    torch.testing.assert_close(mlp(x, depth=1), 4.0 * x + shared_gate * x)
    torch.testing.assert_close(mlp(x, depth=7), 4.0 * x + shared_gate * x)


def test_depth_gated_mlp_only_trains_selected_specialist():
    torch.manual_seed(0)
    mlp = DepthGatedMLP(_make_config(hidden_size=8, intermediate_size=16))
    x = torch.randn(2, 3, 8)

    mlp(x, depth=0).sum().backward()
    assert mlp.shared_expert_gate.weight.grad is not None
    assert all(param.grad is not None for param in mlp.depth0_mlp.parameters())
    assert all(param.grad is None for param in mlp.depthn_mlp.parameters())

    mlp.zero_grad(set_to_none=True)
    mlp(x, depth=2).sum().backward()
    assert mlp.shared_expert_gate.weight.grad is not None
    assert all(param.grad is None for param in mlp.depth0_mlp.parameters())
    assert all(param.grad is not None for param in mlp.depthn_mlp.parameters())


def test_gated_architecture_defaults_to_fc_and_post_norm():
    config = _make_config()
    model = LlamaForCausalLMEagle3Gated(config)

    assert model.fc_norm is not None
    assert len(model.fc_norm) == 3
    assert model.norm_output is True
    assert model.config.fc_norm is True
    assert model.config.post_norm is True
    assert model.config.norm_output is True


def test_gated_architecture_respects_explicit_norm_overrides():
    config = _make_config()
    config.fc_norm = False
    config.post_norm = False
    config.norm_output = False
    model = LlamaForCausalLMEagle3Gated(config)

    assert model.fc_norm is None
    assert model.norm_output is False
    assert model.config.norm_output is False


def test_auto_config_and_model_dispatch_to_gated_architecture():
    config_dict = _make_config().to_dict()
    config = AutoDraftModelConfig.from_dict(config_dict)
    model = AutoEagle3DraftModel.from_config(config)

    assert isinstance(config, LlamaGatedConfig)
    assert isinstance(model, LlamaForCausalLMEagle3Gated)


def test_hugging_face_auto_config_resolves_saved_gated_config(tmp_path):
    config = _make_config()
    config.save_pretrained(tmp_path)

    loaded = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, LlamaGatedConfig)
    assert loaded.model_type == "llama_gated"


def test_gated_model_hugging_face_round_trip(tmp_path):
    model = LlamaForCausalLMEagle3Gated(_make_config())
    model.save_pretrained(tmp_path)

    loaded, loading_info = AutoEagle3DraftModel.from_pretrained(
        tmp_path,
        output_loading_info=True,
    )

    assert isinstance(loaded, LlamaForCausalLMEagle3Gated)
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]


def test_gradient_checkpointed_eagle3_rollout_trains_all_mlp_paths():
    torch.manual_seed(0)
    config = _make_config()
    draft = LlamaForCausalLMEagle3Gated(config)
    model = Eagle3Model(
        draft,
        length=2,
        attention_backend="sdpa",
        gradient_checkpointing=True,
    )
    model.train()

    batch_size, seq_length = 1, 8
    target_p = torch.softmax(
        torch.randn(batch_size, seq_length + model.length, draft.vocab_size),
        dim=-1,
    )
    losses, _, _, _ = model(
        input_ids=torch.randint(0, draft.target_vocab_size, (batch_size, seq_length)),
        attention_mask=torch.ones(batch_size, seq_length, dtype=torch.long),
        target=PrecomputedTarget(target_p),
        loss_mask=torch.ones(batch_size, seq_length),
        hidden_states=torch.randn(
            batch_size,
            seq_length,
            config.hidden_size * draft.num_aux_hidden_states,
        ),
    )
    sum(losses).backward()

    mlp = draft.midlayer.mlp
    assert mlp.shared_expert_gate.weight.grad is not None
    assert all(param.grad is not None for param in mlp.shared_mlp.parameters())
    assert all(param.grad is not None for param in mlp.depth0_mlp.parameters())
    assert all(param.grad is not None for param in mlp.depthn_mlp.parameters())


class _RecordingDraftModel(nn.Module):
    """Minimal draft model used to verify rollout-depth plumbing."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 8
        self.target_vocab_size = 8
        self.norm_output = False
        self.anchor = nn.Parameter(torch.ones(()))
        self.norm = nn.LayerNorm(4)
        self.norm.variance_epsilon = self.norm.eps
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.depths = []

    def get_lm_head_params(self):
        return self.norm.weight, self.lm_head.weight, self.norm.eps

    def project_hidden_states(self, hidden_states):
        return hidden_states

    def prepare_decoder_attention_mask(self, **kwargs):
        return None

    def embed_input_ids(self, input_ids):
        return torch.zeros(*input_ids.shape, 4)

    def backbone(
        self,
        input_embeds,
        hidden_states,
        attention_mask,
        position_ids,
        cache_keys=None,
        cache_values=None,
        use_cache=True,
        depth=0,
    ):
        self.depths.append(depth)
        return hidden_states + self.anchor * 0.0, cache_keys, cache_values


def test_eagle3_model_passes_rollout_depth_to_draft_backbone():
    draft = _RecordingDraftModel()
    model = Eagle3Model(draft, length=3, attention_backend="sdpa")
    batch_size, seq_length = 1, 4
    target = PrecomputedTarget(
        target_p_padded=torch.zeros(batch_size, seq_length + 3, draft.vocab_size),
        position_mask=torch.zeros(batch_size, seq_length),
    )

    model(
        input_ids=torch.zeros(batch_size, seq_length, dtype=torch.long),
        attention_mask=torch.ones(batch_size, seq_length, dtype=torch.long),
        target=target,
        loss_mask=torch.zeros(batch_size, seq_length),
        hidden_states=torch.zeros(batch_size, seq_length, 4),
    )

    assert draft.depths == [0, 1, 2]
