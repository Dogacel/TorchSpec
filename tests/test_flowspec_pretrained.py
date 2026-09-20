"""Pretrained FlowSpec vocabulary weights and frozen-gradient behavior."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.test_flowspec import _make_config
from torchspec.models.draft.flowspec import FlowSpecDraftModel


def checkpoint(path, embedding, head, sharded):
    if sharded:
        save_file({"model.embed_tokens.weight": embedding}, path / "embed.safetensors")
        save_file({"lm_head.weight": head}, path / "head.safetensors")
        (path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "embed.safetensors",
                        "lm_head.weight": "head.safetensors",
                    }
                }
            )
        )
    else:
        save_file(
            {"model.embed_tokens.weight": embedding, "lm_head.weight": head},
            path / "model.safetensors",
        )


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_pretrained_exact_weights_and_gradient_flow(tmp_path, sharded, frozen):
    config = _make_config()
    config.pretrained_embeddings_and_head = True
    config.freeze_embeddings_and_head = frozen
    model = FlowSpecDraftModel(config)
    embedding = torch.randn(config.vocab_size, config.hidden_size)
    head = torch.randn_like(embedding)
    checkpoint(tmp_path, embedding, head, sharded)
    model.load_pretrained_embeddings_and_head(str(tmp_path))
    torch.testing.assert_close(model.vocab_embed.weight, embedding, rtol=0, atol=0)
    torch.testing.assert_close(model.output_layer.linear.weight, head, rtol=0, atol=0)
    assert model.output_layer.linear.bias is None
    assert model.vocab_embed.weight.requires_grad == (not frozen)
    assert model.output_layer.linear.weight.requires_grad == (not frozen)
    ids = torch.tensor([[2, 5]])
    state = torch.nn.functional.one_hot(ids, config.vocab_size).float().requires_grad_()
    embedded = model.vocab_embed(state)
    torch.testing.assert_close(embedded, embedding[ids], rtol=0, atol=0)
    logits = model.output_layer(embedded, torch.randn(1, 2, config.time_condition_size))
    logits.square().mean().backward()
    assert state.grad.abs().sum() > 0
    assert model.output_layer.ada_ln_modulation.weight.grad.abs().sum() > 0
    assert (model.vocab_embed.weight.grad is None) == frozen
    assert (model.output_layer.linear.weight.grad is None) == frozen


def test_pretrained_rejects_shape_mismatch(tmp_path):
    config = _make_config()
    config.pretrained_embeddings_and_head = True
    model = FlowSpecDraftModel(config)
    checkpoint(
        tmp_path,
        torch.randn(1, config.hidden_size),
        torch.randn(config.vocab_size, config.hidden_size),
        False,
    )
    with pytest.raises(ValueError, match="does not match"):
        model.load_pretrained_embeddings_and_head(str(tmp_path))


def test_pretrained_rejects_missing_head(tmp_path):
    config = _make_config()
    config.pretrained_embeddings_and_head = True
    model = FlowSpecDraftModel(config)
    save_file(
        {"model.embed_tokens.weight": torch.randn(config.vocab_size, config.hidden_size)},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(KeyError, match="lm_head.weight"):
        model.load_pretrained_embeddings_and_head(str(tmp_path))
