import json
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from tools.convert_to_hf import _save_with_vocab_pruning


def test_pruned_head_matches_serving_token_map(tmp_path):
    head = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    embeddings = head + 100
    model = SimpleNamespace(
        state_dict=lambda: {
            "lm_head.weight": head,
            "embed_tokens.weight": embeddings,
        }
    )
    target_ids = torch.tensor([2, 5, 9])
    offsets = target_ids - torch.arange(3)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[target_ids] = True
    _save_with_vocab_pruning(
        model,
        str(tmp_path),
        {"model_type": "llama", "vocab_size": 10},
        10,
        3,
        offsets,
        mask,
    )
    saved = load_file(tmp_path / "model.safetensors")
    # Decode the stored map exactly as the serving loader does.
    serving_ids = saved["d2t"] + torch.arange(3)
    assert torch.equal(serving_ids, target_ids)
    assert torch.equal(saved["lm_head.weight"], head[serving_ids])
    assert torch.equal(saved["embed_tokens.weight"], embeddings)
    assert torch.equal(saved["t2d"], mask)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["draft_vocab_size"] == 3
    assert config["vocab_size"] == 10
