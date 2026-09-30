"""A fresh Adam optimizer must restore lazy moments before its next update."""

from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp

from torchspec.training import checkpoint
from torchspec.training.checkpoint import OptimizerState
from torchspec.training.optimizer import BF16Optimizer


def test_bf16_optimizer_resume_preserves_moments_and_next_update(tmp_path, monkeypatch):
    monkeypatch.setattr("torchspec.training.optimizer.print_on_rank0", lambda *args: None)
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2).to(torch.bfloat16)
    opt = BF16Optimizer(model, lr=0.01, total_steps=10, warmup_ratio=0)
    inputs = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.bfloat16)

    def step(m, o):
        m(inputs).float().square().sum().backward()
        o.step()

    step(model, opt)
    step(model, opt)
    dcp.save({"optim_state": OptimizerState(model, opt)}, checkpoint_id=tmp_path)

    restored_model = torch.nn.Linear(3, 2).to(torch.bfloat16)
    restored_model.load_state_dict(model.state_dict())
    restored = BF16Optimizer(restored_model, lr=0.01, total_steps=10, warmup_ratio=0)
    assert not restored.optimizer.state
    dcp.load(
        {"optim_state": OptimizerState(restored_model, restored, tmp_path)},
        checkpoint_id=tmp_path,
    )
    restored.scheduler.load_state_dict(opt.scheduler.state_dict())
    assert len(restored.optimizer.state) == len(opt.optimizer.state)
    for before, after in zip(opt.fp32_params, restored.fp32_params, strict=True):
        assert torch.equal(before, after)
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                opt.optimizer.state[before][field], restored.optimizer.state[after][field]
            )
    step(model, opt)
    step(restored_model, restored)
    for before, after in zip(opt.fp32_params, restored.fp32_params, strict=True):
        assert torch.equal(before, after)
        assert restored.optimizer.state[after]["step"].item() == 3

    # Exercise the production save entry point after restoring and updating.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *args: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args: None)
    actor = SimpleNamespace(
        model=restored_model,
        optimizer=restored,
        lr_scheduler=restored.scheduler,
        global_step=3,
        args=SimpleNamespace(checkpoint_dir=str(tmp_path / "saved")),
    )
    checkpoint.save(actor, step=3)
    saved = tmp_path / "saved" / "iter_0000004" / "optimizer"
    meta = dcp.FileSystemReader(saved).read_metadata()
    steps = {
        k: torch.empty(v.size, dtype=v.properties.dtype)
        for k, v in meta.state_dict_metadata.items()
        if k.endswith(".step")
    }
    dcp.load(steps, checkpoint_id=saved)
    assert all(value.item() == 3 for value in steps.values())
