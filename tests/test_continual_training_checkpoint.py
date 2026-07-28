from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from torchspec.training import checkpoint


class DummyBF16Optimizer:
    def __init__(self, *, lr: float, betas: tuple[float, float], weight_decay: float) -> None:
        self.fp32_params = [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))]
        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def sync_fp32_params_from_model(self):
        raise AssertionError("fallback path should not run in this test")


def test_restore_fp32_master_params_keeps_fresh_optimizer_hparams():
    optimizer = DummyBF16Optimizer(lr=0.123, betas=(0.7, 0.8), weight_decay=0.456)
    actor = SimpleNamespace(model=mock.MagicMock(), optimizer=optimizer)

    checkpoint_optimizer = DummyBF16Optimizer(lr=0.999, betas=(0.1, 0.2), weight_decay=0.0)
    with torch.no_grad():
        checkpoint_optimizer.fp32_params[0].copy_(torch.tensor([42.0], dtype=torch.float32))
    checkpoint_optimizer.optimizer.state[checkpoint_optimizer.fp32_params[0]]["step"] = (
        torch.tensor(7.0)
    )

    checkpoint_state = {
        "optim": checkpoint_optimizer.optimizer.state_dict(),
        "fp32_params": {"0": checkpoint_optimizer.fp32_params[0].detach().clone()},
    }

    def fake_dcp_load(*, state_dict, checkpoint_id):
        assert checkpoint_id == "/tmp/fake_optim"
        state_dict["optim_state"].load_state_dict(checkpoint_state)

    def fake_exists(self):
        return str(self) in {"/tmp/fake_optim", "/tmp/fake_optim/.metadata"}

    with (
        mock.patch("torchspec.training.checkpoint.dcp.load", side_effect=fake_dcp_load),
        mock.patch("pathlib.Path.exists", new=fake_exists),
    ):
        checkpoint._restore_fp32_master_params(actor, Path("/tmp/fake_optim"))

    group = optimizer.optimizer.param_groups[0]
    assert group["lr"] == 0.123
    assert group["betas"] == (0.7, 0.8)
    assert group["weight_decay"] == 0.456
    assert optimizer.optimizer.state == {}
    assert torch.equal(optimizer.fp32_params[0].detach(), torch.tensor([42.0], dtype=torch.float32))


def test_restore_fp32_master_params_can_preserve_adam_state():
    optimizer = DummyBF16Optimizer(lr=0.123, betas=(0.7, 0.8), weight_decay=0.456)
    actor = SimpleNamespace(model=mock.MagicMock(), optimizer=optimizer)

    checkpoint_optimizer = DummyBF16Optimizer(lr=0.999, betas=(0.1, 0.2), weight_decay=0.0)
    parameter = checkpoint_optimizer.fp32_params[0]
    checkpoint_optimizer.optimizer.state[parameter].update(
        step=torch.tensor(7.0),
        exp_avg=torch.tensor([2.0]),
        exp_avg_sq=torch.tensor([3.0]),
    )
    checkpoint_state = {
        "optim": checkpoint_optimizer.optimizer.state_dict(),
        "fp32_params": {"0": parameter.detach().clone()},
    }

    def fake_dcp_load(*, state_dict, checkpoint_id):
        assert checkpoint_id == "/tmp/fake_optim"
        state_dict["optim_state"].load_state_dict(checkpoint_state)

    def fake_exists(self):
        return str(self) in {"/tmp/fake_optim", "/tmp/fake_optim/.metadata"}

    with (
        mock.patch("torchspec.training.checkpoint.dcp.load", side_effect=fake_dcp_load),
        mock.patch("pathlib.Path.exists", new=fake_exists),
    ):
        checkpoint._restore_fp32_master_params(
            actor,
            Path("/tmp/fake_optim"),
            reset_optimizer=False,
        )

    group = optimizer.optimizer.param_groups[0]
    assert group["lr"] == 0.123
    assert group["betas"] == (0.7, 0.8)
    assert group["weight_decay"] == 0.456
    state = optimizer.optimizer.state[optimizer.fp32_params[0]]
    assert state["step"].item() == 7.0
    assert torch.equal(state["exp_avg"], torch.tensor([2.0]))
    assert torch.equal(state["exp_avg_sq"], torch.tensor([3.0]))


def test_resume_restores_global_step():
    actor = SimpleNamespace(
        args=SimpleNamespace(continual_training=False, start_step=None),
        global_step=0,
    )
    payload = {
        "metadata": {"global_step": 2000, "next_step": 2001},
        "iteration": 2001,
    }

    with (
        mock.patch("torchspec.training.checkpoint.torch.cuda.synchronize"),
        mock.patch("torchspec.training.checkpoint.dist.barrier"),
    ):
        checkpoint.finalize_load(actor, payload)

    assert actor.global_step == 2000
    assert actor.args.start_step == 2001


def test_optimizer_continuation_preserves_global_step_with_fresh_scheduler():
    actor = SimpleNamespace(
        args=SimpleNamespace(
            continual_training=True,
            continual_training_reset_optimizer=False,
            start_step=None,
        ),
        global_step=0,
        optimizer=mock.MagicMock(),
    )
    payload = {
        "metadata": {"global_step": 7228, "next_step": 7229},
        "iteration": 7229,
        "optimizer_dir": Path("/tmp/fake_optim"),
    }

    with (
        mock.patch("torchspec.training.checkpoint._restore_fp32_master_params") as restore,
        mock.patch("torchspec.training.checkpoint.torch.cuda.synchronize"),
        mock.patch("torchspec.training.checkpoint.dist.barrier"),
    ):
        checkpoint.finalize_load(actor, payload)

    assert actor.global_step == 7228
    assert actor.args.start_step == 7229
    restore.assert_called_once_with(
        actor,
        Path("/tmp/fake_optim"),
        reset_optimizer=False,
    )
