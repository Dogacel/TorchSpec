"""Adjacent anchors must reach both DFlash training wrappers and respect supervision."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchspec.models.dflash import DFlashModel
from torchspec.models.dflash2 import DFlash2Model
from torchspec.training.dflash2_trainer import DFlash2Trainer
from torchspec.training.dflash_trainer import DFlashTrainer


@pytest.mark.parametrize("trainer_cls", [DFlashTrainer, DFlash2Trainer])
def test_training_wrapper_samples_adjacent_supervised_anchors(trainer_cls):
    trainer = SimpleNamespace(
        block_size=16,
        num_anchors=32,
        anchor_max_gap=0,
        non_overlapping_anchors=False,
        loss_objective="decay",
        dpace_alpha=0.5,
        loss_decay_gamma=7.0,
        ce_loss_alpha=1.0,
        l1_loss_alpha=0.0,
        selector_loss_alpha=1.0,
    )
    draft = nn.Linear(2, 2)
    draft.config = SimpleNamespace(num_hidden_layers=1)
    model = trainer_cls._build_training_wrapper(trainer, draft)
    mask = torch.zeros(2, 256)
    mask[0, 40:200] = 1
    mask[1, 80:90] = 1
    torch.manual_seed(42)
    anchors, keep = model._sample_anchor_positions(256, mask, torch.device("cpu"))
    assert anchors.shape == keep.shape == (2, 32)
    assert keep.sum(dim=1).tolist() == [32, 9]
    for row, (start, count) in enumerate([(40, 159), (80, 9)]):
        valid = anchors[row, keep[row]]
        assert valid.unique().numel() == valid.numel()
        assert mask[row, valid].all() and mask[row, valid + 1].all()
        # The randomly phased window may wrap from the last valid anchor to the first.
        ranks = valid - start
        assert torch.all((ranks[1:] - ranks[:-1]) % count == 1)


@pytest.mark.parametrize("model_cls", [DFlashModel, DFlash2Model])
def test_negative_anchor_gap_is_rejected(model_cls):
    with pytest.raises(ValueError, match="nonnegative"):
        model_cls(nn.Linear(2, 2), anchor_max_gap=-1)


@pytest.mark.parametrize("trainer_cls", [DFlashTrainer, DFlash2Trainer])
@pytest.mark.parametrize("length", [8, 40, 256, 12000])
def test_nonoverlapping_blocks_remain_disjoint(trainer_cls, length):
    trainer = SimpleNamespace(
        block_size=16,
        num_anchors=512,
        anchor_max_gap=None,
        non_overlapping_anchors=True,
        loss_objective="decay",
        dpace_alpha=0.5,
        loss_decay_gamma=7.0,
        ce_loss_alpha=1.0,
        l1_loss_alpha=0.0,
        selector_loss_alpha=1.0,
    )
    draft = nn.Linear(2, 2)
    draft.config = SimpleNamespace(num_hidden_layers=1)
    model = trainer_cls._build_training_wrapper(trainer, draft)
    mask = torch.zeros(3, length)
    mask[0, 4:] = 1
    mask[1, 4 : length // 2] = 1
    mask[1, length // 2 + 7 :] = 1
    for seed in range(10):
        torch.manual_seed(seed)
        anchors, keep = model._sample_anchor_positions(length, mask, torch.device("cpu"))
        assert anchors.shape == keep.shape == (3, 512)
        assert not keep[2].any()
        for row in (0, 1):
            valid = anchors[row, keep[row]].sort().values
            assert torch.all(valid[1:] - valid[:-1] >= 16)
            assert mask[row, valid].all() and mask[row, valid + 1].all()
            if length >= 40:
                assert valid.numel() > 0
                assert torch.all(valid + 16 <= length)
        if length == 12000:
            assert keep[0].sum() == 512
        if length == 256:
            assert keep[0].sum() < 20
