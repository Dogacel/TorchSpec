from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.eagle3 import Eagle3Model, PrecomputedTarget
from torchspec.training.eagle3_trainer import Eagle3Trainer


class _FakeDraftModel(nn.Module):
    def __init__(self, hidden_size: int = 8, vocab_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.target_vocab_size = vocab_size
        self.norm_output = False
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.lm_head_weight = nn.Parameter(torch.randn(vocab_size, hidden_size))

    def get_lm_head_params(self):
        return self.norm_weight, self.lm_head_weight, 1e-6

    def project_hidden_states(self, hidden_states):
        return hidden_states[..., : self.hidden_size]

    def compute_logits(self, hidden_states):
        return F.linear(hidden_states, self.lm_head_weight)

    def embed_input_ids(self, input_ids):
        return torch.zeros(
            *input_ids.shape,
            self.hidden_size,
            dtype=self.norm_weight.dtype,
            device=input_ids.device,
        )

    def backbone(
        self,
        input_embeds,
        hidden_states,
        attention_mask,
        position_ids,
        cache_keys,
        cache_values,
        use_cache,
        depth=None,
    ):
        return hidden_states + input_embeds, cache_keys, cache_values


def _make_fake_model(*, length=3, loss_mask_mode="assistant", hidden_state_dropout=0.0):
    return Eagle3Model(
        _FakeDraftModel(),
        length=length,
        attention_backend="flex_attention",
        loss_mask_mode=loss_mask_mode,
        hidden_state_dropout=hidden_state_dropout,
    )


def _run_and_capture_masks(
    model,
    attention_mask,
    loss_mask,
    matches_by_depth=None,
    position_mask=None,
):
    captured = []
    match_calls = []

    def fake_calculate_loss(
        self,
        hidden_states,
        target,
        mask,
        idx,
        seq_length,
        norm_weight,
        lm_head_weight,
        norm_eps,
    ):
        captured.append(mask.detach().clone())
        zero = hidden_states.sum() * 0.0
        count = mask.sum().float()
        return zero, count.detach(), count.detach()

    def fake_top1_match_mask(self, hidden_states, target, mask, idx, seq_length):
        match_calls.append(idx)
        matches = torch.ones_like(mask, dtype=torch.bool)
        if matches_by_depth is not None and idx in matches_by_depth:
            active = mask.bool()
            matches[active] = matches_by_depth[idx].to(mask.device)[active]
        return matches

    model._calculate_loss = MethodType(fake_calculate_loss, model)
    model._top1_match_mask = MethodType(fake_top1_match_mask, model)
    batch_size, seq_length = loss_mask.shape
    hidden_size = model.draft_model.hidden_size
    vocab_size = model.draft_model.vocab_size
    target_p = F.softmax(
        torch.randn(batch_size, seq_length + model.length, vocab_size),
        dim=-1,
    )
    model(
        input_ids=torch.ones(batch_size, seq_length, dtype=torch.long),
        attention_mask=attention_mask,
        target=PrecomputedTarget(target_p, position_mask),
        loss_mask=loss_mask,
        hidden_states=torch.randn(batch_size, seq_length, hidden_size * 3),
    )
    return captured, match_calls


def test_top1_path_trains_all_valid_continuations_and_filters_future_steps():
    model = _make_fake_model(loss_mask_mode="top1_path")
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32)
    matches_by_depth = {
        0: torch.tensor([[1, 0, 1, 0, 1, 1]], dtype=torch.bool),
        1: torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.bool),
    }

    masks, match_calls = _run_and_capture_masks(
        model,
        attention_mask,
        loss_mask,
        matches_by_depth=matches_by_depth,
    )

    # Every depth includes all remaining valid continuations, independent of
    # role, and only retains paths that have matched at every previous depth.
    torch.testing.assert_close(masks[0], torch.tensor([[1, 1, 1, 1, 0, 0.0]]))
    torch.testing.assert_close(masks[1], torch.tensor([[1, 0, 1, 0, 0, 0.0]]))
    torch.testing.assert_close(masks[2], torch.tensor([[1, 0, 0, 0, 0, 0.0]]))
    # The last depth has no future loss to gate.
    assert match_calls == [0, 1]


def test_assistant_mode_preserves_original_shifted_masks():
    model = _make_fake_model(length=3)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32)

    masks, match_calls = _run_and_capture_masks(model, attention_mask, loss_mask)

    torch.testing.assert_close(masks[0], loss_mask)
    torch.testing.assert_close(masks[1], torch.tensor([[0, 1, 1, 1, 0, 0.0]]))
    torch.testing.assert_close(masks[2], torch.tensor([[1, 1, 1, 0, 0, 0.0]]))
    assert match_calls == []


def test_top1_path_invalid_vocab_position_cannot_reenter():
    model = _make_fake_model(length=2, loss_mask_mode="top1_path")
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    loss_mask = torch.ones_like(attention_mask, dtype=torch.float32)
    position_mask = torch.tensor([[1, 0, 1, 1, 0, 0]], dtype=torch.float32)

    masks, _ = _run_and_capture_masks(
        model,
        attention_mask,
        loss_mask,
        position_mask=position_mask,
    )

    torch.testing.assert_close(masks[0], position_mask)
    # Position 1 becomes vocabulary-valid after shifting, but its path was
    # terminated when the preceding target was absent from the draft vocab.
    torch.testing.assert_close(masks[1], torch.tensor([[0, 0, 1, 0, 0, 0.0]]))


def test_eval_top1_path_uses_assistant_mask_without_pruning():
    model = _make_fake_model(length=3, loss_mask_mode="top1_path")
    model.eval()
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32)
    matches_by_depth = {
        0: torch.zeros_like(attention_mask, dtype=torch.bool),
        1: torch.zeros_like(attention_mask, dtype=torch.bool),
    }

    masks, match_calls = _run_and_capture_masks(
        model,
        attention_mask,
        loss_mask,
        matches_by_depth=matches_by_depth,
    )

    torch.testing.assert_close(masks[0], loss_mask)
    torch.testing.assert_close(masks[1], torch.tensor([[0, 1, 1, 1, 0, 0.0]]))
    torch.testing.assert_close(masks[2], torch.tensor([[1, 1, 1, 0, 0, 0.0]]))
    assert match_calls == []


def test_eval_assistant_mask_excludes_unavailable_pruned_vocab_targets():
    model = _make_fake_model(length=2, loss_mask_mode="top1_path")
    model.eval()
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32)
    position_mask = torch.tensor([[1, 1, 1, 0, 1, 0]], dtype=torch.float32)

    masks, match_calls = _run_and_capture_masks(
        model,
        attention_mask,
        loss_mask,
        position_mask=position_mask,
    )

    torch.testing.assert_close(masks[0], torch.tensor([[0, 0, 1, 0, 1, 0.0]]))
    torch.testing.assert_close(masks[1], torch.tensor([[0, 1, 0, 1, 0, 0.0]]))
    assert match_calls == []


def test_hidden_state_dropout_keeps_depth0_and_zeros_later_complete_token_vectors():
    model = _make_fake_model(hidden_state_dropout=1.0)
    hidden_states = torch.randn(2, 4, 24)

    model.train()
    retained_depth0 = model._apply_hidden_state_dropout(hidden_states, depth=0)
    dropped = model._apply_hidden_state_dropout(hidden_states, depth=1)
    assert retained_depth0 is hidden_states
    torch.testing.assert_close(dropped, torch.zeros_like(hidden_states))

    model.eval()
    retained = model._apply_hidden_state_dropout(hidden_states, depth=2)
    assert retained is hidden_states


@pytest.mark.parametrize("dropout", [-0.1, 1.1])
def test_hidden_state_dropout_rejects_invalid_probability(dropout):
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        _make_fake_model(hidden_state_dropout=dropout)


def test_top1_match_mask_is_neutral_outside_active_positions():
    model = _make_fake_model(length=2, loss_mask_mode="top1_path")
    model.draft_model.lm_head_weight.data.copy_(torch.eye(16, 8))

    hidden_states = torch.zeros(1, 3, 8)
    hidden_states[0, 0, 0] = 1
    hidden_states[0, 1, 1] = 1
    hidden_states[0, 2, 2] = 1
    target_p = torch.zeros(1, 5, 16)
    target_p[0, 0, 0] = 1
    target_p[0, 1, 3] = 1
    target_p[0, 2, 4] = 1
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    matches = model._top1_match_mask(
        hidden_states,
        PrecomputedTarget(target_p),
        mask,
        idx=0,
        seq_length=3,
    )

    torch.testing.assert_close(matches, torch.tensor([[True, False, True]]))


def test_eval_metrics_weight_accuracy_by_valid_token_count(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    trainer = Eagle3Trainer.__new__(Eagle3Trainer)
    trainer.args = SimpleNamespace(ploss_weights=None)

    metrics = trainer._aggregate_eval_metrics(
        [
            {
                "vlosses": torch.tensor([1.0, 1.0]),
                "acces": torch.tensor([1.0, 0.0]),
                "acc_counts": torch.tensor([1.0, 9.0]),
            },
            {
                "vlosses": torch.tensor([1.0, 1.0]),
                "acces": torch.tensor([0.0, 1.0]),
                "acc_counts": torch.tensor([9.0, 1.0]),
            },
        ]
    )

    assert metrics["eval/acc_0"] == pytest.approx(0.1)
    assert metrics["eval/acc_1"] == pytest.approx(0.1)
    assert metrics["eval/avg_acc"] == pytest.approx(0.1)
    assert metrics["eval/simulated_acc_len"] == pytest.approx(0.11)
