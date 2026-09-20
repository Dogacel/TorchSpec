"""Realized-token endpoint semantics, shared supervision, and backward compatibility."""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from tests.test_flowspec import _make_config, _target_inputs
from torchspec.models.draft.flowspec import FlowSpecDraftModel
from torchspec.models.flowspec import FlowSpecModel


def make_model(**kwargs):
    return FlowSpecModel(
        FlowSpecDraftModel(_make_config(self_conditioning=True)),
        block_size=3,
        num_anchors=2,
        anchor_max_gap=0,
        time_schedule="soft_target_argmax_uniform_blend",
        noise_type="standard_gaussian",
        use_diffusion_forcing_schedule=False,
        self_condition_probability=0.5,
        boundary_probability=1 / 32,
        use_flex_attention=False,
        **kwargs,
    )


def test_data_endpoint_controls_corruption_ce_and_hard_anchor():
    torch.manual_seed(42)
    model = make_model(endpoint_target="data_token")
    ids = torch.tensor([[7, 11, 13, 17, 19, 23, 29, 31]])
    inputs = dict(
        input_ids=ids,
        hidden_states_list=[torch.randn(1, 8, 32) for _ in range(2)],
        loss_mask=torch.ones(1, 8),
    )
    anchors = torch.tensor([[1, 4]])
    tau = torch.tensor([[0.0, 0.6]])
    clean_ids = torch.tensor([[[11, 13, 17], [19, 23, 29]]])
    clean = F.one_hot(clean_ids, 64).float()
    noise = torch.randn_like(clean)
    with torch.no_grad():
        model.draft_model.output_layer.linear.bias.copy_(torch.linspace(-1, 1, 64))
    with (
        mock.patch.object(
            model,
            "_sample_anchor_positions",
            return_value=(anchors, torch.ones_like(anchors, dtype=torch.bool)),
        ),
        mock.patch.object(model, "_sample_training_times", return_value=tau),
        mock.patch.object(model, "_sample_noise", return_value=noise),
        mock.patch.object(
            model,
            "_target_logits",
            side_effect=AssertionError("Must not use teacher probabilities"),
        ),
        mock.patch.object(
            model, "_corrupt_distribution", wraps=model._corrupt_distribution
        ) as corrupt,
        mock.patch.object(
            model, "_prediction_loss", wraps=model._prediction_loss
        ) as prediction_loss,
        mock.patch.object(model.draft_model, "forward", wraps=model.draft_model.forward) as forward,
    ):
        loss = model(**inputs)[0]
    torch.testing.assert_close(corrupt.call_args.args[0], clean, rtol=0, atol=0)
    torch.testing.assert_close(prediction_loss.call_args.args[1], clean, rtol=0, atol=0)
    t = model._time_coordinates(tau)[1][..., None]
    expected_state = (1 - t) * noise + t * clean
    for call in forward.call_args_list:
        torch.testing.assert_close(call.kwargs["flow_state"].view_as(clean), expected_state)
        torch.testing.assert_close(
            call.kwargs["self_condition_state"].view_as(clean)[:, :, 0], clean[:, :, 0]
        )
    bias = model.draft_model.output_layer.linear.bias
    ce = F.cross_entropy(bias.expand(4, -1), clean_ids[..., 1:].flatten(), reduction="none").view(
        1, 2, 2
    )
    weights = model.loss_decay_weights[1:]
    expected_loss = (ce * weights).sum() / (2 * weights.sum())
    torch.testing.assert_close(loss, expected_loss)
    expected_grad = torch.autograd.grad(expected_loss, bias)[0]
    loss.backward()
    torch.testing.assert_close(bias.grad, expected_grad)
    assert torch.isfinite(model.draft_model.output_layer.linear.weight.grad).all()


def test_data_endpoint_ignores_teacher_targets():
    model = make_model(endpoint_target="data_token")
    inputs = dict(
        input_ids=torch.randint(64, (1, 8)),
        hidden_states_list=[torch.randn(1, 8, 32) for _ in range(2)],
        loss_mask=torch.ones(1, 8),
    )
    teacher = _target_inputs(model, 1, 8)
    torch.manual_seed(10)
    expected = model(**inputs)
    torch.manual_seed(10)
    actual = model(**inputs, **teacher)
    for a, b in zip(actual[:-1], expected[:-1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_endpoint_default_preserves_teacher_requirement_and_invalid_values_fail():
    model = make_model()
    assert model.endpoint_target == "teacher_distribution"
    with pytest.raises(ValueError, match="target LM logits"):
        model(
            input_ids=torch.ones(1, 8, dtype=torch.long),
            hidden_states_list=[torch.randn(1, 8, 32) for _ in range(2)],
            loss_mask=torch.ones(1, 8),
        )
    with pytest.raises(ValueError, match="endpoint_target"):
        make_model(endpoint_target="unknown")


def test_token_endpoint_with_distribution_anchor_still_requires_teacher():
    model = make_model(endpoint_target="data_token", target_distribution_anchor=True)
    with pytest.raises(ValueError, match="target LM logits"):
        model(
            input_ids=torch.ones(1, 8, dtype=torch.long),
            hidden_states_list=[torch.randn(1, 8, 32) for _ in range(2)],
            loss_mask=torch.ones(1, 8),
        )
