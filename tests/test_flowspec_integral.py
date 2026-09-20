"""Objective identities, sampling correction, and distribution-anchor semantics."""
from unittest import mock

import pytest
import torch

from tests.test_flowspec import _make_config, _target_inputs
from torchspec.models.draft.flowspec import FlowSpecDraftModel
from torchspec.models.flowspec import FlowSpecModel


def make_model(**kwargs):
    return FlowSpecModel(
        FlowSpecDraftModel(_make_config(self_conditioning=True)),
        block_size=3, num_anchors=2, anchor_max_gap=0,
        time_schedule='soft_target_argmax_uniform_blend',
        noise_type='standard_gaussian', use_diffusion_forcing_schedule=False,
        self_condition_probability=0.5, boundary_probability=1 / 32,
        loss_type='velocity_mse', integral_time_weighting=True,
        target_distribution_anchor=True, use_flex_attention=False, **kwargs,
    )


def test_velocity_loss_value_and_gradient_match_direct_regression():
    torch.manual_seed(3)
    model = make_model()
    q = torch.randn(1, 2, 3, 64).softmax(-1)
    t = torch.tensor([[[1., 0., 0.], [1., .83, .83]]])
    noise = torch.randn_like(q)
    z = (1 - t[..., None]) * noise + t[..., None] * q
    logits = torch.randn(1, 6, 64, requires_grad=True)
    loss = model._prediction_loss(logits, q, t)[..., 1:].sum()
    pred = logits.softmax(-1).view_as(q)[..., 1:, :]
    direct = ((pred - z[..., 1:, :]) / (1-t[..., 1:, None])
              - (q - noise)[..., 1:, :]).square().sum()
    torch.testing.assert_close(loss, direct)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0],
                               torch.autograd.grad(direct, logits)[0])


def test_jacobian_matches_finite_difference_and_change_of_variables():
    model = make_model()
    schedule = model.time_from_tau
    tau = torch.linspace(.01, .99, 100)
    finite_difference = (schedule(tau + 1e-4) - schedule(tau - 1e-4)) / 2e-4
    torch.testing.assert_close(schedule.derivative(tau), finite_difference, rtol=.01, atol=.002)
    # Integral of 1, t, t^2 on a bounded interval, evaluated in schedule coordinates.
    tau = torch.linspace(.001, .999, 100001)
    t = schedule(tau)
    jac = schedule.derivative(tau)
    for power in (0, 1, 2):
        integral = torch.trapezoid(jac * t.pow(power), tau)
        expected = (t[-1]**(power+1) - t[0]**(power+1)) / (power+1)
        torch.testing.assert_close(integral, expected, rtol=2e-4, atol=2e-5)


def test_boundary_has_unit_weight_and_no_derivative_at_zero():
    model = make_model()
    tau = torch.tensor([[0., .3]])
    with mock.patch.object(model.time_from_tau, 'derivative', wraps=model.time_from_tau.derivative) as fn:
        weights = model._time_objective_weights(tau)
    assert weights[0, 0] == 1
    assert (fn.call_args.args[0] > 0).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights[0, 1], model.time_from_tau.derivative(tau[:, 1])[0])


def test_anchor_alignment_excludes_zero_and_uses_only_previous_target_state():
    model = make_model()
    anchors = torch.tensor([[2, 4]])
    target = _target_inputs(model, 1, 8)
    hidden, head = target['last_hidden_states'], target['lm_head_weight']
    q = model._target_anchor_distribution(hidden, head, anchors)
    expected = (hidden[:, [1, 3]] @ head.T).softmax(-1)
    torch.testing.assert_close(q, expected)
    # Perturb states at/after each anchor independently: earlier anchor cannot change.
    changed = hidden.clone(); changed[:, 2:] += 100
    torch.testing.assert_close(model._target_anchor_distribution(changed, head, anchors)[:, 0], q[:, 0])
    sampled, keep = model._sample_anchor_positions(torch.ones(1, 8))
    assert (sampled[keep] > 0).all()


def test_forward_uses_distribution_anchor_and_exact_weighted_denominator():
    torch.manual_seed(4)
    model = make_model()
    inputs = dict(input_ids=torch.randint(0, 64, (1, 8)),
                  hidden_states_list=[torch.randn(1, 8, 32) for _ in range(2)],
                  loss_mask=torch.ones(1, 8), **_target_inputs(model, 1, 8))
    anchors = torch.tensor([[2, 4]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    tau = torch.tensor([[0., .7]])
    with (mock.patch.object(model, '_sample_anchor_positions', return_value=(anchors, keep)),
          mock.patch.object(model, '_sample_training_times', return_value=tau),
          mock.patch.object(model.draft_model, 'forward', wraps=model.draft_model.forward) as forward):
        loss = model(**inputs)[0]
    q_anchor = model._target_anchor_distribution(inputs['last_hidden_states'], inputs['lm_head_weight'], anchors)
    for call in forward.call_args_list:
        state = call.kwargs['flow_state'].view(1, 2, 3, 64)
        sc = call.kwargs['self_condition_state'].view_as(state)
        torch.testing.assert_close(state[:, :, 0], q_anchor)
        torch.testing.assert_close(sc[:, :, 0], q_anchor)
    q = model._target_logits(inputs['last_hidden_states'], inputs['lm_head_weight'], anchors).softmax(-1)
    # Zero-initialized head gives a uniform prediction, independent of noisy inputs.
    mse = (q - 1/64).square().sum(-1) / (1-model.tau_to_time(tau)[..., None]).square()
    position_weights = model.loss_decay_weights[1:]
    expected = (mse * position_weights * model._time_objective_weights(tau)[..., None]).sum() / (2 * position_weights.sum())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(model.draft_model.output_layer.linear.weight.grad).all()


def test_integrator_keeps_distribution_in_both_paths_and_ignores_hard_anchor_ids():
    model = make_model()
    anchors = torch.tensor([[2, 4]])
    hidden = [torch.randn(1, 8, 32) for _ in range(2)]
    q = torch.randn(1, 2, 64).softmax(-1)
    noise = torch.randn(1, 2, 3, 64)
    with mock.patch.object(model.draft_model, 'forward', wraps=model.draft_model.forward) as forward:
        result = model.integrate(hidden, anchors, torch.tensor([[1, 2]]), 3,
                                 noise=noise, anchor_distribution=q)
    torch.testing.assert_close(result[:, :, 0], q)
    for call in forward.call_args_list:
        for key in ('flow_state', 'self_condition_state'):
            torch.testing.assert_close(call.kwargs[key].view_as(result)[:, :, 0], q)
    other = model.integrate(hidden, anchors, torch.tensor([[20, 30]]), 3,
                            noise=noise, anchor_distribution=q)
    torch.testing.assert_close(result, other, rtol=0, atol=0)
    with pytest.raises(ValueError, match='anchor_distribution'):
        model.integrate(hidden, anchors, torch.tensor([[1, 2]]), 1)


def test_integrated_evaluation_requires_target_inputs():
    model = make_model()
    inputs = dict(input_ids=torch.randint(0, 64, (1, 8)),
                  hidden_states_list=[torch.randn(1, 8, 32) for _ in range(2)],
                  loss_mask=torch.ones(1, 8), num_steps=2)
    with pytest.raises(ValueError, match='target hidden states'):
        model.evaluate_integrated(**inputs)
    metrics = model.evaluate_integrated(**inputs, **_target_inputs(model, 1, 8))
    assert metrics['count_per_position'][0] == 0
    assert metrics['count_per_position'][1] == 2
