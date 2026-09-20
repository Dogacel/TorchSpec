"""XM winner selection, gradients, RNG pairing, and honest evaluation."""

import copy
from unittest import mock

import pytest
import torch

from tests.test_flowspec import _target_inputs
from tests.test_flowspec_token_endpoint import make_model


def inputs_for(model):
    return dict(
        input_ids=torch.randint(64, (1, 8)),
        hidden_states_list=[torch.randn(1, 8, 32) for _ in range(2)],
        loss_mask=torch.ones(1, 8),
        **_target_inputs(model, 1, 8),
    )


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        make_model(xm_best_of_k=k)


def test_k1_default_exact_and_eval_never_selects():
    torch.manual_seed(42)
    baseline = make_model()
    xm = copy.deepcopy(baseline)
    xm.xm_best_of_k = 4
    inputs = inputs_for(baseline)
    baseline.eval()
    xm.eval()
    torch.manual_seed(19)
    expected = baseline(**inputs)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    with mock.patch.object(
        xm, "_select_xm_candidates", side_effect=AssertionError("eval selection")
    ):
        actual = xm(**inputs)
    for a, b in zip(actual[:-1], expected[:-1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("sc_enabled", [False, True])
def test_streamed_search_matches_explicit_block_min_and_gradients(sc_enabled):
    torch.manual_seed(104)
    xm = make_model(xm_best_of_k=4)
    # Break the initial zero output projections so selection depends on noise.
    with torch.no_grad():
        for module in xm.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(0, 0.10)
    baseline = copy.deepcopy(xm)
    baseline.xm_best_of_k = 1
    xm.self_condition_probability = baseline.self_condition_probability = float(sc_enabled)
    inputs = inputs_for(xm)
    noises = [torch.randn(1, 2, 3, 64) for _ in range(4)]
    anchors = torch.tensor([[1, 4]])
    tau = torch.tensor([[0.0, 0.55]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    weights = xm.loss_decay_weights.view(1, 1, -1).expand(1, 2, -1).clone()
    weights[..., 0] = 0

    candidate_tokens = []

    def capture(logits, target, time):
        result = baseline._prediction_loss_original(logits, target, time)
        candidate_tokens.append(result)
        return result

    baseline._prediction_loss_original = baseline._prediction_loss
    for noise in noises:
        torch.manual_seed(29)
        with (
            mock.patch.object(baseline, "_sample_anchor_positions", return_value=(anchors, keep)),
            mock.patch.object(baseline, "_sample_training_times", return_value=tau),
            mock.patch.object(baseline, "_sample_noise", return_value=noise),
            mock.patch.object(baseline, "_prediction_loss", side_effect=capture),
        ):
            baseline(**inputs)
    expected_rng = torch.get_rng_state()
    scores = torch.stack([(ce * weights).sum(-1) for ce in candidate_tokens])
    expected = scores.min(0).values.sum() / weights.sum()
    expected.backward()

    torch.manual_seed(29)
    with (
        mock.patch.object(xm, "_sample_anchor_positions", return_value=(anchors, keep)),
        mock.patch.object(xm, "_sample_training_times", return_value=tau),
        mock.patch.object(xm, "_sample_noise", side_effect=noises),
    ):
        actual = xm(**inputs)
    assert torch.equal(expected_rng, torch.get_rng_state())
    torch.testing.assert_close(actual[0], expected, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(actual[-1]["xm_selected_loss"], actual[0].detach())
    assert actual[-1]["xm_selection_gain"] >= 0
    actual[0].backward()
    checked = 0
    for (name, a), (_, b) in zip(xm.named_parameters(), baseline.named_parameters()):
        if a.grad is not None or b.grad is not None:
            assert a.grad is not None and b.grad is not None, name
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6, msg=name)
            checked += 1
    assert checked > 10
