"""Semantic checks for the FlowSpec conditioning ablations."""

import pytest
import torch

from tests.test_flowspec import _make_wrapper
from torchspec.models.flowspec import _create_flowspec_dense_mask


def test_causal_mask_preserves_prefix_and_isolates_overlapping_blocks():
    anchors = torch.tensor([[4, 5, 0]])
    keep = torch.tensor([[True, True, False]])
    context_len, block_size = 12, 8
    mask = _create_flowspec_dense_mask(anchors, keep, context_len, block_size, causal=True)[0, 0]
    for query in range(24):
        block, offset = divmod(query, block_size)
        expected = torch.zeros(36, dtype=torch.bool)
        if keep[0, block]:
            expected[: anchors[0, block]] = True
            start = context_len + block * block_size
            expected[start : start + offset + 1] = True
        torch.testing.assert_close(mask[query], expected)


def test_causal_network_cannot_use_future_draft_states():
    torch.manual_seed(7)
    model = _make_wrapper(block_size=4, num_anchors=1)
    model.causal_attention = True
    # Exercise attention instead of the zero-initialized identity/output paths.
    for layer in model.draft_model.layers:
        torch.nn.init.normal_(layer.ada_ln_modulation.weight, std=0.1)
        torch.nn.init.normal_(layer.ada_ln_modulation.bias, std=0.1)
    torch.nn.init.normal_(model.draft_model.output_layer.linear.weight, std=0.1)
    anchors = torch.tensor([[4]])
    positions, draft_positions = model._create_position_ids(anchors, 8)
    state = torch.randn(1, 4, 64)
    kwargs = dict(
        timesteps=torch.zeros(1, 4),
        context_feature=torch.randn(1, 8, 32),
        draft_position_ids=draft_positions,
        context_position_ids=positions,
        block_mask=model._create_attention_mask(
            anchors, torch.ones_like(anchors, dtype=torch.bool), 8
        ),
    )
    original = model.draft_model(flow_state=state, **kwargs)
    changed = state.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 5
    perturbed = model.draft_model(flow_state=changed, **kwargs)
    torch.testing.assert_close(original[:, :2], perturbed[:, :2], rtol=0, atol=0)
    assert not torch.allclose(original[:, 2:], perturbed[:, 2:])


@pytest.mark.parametrize("diffusion_forcing", [False, True])
def test_schedule_ablation_keeps_anchor_clean(diffusion_forcing):
    model = _make_wrapper(
        block_size=8, num_anchors=1, use_diffusion_forcing_schedule=diffusion_forcing
    )
    times, _ = model._time_coordinates(torch.tensor([[0.4]]))
    assert times[0, 0, 0] == 1
    if diffusion_forcing:
        assert times[0, 0, 1] > times[0, 0, -1]
    else:
        torch.testing.assert_close(times[0, 0, 1:], torch.full((7,), 0.4))


def _scratch_model(self_conditioning=False, device="cpu", flex=False, hidden_size=32):
    from tests.test_flowspec import _make_config
    from torchspec.models.draft.flowspec import FlowSpecDraftModel
    from torchspec.models.flowspec import FlowSpecModel

    config = _make_config(self_conditioning=self_conditioning, hidden_size=hidden_size)
    config.scratch_tokens = 8
    config.flow_block_size = 9
    return FlowSpecModel(
        FlowSpecDraftModel(config),
        block_size=9,
        num_anchors=2,
        self_condition_probability=0.5 if self_conditioning else 0,
        use_flex_attention=flex,
        anchor_max_gap=0,
    ).to(device)


def _activate_attention(model):
    for layer in model.draft_model.layers:
        torch.nn.init.normal_(layer.ada_ln_modulation.weight, std=0.1)
        torch.nn.init.normal_(layer.ada_ln_modulation.bias, std=0.1)
    torch.nn.init.normal_(model.draft_model.output_layer.linear.weight, std=0.1)


def test_scratch_has_no_output_slots_and_receives_only_attention_gradient():
    torch.manual_seed(19)
    model = _scratch_model()
    _activate_attention(model)
    anchors = torch.tensor([[3, 4]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    context_positions, draft_positions = model._create_position_ids(anchors, 16)
    mask = model._create_attention_mask(anchors, keep, 16)
    assert mask.shape == (1, 1, 34, 50)
    # Overlapping token spans still cannot see each other's scratch or prediction slots.
    assert not mask[0, 0, :17, 33:].any()
    assert not mask[0, 0, 17:, 16:33].any()
    head_lengths = []
    hook = model.draft_model.output_layer.register_forward_pre_hook(
        lambda module, args: head_lengths.append(args[0].shape[1])
    )
    kwargs = dict(
        flow_state=torch.randn(1, 18, 64),
        timesteps=torch.zeros(1, 18),
        context_feature=torch.randn(1, 16, 32),
        draft_position_ids=draft_positions,
        context_position_ids=context_positions,
        block_mask=mask,
    )
    logits = model.draft_model(**kwargs)
    assert logits.shape == (1, 18, 64)
    assert head_lengths == [18]  # 2 * (anchor + 8 predictions), never scratch positions.
    logits.reshape(1, 2, 9, 64)[:, :, 1:].square().mean().backward()
    assert model.draft_model.scratch_embedding.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    blocked = mask.clone()
    for block in range(2):
        scratch_start = 16 + block * 17 + 1
        blocked[..., scratch_start : scratch_start + 8] = False
    kwargs["block_mask"] = blocked
    model.draft_model(**kwargs).reshape(1, 2, 9, 64)[:, :, 1:].square().mean().backward()
    assert model.draft_model.scratch_embedding.grad.abs().sum() == 0
    hook.remove()


@pytest.mark.parametrize("self_conditioning", [False, True])
def test_scratch_train_and_ode_score_eight_predictions(self_conditioning):
    from unittest import mock

    from tests.test_flowspec import _target_inputs

    model = _scratch_model(self_conditioning)
    ids = torch.randint(0, 64, (1, 20))
    hidden = [torch.randn(1, 20, 32) for _ in range(2)]
    mask = torch.ones_like(ids)
    # Force noise boundary so all eight prediction slots are included in loss.
    with mock.patch.object(model, "_sample_training_times", return_value=torch.zeros(1, 2)):
        result = model(ids, hidden, mask, **_target_inputs(model, 1, 20))
    assert torch.isfinite(result[0])
    result[0].backward()
    assert result[4].tolist() == [0] + [2] * 8
    for steps in (1, 32):
        metrics = model.evaluate_integrated(ids, hidden, mask, num_steps=steps)
        assert metrics["count_per_position"].tolist() == [0] + [2] * 8
        assert all(torch.isfinite(v).all() for v in metrics.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
def test_scratch_flex_and_dense_attention_agree():
    torch.manual_seed(29)
    model = _scratch_model(device="cuda", hidden_size=64)
    _activate_attention(model)
    anchors = torch.tensor([[3, 4]], device="cuda")
    tokens = torch.tensor([[7, 8]], device="cuda")
    hidden = [torch.randn(1, 16, 64, device="cuda") for _ in range(2)]
    noise = torch.randn(1, 2, 9, 64, device="cuda")
    dense = model.integrate(hidden, anchors, tokens, 2, noise=noise)
    model.use_flex_attention = True
    flex = model.integrate(hidden, anchors, tokens, 2, noise=noise)
    torch.testing.assert_close(flex, dense, rtol=1e-4, atol=1e-5)

    inputs = dict(
        input_ids=torch.randint(0, 64, (1, 16), device="cuda"),
        hidden_states_list=hidden,
        loss_mask=torch.ones(1, 16, device="cuda"),
        last_hidden_states=torch.randn(1, 16, 64, device="cuda"),
        lm_head_weight=torch.randn(64, 64, device="cuda"),
    )
    loss = model(**inputs)[0]
    loss.backward()
    assert torch.isfinite(model.draft_model.scratch_embedding.grad).all()
    assert model.draft_model.scratch_embedding.grad.abs().sum() > 0
