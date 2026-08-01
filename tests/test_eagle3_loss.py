"""Tests for Eagle3 loss computation paths.

Verifies that:
1. compiled_forward_kl_loss matches a naive reference implementation.
2. compute_target_p_padded produces correct shapes and valid probabilities
   for both pruning and non-pruning paths.
3. The lazy target path (non-pruning, target_p_padded=None) produces identical
   losses to the pre-computed target_p_padded path.
"""

import unittest

import torch
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig

from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from torchspec.models.eagle3 import (
    Eagle3Model,
    _build_loss_candidates,
    compute_lazy_target_padded,
    compute_target_p_padded,
)


from torchspec.models.ops.loss import (
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,
)


def _shift_loss_candidates(
    loss_valid_idx: torch.Tensor,
    *,
    depth: int,
    base_seq_length: int,
    step_seq_length: int,
    candidate_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-depth reference for ``_build_loss_candidates``.

    Kept as a test oracle only: production builds every TTT depth in one shot.
    """
    batch_idx = torch.div(loss_valid_idx, base_seq_length, rounding_mode="floor")
    base_position = loss_valid_idx.remainder(base_seq_length)
    shifted_position = base_position - depth
    valid_weights = (shifted_position >= 0) & (shifted_position < step_seq_length)
    valid_idx = batch_idx * step_seq_length + shifted_position.clamp(0, step_seq_length - 1)
    if candidate_valid is not None:
        valid_weights = valid_weights & candidate_valid
    return valid_idx, valid_weights


def _reference_forward_kl_loss(hs_flat, target_p_flat, norm_weight, lm_head_weight, norm_eps):
    """Pure-Python reference (no torch.compile) for validation."""
    hs_f32 = hs_flat.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs_flat.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    log_p = F.log_softmax(logits.float(), dim=-1)
    loss = -(target_p_flat * log_p).sum(-1).mean()
    acc = (logits.argmax(-1) == target_p_flat.argmax(-1)).float().mean()
    return loss, acc


def _make_config(H=128, V=256, draft_V=None, num_heads=4, num_kv_heads=2):
    config = LlamaConfig(
        hidden_size=H,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=H * 4,
        max_position_embeddings=1024,
        vocab_size=V,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_scaling=None,
        pretraining_tp=1,
        pad_token_id=0,
    )
    config.draft_vocab_size = draft_V or V
    return config


def _make_model(config, length=3, attention_backend="sdpa", device="cpu"):
    draft_model = LlamaForCausalLMEagle3(config, attention_backend=attention_backend)
    draft_model = draft_model.to(device=device, dtype=torch.bfloat16)
    model = Eagle3Model(
        draft_model,
        length=length,
        attention_backend=attention_backend,
    )
    model.eval()
    return model


def _make_batch(B, T, H, V, device="cpu"):
    input_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    loss_mask = torch.zeros(B, T, device=device)
    loss_mask[:, T // 4 :] = 1.0
    hidden_states = torch.randn(B, T, H * 3, device=device, dtype=torch.bfloat16)
    target_hidden_states = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target_hidden_states": target_hidden_states,
    }


class TestCompiledForwardKLLoss(unittest.TestCase):
    """compiled_forward_kl_loss should match the reference implementation."""

    def test_matches_reference(self):
        torch.manual_seed(42)
        N, H, V = 32, 128, 256
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        raw_logits = F.linear(hs.float(), lm_head_weight.float())
        target_p = F.softmax(raw_logits + torch.randn_like(raw_logits) * 0.5, dim=-1)

        loss_sum, correct, count = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, norm_eps
        )
        loss = loss_sum / count
        acc = correct / count
        ref_loss, ref_acc = _reference_forward_kl_loss(
            hs, target_p, norm_weight, lm_head_weight, norm_eps
        )

        torch.testing.assert_close(loss, ref_loss, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(acc, ref_acc, atol=1e-3, rtol=1e-3)

    def test_perfect_prediction_equals_entropy(self):
        """When draft == target, cross-entropy loss equals target entropy."""
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        norm_weight = torch.ones(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        hs = torch.randn(N, H, dtype=torch.float32)
        # The loss function computes H(target, draft) = H(target) + KL(target||draft).
        # When target_p is derived from the same logits, KL ≈ 0 so loss ≈ H(target).
        variance = hs.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + norm_eps)
        norm_hs = hs * rstd * norm_weight
        logits = F.linear(norm_hs, lm_head_weight)
        target_p = F.softmax(logits, dim=-1)
        expected_entropy = -(target_p * target_p.log()).sum(-1).mean()

        loss_sum, correct, count = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, norm_eps
        )
        loss = loss_sum / count
        acc = correct / count
        torch.testing.assert_close(loss, expected_entropy, atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(acc.item(), 1.0, places=2)

    def test_loss_non_negative_and_finite(self):
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        target_p = F.softmax(torch.randn(N, V), dim=-1)
        valid_idx = torch.arange(N)

        loss_sum, correct, count = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, 1e-6
        )
        loss = loss_sum / count
        acc = correct / count
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertGreaterEqual(acc.item(), 0.0)
        self.assertLessEqual(acc.item(), 1.0)

    def test_fixed_capacity_weights_match_compacted_loss(self):
        torch.manual_seed(1)
        N, H, V = 8, 32, 24
        hs = torch.randn(N, H)
        target_p = F.softmax(torch.randn(N, V), dim=-1)
        norm_weight = torch.randn(H)
        lm_head_weight = torch.randn(V, H)
        valid_idx = torch.tensor([0, 2, 4, 6, 1, 3])
        valid_weights = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)

        packed = compiled_forward_kl_loss(
            hs,
            target_p,
            valid_idx,
            norm_weight,
            lm_head_weight,
            1e-6,
            valid_weights,
        )
        compacted = compiled_forward_kl_loss(
            hs,
            target_p,
            valid_idx[valid_weights],
            norm_weight,
            lm_head_weight,
            1e-6,
        )
        for actual, expected in zip(packed, compacted):
            torch.testing.assert_close(actual, expected)

    def test_compact_target_matches_dense_target(self):
        torch.manual_seed(2)
        N, H, V = 10, 32, 24
        hs = torch.randn(N, H)
        dense_target = F.softmax(torch.randn(N, V), dim=-1)
        norm_weight = torch.randn(H)
        lm_head_weight = torch.randn(V, H)
        valid_idx = torch.tensor([1, 3, 4, 8])

        dense = compiled_forward_kl_loss(
            hs,
            dense_target,
            valid_idx,
            norm_weight,
            lm_head_weight,
            1e-6,
        )
        compact_target = dense_target.index_select(0, valid_idx)
        compact = compiled_forward_kl_loss(
            hs,
            compact_target,
            valid_idx,
            norm_weight,
            lm_head_weight,
            1e-6,
            None,
            True,
            compact_target.argmax(dim=-1),
        )
        for actual, expected in zip(compact, dense):
            torch.testing.assert_close(actual, expected)

        dense_hs = hs.detach().clone().requires_grad_(True)
        compact_hs = hs.detach().clone().requires_grad_(True)
        dense_loss = compiled_forward_kl_loss(
            dense_hs,
            dense_target,
            valid_idx,
            norm_weight,
            lm_head_weight,
            1e-6,
        )[0]
        compact_loss = compiled_forward_kl_loss(
            compact_hs,
            compact_target,
            valid_idx,
            norm_weight,
            lm_head_weight,
            1e-6,
            None,
            True,
            compact_target.argmax(dim=-1),
        )[0]
        dense_loss.backward()
        compact_loss.backward()
        torch.testing.assert_close(compact_hs.grad, dense_hs.grad)


class TestShiftLossCandidates(unittest.TestCase):
    def test_depth_shift_preserves_batch_boundaries_and_masks_borders(self):
        base_idx = torch.tensor([0, 2, 5, 6, 8, 11])
        candidate_valid = torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool)
        shifted, weights = _shift_loss_candidates(
            base_idx,
            depth=2,
            base_seq_length=6,
            step_seq_length=6,
            candidate_valid=candidate_valid,
        )

        torch.testing.assert_close(shifted, torch.tensor([0, 0, 3, 6, 6, 9]))
        torch.testing.assert_close(
            weights,
            torch.tensor([0, 1, 0, 0, 1, 1], dtype=torch.bool),
        )

    def test_vectorized_depth_candidates_match_scalar_reference(self):
        base_idx = torch.tensor([0, 2, 5, 6, 8, 11])
        candidate_valid = torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool)
        indices, weights = _build_loss_candidates(
            base_idx,
            depth_count=4,
            base_seq_length=6,
            step_seq_length=6,
            candidate_valid=candidate_valid,
        )
        for depth in range(4):
            expected_indices, expected_weights = _shift_loss_candidates(
                base_idx,
                depth=depth,
                base_seq_length=6,
                step_seq_length=6,
                candidate_valid=candidate_valid,
            )
            torch.testing.assert_close(indices[depth], expected_indices)
            torch.testing.assert_close(weights[depth], expected_weights)


class TestComputeTargetPPadded(unittest.TestCase):
    """compute_target_p_padded: shape, dtype, and probability correctness."""

    def test_pruning_shapes_and_candidate_mask(self):
        torch.manual_seed(0)
        B, T, D = 2, 16, 64
        V_target, V_draft = 128, 32
        length = 3
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V_target, D, dtype=torch.bfloat16)
        loss_mask = torch.ones(B, T)

        t2d = torch.zeros(V_target, dtype=torch.bool)
        t2d[:V_draft] = True

        result = compute_target_p_padded(
            hs,
            weight,
            t2d=t2d,
            loss_mask=loss_mask,
            length=length,
        )

        self.assertIsInstance(result, PrecomputedTarget)
        self.assertTrue(result.compact)
        self.assertEqual(result.target_p_padded.shape, (B * T, V_draft))
        self.assertIsNotNone(result.candidate_valid)
        self.assertEqual(result.candidate_valid.shape, (B * T,))
        self.assertEqual(result.target_argmax.shape, (B * T,))
        sums = result.target_p_padded.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)

    def test_loss_mask_respected_in_candidate_mask(self):
        """Only loss-mask positions should be represented as candidates."""
        torch.manual_seed(0)
        B, T, D = 1, 32, 64
        V_target, V_draft = 128, 32
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V_target, D, dtype=torch.bfloat16)
        loss_mask = torch.zeros(B, T)
        loss_mask[:, T // 2 :] = 1.0

        t2d = torch.zeros(V_target, dtype=torch.bool)
        t2d[:V_draft] = True

        result = compute_target_p_padded(
            hs,
            weight,
            t2d=t2d,
            loss_mask=loss_mask,
            length=3,
        )

        self.assertEqual(result.candidate_valid.shape, (T // 2,))

    def test_full_vocab_target_is_compacted_to_loss_positions(self):
        torch.manual_seed(3)
        B, T, D, V = 2, 12, 32, 48
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V, D, dtype=torch.bfloat16)
        loss_valid_idx = torch.tensor([2, 5, 12, 20])

        result = compute_target_p_padded(
            hs,
            weight,
            loss_valid_idx=loss_valid_idx,
        )
        selected_hs = hs.reshape(-1, D).index_select(0, loss_valid_idx)
        expected = F.softmax(F.linear(selected_hs, weight).float(), dim=-1)

        self.assertTrue(result.compact)
        self.assertIsNone(result.candidate_valid)
        torch.testing.assert_close(result.target_p_padded, expected)
        torch.testing.assert_close(result.target_argmax, expected.argmax(dim=-1))


class TestLazyVsPrecomputedTarget(unittest.TestCase):
    """The lazy path (target_p_padded=None) must produce identical losses."""

    def _run_both_paths(self, device="cpu"):
        torch.manual_seed(42)
        H, V, B, T, length = 128, 256, 1, 32, 3

        config = _make_config(H=H, V=V)
        model = _make_model(config, length=length, device=device)
        batch = _make_batch(B, T, H, V, device=device)

        draft_model = model.draft_model
        _, lm_head_weight, _ = draft_model.get_lm_head_params()

        loss_valid_idx = batch["loss_mask"].reshape(-1).bool().nonzero().flatten()
        precomputed = compute_target_p_padded(
            batch["target_hidden_states"],
            lm_head_weight.detach(),
            loss_valid_idx=loss_valid_idx,
            length=length,
        )
        with torch.no_grad():
            plosses_pre, _, acces_pre, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target=precomputed,
                loss_mask=batch["loss_mask"],
                loss_valid_idx=loss_valid_idx,
                hidden_states=batch["hidden_states"],
            )

        lazy = compute_lazy_target_padded(
            batch["target_hidden_states"],
            lm_head_weight,
            length,
        )
        with torch.no_grad():
            plosses_lazy, _, acces_lazy, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target=lazy,
                loss_mask=batch["loss_mask"],
                hidden_states=batch["hidden_states"],
            )

        return plosses_pre, acces_pre, plosses_lazy, acces_lazy

    def test_losses_match_cpu(self):
        plosses_pre, acces_pre, plosses_lazy, acces_lazy = self._run_both_paths("cpu")
        for i, (pre, lazy) in enumerate(zip(plosses_pre, plosses_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Loss mismatch at position {i}",
            )
        for i, (pre, lazy) in enumerate(zip(acces_pre, acces_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Accuracy mismatch at position {i}",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_losses_match_cuda(self):
        plosses_pre, acces_pre, plosses_lazy, acces_lazy = self._run_both_paths("cuda")
        for i, (pre, lazy) in enumerate(zip(plosses_pre, plosses_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=1e-3,
                rtol=1e-3,
                msg=f"Loss mismatch at position {i}",
            )
        for i, (pre, lazy) in enumerate(zip(acces_pre, acces_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=1e-3,
                rtol=1e-3,
                msg=f"Accuracy mismatch at position {i}",
            )


class TestRotaryConfigWiring(unittest.TestCase):
    """Model config should fully wire RoPE settings into rotary embeddings."""

    def test_yarn_uses_rope_theta_as_base(self):
        config = LlamaConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=262144,
            vocab_size=256,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=50000.0,
            rope_scaling={
                "type": "yarn",
                "factor": 64.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
            pretraining_tp=1,
            pad_token_id=0,
        )
        config.draft_vocab_size = 256

        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        rotary = model.midlayer.self_attn.rotary_emb

        self.assertEqual(rotary.base, 50000.0)
        self.assertEqual(rotary.original_max_position_embeddings, 4096)
        self.assertEqual(rotary.scaling_factor, 64.0)


def _make_mask_patterns(BT):
    """Return (name, valid_idx) pairs covering diverse masking patterns."""
    patterns = []

    # contiguous first half
    m = torch.zeros(BT, dtype=torch.bool)
    m[: BT // 2] = True
    patterns.append(("first_half", m.nonzero().squeeze(-1)))

    # contiguous second half
    m = torch.zeros(BT, dtype=torch.bool)
    m[BT // 2 :] = True
    patterns.append(("second_half", m.nonzero().squeeze(-1)))

    # every other position (strided)
    m = torch.zeros(BT, dtype=torch.bool)
    m[::2] = True
    patterns.append(("strided", m.nonzero().squeeze(-1)))

    # random sparse (~25%)
    g = torch.Generator().manual_seed(99)
    m = torch.rand(BT, generator=g) < 0.25
    patterns.append(("random_sparse", m.nonzero().squeeze(-1)))

    # single valid position
    patterns.append(("single", torch.tensor([BT // 3])))

    # all valid
    patterns.append(("all", torch.arange(BT)))

    return patterns


class TestValidIdxSubsetting(unittest.TestCase):
    """valid_idx filtering must produce the same loss as manual pre-filtering."""

    BT, H, V = 64, 128, 256

    def _check_forward_kl(self, valid_idx):
        torch.manual_seed(7)
        hs_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        norm_weight = torch.randn(self.H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        tp_flat = F.softmax(torch.randn(self.BT, self.V), dim=-1)
        norm_eps = 1e-6

        loss_sum, correct, count = compiled_forward_kl_loss(
            hs_flat,
            tp_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            norm_eps,
        )
        loss = loss_sum / count
        acc = correct / count

        hs_valid = hs_flat[valid_idx]
        tp_valid = tp_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_sum_ref, correct_ref, count_ref = compiled_forward_kl_loss(
            hs_valid,
            tp_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            norm_eps,
        )
        loss_ref = loss_sum_ref / count_ref
        acc_ref = correct_ref / count_ref

        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acc, acc_ref, atol=1e-5, rtol=1e-5)

    def _check_forward_kl_from_hs(self, valid_idx):
        torch.manual_seed(7)
        hs_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        ths_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        norm_weight = torch.randn(self.H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        target_lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        norm_eps = 1e-6

        loss_sum, correct, count = compiled_forward_kl_loss_from_hs(
            hs_flat,
            ths_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            target_lm_head_weight,
            norm_eps,
        )
        loss = loss_sum / count
        acc = correct / count

        hs_valid = hs_flat[valid_idx]
        ths_valid = ths_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_sum_ref, correct_ref, count_ref = compiled_forward_kl_loss_from_hs(
            hs_valid,
            ths_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            target_lm_head_weight,
            norm_eps,
        )
        loss_ref = loss_sum_ref / count_ref
        acc_ref = correct_ref / count_ref

        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acc, acc_ref, atol=1e-5, rtol=1e-5)


# Dynamically generate one test method per mask pattern per loss function.
for _name, _vidx in _make_mask_patterns(TestValidIdxSubsetting.BT):

    def _make_kl(vidx=_vidx):
        def test(self):
            self._check_forward_kl(vidx)

        return test

    def _make_kl_from_hs(vidx=_vidx):
        def test(self):
            self._check_forward_kl_from_hs(vidx)

        return test

    setattr(TestValidIdxSubsetting, f"test_forward_kl_{_name}", _make_kl())
    setattr(TestValidIdxSubsetting, f"test_forward_kl_from_hs_{_name}", _make_kl_from_hs())


if __name__ == "__main__":
    unittest.main()
