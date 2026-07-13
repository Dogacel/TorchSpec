"""Tests for ReMDM-style self-conditioning + remask sampling on DFlash.

Covers:
1. Default-off equivalence: self_cond_frac=0 / remask_eval_keep=0 run exactly one
   draft pass and leave the base DFlash training path untouched
2. _reveal_from_confidence: anchor slot never revealed, top-confidence selection,
   per-block reveal counts, invalid/unselected blocks untouched
3. Self-conditioned forward: loss finite, gradients flow to all trainable params,
   frozen embedding gets no gradient, dry-run pass carries no grad
4. Metric exclusion: self-conditioned blocks drop out of metrics but stay in loss
5. Eval two-pass remask metrics present and finite; absent in train mode
6. _consecutive_len run-length math
7. (GPU) block-mask content independence: perturbing one block's content leaves
   every other block's output unchanged (attention correctness under reuse)
"""

import unittest
from unittest import mock

import torch

from torchspec.models.dflash import DFlashModel
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel

V = 128
H = 64
NUM_TARGET_LAYERS = 2


def _make_model(block_size=4, num_anchors=4, **kwargs):
    config = DFlashConfig(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=NUM_TARGET_LAYERS,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        mask_token_id=V - 1,
    )
    draft_model = DFlashDraftModel(config).to(dtype=torch.float32)
    draft_model.freeze_embedding()
    return DFlashModel(
        draft_model=draft_model,
        block_size=block_size,
        num_anchors=num_anchors,
        loss_decay_gamma=7.0,
        **kwargs,
    )


def _make_batch(bsz=2, seq_len=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, V - 1, (bsz, seq_len), generator=g)
    hidden_states_list = [
        torch.randn(bsz, seq_len, H, generator=g) for _ in range(NUM_TARGET_LAYERS)
    ]
    loss_mask = torch.ones(bsz, seq_len)
    lm_head_weight = torch.randn(V, H, generator=g)
    return input_ids, hidden_states_list, loss_mask, lm_head_weight


class TestDefaultOff(unittest.TestCase):
    """self_cond_frac=0 + remask_eval_keep=0 must reproduce stock DFlash exactly."""

    def test_train_runs_single_draft_pass(self):
        model = _make_model()
        batch = _make_batch()
        with mock.patch.object(model, "_run_draft", wraps=model._run_draft) as run:
            loss, acc, *_rest, components = model(*batch)
        self.assertEqual(run.call_count, 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("block_acc_len_det", components)
        self.assertNotIn("block_acc_len_remask", components)

    def test_eval_runs_single_draft_pass_when_disabled(self):
        model = _make_model()
        model.eval()
        batch = _make_batch()
        with mock.patch.object(model, "_run_draft", wraps=model._run_draft) as run:
            with torch.no_grad():
                _, _, *_rest, components = model(*batch)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("block_acc_len_remask", components)

    def test_metrics_counted_on_all_valid_blocks(self):
        model = _make_model()
        batch = _make_batch()
        _, _, _, _, count_per_position, _ = model(*batch)
        self.assertGreater(count_per_position.sum().item(), 0)


class TestRevealFromConfidence(unittest.TestCase):
    def test_top_confidence_selection_and_counts(self):
        model = _make_model(block_size=4, num_anchors=3)
        bsz, n, bs = 2, 3, 4
        conf = torch.randn(bsz, n * bs)
        keep_mask = torch.ones(bsz, n, dtype=torch.bool)
        keep_mask[1, 2] = False  # one invalid block
        reveal_blocks = torch.ones(bsz, n, dtype=torch.bool)
        reveal_blocks[0, 1] = False  # one unselected block
        keep_frac = torch.full((bsz, n), 0.7)  # floor(0.7 * 3) = 2 slots per block

        revealed = model._reveal_from_confidence(conf, keep_mask, reveal_blocks, keep_frac)

        self.assertEqual(revealed.shape, (bsz, n, bs))
        # Anchor slot never revealed
        self.assertFalse(revealed[..., 0].any())
        # Unselected / invalid blocks untouched
        self.assertFalse(revealed[0, 1].any())
        self.assertFalse(revealed[1, 2].any())
        conf3 = conf.view(bsz, n, bs)
        for b in range(bsz):
            for i in range(n):
                if not (reveal_blocks[b, i] and keep_mask[b, i]):
                    continue
                self.assertEqual(revealed[b, i].sum().item(), 2)
                # Revealed slots are exactly the top-2 confident non-anchor slots
                top2 = conf3[b, i, 1:].topk(2).indices + 1
                self.assertEqual(
                    set(revealed[b, i].nonzero().flatten().tolist()), set(top2.tolist())
                )

    def test_zero_keep_frac_reveals_nothing(self):
        model = _make_model(block_size=4, num_anchors=3)
        conf = torch.randn(2, 12)
        ones = torch.ones(2, 3, dtype=torch.bool)
        revealed = model._reveal_from_confidence(conf, ones, ones, torch.zeros(2, 3))
        self.assertFalse(revealed.any())


class TestSelfConditionedTraining(unittest.TestCase):
    def test_two_passes_and_gradient_flow(self):
        model = _make_model(self_cond_frac=1.0, self_cond_keep_max=0.75)
        batch = _make_batch()
        with mock.patch.object(model, "_run_draft", wraps=model._run_draft) as run:
            loss, _, *_rest, _ = model(*batch)
        self.assertEqual(run.call_count, 2)  # dry pass + gradient pass
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)

        loss.backward()
        draft = model.draft_model
        self.assertIsNotNone(draft.context_proj.weight.grad)
        self.assertTrue(draft.context_proj.weight.grad.abs().sum() > 0)
        attn = draft.layers[0].self_attn
        self.assertIsNotNone(attn.q_proj.weight.grad)
        self.assertTrue(attn.q_proj.weight.grad.abs().sum() > 0)
        # Frozen embedding: no gradient may accumulate
        self.assertIsNone(draft.embed_tokens.weight.grad)

    def test_revealed_inputs_are_dry_run_predictions(self):
        model = _make_model(self_cond_frac=1.0, self_cond_keep_max=0.75)
        batch = _make_batch()
        captured = []
        orig = model._run_draft

        def spy(prep, noise_ids):
            captured.append(noise_ids.clone())
            return orig(prep, noise_ids)

        with mock.patch.object(model, "_run_draft", side_effect=spy):
            model(*batch)

        self.assertEqual(len(captured), 2)
        dry_ids, mixed_ids = captured
        changed = dry_ids != mixed_ids
        bs = model.block_size
        changed3 = changed.view(changed.shape[0], -1, bs)
        # Anchor slots never change
        self.assertFalse(changed3[..., 0].any())
        # Every change replaces a MASK token with a real (non-MASK) token
        mask_id = model.draft_model.mask_token_id
        self.assertTrue((dry_ids[changed] == mask_id).all())
        self.assertTrue((mixed_ids[changed] != mask_id).all() or changed.sum() == 0)

    def test_metric_exclusion_and_loss_retention(self):
        # frac=1.0: every valid block is self-conditioned → metrics empty, loss alive
        model = _make_model(self_cond_frac=1.0, self_cond_keep_max=0.75)
        batch = _make_batch()
        loss, acc, loss_pp, acc_pp, count_pp, components = model(*batch)
        self.assertEqual(count_pp.sum().item(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(acc))
        self.assertTrue(torch.isfinite(loss_pp).all())
        self.assertTrue(torch.isfinite(acc_pp).all())
        self.assertEqual(components["block_acc_len_det"].item(), 0.0)


class TestRemaskEval(unittest.TestCase):
    def test_eval_emits_remask_metrics(self):
        model = _make_model(remask_eval_keep=0.5)
        model.eval()
        batch = _make_batch()
        with mock.patch.object(model, "_run_draft", wraps=model._run_draft) as run:
            with torch.no_grad():
                _, _, *_rest, components = model(*batch)
        self.assertEqual(run.call_count, 2)  # main pass + remask pass
        for key in ("block_acc_len_remask", "block_acc_len_revise", "remask_keep_acc"):
            self.assertIn(key, components)
            self.assertTrue(torch.isfinite(components[key]))
        self.assertGreaterEqual(components["remask_keep_acc"].item(), 0.0)
        self.assertLessEqual(components["remask_keep_acc"].item(), 1.0)

    def test_train_mode_does_not_remask(self):
        model = _make_model(remask_eval_keep=0.5)
        batch = _make_batch()
        _, _, *_rest, components = model(*batch)
        self.assertNotIn("block_acc_len_remask", components)


class TestConsecutiveLen(unittest.TestCase):
    def test_run_length_math(self):
        # block 0: slots 1,2 correct, slot 3 wrong → run 2
        # block 1: slot 1 correct, slot 2 wrong, slot 3 correct → run 1
        correct = torch.tensor([[[True, True, True, False], [True, True, False, True]]])
        valid = torch.ones(1, 2, 4, dtype=torch.bool)
        val = DFlashModel._consecutive_len(correct, valid)
        self.assertAlmostEqual(val.item(), 1.5, places=6)

    def test_invalid_blocks_excluded(self):
        correct = torch.tensor([[[True, True, True, True], [True, True, True, True]]])
        valid = torch.ones(1, 2, 4, dtype=torch.bool)
        valid[0, 1] = False  # block 1 has no valid labels → excluded entirely
        val = DFlashModel._consecutive_len(correct, valid)
        self.assertAlmostEqual(val.item(), 3.0, places=6)


class TestTwoRoundSelfCond(unittest.TestCase):
    def test_three_passes_and_gradient_flow(self):
        model = _make_model(
            self_cond_frac=1.0,
            self_cond_keep_max=0.95,
            self_cond_rounds=2,
            self_cond_random_frac=0.2,
        )
        batch = _make_batch()
        with mock.patch.object(model, "_run_draft", wraps=model._run_draft) as run:
            loss, _, *_rest, _ = model(*batch)
        self.assertEqual(run.call_count, 3)  # dry1 + dry2 + gradient pass
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.draft_model.context_proj.weight.grad)
        self.assertIsNone(model.draft_model.embed_tokens.weight.grad)

    def test_rounds_default_keeps_single_dry_pass(self):
        model = _make_model(self_cond_frac=1.0)
        batch = _make_batch()
        with mock.patch.object(model, "_run_draft", wraps=model._run_draft) as run:
            model(*batch)
        self.assertEqual(run.call_count, 2)

    def test_eval_mode_skips_all_dry_passes(self):
        model = _make_model(self_cond_frac=1.0, self_cond_rounds=2)
        model.eval()
        batch = _make_batch()
        with mock.patch.object(model, "_run_draft", wraps=model._run_draft) as run:
            with torch.no_grad():
                model(*batch)
        self.assertEqual(run.call_count, 1)

    def test_invalid_rounds_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(self_cond_rounds=3)


class TestScheduledSampler(unittest.TestCase):
    def _prep(self, model, batch):
        input_ids, hidden_states_list, loss_mask, head = batch
        prep = model._prepare_draft_inputs(input_ids, hidden_states_list, loss_mask)
        return prep, head

    def test_single_pass_equals_one_jump(self):
        model = _make_model()
        model.eval()
        prep, head = self._prep(model, _make_batch())
        preds = model.sample_scheduled(prep, head, num_passes=1)
        hidden = model._run_draft(prep, prep["noise_ids"])
        ref_preds, _ = model._predict_tokens(hidden, head)
        self.assertTrue(torch.equal(preds, ref_preds))

    def test_pass_count_and_monotone_commit_schedule(self):
        model = _make_model(block_size=8, num_anchors=4)
        model.eval()
        prep, head = self._prep(model, _make_batch())
        fracs = []
        orig = model._reveal_from_confidence

        def spy(conf, keep, reveal, keep_frac):
            fracs.append(keep_frac[0, 0].item())
            return orig(conf, keep, reveal, keep_frac)

        with (
            mock.patch.object(model, "_reveal_from_confidence", side_effect=spy),
            mock.patch.object(model, "_run_draft", wraps=model._run_draft) as run,
        ):
            preds = model.sample_scheduled(prep, head, num_passes=4, schedule_alpha=4.0)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(len(fracs), 3)  # selection happens between passes only
        self.assertTrue(all(b > a for a, b in zip(fracs, fracs[1:])))
        self.assertTrue(0.0 < fracs[0] < fracs[-1] < 1.0)
        self.assertEqual(preds.shape, prep["noise_ids"].shape)

    def test_logit_schedule_shape(self):
        # endpoints exact; alpha=0 linear; S-curve steeper than linear mid-course
        self.assertAlmostEqual(DFlashModel._logit_schedule(0.0, 4.0), 0.0, places=9)
        self.assertAlmostEqual(DFlashModel._logit_schedule(1.0, 4.0), 1.0, places=9)
        self.assertAlmostEqual(DFlashModel._logit_schedule(0.5, 4.0), 0.5, places=9)
        self.assertAlmostEqual(DFlashModel._logit_schedule(0.3, 0.0), 0.3, places=9)
        self.assertLess(DFlashModel._logit_schedule(0.25, 6.0), 0.25)
        self.assertGreater(DFlashModel._logit_schedule(0.75, 6.0), 0.75)

    def test_deterministic_given_prep(self):
        model = _make_model()
        model.eval()
        prep, head = self._prep(model, _make_batch())
        a = model.sample_scheduled(prep, head, num_passes=3)
        b = model.sample_scheduled(prep, head, num_passes=3)
        self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for FlexAttention path")
class TestBlockIsolationGPU(unittest.TestCase):
    """The block mask must isolate blocks from each other even when blocks carry
    revealed (non-MASK) content: perturbing one block's tokens may only change
    that block's own outputs."""

    def test_content_independence_across_blocks(self):
        torch.manual_seed(0)
        model = _make_model(block_size=4, num_anchors=32).cuda().to(torch.bfloat16)
        model.eval()
        bsz, seq_len = 1, 128
        input_ids = torch.randint(0, V - 1, (bsz, seq_len), device="cuda")
        hidden_states_list = [
            torch.randn(bsz, seq_len, H, device="cuda", dtype=torch.bfloat16)
            for _ in range(NUM_TARGET_LAYERS)
        ]
        loss_mask = torch.ones(bsz, seq_len, device="cuda")

        with torch.no_grad():
            prep = model._prepare_draft_inputs(input_ids, hidden_states_list, loss_mask)
            base_ids = prep["noise_ids"]
            out_a = model._run_draft(prep, base_ids)

            # Reveal a token inside block 5 only (slot 2)
            bs = model.block_size
            mod_ids = base_ids.clone()
            mod_ids[0, 5 * bs + 2] = 7
            out_b = model._run_draft(prep, mod_ids)

        diff = (out_a - out_b).abs().sum(dim=-1).view(bsz, -1, bs)
        # Block 5 must change...
        self.assertGreater(diff[0, 5].sum().item(), 0.0)
        # ...and no other block may change at all.
        other = torch.cat([diff[0, :5], diff[0, 6:]], dim=0)
        self.assertEqual(other.sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
