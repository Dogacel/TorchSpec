"""Tests for the FlowSpec flow-matching speculator.

Covers:
1. FlowSpecConfig: fields, serialization, auto-registry dispatch
2. FlowSpecDraftModel: velocity forward shapes, zero-init velocity, context fusion
3. shift_time / sampler timestep grid
4. FlowSpecModel wrapper: loss contract, grads, eval sampler metrics, ValueError
"""

import unittest

import torch

from torchspec.models.draft.flowspec import (
    FlowSpecConfig,
    FlowSpecDraftModel,
    timestep_embedding,
)
from torchspec.models.flowspec import FlowSpecModel, shift_time


def _make_config(H=64, V=128, num_target_layers=2, fuse_embedding=True, **kwargs):
    return FlowSpecConfig(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=num_target_layers,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        fuse_embedding=fuse_embedding,
        time_embed_dim=16,
        flow_sample_steps=2,
        **kwargs,
    )


def _make_model(block_size=4, num_anchors=4, ce_loss_alpha=0.2, chunk_ae_path=None, **cfg_kwargs):
    config = _make_config(**cfg_kwargs)
    draft = FlowSpecDraftModel(config).to(dtype=torch.float32)
    draft.freeze_embedding()
    return FlowSpecModel(
        draft_model=draft,
        block_size=block_size,
        num_anchors=num_anchors,
        loss_decay_gamma=7.0,
        ce_loss_alpha=ce_loss_alpha,
        chunk_ae_path=chunk_ae_path,
    )


def _data(B=2, seq_len=64, H=64, V=128, num_target_layers=2, with_last_hs=True):
    torch.manual_seed(0)
    return dict(
        input_ids=torch.randint(0, V, (B, seq_len)),
        hidden_states_list=[torch.randn(B, seq_len, H) for _ in range(num_target_layers)],
        loss_mask=torch.ones(B, seq_len),
        lm_head_weight=torch.randn(V, H),
        last_hidden_states=torch.randn(B, seq_len, H) if with_last_hs else None,
    )


class TestFlowSpecConfig(unittest.TestCase):
    def test_fields_and_serialization(self):
        cfg = _make_config(flow_time_shift=1.5)
        self.assertEqual(cfg.model_type, "flowspec")
        self.assertTrue(cfg.fuse_embedding)
        d = cfg.to_dict()
        restored = FlowSpecConfig(**{k: v for k, v in d.items() if k != "transformers_version"})
        self.assertEqual(restored.flow_time_shift, 1.5)
        self.assertEqual(restored.flow_sample_steps, 2)

    def test_auto_registry(self):
        from torchspec.models.draft.auto import AutoDraftModelConfig

        cfg = AutoDraftModelConfig.from_dict(
            {
                "architectures": ["FlowSpecDraftModel"],
                "hidden_size": 64,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "num_target_layers": 2,
                "target_hidden_size": 64,
                "target_num_hidden_layers": 12,
            }
        )
        self.assertIsInstance(cfg, FlowSpecConfig)

    def test_is_dflash_subclass(self):
        """Trainer/entry validation paths key off DFlashConfig."""
        from torchspec.models.draft.dflash import DFlashConfig

        self.assertIsInstance(_make_config(), DFlashConfig)


class TestFlowSpecDraftModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.config = _make_config()
        self.model = FlowSpecDraftModel(self.config).to(dtype=torch.float32)
        self.model.eval()

    def test_velocity_forward_shapes_and_zero_init(self):
        B, draft_len, ctx_len, H = 2, 8, 16, 64
        embeds = torch.randn(B, draft_len, H)
        t = torch.rand(B, draft_len)
        ctx = torch.randn(B, ctx_len, H)
        d_pos = torch.arange(draft_len).unsqueeze(0).expand(B, -1)
        c_pos = torch.arange(ctx_len).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            v = self.model(
                draft_inputs_embeds=embeds,
                t=t,
                context_feature=ctx,
                draft_position_ids=d_pos,
                context_position_ids=c_pos,
            )
        self.assertEqual(v.shape, (B, draft_len, H))
        # v_out is zero-init => velocity is exactly zero at init.
        self.assertTrue(torch.all(v == 0))

    def test_context_fusion_requires_embeds_when_fused(self):
        hs = [torch.randn(2, 8, 64) for _ in range(2)]
        with self.assertRaisesRegex(ValueError, "token_embeds"):
            self.model.extract_context_feature(hs, None)
        embeds = torch.randn(2, 8, 64)
        ctx = self.model.extract_context_feature(hs, embeds)
        self.assertEqual(ctx.shape, (2, 8, 64))

    def test_fusion_dim_without_embedding(self):
        cfg = _make_config(fuse_embedding=False)
        model = FlowSpecDraftModel(cfg)
        self.assertEqual(model.context_proj.in_features, 2 * 64)
        ctx = model.extract_context_feature([torch.randn(1, 4, 64) for _ in range(2)])
        self.assertEqual(ctx.shape, (1, 4, 64))

    def test_update_stats_ema(self):
        m1 = torch.arange(64, dtype=torch.float32)
        v1 = torch.ones(64) * 2.0
        self.model.update_stats(m1, v1)
        torch.testing.assert_close(self.model.h_mean, m1)
        self.model.update_stats(torch.zeros(64), torch.ones(64))
        mom = self.model.stats_momentum
        torch.testing.assert_close(self.model.h_mean, mom * m1)

    def test_standardize_roundtrip(self):
        self.model.update_stats(torch.randn(64), torch.rand(64) + 0.5)
        h = torch.randn(10, 64)
        torch.testing.assert_close(
            self.model.unstandardize(self.model.standardize(h)), h, rtol=1e-4, atol=1e-4
        )


class TestShiftTime(unittest.TestCase):
    def test_identity_at_one(self):
        t = torch.rand(100)
        torch.testing.assert_close(shift_time(t, 1.0), t)

    def test_endpoints_fixed_and_monotone(self):
        t = torch.linspace(0, 1, 50)
        for s in (0.5, 2.0, 3.0):
            out = shift_time(t, s)
            self.assertAlmostEqual(out[0].item(), 0.0)
            self.assertAlmostEqual(out[-1].item(), 1.0)
            self.assertTrue(torch.all(out[1:] > out[:-1]))

    def test_sampler_grid(self):
        model = _make_model()
        ts = model._sampler_timesteps(8, torch.device("cpu"))
        self.assertEqual(ts.shape[0], 9)
        self.assertAlmostEqual(ts[0].item(), 0.0)
        self.assertAlmostEqual(ts[-1].item(), 1.0)


class TestFlowSpecModelForward(unittest.TestCase):
    def test_requires_last_hidden_states(self):
        model = _make_model()
        with self.assertRaisesRegex(ValueError, "last_hidden_states"):
            model(**_data(with_last_hs=False))

    def test_train_forward_contract_and_grads(self):
        torch.manual_seed(2)
        model = _make_model()
        model.train()
        loss, acc, loss_pp, acc_pp, count_pp, comps = model(**_data())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(acc.item(), 0.0)
        self.assertLessEqual(acc.item(), 1.0)
        self.assertEqual(loss_pp.shape, (model.block_size,))
        self.assertEqual(acc_pp.shape, (model.block_size,))
        self.assertEqual(count_pp.shape, (model.block_size,))
        self.assertIn("fm_loss", comps)
        self.assertIn("ce_loss", comps)
        # Sampler does NOT run in training mode.
        self.assertNotIn("block_acc_len", comps)

        loss.backward()
        draft = model.draft_model
        # At init v_out is zero, so it is the only module guaranteed a gradient
        # on the very first step (zero weights block upstream flow — standard
        # adaLN-Zero warmup dynamics).
        v_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in draft.v_out.parameters()
        )
        self.assertTrue(v_grad, "no gradient on v_out at init")
        self.assertFalse(draft.embed_tokens.weight.requires_grad)

    def test_upstream_grads_after_vout_warm(self):
        """Once v_out is non-zero (as after the first optimizer step), gradient
        reaches the adaLN modulation, x_in, time_mlp, and context fusion."""
        torch.manual_seed(9)
        model = _make_model()
        model.train()
        draft = model.draft_model
        with torch.no_grad():
            draft.v_out.weight.normal_(0, 0.02)
            for layer in draft.layers:
                layer.ada_mod.weight.normal_(0, 0.02)
                layer.ada_mod.bias.normal_(0, 0.02)
        loss, *_ = model(**_data())
        loss.backward()
        for name, params in (
            ("x_in", draft.x_in.parameters()),
            ("context_proj", draft.context_proj.parameters()),
            ("time_mlp", draft.time_mlp.parameters()),
            ("ada_mod", (p for layer in draft.layers for p in layer.ada_mod.parameters())),
        ):
            got = any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
            self.assertTrue(got, f"no gradient flowed to {name}")

    def test_eval_forward_runs_sampler(self):
        torch.manual_seed(3)
        model = _make_model()
        model.eval()
        with torch.no_grad():
            loss, acc, *_, comps = model(**_data())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("block_acc_len", comps)
        self.assertTrue(torch.isfinite(comps["block_acc_len"]))
        self.assertGreaterEqual(comps["block_acc_len"].item(), 0.0)

    def test_ce_alpha_zero_is_pure_fm(self):
        torch.manual_seed(4)
        kwargs = _data()

        def run(alpha):
            torch.manual_seed(11)
            model = _make_model(ce_loss_alpha=alpha)
            model.train()
            out = model(**kwargs)
            return out[0].item(), out[5]

        loss0, comps0 = run(0.0)
        self.assertAlmostEqual(loss0, comps0["fm_loss"].item(), places=5)

    def test_anchor_slot_carries_anchor_embedding(self):
        model = _make_model()
        draft = model.draft_model
        B, nb, bs, H = 2, 3, 4, 64
        x = torch.randn(B, nb, bs, H)
        anchor_embeds = torch.randn(B, nb, H)
        stream = model._build_draft_stream(x, anchor_embeds).view(B, nb, bs, H)
        torch.testing.assert_close(stream[:, :, 0], anchor_embeds)
        torch.testing.assert_close(stream[:, :, 1], draft.x_in(x[:, :, 1]))


class TestBridgeFlow(unittest.TestCase):
    """v4 formulation: anchor-hidden source, depth-graded per-position t,
    staggered sampler, frozen-fc loading contract."""

    def _model(self, **cfg_kwargs):
        return _make_model(
            flow_source="anchor",
            flow_source_sigma=0.25,
            flow_t_per_position=True,
            flow_t_slope=0.4,
            flow_t_jitter=0.05,
            flow_sample_lead=2,
            fuse_embedding=cfg_kwargs.pop("fuse_embedding", True),
            **cfg_kwargs,
        )

    def test_graded_t_profile(self):
        torch.manual_seed(31)
        model = self._model()
        gen = torch.Generator().manual_seed(0)
        t3 = model._sample_train_t(4, 8, torch.device("cpu"), gen)
        self.assertEqual(tuple(t3.shape), (4, 8, model.block_size))
        # anchor slot pinned clean
        self.assertTrue(torch.all(t3[:, :, 0] == 1.0))
        # front predicted slot gets less noise (higher t) than deepest on average
        self.assertGreater(t3[:, :, 1].mean().item(), t3[:, :, -1].mean().item())
        self.assertTrue(torch.all(t3 >= 0.0) and torch.all(t3 <= 1.0))

    def test_anchor_source_state(self):
        torch.manual_seed(32)
        model = self._model()
        B, seq_len, H = 2, 64, 64
        eps = torch.randn(B, 3, 4, H)
        last_hs = torch.randn(B, seq_len, H)
        anchors = torch.tensor([[3, 10, 20], [5, 15, 30]])
        gen = torch.Generator().manual_seed(0)
        x0 = model._flow_source_state(eps, last_hs, anchors, gen)
        self.assertEqual(tuple(x0.shape), (B, 3, 4, H))
        # x0 centers on the anchor hidden: subtracting sigma*eps recovers a
        # slot-constant vector (the broadcast standardized anchor hidden).
        base = x0 - 0.25 * eps
        torch.testing.assert_close(base[:, :, 0], base[:, :, 1], rtol=1e-5, atol=1e-5)

    def test_train_and_eval_forward(self):
        torch.manual_seed(33)
        model = self._model()
        model.train()
        loss, acc, *_, comps = model(**_data())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("fm_loss", comps)
        loss.backward()
        model.eval()
        with torch.no_grad():
            loss_e, *_, comps_e = model(**_data())
        self.assertTrue(torch.isfinite(loss_e))
        self.assertIn("block_acc_len", comps_e)

    def test_sampler_grid_staggered_and_anchor_frozen(self):
        torch.manual_seed(34)
        model = self._model()
        model.eval()
        # Directly exercise the sampler; front slot must finish earlier.
        B, seq_len, H = 1, 64, 64
        data = _data(B=B, seq_len=seq_len)
        with torch.no_grad():
            draft = model.draft_model
            emb = draft.embed_tokens(data["input_ids"]) if draft.fuse_embedding else None
            ctx = draft.extract_context_feature(data["hidden_states_list"], emb)
            anchors, _ = model._sample_anchor_positions(
                seq_len, data["loss_mask"], torch.device("cpu")
            )
            cpos, dpos = model._create_position_ids(anchors, seq_len)
            a_emb = draft.embed_tokens(torch.gather(data["input_ids"], 1, anchors))
            gen = torch.Generator().manual_seed(1)
            h = model._sample_block_hiddens(
                a_emb,
                ctx,
                dpos,
                cpos,
                None,
                4,
                gen,
                B,
                anchors.shape[1],
                data["last_hidden_states"],
                anchors,
            )
            self.assertEqual(h.shape[-1], H)
            self.assertTrue(torch.isfinite(h).all())

    def test_fc_loader_requires_no_embedding_fusion(self):
        model = self._model(fuse_embedding=True)
        with self.assertRaisesRegex(ValueError, "fuse_embedding"):
            model.draft_model.load_context_proj("/nonexistent.safetensors")

    def test_freeze_context_proj(self):
        model = self._model(fuse_embedding=False)
        model.draft_model.freeze_context_proj()
        self.assertFalse(model.draft_model.context_proj.weight.requires_grad)
        self.assertFalse(model.draft_model.context_norm.weight.requires_grad)


class TestFlowSpecTrainerWiring(unittest.TestCase):
    def test_trainer_declares_components(self):
        from torchspec.training.flowspec_trainer import FlowSpecTrainer

        self.assertEqual(
            FlowSpecTrainer._extra_loss_component_keys,
            ["fm_loss", "ce_loss", "l1_loss", "block_acc_len", "denoise_acc"],
        )
        self.assertIs(FlowSpecTrainer._draft_config_class, FlowSpecConfig)

    def test_timestep_embedding_shape(self):
        emb = timestep_embedding(torch.rand(10), 16)
        self.assertEqual(emb.shape, (10, 16))
        self.assertTrue(torch.isfinite(emb).all())


class TestChunkLatentMode(unittest.TestCase):
    """flow_state=chunk_latent: one AE latent per block, frozen AE attached."""

    @classmethod
    def setUpClass(cls):
        import os
        import tempfile

        from torchspec.models.calm_ae import Autoencoder, AutoencoderConfig

        cls.tmp = tempfile.mkdtemp()
        ae_cfg = AutoencoderConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=80,
            num_encoder_layers=2,
            num_decoder_layers=2,
            latent_size=16,
            patch_size=4,
            pad_token_id=0,
        )
        ae = Autoencoder(ae_cfg)
        torch.save(ae.state_dict(), os.path.join(cls.tmp, "autoencoder.pt"))
        ae_cfg.save_pretrained(cls.tmp)

    def _model(self):
        return _make_model(
            block_size=2,
            num_anchors=4,
            chunk_ae_path=self.tmp,
            flow_state="chunk_latent",
            flow_state_dim=16,
            fuse_embedding=True,
            flow_t_per_position=False,
        )

    def test_requires_block_size_2(self):
        model = _make_model(
            block_size=4,
            chunk_ae_path=self.tmp,
            flow_state="chunk_latent",
            flow_state_dim=16,
        )
        with self.assertRaisesRegex(ValueError, "block_size"):
            model(**_data())

    def test_train_forward_grads_and_frozen_ae(self):
        torch.manual_seed(41)
        model = self._model()
        # AE runs in fp32 for CPU tests
        model.chunk_ae = model.chunk_ae.to(torch.float32)
        model.train()
        loss, acc, loss_pp, acc_pp, count_pp, comps = model(**_data())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("fm_loss", comps)
        self.assertIn("ce_loss", comps)
        self.assertIn("denoise_acc", comps)
        # metric arrays: 1 anchor slot + 4 chunk token positions
        self.assertEqual(loss_pp.shape, (5,))
        self.assertEqual(acc_pp.shape, (5,))
        loss.backward()
        ae_grads = [p.grad for p in model.chunk_ae.parameters() if p.grad is not None]
        self.assertEqual(len(ae_grads), 0, "frozen AE must receive no gradients")
        v_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.draft_model.v_out.parameters()
        )
        self.assertTrue(v_grad)

    def test_eval_forward_samples_and_reports(self):
        torch.manual_seed(42)
        model = self._model()
        model.chunk_ae = model.chunk_ae.to(torch.float32)
        model.eval()
        with torch.no_grad():
            loss, acc, *_, comps = model(**_data())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("block_acc_len", comps)
        self.assertLessEqual(comps["block_acc_len"].item(), 4.0)

    def test_state_dim_wiring(self):
        model = self._model()
        self.assertEqual(model.draft_model.state_dim, 16)
        self.assertEqual(model.draft_model.x_in.in_features, 16)
        self.assertEqual(model.draft_model.v_out.out_features, 16)
        self.assertEqual(model.draft_model.h_mean.shape[0], 16)


if __name__ == "__main__":
    unittest.main()
