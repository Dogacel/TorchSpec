# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU contract tests for the FlowSpec draft network."""

import math
import unittest

import torch
import torch.nn.functional as F

from torchspec.config.train_config import Config
from torchspec.models.draft.auto import AutoDraftModelConfig
from torchspec.models.draft.flowspec import (
    FlowSpecAttention,
    FlowSpecConfig,
    FlowSpecDecoderLayer,
    FlowSpecDraftModel,
    FlowSpecFinalLayer,
    FlowSpecTimestepEmbedding,
    FlowSpecVocabularyEmbedding,
    build_target_layer_ids,
)
from torchspec.models.flowspec import (
    FlowSpecModel,
    _create_flowspec_dense_mask,
    _create_flowspec_mask_mod,
)
from torchspec.models.flowspec_schedule import (
    LogitNormalMixtureTimeSchedule,
    RecognitionTimeSchedule,
)
from torchspec.training.flowspec_trainer import (
    FlowSpecTrainer,
    _rank_sample_limit,
)


def _make_config(
    hidden_size: int = 32,
    vocab_size: int = 64,
    num_hidden_layers: int = 2,
    num_target_layers: int = 2,
    self_conditioning: bool = False,
) -> FlowSpecConfig:
    return FlowSpecConfig(
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        time_condition_size=hidden_size,
        time_frequency_size=16,
        num_target_layers=num_target_layers,
        target_hidden_size=hidden_size,
        target_num_hidden_layers=12,
        output_bias=True,
        tie_word_embeddings=False,
        self_conditioning=self_conditioning,
    )


def _make_wrapper(
    block_size: int = 3,
    num_anchors: int = 4,
    boundary_probability: float = 0.0,
    clean_deadline_jitter: float = 0.0,
    block_schedule_overlap: float = 1.0,
    use_diffusion_forcing_schedule: bool = True,
    time_schedule: str = "simplex_sharp",
    noise_type: str = "simplex",
    self_condition_probability: float = 0.0,
) -> FlowSpecModel:
    draft_model = FlowSpecDraftModel(
        _make_config(self_conditioning=self_condition_probability > 0.0)
    )
    return FlowSpecModel(
        draft_model=draft_model,
        block_size=block_size,
        num_anchors=num_anchors,
        boundary_probability=boundary_probability,
        clean_deadline_jitter=clean_deadline_jitter,
        block_schedule_overlap=block_schedule_overlap,
        use_diffusion_forcing_schedule=use_diffusion_forcing_schedule,
        time_schedule=time_schedule,
        noise_type=noise_type,
        self_condition_probability=self_condition_probability,
        use_flex_attention=False,
    )


def _target_inputs(model: FlowSpecModel, batch_size: int, sequence_len: int) -> dict:
    hidden_size = model.draft_model.config.target_hidden_size
    return {
        "last_hidden_states": torch.randn(batch_size, sequence_len, hidden_size),
        "lm_head_weight": torch.randn(model.draft_model.config.vocab_size, hidden_size),
    }


class TestFlowSpecConfig(unittest.TestCase):
    def test_attributes_and_serialization(self):
        config = _make_config()
        restored = FlowSpecConfig.from_dict(config.to_dict())

        self.assertEqual(config.model_type, "flowspec")
        self.assertEqual(restored.hidden_size, config.hidden_size)
        self.assertEqual(restored.vocab_size, config.vocab_size)
        self.assertEqual(restored.time_frequency_size, 16)
        self.assertFalse(restored.tie_word_embeddings)
        self.assertFalse(restored.self_conditioning)

    def test_rejects_invalid_attention_dimensions(self):
        invalid = (
            {"hidden_size": 30, "num_attention_heads": 4},
            {"num_attention_heads": 3, "num_key_value_heads": 2},
            {"head_dim": 7},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FlowSpecConfig(**kwargs)

    def test_rejects_invalid_time_dimensions(self):
        for frequency_size in (0, 1, 3, 255):
            with self.subTest(frequency_size=frequency_size), self.assertRaises(ValueError):
                FlowSpecConfig(time_frequency_size=frequency_size)

        for frequency_size in (1, 3, 15):
            with self.subTest(frequency_size=frequency_size), self.assertRaises(ValueError):
                FlowSpecTimestepEmbedding(32, frequency_size=frequency_size)

    def test_target_layer_selection(self):
        self.assertEqual(build_target_layer_ids(1, 36), [18])
        self.assertEqual(build_target_layer_ids(5, 36), [1, 9, 17, 25, 33])

    def test_auto_config_dispatch(self):
        config = AutoDraftModelConfig.from_dict(
            {
                **_make_config().to_dict(),
                "architectures": ["FlowSpecDraftModel"],
            }
        )
        self.assertIsInstance(config, FlowSpecConfig)

    def test_ode_eval_defaults_are_disabled(self):
        training = Config().training
        self.assertEqual(training.flowspec_loss_decay_gamma, 7.0)
        self.assertEqual(training.flowspec_boundary_probability, 0.0)
        self.assertEqual(training.flowspec_clean_deadline_jitter, 0.0)
        self.assertEqual(training.flowspec_block_schedule_overlap, 1.0)
        self.assertTrue(training.flowspec_use_diffusion_forcing_schedule)
        self.assertEqual(training.flowspec_time_schedule, "simplex_sharp")
        self.assertEqual(training.flowspec_noise_type, "simplex")
        self.assertEqual(training.flowspec_self_condition_probability, 0.0)
        self.assertEqual(training.flowspec_ode_eval_steps, 0)
        self.assertIsNone(training.flowspec_ode_eval_step_counts)
        self.assertEqual(training.flowspec_ode_eval_max_samples, 0)
        self.assertEqual(training.flowspec_ode_eval_seed, 1234)


class TestFlowSpecTimestepEmbedding(unittest.TestCase):
    def test_fractional_times_with_leading_dimensions(self):
        embedding = FlowSpecTimestepEmbedding(
            condition_size=24,
            frequency_size=16,
        )
        output = embedding(torch.rand(2, 3, 4))
        self.assertEqual(output.shape, (2, 3, 4, 24))

    def test_zero_time_frequency_features(self):
        embedding = FlowSpecTimestepEmbedding(
            condition_size=16,
            frequency_size=8,
        )
        features = embedding._frequency_embedding(torch.zeros(2))
        torch.testing.assert_close(features[:, :4], torch.ones(2, 4))
        torch.testing.assert_close(features[:, 4:], torch.zeros(2, 4))

    def test_mlp_uses_bias(self):
        embedding = FlowSpecTimestepEmbedding(
            condition_size=16,
            frequency_size=8,
        )
        self.assertIsNotNone(embedding.mlp[0].bias)
        self.assertIsNotNone(embedding.mlp[2].bias)

    def test_bfloat16_mlp_accepts_float_time_features(self):
        embedding = FlowSpecTimestepEmbedding(
            condition_size=16,
            frequency_size=8,
        ).to(torch.bfloat16)
        output = embedding(torch.tensor([0.25, 0.75], dtype=torch.float32))
        self.assertEqual(output.dtype, torch.bfloat16)


class TestFlowSpecVocabularyEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.embedding = FlowSpecVocabularyEmbedding(vocab_size=11, hidden_size=7)

    def test_dense_projection_matches_matrix_multiplication(self):
        flow_state = torch.randn(2, 5, 11)
        expected = torch.matmul(flow_state.float(), self.embedding.weight.float())
        actual = self.embedding(flow_state)
        torch.testing.assert_close(actual, expected)

    def test_dense_projection_uses_embedding_dtype(self):
        embedding = self.embedding.to(torch.bfloat16)
        flow_state = torch.randn(2, 5, 11, dtype=torch.float32)

        actual = embedding(flow_state)
        expected = torch.matmul(flow_state.to(torch.bfloat16), embedding.weight)

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected)

    def test_token_ids_index_the_same_projection(self):
        token_ids = torch.tensor([[0, 3, 10], [2, 4, 6]])
        actual = self.embedding(token_ids)
        expected = self.embedding.weight[token_ids]
        torch.testing.assert_close(actual, expected)

    def test_dense_projection_backpropagates(self):
        flow_state = torch.randn(2, 5, 11, requires_grad=True)
        self.embedding(flow_state).square().mean().backward()
        self.assertIsNotNone(flow_state.grad)
        self.assertIsNotNone(self.embedding.weight.grad)
        self.assertGreater(self.embedding.weight.grad.abs().sum().item(), 0)


class TestFlowSpecAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.config = _make_config(hidden_size=16, num_hidden_layers=1)
        self.attention = FlowSpecAttention(self.config).eval()
        self.draft_hidden = torch.randn(1, 2, 16)
        self.context_hidden = torch.randn(1, 3, 16)
        self.draft_positions = torch.tensor([[3, 4]])
        self.context_positions = torch.tensor([[0, 1, 2]])

    def _forward(self, context_hidden: torch.Tensor) -> torch.Tensor:
        dense_mask = torch.zeros(1, 1, 2, 5, dtype=torch.bool)
        dense_mask[..., 0] = True
        return self.attention(
            draft_hidden=self.draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=self.draft_positions,
            context_position_ids=self.context_positions,
            block_mask=dense_mask,
        )

    def test_dense_mask_excludes_context_keys(self):
        baseline = self._forward(self.context_hidden)

        masked_context_changed = self.context_hidden.clone()
        masked_context_changed[:, 1:] += 100
        torch.testing.assert_close(self._forward(masked_context_changed), baseline)

        visible_context_changed = self.context_hidden.clone()
        visible_context_changed[:, 0] += 100
        self.assertFalse(torch.allclose(self._forward(visible_context_changed), baseline))


class TestFlowSpecBlockMask(unittest.TestCase):
    def setUp(self):
        self.anchors = torch.tensor([[2, 5]])
        self.keep = torch.tensor([[True, True]])
        self.context_len = 8
        self.block_size = 3

    def test_prefix_and_block_visibility(self):
        mask = _create_flowspec_dense_mask(
            self.anchors,
            self.keep,
            self.context_len,
            self.block_size,
        )[0, 0]

        expected_first_block = torch.zeros(14, dtype=torch.bool)
        expected_first_block[:2] = True
        expected_first_block[8:11] = True
        torch.testing.assert_close(mask[0], expected_first_block)
        torch.testing.assert_close(mask[2], expected_first_block)

        expected_second_block = torch.zeros(14, dtype=torch.bool)
        expected_second_block[:5] = True
        expected_second_block[11:14] = True
        torch.testing.assert_close(mask[3], expected_second_block)
        torch.testing.assert_close(mask[5], expected_second_block)

    def test_invalid_block_sees_nothing(self):
        keep = torch.tensor([[True, False]])
        mask_mod = _create_flowspec_mask_mod(
            self.anchors,
            keep,
            self.context_len,
            self.block_size,
        )
        for query in range(3, 6):
            for key in range(14):
                self.assertFalse(bool(mask_mod(0, 0, query, key)))


class TestFlowSpecInitialization(unittest.TestCase):
    def test_decoder_is_identity_at_initialization(self):
        config = _make_config(hidden_size=16, num_hidden_layers=1)
        layer = FlowSpecDecoderLayer(config).eval()
        draft_hidden = torch.randn(1, 3, 16)
        context_hidden = torch.randn(1, 4, 16)
        time_condition = torch.randn(1, 3, 16)
        draft_positions = torch.tensor([[4, 5, 6]])
        context_positions = torch.tensor([[0, 1, 2, 3]])
        dense_mask = torch.ones(1, 1, 3, 7, dtype=torch.bool)

        actual = layer(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            time_condition=time_condition,
            draft_position_ids=draft_positions,
            context_position_ids=context_positions,
            block_mask=dense_mask,
        )
        torch.testing.assert_close(actual, draft_hidden, rtol=0, atol=0)

    def test_final_layer_starts_with_uniform_logits(self):
        config = _make_config(hidden_size=16, vocab_size=31, num_hidden_layers=1)
        layer = FlowSpecFinalLayer(config)
        logits = layer(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        torch.testing.assert_close(logits, torch.zeros(2, 4, 31))


class TestFlowSpecDraftModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.config = _make_config()
        self.model = FlowSpecDraftModel(self.config).eval()
        self.batch_size = 2
        self.draft_len = 6
        self.context_len = 8
        self.hidden_states = [
            torch.randn(self.batch_size, self.context_len, self.config.target_hidden_size)
            for _ in range(self.config.num_target_layers)
        ]
        self.context_feature = self.model.extract_context_feature(self.hidden_states)
        self.flow_state = torch.randn(
            self.batch_size,
            self.draft_len,
            self.config.vocab_size,
        )
        self.draft_positions = torch.arange(self.draft_len).expand(self.batch_size, -1)
        self.context_positions = torch.arange(self.context_len).expand(self.batch_size, -1)
        self.dense_mask = torch.ones(
            self.batch_size,
            1,
            self.draft_len,
            self.context_len + self.draft_len,
            dtype=torch.bool,
        )

    def test_context_projection_shape(self):
        self.assertEqual(
            self.context_feature.shape,
            (self.batch_size, self.context_len, self.config.hidden_size),
        )

    def test_time_broadcasting(self):
        scalar = self.model._prepare_time_condition(
            torch.tensor(0.5), self.batch_size, self.draft_len
        )
        per_batch = self.model._prepare_time_condition(
            torch.tensor([0.25, 0.75]), self.batch_size, self.draft_len
        )
        explicit = self.model._prepare_time_condition(
            torch.rand(self.batch_size, self.draft_len),
            self.batch_size,
            self.draft_len,
        )

        expected_shape = (self.batch_size, self.draft_len, self.config.time_condition_size)
        self.assertEqual(scalar.shape, expected_shape)
        self.assertEqual(per_batch.shape, expected_shape)
        self.assertEqual(explicit.shape, expected_shape)
        torch.testing.assert_close(per_batch[:, :1].expand_as(per_batch), per_batch)

    def test_forward_and_first_step_gradient(self):
        logits = self.model(
            flow_state=self.flow_state,
            timesteps=torch.rand(self.batch_size, self.draft_len),
            context_feature=self.context_feature,
            draft_position_ids=self.draft_positions,
            context_position_ids=self.context_positions,
            block_mask=self.dense_mask,
        )
        self.assertEqual(
            logits.shape,
            (self.batch_size, self.draft_len, self.config.vocab_size),
        )
        torch.testing.assert_close(logits, torch.zeros_like(logits))

        targets = torch.randint(
            0,
            self.config.vocab_size,
            (self.batch_size, self.draft_len),
        )
        F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
        head_gradient = self.model.output_layer.linear.weight.grad
        self.assertIsNotNone(head_gradient)
        self.assertGreater(head_gradient.abs().sum().item(), 0)

    def test_rejects_invalid_timestep_shape(self):
        with self.assertRaisesRegex(ValueError, "timesteps must be scalar"):
            self.model._prepare_time_condition(
                torch.rand(self.batch_size, 2, 3),
                self.batch_size,
                self.draft_len,
            )

    def test_self_conditioning_projection_matches_elf_parameterization(self):
        config = _make_config(self_conditioning=True)
        model = FlowSpecDraftModel(config)
        hidden_size = config.hidden_size

        self.assertEqual(
            model.self_condition_proj.weight.shape,
            (hidden_size, 2 * hidden_size),
        )
        self.assertIsNotNone(model.self_condition_proj.bias)
        torch.testing.assert_close(
            model.self_condition_proj.bias,
            torch.zeros(hidden_size),
        )
        self.assertGreater(
            model.self_condition_proj.weight[:, :hidden_size].abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            model.self_condition_proj.weight[:, hidden_size:].abs().sum().item(),
            0.0,
        )

        logits = model(
            flow_state=self.flow_state,
            self_condition_state=torch.rand_like(self.flow_state),
            timesteps=torch.rand(self.batch_size, self.draft_len),
            context_feature=model.extract_context_feature(self.hidden_states),
            draft_position_ids=self.draft_positions,
            context_position_ids=self.context_positions,
            block_mask=self.dense_mask,
        )
        self.assertEqual(logits.shape, self.flow_state.shape)


class TestFlowSpecWrapper(unittest.TestCase):
    def test_soft_target_logit_normal_mixture_matches_fitted_quantiles(self):
        schedule = LogitNormalMixtureTimeSchedule()
        tau = torch.linspace(0.0, 1.0, 1_001)
        physical_time = schedule(tau)

        self.assertTrue(bool((physical_time[1:] >= physical_time[:-1]).all()))
        torch.testing.assert_close(physical_time[[0, -1]], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(
            schedule(torch.tensor([0.1, 0.5, 0.9, 0.95, 0.99])),
            torch.tensor([0.7699685, 0.8297437, 0.8868507, 0.9104648, 0.9457566]),
            atol=2e-6,
            rtol=0.0,
        )

    def test_soft_target_argmax_uniform_blend_uses_ten_percent_uniform_time(self):
        model = _make_wrapper(time_schedule="soft_target_argmax_uniform_blend")
        tau = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
        base = LogitNormalMixtureTimeSchedule()(tau)

        torch.testing.assert_close(model.tau_to_time(tau), 0.9 * base + 0.1 * tau)

    def test_recognition_schedule_is_monotone_and_matches_landmarks(self):
        schedule = RecognitionTimeSchedule()
        tau = torch.linspace(0.0, 1.0, 1_001)
        physical_time = schedule(tau)

        self.assertTrue(bool((physical_time[1:] >= physical_time[:-1]).all()))
        torch.testing.assert_close(physical_time[[0, -1]], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(
            schedule(torch.tensor([0.1, 0.5, 0.9])),
            torch.tensor([0.3999181, 0.5, 0.6000819]),
            atol=2e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            physical_time + physical_time.flip(0),
            torch.ones_like(physical_time),
            atol=2e-5,
            rtol=0.0,
        )

    def test_default_block_size_is_eight(self):
        draft_model = FlowSpecDraftModel(_make_config())
        model = FlowSpecModel(
            draft_model,
            num_anchors=2,
            use_flex_attention=False,
        )
        self.assertEqual(model.block_size, 8)
        self.assertEqual(model.num_predictions, 7)
        self.assertEqual(FlowSpecTrainer._anchor_slot_offset, 1)

    def test_all_campaign_time_schedules_are_monotone(self):
        for schedule_name in (
            "onehot_lut",
            "simplex_sharp",
            "simplex_visibility",
            "affine_balanced",
            "soft_target_logit_normal_mixture",
            "soft_target_argmax_uniform_blend",
            "identity",
        ):
            with self.subTest(schedule=schedule_name):
                model = _make_wrapper(time_schedule=schedule_name)
                tau = torch.linspace(0.0, 1.0, 1_001)
                physical_time = model.tau_to_time(tau)
                self.assertTrue(bool((physical_time[1:] >= physical_time[:-1]).all()))
                torch.testing.assert_close(
                    physical_time[[0, -1]],
                    torch.tensor([0.0, 1.0]),
                )

    def test_campaign_schedule_shapes_are_distinct(self):
        signatures = {
            name: tuple(
                round(value, 4)
                for value in _make_wrapper(time_schedule=name)
                .tau_to_time(torch.tensor([0.25, 0.5, 0.75]))
                .tolist()
            )
            for name in (
                "onehot_lut",
                "simplex_sharp",
                "simplex_visibility",
                "affine_balanced",
                "soft_target_logit_normal_mixture",
                "soft_target_argmax_uniform_blend",
                "identity",
            )
        }
        self.assertEqual(len(set(signatures.values())), len(signatures))

    def test_block_requires_anchor_and_prediction(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            FlowSpecModel(
                FlowSpecDraftModel(_make_config()),
                block_size=1,
                use_flex_attention=False,
            )

    def test_rejects_invalid_loss_decay_gamma(self):
        for gamma in (-0.1, 0.0, float("inf"), float("nan")):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                FlowSpecModel(
                    FlowSpecDraftModel(_make_config()),
                    loss_decay_gamma=gamma,
                    use_flex_attention=False,
                )

    def test_rejects_invalid_boundary_probability(self):
        for probability in (-0.1, 1.0):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                FlowSpecModel(
                    FlowSpecDraftModel(_make_config()),
                    boundary_probability=probability,
                    use_flex_attention=False,
                )

    def test_rejects_invalid_clean_deadline_jitter(self):
        for jitter in (-0.1, 1.0, float("inf"), float("nan")):
            with self.subTest(jitter=jitter), self.assertRaises(ValueError):
                _make_wrapper(clean_deadline_jitter=jitter)

    def test_boundary_sampling_rescales_and_clamps_uniform_times(self):
        model = _make_wrapper(
            num_anchors=16,
            boundary_probability=0.25,
        )
        actual_generator = torch.Generator().manual_seed(17)
        expected_generator = torch.Generator().manual_seed(17)
        uniform = torch.rand(8, 16, generator=expected_generator)
        expected = ((uniform - 0.25) / 0.75).clamp_min(0.0)

        actual = model._sample_training_times(
            batch_size=8,
            device=torch.device("cpu"),
            generator=actual_generator,
        )

        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool((actual == 0.0).any()))

    def test_block_schedule_denoises_one_token_per_interval(self):
        model = _make_wrapper(block_size=8)
        tau = torch.tensor([[0.0, 1.0 / 14.0, 1.0 / 7.0, 3.0 / 14.0, 2.0 / 7.0, 1.0]])

        token_tau, physical_time = model._block_time_coordinates(tau)

        torch.testing.assert_close(token_tau[..., 0], torch.ones_like(tau))
        torch.testing.assert_close(token_tau[:, 0, 1:], torch.zeros(1, 7))
        torch.testing.assert_close(token_tau[:, 1, 1:], torch.tensor([[0.5, 0, 0, 0, 0, 0, 0]]))
        torch.testing.assert_close(
            token_tau[:, 2, 1:],
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(token_tau[:, 3, 1:], torch.tensor([[1, 0.5, 0, 0, 0, 0, 0]]))
        torch.testing.assert_close(
            token_tau[:, 4, 1:],
            torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(token_tau[:, -1:, 1:], torch.ones(1, 1, 7))
        torch.testing.assert_close(physical_time, model.tau_to_time(token_tau))

    def test_block_schedule_maps_token_tau_to_recognition_time(self):
        model = _make_wrapper(block_size=8)
        tau = torch.tensor([[3.0 / 14.0]])

        token_tau, physical_time = model._block_time_coordinates(tau)

        expected_tau = torch.tensor([[[1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        torch.testing.assert_close(token_tau, expected_tau)
        torch.testing.assert_close(physical_time, model.tau_to_time(expected_tau))

    def test_overlap_two_softens_adjacent_block_transitions(self):
        model = _make_wrapper(block_size=8, block_schedule_overlap=2.0)
        tau = torch.tensor([[0.2, 0.3]])

        token_tau, physical_time = model._block_time_coordinates(tau)

        torch.testing.assert_close(
            token_tau[..., 1:],
            torch.tensor(
                [
                    [
                        [0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertTrue(bool((token_tau[..., 1:-1] >= token_tau[..., 2:]).all()))
        torch.testing.assert_close(physical_time, model.tau_to_time(token_tau))

    def test_overlap_two_schedule_weights_use_overlapping_deadlines(self):
        model = _make_wrapper(block_size=8, block_schedule_overlap=2.0)
        active_probability = torch.arange(2, 9, dtype=torch.float32) / 8.0

        torch.testing.assert_close(
            model.schedule_loss_weights[1:] * active_probability,
            torch.ones(7),
        )

    def test_regular_schedule_shares_time_across_predictions(self):
        model = _make_wrapper(
            block_size=8,
            use_diffusion_forcing_schedule=False,
        )
        tau = torch.tensor([[0.0, 0.25, 1.0]])

        token_tau, physical_time = model._time_coordinates(tau, training=True)

        torch.testing.assert_close(token_tau[..., 0], torch.ones_like(tau))
        torch.testing.assert_close(
            token_tau[..., 1:],
            tau.unsqueeze(-1).expand(-1, -1, 7),
        )
        torch.testing.assert_close(physical_time, model.tau_to_time(token_tau))
        torch.testing.assert_close(model.schedule_loss_weights, torch.ones(8))

    def test_deadline_jitter_is_early_bounded_and_ordered(self):
        model = _make_wrapper(
            block_size=8,
            clean_deadline_jitter=0.1,
        )
        tau = torch.tensor([[0.1, 0.2, 0.3]])
        deadlines = model._sample_clean_deadlines(
            tau,
            generator=torch.Generator().manual_seed(7),
        )
        base = model.clean_deadlines.expand_as(deadlines)

        self.assertTrue(bool((deadlines <= base).all()))
        self.assertTrue(bool((base - deadlines <= 0.1 / 7.0).all()))
        self.assertTrue(bool((deadlines[..., 1:] > deadlines[..., :-1]).all()))
        self.assertFalse(torch.equal(deadlines, base))

    def test_deadline_jitter_changes_training_path_in_tau_space(self):
        model = _make_wrapper(
            block_size=8,
            clean_deadline_jitter=0.1,
        )
        tau = torch.tensor([[0.12, 0.24, 0.36]])
        deterministic_tau, _ = model._block_time_coordinates(tau)
        token_tau, physical_time = model._sample_training_time_coordinates(
            tau,
            generator=torch.Generator().manual_seed(11),
        )

        self.assertTrue(bool((token_tau[..., 1:-1] >= token_tau[..., 2:]).all()))
        self.assertFalse(torch.equal(token_tau, deterministic_tau))
        torch.testing.assert_close(physical_time, model.tau_to_time(token_tau))

    def test_schedule_loss_weights_cancel_expected_active_frequency(self):
        model = _make_wrapper(block_size=8)
        active_probability = torch.arange(1, 8, dtype=torch.float32) / 7.0

        torch.testing.assert_close(
            model.schedule_loss_weights[1:] * active_probability,
            torch.ones(7),
        )

    def test_schedule_loss_weights_include_boundary_and_jitter(self):
        boundary_probability = 1.0 / 32.0
        jitter = 0.1
        model = _make_wrapper(
            block_size=8,
            boundary_probability=boundary_probability,
            clean_deadline_jitter=jitter,
        )
        expected_deadlines = (torch.arange(1, 8, dtype=torch.float32) - jitter / 2.0) / 7.0
        active_probability = (
            boundary_probability + (1.0 - boundary_probability) * expected_deadlines
        )

        torch.testing.assert_close(
            model.schedule_loss_weights[1:] * active_probability,
            torch.ones(7),
        )

    def test_anchor_requires_complete_supervised_aligned_block(self):
        model = _make_wrapper(block_size=3, num_anchors=4)
        loss_mask = torch.zeros(1, 12)
        loss_mask[:, 2:5] = 1
        loss_mask[:, 6:9] = 1

        anchors, keep = model._sample_anchor_positions(loss_mask)
        torch.testing.assert_close(anchors, torch.tensor([[2, 6, 0, 0]]))
        torch.testing.assert_close(keep, torch.tensor([[True, True, False, False]]))

    def test_seeded_anchor_sampling_is_repeatable(self):
        model = _make_wrapper(block_size=3, num_anchors=4)
        loss_mask = torch.ones(1, 16)
        first_generator = torch.Generator().manual_seed(123)
        second_generator = torch.Generator().manual_seed(123)

        first = model._sample_anchor_positions(loss_mask, generator=first_generator)
        second = model._sample_anchor_positions(loss_mask, generator=second_generator)

        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])

    def test_positions_start_at_anchor(self):
        model = _make_wrapper(block_size=3, num_anchors=2)
        context_positions, draft_positions = model._create_position_ids(
            torch.tensor([[1, 5]]),
            context_len=12,
        )
        torch.testing.assert_close(context_positions, torch.arange(12).unsqueeze(0))
        torch.testing.assert_close(draft_positions, torch.tensor([[1, 2, 3, 5, 6, 7]]))

    def test_soft_distribution_corruption_endpoints(self):
        model = _make_wrapper(block_size=3, num_anchors=2)
        target = torch.rand(1, 2, 3, model.draft_model.config.vocab_size)
        target = target / target.sum(dim=-1, keepdim=True)
        noise = torch.randn_like(target)
        corrupted = model._corrupt_distribution(
            target,
            physical_time=torch.tensor([[0.0, 1.0]]),
            noise=noise,
        )
        torch.testing.assert_close(corrupted[:, 0], noise[:, 0])
        torch.testing.assert_close(corrupted[:, 1], target[:, 1])

    def test_default_noise_is_a_probability_distribution(self):
        model = _make_wrapper(block_size=3, num_anchors=2)
        target = torch.rand(1, 2, 3, model.draft_model.config.vocab_size)
        target /= target.sum(dim=-1, keepdim=True)

        corrupted = model._corrupt_distribution(
            target,
            physical_time=torch.zeros(1, 2),
        )

        self.assertTrue(bool((corrupted >= 0.0).all()))
        torch.testing.assert_close(
            corrupted.sum(dim=-1),
            torch.ones_like(corrupted[..., 0]),
        )

    def test_affine_gaussian_matches_row_sum_and_second_moment(self):
        model = _make_wrapper(noise_type="affine_gaussian")
        noise = model._sample_noise(
            torch.Size((8_192, model.draft_model.config.vocab_size)),
            torch.device("cpu"),
            generator=torch.Generator().manual_seed(9),
        )

        torch.testing.assert_close(
            noise.sum(dim=-1),
            torch.ones(noise.shape[0]),
            atol=2e-6,
            rtol=0.0,
        )
        self.assertAlmostEqual(noise.square().sum(dim=-1).mean().item(), 0.8492, places=2)
        self.assertTrue(bool((noise < 0.0).any()))

    def test_standard_gaussian_noise_retains_original_recipe(self):
        model = _make_wrapper(noise_type="standard_gaussian")
        noise = model._sample_noise(
            torch.Size((4_096, 64)),
            torch.device("cpu"),
            generator=torch.Generator().manual_seed(3),
        )
        self.assertAlmostEqual(noise.mean().item(), 0.0, places=2)
        self.assertAlmostEqual(noise.std().item(), 1.0, places=2)

    def test_self_conditioning_uses_detached_model_prediction(self):
        model = _make_wrapper(
            block_size=3,
            num_anchors=1,
            self_condition_probability=1.0,
        )
        calls = []

        def record_forward(**kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 3, model.draft_model.config.vocab_size)

        model.draft_model.forward = record_forward
        model._sample_training_times = lambda *args, **kwargs: torch.zeros(1, 1)
        model(
            input_ids=torch.arange(1, 9).unsqueeze(0),
            hidden_states_list=[
                torch.randn(1, 8, model.draft_model.config.target_hidden_size)
                for _ in range(model.draft_model.config.num_target_layers)
            ],
            loss_mask=torch.ones(1, 8),
            **_target_inputs(model, 1, 8),
        )

        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[0]["self_condition_state"])
        self.assertIsNotNone(calls[1]["self_condition_state"])
        self.assertEqual(calls[0]["self_condition_state"].shape, (1, 3, 64))
        self.assertEqual(calls[1]["self_condition_state"].shape, (1, 3, 64))
        zero_condition = calls[0]["self_condition_state"].view(1, 1, 3, -1)
        learned_condition = calls[1]["self_condition_state"].view(1, 1, 3, -1)
        torch.testing.assert_close(
            zero_condition[..., 1:, :], torch.zeros_like(zero_condition[..., 1:, :])
        )
        torch.testing.assert_close(zero_condition[..., :1, :], learned_condition[..., :1, :])
        self.assertFalse(calls[1]["self_condition_state"].requires_grad)

    def test_ode_carries_previous_clean_prediction_as_self_condition(self):
        model = _make_wrapper(
            block_size=3,
            num_anchors=1,
            self_condition_probability=0.5,
        )
        calls = []

        def record_forward(**kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 3, model.draft_model.config.vocab_size)

        model.draft_model.forward = record_forward
        hidden_states = [
            torch.randn(1, 6, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        model.integrate(
            hidden_states_list=hidden_states,
            anchor_positions=torch.tensor([[2]]),
            anchor_token_ids=torch.tensor([[7]]),
            num_steps=2,
            noise=torch.zeros(1, 1, 3, model.draft_model.config.vocab_size),
        )

        self.assertEqual(len(calls), 2)
        first_condition = calls[0]["self_condition_state"].view(1, 1, 3, -1)
        second_condition = calls[1]["self_condition_state"].view(1, 1, 3, -1)
        expected_anchor = F.one_hot(
            torch.tensor(7),
            model.draft_model.config.vocab_size,
        ).float()
        torch.testing.assert_close(first_condition[0, 0, 0], expected_anchor)
        torch.testing.assert_close(
            first_condition[0, 0, 1:], torch.zeros_like(first_condition[0, 0, 1:])
        )
        torch.testing.assert_close(second_condition[0, 0, 0], expected_anchor)
        torch.testing.assert_close(
            second_condition[0, 0, 1:],
            torch.full_like(
                second_condition[0, 0, 1:],
                1.0 / model.draft_model.config.vocab_size,
            ),
        )

    def test_forward_masks_anchor_and_scores_seven_predictions(self):
        model = _make_wrapper(block_size=8, num_anchors=2)
        batch_size, sequence_len = 1, 20
        input_ids = torch.arange(1, sequence_len + 1).unsqueeze(0)
        loss_mask = torch.ones(batch_size, sequence_len)
        hidden_states = [
            torch.randn(batch_size, sequence_len, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        model._sample_training_times = lambda *args, **kwargs: torch.zeros(1, 2)

        loss, accuracy, loss_per_position, _, count_per_position, components = model(
            input_ids=input_ids,
            hidden_states_list=hidden_states,
            loss_mask=loss_mask,
            **_target_inputs(model, batch_size, sequence_len),
        )
        uniform_ce = math.log(model.draft_model.config.vocab_size)
        self.assertAlmostEqual(loss.item(), uniform_ce, places=5)
        self.assertEqual(accuracy.item(), 0.0)
        self.assertEqual(loss_per_position.shape, (8,))
        torch.testing.assert_close(
            count_per_position,
            torch.tensor([0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        )
        torch.testing.assert_close(
            model.loss_decay_weights,
            torch.exp(-(torch.arange(8, dtype=torch.float32) - 1.0).clamp_min(0.0) / 7.0),
        )
        torch.testing.assert_close(model.loss_decay_weights[:2], torch.ones(2))

    def test_forward_supplies_clean_anchor_at_time_one(self):
        model = _make_wrapper(block_size=3, num_anchors=1)
        input_ids = torch.tensor([[3, 4, 7, 8, 9, 10]])
        loss_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        hidden_states = [
            torch.randn(1, 6, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        captured = {}

        def record_forward(**kwargs):
            captured.update(kwargs)
            return torch.zeros(
                1,
                3,
                model.draft_model.config.vocab_size,
            )

        model.draft_model.forward = record_forward
        model._sample_training_times = lambda *args, **kwargs: torch.zeros(1, 1)
        _, _, _, _, count_per_position, _ = model(
            input_ids=input_ids,
            hidden_states_list=hidden_states,
            loss_mask=loss_mask,
            **_target_inputs(model, 1, 6),
        )

        flow_state = captured["flow_state"].view(
            1,
            1,
            3,
            model.draft_model.config.vocab_size,
        )
        expected_anchor = F.one_hot(
            torch.tensor(7),
            model.draft_model.config.vocab_size,
        ).float()
        torch.testing.assert_close(flow_state[0, 0, 0], expected_anchor)
        self.assertEqual(captured["timesteps"][0, 0].item(), 1.0)
        torch.testing.assert_close(count_per_position, torch.tensor([0.0, 1.0, 1.0]))

    def test_forward_does_not_score_tokens_that_are_already_clean(self):
        model = _make_wrapper(block_size=8, num_anchors=1)
        input_ids = torch.arange(1, 13).unsqueeze(0)
        hidden_states = [
            torch.randn(1, 12, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        model._sample_training_times = lambda *args, **kwargs: torch.tensor([[0.2]])

        _, _, _, _, count_per_position, _ = model(
            input_ids=input_ids,
            hidden_states_list=hidden_states,
            loss_mask=torch.ones_like(input_ids, dtype=torch.float32),
            **_target_inputs(model, 1, 12),
        )

        torch.testing.assert_close(
            count_per_position,
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        )

    def test_forward_uses_jittered_clean_deadlines(self):
        model = _make_wrapper(
            block_size=8,
            num_anchors=2,
            clean_deadline_jitter=0.1,
        )
        input_ids = torch.arange(1, 21).unsqueeze(0)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        hidden_states = [
            torch.randn(1, 20, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        captured = {}

        def record_forward(**kwargs):
            captured.update(kwargs)
            return torch.zeros(1, 16, model.draft_model.config.vocab_size)

        model.draft_model.forward = record_forward
        fixed_tau = torch.tensor([[0.12, 0.24]])
        model._sample_training_times = lambda *args, **kwargs: fixed_tau
        torch.manual_seed(19)
        model(
            input_ids=input_ids,
            hidden_states_list=hidden_states,
            loss_mask=loss_mask,
            **_target_inputs(model, 1, 20),
        )

        token_tau = captured["timesteps"].view(1, 2, 8)
        deterministic_tau, _ = model._block_time_coordinates(fixed_tau)
        self.assertTrue(bool((token_tau[..., 1:-1] >= token_tau[..., 2:]).all()))
        self.assertFalse(torch.equal(token_tau, deterministic_tau))

    def test_eval_forward_uses_deterministic_block_path(self):
        model = _make_wrapper(
            block_size=8,
            num_anchors=2,
            clean_deadline_jitter=0.1,
        ).eval()
        input_ids = torch.arange(1, 21).unsqueeze(0)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        hidden_states = [
            torch.randn(1, 20, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        captured = {}

        def record_forward(**kwargs):
            captured.update(kwargs)
            return torch.zeros(1, 16, model.draft_model.config.vocab_size)

        model.draft_model.forward = record_forward
        fixed_tau = torch.tensor([[0.12, 0.24]])
        model._sample_training_times = lambda *args, **kwargs: fixed_tau
        model(
            input_ids=input_ids,
            hidden_states_list=hidden_states,
            loss_mask=loss_mask,
            **_target_inputs(model, 1, 20),
        )

        token_tau = captured["timesteps"].view(1, 2, 8)
        deterministic_tau, _ = model._block_time_coordinates(fixed_tau)
        torch.testing.assert_close(token_tau, deterministic_tau)

    def test_forward_can_train_on_qwen_distribution(self):
        model = _make_wrapper(block_size=3, num_anchors=2)
        batch_size, sequence_len = 1, 8
        hidden_size = model.draft_model.config.target_hidden_size
        hidden_states = [
            torch.randn(batch_size, sequence_len, hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        model._sample_training_times = lambda *args, **kwargs: torch.zeros(1, 2)

        loss, accuracy, *_, components = model(
            input_ids=torch.arange(1, sequence_len + 1).unsqueeze(0),
            hidden_states_list=hidden_states,
            loss_mask=torch.ones(batch_size, sequence_len),
            last_hidden_states=torch.randn(batch_size, sequence_len, hidden_size),
            lm_head_weight=torch.zeros(model.draft_model.config.vocab_size, hidden_size),
        )

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(accuracy.item(), 0.0)
        self.assertEqual(components["block_match_length"].item(), 0.0)

    def test_forward_requires_target_hidden_states(self):
        model = _make_wrapper(block_size=3, num_anchors=1)
        hidden_states = [
            torch.randn(1, 6, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        with self.assertRaisesRegex(ValueError, "requires target LM logits"):
            model(
                input_ids=torch.arange(6).unsqueeze(0),
                hidden_states_list=hidden_states,
                loss_mask=torch.ones(1, 6),
            )

    def test_target_logits_use_causally_aligned_hidden_states(self):
        model = _make_wrapper(block_size=3, num_anchors=1)
        hidden_size = model.draft_model.config.target_hidden_size
        vocab_size = model.draft_model.config.vocab_size
        last_hidden_states = torch.zeros(1, 8, hidden_size)
        for position in range(8):
            last_hidden_states[0, position, position] = 1.0
        lm_head_weight = torch.zeros(vocab_size, hidden_size)
        lm_head_weight[:hidden_size] = 10.0 * torch.eye(hidden_size)

        predictions = model._target_logits(
            last_hidden_states,
            lm_head_weight,
            anchor_positions=torch.tensor([[2]]),
        ).argmax(dim=-1)

        torch.testing.assert_close(predictions, torch.tensor([[[2, 3]]]))

    def test_integration_returns_clean_probabilities(self):
        model = _make_wrapper(block_size=3, num_anchors=1)
        batch_size, context_len = 1, 6
        hidden_states = [
            torch.randn(batch_size, context_len, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        noise = torch.randn(
            batch_size,
            1,
            3,
            model.draft_model.config.vocab_size,
        )
        probabilities = model.integrate(
            hidden_states_list=hidden_states,
            anchor_positions=torch.tensor([[2]]),
            anchor_token_ids=torch.tensor([[7]]),
            num_steps=2,
            noise=noise,
        )
        expected_predictions = torch.full_like(
            probabilities[:, :, 1:],
            1.0 / model.draft_model.config.vocab_size,
        )
        torch.testing.assert_close(probabilities[:, :, 1:], expected_predictions)
        torch.testing.assert_close(
            probabilities[:, :, 0],
            F.one_hot(
                torch.tensor([[7]]),
                model.draft_model.config.vocab_size,
            ).float(),
        )

    def test_integration_freezes_each_token_after_its_clean_deadline(self):
        model = _make_wrapper(block_size=3, num_anchors=1)
        vocab_size = model.draft_model.config.vocab_size
        call_targets = ((1, 2), (3, 4), (5, 6))
        call_index = 0

        def changing_predictions(**kwargs):
            nonlocal call_index
            logits = torch.full((1, 3, vocab_size), -20.0)
            first, second = call_targets[call_index]
            logits[0, 1, first] = 20.0
            logits[0, 2, second] = 20.0
            call_index += 1
            return logits

        model.draft_model.forward = changing_predictions
        hidden_states = [
            torch.randn(1, 6, model.draft_model.config.target_hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]

        predictions = model.generate(
            hidden_states_list=hidden_states,
            anchor_positions=torch.tensor([[2]]),
            anchor_token_ids=torch.tensor([[7]]),
            num_steps=3,
            generator=torch.Generator().manual_seed(7),
        )

        torch.testing.assert_close(predictions, torch.tensor([[[7, 3, 6]]]))

    def test_integrated_eval_reports_exact_block_matches(self):
        model = _make_wrapper(block_size=3, num_anchors=2)
        batch_size, context_len = 1, 8
        hidden_size = model.draft_model.config.target_hidden_size
        hidden_states = [
            torch.randn(batch_size, context_len, hidden_size)
            for _ in range(model.draft_model.config.num_target_layers)
        ]
        totals = model.evaluate_integrated(
            input_ids=torch.zeros(batch_size, context_len, dtype=torch.long),
            hidden_states_list=hidden_states,
            loss_mask=torch.ones(batch_size, context_len),
            num_steps=2,
            generator=torch.Generator().manual_seed(7),
        )

        torch.testing.assert_close(
            totals["token_match_per_position"],
            torch.tensor([0.0, 2.0, 2.0]),
        )
        torch.testing.assert_close(
            totals["count_per_position"],
            torch.tensor([0.0, 2.0, 2.0]),
        )
        torch.testing.assert_close(totals["block_match_length_sum"], torch.tensor(4.0))
        torch.testing.assert_close(
            totals["prefix_match_per_position"],
            torch.tensor([0.0, 2.0, 2.0]),
        )
        torch.testing.assert_close(totals["block_count"], torch.tensor(2.0))
        self.assertEqual(
            set(totals),
            {
                "token_match_per_position",
                "count_per_position",
                "prefix_match_per_position",
                "block_match_length_sum",
                "block_count",
                "row_count",
            },
        )


class TestFlowSpecOdeEvalPartition(unittest.TestCase):
    def test_one_hundred_rows_are_split_exactly_across_six_ranks(self):
        limits = [_rank_sample_limit(100, rank, 6) for rank in range(6)]
        self.assertEqual(limits, [17, 17, 17, 17, 16, 16])
        self.assertEqual(sum(limits), 100)


if __name__ == "__main__":
    unittest.main()
