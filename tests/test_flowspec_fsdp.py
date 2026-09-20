"""Two-GPU regression for FlowSpec ODE evaluation after FSDP checkpointing."""

from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh


def _checkpoint_eval_worker(rank, rendezvous, checkpoint_dir, ablation):
    from torchspec.models.draft.flowspec import FlowSpecConfig
    from torchspec.training import checkpoint
    from torchspec.training.flowspec_trainer import FlowSpecTrainer
    from torchspec.training.trainer import Trainer

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        mesh = init_device_mesh("cuda", (2,))
        args = Namespace(
            flowspec_block_size=8,
            flowspec_num_anchors=4,
            flowspec_anchor_max_gap=0,
            flowspec_num_target_layers=2,
            flowspec_self_condition_probability=0.5,
            flowspec_noise_type="standard_gaussian",
            flowspec_time_schedule="soft_target_argmax_uniform_blend",
            attention_backend="sdpa",
            fsdp_strategy="FULL_SHARD",
            fsdp_reduce_dtype="bfloat16",
            learning_rate=1e-4,
            lr_total_steps=8,
            warmup_ratio=0.0,
            max_grad_norm=0.5,
            checkpoint_dir=checkpoint_dir,
        )

        for key, value in ablation.items():
            setattr(args, key, value)

        def base_init(self, args):
            self.args = args
            self.dp_mesh = mesh
            self.dp_rank = rank
            self.fsdp_cpu_offload = False
            self.global_step = 0
            self.prof = SimpleNamespace(on_init_end=lambda: None)

        def target_head(self, _path):
            self.target_lm_head = SimpleNamespace(
                lm_head=torch.nn.Linear(32, 64, bias=False, device="cuda", dtype=torch.bfloat16)
            )

        config = FlowSpecConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=128,
            time_condition_size=32,
            time_frequency_size=16,
            num_target_layers=2,
            target_hidden_size=32,
            target_num_hidden_layers=12,
            self_conditioning=True,
        )
        torch.manual_seed(42)

        def create_trainer():
            with (
                mock.patch.object(Trainer, "__init__", base_init),
                mock.patch.object(FlowSpecTrainer, "_init_target_lm_head", target_head),
                mock.patch(
                    "torchspec.training.flowspec_trainer.get_gloo_group",
                    return_value=dist.group.WORLD,
                ),
            ):
                result = FlowSpecTrainer(args)
                result.init_model(config, str(Path(checkpoint_dir).parent / "target"))
                return result

        trainer = create_trainer()
        assert trainer.flowspec.endpoint_target == getattr(
            args, "flowspec_endpoint_target", "teacher_distribution"
        )

        def check_frozen_target(result):
            if not getattr(args, "flowspec_pretrained_embeddings_and_head", False):
                return
            expected = torch.arange(64 * 32).reshape(64, 32).float().div(2048).bfloat16().cuda()
            for parameter, reference in (
                (result.draft_model.vocab_embed.weight, expected),
                (result.draft_model.output_layer.linear.weight, expected.flip(0)),
            ):
                assert not parameter.requires_grad
                assert parameter.grad is None
                actual = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert all(parameter is not p for p in result.optimizer.model_params)
            assert result.draft_model.output_layer.linear.bias is None

        check_frozen_target(trainer)

        inputs = dict(
            input_ids=torch.randint(0, 64, (1, 16), device="cuda"),
            hidden_states_list=[
                torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2)
            ],
            loss_mask=torch.ones(1, 16, device="cuda"),
            last_hidden_states=torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16),
            lm_head_weight=trainer.target_lm_head_weight,
        )
        trainer.model.train()
        loss = trainer.model(**inputs)[0]
        loss.backward()
        grad_norm = trainer.optimizer.step()
        assert torch.isfinite(grad_norm).item()
        check_frozen_target(trainer)
        trainer.global_step = 1
        trainer.model.eval()
        with torch.no_grad():
            assert torch.isfinite(trainer.model(**inputs)[0])
        checkpoint.save(trainer, 1)
        eval_inputs = {k: inputs[k] for k in ("input_ids", "hidden_states_list", "loss_mask")}
        if trainer.flowspec.target_distribution_anchor:
            eval_inputs.update({k: inputs[k] for k in ("last_hidden_states", "lm_head_weight")})
        for num_steps in (1, 2, 4, 32):
            metrics = trainer.flowspec.evaluate_integrated(**eval_inputs, num_steps=num_steps)
            assert all(torch.isfinite(value).all() for value in metrics.values())
            assert metrics["row_count"].item() == 1

        args.load_path = checkpoint_dir
        restored = create_trainer()
        assert restored.global_step == 1
        check_frozen_target(restored)
        assert len(restored.optimizer.state) == len(trainer.optimizer.state) > 0
        for original_param, loaded_param in zip(
            trainer.optimizer.fp32_params, restored.optimizer.fp32_params, strict=True
        ):
            original_state = trainer.optimizer.state[original_param]
            loaded_state = restored.optimizer.state[loaded_param]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                original, loaded = original_state[key], loaded_state[key]
                if hasattr(original, "to_local"):
                    original, loaded = original.to_local(), loaded.to_local()
                torch.testing.assert_close(loaded, original, rtol=0, atol=0)
        metrics = restored.flowspec.evaluate_integrated(**eval_inputs, num_steps=1)
        assert torch.isfinite(metrics["block_match_length_sum"])
        restored.model.train()
        restored.model(**inputs)[0].backward()
        assert torch.isfinite(restored.optimizer.step()).item()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize(
    "ablation",
    [
        {
            "flowspec_pretrained_embeddings_and_head": True,
            "flowspec_use_diffusion_forcing_schedule": False,
            "flowspec_boundary_probability": 1 / 32,
            "flowspec_xm_best_of_k": 4,
        },
        {},
        {"flowspec_use_diffusion_forcing_schedule": False},
        {"flowspec_causal_attention": True},
        {"flowspec_self_condition_probability": 0.0},
        {"flowspec_scratch_tokens": 8, "flowspec_block_size": 9},
        {"flowspec_pretrained_embeddings_and_head": True},
        {
            "flowspec_pretrained_embeddings_and_head": True,
            "flowspec_self_condition_probability": 0.0,
        },
        {
            "flowspec_pretrained_embeddings_and_head": True,
            "flowspec_scratch_tokens": 8,
            "flowspec_block_size": 9,
        },
        {
            "flowspec_pretrained_embeddings_and_head": True,
            "flowspec_use_diffusion_forcing_schedule": False,
            "flowspec_loss_type": "velocity_mse",
            "flowspec_integral_time_weighting": True,
            "flowspec_target_distribution_anchor": True,
            "flowspec_boundary_probability": 1 / 32,
        },
        {
            "flowspec_pretrained_embeddings_and_head": True,
            "flowspec_use_diffusion_forcing_schedule": False,
            "flowspec_loss_type": "cross_entropy",
            "flowspec_integral_time_weighting": False,
            "flowspec_target_distribution_anchor": True,
            "flowspec_boundary_probability": 1 / 32,
        },
        {
            "flowspec_pretrained_embeddings_and_head": True,
            "flowspec_use_diffusion_forcing_schedule": False,
            "flowspec_loss_type": "cross_entropy",
            "flowspec_endpoint_target": "data_token",
            "flowspec_integral_time_weighting": False,
            "flowspec_target_distribution_anchor": False,
            "flowspec_boundary_probability": 1 / 32,
        },
    ],
    ids=[
        "xm_best_of_4",
        "baseline",
        "no_diffusion_forcing",
        "causal",
        "no_self_conditioning",
        "scratch",
        "pretrained",
        "pretrained_no_self_conditioning",
        "pretrained_scratch",
        "integral_mse_distribution_anchor",
        "ce_distribution_anchor",
        "ce_data_token_endpoint",
    ],
)
def test_flowspec_ode_eval_after_checkpoint_and_reload(tmp_path, ablation):
    if ablation.get("flowspec_pretrained_embeddings_and_head"):
        from safetensors.torch import save_file

        target = tmp_path / "target"
        target.mkdir()
        embedding = torch.arange(64 * 32).reshape(64, 32).float().div(2048).bfloat16()
        save_file(
            {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding.flip(0)},
            target / "model.safetensors",
        )
    mp.spawn(
        _checkpoint_eval_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "checkpoints"), ablation),
        nprocs=2,
        join=True,
    )
