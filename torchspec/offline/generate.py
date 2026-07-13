"""Materialize live target-model outputs for offline training."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.controller.inference_manager import AsyncInferenceManager
from torchspec.controller.loop import _safe_training_cleanup
from torchspec.controller.setup import build_mooncake_config
from torchspec.controller.training_controller import AsyncTrainingController
from torchspec.inference.factory import create_inference_engines
from torchspec.offline.dataset import OfflineDataset
from torchspec.offline.saving_actor import OfflineSavingActor
from torchspec.ray.placement_group import _ensure_ray_initialized, create_placement_groups
from torchspec.train_entry import (
    _get_draft_model_config,
    _resolve_batch_size,
    _validate_and_configure_dflash,
)
from torchspec.transfer.mooncake.utils import launch_mooncake_master
from torchspec.utils.env import get_torchspec_env_vars
from torchspec.utils.logging import logger


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--overwrite", action="store_true")
    cli, unknown = parser.parse_known_args()
    args = config_to_flat_args(load_config(cli.config, cli_args=unknown or None))
    args.rank = 0
    args.world_size = args.training_num_nodes * args.training_num_gpus_per_node
    if getattr(args, "attention_backend", None) == "usp":
        raise ValueError("Offline materialization does not support USP")
    _resolve_batch_size(args)
    return cli, args


def _save_vocab_mapping(controller, output_dir: str, draft_config) -> None:
    draft_vocab_size = getattr(draft_config, "draft_vocab_size", None)
    if draft_vocab_size is None or draft_vocab_size == draft_config.vocab_size:
        return
    path = Path(output_dir) / "vocab_mapping.pt"
    d2t, t2d = ray.get(
        controller.compute_vocab_mapping.remote(draft_config.vocab_size, draft_vocab_size)
    )
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"d2t": d2t.cpu(), "t2d": t2d.cpu()}, temporary)
    os.replace(temporary, path)


def materialize(args, output_dir: str, *, overwrite: bool = False) -> dict[str, int]:
    if getattr(args, "inference_engine_type", None) == "offline":
        raise ValueError("Materialization requires a live inference engine")

    draft_config = _get_draft_model_config(args)
    args.draft_model_config_obj = draft_config
    _validate_and_configure_dflash(args, draft_config)
    # A one-item dispatch lets the saving actor persist every source sample,
    # including a final partial inference batch.
    args.per_dp_rank_batch_size = 1
    OfflineDataset(
        output_dir,
        create=True,
        last_hidden_states_prenorm=args.last_hidden_states_prenorm,
        overwrite=overwrite,
    )

    _ensure_ray_initialized()
    driver_node_id = ray.get_runtime_context().get_node_id()
    controller = AsyncTrainingController.options(
        runtime_env={"env_vars": get_torchspec_env_vars()},
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
    ).remote(args, 1)
    train_size, eval_size = ray.get(
        [controller.load_dataset.remote(args), controller.load_eval_dataset.remote(args)]
    )
    _save_vocab_mapping(controller, output_dir, draft_config)

    inference_pg = create_placement_groups(args, roles={"inference"})["inference"]
    launch_mooncake_master(args)
    mooncake_config = build_mooncake_config(args)
    engines = create_inference_engines(args, inference_pg, mooncake_config)

    saver = OfflineSavingActor.remote(output_dir, mooncake_config)
    train_queues, eval_queues = ray.get(
        [controller.get_train_queues.remote(), controller.get_eval_queues.remote()]
    )
    ray.get(saver.set_queues.remote(train_queues[0], eval_queues[0]))

    if getattr(args, "max_sample_pool_size", 0) <= 0:
        args.max_sample_pool_size = max(
            8,
            getattr(args, "inference_batch_size", 1)
            * getattr(args, "max_concurrent_batches", 1)
            * 4,
        )
    manager = AsyncInferenceManager.remote(
        args,
        controller,
        inference_engines=engines,
        max_concurrent_batches=getattr(args, "max_concurrent_batches", 1),
    )
    if eval_size:
        ray.get(controller.submit_eval_chunk.remote(0, eval_size))
    ray.get(controller.submit_training_dataset.remote())

    manager_future = manager.run.remote()
    expected = {"train": train_size, "eval": eval_size}
    processed = {"train": 0, "eval": 0}
    counts = None
    try:
        while processed != expected:
            made_progress = False
            for split, method in (
                ("eval", controller.try_dispatch_eval_batch),
                ("train", controller.try_dispatch_batch),
            ):
                if processed[split] < expected[split] and ray.get(method.remote()):
                    ray.get(saver.save_from_queue.remote(split))
                    processed[split] += 1
                    made_progress = True

            if made_progress:
                logger.info("Materialization progress: %s / %s", processed, expected)
                continue

            manager_status, controller_status = ray.get(
                [manager.get_status.remote(), controller.get_full_status.remote()]
            )
            drained = (
                manager_status["prompt_buffer_size"] == 0
                and manager_status["pending_tasks"] == 0
                and controller_status["prompt_buffer_size"] == 0
                and controller_status["sample_pool_size"] == 0
                and ray.get(controller.get_eval_pool_size.remote()) == 0
            )
            if drained:
                raise RuntimeError(
                    f"Inference drained early: processed={processed}, expected={expected}"
                )
            time.sleep(0.05)

        if eval_size:
            ray.get(controller.finalize_eval_dispatch.remote())
        counts = ray.get(saver.counts.remote())
    finally:
        try:
            ray.get(saver.close.remote())
        except Exception as exc:
            logger.warning("Failed to close offline saver: %s", exc)
        _safe_training_cleanup(args, manager, manager_future, engines)
        try:
            ray.get(controller.shutdown.remote())
        except Exception as exc:
            logger.warning("Failed to stop materialization controller: %s", exc)

    logger.info("Offline dataset ready at %s: %s", output_dir, counts)
    return counts


def main() -> None:
    cli, args = _parse_args()
    materialize(args, cli.output, overwrite=cli.overwrite)


if __name__ == "__main__":
    main()
