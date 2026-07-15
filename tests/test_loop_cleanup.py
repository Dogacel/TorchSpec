from types import SimpleNamespace
from unittest import mock

import pytest

from torchspec.controller import eval as eval_utils
from torchspec.controller import loop


class TestTrainingLoopCleanup:
    def test_safe_cleanup_stops_inference_and_shutdowns_mooncake_actor(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = object()

        stop_ref = inference_manager.stop.remote.return_value
        shutdown_ref = args._mooncake_master_actor.shutdown.remote.return_value

        with mock.patch("torchspec.controller.loop.ray.get") as mock_ray_get:
            loop._safe_training_cleanup(args, inference_manager, inference_future)

        mock_ray_get.assert_any_call(stop_ref)
        mock_ray_get.assert_any_call(inference_future)
        mock_ray_get.assert_any_call(shutdown_ref, timeout=10)

    def test_safe_cleanup_skips_mooncake_shutdown_when_actor_missing(self):
        args = SimpleNamespace()
        inference_manager = mock.MagicMock()
        inference_future = object()

        with mock.patch("torchspec.controller.loop.ray.get"):
            loop._safe_training_cleanup(args, inference_manager, inference_future)

        inference_manager.stop.remote.assert_called_once()

    def test_run_training_loop_finally_runs_cleanup_on_success(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = inference_manager.run.remote.return_value

        with (
            mock.patch("torchspec.controller.loop.training_loop", return_value="ok") as mock_impl,
            mock.patch("torchspec.controller.loop._safe_training_cleanup") as mock_cleanup,
        ):
            result = loop.run_training_loop(args, object(), inference_manager, object())

        assert result == "ok"
        mock_impl.assert_called_once()
        mock_cleanup.assert_called_once_with(
            args=args,
            inference_manager=inference_manager,
            inference_future=inference_future,
            inference_engines=None,
        )


def test_maybe_run_initial_eval_reports_loaded_checkpoint_before_training(tmp_path):
    args = SimpleNamespace(eval_at_start=True, output_dir=str(tmp_path))
    train_group = object()

    with (
        mock.patch(
            "torchspec.controller.loop.run_eval",
            return_value={"eval/avg_loss": 1.5},
        ) as mock_eval,
        mock.patch(
            "torchspec.controller.loop.run_flowspec_ode_eval",
            return_value={"ode_eval/acc_0": 0.6},
        ) as mock_ode_eval,
    ):
        metrics = loop._maybe_run_initial_eval(args, 0, train_group, eval_enabled=True)

    assert metrics == {"eval/avg_loss": 1.5, "ode_eval/acc_0": 0.6}
    assert eval_utils._read_checkpoint_metadata(tmp_path / "initial_eval_metrics.json") == metrics
    mock_eval.assert_called_once_with(0, train_group, True)
    mock_ode_eval.assert_called_once_with(0, train_group, True, args)


def test_maybe_run_initial_eval_is_opt_in():
    args = SimpleNamespace(eval_at_start=False)

    with (
        mock.patch("torchspec.controller.loop.run_eval") as mock_eval,
        mock.patch("torchspec.controller.loop.run_flowspec_ode_eval") as mock_ode_eval,
    ):
        metrics = loop._maybe_run_initial_eval(args, 0, object(), eval_enabled=True)

    assert metrics == {}
    mock_eval.assert_not_called()
    mock_ode_eval.assert_not_called()


def test_run_training_loop_finally_runs_cleanup_on_exception():
    args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
    inference_manager = mock.MagicMock()
    inference_future = inference_manager.run.remote.return_value

    with (
        mock.patch(
            "torchspec.controller.loop.training_loop",
            side_effect=RuntimeError("training failed"),
        ),
        mock.patch("torchspec.controller.loop._safe_training_cleanup") as mock_cleanup,
    ):
        with pytest.raises(RuntimeError, match="training failed"):
            loop.run_training_loop(args, object(), inference_manager, object())

    mock_cleanup.assert_called_once_with(
        args=args,
        inference_manager=inference_manager,
        inference_future=inference_future,
        inference_engines=None,
    )


def test_generate_eval_cache_dispatches_and_finalizes():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()

    controller.try_dispatch_eval_batch.remote.side_effect = [True, True, True, True]

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=8,
    )

    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep"),
    ):
        eval_utils.generate_eval_cache(controller, train_group, state)

    assert train_group.cache_eval_samples.call_count == 4
    train_group.cache_eval_samples.assert_called_with(1)
    controller.finalize_eval_dispatch.remote.assert_called_once()


def test_run_eval_writes_all_scalar_metrics_to_tensorboard():
    train_group = mock.MagicMock()
    train_group.run_eval.return_value = [
        {
            "eval/avg_loss": 2.5,
            "eval/acc_0": 0.4,
            "eval/greedy_acceptance_length": 0.7,
            "ignored": object(),
        }
    ]
    writer = mock.MagicMock()

    with (
        mock.patch("torchspec.controller.eval.get_tb_writer", return_value=writer),
        mock.patch("torchspec.controller.eval.wandb.run", None),
    ):
        metrics = eval_utils.run_eval(100, train_group, eval_enabled=True)

    assert metrics["eval/step"] == 100
    assert writer.add_scalar.call_args_list == [
        mock.call("eval/avg_loss", 2.5, 100),
        mock.call("eval/acc_0", 0.4, 100),
        mock.call("eval/greedy_acceptance_length", 0.7, 100),
        mock.call("eval/step", 100, 100),
    ]


def test_run_flowspec_ode_eval_logs_separate_integrated_metrics():
    train_group = mock.MagicMock()
    train_group.run_flowspec_ode_eval.return_value = [
        {
            "ode_eval/acc_0": 0.35,
            "ode_eval/block_match_length": 0.6,
            "ode_eval/num_rows": 100.0,
        }
    ]
    args = SimpleNamespace(
        flowspec_ode_eval_steps=32,
        flowspec_ode_eval_max_samples=100,
        flowspec_ode_eval_seed=1234,
    )
    writer = mock.MagicMock()

    with (
        mock.patch("torchspec.controller.eval.get_tb_writer", return_value=writer),
        mock.patch("torchspec.controller.eval.wandb.run", None),
    ):
        metrics = eval_utils.run_flowspec_ode_eval(
            4000,
            train_group,
            eval_enabled=True,
            args=args,
        )

    train_group.run_flowspec_ode_eval.assert_called_once_with(32, 100, 1234)
    assert metrics["ode_eval/acc_0"] == 0.35
    assert metrics["ode_eval/step"] == 4000
    assert writer.add_scalar.call_args_list == [
        mock.call("ode_eval/acc_0", 0.35, 4000),
        mock.call("ode_eval/block_match_length", 0.6, 4000),
        mock.call("ode_eval/num_rows", 100.0, 4000),
        mock.call("ode_eval/step", 4000, 4000),
    ]


def test_run_flowspec_ode_eval_supports_multiple_step_counts():
    train_group = mock.MagicMock()
    train_group.run_flowspec_ode_eval.side_effect = [
        [{"ode_eval/acc_0": 0.6, "ode_eval/num_steps": 1}],
        [{"ode_eval/acc_0": 0.5, "ode_eval/num_steps": 32}],
    ]
    args = SimpleNamespace(
        flowspec_ode_eval_step_counts=[1, 32],
        flowspec_ode_eval_max_samples=100,
        flowspec_ode_eval_seed=1234,
    )
    writer = mock.MagicMock()

    with (
        mock.patch("torchspec.controller.eval.get_tb_writer", return_value=writer),
        mock.patch("torchspec.controller.eval.wandb.run", None),
    ):
        metrics = eval_utils.run_flowspec_ode_eval(
            1000,
            train_group,
            eval_enabled=True,
            args=args,
        )

    assert train_group.run_flowspec_ode_eval.call_args_list == [
        mock.call(1, 100, 1234),
        mock.call(32, 100, 1234),
    ]
    assert metrics["ode_eval/ode1/acc_0"] == 0.6
    assert metrics["ode_eval/ode32/acc_0"] == 0.5
    assert metrics["ode_eval/step"] == 1000
    assert writer.add_scalar.call_args_list == [
        mock.call("ode_eval/ode1/acc_0", 0.6, 1000),
        mock.call("ode_eval/ode1/num_steps", 1, 1000),
        mock.call("ode_eval/ode32/acc_0", 0.5, 1000),
        mock.call("ode_eval/ode32/num_steps", 32, 1000),
        mock.call("ode_eval/step", 1000, 1000),
    ]


def test_checkpoint_meta_keeps_all_scalar_eval_metrics(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_dir / "iter_0000101"
    checkpoint_path.mkdir(parents=True)
    eval_utils._write_checkpoint_metadata(
        checkpoint_path / "meta.json",
        {"global_step": 100},
    )
    metrics = {
        "eval/avg_loss": 1.5,
        "eval/simulated_acc_len": 2.0,
        "eval/block_match_length": 1.25,
        "ode_eval/block_match_length": 1.1,
        "eval/step": 100,
        "ode_eval/step": 100,
        "eval/non_scalar": object(),
        "train/avg_loss": 2.5,
    }

    best = eval_utils.update_checkpoint_eval_meta(
        str(checkpoint_dir),
        step=100,
        eval_metrics=metrics,
        current_best=-float("inf"),
    )
    stored = eval_utils._read_checkpoint_metadata(checkpoint_path / "meta.json")

    assert best == 2.0
    assert stored["eval/block_match_length"] == 1.25
    assert stored["ode_eval/block_match_length"] == 1.1
    assert "eval/step" not in stored
    assert "ode_eval/step" not in stored
    assert "eval/non_scalar" not in stored
    assert "train/avg_loss" not in stored


def test_generate_eval_cache_interleaves_refill_and_drain_without_stalling():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()

    controller.try_dispatch_eval_batch.remote.side_effect = [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=4,
    )

    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep") as mock_sleep,
    ):
        eval_utils.generate_eval_cache(controller, train_group, state)

    assert train_group.cache_eval_samples.call_count == 4
    assert controller.submit_eval_chunk.remote.call_args_list == [
        mock.call(4, 6),
        mock.call(6, 8),
    ]
    controller.finalize_eval_dispatch.remote.assert_called_once()
    assert mock_sleep.call_count == 4


def test_generate_eval_cache_times_out_when_eval_never_arrives():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=4,
    )

    controller.try_dispatch_eval_batch.remote.return_value = False
    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep") as mock_sleep,
        mock.patch("torchspec.controller.eval.EVAL_CACHE_IDLE_TIMEOUT", 0.0),
    ):
        with pytest.raises(TimeoutError, match="Timed out while waiting for eval cache generation"):
            eval_utils.generate_eval_cache(controller, train_group, state)

    train_group.cache_eval_samples.assert_not_called()
    controller.finalize_eval_dispatch.remote.assert_not_called()
    mock_sleep.assert_not_called()


def test_setup_eval_dispatch_bs_is_dp_size():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 4

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=2,
        inference_batch_size=4,
        max_sample_pool_size=64,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path="model",
        max_seq_length=4096,
    )

    with mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x):
        state = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=16,
        )

    assert state.eval_enabled is True
    assert state.eval_dispatch_bs == 2
    assert state.eval_dataset_size == 16
    assert state.initial_eval_submit_count == 4
    controller.submit_eval_chunk.remote.assert_called_once_with(0, 4)


def test_setup_eval_dispatch_bs_caps_at_dataset_size():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 4

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=8,
        inference_batch_size=4,
        max_sample_pool_size=64,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path="model",
        max_seq_length=4096,
    )

    with mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x):
        state = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=4,
        )

    assert state.eval_dispatch_bs == 4
    assert state.initial_eval_submit_count == 4
    controller.submit_eval_chunk.remote.assert_called_once_with(0, 4)
