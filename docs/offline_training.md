# Offline Training

Offline training lets you run target-model inference once, save the hidden
states, and reuse them for draft-model training. This is useful for development
because training no longer needs a target inference GPU, thus you can test your
workflows on only 1 GPU. **Warning:** This workflow is not optimized for
throughput yet.

## One GPU, 1,000 samples

Run these commands from the TorchSpec repository root.

```bash
conda activate torchspec
```

### 1. Save the target hidden states

```bash
python -m torchspec.offline.generate \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  --output outputs/qwen3-8b-dflash-offline-1000 \
  inference.inference_num_gpus=1 \
  inference.inference_num_gpus_per_node=1
```

This runs Qwen3-8B with SGLang on one GPU. The config uses the bundled sample
dataset and produces 1,000 replay samples. Expect the output to use about 6 GB.

Materialization resumes by default. Add `--overwrite` to replace an existing
output directory.

### 2. Train from the saved data

```bash
python -m torchspec.train_entry \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  inference.inference_engine_type=offline \
  inference.offline.data_path=outputs/qwen3-8b-dflash-offline-1000 \
  inference.offline.num_engines=1 \
  training.training_num_gpus_per_node=1 \
  output_dir=outputs/qwen3-8b-dflash-offline-dev
```

This uses one GPU for training and does not start an inference engine. The
saved dataset can be reused for as many training runs as needed.

## Use your own data

Override the dataset path while materializing:

```bash
python -m torchspec.offline.generate \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  --output /path/to/offline-data \
  dataset.train_data_path=/path/to/train.jsonl \
  inference.inference_num_gpus=1 \
  inference.inference_num_gpus_per_node=1
```

Then pass that output directory to
`inference.offline.data_path` when training.

The materialization and training configs must describe the same target model,
tokenizer, draft method, and hidden-state layout. Target-model weights must
remain available during training because TorchSpec still uses the target
embedding, normalization, and LM-head weights.

Offline training does not currently support USP or `train_with_decode`.
