#!/usr/bin/env python3
"""Train the CALM-style chunk autoencoder on the Qwen tokenizer.

The model and objective are CALM's, vendored verbatim (torchspec/models/calm_ae,
MIT). This adapter only provides: Qwen3 tokenization of open-perfectblend with
chat templating, per-sample random offsets (offset-invariant chunks), padding
to patch multiples, block grouping, and an HF Trainer with wandb reporting.

Usage (8 GPUs):
  torchrun --nproc_per_node=8 scripts/tools/train_chunk_ae.py \
      --output-dir outputs/chunk-ae-k4 --epochs 1
"""

import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from torchspec.models.calm_ae import Autoencoder, AutoencoderConfig

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
}

PATCH = 4
HOLDOUT = 5000
SEED = 42


def normalize(conv_list):
    out = []
    for turn in conv_list:
        role = turn.get("from") or turn.get("role")
        content = turn.get("value") or turn.get("content") or ""
        role = ROLE_MAPPING.get(role, role)
        if role in ("user", "assistant", "system") and content:
            out.append({"role": role, "content": content})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/chunk-ae-k4")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--per-device-batch", type=int, default=32)
    ap.add_argument("--wandb-project", default="qwen3-8b-dflash-flow")
    ap.add_argument("--wandb-run", default="chunk-ae-k4")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    pad_id = tok.pad_token_id

    ds = load_dataset("mlabonne/open-perfectblend", split="train")
    ds = ds.shuffle(seed=SEED)
    holdout = ds.select(range(HOLDOUT))
    train = ds.select(range(HOLDOUT, len(ds)))

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "holdout.jsonl"), "w") as f:
            for ex in holdout:
                f.write(json.dumps({"conversations": normalize(ex["conversations"])}) + "\n")

    def to_ids(batch, indices):
        outs = []
        for idx, conv in zip(indices, batch["conversations"]):
            msgs = normalize(conv)
            if not msgs:
                outs.append([])
                continue
            text = tok.apply_chat_template(msgs, tokenize=False)
            ids = tok(text, truncation=True, max_length=4096).input_ids
            # Per-sample random offset so patch boundaries cover all phases
            # (chunks at serving time are anchor-relative, not stride-aligned).
            off = random.Random(SEED + idx).randrange(PATCH)
            ids = ids[off:]
            if len(ids) % PATCH:
                ids = ids + [pad_id] * (PATCH - len(ids) % PATCH)
            outs.append(ids)
        return {"input_ids": outs}

    train = train.map(
        to_ids,
        batched=True,
        with_indices=True,
        num_proc=32,
        remove_columns=train.column_names,
        desc="tokenize",
    )

    block = args.block_size

    def group(batch):
        flat = [t for ids in batch["input_ids"] for t in ids]
        total = (len(flat) // block) * block
        chunks = [flat[i : i + block] for i in range(0, total, block)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    train = train.map(group, batched=True, num_proc=32, desc="group")

    cfg = AutoencoderConfig(
        vocab_size=len(tok),
        hidden_size=512,
        intermediate_size=1280,
        num_encoder_layers=2,
        num_decoder_layers=2,
        latent_size=128,
        patch_size=PATCH,
        ae_dropout=0.15,
        kl_clamp=0.5,
        kl_weight=1e-3,
        pad_token_id=pad_id,
    )
    model = Autoencoder(cfg)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_strategy="no",  # HF save_pretrained chokes on CALM's tied-weights
        # metadata under this transformers version; we save
        # the state dict directly below instead.
        report_to="wandb",
        run_name=args.wandb_run,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train)
    trainer.train()
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "autoencoder.pt"))
        cfg.save_pretrained(args.output_dir)
        tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
