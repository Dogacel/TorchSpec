#!/usr/bin/env python3
"""Acceptance gate for the chunk autoencoder.

Encodes held-out chunks with the MEAN latent (no sampling noise), decodes, and
reports per-token and per-chunk exact reconstruction accuracy — at all four
patch offsets, to verify offset invariance. Prints human-readable examples.

Pass criterion: >99% per-token accuracy at every offset.

Usage: python scripts/tools/chunk_ae_gate.py --model outputs/chunk-ae-k4
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from torchspec.models.calm_ae import Autoencoder

PATCH = 4


@torch.no_grad()
def encode_mean(model, ids):
    latent = model.encoder(input_ids=ids)
    mean, _ = torch.chunk(latent, 2, dim=-1)
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/chunk-ae-k4")
    ap.add_argument("--num-convs", type=int, default=500)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    from torchspec.models.calm_ae import AutoencoderConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoencoderConfig.from_pretrained(args.model)
    model = Autoencoder(cfg)
    model.load_state_dict(
        torch.load(f"{args.model}/autoencoder.pt", map_location="cpu", weights_only=True)
    )
    model = model.to(torch.bfloat16).cuda()
    model.eval()
    pad_id = tok.pad_token_id

    convs = []
    with open(f"{args.model}/holdout.jsonl") as f:
        for i, line in enumerate(f):
            if i >= args.num_convs:
                break
            convs.append(json.loads(line)["conversations"])

    print(f"{'offset':>7} {'token_acc':>10} {'chunk_acc':>10} {'n_chunks':>9}")
    worst = 1.0
    for offset in range(PATCH):
        n_tok = n_tok_ok = n_chunk = n_chunk_ok = 0
        for conv in convs:
            if not conv:
                continue
            text = tok.apply_chat_template(conv, tokenize=False)
            ids = tok(text, truncation=True, max_length=2048).input_ids[offset:]
            if len(ids) < PATCH:
                continue
            ids = ids[: (len(ids) // PATCH) * PATCH]
            t = torch.tensor(ids, device="cuda").view(1, -1)
            z = encode_mean(model, t)
            logits = model.decoder(latent_states=z.to(torch.bfloat16))
            pred = logits.argmax(-1).view(-1)
            tgt = t.view(-1)
            ok = pred == tgt
            n_tok += tgt.numel()
            n_tok_ok += int(ok.sum())
            ok_c = ok.view(-1, PATCH).all(-1)
            n_chunk += ok_c.numel()
            n_chunk_ok += int(ok_c.sum())
        tok_acc = n_tok_ok / max(n_tok, 1)
        chunk_acc = n_chunk_ok / max(n_chunk, 1)
        worst = min(worst, tok_acc)
        print(f"{offset:>7} {tok_acc:>10.4f} {chunk_acc:>10.4f} {n_chunk:>9}")

    print("\nEXAMPLES (chunk in -> latent(128) -> chunk out):")
    conv = convs[0]
    text = tok.apply_chat_template(conv, tokenize=False)
    ids = tok(text).input_ids[: PATCH * args.examples]
    ids = ids[: (len(ids) // PATCH) * PATCH]
    t = torch.tensor(ids, device="cuda").view(1, -1)
    z = encode_mean(model, t)
    pred = model.decoder(latent_states=z.to(torch.bfloat16)).argmax(-1).view(-1, PATCH)
    for i, (a, b) in enumerate(zip(t.view(-1, PATCH), pred)):
        ok = "OK " if torch.equal(a, b) else "BAD"
        print(f"  [{ok}] {tok.decode(a.tolist())!r} -> {tok.decode(b.tolist())!r}")

    print(
        f"\nGATE: {'PASS' if worst > 0.99 else 'FAIL'} (worst offset token acc {worst:.4f}, need > 0.99)"
    )


if __name__ == "__main__":
    main()
