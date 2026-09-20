#!/usr/bin/env python3
"""Where does the trained compressor actually look?

The recent-window baseline collapsed (ppl 2574 vs 54 for no context at all),
which is the attention-sink failure StreamingLLM describes: drop the first few
tokens and the model has nowhere to dump its spare attention mass. So the
question is whether the learned compressor rediscovered the sink on its own.

Prints the marginal source-position mass, summed over slots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compress import KVCompressor
from train import load_chunks

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    cfg = model.config
    comp = KVCompressor(cfg.num_hidden_layers, cfg.head_dim, args.m, args.seg).cuda()
    comp.load_state_dict(torch.load(HERE / "ckpt" / f"comp_m{args.m}.pt"))
    comp.eval()

    data = load_chunks(tok, args.seg, args.n)
    mass = torch.zeros(cfg.num_hidden_layers, args.seg)
    with torch.no_grad():
        for i in range(args.n):
            ids = data[i : i + 1, : args.seg].cuda()
            out = model(ids, use_cache=True)
            for li, l in enumerate(out.past_key_values.layers):
                _, _, w = comp.compress_layer(l.keys, l.values, li)
                # w: [B,H,S,m] -> total mass each source position receives
                mass[li] += w.sum(dim=3).mean(dim=1)[0].float().cpu()
    mass /= args.n

    avg = mass.mean(dim=0)
    uniform = args.m / args.seg
    print(f"m={args.m}  uniform mass per position = {uniform:.4f}\n")
    print("first 12 positions (attention-sink region):")
    for j in range(12):
        print(f"  pos {j:3d}: {avg[j]:.4f}  ({avg[j]/uniform:5.1f}x uniform)")
    print("\nlast 12 positions (recency region):")
    for j in range(args.seg - 12, args.seg):
        print(f"  pos {j:3d}: {avg[j]:.4f}  ({avg[j]/uniform:5.1f}x uniform)")

    print("\nmass by region:")
    for name, sl in [("0-3   sink", slice(0, 4)), ("4-15", slice(4, 16)),
                     ("16-127", slice(16, 128)), ("128-383 middle", slice(128, 384)),
                     ("384-479", slice(384, 480)), ("480-511 recent", slice(480, 512))]:
        blk = avg[sl]
        print(f"  {name:16s} {blk.sum():7.3f} total  ({blk.mean()/uniform:5.1f}x uniform)")

    torch.save(mass, HERE / "out" / f"mass_m{args.m}.pt")
    print(f"\nsaved out/mass_m{args.m}.pt")


if __name__ == "__main__":
    main()
