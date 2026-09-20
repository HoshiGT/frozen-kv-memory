#!/usr/bin/env python3
"""Score every training-free way of keeping m KV entries, plus the trained one.

The first recent-window number was unfair: StreamingLLM keeps a few initial
"sink" tokens alongside the recent window precisely because dropping them
destroys the model. A compressor is only interesting if it beats the honest
version of that trick at the same budget.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compress import KVCompressor
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    cfg, m, S = model.config, args.m, args.seg

    comp = None
    ck = HERE / "ckpt" / f"comp_m{m}.pt"
    if ck.exists():
        comp = KVCompressor(cfg.num_hidden_layers, cfg.head_dim, m, S).cuda().eval()
        comp.load_state_dict(torch.load(ck))

    data = load_chunks(tok, S, args.n)

    def keep(pairs, idx):
        return [(k[:, :, idx], v[:, :, idx]) for k, v in pairs]

    sink4 = torch.cat([torch.arange(4), torch.arange(S - m + 4, S)])
    sink1 = torch.cat([torch.arange(1), torch.arange(S - m + 1, S)])
    stride = torch.linspace(0, S - 1, m).long()
    strided_sink = torch.cat([torch.arange(4), torch.linspace(4, S - 1, m - 4).long()])

    names = ["upper(512)", "lower(0)", f"recent({m})", f"sink1+recent({m-1})",
             f"sink4+recent({m-4})", f"stride({m})", f"sink4+stride({m-4})"]
    if comp:
        names.append(f"learned({m})")
    tot = {n: 0.0 for n in names}

    with torch.no_grad():
        for i in range(0, args.n, args.bs):
            b = data[i : i + args.bs].cuda()
            a, bb = b[:, :S], b[:, S:]
            pairs = [(l.keys, l.values) for l in model(a, use_cache=True).past_key_values.layers]
            n = b.shape[0]
            runs = {
                "upper(512)": pairs,
                "lower(0)": None,
                f"recent({m})": keep(pairs, torch.arange(S - m, S)),
                f"sink1+recent({m-1})": keep(pairs, sink1),
                f"sink4+recent({m-4})": keep(pairs, sink4),
                f"stride({m})": keep(pairs, stride),
                f"sink4+stride({m-4})": keep(pairs, strided_sink),
            }
            if comp:
                runs[f"learned({m})"] = [
                    comp.compress_layer(k, v, li)[:2] for li, (k, v) in enumerate(pairs)
                ]
            for name, pk in runs.items():
                tot[name] += seg_b_loss(model, cfg, bb, pk, S).item() * n

    for k in tot:
        tot[k] /= args.n
    up, lo = tot["upper(512)"], tot["lower(0)"]
    gap = lo - up
    print(f"\nseg={S} m={m} n={args.n} chunks")
    print(f"gap = {gap:.4f} nats   (ppl {math.exp(lo):.1f} -> {math.exp(up):.1f})\n")
    print(f"{'method':24s} {'CE':>8s} {'ppl':>9s} {'recovery':>10s}")
    print("-" * 54)
    for name in names:
        ce = tot[name]
        r = (lo - ce) / gap
        print(f"{name:24s} {ce:8.4f} {math.exp(ce):9.1f} {r:+9.1%}")


if __name__ == "__main__":
    main()
