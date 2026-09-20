#!/usr/bin/env python3
"""Recurrent compression: a constant KV budget over an arbitrarily long text.

Everything so far compressed one segment in isolation. That is not the problem
worth solving -- the problem is that KV grows linearly forever. So:

    seg 1 -> [state_1]
    seg 2 + state_1 -> [state_2]          state_2 is the same size as state_1
    seg 3 + state_2 -> [state_3]
    ...

The state never grows. Each new set of memory tokens attends to the previous
state while reading its own segment, so it compresses on top of what is already
there rather than starting over. After n segments the cache holds k entries,
not n*512.

This is also where this differs from the KV-compression literature: those
methods compress a prefix once, offline. Nothing there has to survive being
re-compressed twenty times, and that is exactly what breaks first -- each pass
re-encodes an already lossy state, so errors compound. Measuring that decay is
the point of this file.

SUPERSEDED by train_recur.py, which also evaluates -- and fixes a position bug
here. This file places the tail at seg*hops+k while the state's own KV sits at
positions 0..k-1, leaving a gap of thousands of positions between the state and
what it is meant to inform. Only the mem condition pays that (upper reads a
contiguous prefix), so the numbers below understate it: 2 hops reads +65.6% here
and +87% under the correct convention. Kept for the raw single-hop-checkpoint
curve it produced; use train_recur.py for anything new.

Reported: loss on a held-out continuation after n hops, against two references
  - full   : the entire preceding text verbatim (an upper bound that costs O(n))
  - none   : no context at all
so recovery stays comparable to the single-hop numbers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def long_chunks(tok, seg: int, hops: int, limit: int) -> torch.Tensor:
    """Windows long enough for `hops` segments plus one to predict."""
    need = seg * (hops + 1)
    out = []
    for p in sorted((HERE / "data").glob("*.txt")):
        ids = tok(p.read_text(encoding="utf-8"), return_tensors="pt").input_ids[0]
        for s in range(0, len(ids) - need, need):
            out.append(ids[s : s + need])
            if len(out) >= limit:
                return torch.stack(out)
    return torch.stack(out) if out else torch.empty(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--lora", default="")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--mem-depth", action="store_true",
                    help="checkpoint was trained with per-layer memory biases")
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    dev = "cuda"
    DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DT).to(dev).eval()
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).eval()
    # PeftModel proxies .config, and HF's own base_model property already unwraps
    # a bare model one level -- descending into it by hand overshoots
    cfg = model.config

    depth = (cfg.num_hidden_layers - 1) if args.mem_depth else 0
    # random-budget training allocates max(k, k_hi) slots, so the checkpoint is
    # usually wider than the budget being tested here; build it at its own width
    # and use the first args.k slots, exactly as training did
    slots = args.k
    sd = None
    if args.ckpt:
        sd = torch.load(args.ckpt)
        sd = sd["mem"] if "mem" in sd else sd
        slots = sd["emb"].shape[0]
    mem = MemoryTokens(slots, cfg.hidden_size, depth=depth).to(dev)
    if sd is not None:
        mem.load_state_dict(sd)
        print(f"loaded {Path(args.ckpt).name} ({slots} slots, using first {args.k})")

    data = long_chunks(tok, args.seg, args.hops, args.n)
    if len(data) == 0:
        print("not enough long text for this many hops"); return
    print(f"{len(data)} windows of {args.seg*(args.hops+1)} tokens | "
          f"seg={args.seg} k={args.k} hops={args.hops}")
    print(f"constant budget: {args.k} entries vs {args.seg*args.hops} verbatim "
          f"({args.seg*args.hops/args.k:.0f}:1 after {args.hops} hops)\n")

    tot = {h: 0.0 for h in range(1, args.hops + 1)}
    ref = {"full": 0.0, "none": 0.0}
    with torch.no_grad():
        for i in range(len(data)):
            w = data[i : i + 1].to(dev)
            tail = w[:, args.seg * args.hops :]

            # recurrent pass: state carries forward, never grows
            state = None
            for h in range(args.hops):
                seg_ids = w[:, h * args.seg : (h + 1) * args.seg]
                past_len = 0 if state is None else args.k
                state = compress(model, mem, seg_ids, past=state,
                                 past_len=past_len, grad=False, k=args.k)
                # score the tail using only the state built so far
                tot[h + 1] += seg_b_loss(model, cfg, tail, state,
                                         args.seg * (h + 1) + args.k).item()

            full = [(l.keys, l.values) for l in
                    model(w[:, : args.seg * args.hops], use_cache=True).past_key_values.layers]
            ref["full"] += seg_b_loss(model, cfg, tail, full, args.seg * args.hops).item()
            ref["none"] += seg_b_loss(model, cfg, tail, None, args.seg * args.hops).item()

    n = len(data)
    up, lo = ref["full"] / n, ref["none"] / n
    gap = lo - up
    print(f"full(all {args.seg*args.hops} KV) CE={up:.4f} ppl={math.exp(up):.2f}")
    print(f"none                CE={lo:.4f} ppl={math.exp(lo):.2f}   gap={gap:.4f}\n")
    print(f"{'hops':>5} {'kept':>6} {'CE':>8} {'ppl':>8} {'recovery':>10}")
    print("-" * 42)
    rows = []
    for h in range(1, args.hops + 1):
        ce = tot[h] / n
        r = (lo - ce) / gap if gap > 1e-6 else float("nan")
        print(f"{h:5d} {args.k:6d} {ce:8.4f} {math.exp(ce):8.2f} {r:+9.1%}")
        rows.append({"hops": h, "ce": ce, "recovery": r})

    (HERE / "out" / f"recur_k{args.k}_h{args.hops}.json").write_text(
        json.dumps({"args": vars(args), "upper": up, "lower": lo, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
