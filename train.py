#!/usr/bin/env python3
"""Train the KV bottleneck and score it against both bounds.

Setup is deliberately non-recursive for v0: one 1024-token window split into
segment A (the past, to be compressed) and segment B (what we predict).

  upper : B attends to all 512 of A's KV      -- nothing lost
  lower : B attends to nothing                -- everything lost
  comp  : B attends to m compressed slots     -- what we are training

The headline number is recovery = (lower - comp) / (lower - upper): the
fraction of the truncation penalty the compressor buys back. 1.0 would be
lossless at m slots; 0.0 means the slots carry nothing useful.

Segment B keeps its true absolute positions (512..1023) in every condition, so
the only variable is what it can see, never where it thinks it is.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from compress import KVCompressor

HERE = Path(__file__).resolve().parent


def load_chunks(tok, seg: int, limit: int) -> torch.Tensor:
    L = seg * 2
    out = []
    # MEMZIP_DATA points the loader at a different corpus without touching any
    # call site: the memory module is domain-specific (dialogue +53.5% against
    # -54.4% on novels with the same checkpoint), so retraining per domain is
    # the expected workflow rather than an edge case.
    import os
    _dir = HERE / os.environ.get("MEMZIP_DATA", "data")
    for p in sorted(_dir.glob("*.txt")):
        ids = tok(p.read_text(encoding="utf-8"), return_tensors="pt").input_ids[0]
        for s in range(0, len(ids) - L, L):
            out.append(ids[s : s + L])
            if len(out) >= limit:
                return torch.stack(out)
    return torch.stack(out)


def cache_from(pairs, cfg) -> DynamicCache:
    c = DynamicCache(config=cfg)
    for i, (k, v) in enumerate(pairs):
        c.update(k, v, i)
    return c


def seg_b_loss(model, cfg, ids_b, past_pairs, seg: int):
    """CE on segment B given an optional compressed/full past."""
    B, S = ids_b.shape
    pos = torch.arange(seg, seg + S, device=ids_b.device).unsqueeze(0).expand(B, -1)
    kw = {}
    if past_pairs is not None:
        cache = cache_from(past_pairs, cfg)
        plen = past_pairs[0][0].shape[2]
        kw["past_key_values"] = cache
        kw["attention_mask"] = torch.ones(B, plen + S, device=ids_b.device, dtype=torch.long)
        kw["cache_position"] = torch.arange(plen, plen + S, device=ids_b.device)
    logits = model(ids_b, position_ids=pos, use_cache=False, **kw).logits
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                           ids_b[:, 1:].reshape(-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--eval-n", type=int, default=24)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--mix", default="softmax", choices=["softmax", "sigmoid", "linear"])
    ap.add_argument("--no-realign", action="store_true",
                    help="ablation: mix K in rotated space (the v0 failure mode)")
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config

    data = load_chunks(tok, args.seg, args.chunks)
    n_eval = args.eval_n
    ev, tr = data[:n_eval], data[n_eval:]
    print(f"train {len(tr)} chunks / eval {len(ev)} chunks, seg={args.seg} m={args.m}")

    comp = KVCompressor(cfg.num_hidden_layers, cfg.head_dim, args.m, args.seg,
                        model.model.rotary_emb.inv_freq, n_sink=args.sink,
                        realign=not args.no_realign, mix_mode=args.mix).to(dev)  # fp32
    n_par = sum(p.numel() for p in comp.parameters())
    print(f"compressor params: {n_par/1e6:.2f}M")
    opt = torch.optim.AdamW(comp.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)

    free_idx = torch.cat([
        torch.arange(args.sink),
        torch.linspace(args.sink, args.seg - 1, args.m - args.sink).long(),
    ]).to(dev)

    def encode_a(ids_a):
        with torch.no_grad():
            out = model(ids_a, use_cache=True)
        return [(l.keys, l.values) for l in out.past_key_values.layers]

    @torch.no_grad()
    def evaluate():
        tot = {"upper": 0.0, "lower": 0.0, "comp": 0.0, "recent": 0.0}
        stats = None
        for i in range(0, len(ev), args.bs):
            b = ev[i : i + args.bs].to(dev)
            a, bb = b[:, : args.seg], b[:, args.seg :]
            pairs = encode_a(a)
            cpairs = []
            for li, (k, v) in enumerate(pairs):
                kc, vc, w = comp.compress_layer(k, v, li)
                cpairs.append((kc, vc))
                if li == cfg.num_hidden_layers // 2:
                    stats = comp.selection_stats(w)
            # the free trick worth beating: sink + evenly spaced picks (+21.7%)
            rpairs = [(k[:, :, free_idx], v[:, :, free_idx]) for k, v in pairs]
            n = b.shape[0]
            tot["upper"] += seg_b_loss(model, cfg, bb, pairs, args.seg).item() * n
            tot["lower"] += seg_b_loss(model, cfg, bb, None, args.seg).item() * n
            tot["comp"] += seg_b_loss(model, cfg, bb, cpairs, args.seg).item() * n
            tot["recent"] += seg_b_loss(model, cfg, bb, rpairs, args.seg).item() * n
        for k in tot:
            tot[k] /= len(ev)
        gap = tot["lower"] - tot["upper"]
        rec = lambda x: (tot["lower"] - x) / gap if gap > 1e-6 else float("nan")
        tot["recovery"] = rec(tot["comp"])
        tot["rec_recent"] = rec(tot["recent"])
        tot["gap"] = gap
        if stats:
            tot.update(stats)
        return tot

    e0 = evaluate()
    print(f"\n[init] upper={e0['upper']:.4f} lower={e0['lower']:.4f} gap={e0['gap']:.4f}")
    print(f"       recent(m={args.m})={e0['recent']:.4f}  recovery={e0['rec_recent']:+.1%}   <- to beat")
    print(f"       comp            ={e0['comp']:.4f}  recovery={e0['recovery']:+.1%}")
    print(f"       ppl: upper={math.exp(e0['upper']):.2f} lower={math.exp(e0['lower']):.2f} "
          f"recent={math.exp(e0['recent']):.2f} comp={math.exp(e0['comp']):.2f}")
    print(f"       select entropy={e0['entropy']:.3f} (uniform={math.log(args.seg):.3f}) "
          f"max_w={e0['max_w']:.4f}\n")

    log = [{"step": 0, **e0}]
    t0 = time.time()
    perm = torch.randperm(len(tr))
    ptr = 0
    for step in range(1, args.steps + 1):
        if ptr + args.bs > len(tr):
            perm = torch.randperm(len(tr)); ptr = 0
        b = tr[perm[ptr : ptr + args.bs]].to(dev); ptr += args.bs
        a, bb = b[:, : args.seg], b[:, args.seg :]

        pairs = encode_a(a)
        cpairs = [comp.compress_layer(k, v, li)[:2] for li, (k, v) in enumerate(pairs)]
        loss = seg_b_loss(model, cfg, bb, cpairs, args.seg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(comp.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 50 == 0 or step == args.steps:
            e = evaluate()
            log.append({"step": step, **e})
            print(f"[{step:4d}] loss={loss.item():.4f} | comp={e['comp']:.4f} "
                  f"recovery={e['recovery']:+.1%} (recent {e['rec_recent']:+.1%}) | "
                  f"ent={e['entropy']:.2f} max_w={e['max_w']:.3f} | {time.time()-t0:.0f}s")

    tag = f"m{args.m}" + ("_norealign" if args.no_realign else "") + \
          ("" if args.mix == "softmax" else f"_{args.mix}")
    out = HERE / "out" / f"train_{tag}.json"
    out.write_text(json.dumps({"args": vars(args), "log": log}, indent=1))
    torch.save(comp.state_dict(), HERE / "ckpt" / f"comp_{tag}.pt")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
