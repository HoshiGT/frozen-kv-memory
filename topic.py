#!/usr/bin/env python3
"""Does a segment's topic spread predict how much budget it needs?

Hoshi's conjecture: a single-topic stretch of conversation should compress
tighter than one that jumps between topics, because what has to be understood
is less entangled. If that holds, budget should not be uniform across segments
-- and, more usefully, the signal that decides it should be cheap.

Two measurements per segment, then their rank correlation:

  spread   how much the segment moves around semantically. The segment is cut
           into `--blocks` sub-blocks, each reduced to the mean hidden state at
           a middle layer, and spread is their mean pairwise cosine distance.
           One topic -> blocks resemble each other -> low. Costs one forward
           pass and no compression at all, which is the point: if this predicts
           hunger, it can allocate budget online.

  hunger   how much the segment gains from more slots:
           recovery(k_hi) - recovery(k_lo). A segment that is already saturated
           at 16 slots gains nothing from 128 and should not be given them.

A positive correlation means budget should follow content rather than be split
evenly -- the concrete form of the "state should grow" plan.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                     # average ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 1e-9 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--k-lo", type=int, default=16)
    ap.add_argument("--k-hi", type=int, default=128)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_deep_frozen.pt"))
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config

    sd = torch.load(args.ckpt)
    sd = sd["mem"] if "mem" in sd else sd
    depth = sd["deep"].shape[0] if "deep" in sd else 0
    mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size, depth=depth).to(dev)
    mem.load_state_dict(sd)

    data = load_chunks(tok, args.seg * 2, args.n)
    mid = cfg.num_hidden_layers // 2
    bl = args.seg // args.blocks

    spread, hunger, rec_lo, rec_hi, gaps, selfce = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(data)):
            w = data[i : i + 1].to(dev)
            a, b = w[:, : args.seg], w[:, args.seg :]

            h = model(a, output_hidden_states=True).hidden_states[mid][0]
            blocks = F.normalize(
                h[: args.blocks * bl].view(args.blocks, bl, -1).mean(1).float(), dim=-1)
            sim = blocks @ blocks.T
            off = ~torch.eye(args.blocks, dtype=torch.bool, device=dev)
            spread.append((1 - sim[off]).mean().item())
            # A more principled density signal: how hard the segment is to
            # predict from itself. High surprise = high information content =
            # plausibly more to store. Free at compression time, unlike spread.
            selfce.append(seg_b_loss(model, cfg, a, None, 0).item())

            lower = seg_b_loss(model, cfg, b, None, args.seg).item()
            full = [(l.keys, l.values) for l in model(a, use_cache=True).past_key_values.layers]
            upper = seg_b_loss(model, cfg, b, full, args.seg).item()
            gap = lower - upper
            ce_lo, ce_hi = [], []
            def rec(k, sink):
                st = compress(model, mem, a, grad=False, k=k)
                ce = seg_b_loss(model, cfg, b, st, args.seg + k).item()
                sink.append(ce)
                return (lower - ce) / gap if gap > 1e-6 else float("nan")
            lo, hi = rec(args.k_lo, ce_lo), rec(args.k_hi, ce_hi)
            rec_lo.append(lo); rec_hi.append(hi)
            # absolute nats, not a ratio: dividing by each segment's own gap
            # mixes "how much this context helps at all" into "how much this
            # segment wants more slots", and they are different questions
            hunger.append(ce_lo[0] - ce_hi[0])
            gaps.append(gap)

    n = len(spread)
    r = spearman(spread, hunger)
    print(f"{n} segments | blocks={args.blocks} | k {args.k_lo} vs {args.k_hi}\n")
    print(f"  spread : {min(spread):.3f} .. {max(spread):.3f}   "
          f"mean {sum(spread)/n:.3f}")
    print(f"  hunger : {min(hunger):+.3f} .. {max(hunger):+.3f} nats   "
          f"mean {sum(hunger)/n:+.3f}")
    print(f"  gap    : {min(gaps):.3f} .. {max(gaps):.3f} nats   mean {sum(gaps)/n:.3f}")
    print(f"  selfCE : {min(selfce):.3f} .. {max(selfce):.3f} nats   mean {sum(selfce)/n:.3f}")
    print(f"  recovery@{args.k_lo:<3}: {sum(rec_lo)/n:+.1%}   "
          f"recovery@{args.k_hi}: {sum(rec_hi)/n:+.1%}\n")
    print(f"  Spearman(spread, hunger_nats) = {r:+.3f}")
    print(f"  Spearman(spread, gap)         = {spearman(spread, gaps):+.3f}"
          f"   (does spread predict how useful the context is at all)")
    print(f"  Spearman(gap, hunger_nats)    = {spearman(gaps, hunger):+.3f}"
          f"   (do informative segments want more slots)")
    print(f"  Spearman(selfCE, hunger_nats) = {spearman(selfce, hunger):+.3f}"
          f"   (do surprising segments want more slots)")
    print(f"  Spearman(selfCE, spread)      = {spearman(selfce, spread):+.3f}"
          f"   (are the two cheap signals even measuring the same thing)")
    print("  positive => topic-spread segments need more slots, i.e. budget")
    print("             should follow content, and spread can decide it cheaply\n")

    # split into halves by spread: the practical question is whether allocating
    # by this signal would actually have helped
    order = sorted(range(n), key=lambda i: selfce[i])
    half = n // 2
    for name, idx in (("predictable (low CE)", order[:half]),
                      ("surprising (high CE)", order[half:])):
        print(f"  {name:<20} selfCE {sum(selfce[i] for i in idx)/len(idx):.3f}  "
              f"rec@{args.k_lo} {sum(rec_lo[i] for i in idx)/len(idx):+.1%}  "
              f"rec@{args.k_hi} {sum(rec_hi[i] for i in idx)/len(idx):+.1%}  "
              f"hunger {sum(hunger[i] for i in idx)/len(idx):+.3f} nats")


if __name__ == "__main__":
    main()
