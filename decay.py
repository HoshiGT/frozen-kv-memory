#!/usr/bin/env python3
"""Should anchors thin out with age?

Anchors are currently spread uniformly: 12.5% of every part of the history,
whether it is ten turns old or a thousand. That is almost certainly wrong. What
was said recently gets quoted back; what was said long ago survives as an
impression. If the probability that a position is needed falls with distance,
anchor density should fall with it -- and if it falls like 1/d, the total is
the integral of 1/d, which is log(n) rather than n.

That would be the difference between flattening the slope of KV growth, which
is what the hybrid does today, and actually breaking it.

Two measurements, in order of cheapness:

  1. distribution  where do the positions that actually matter live? Two
                   readings of "matter": every context position the
                   continuation reuses at all, and the positions the oracle
                   would pick, which weights by how much the reuse is worth.

  2. allocation    does acting on it help? The same budget of anchors, spread
                   uniformly versus concentrated near the recent end, scored
                   the usual way. A distribution that is only mildly skewed
                   will show nothing here, which is itself the answer.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import torch

from hybrid import oracle_anchors, query_anchors, speculate
from memory import kv_only
from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


# oracle-selected anchors per available position, measured at 8192 tokens over
# 8 log-spaced bands (decay.py, n=10). Peaks at 11-29 tokens back, not at the
# most recent, and falls only 4.5x across the whole range -- against a 114x fall
# in raw reuse. What gets quoted again is not what needs anchoring.
ORACLE_DENSITY = [0.1000, 0.1143, 0.2421, 0.1839, 0.1590, 0.1180, 0.0531, 0.0544]


def log_bins(n_ctx: int, n_bins: int = 8):
    """Edges spaced geometrically in distance from the end of the context."""
    edges = [0]
    for i in range(1, n_bins + 1):
        edges.append(int(round(n_ctx ** (i / n_bins))))
    edges[-1] = n_ctx
    return edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--hops", type=int, default=16)
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--bins", type=int, default=8)
    # query_anchors reads these off the namespace it is handed
    ap.add_argument("--score-mode", default="mean", choices=["mean", "maxhead", "sumsoft"])
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--weight", default="inv", choices=["inv", "measured"],
                    help="how the budget is split across distance bands")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_deep_frozen.pt"))
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    sd = torch.load(a.ckpt)
    sd = sd["mem"] if "mem" in sd else sd
    mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size,
                       depth=sd["deep"].shape[0] if "deep" in sd else 0).to(dev)
    mem.load_state_dict(sd)

    cut = a.seg * a.hops
    data = load_chunks(tok, cut + a.seg, a.n)
    edges = log_bins(cut, a.bins)
    reuse = [0] * a.bins          # positions the tail reuses at all
    picked = [0] * a.bins         # positions the oracle would anchor
    avail = [0] * a.bins          # positions existing in each band
    print(f"{len(data)} 个窗口 | 上下文 {cut} token | 距离按对数分 {a.bins} 段\n")

    def band(dist):
        for b in range(a.bins):
            if edges[b] < dist <= edges[b + 1]:
                return b
        return a.bins - 1

    tot = {"uniform": 0.0, "recency": 0.0, "upper": 0.0, "lower": 0.0}
    for i in range(len(data)):
        w = data[i : i + 1].to(dev)
        ctx, tail = w[:, :cut], w[:, cut : cut + a.seg]
        seq = ctx[0].tolist()

        # --- 1. where do reused positions live ---
        last = {}
        for pos, t in enumerate(seq):
            last[t] = pos                      # most recent occurrence wins
        for t in set(tail[0].tolist()):
            if t in last:
                reuse[band(cut - last[t])] += 1
        for pos in range(cut):
            avail[band(cut - pos)] += 1
        for p in oracle_anchors(ctx, tail, a.anchors, a.sink).tolist():
            picked[band(cut - p)] += 1

        # --- 2. does concentrating anchors near the recent end help ---
        full = kv_only(model, ctx)
        st = None
        for h in range(a.hops):
            st = compress(model, mem, w[:, h * a.seg : (h + 1) * a.seg],
                          past=st, past_len=h * a.seg, grad=False, k=a.k)
        base = [(torch.cat([kk[:, :, : a.sink], sk], dim=2),
                 torch.cat([vv[:, :, : a.sink], sv], dim=2))
                for (kk, vv), (sk, sv) in zip(full, st)]
        guess = speculate(model, cfg, base, tail[:, :1], cut + a.k, a.probe)

        tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
        tot["lower"] += seg_b_loss(model, cfg, tail, None, cut).item()

        def score_with(idx):
            anc = [(kk[:, :, idx], vv[:, :, idx]) for kk, vv in full]
            mix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                   for (ak, av), (sk, sv) in zip(anc, st)]
            return seg_b_loss(model, cfg, tail, mix, cut + a.k).item()

        uni = query_anchors(model, cfg, full, cut, guess, a.anchors, a.sink, a)
        tot["uniform"] += score_with(uni)

        # recency-weighted: split the budget across log-spaced bands so that
        # each band gets a share proportional to 1/distance, then let the same
        # query pick the best positions inside each band
        sc = torch.full((cut,), float("-inf"), device=dev)
        sc[uni] = 0.0                                    # reuse the ranking
        full_score = query_anchors(model, cfg, full, cut, guess, cut, a.sink, a)
        rank = torch.empty(cut, device=dev)
        rank[full_score] = torch.arange(len(full_score), device=dev).float()
        # Two weightings. 1/d is the hypothesis; "measured" uses the oracle's own
        # density per band, which is the best any distance-based rule could do,
        # since it is read off the answer. If even that loses to uniform, the
        # value of a position simply is not a function of its age.
        if a.weight == "inv":
            want = [1.0 / max(1, (edges[b] + edges[b + 1]) / 2) for b in range(a.bins)]
        else:
            want = [ORACLE_DENSITY[b] for b in range(a.bins)]
        wsum = sum(want)
        chosen = list(range(a.sink))
        for b in range(a.bins):
            quota = int(round((a.anchors - a.sink) * want[b] / wsum))
            lo_d, hi_d = edges[b], edges[b + 1]
            pos = [p for p in range(a.sink, cut) if lo_d < cut - p <= hi_d]
            if not pos or quota <= 0:
                continue
            pos.sort(key=lambda p: rank[p].item())
            chosen.extend(pos[:quota])
        ridx = torch.tensor(sorted(set(chosen)), dtype=torch.long, device=dev)
        tot["recency"] += score_with(ridx)
        del full
        torch.cuda.empty_cache()

    n = len(data)
    print(f"  {'距离':>14} {'该段位置数':>10} {'被复用':>8} {'密度':>8} {'oracle选中':>10} {'相对密度':>9}")
    print("  " + "-" * 68)
    for b in range(a.bins):
        if avail[b] == 0:
            continue
        dens = reuse[b] / avail[b]
        odens = picked[b] / avail[b]
        print(f"  {edges[b]+1:>6}-{edges[b+1]:<7} {avail[b]//n:>10} {reuse[b]:>8} "
              f"{dens:>8.4f} {picked[b]:>10} {odens:>9.4f}")

    for kk in tot:
        tot[kk] /= n
    gap = tot["lower"] - tot["upper"]
    print(f"\n  同样 {a.anchors} 条 anchor：")
    print(f"    均匀挑选      recovery {(tot['lower']-tot['uniform'])/gap:+.1%}")
    lbl = "按 1/距离 加权" if a.weight == "inv" else "按实测密度加权"
    print(f"    {lbl} recovery {(tot['lower']-tot['recency'])/gap:+.1%}")


if __name__ == "__main__":
    main()
