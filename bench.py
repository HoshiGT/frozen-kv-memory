#!/usr/bin/env python3
"""Do the numbers survive outside the corpus they were trained on?

Every result so far is on real Claude dialogue -- the same distribution the
memory module was trained on. domain.py already showed the compressor is
strongly domain-sensitive (dialogue +70.7%, docs +54.5%, code +13.6%), which
makes "+86% on our own logs" a weak claim about anything else.

This runs the same conditions on public-domain novels, the same material PG19 is
drawn from, with the checkpoint left exactly as it is -- trained on dialogue,
evaluated on prose. A method that only works in-domain will say so here.

Reported per corpus so the comparison is direct:

  state only        the abstract summary of the recent span, nothing else
  anchors only      retrieved verbatim KV, no summary
  hybrid            both, which is what memory.py does
  oracle            anchors chosen knowing the answer, an upper bound
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

from hybrid import oracle_anchors, query_anchors, speculate
from memory import kv_only
from memtok import MemoryTokens, compress
from train import seg_b_loss

HERE = Path(__file__).resolve().parent


def load(tok, paths, need, limit, skip_head=2000):
    """Windows of `need` tokens, skipping each file's licence boilerplate."""
    out = []
    for p in sorted(paths):
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        ids = tok(txt[skip_head:], return_tensors="pt").input_ids[0]
        step = need * 3                      # spread windows across the book
        for s in range(0, len(ids) - need, step):
            out.append(ids[s : s + need])
            if len(out) >= limit:
                return torch.stack(out)
    return torch.stack(out) if out else torch.empty(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--pred", type=int, default=512)
    ap.add_argument("--seg", type=int, default=1024, help="span that gets summarised")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--probe", type=int, default=16)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--score-mode", default="mean", choices=["mean", "maxhead", "sumsoft"])
    ap.add_argument("--block", type=int, default=1)
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

    corpora = {
        "对话（训练域）": sorted(glob.glob(str(HERE / "data" / "*.txt"))),
        "古腾堡长篇小说": sorted(glob.glob(str(HERE / "data_books" / "*.txt"))),
    }
    print(f"上下文 {a.ctx} | 摘要最后 {a.seg} → {a.k} 槽 | anchor {a.anchors} "
          f"| draft {a.probe}\n")
    print(f"  {'语料':<16} {'窗口':>4} {'gap':>6} {'只有状态':>9} {'只有anchor':>11} "
          f"{'混合':>8} {'oracle':>9}")
    print("  " + "-" * 70)

    for name, paths in corpora.items():
        data = load(tok, paths, a.ctx + a.pred, a.n)
        if len(data) == 0:
            print(f"  {name:<16} 文本不足"); continue
        tot = dict.fromkeys(["upper", "lower", "state", "anc", "hyb", "orc"], 0.0)
        for i in range(len(data)):
            w = data[i : i + 1].to(dev)
            ctx, tail = w[:, : a.ctx], w[:, a.ctx : a.ctx + a.pred]
            full = kv_only(model, ctx)
            tot["upper"] += seg_b_loss(model, cfg, tail, full, a.ctx).item()
            tot["lower"] += seg_b_loss(model, cfg, tail, None, a.ctx).item()

            st = compress(model, mem, w[:, a.ctx - a.seg : a.ctx], past=None,
                          past_len=a.ctx - a.seg, grad=False, k=a.k)
            tot["state"] += seg_b_loss(model, cfg, tail, st, a.ctx + a.k).item()

            sink = [(kk[:, :, : a.sink], vv[:, :, : a.sink]) for kk, vv in full]
            base = [(torch.cat([sk, ak], dim=2), torch.cat([sv, av], dim=2))
                    for (sk, sv), (ak, av) in zip(sink, st)]
            guess = speculate(model, cfg, base, tail[:, :1], a.ctx + a.k, a.probe)

            idx = query_anchors(model, cfg, full, a.ctx, guess, a.anchors, a.sink, a)
            anc = [(kk[:, :, idx], vv[:, :, idx]) for kk, vv in full]
            tot["anc"] += seg_b_loss(model, cfg, tail, anc, a.ctx).item()
            mix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                   for (ak, av), (sk, sv) in zip(anc, st)]
            tot["hyb"] += seg_b_loss(model, cfg, tail, mix, a.ctx + a.k).item()

            oidx = oracle_anchors(ctx, tail, a.anchors, a.sink)
            oanc = [(kk[:, :, oidx], vv[:, :, oidx]) for kk, vv in full]
            omix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                    for (ak, av), (sk, sv) in zip(oanc, st)]
            tot["orc"] += seg_b_loss(model, cfg, tail, omix, a.ctx + a.k).item()
            del full
            torch.cuda.empty_cache()

        n = len(data)
        for kk in tot:
            tot[kk] /= n
        g = tot["lower"] - tot["upper"]
        r = lambda key: (tot["lower"] - tot[key]) / g
        print(f"  {name:<16} {n:>4} {g:>6.3f} {r('state'):>+8.1%} {r('anc'):>+10.1%} "
              f"{r('hyb'):>+7.1%} {r('orc'):>+8.1%}")


if __name__ == "__main__":
    main()
