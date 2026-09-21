#!/usr/bin/env python3
"""Which kinds of text does an abstract state actually work on?

The checkpoint was trained on real dialogue, and every number so far is from
that. But the diagnosis said what the state can and cannot hold -- gist yes,
specifics no -- which predicts that text whose value IS its specifics should
compress badly no matter how well the compressor was trained.

System prompts are the extreme case: constraints, tool names, formats, none of
it derivable from what came before. Dialogue is the opposite: mostly derivable,
with occasional hard facts. Code sits in between, being highly structured but
full of exact identifiers.

Run the same compressor over each and compare. A large spread means "what to
compress" is a deployment decision, not a detail.
"""
from __future__ import annotations
import argparse, glob, math
from pathlib import Path
import torch
from memtok import MemoryTokens, compress
from train import seg_b_loss

HERE = Path(__file__).resolve().parent


def load(tok, paths, need, limit):
    out = []
    for p in paths:
        try:
            txt = Path(p).read_text(encoding="utf-8")
        except Exception:
            continue
        ids = tok(txt, return_tensors="pt").input_ids[0]
        for s in range(0, len(ids) - need, need):
            out.append(ids[s : s + need])
            if len(out) >= limit:
                return torch.stack(out)
    return torch.stack(out) if out else torch.empty(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_deep_frozen.pt"))
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    sd = torch.load(a.ckpt); sd = sd["mem"] if "mem" in sd else sd
    mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size,
                       depth=sd["deep"].shape[0] if "deep" in sd else 0).to(dev)
    mem.load_state_dict(sd)

    corpora = {
        "对话（训练域）": sorted(glob.glob(str(HERE / "data" / "*.txt"))),
        "指令/文档": [str(HERE / "README.md"), str(HERE / "README-usage.md"),
                     str(Path.home() / ".claude" / "CLAUDE.md"),
                     str(HERE / "NEXT.md")],
        "代码": sorted(glob.glob(str(HERE / "*.py"))),
    }
    cut = a.seg * a.hops
    print(f"{a.hops} 跳 / {cut} token → {a.k} 槽（{cut//a.k}:1），纯抽象，无 anchor\n")
    print(f"  {'语料':<16} {'窗口':>4} {'upper':>7} {'lower':>7} {'gap':>6} {'recovery':>9}")
    print("  " + "-" * 56)
    for name, paths in corpora.items():
        data = load(tok, paths, cut + a.seg, a.n)
        if len(data) == 0:
            print(f"  {name:<16} 文本不足"); continue
        up = lo = me = 0.0
        for i in range(len(data)):
            w = data[i : i + 1].to(dev)
            ctx, tail = w[:, :cut], w[:, cut : cut + a.seg]
            st = None
            for h in range(a.hops):
                st = compress(model, mem, w[:, h*a.seg:(h+1)*a.seg],
                              past=st, past_len=h*a.seg, grad=False, k=a.k)
            full = [(l.keys, l.values) for l in
                    model.model(input_ids=ctx, use_cache=True).past_key_values.layers]
            up += seg_b_loss(model, cfg, tail, full, cut).item()
            lo += seg_b_loss(model, cfg, tail, None, cut).item()
            me += seg_b_loss(model, cfg, tail, st, cut + a.k).item()
            del full
            torch.cuda.empty_cache()
        n = len(data); up, lo, me = up/n, lo/n, me/n
        print(f"  {name:<16} {n:>4} {up:>7.3f} {lo:>7.3f} {lo-up:>6.3f} "
              f"{(lo-me)/(lo-up):>+8.1%}")


if __name__ == "__main__":
    main()
