#!/usr/bin/env python3
"""Can memory be stored per session and loaded back together?

Everything so far compressed one continuous stretch in one pass. Real use is not
like that: a companion has N past conversations, each ended at a different time,
and the next one should begin with all of them available. That only works if
each session can be compressed on its own, written to disk, and later
concatenated -- a single rolling state would mean recompressing the entire
history every time anything is added.

The question is whether concatenation survives at all. Each state was produced
at its own positions, believing itself to be the only thing in the cache, so
putting several side by side is a situation none of them was built for.

Three ways to lay them out, measured against rolling one state through
everything (what the experiments have been doing) and against the full cache:

  native    every session keeps the positions it was compressed at; they
            overlap, since each starts from zero
  stacked   sessions are placed end to end, k positions apart, in order
  spread    sessions are spaced as far apart as the original text was, so the
            gaps between memories match the gaps between the conversations

Also reports what it costs: bytes on disk, and the time to load versus the time
to recompute the same KV from scratch.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--sessions", type=int, default=4, help="past conversations")
    ap.add_argument("--hops", type=int, default=2, help="segments per conversation")
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

    sess_len = a.seg * a.hops
    total = sess_len * a.sessions
    data = load_chunks(tok, total + a.seg, a.n)
    print(f"{a.sessions} 段历史 × {sess_len} token = {total}，各自压成 {a.k} 槽 "
          f"（合计 {a.k*a.sessions} 槽，{total//(a.k*a.sessions)}:1）\n")

    def roll(w, start, hops, base_pos, past=None, past_len=0):
        """Compress `hops` segments starting at token `start`.

        `past` lets a new session attend to states already stored, which is the
        difference between compressing it in isolation and compressing it as a
        continuation -- without recomputing anything that came before.
        """
        st = past
        for h in range(hops):
            s0 = start + h * a.seg
            pl = (past_len if st is past and past is not None
                  else base_pos + h * a.seg)
            st = compress(model, mem, w[:, s0 : s0 + a.seg], past=st,
                          past_len=pl, grad=False, k=a.k)
        return st

    def shift(state, delta):
        """Move a state's KV to different positions is NOT possible without
        re-rotating; instead this only records where it will be *told* it sits.
        Kept as a no-op so the layouts differ solely in the offsets used."""
        return state

    tot = dict.fromkeys(["upper", "lower", "rolling", "native", "stacked",
                         "spread", "incremental"], 0.0)
    for i in range(len(data)):
        w = data[i : i + 1].to(dev)
        ctx, tail = w[:, :total], w[:, total : total + a.seg]

        full = [(l.keys, l.values) for l in
                model.model(input_ids=ctx, use_cache=True).past_key_values.layers]
        tot["upper"] += seg_b_loss(model, cfg, tail, full, total).item()
        tot["lower"] += seg_b_loss(model, cfg, tail, None, total).item()

        # one state rolled through the whole history, positions accumulating
        st_roll = roll(w, 0, a.sessions * a.hops, 0)
        tot["rolling"] += seg_b_loss(model, cfg, tail, st_roll,
                                     total + a.k).item()

        # each session compressed alone, believing it starts at position 0
        per = [roll(w, s * sess_len, a.hops, 0) for s in range(a.sessions)]
        cat = [tuple(torch.cat([p[l][j] for p in per], dim=2) for j in range(2))
               for l in range(len(per[0]))]
        tot["native"] += seg_b_loss(model, cfg, tail, cat, sess_len + a.k).item()
        tot["stacked"] += seg_b_loss(model, cfg, tail, cat,
                                     a.k * a.sessions).item()
        tot["spread"] += seg_b_loss(model, cfg, tail, cat, total + a.k).item()

        # incremental: each session is compressed while attending to the states
        # already stored, so it is written as a continuation rather than in
        # isolation -- and nothing earlier is recomputed. This is what adding a
        # conversation to an existing memory would actually do.
        # Only well-defined when a session is a single segment: with more hops
        # the second hop's `past` is just the first hop's memory, which drops
        # every earlier session and leaves the positions inconsistent. That is
        # what the -67.8% reading was measuring.
        inc = []
        for sess in range(a.sessions):
            prev = None
            if inc:
                prev = [tuple(torch.cat([q[l][j] for q in inc], dim=2)
                              for j in range(2)) for l in range(len(inc[0]))]
            st = compress(model, mem, w[:, sess * sess_len : (sess + 1) * sess_len],
                          past=prev, past_len=a.k * len(inc), grad=False, k=a.k)
            inc.append(st)
        cat_inc = [tuple(torch.cat([q[l][j] for q in inc], dim=2) for j in range(2))
                   for l in range(len(inc[0]))]
        tot["incremental"] += seg_b_loss(model, cfg, tail, cat_inc,
                                         a.k * a.sessions + a.seg).item()
        del full
        torch.cuda.empty_cache()

    n = len(data)
    for kk in tot:
        tot[kk] /= n
    gap = tot["lower"] - tot["upper"]
    print(f"upper (全部 {total} KV) CE={tot['upper']:.4f}")
    print(f"lower (什么都没有)      CE={tot['lower']:.4f}   gap={gap:.4f}\n")
    print(f"  {'布局':<28} {'CE':>8} {'recovery':>10}")
    print("  " + "-" * 50)
    for key, label in (("rolling", "单状态滚过全部历史"),
                       ("native", "各自压缩·沿用原位置"),
                       ("stacked", "各自压缩·首尾相接"),
                       ("spread", "各自压缩·按原间隔"),
                       ("incremental", "增量压缩·看得见已存的")):
        ce = tot[key]
        print(f"  {label:<28} {ce:8.4f} {(tot['lower']-ce)/gap:>+9.1%}")

    # what it costs to keep
    path = Path("/tmp/memzip_state.pt")
    st = roll(data[0:1].to(dev), 0, a.sessions * a.hops, 0)
    torch.save([(k.cpu(), v.cpu()) for k, v in st], path)
    size = path.stat().st_size
    t0 = time.time()
    for _ in range(5):
        loaded = torch.load(path)
        loaded = [(k.to(dev), v.to(dev)) for k, v in loaded]
    t_load = (time.time() - t0) / 5
    w = data[0:1].to(dev)
    t0 = time.time()
    for _ in range(5):
        _ = model.model(input_ids=w[:, :total], use_cache=True).past_key_values
    t_calc = (time.time() - t0) / 5
    per_tok = cfg.num_hidden_layers * cfg.num_key_value_heads * getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads) * 2 * 2
    print(f"\n  存一份状态      {size/2**20:6.1f} MB   加载 {t_load*1000:6.1f} ms")
    print(f"  存全量 KV      {total*per_tok/2**20:6.1f} MB   重算 {t_calc*1000:6.1f} ms")
    print(f"  体积比 {total*per_tok/size:.0f}:1")
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
