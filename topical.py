#!/usr/bin/env python3
"""Recall one understanding, not all of them.

persist.py stored a state per session and loaded them all together: eight
states, eight times the storage, and a worse result than one rolling state.
tiered.py added a coarse state for the distant past and it contributed nothing.
Both of those loaded everything they had.

Hoshi's proposal is different in one respect that matters: keep an
understanding per topic, and recall only the one the conversation has arrived
at. Selection instead of concatenation. That is the same move that made anchors
work -- 512 positions chosen by the query beat 512 chosen by rarity by 29
points -- applied one level up, to whole states rather than to single positions.

The attraction is storage. Eight topical states are 56MB where the verbatim KV
they summarise is 448MB.

The reservation is that a recalled abstract state is still an abstract state.
Tokens seen exactly once recover at +46% no matter which state holds them, so
this should restore gist and not specifics. What it might do is replace a large
share of the anchors, which is worth knowing.

Conditions (distant history split into `--blocks` chunks, each compressed alone):

  recent only        just the last segment, distant history discarded
  all blocks         every topical state concatenated -- the thing that failed
  recalled           the top-m states chosen by the same speculative query
  recalled+anchors   those states plus a reduced anchor budget
  anchors only       today's best, for reference
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hybrid import query_anchors, speculate
from memory import kv_only
from memtok import MemoryTokens, compress, _layers
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def score_states(model, cfg, states, query):
    """Attention from the query onto each state's keys; one score per state."""
    dev = query.device
    base = getattr(model, "model", model)
    h = base(input_ids=query, output_hidden_states=True).hidden_states[-2][0]
    layer = _layers(model)[-1]
    attn = layer.self_attn
    q = attn.q_proj(layer.input_layernorm(h))
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    q = q.view(-1, cfg.num_attention_heads, hd)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    out = []
    for st in states:
        k = st[-1][0][0]
        k = k.repeat_interleave(cfg.num_attention_heads // k.shape[0], dim=0)
        s = torch.einsum("thd,hcd->", q.float(), k.float())
        out.append((s / (len(q) * k.shape[1] * hd ** 0.5)).item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--hops", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=5, help="topical chunks of the past")
    ap.add_argument("--recall", type=int, default=2, help="how many to bring back")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--anchors-reduced", type=int, default=128)
    ap.add_argument("--probe", type=int, default=128)
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

    cut = a.seg * a.hops
    past_len_tok = cut - a.seg                       # everything but the last segment
    blk = past_len_tok // a.blocks
    data = load_chunks(tok, cut + a.seg, a.n)
    print(f"上下文 {cut} | 远期 {past_len_tok} 切成 {a.blocks} 块（每块 {blk}）")
    print(f"每块压成 {a.k} 槽 = {blk//a.k}:1 | 召回 {a.recall} 块\n")

    def roll(w, start, end, k):
        st = None
        for s0 in range(start, end, a.seg):
            st = compress(model, mem, w[:, s0 : min(s0 + a.seg, end)], past=st,
                          past_len=s0, grad=False, k=k)
        return st

    def cat(states):
        return [tuple(torch.cat([s[l][j] for s in states], dim=2) for j in range(2))
                for l in range(len(states[0]))]

    names = ["upper", "lower", "recent", "all_blocks", "recalled",
             "recalled_anchors", "anchors_only"]
    tot = dict.fromkeys(names, 0.0)
    hit = [0, 0]
    for i in range(len(data)):
        w = data[i : i + 1].to(dev)
        tail = w[:, cut : cut + a.seg]
        full = kv_only(model, w[:, :cut])
        tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
        tot["lower"] += seg_b_loss(model, cfg, tail, None, cut).item()

        recent = compress(model, mem, w[:, cut - a.seg : cut], past=None,
                          past_len=cut - a.seg, grad=False, k=a.k)
        tot["recent"] += seg_b_loss(model, cfg, tail, recent, cut + a.k).item()

        blocks = [roll(w, b * blk, (b + 1) * blk, a.k) for b in range(a.blocks)]
        tot["all_blocks"] += seg_b_loss(
            model, cfg, tail, cat(blocks + [recent]), cut + a.k).item()

        # the query: what the recent state thinks is coming
        sink = [(kk[:, :, : a.sink], vv[:, :, : a.sink]) for kk, vv in full]
        base = [(torch.cat([sk, ak], dim=2), torch.cat([sv, av], dim=2))
                for (sk, sv), (ak, av) in zip(sink, recent)]
        guess = speculate(model, cfg, base, tail[:, :1], cut + a.k, a.probe)

        sc = score_states(model, cfg, blocks, guess)
        top = sorted(range(a.blocks), key=lambda b: -sc[b])[: a.recall]
        hit[0] += 1 if (a.blocks - 1) in top else 0     # did it pick the latest block
        hit[1] += 1
        tot["recalled"] += seg_b_loss(
            model, cfg, tail, cat([blocks[b] for b in sorted(top)] + [recent]),
            cut + a.k).item()

        idx = query_anchors(model, cfg, full, cut, guess, a.anchors, a.sink, a)
        anc = [(kk[:, :, idx], vv[:, :, idx]) for kk, vv in full]
        tot["anchors_only"] += seg_b_loss(
            model, cfg, tail, cat([anc, recent]), cut + a.k).item()

        ridx = query_anchors(model, cfg, full, cut, guess, a.anchors_reduced, a.sink, a)
        ranc = [(kk[:, :, ridx], vv[:, :, ridx]) for kk, vv in full]
        tot["recalled_anchors"] += seg_b_loss(
            model, cfg, tail,
            cat([ranc] + [blocks[b] for b in sorted(top)] + [recent]),
            cut + a.k).item()
        del full
        torch.cuda.empty_cache()

    n = len(data)
    for kk in tot:
        tot[kk] /= n
    gap = tot["lower"] - tot["upper"]
    st_mb = a.k * cfg.num_hidden_layers * cfg.num_key_value_heads * getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads) * 2 * 2 / 2**20
    print(f"upper CE={tot['upper']:.4f}  lower CE={tot['lower']:.4f}  gap={gap:.4f}")
    print(f"一个状态 {st_mb:.1f} MB | 全部 {a.blocks} 块 {st_mb*a.blocks:.0f} MB "
          f"| 原文 KV {st_mb*cut/a.k:.0f} MB\n")
    print(f"  {'条件':<32} {'存储':>8} {'CE':>8} {'recovery':>10}")
    print("  " + "-" * 62)
    for key, label, store in (
            ("recent", "只有最近一段", st_mb),
            ("all_blocks", f"全部 {a.blocks} 块都加载", st_mb * (a.blocks + 1)),
            ("recalled", f"按语境召回 {a.recall} 块", st_mb * (a.blocks + 1)),
            ("recalled_anchors", f"召回 {a.recall} 块 + {a.anchors_reduced} anchor",
             st_mb * (a.blocks + 1) + st_mb * a.anchors_reduced / a.k),
            ("anchors_only", f"{a.anchors} anchor（今日最佳）",
             st_mb + st_mb * cut / a.k)):
        ce = tot[key]
        print(f"  {label:<32} {store:>7.0f}M {ce:8.4f} {(tot['lower']-ce)/gap:>+9.1%}")
    print(f"\n  召回时选中最后一块的比例：{hit[0]/hit[1]:.0%}"
          f"（{a.recall}/{a.blocks} 随机命中率 {a.recall/a.blocks:.0%}）")


if __name__ == "__main__":
    main()
