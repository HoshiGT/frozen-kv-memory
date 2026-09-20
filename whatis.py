#!/usr/bin/env python3
"""Is the state a compressed copy of the text, or something else?

The whole point was never to store tokens. But memory tokens do occupy k
sequence positions with their own RoPE phases, so "it is just token compression
wearing a hat" is a fair accusation and deserves a direct answer rather than the
indirect evidence (recovery > 100%, AE failing, recursion converging).

Three probes, each of which a token-compressor would fail differently:

  1. decode   -- push the memory positions' hidden states through lm_head.
                 If the slots hold text, the top tokens are words from the
                 segment. If they hold something else, they decode to noise.
  2. nearest  -- cosine between each memory slot's KV and every real position's
                 KV, per layer. A selection/pruning scheme (which is what v1
                 was) has each slot sitting on top of some source position.
                 A synthesised state sits nowhere near any of them.
  3. rebuild  -- reconstruct the segment from its own memory, scored against
                 predicting it with no context at all. This is the AE objective
                 measured, not trained.

Run against whatever checkpoint trained downstream-only.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from memtok import MemoryTokens, compress, embed_tokens, deep_prefix
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_deep_frozen.pt"))
    ap.add_argument("--mem-depth", action="store_true", default=True)
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config

    sd = torch.load(args.ckpt)
    sd = sd["mem"] if "mem" in sd else sd
    slots = sd["emb"].shape[0]
    depth = sd["deep"].shape[0] if "deep" in sd else 0
    mem = MemoryTokens(slots, cfg.hidden_size, depth=depth).to(dev)
    mem.load_state_dict(sd)
    print(f"{Path(args.ckpt).name}: {slots} slots, depth={depth}, using k={args.k}\n")

    data = load_chunks(tok, args.seg * 2, args.n)

    # ---------- 1. decode the memory positions ----------
    w = data[0:1, : args.seg].to(dev)
    with torch.no_grad():
        x = mem.append(embed_tokens(model, w), args.k)
        pos = torch.arange(0, args.seg + args.k, device=dev)
        with deep_prefix(model, mem, args.k):
            out = model(inputs_embeds=x, position_ids=pos.unsqueeze(0),
                        use_cache=True, output_hidden_states=True)
        h_mem = out.hidden_states[-1][0, -args.k:]            # [k, d]
        logits = model.lm_head(model.model.norm(h_mem))       # same head the model uses
        top = logits.topk(4, dim=-1).indices

    seg_ids = set(w[0].tolist())
    hits = sum(1 for row in top for t in row.tolist() if t in seg_ids)
    print("=== 1. what do the slots decode to ===")
    for i in range(min(6, args.k)):
        words = " / ".join(repr(tok.decode([t]))[1:-1] for t in top[i].tolist())
        print(f"  slot {i:2d}: {words}")
    print(f"  ... top-4 tokens landing inside the source segment: "
          f"{hits}/{args.k*4} ({hits/(args.k*4):.1%})")
    print(f"  (chance level for this segment: {len(seg_ids)/cfg.vocab_size:.2%})\n")

    # ---------- 2. is each slot sitting on a source position ----------
    with torch.no_grad():
        full = model(w, use_cache=True).past_key_values.layers
        state = compress(model, mem, w, grad=False, k=args.k)
    # Controls, without which the raw number means nothing: key vectors may simply
    # all live in one narrow cone, in which case any vector scores high.
    #   other  : the same slots compressing a DIFFERENT segment, matched against
    #            THIS segment's keys -- content-free similarity
    #   shuffle: this segment's own keys with their head assignment permuted
    other_seg = data[1:2, : args.seg].to(dev)
    with torch.no_grad():
        other = compress(model, mem, other_seg, grad=False, k=args.k)
    print("=== 2. cosine to the nearest real position (key vectors) ===")
    print(f"  {'layer':>6} {'this seg':>9} {'other seg':>11} {'random':>9}  {'excess':>8}")
    for li in (0, cfg.num_hidden_layers // 2, cfg.num_hidden_layers - 1):
        src = F.normalize(full[li].keys[0].transpose(0, 1).reshape(args.seg, -1).float(), dim=-1)
        def nearest(v):
            return (F.normalize(v.float(), dim=-1) @ src.T).max(dim=-1).values.mean().item()
        mk = state[li][0][0].transpose(0, 1).reshape(args.k, -1)
        ok = other[li][0][0].transpose(0, 1).reshape(args.k, -1)
        rnd = torch.randn_like(mk.float()) * mk.float().std()
        a, b, c = nearest(mk), nearest(ok), nearest(rnd)
        print(f"  {li:6d} {a:9.3f} {b:11.3f} {c:9.3f}  {a-b:+8.3f}")
    print("  this seg vs other seg is the real signal: how much of the similarity")
    print("  is about THIS content rather than about where keys live in general.")
    print("  (a pruning scheme would show ~1.0 here and a large positive excess)\n")

    # ---------- 3. can the segment be rebuilt from its own memory ----------
    tot_mem = tot_none = 0.0
    with torch.no_grad():
        for i in range(args.n):
            a = data[i : i + 1, : args.seg].to(dev)
            st = compress(model, mem, a, grad=False, k=args.k)
            tot_mem += seg_b_loss(model, cfg, a, st, args.seg + args.k).item()
            tot_none += seg_b_loss(model, cfg, a, None, args.seg + args.k).item()
    m, n = tot_mem / args.n, tot_none / args.n
    print("=== 3. rebuilding the segment from its own memory ===")
    print(f"  from memory : CE={m:.4f} ppl={math.exp(m):7.2f}")
    print(f"  no context  : CE={n:.4f} ppl={math.exp(n):7.2f}")
    print(f"  the memory buys back {(n-m)/n:+.1%} of the reconstruction loss")
    print("  (a compressed copy of the text would rebuild it far better than this)")


if __name__ == "__main__":
    main()
