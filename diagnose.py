#!/usr/bin/env python3
"""What exactly is the ceiling made of?

Three levers are exhausted -- training, scale and budget all fail to move the
8-hop number past ~67%, and carrying slots over untouched changes nothing. So
the missing third is not something the compressor is failing to do; it is
something an abstract state cannot hold at all.

The candidate: precise recall. The probes in whatis.py already showed this state
rebuilds its own segment at only +16.3% while predicting the next one at +104%
-- it keeps the gist and drops the specifics. Long-context prediction earns part
of its advantage from exactly those specifics: the variable name introduced
2000 tokens ago, the number, the proper noun. Verbatim KV can copy them. A
summary cannot, at any budget.

So split the tail's tokens by whether the answer was available to copy:

  seen    this token appeared somewhere in the preceding context
  novel   it did not

and compare per-token recovery between the groups. If the ceiling is about
precise recall, `seen` tokens are where verbatim KV pulls ahead and the state
cannot follow -- and the gap between the groups is the shape of what no amount
of compression will buy.

The practical reading, if it holds: pair the abstract state with a small set of
exact anchors rather than spending more slots on abstraction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from memtok import MemoryTokens, compress
from train import cache_from, load_chunks

HERE = Path(__file__).resolve().parent


def per_token_ce(model, cfg, ids_b, past, offset):
    """CE for every token of ids_b, rather than the mean seg_b_loss returns."""
    B, S = ids_b.shape
    pos = torch.arange(offset, offset + S, device=ids_b.device)
    kw = {}
    if past is not None:
        kw["past_key_values"] = cache_from(past, cfg)
        plen = past[0][0].shape[2]
        kw["attention_mask"] = torch.ones(B, plen + S, device=ids_b.device, dtype=torch.long)
        kw["cache_position"] = torch.arange(plen, plen + S, device=ids_b.device)
    logits = model(input_ids=ids_b, position_ids=pos.unsqueeze(0).expand(B, -1), **kw).logits
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                           ids_b[:, 1:].reshape(-1), reduction="none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--hops", type=int, default=8)
    ap.add_argument("--n", type=int, default=24)
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

    cut = args.seg * args.hops
    data = load_chunks(tok, cut + args.seg, args.n)
    print(f"{len(data)} windows | {args.hops} hops | k={args.k} "
          f"| {cut // args.k}:1 after the last hop\n")

    # accumulate per-token numbers, grouped by whether the token was copyable
    groups = {"seen": [0.0, 0.0, 0.0, 0], "novel": [0.0, 0.0, 0.0, 0]}
    freq_bucket = {}
    with torch.no_grad():
        for i in range(len(data)):
            w = data[i : i + 1].to(dev)
            ctx, tail = w[:, :cut], w[:, cut : cut + args.seg]

            state = None
            for h in range(args.hops):
                state = compress(model, mem, w[:, h * args.seg : (h + 1) * args.seg],
                                 past=state, past_len=0 if state is None else args.k,
                                 grad=False, k=args.k)
            full = [(l.keys, l.values) for l in model(ctx, use_cache=True).past_key_values.layers]

            up = per_token_ce(model, cfg, tail, full, cut)
            lo = per_token_ce(model, cfg, tail, None, cut)
            me = per_token_ce(model, cfg, tail, state, args.seg + 2 * args.k)

            seen_ids = set(ctx[0].tolist())
            targets = tail[0, 1:].tolist()
            counts = Counter(ctx[0].tolist())
            for j, t in enumerate(targets):
                g = groups["seen" if t in seen_ids else "novel"]
                g[0] += up[j].item(); g[1] += lo[j].item(); g[2] += me[j].item(); g[3] += 1
                # how often it appeared, for the copyable group
                if t in seen_ids:
                    b = "1" if counts[t] == 1 else ("2-5" if counts[t] <= 5 else "6+")
                    e = freq_bucket.setdefault(b, [0.0, 0.0, 0.0, 0])
                    e[0] += up[j].item(); e[1] += lo[j].item()
                    e[2] += me[j].item(); e[3] += 1

    def row(name, g):
        u, l, m, n = g[0] / g[3], g[1] / g[3], g[2] / g[3], g[3]
        gap = l - u
        rec = (l - m) / gap if abs(gap) > 1e-6 else float("nan")
        print(f"  {name:<12} n={n:6d}  upper={u:.3f}  lower={l:.3f}  mem={m:.3f}  "
              f"gap={gap:+.3f}  recovery={rec:+7.1%}")

    tot = [sum(g[i] for g in groups.values()) for i in range(4)]
    print("=== was the answer available to copy from the context? ===")
    row("all", tot)
    row("seen", groups["seen"])
    row("novel", groups["novel"])
    s, nv = groups["seen"], groups["novel"]
    print(f"\n  {s[3]/tot[3]:.1%} of tokens had appeared before; "
          f"they carry {(s[1]-s[0])*s[3]/((tot[1]-tot[0])*tot[3]):.1%} of the total gap\n")

    print("=== among copyable tokens, by how often they appeared ===")
    for b in ("1", "2-5", "6+"):
        if b in freq_bucket:
            row(f"seen x{b}", freq_bucket[b])
    print("\n  a state that keeps the gist should do worst on tokens seen exactly once:")
    print("  those are the specifics -- a name, a number -- that verbatim KV copies")
    print("  and a summary has no way to hold.")


if __name__ == "__main__":
    main()
