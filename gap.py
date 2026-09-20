#!/usr/bin/env python3
"""Step 1: measure the headroom a context compressor could possibly recover.

Two bounds, no training involved:

  full   -- every segment sees the whole prefix (upper bound, m = inf)
  none   -- every segment sees only itself  (lower bound, m = 0)

A compressor that squeezes each 512-token segment into m slots lives strictly
between these. If the gap is small the whole idea is moot on this data, so this
runs before anything else gets built.

Reported per position-within-segment, because that curve is the interesting
part: segment-initial tokens have nothing local to lean on and should show the
widest gap, decaying as the segment accumulates its own context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-0.6B"


def token_losses(model, ids: torch.Tensor) -> torch.Tensor:
    """Per-token CE for ids[1:], shape [L-1]. ids: [1, L]"""
    with torch.no_grad():
        logits = model(ids).logits.float()
    return F.cross_entropy(
        logits[0, :-1], ids[0, 1:], reduction="none"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--nseg", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    L = args.seg * args.nseg
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    texts = sorted((HERE / "data").glob("*.txt"))
    ids_all = []
    for p in texts:
        enc = tok(p.read_text(encoding="utf-8"), return_tensors="pt").input_ids[0]
        for s in range(0, len(enc) - L, L):
            ids_all.append(enc[s : s + L])
            if len(ids_all) >= args.chunks:
                break
        if len(ids_all) >= args.chunks:
            break

    print(f"{len(ids_all)} chunks x {L} tokens, seg={args.seg}\n")

    # loss accumulators indexed by position-within-segment
    acc = {"full": torch.zeros(args.seg), "none": torch.zeros(args.seg)}
    cnt = torch.zeros(args.seg)

    for ids in ids_all:
        ids = ids.unsqueeze(0).cuda()

        full = token_losses(model, ids).cpu()  # [L-1], loss for token i+1

        # "none": each segment forwarded alone, no prefix at all
        none = torch.zeros_like(full)
        for k in range(args.nseg):
            a, b = k * args.seg, (k + 1) * args.seg
            seg_loss = token_losses(model, ids[:, a:b]).cpu()  # [seg-1]
            # seg_loss[j] is loss of absolute token a+j+1
            none[a : a + args.seg - 1] = seg_loss
        # token at each segment boundary has no "none" counterpart; mask it
        valid = torch.ones_like(full, dtype=torch.bool)
        for k in range(1, args.nseg):
            valid[k * args.seg - 1] = False

        for k in range(args.nseg):
            if k == 0:
                continue  # segment 0 has no prefix either way -> no gap by construction
            a = k * args.seg
            for j in range(args.seg - 1):
                i = a + j
                if not valid[i]:
                    continue
                acc["full"][j] += full[i]
                acc["none"][j] += none[i]
                cnt[j] += 1

    ok = cnt > 0
    mf = (acc["full"][ok] / cnt[ok])
    mn = (acc["none"][ok] / cnt[ok])
    print(f"mean CE   full={mf.mean():.4f}   none={mn.mean():.4f}   gap={mn.mean()-mf.mean():.4f}")
    print(f"ppl       full={mf.mean().exp():.2f}     none={mn.mean().exp():.2f}")
    print("\npos-in-seg :   full     none      gap")
    for j in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 400, 510]:
        if j < len(mf):
            print(f"  {j:4d}     : {mf[j]:7.4f}  {mn[j]:7.4f}  {mn[j]-mf[j]:7.4f}")

    out = HERE / "out" / "gap.json"
    out.write_text(json.dumps({
        "seg": args.seg, "nseg": args.nseg, "chunks": len(ids_all),
        "full": mf.tolist(), "none": mn.tolist(),
    }))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
