#!/usr/bin/env python3
"""Does h carry information at all, or does the reader just fail to use it?

Two very different failures look identical from the loss curve:

  A. the GRU collapses -- every segment produces nearly the same h, so there is
     nothing to read. Diagnosed by pairwise cosine between states: if it sits
     near 1.0, h is a constant with noise on top.

  B. h is informative but the read-out/attention path cannot deliver it.
     Diagnosed by retrieval: freeze h, train a linear map against the segment's
     own mean hidden state, and see whether h_i can pick A_i out of a lineup.

Retrieval well above chance plus flat recovery means the bottleneck is the
interface, not the memory -- a completely different thing to go fix.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from state import StateMemory
from train import load_chunks

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--d-state", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config

    mem = StateMemory(cfg.hidden_size, cfg.num_hidden_layers, cfg.num_key_value_heads,
                      cfg.head_dim, m=args.m, d_state=args.d_state, chunk=args.chunk).to(dev)
    ck = Path(args.ckpt) if args.ckpt else HERE / "ckpt" / f"state_w64_m{args.m}_d{args.d_state}.pt"
    if ck.exists():
        mem.load_state_dict(torch.load(ck)); print(f"loaded {ck.name}")
    else:
        print("!! no checkpoint, probing an untrained state (this is the control)")
    mem.eval()

    data = load_chunks(tok, args.seg, args.n)
    H, T = [], []
    with torch.no_grad():
        for i in range(0, len(data), 4):
            a = data[i : i + 4, : args.seg].to(dev)
            hs = model(a, output_hidden_states=True).hidden_states[-1]
            H.append(mem.absorb(hs).float().cpu())
            T.append(hs.float().mean(dim=1).cpu())
    H, T = torch.cat(H), torch.cat(T)
    n = len(H)
    print(f"\n{n} segments, h dim {H.shape[1]}")

    # --- A. collapse check -------------------------------------------------
    Hn = F.normalize(H, dim=1)
    sim = Hn @ Hn.T
    off = sim[~torch.eye(n, dtype=torch.bool)]
    print(f"\n[collapse] pairwise cosine between states:")
    print(f"  mean {off.mean():.4f}  p10 {off.quantile(0.1):.4f}  p90 {off.quantile(0.9):.4f}")
    print(f"  per-dim std across segments: {H.std(dim=0).mean():.4f}")
    if off.mean() > 0.95:
        print("  -> COLLAPSED: h is essentially constant, nothing was encoded")

    # --- B. retrieval probe ------------------------------------------------
    split = n // 2
    W = torch.nn.Linear(H.shape[1], T.shape[1]).cuda()
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3)
    Htr, Ttr = H[:split].cuda(), T[:split].cuda()
    Hte, Tte = H[split:].cuda(), T[split:].cuda()
    for _ in range(600):
        logits = F.normalize(W(Htr), dim=1) @ F.normalize(Ttr, dim=1).T * 10
        loss = F.cross_entropy(logits, torch.arange(split, device="cuda"))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        lg = F.normalize(W(Hte), dim=1) @ F.normalize(Tte, dim=1).T
        k = len(Hte)
        top1 = (lg.argmax(dim=1) == torch.arange(k, device="cuda")).float().mean().item()
        rank = (lg > lg.diag().unsqueeze(1)).sum(dim=1).float().mean().item() + 1
    print(f"\n[retrieval] held-out {k}-way: top1 {top1:.1%} (chance {1/k:.1%}), "
          f"mean rank {rank:.1f}/{k}")
    if top1 > 3 / k:
        print("  -> h IS informative; if recovery is still flat the interface is the problem")


if __name__ == "__main__":
    main()
