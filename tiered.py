#!/usr/bin/env python3
"""Does a coarse memory of the distant past add anything a fine one does not?

Hoshi's observation: human memory holds the recent stretch in some detail, and
further back it thins into an impression -- a diary or a photograph is what
brings the rest back. That suggests tiers rather than one uniform state.

This is NOT the thinning already measured in decay.py. That put *fewer verbatim
anchors* on distant text and found the value of a position barely falls with
age. This puts a *coarser representation* there: the same compressor run at a
much higher ratio over a much longer span.

One warning from persist.py: eight per-session states, 8x the storage, scored
+46.3% against +70.0% for a single rolling state. States of the same granularity
sitting side by side mostly duplicate each other. The claim here is that
different granularities do not -- that "we have been at this for weeks" is not
recoverable from a detailed record of the last few turns.

Conditions, over one long window split into a distant part and a recent part:

  single      one state rolled over everything, today's scheme
  single+     same, with the tier budget, so the comparison is budget-matched
  fine only   the recent part compressed, the distant part discarded entirely
  tiered      coarse state over the distant part + fine state over the recent

tiered vs fine-only is the real question: it isolates what the coarse tier
contributes. tiered vs single+ asks whether splitting the budget beats spending
it all on one state.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from memory import kv_only
from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--hops", type=int, default=16, help="total segments")
    ap.add_argument("--recent", type=int, default=4, help="segments counted as recent")
    ap.add_argument("--k-fine", type=int, default=64)
    ap.add_argument("--k-coarse", type=int, default=16)
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
    split = a.seg * (a.hops - a.recent)          # where "recent" begins
    budget = a.k_fine + a.k_coarse
    data = load_chunks(tok, cut + a.seg, a.n)
    print(f"上下文 {cut} token = 远期 {split} + 近期 {cut-split}")
    print(f"分层预算 {a.k_coarse}(粗,{split//a.k_coarse}:1) + {a.k_fine}"
          f"(细,{(cut-split)//a.k_fine}:1) = {budget} 槽\n")

    def roll(w, start, end, k):
        """Compress w[start:end] into k slots; positions stay absolute."""
        st = None
        for s0 in range(start, end, a.seg):
            st = compress(model, mem, w[:, s0 : s0 + a.seg], past=st,
                          past_len=s0, grad=False, k=k)
        return st

    tot = dict.fromkeys(
        ["upper", "lower", "single", "single_plus", "fine_only", "tiered"], 0.0)
    for i in range(len(data)):
        w = data[i : i + 1].to(dev)
        tail = w[:, cut : cut + a.seg]

        full = kv_only(model, w[:, :cut])
        tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
        tot["lower"] += seg_b_loss(model, cfg, tail, None, cut).item()
        del full

        st1 = roll(w, 0, cut, a.k_fine)
        tot["single"] += seg_b_loss(model, cfg, tail, st1, cut + a.k_fine).item()

        st2 = roll(w, 0, cut, budget)
        tot["single_plus"] += seg_b_loss(model, cfg, tail, st2, cut + budget).item()

        fine = roll(w, split, cut, a.k_fine)
        tot["fine_only"] += seg_b_loss(model, cfg, tail, fine, cut + a.k_fine).item()

        coarse = roll(w, 0, split, a.k_coarse)
        mix = [(torch.cat([ck, fk], dim=2), torch.cat([cv, fv], dim=2))
               for (ck, cv), (fk, fv) in zip(coarse, fine)]
        tot["tiered"] += seg_b_loss(model, cfg, tail, mix, cut + a.k_fine).item()
        torch.cuda.empty_cache()

    n = len(data)
    for kk in tot:
        tot[kk] /= n
    gap = tot["lower"] - tot["upper"]
    print(f"upper CE={tot['upper']:.4f}  lower CE={tot['lower']:.4f}  gap={gap:.4f}\n")
    print(f"  {'条件':<34} {'槽数':>5} {'CE':>8} {'recovery':>10}")
    print("  " + "-" * 62)
    for key, label, slots in (
            ("single", f"单一状态滚过全部", a.k_fine),
            ("single_plus", f"单一状态·同等预算", budget),
            ("fine_only", f"只有近期细状态·远期丢弃", a.k_fine),
            ("tiered", f"分层：粗远期 + 细近期", budget)):
        ce = tot[key]
        print(f"  {label:<34} {slots:>5} {ce:8.4f} {(tot['lower']-ce)/gap:>+9.1%}")
    d = (tot["fine_only"] - tot["tiered"]) / gap
    print(f"\n  粗粒度那层的贡献：{d:+.1%}"
          f"（分层 − 只有近期）")


if __name__ == "__main__":
    main()
