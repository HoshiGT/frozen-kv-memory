#!/usr/bin/env python3
"""How long should the one summarised segment be?

tiered.py compared keeping the last N segments, but those were rolled: compress
512, then compress the next 512 with the first as past. Rolling turned out to
dilute, so "the last 1024 tokens" was never actually measured -- only "512
rolled into 512". This compresses a span of the chosen length in one pass.

If a longer span holds up, the effective window widens for free: same slots,
same compute, more of the recent past summarised.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from memory import kv_only
from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--pred", type=int, default=512)
ap.add_argument("--k", type=int, default=64)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_deep_frozen.pt"))
ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda"; torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
cfg = model.config
sd = torch.load(a.ckpt); sd = sd["mem"] if "mem" in sd else sd
mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size,
                   depth=sd["deep"].shape[0] if "deep" in sd else 0).to(dev)
mem.load_state_dict(sd)

data = load_chunks(tok, a.ctx + a.pred, a.n)
spans = [256, 512, 1024, 2048]
tot = {s: 0.0 for s in spans}; tot["upper"] = tot["lower"] = 0.0
for i in range(len(data)):
    w = data[i:i+1].to(dev)
    tail = w[:, a.ctx : a.ctx + a.pred]
    full = kv_only(model, w[:, :a.ctx])
    tot["upper"] += seg_b_loss(model, cfg, tail, full, a.ctx).item()
    tot["lower"] += seg_b_loss(model, cfg, tail, None, a.ctx).item()
    del full
    for s in spans:
        st = compress(model, mem, w[:, a.ctx - s : a.ctx], past=None,
                      past_len=a.ctx - s, grad=False, k=a.k)
        tot[s] += seg_b_loss(model, cfg, tail, st, a.ctx + a.k).item()
    torch.cuda.empty_cache()
n = len(data)
for kk in tot: tot[kk] /= n
gap = tot["lower"] - tot["upper"]
print(f"上下文 {a.ctx} | {a.k} 槽 | 一次性压缩（无滚动）\n")
print(f"  {'压缩跨度':>10} {'压缩比':>8} {'CE':>8} {'recovery':>10}")
print("  " + "-" * 42)
for s in spans:
    print(f"  {s:>10} {s//a.k:>7}:1 {tot[s]:8.4f} {(tot['lower']-tot[s])/gap:>+9.1%}")
