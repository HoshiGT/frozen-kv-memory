#!/usr/bin/env python3
"""Train the recursion itself.

Every number so far came from a checkpoint that only ever saw one hop. Rolling
it 8 hops worked (+50.8% at 128:1) but that was zero-shot extrapolation -- the
model was never asked to compress something that had already been compressed.
That is the one part of the architecture nothing has verified.

    seg 1 -> [state]
    seg 2 + [state] -> [state']        same size
    seg 3 + [state'] -> [state'']
    ...                                 predict the tail from the last state

Two things make this trainable on an 8GB card:

  truncated BPTT -- gradient flows through the last `--bptt` hops only, the
  earlier ones run under no_grad. Memory is then constant in hop count while
  the input state is still a genuinely recursive one, which is the part that
  has to be learned.

  random hop count -- each step samples h, so one set of weights covers every
  depth instead of overfitting to one. Same trick as the random budget, which
  turned out to be curriculum learning rather than regularisation.

Position convention, which the earlier evaluation got wrong:

  hop 1    state absent   segment at 0..seg-1     memory at seg..seg+k-1
  hop n>1  state at 0..k-1, segment at k..k+seg-1, memory at k+seg..k+seg+k-1

So the state always sits at the very front, like a resident prefix, and
positions do not accumulate with hop count. The tail must then start right
after the last memory token -- `tail_offset()` below. recur.py instead placed
the tail at seg*hops+k, leaving a gap of thousands of positions between the
state and what it was meant to inform, which only penalised the mem condition.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k-random", default="", metavar="LO,HI")
    ap.add_argument("--hops", default="1,4", metavar="LO,HI",
                    help="sample hop count per step from this range")
    ap.add_argument("--bptt", type=int, default=2,
                    help="how many trailing hops carry gradient; earlier hops run "
                         "under no_grad so memory stays flat in hop count")
    ap.add_argument("--pred-len", type=int, default=0,
                    help="score only this many of the tail's tokens (0 = all). The "
                         "logits alone are seq*vocab*4 bytes -- 311MB for 512 tokens "
                         "on a 151k vocab, which is what puts 1.7B over an 8GB card "
                         "here. Compression ratio is set by seg/k and is untouched.")
    ap.add_argument("--eval-hops", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--keep", type=int, default=0,
                    help="how many slots carry over from the previous state untouched "
                         "instead of being rewritten. 0 = today's behaviour, every slot "
                         "re-encoded every hop. The budget sweep showed capacity is not "
                         "the bottleneck, so what is lost is lost during re-encoding -- "
                         "slots that are never re-encoded cannot lose anything.")
    ap.add_argument("--keep-which", default="old", choices=["old", "new"],
                    help="old: the earliest-written slots stay resident, like an "
                         "attention sink for memory. new: a FIFO window, oldest evicted.")
    ap.add_argument("--eval-only", action="store_true",
                    help="report the init evaluation and stop; for sweeping a loaded "
                         "checkpoint across budgets without touching it")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--mem-depth", action="store_true")
    ap.add_argument("--init", default="", help="start from a single-hop checkpoint")
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DT).to(dev).eval()
    cfg = model.config
    for p in model.parameters():
        p.requires_grad_(False)

    k_lo, k_hi = (args.k, args.k)
    if args.k_random:
        k_lo, k_hi = (int(x) for x in args.k_random.split(","))
    h_lo, h_hi = (int(x) for x in args.hops.split(","))
    eval_hops = [int(x) for x in args.eval_hops.split(",")]
    depth = (cfg.num_hidden_layers - 1) if args.mem_depth else 0

    # a single-hop checkpoint was usually trained with a wider slot allocation
    # (max of k and the random-budget ceiling); adopt its width so it loads
    sd = None
    slots = max(args.k, k_hi)
    if args.init:
        sd = torch.load(args.init)
        sd = sd["mem"] if "mem" in sd else sd
        slots = max(slots, sd["emb"].shape[0])
    mem = MemoryTokens(slots, cfg.hidden_size, depth=depth).to(dev)
    if sd is not None:
        mem.load_state_dict(sd)
        print(f"init from {Path(args.init).name} ({slots} slots)")
    print(f"memory params: {sum(p.numel() for p in mem.parameters())/1e6:.3f}M "
          f"| hops {h_lo}-{h_hi} | bptt {args.bptt} | backbone frozen")

    pred_len = args.pred_len or args.seg
    need = args.seg * (max(h_hi, max(eval_hops)) + 1)
    data = load_chunks(tok, need, args.chunks)
    ev, tr = data[: args.eval_n], data[args.eval_n :]
    print(f"train {len(tr)} / eval {len(ev)} windows of {need} tokens\n")

    def tail_offset(hops: int, k: int) -> int:
        """Where the predicted segment starts, given the convention above."""
        return (k if hops > 1 else 0) + args.seg + k

    def roll(w, hops: int, k: int, grad: bool):
        """Run the recursion; gradient only through the last `bptt` hops.

        With --keep, only k-keep slots are written per hop and the rest are
        carried over from the previous state as-is. The state stays exactly k
        wide either way; what changes is how often a given slot is re-encoded.
        """
        keep = min(args.keep, k - 1)
        state = None
        first_grad = max(0, hops - args.bptt) if grad else hops
        for h in range(hops):
            seg_ids = w[:, h * args.seg : (h + 1) * args.seg]
            past_len = 0 if state is None else k
            k_new = k if state is None else k - keep
            new = compress(model, mem, seg_ids, past=state, past_len=past_len,
                           grad=grad and h >= first_grad, k=k_new)
            if state is None or keep == 0:
                state = new
            else:
                sl = slice(0, keep) if args.keep_which == "old" else slice(k - keep, k)
                state = [(torch.cat([ok[:, :, sl], nk], dim=2),
                          torch.cat([ov[:, :, sl], nv], dim=2))
                         for (ok, ov), (nk, nv) in zip(state, new)]
            if h + 1 == first_grad:      # cut the graph before the trailing hops
                state = [(kk.detach(), vv.detach()) for kk, vv in state]
        return state

    @torch.no_grad()
    def evaluate():
        out = {}
        for hops in eval_hops:
            cut = args.seg * hops
            tot = {"upper": 0.0, "lower": 0.0, "mem": 0.0}
            for i in range(len(ev)):
                w = ev[i : i + 1].to(dev)
                tail = w[:, cut : cut + args.seg]
                full = [(l.keys, l.values) for l in
                        model(w[:, :cut], use_cache=True).past_key_values.layers]
                tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
                tot["lower"] += seg_b_loss(model, cfg, tail, None, cut).item()
                st = roll(w, hops, args.k, grad=False)
                tot["mem"] += seg_b_loss(model, cfg, tail, st,
                                         tail_offset(hops, args.k)).item()
            for kk in tot:
                tot[kk] /= len(ev)
            gap = tot["lower"] - tot["upper"]
            tot["recovery"] = (tot["lower"] - tot["mem"]) / gap if gap > 1e-6 else float("nan")
            tot["ratio"] = cut / args.k
            out[hops] = tot
        return out

    def line(tag, e):
        return f"{tag} | " + "  ".join(
            f"h{h}({e[h]['ratio']:.0f}:1):{e[h]['recovery']:+.0%}" for h in eval_hops)

    e0 = evaluate()
    print(line("[init]", e0) + "\n")
    if args.eval_only:
        for h in eval_hops:
            e = e0[h]
            print(f"  hops={h:2d}  ratio={e['ratio']:5.0f}:1  "
                  f"upper={e['upper']:.4f}  lower={e['lower']:.4f}  "
                  f"mem={e['mem']:.4f}  recovery={e['recovery']:+.1%}")
        return

    opt = torch.optim.AdamW([{"params": list(mem.parameters()), "lr": args.lr}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr], total_steps=args.steps, pct_start=0.1)

    log, t0 = [{"step": 0, **{str(h): e0[h]["recovery"] for h in eval_hops}}], time.time()
    perm, ptr = torch.randperm(len(tr)), 0
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            if ptr + 1 > len(tr):
                perm, ptr = torch.randperm(len(tr)), 0
            w = tr[perm[ptr : ptr + 1]].to(dev); ptr += 1
            hops = random.randint(h_lo, h_hi)
            k = random.randint(k_lo, k_hi) if args.k_random else args.k
            tail = w[:, args.seg * hops : args.seg * hops + pred_len]
            st = roll(w, hops, k, grad=True)
            loss = seg_b_loss(model, cfg, tail, st, tail_offset(hops, k)) / args.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(mem.parameters()), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == args.steps:
            e = evaluate()
            print(line(f"[{step:5d}]", e) + f" | {time.time()-t0:.0f}s")
            log.append({"step": step, **{str(h): e[h]["recovery"] for h in eval_hops}})

    tag = f"recur_k{args.k}_h{h_lo}-{h_hi}" + ("_deep" if args.mem_depth else "")
    (HERE / "out" / f"{tag}.json").write_text(
        json.dumps({"args": vars(args), "log": log}, indent=1))
    torch.save({"mem": mem.state_dict()}, HERE / "ckpt" / f"{tag}.pt")
    print(f"\nsaved out/{tag}.json")


if __name__ == "__main__":
    main()
