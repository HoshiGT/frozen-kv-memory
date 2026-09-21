#!/usr/bin/env python3
"""Train the abstract state knowing that anchors will be there too.

Every checkpoint so far was trained alone: nothing but k slots stood between the
context and the prediction, so the slots had to try to cover everything --
including the specifics that diagnose.py showed they cannot hold anyway (+46%
on tokens seen once) and that an anchor supplies exactly.

Put the anchors in the past during training and that effort has nowhere useful
to go. The gradient through the slots only rewards what the anchors did not
already provide, which is the division of labour the diagnosis implies:

    anchors   the specifics, verbatim, retrieved       seen x1 tokens: +46% alone
    slots     the gist, what is coming, the shape      novel tokens:  +231% alone

Anchors enter as detached original KV, so nothing flows into them; they are a
fixed fact of the environment the slots are learning to complement. Selection is
by surprisal, which needs one forward pass and no query -- speculative drafting
is better at eval time but far too slow to run inside a training step.

The honest null result, worth recording if it comes out that way: the loss
already avoids spending capacity on what the anchors cover, in which case the
division of labour is emergent and needs no special training.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from hybrid import surprisal_anchors
from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--k", type=int, default=64, help="abstract slots")
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--hops", default="2,4", metavar="LO,HI",
                    help="hops sampled per step; training long is expensive because "
                         "the anchors need a full-context forward each step")
    ap.add_argument("--eval-hops", default="4,8")
    ap.add_argument("--bptt", type=int, default=2)
    ap.add_argument("--pred-len", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--mem-depth", action="store_true")
    ap.add_argument("--init", default=str(HERE / "ckpt" / "memtok_k32_deep_frozen.pt"))
    ap.add_argument("--no-anchors-in-training", action="store_true",
                    help="control: same schedule, anchors absent while training but "
                         "present at evaluation. Isolates whether the gain comes from "
                         "training with anchors or merely from more training.")
    ap.add_argument("--chunks", type=int, default=160)
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    for p in model.parameters():
        p.requires_grad_(False)

    h_lo, h_hi = (int(x) for x in args.hops.split(","))
    eval_hops = [int(x) for x in args.eval_hops.split(",")]
    depth = (cfg.num_hidden_layers - 1) if args.mem_depth else 0

    sd = torch.load(args.init)
    sd = sd["mem"] if "mem" in sd else sd
    slots = max(args.k, sd["emb"].shape[0])
    if "deep" in sd:
        depth = sd["deep"].shape[0]
    mem = MemoryTokens(slots, cfg.hidden_size, depth=depth).to(dev)
    mem.load_state_dict(sd)
    print(f"init {Path(args.init).name} ({slots} slots, depth={depth}) | k={args.k} "
          f"| anchors={args.anchors} | backbone frozen")

    pred_len = args.pred_len or args.seg
    need = args.seg * (max(h_hi, max(eval_hops)) + 1)
    data = load_chunks(tok, need, args.chunks)
    ev, tr = data[: args.eval_n], data[args.eval_n :]
    print(f"train {len(tr)} / eval {len(ev)} windows of {need} tokens\n")

    def roll(w, hops, grad):
        """Positions accumulate: measured 4 points better than resetting them."""
        state = None
        first_grad = max(0, hops - args.bptt) if grad else hops
        for h in range(hops):
            state = compress(model, mem, w[:, h * args.seg : (h + 1) * args.seg],
                             past=state, past_len=h * args.seg,
                             grad=grad and h >= first_grad, k=args.k)
            if h + 1 == first_grad:
                state = [(kk.detach(), vv.detach()) for kk, vv in state]
        return state

    def anchors_for(ctx):
        """Detached original KV at the most surprising positions, plus the sink."""
        with torch.no_grad():
            full = [(l.keys, l.values) for l in
                    model(ctx, use_cache=True).past_key_values.layers]
            idx = surprisal_anchors(model, cfg, ctx, args.anchors, args.sink)
        return [(kk[:, :, idx], vv[:, :, idx]) for kk, vv in full], full

    def mix(anc, state):
        return [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                for (ak, av), (sk, sv) in zip(anc, state)]

    @torch.no_grad()
    def evaluate():
        out = {}
        for hops in eval_hops:
            cut = args.seg * hops
            tot = {"upper": 0.0, "lower": 0.0, "solo": 0.0, "hybrid": 0.0}
            for i in range(len(ev)):
                w = ev[i : i + 1].to(dev)
                ctx, tail = w[:, :cut], w[:, cut : cut + pred_len]
                anc, full = anchors_for(ctx)
                st = roll(w, hops, False)
                tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
                tot["lower"] += seg_b_loss(model, cfg, tail, None, cut).item()
                tot["solo"] += seg_b_loss(model, cfg, tail, st, cut + args.k).item()
                tot["hybrid"] += seg_b_loss(model, cfg, tail, mix(anc, st),
                                            cut + args.k).item()
            for kk in tot:
                tot[kk] /= len(ev)
            g = tot["lower"] - tot["upper"]
            out[hops] = {"solo": (tot["lower"] - tot["solo"]) / g,
                         "hybrid": (tot["lower"] - tot["hybrid"]) / g}
        return out

    def line(tag, e):
        return f"{tag} | " + "  ".join(
            f"h{h}: solo {e[h]['solo']:+.0%} hybrid {e[h]['hybrid']:+.0%}"
            for h in eval_hops)

    e0 = evaluate()
    print(line("[init]", e0) + "\n")

    opt = torch.optim.AdamW([{"params": list(mem.parameters()), "lr": args.lr}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr], total_steps=args.steps, pct_start=0.1)

    log, t0 = [{"step": 0, **{str(h): e0[h] for h in eval_hops}}], time.time()
    perm, ptr = torch.randperm(len(tr)), 0
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            if ptr + 1 > len(tr):
                perm, ptr = torch.randperm(len(tr)), 0
            w = tr[perm[ptr : ptr + 1]].to(dev); ptr += 1
            hops = random.randint(h_lo, h_hi)
            cut = args.seg * hops
            ctx = w[:, :cut]
            tail = w[:, cut : cut + pred_len]
            st = roll(w, hops, True)
            if args.no_anchors_in_training:
                past = st
            else:
                anc, _ = anchors_for(ctx)
                past = mix(anc, st)
            loss = seg_b_loss(model, cfg, tail, past, cut + args.k) / args.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(mem.parameters()), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == args.steps:
            e = evaluate()
            print(line(f"[{step:4d}]", e) + f" | {time.time()-t0:.0f}s")
            log.append({"step": step, **{str(h): e[h] for h in eval_hops}})

    tag = f"hybrid_k{args.k}_a{args.anchors}" + (
        "_noanc" if args.no_anchors_in_training else "")
    (HERE / "out" / f"{tag}.json").write_text(
        json.dumps({"args": vars(args), "log": log}, indent=1))
    torch.save({"mem": mem.state_dict()}, HERE / "ckpt" / f"{tag}.pt")
    print(f"\nsaved out/{tag}.json")


if __name__ == "__main__":
    main()
