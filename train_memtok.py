#!/usr/bin/env python3
"""Train the model to compress a segment into k memory tokens, by itself.

    upper  : segment B sees all 512 verbatim KV       (nothing lost)
    lower  : segment B sees nothing                   (everything lost)
    mem    : segment B sees only the k memory KV      (what we train)

    recovery = (lower - mem) / (lower - upper)

Mixed-condition training, as before: each step samples which condition the
model is scored under, so it cannot quietly wreck the upper bound to make
recovery look good.

The memory tokens occupy positions seg..seg+k, so segment B has to start at
seg+k or the two collide: giving the state and the live tokens identical RoPE
phases leaves the model no way to tell "compressed past" from "what I am
reading now". That collision alone drove recovery to -31%. The same offset is
applied under every condition, so the comparison stays fair -- upper simply
sees a k-position gap between its verbatim KV and segment B.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--pred-len", type=int, default=0,
                    help="how many of segment B's tokens to score (0 = all). "
                         "Shrinking this saves activation memory without touching "
                         "the compression ratio, which is set by seg/k.")
    ap.add_argument("--k", type=int, default=32, help="memory tokens = the whole state")
    ap.add_argument("--k-curriculum", default="", metavar="START,END",
                    help="squeeze geometrically from START slots down to END over "
                         "training; 16:1 from scratch gives the model no foothold, "
                         "so teach the mechanism at an easy ratio first")
    ap.add_argument("--k-random", default="", metavar="LO,HI",
                    help="sample k per step from [LO,HI]; one set of weights then "
                         "covers every budget instead of overfitting to one ratio")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--ae-steps", type=int, default=0,
                    help="ICAE-style autoencoding warmup: before the downstream "
                         "objective, ask the model to reconstruct the segment "
                         "itself from the memory tokens. Next-token prediction on "
                         "a *later* segment is a weak signal (0.6B needed 200 "
                         "steps just to turn positive); reconstruction demands "
                         "the state actually carry the segment.")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3, help="for the memory embeddings")
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4-quantise the frozen backbone. GGUF's Q4/Q5/Q6 cannot be "
                         "trained through (k-quant blocks have no autograd path); this is "
                         "the trainable equivalent, and it is what lets an 8GB card reach "
                         "8B. Safe in principle because upper/lower/mem are all measured "
                         "on the same quantised model, so recovery is a ratio of errors "
                         "that largely cancel -- which is exactly what this flag tests.")
    ap.add_argument("--mem-depth", action="store_true",
                    help="give the memory slots a per-layer bias of their own (P-tuning v2 "
                         "style). Causality keeps the backbone's behaviour on real tokens "
                         "exactly unchanged; only the state gains capacity.")
    ap.add_argument("--no-lora", action="store_true",
                    help="freeze the backbone completely: the only trainable thing is the "
                         "memory module itself. Tests whether a compression policy can be "
                         "learned from the input side alone, with a reader that never adapts.")
    ap.add_argument("--lora-attn-only", action="store_true",
                    help="skip MLP adapters; their 6144-dim activations are what "
                         "pushes 1.7B over an 8GB card")
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--eval-n", type=int, default=24)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="V100 is Volta: it has fp16 tensor cores but no bf16, so "
                         "renting one means fp16 + loss scaling")
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    dev = "cuda"
    DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        # double quantisation on top: the quantisation constants themselves get
        # quantised, which is most of the difference between NF4 and Q4_0
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=DT, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DT).to(dev)
    cfg = model.config
    for p in model.parameters():
        p.requires_grad_(False)
    if not args.no_lora:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=(["q_proj", "k_proj", "v_proj", "o_proj"] if args.lora_attn_only
                            else ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]),
            task_type="CAUSAL_LM",
        ))
    # fp16 has far less dynamic range than bf16; keep the trained parameters in
    # fp32 and let the adapters cast at the boundary, or Adam updates vanish.
    if DT is torch.float16:
        for prm in model.parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA params: {n_lora/1e6:.2f}M"
          f"{'  (backbone FROZEN)' if args.no_lora else ''} | dtype={args.dtype}")
    model.eval()

    k_lo, k_hi = (args.k, args.k)
    if args.k_random:
        k_lo, k_hi = (int(x) for x in args.k_random.split(","))
    k_start = k_end = args.k
    if args.k_curriculum:
        k_start, k_end = (int(x) for x in args.k_curriculum.split(","))
        k_lo, k_hi = min(k_start, k_end), max(k_start, k_end)
    depth = (cfg.num_hidden_layers - 1) if args.mem_depth else 0
    mem = MemoryTokens(max(args.k, k_hi), cfg.hidden_size, depth=depth).to(dev)
    print(f"memory params: {sum(p.numel() for p in mem.parameters())/1e6:.3f}M"
          f"{f' (deep, {depth} layers)' if depth else ' (flat)'}")

    def k_at(step: int) -> int:
        """Geometric squeeze: equal ratio per step, not equal slot count."""
        if not args.k_curriculum:
            return args.k
        frac = min(1.0, step / max(1, int(args.steps * 0.8)))
        return max(k_end, int(round(k_start * (k_end / k_start) ** frac)))
    pred_len = args.pred_len or args.seg
    data = load_chunks(tok, args.seg, args.chunks)
    ev, tr = data[: args.eval_n], data[args.eval_n :]
    print(f"train {len(tr)} / eval {len(ev)} | seg={args.seg} k={args.k} "
          f"({args.seg / args.k:.0f}:1)")

    scaler = torch.amp.GradScaler("cuda", enabled=(DT is torch.float16))
    groups = [{"params": list(mem.parameters()), "lr": args.lr}]
    peak = [args.lr]
    # an empty param group would make OneCycleLR's max_lr list mismatch, so the
    # frozen-backbone run simply has one group
    if n_lora:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": args.lora_lr})
        peak.append(args.lora_lr)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=peak, total_steps=args.steps, pct_start=0.1)

    def full_kv(ids_a, grad: bool):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = model(ids_a, use_cache=True)
        return [(l.keys, l.values) for l in out.past_key_values.layers]

    # segment B always starts after the space the memory tokens occupy
    def b_offset(k: int) -> int:
        return args.seg + k

    def past_for(mode, ids_a, grad, k: int | None = None):
        if mode == "lower":
            return None
        if mode == "upper":
            return full_kv(ids_a, grad)
        return compress(model, mem, ids_a, grad=grad, k=k)

    @torch.no_grad()
    def evaluate(k_now: int | None = None):
        k_now = args.k if k_now is None else k_now
        tot = {"upper": 0.0, "lower": 0.0, "mem": 0.0}
        for i in range(0, len(ev), args.bs):
            b = ev[i : i + args.bs].to(dev)
            a, bb = b[:, : args.seg], b[:, args.seg : args.seg + pred_len]
            n = b.shape[0]
            for name in tot:
                tot[name] += seg_b_loss(
                    model, cfg, bb, past_for(name, a, False, k_now),
                    b_offset(k_now)).item() * n
        for kk in tot:
            tot[kk] /= len(ev)
        gap = tot["lower"] - tot["upper"]
        tot["gap"] = gap
        tot["recovery"] = (tot["lower"] - tot["mem"]) / gap if gap > 1e-6 else float("nan")
        return tot

    def eval_curve():
        ks = sorted({k_lo, k_hi, args.k, (k_lo + k_hi) // 2}) if args.k_random else [args.k]
        out = {}
        for kk in ks:
            tot, nn_ = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(ev), args.bs):
                    b = ev[i : i + args.bs].to(dev)
                    a, bb = b[:, : args.seg], b[:, args.seg : args.seg + pred_len]
                    tot += seg_b_loss(model, cfg, bb,
                                      past_for("mem", a, False, kk),
                                      b_offset(kk)).item() * b.shape[0]
                    nn_ += b.shape[0]
            out[kk] = tot / nn_
        return out

    e0 = evaluate()
    print(f"\n[init] upper={e0['upper']:.4f} lower={e0['lower']:.4f} mem={e0['mem']:.4f}")
    print(f"       gap={e0['gap']:.4f} nats  recovery={e0['recovery']:+.1%}\n")

    def ae_loss(a, k, grad):
        """Reconstruct segment A from its own memory tokens."""
        state = past_for("mem", a, grad, k)
        return seg_b_loss(model, cfg, a, state, b_offset(k))

    if args.ae_steps:
        print(f"=== autoencoding warmup: {args.ae_steps} steps ===")
        ae_perm, ae_ptr = torch.randperm(len(tr)), 0
        opt.zero_grad(set_to_none=True)
        for astep in range(1, args.ae_steps + 1):
            for _ in range(args.accum):
                if ae_ptr + args.bs > len(tr):
                    ae_perm, ae_ptr = torch.randperm(len(tr)), 0
                b = tr[ae_perm[ae_ptr : ae_ptr + args.bs]].to(dev); ae_ptr += args.bs
                a = b[:, : args.seg]
                kk = random.randint(k_lo, k_hi) if args.k_random else args.k
                scaler.scale(ae_loss(a, kk, True) / args.accum).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if astep % 50 == 0 or astep == args.ae_steps:
                with torch.no_grad():
                    v = sum(ae_loss(ev[i:i+1, :args.seg].to(dev), args.k, False).item()
                            for i in range(min(8, len(ev)))) / min(8, len(ev))
                print(f"[ae {astep:4d}] reconstruct CE={v:.4f} ppl={math.exp(v):.1f}")
        e1 = evaluate()
        print(f"\nafter warmup: recovery={e1['recovery']:+.1%} "
              f"(was {e0['recovery']:+.1%})\n")

    # Mixing in the upper/lower conditions exists to stop LoRA from wrecking the
    # references it is measured against. With the backbone frozen those two
    # references cannot move, so the antidote is unnecessary -- and those steps
    # would carry no trainable parameter at all, since only mem.emb is left.
    if args.no_lora:
        modes, weights = ["mem"], [1]
    else:
        modes, weights = ["upper", "lower", "mem"], [1, 1, 3]
    log, t0 = [{"step": 0, **e0}], time.time()
    perm, ptr = torch.randperm(len(tr)), 0
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            if ptr + args.bs > len(tr):
                perm, ptr = torch.randperm(len(tr)), 0
            b = tr[perm[ptr : ptr + args.bs]].to(dev); ptr += args.bs
            a, bb = b[:, : args.seg], b[:, args.seg : args.seg + pred_len]
            mode = random.choices(modes, weights=weights)[0]
            k_step = (random.randint(k_lo, k_hi) if args.k_random else k_at(step))
            loss = seg_b_loss(model, cfg, bb, past_for(mode, a, True, k_step),
                              b_offset(k_step)) / args.accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == args.steps:
            e = evaluate(k_at(step))
            log.append({"step": step, "k": k_at(step), **e})
            line = (f"[{step:4d}] k={k_at(step):3d} up={e['upper']:.3f} "
                    f"low={e['lower']:.3f} mem={e['mem']:.4f} gap={e['gap']:.3f} "
                    f"recovery={e['recovery']:+.1%}")
            if args.k_random:
                cur = eval_curve()
                line += " | " + " ".join(
                    f"k{kk}:{(e['lower']-v)/e['gap']:+.0%}" for kk, v in cur.items())
            print(line + f" | {time.time()-t0:.0f}s")

    # The corpus belongs in the filename. A memory module is domain-specific
    # (+53.5% on dialogue against -54.4% on novels with one checkpoint), and two
    # domains sharing a name silently destroyed a trained checkpoint once.
    import os
    _dom = os.environ.get("MEMZIP_DATA", "data")
    tag = f"k{args.k}" + ("" if _dom == "data" else f"_{_dom}")
    tag += f"_curr{k_start}-{k_end}" if args.k_curriculum else ""
    if args.mem_depth:
        tag += "_deep"
    if args.load_4bit:
        tag += "_nf4"
    if args.no_lora:
        tag += "_frozen"
    (HERE / "out" / f"memtok_{tag}.json").write_text(
        json.dumps({"args": vars(args), "log": log}, indent=1))
    torch.save({"mem": mem.state_dict()}, HERE / "ckpt" / f"memtok_{tag}.pt")
    if n_lora:
        model.save_pretrained(str(HERE / "ckpt" / f"memtok_{tag}_lora"))
    print(f"\nsaved out/memtok_{tag}.json")


if __name__ == "__main__":
    main()
