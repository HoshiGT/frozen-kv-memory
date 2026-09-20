#!/usr/bin/env python3
"""Train an abstract state jointly with the model that has to read it.

    [h: fixed-size abstract state] + [recent W tokens of verbatim KV] + [now]

Total KV stays constant however long the conversation runs. Removing the linear
growth is the point -- not compressing one segment in isolation.

Why the model is trained too
----------------------------
A frozen model has only ever seen KV produced by real tokens. The slots read
out of h are synthesised vectors with no reason to land anywhere near that
distribution, so asking a frozen reader to understand them is asking it to
interpret noise. Encoding and decoding have to co-evolve -- one model, one
LoRA, learning to write the state and to read it back. Gradients flow through
the encoding pass as well, so the model can also learn to produce hidden states
that are easy to compress.

The mixed-condition trap
------------------------
If training only ever shows "window + h", the model quietly specialises to that
and degrades on full KV. The upper bound drops, the gap shrinks, and recovery
looks great for the wrong reason. So each step samples a condition at random
and the model must stay good at all three.
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

from state import StateMemory, StateInjector, attach_injector
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--inject", default="gated", choices=["gated", "slots"],
                    help="gated = residual-stream modulation; slots = pseudo-KV (v1 route)")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--gate-lr-mult", type=float, default=30.0)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--d-state", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-lora", action="store_true", help="ablation: frozen reader")
    ap.add_argument("--grad-encode", type=int, default=1,
                    help="1 = let gradients shape the encoding pass too")
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
    cfg = model.config
    for p in model.parameters():
        p.requires_grad_(False)

    if not args.no_lora:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ))
        n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA params: {n_lora/1e6:.2f}M")
    model.eval()   # no dropout; LoRA params still train

    data = load_chunks(tok, args.seg, args.chunks)
    ev, tr = data[: args.eval_n], data[args.eval_n :]
    print(f"train {len(tr)} / eval {len(ev)} | seg={args.seg} window={args.window} "
          f"m={args.m} d_state={args.d_state}")

    mem = StateMemory(cfg.hidden_size, cfg.num_hidden_layers, cfg.num_key_value_heads,
                      cfg.head_dim, m=args.m, d_state=args.d_state, chunk=args.chunk).to(dev)
    inj = StateInjector(args.d_state, cfg.hidden_size, cfg.num_hidden_layers,
                        rank=args.rank).to(dev)
    holder: dict = {"deltas": None}
    handles = attach_injector(model, holder)
    print(f"state params: {sum(p.numel() for p in mem.parameters())/1e6:.2f}M "
          f"+ injector {sum(p.numel() for p in inj.parameters())/1e6:.2f}M "
          f"(inject={args.inject})")

    # The gate is a single scalar per layer, so its gradient is tiny next to the
    # matrices'; on a shared lr it creeps (0.0035 -> 0.0064 over 300 steps) and
    # the injection never reaches a scale that matters. Give it its own faster
    # group so it can actually open.
    gate_params = [inj.gate]
    body = list(mem.parameters()) + [p for n, p in inj.named_parameters() if n != "gate"]
    groups = [{"params": body, "lr": args.lr},
              {"params": gate_params, "lr": args.lr * args.gate_lr_mult}]
    if not args.no_lora:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": args.lora_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps, pct_start=0.1)

    win_idx = torch.cat([torch.arange(args.sink),
                         torch.arange(args.seg - args.window, args.seg)]).to(dev)

    def encode_a(ids_a, grad: bool):
        holder["deltas"] = None          # encoding is never modulated
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = model(ids_a, use_cache=True, output_hidden_states=True)
        return ([(l.keys, l.values) for l in out.past_key_values.layers],
                out.hidden_states[-1])

    def build(pairs, hs, mode: str):
        """Returns the past to attend over; may also arm the residual injector."""
        holder["deltas"] = None
        if mode == "full":
            return pairs
        wp = [(k[:, :, win_idx], v[:, :, win_idx]) for k, v in pairs]
        if mode == "window":
            return wp
        h = mem.absorb(hs)
        if args.inject == "gated":
            holder["deltas"] = inj.deltas(h, pairs[0][0].dtype)
            return wp
        slots = mem.read(h, pairs[0][0].dtype)
        return [(torch.cat([k, sk], dim=2), torch.cat([v, sv], dim=2))
                for (k, v), (sk, sv) in zip(wp, slots)]

    @torch.no_grad()
    def evaluate():
        tot = {"upper": 0.0, "window": 0.0, "state": 0.0}
        for i in range(0, len(ev), args.bs):
            b = ev[i : i + args.bs].to(dev)
            a, bb = b[:, : args.seg], b[:, args.seg :]
            pairs, hs = encode_a(a, grad=False)
            n = b.shape[0]
            for name, mode in [("upper", "full"), ("window", "window"), ("state", "state")]:
                tot[name] += seg_b_loss(model, cfg, bb, build(pairs, hs, mode), args.seg).item() * n
        for k in tot:
            tot[k] /= len(ev)
        gap = tot["window"] - tot["upper"]
        tot["gap"] = gap
        tot["recovery"] = (tot["window"] - tot["state"]) / gap if abs(gap) > 1e-6 else float("nan")
        return tot

    # calibrate the read-out against genuine cache statistics before step 1
    with torch.no_grad():
        cal = tr[:4].to(dev)[:, : args.seg]
        cpairs, _ = encode_a(cal, grad=False)
        mem.calibrate(cpairs)
    print("calibrated K/V targets: "
          f"K {mem.k_target[0]:.2f}->{mem.k_target[-1]:.2f}  "
          f"V {mem.v_target[0]:.2f}->{mem.v_target[-1]:.2f}")

    e0 = evaluate()
    print(f"\n[init] upper={e0['upper']:.4f} window={e0['window']:.4f} state={e0['state']:.4f}")
    print(f"       residual gap after window = {e0['gap']:.4f} nats "
          f"(ppl {math.exp(e0['window']):.2f} -> {math.exp(e0['upper']):.2f})")
    print(f"       recovery={e0['recovery']:+.1%}\n")

    modes, weights = ["full", "window", "state"], [1, 1, 2]
    log, t0 = [{"step": 0, **e0}], time.time()
    perm, ptr = torch.randperm(len(tr)), 0
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            if ptr + args.bs > len(tr):
                perm, ptr = torch.randperm(len(tr)), 0
            b = tr[perm[ptr : ptr + args.bs]].to(dev); ptr += args.bs
            a, bb = b[:, : args.seg], b[:, args.seg :]
            mode = random.choices(modes, weights=weights)[0]
            pairs, hs = encode_a(a, grad=bool(args.grad_encode) and not args.no_lora)
            loss = seg_b_loss(model, cfg, bb, build(pairs, hs, mode), args.seg) / args.accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == args.steps:
            e = evaluate()
            log.append({"step": step, **e})
            g = inj.gate.abs().mean().item()
            print(f"[{step:4d}] up={e['upper']:.3f} win={e['window']:.3f} "
                  f"state={e['state']:.4f} gap={e['gap']:.3f} "
                  f"recovery={e['recovery']:+.1%} |gate|={g:.4f} | {time.time()-t0:.0f}s")

    tag = f"{args.inject}_w{args.window}_m{args.m}_d{args.d_state}" + ("_nolora" if args.no_lora else "")
    (HERE / "out" / f"state_{tag}.json").write_text(
        json.dumps({"args": vars(args), "log": log}, indent=1))
    torch.save({"mem": mem.state_dict(), "inj": inj.state_dict()},
               HERE / "ckpt" / f"state_{tag}.pt")
    for h_ in handles:
        h_.remove()
    print(f"\nsaved out/state_{tag}.json")


if __name__ == "__main__":
    main()
