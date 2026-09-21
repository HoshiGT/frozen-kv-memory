#!/usr/bin/env python3
"""Abstract state plus exact anchors.

diagnose.py located the ceiling. Tokens that never appeared before are recovered
at +231% -- the state's gist is better for those than verbatim KV. Tokens that
did appear are recovered at +59%, and the rarer they are the worse it gets:
+46% for something seen exactly once. Those are the specifics, and 76% of the
total gap sits in that group.

No budget fixes it, because the problem is not size. A summary has no mechanism
for "the variable was called k_hi". Verbatim KV has exactly one: it kept the
token.

So keep both kinds of memory, and spend the budget where each works:

    [ a exact anchors ][ k abstract slots ]
      rare tokens'       compressed gist
      original KV        of everything

The anchors are real KV from real positions, chosen by rarity, carrying their
original RoPE phase. The comparison that matters is against the same total
budget spent entirely on abstraction: 64+64 hybrid vs 128 abstract, where 128
abstract is known to be saturated (+67.6%, no better than 64).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnose import per_token_ce
from memtok import MemoryTokens, compress, _layers
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def query_anchors(model, cfg, full, ctx_len: int, probe: torch.Tensor,
                  n: int, sink: int = 4, args=None,
                  restrict: torch.Tensor | None = None) -> torch.Tensor:
    """Deployable retrieval: pick anchors by what the text so far is attending to.

    The oracle looks at the whole continuation. Decoding does not need to --
    it has already emitted a prefix, and that prefix is a legitimate query. So
    run the probe through the model, take its attention over the context keys,
    and keep the positions it actually looks at. This is what "going back to
    look it up" is, mechanically: the query decides what gets paged in.

    Scored on the last layer's queries, averaged over heads and probe tokens.
    """
    with torch.no_grad():
        out = model(probe, position_ids=torch.arange(
            ctx_len, ctx_len + probe.shape[1], device=probe.device).unsqueeze(0),
            output_hidden_states=True)
        h = out.hidden_states[-2][0]                       # into the last block
    layer = _layers(model)[-1]
    attn = layer.self_attn
    q = attn.q_proj(layer.input_layernorm(h))
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    q = q.view(-1, cfg.num_attention_heads, hd)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    k = full[-1][0][0]                                     # [kv_heads, ctx, hd]
    rep = cfg.num_attention_heads // k.shape[0]
    k = k.repeat_interleave(rep, dim=0)
    # Per (head, probe token) attention, then pooled. Averaging first dilutes
    # exactly the signal being looked for: one head wanting one position badly
    # is the whole point of an anchor, and a mean over 16 heads buries it.
    att = torch.einsum("thd,hcd->thc", q.float(), k.float()) / hd ** 0.5
    if args.score_mode == "mean":
        score = att.mean(dim=(0, 1))
    elif args.score_mode == "maxhead":
        score = att.softmax(-1).amax(dim=(0, 1))       # strongest single demand
    else:                                              # "sumsoft"
        score = att.softmax(-1).sum(dim=(0, 1))        # total attention received
    head = torch.arange(min(sink, ctx_len), device=probe.device)
    score[:sink] = float("-inf")                           # sink is kept anyway
    if restrict is not None:
        # two-stage: surprisal already decided what is worth keeping at all,
        # the query only decides which of those to page in now
        mask = torch.full_like(score, float("-inf"))
        mask[restrict] = 0.0
        mask[:sink] = float("-inf")
        score = score + mask
        avail = int((score > float("-inf")).sum())
    else:
        avail = ctx_len - sink
    top = score.topk(min(max(0, n - len(head)), avail)).indices
    return torch.cat([head, top]).sort().values


def surprisal_anchors(model, cfg, ctx: torch.Tensor, n: int, sink: int = 4,
                      chunk: int = 256) -> torch.Tensor:
    """Anchor whatever the model failed to predict while reading.

    Hoshi's proposal, and the one signal here that needs neither the future nor
    a query: if the model could not predict a token from what came before, that
    token carried information nothing else in the context implies -- exactly the
    thing a gist cannot regenerate later. Prediction error driving what gets
    written down is also how encoding appears to work in brains, where surprise
    and memory formation share a signal.

    Rarity, which failed, is only a proxy for this: a rare token in a predictable
    phrase is cheap to reconstruct, while a common word in an unexpected place is
    not, and frequency cannot tell those apart.

    Logits for 4096 positions would be 2.5GB, so the head is applied in chunks
    over hidden states that cost a thousandth of that.
    """
    with torch.no_grad():
        h = model(ctx, output_hidden_states=True).hidden_states[-1][0]
        h = model.model.norm(h)
        ces = []
        for i in range(0, len(h) - 1, chunk):
            j = min(i + chunk, len(h) - 1)
            lg = model.lm_head(h[i:j]).float()
            ces.append(F.cross_entropy(lg, ctx[0, i + 1 : j + 1], reduction="none"))
        ce = torch.cat(ces)                       # ce[i] = surprise of token i+1
    score = torch.full((ctx.shape[1],), float("-inf"), device=ctx.device)
    score[1:] = ce
    score[:sink] = float("-inf")
    head = torch.arange(min(sink, ctx.shape[1]), device=ctx.device)
    top = score.topk(min(max(0, n - len(head)), ctx.shape[1] - sink)).indices
    return torch.cat([head, top]).sort().values


def oracle_anchors(ctx: torch.Tensor, tail: torch.Tensor, n: int,
                   sink: int = 4) -> torch.Tensor:
    """Cheating upper bound: anchor the context tokens that the tail actually reuses.

    Rarity is a guess at which specifics will matter. This looks at the answer.
    It is not deployable, and that is the point -- if even a perfect chooser
    cannot beat spending the same budget on abstraction, the anchor idea is
    dead on its merits rather than on my selection heuristic.
    """
    want = set(tail[0].tolist())
    seq = ctx[0].tolist()
    counts = Counter(seq)
    head = list(range(min(sink, len(seq))))
    cand = [(counts[t], i) for i, t in enumerate(seq) if t in want and i >= sink]
    cand.sort(key=lambda ci: (ci[0], -ci[1]))     # rare-and-reused first
    keep = sorted(set(head + [i for _, i in cand[: max(0, n - len(head))]]))
    return torch.tensor(keep, dtype=torch.long, device=ctx.device)


def pick_anchors(ids: torch.Tensor, n: int, max_count: int = 5,
                 sink: int = 4) -> torch.Tensor:
    """Positions of the rarest tokens, plus the attention sink.

    Rarity is the signal diagnose.py implicated, and it needs no model: a token
    appearing once is where the state does worst and verbatim KV does best.
    Ties are broken toward later positions, which are likelier to still be
    relevant to what comes next.

    The first `sink` positions are kept unconditionally. Attention needs
    somewhere to dump probability mass it does not want to spend, and the
    sequence opening is where a trained model puts it -- baselines.py measured
    the cost of omitting it at ppl 2574 vs 48.8, a single token's difference.
    Selecting anchors purely by rarity excludes those opening tokens precisely
    because they are common, which poisons the whole set.
    """
    seq = ids[0].tolist()
    counts = Counter(seq)
    head = list(range(min(sink, len(seq))))
    cand = [(counts[t], i) for i, t in enumerate(seq)
            if counts[t] <= max_count and i >= sink]
    cand.sort(key=lambda ci: (ci[0], -ci[1]))        # rarest first, then latest
    keep = sorted(set(head + [i for _, i in cand[: max(0, n - len(head))]]))
    return torch.tensor(keep, dtype=torch.long, device=ids.device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--hops", type=int, default=8)
    ap.add_argument("--k", type=int, default=64, help="abstract slots")
    ap.add_argument("--anchors", type=int, default=64, help="exact KV entries")
    ap.add_argument("--sink", type=int, default=4,
                    help="opening positions kept unconditionally; see pick_anchors")
    ap.add_argument("--max-count", type=int, default=5,
                    help="only tokens appearing at most this often are anchor candidates")
    ap.add_argument("--probe", type=int, default=32,
                    help="how many tail tokens act as the retrieval query; a decoder "
                         "has these already, so using them is not cheating")
    ap.add_argument("--pool", type=int, default=512,
                    help="how many positions surprisal pins at encoding time; the "
                         "query then pages in --anchors of them")
    ap.add_argument("--score-mode", default="sumsoft",
                    choices=["mean", "maxhead", "sumsoft"],
                    help="how per-head attention is pooled into one score per position")
    ap.add_argument("--chunk", type=int, default=64,
                    help="re-run retrieval every this many generated tokens; 0 disables")
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
    total_budget = args.k + args.anchors
    print(f"{len(data)} windows | {args.hops} hops over {cut} tokens")
    print(f"budget {total_budget} = {args.anchors} anchors + {args.k} abstract "
          f"({cut/total_budget:.0f}:1)\n")

    def roll(w, k, accum=False):
        """accum: let positions advance with the text instead of resetting.

        The compact convention (state always at the front, positions reset each
        hop) is right when the state is the only thing the tail sees. It is
        wrong for the hybrid: anchors sit at their original positions near 4096
        while a reset state sits near 640, so whichever offset the tail uses,
        one of the two is thousands of positions adrift. Advancing positions
        puts the final memory tokens at the end of the context, right where the
        anchors already are, and the tail then continues from cut+k with no gap
        for either.
        """
        state = None
        for h in range(args.hops):
            past_len = h * args.seg if accum else (0 if state is None else k)
            state = compress(model, mem, w[:, h * args.seg : (h + 1) * args.seg],
                             past=state, past_len=past_len, grad=False, k=k)
        return state

    tot = dict.fromkeys(
        ["upper", "lower", "abstract_all", "abstract_accum",
         "anchors_only", "hybrid", "query_hybrid",
         "chunked_query", "surprisal_hybrid", "two_stage",
         "oracle_only", "oracle_hybrid"], 0.0)
    n_anchor = 0
    with torch.no_grad():
        for i in range(len(data)):
            w = data[i : i + 1].to(dev)
            ctx, tail = w[:, :cut], w[:, cut : cut + args.seg]
            full = [(l.keys, l.values) for l in model(ctx, use_cache=True).past_key_values.layers]

            tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
            tot["lower"] += seg_b_loss(model, cfg, tail, None, cut).item()

            # the whole budget spent on abstraction -- the thing to beat
            st_all = roll(w, total_budget)
            tot["abstract_all"] += seg_b_loss(
                model, cfg, tail, st_all, args.seg + 2 * total_budget).item()
            st_acc = roll(w, total_budget, accum=True)
            tot["abstract_accum"] += seg_b_loss(
                model, cfg, tail, st_acc, cut + total_budget).item()

            idx = pick_anchors(ctx, args.anchors, args.max_count, args.sink)
            n_anchor += len(idx)
            anc = [(kk[:, :, idx], vv[:, :, idx]) for kk, vv in full]
            # anchors keep their own positions, so the tail continues after the
            # context just as it does under `upper`
            tot["anchors_only"] += seg_b_loss(model, cfg, tail, anc, cut).item()

            # hybrid: anchors first (original positions), abstract state after
            st = roll(w, args.k, accum=True)
            mix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                   for (ak, av), (sk, sv) in zip(anc, st)]
            tot["hybrid"] += seg_b_loss(model, cfg, tail, mix, cut + args.k).item()

            # deployable: query built from the first --probe tokens of the tail,
            # which a decoder would already have emitted
            qidx = query_anchors(model, cfg, full, cut, tail[:, : args.probe],
                                 args.anchors, args.sink, args)
            qanc = [(kk[:, :, qidx], vv[:, :, qidx]) for kk, vv in full]
            qmix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                    for (ak, av), (sk, sv) in zip(qanc, st)]
            tot["query_hybrid"] += seg_b_loss(
                model, cfg, tail, qmix, cut + args.k).item()

            # re-retrieve as decoding proceeds: the query is whatever has been
            # emitted so far, so it sharpens chunk by chunk. One retrieval for a
            # whole 512-token continuation is the worst case, not the real one.
            if args.chunk:
                # Re-retrieve as decoding proceeds. The previous version scored
                # each chunk in isolation, which also removed the tail's own
                # preceding tokens -- measuring the loss of local context, not
                # the retrieval policy. Rerunning the whole prefix each time and
                # scoring only the new chunk keeps causal attention intact, so
                # the only thing that varies between chunks is which anchors the
                # query pulled in.
                ce_sum, ntok = 0.0, 0
                for a0 in range(0, tail.shape[1], args.chunk):
                    a1 = min(a0 + args.chunk, tail.shape[1])
                    pr = tail[:, max(0, a0 - args.probe) : a0] if a0 else tail[:, :1]
                    ci = query_anchors(model, cfg, full, cut, pr, args.anchors, args.sink, args)
                    canc = [(kk[:, :, ci], vv[:, :, ci]) for kk, vv in full]
                    cmix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                            for (ak, av), (sk, sv) in zip(canc, st)]
                    ce = per_token_ce(model, cfg, tail[:, :a1], cmix, cut + args.k)
                    lo_ = max(0, a0 - 1)
                    ce_sum += ce[lo_ : a1 - 1].sum().item()
                    ntok += (a1 - 1) - lo_
                tot["chunked_query"] += ce_sum / ntok

            sidx = surprisal_anchors(model, cfg, ctx, args.anchors, args.sink)
            sanc = [(kk[:, :, sidx], vv[:, :, sidx]) for kk, vv in full]
            smix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                    for (ak, av), (sk, sv) in zip(sanc, st)]
            tot["surprisal_hybrid"] += seg_b_loss(
                model, cfg, tail, smix, cut + args.k).item()

            # write with surprisal, read with the query: a large pool is pinned
            # at encoding time, and only args.anchors of it is paged in
            pool = surprisal_anchors(model, cfg, ctx, args.pool, args.sink)
            tidx = query_anchors(model, cfg, full, cut, tail[:, : args.probe],
                                 args.anchors, args.sink, args, restrict=pool)
            tanc = [(kk[:, :, tidx], vv[:, :, tidx]) for kk, vv in full]
            tmix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                    for (ak, av), (sk, sv) in zip(tanc, st)]
            tot["two_stage"] += seg_b_loss(
                model, cfg, tail, tmix, cut + args.k).item()

            oidx = oracle_anchors(ctx, tail, args.anchors, args.sink)
            oanc = [(kk[:, :, oidx], vv[:, :, oidx]) for kk, vv in full]
            tot["oracle_only"] += seg_b_loss(model, cfg, tail, oanc, cut).item()
            omix = [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                    for (ak, av), (sk, sv) in zip(oanc, st)]
            tot["oracle_hybrid"] += seg_b_loss(
                model, cfg, tail, omix, cut + args.k).item()

    n = len(data)
    for kk in tot:
        tot[kk] /= n
    gap = tot["lower"] - tot["upper"]
    print(f"upper (all {cut} KV) CE={tot['upper']:.4f}")
    print(f"lower (nothing)      CE={tot['lower']:.4f}   gap={gap:.4f} nats")
    print(f"anchors actually placed: {n_anchor/n:.1f} of {args.anchors} requested\n")
    print(f"  {'condition':<26} {'CE':>8} {'recovery':>10}")
    print("  " + "-" * 46)
    for name, label in (("abstract_all", f"{total_budget} abstract (compact pos)"),
                        ("abstract_accum", f"{total_budget} abstract (accum pos)"),
                        ("anchors_only", f"{args.anchors} anchors only"),
                        ("hybrid", f"{args.anchors} anchors + {args.k} abstract"),
                        ("query_hybrid", f"{args.anchors} QUERY + {args.k} abstract"),
                        ("chunked_query", f"{args.anchors} QUERY re-retrieved/{args.chunk}"),
                        ("surprisal_hybrid", f"{args.anchors} SURPRISAL + {args.k} abstract"),
                        ("two_stage", f"SURPRISAL{args.pool}->QUERY{args.anchors} + {args.k}"),
                        ("oracle_only", f"{args.anchors} ORACLE anchors only"),
                        ("oracle_hybrid", f"{args.anchors} ORACLE + {args.k} abstract")):
        ce = tot[name]
        print(f"  {label:<26} {ce:8.4f} {(tot['lower']-ce)/gap:+9.1%}")


if __name__ == "__main__":
    main()
