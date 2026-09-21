#!/usr/bin/env python3
"""The working form of everything the experiments concluded.

hybrid.py is a measurement rig -- it computes eight conditions to compare them.
This is the one configuration that won, with the comparisons removed:

    64 abstract slots over the last segment   the gist, what comes next
    original KV, offloaded                    every specific, verbatim
    512 entries paged in per query            what this moment needs

Measured on Qwen3-0.6B over 4096 tokens of real dialogue at 7:1: +86.8% of the
information a full cache provides, against +69.4% for spending the same budget
entirely on abstraction.

VRAM is O(n/8) rather than O(1): the anchors are paged into the same scarce
memory and have to grow with the context to hold coverage. What the scheme
actually buys is moving the O(n) bulk off the GPU.

Three findings are baked in rather than configurable, because getting them wrong
is expensive and none of them is a preference:

  only the last segment is summarised     rolling dilutes it; 4 points
  the sink is always anchored             without it, partial KV scores -400%
  the query is what the state imagines    beats using the real prefix as query

Usage:

    mem = HybridMemory.load("ckpt/memtok_k32_deep_frozen.pt", model, tok)
    handle = mem.compress(long_context_ids)        # state + offloaded KV
    past = mem.recall(handle, prefix_ids)          # what to put in the cache
    out = model(next_ids, past_key_values=past)

`recall` is the only thing on the hot path, and it costs one short generation
plus one attention scoring pass over the offloaded keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from memtok import MemoryTokens, compress, _layers


@torch.no_grad()
def kv_only(model, ids: torch.Tensor):
    """KV for a sequence, without paying for its logits.

    Inference-only by construction: filling a cache never needs a graph, and
    building one over 4096 positions costs gigabytes of stored activations.

    Calling the causal-LM wrapper computes a distribution at every position:
    4096 tokens over a 151k vocabulary is 1.2GB in bf16, none of which is wanted
    when the point is to fill a cache. The inner model stops before the head.
    """
    base = getattr(model, "model", model)
    out = base(input_ids=ids, use_cache=True)
    return [(l.keys, l.values) for l in out.past_key_values.layers]


@dataclass
class Handle:
    """What compression leaves behind: a small resident part and a large cold one."""
    state: list[tuple[torch.Tensor, torch.Tensor]]   # k slots, stays on GPU
    kv: list[tuple[torch.Tensor, torch.Tensor]]      # full KV, wherever it fits
    length: int                                      # tokens the context held

    def resident_bytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                   for k, v in self.state)

    def cold_bytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                   for k, v in self.kv)


class HybridMemory:
    def __init__(self, model, tok, mem: MemoryTokens, k: int = 64,
                 seg: int = 1024, sink: int = 4, offload: str = "cpu"):
        self.model, self.tok, self.mem = model, tok, mem
        self.cfg = model.config
        self.k, self.seg, self.sink = k, seg, sink
        self.offload = offload

    @classmethod
    def load(cls, ckpt: str | Path, model, tok, **kw) -> "HybridMemory":
        sd = torch.load(str(ckpt))
        sd = sd["mem"] if "mem" in sd else sd
        depth = sd["deep"].shape[0] if "deep" in sd else 0
        mem = MemoryTokens(sd["emb"].shape[0], model.config.hidden_size,
                           depth=depth).to(next(model.parameters()).device)
        mem.load_state_dict(sd)
        return cls(model, tok, mem, **kw)

    # ---------------------------------------------------------------- compress

    @torch.no_grad()
    def compress(self, ids: torch.Tensor) -> Handle:
        """Summarise the most recent segment and keep the full KV to page from.

        Earlier versions rolled one state across the whole context, hop by hop.
        tiered.py showed that does not accumulate: compressing only the last
        segment scores +76.1% inside the hybrid against +72.2% for a 16-hop
        roll over the same 8192 tokens, at a sixteenth of the compute. The
        state's effective window is about one segment, so rolling mostly
        dilutes what that segment contributed.

        Everything older is not summarised at all -- it is carried verbatim in
        `kv` and retrieved on demand. That division is not a preference: an
        abstract state holds what the model could have predicted anyway, and
        what it could not is exactly what only the original KV retains.
        """
        n = ids.shape[1]
        start = max(0, n - self.seg)
        state = compress(self.model, self.mem, ids[:, start:], past=None,
                         past_len=start, grad=False, k=self.k)
        full = kv_only(self.model, ids)
        if self.offload != "cuda":
            full = [(k.to(self.offload, non_blocking=True),
                     v.to(self.offload, non_blocking=True)) for k, v in full]
        return Handle(state=state, kv=full, length=n)

    # ------------------------------------------------------------------ recall

    @torch.no_grad()
    def _imagine(self, handle: Handle, prefix: torch.Tensor, n_tok: int):
        """Generate from the state alone; the guess is the retrieval query.

        Deliberately not given the anchors: this runs before retrieval, and its
        job is only to say where the text is heading. Drafting with anchors, and
        drafting longer, were both measured and neither helped -- the draft is
        useful for direction, not detail.
        """
        from train import cache_from

        dev = prefix.device
        sink = [(k[:, :, : self.sink].to(dev), v[:, :, : self.sink].to(dev))
                for k, v in handle.kv]
        base = [(torch.cat([sk, ak], dim=2), torch.cat([sv, av], dim=2))
                for (sk, sv), (ak, av) in zip(sink, handle.state)]
        cache = cache_from([(k.clone(), v.clone()) for k, v in base], self.cfg)
        plen = base[0][0].shape[2]
        ids, out = prefix[:, -1:], []
        for _ in range(n_tok):
            pos = torch.arange(handle.length + self.k + len(out),
                               handle.length + self.k + len(out) + 1, device=dev)
            o = self.model(input_ids=ids, position_ids=pos.unsqueeze(0),
                           past_key_values=cache, use_cache=True,
                           cache_position=torch.arange(plen, plen + 1, device=dev),
                           attention_mask=torch.ones(1, plen + 1, device=dev,
                                                     dtype=torch.long))
            cache, plen = o.past_key_values, plen + 1
            ids = o.logits[:, -1].argmax(-1, keepdim=True)
            out.append(ids)
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def _score(self, handle: Handle, query: torch.Tensor) -> torch.Tensor:
        """Attention from the query over the cold keys, one score per position."""
        dev = query.device
        base = getattr(self.model, "model", self.model)
        h = base(input_ids=query, output_hidden_states=True).hidden_states[-2][0]
        layer = _layers(self.model)[-1]
        attn = layer.self_attn
        q = attn.q_proj(layer.input_layernorm(h))
        hd = getattr(self.cfg, "head_dim",
                     self.cfg.hidden_size // self.cfg.num_attention_heads)
        q = q.view(-1, self.cfg.num_attention_heads, hd)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        k = handle.kv[-1][0][0].to(dev)
        k = k.repeat_interleave(self.cfg.num_attention_heads // k.shape[0], dim=0)
        return torch.einsum("thd,hcd->c", q.float(), k.float()) / (len(q) * hd ** 0.5)

    @torch.no_grad()
    def recall(self, handle: Handle, prefix: torch.Tensor, n: int = 512,
               draft: int = 16):
        """Assemble the cache for the next step: sink + anchors + abstract state.

        `draft` is the whole hot-path cost -- one forward per token generated.
        Measured at 8/16/32/128 it scores 70.7/70.9/70.9/71.4%, so 16 gives away
        half a point and runs eight times faster. The draft only has to point in
        the right direction; it was never going to get the details right.
        """
        dev = prefix.device
        guess = self._imagine(handle, prefix, draft)
        score = self._score(handle, guess)
        score[: self.sink] = float("-inf")
        top = score.topk(min(max(0, n - self.sink), handle.length - self.sink)).indices
        idx = torch.cat([torch.arange(self.sink, device=score.device), top]).sort().values
        anc = [(k[:, :, idx.to(k.device)].to(dev), v[:, :, idx.to(v.device)].to(dev))
               for k, v in handle.kv]
        return [(torch.cat([ak, sk], dim=2), torch.cat([av, sv], dim=2))
                for (ak, av), (sk, sv) in zip(anc, handle.state)]

    def offset(self, handle: Handle) -> int:
        """Where the next tokens' positions start, given what recall assembled."""
        return handle.length + self.k


if __name__ == "__main__":
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from train import load_chunks, seg_b_loss

    ap = argparse.ArgumentParser(description="sanity-check the packaged form")
    ap.add_argument("--ckpt", default="ckpt/memtok_k32_deep_frozen.pt")
    ap.add_argument("--model", default="./qwen3-0.6b")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    mem = HybridMemory.load(a.ckpt, model, tok)

    data = load_chunks(tok, a.ctx + 512, a.n)
    lo = hi = hy = 0.0
    stats = (1, 1)
    torch.set_grad_enabled(False)          # nothing here trains
    for i in range(a.n):
        w = data[i : i + 1].to(dev)
        ctx, tail = w[:, : a.ctx], w[:, a.ctx : a.ctx + 512]
        hnd = mem.compress(ctx)
        past = mem.recall(hnd, tail[:, :1])
        hy += seg_b_loss(model, mem.cfg, tail, past, mem.offset(hnd)).item()
        lo += seg_b_loss(model, mem.cfg, tail, None, a.ctx).item()
        full = kv_only(model, ctx)
        hi += seg_b_loss(model, mem.cfg, tail, full, a.ctx).item()
        stats = (hnd.resident_bytes(), hnd.cold_bytes())
        # a full cache is ~470MB here; one per window does not fit eight times
        del hnd, past, full, w, ctx, tail
        torch.cuda.empty_cache()
    lo, hi, hy = lo / a.n, hi / a.n, hy / a.n
    print(f"upper {hi:.4f}  lower {lo:.4f}  hybrid {hy:.4f}")
    print(f"recovery {(lo - hy) / (lo - hi):+.1%}")
    print(f"resident {stats[0]/2**20:.1f} MiB  cold {stats[1]/2**20:.1f} MiB  "
          f"({stats[1]/stats[0]:.0f}x offloaded)")
