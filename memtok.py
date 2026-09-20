#!/usr/bin/env python3
"""Let the model compress by itself.

Every failure today came from the same place: I built the state outside the
model and then tried to hand it over. Norm calibration, gating, injection
points, zero-init deadlocks -- none of those are difficulties of compression,
they are difficulties of bolting a foreign object onto a transformer.

So don't bolt anything on. Append k learnable memory tokens to the segment, run
the model's own forward, and take the KV at those positions as the state:

    [ segment A (512 tok) ][ mem_1 ... mem_k ]
                              |
                              +-- their KV is the whole state
    [ segment B ] attends to that, and nothing else

The state is now produced by the model, so it lives in the model's own
distribution by construction -- nothing to calibrate, nothing to gate. LoRA
learns one thing: route what matters into those k positions, and read it back
out. The compression policy is the model's, not mine; the only thing imposed is
the constraint that just k positions survive.

Recurrent by construction: the next segment appends a fresh set of memory
tokens while attending to the previous state, so it compresses on top of what
is already there.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class MemoryTokens(nn.Module):
    """k learnable slots, optionally with a per-layer bias of their own.

    `depth` turns the flat prefix into a deep one: layer i adds `deep[i]` to the
    hidden states at the memory positions only. Attention is causal, so the 512
    real tokens ahead of them cannot see those positions -- the backbone's
    behaviour on ordinary text is bit-for-bit unchanged, and the extra capacity
    lands entirely inside the state being built. Costs L*k*d parameters (~0.9M
    on 0.6B) against 32k for the flat version.
    """

    def __init__(self, k: int, d_model: int, init_std: float = 0.02,
                 depth: int = 0):
        super().__init__()
        self.k = k
        self.emb = nn.Parameter(torch.randn(k, d_model) * init_std)
        # zero-init is safe here: this is an additive residual with no gate, so
        # the gradient w.r.t. it is nonzero from the first step
        self.deep = nn.Parameter(torch.zeros(depth, k, d_model)) if depth else None

    def append(self, tok_emb: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """tok_emb: [B, S, d] -> [B, S + k, d]; k <= self.k uses the first k slots."""
        B = tok_emb.shape[0]
        k = self.k if k is None else min(k, self.k)
        mem = self.emb[:k].unsqueeze(0).expand(B, -1, -1).to(tok_emb.dtype)
        return torch.cat([tok_emb, mem], dim=1)


def embed_tokens(model, ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(ids)


def _layers(model):
    """The transformer blocks, through PEFT wrapping or none.

    state.py's version assumed a PEFT wrapper: on a bare model, HF's own
    `base_model` property already unwraps to Qwen3Model, and descending once more
    overshoots. Walking down until something actually has `.layers` handles both.
    """
    m = model
    for _ in range(4):
        if hasattr(m, "layers"):
            return m.layers
        nxt = getattr(m, "model", None) or getattr(m, "base_model", None)
        if nxt is None or nxt is m:
            break
        m = nxt
    raise AttributeError(f"no .layers under {type(model).__name__}")


@contextmanager
def deep_prefix(model, mem: MemoryTokens, k: int):
    """While active, layer i adds mem.deep[i] to the last k positions.

    The hook fires on layer i's *output*, so layer i's own KV is already
    computed -- deep[i] shapes the KV of layer i+1 onwards. A bias on the final
    layer would therefore reach no KV at all, which is why depth is set to
    n_layers - 1.

    Only live during compression: once the state exists, segment B reads it as
    an ordinary cache and no hook is involved.
    """
    if mem.deep is None or k == 0:
        yield
        return

    handles = []
    for i, layer in enumerate(_layers(model)[: mem.deep.shape[0]]):
        def hook(_mod, _inp, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            bias = mem.deep[i, :k].to(h.dtype)
            h = torch.cat([h[:, :-k], h[:, -k:] + bias], dim=1)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        handles.append(layer.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def compress(model, mem: MemoryTokens, ids: torch.Tensor,
             past=None, past_len: int = 0, grad: bool = True, k: int | None = None):
    """Run segment `ids` with memory tokens appended; return their KV.

    `past` lets the new memory tokens see the previous state, which is what
    makes the scheme recurrent rather than one-shot.
    """
    from train import cache_from   # local import keeps module import-light

    B, S = ids.shape
    k = mem.k if k is None else min(k, mem.k)
    x = mem.append(embed_tokens(model, ids), k)
    total = S + k
    pos = torch.arange(past_len, past_len + total, device=ids.device)
    kw = {}
    if past is not None:
        kw["past_key_values"] = cache_from(past, model.config)
        plen = past[0][0].shape[2]
        kw["attention_mask"] = torch.ones(B, plen + total, device=ids.device, dtype=torch.long)
        kw["cache_position"] = torch.arange(plen, plen + total, device=ids.device)

    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, deep_prefix(model, mem, k):
        out = model(inputs_embeds=x, position_ids=pos.unsqueeze(0).expand(B, -1),
                    use_cache=True, **kw)
    # keep only the memory positions: that is the entire surviving state
    return [(l.keys[:, :, -k:], l.values[:, :, -k:])
            for l in out.past_key_values.layers]
