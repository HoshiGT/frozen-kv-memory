#!/usr/bin/env python3
"""A recurrent abstract state -- not a selection over tokens.

The KV-compression line (compress.py) kept failing the same way: whatever the
mixing rule, a slot stayed a combination of token representations, so you could
always ask "which tokens does this slot stand for". That question is the tell.
Here it has no answer by construction.

    h : [B, d_state]

h is built by walking the segment and updating, the way an RNN or an SSM state
is built. Its width has nothing to do with how many tokens went in -- 512
tokens or 5000, h is the same size. Nothing in it is anchored to a position.

Attention cannot read a bare vector, so a read-out projects h into m pseudo-KV
entries. Those entries are an *interface*, not the memory: they are read out of
h, never assembled from tokens. This is the whole difference from v1.

Because the update is recurrent, this also does online compression for free --
h can keep absorbing further segments, which the one-shot KV compressor could
not do.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StateMemory(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_kv_heads: int, head_dim: int,
                 m: int = 8, d_state: int = 1024, d_slot: int = 256,
                 chunk: int = 32):
        super().__init__()
        self.d_state, self.m, self.chunk = d_state, m, chunk
        self.n_layers, self.n_kv_heads, self.head_dim = n_layers, n_kv_heads, head_dim

        self.inp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_state),
            nn.GELU(),
        )
        self.cell = nn.GRUCell(d_state, d_state)
        self.h0 = nn.Parameter(torch.zeros(d_state))
        # States share a large common component; the part that differs between
        # segments is a small perturbation on top. Without removing that DC the
        # read-out reproduces it faithfully and emits near-constant slots
        # (measured: 7% variation across segments, vs 85% for real KV).
        self.register_buffer("h_mean", torch.zeros(d_state))

        # read-out: h -> m slot vectors -> per-layer K and V
        self.to_slots = nn.Sequential(
            nn.LayerNorm(d_state),
            nn.Linear(d_state, m * d_slot),
        )
        self.slot_norm = nn.LayerNorm(d_slot)
        self.k_read = nn.ModuleList(
            [nn.Linear(d_slot, n_kv_heads * head_dim) for _ in range(n_layers)]
        )
        self.v_read = nn.ModuleList(
            [nn.Linear(d_slot, n_kv_heads * head_dim) for _ in range(n_layers)]
        )
        # Real K RMS runs 21.3 -> 2.8 across layers and real V runs 0.15 -> 7.5,
        # a 50x spread; a read-out that emits ~1.0 everywhere is invisible in
        # some layers and drowns the real entries in others. Calibrate to the
        # measured per-layer RMS and leave a learned trim on top.
        self.register_buffer("k_target", torch.ones(n_layers))
        self.register_buffer("v_target", torch.ones(n_layers))
        # Zero-init residual. A synthesised K with a realistic norm but a random
        # direction produces extreme attention logits and wrecks the forward
        # pass (measured: -2797% recovery). Start the slots harmless instead:
        # v_scale = 0 means they contribute nothing to the attention output, no
        # matter what weight they draw, and k_scale starts at half strength so
        # they cannot dominate the softmax either. Training grows them.
        self.k_scale = nn.Parameter(torch.full((n_layers,), 0.15))
        self.v_scale = nn.Parameter(torch.zeros(n_layers))

    @torch.no_grad()
    def calibrate(self, pairs):
        """Record the RMS of genuine cache entries, layer by layer."""
        for i, (k, v) in enumerate(pairs):
            self.k_target[i] = k.float().pow(2).mean().sqrt()
            self.v_target[i] = v.float().pow(2).mean().sqrt()

    def absorb(self, hs: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """hs: [B, S, d_model] hidden states of a segment -> updated state [B, d_state].

        Call it again with the previous h to keep compressing further segments:
        that is the online/recurrent path the one-shot compressor never had.
        """
        B, S, _ = hs.shape
        n = S // self.chunk
        # the module runs in fp32 while the frozen model hands us bf16
        x = hs[:, : n * self.chunk].reshape(B, n, self.chunk, -1).mean(dim=2).float()
        x = self.inp(x)                                  # [B, n, d_state]
        if h is None:
            h = self.h0.unsqueeze(0).expand(B, -1).contiguous()
        for t in range(n):
            h = self.cell(x[:, t], h)
        return h

    def read(self, h: torch.Tensor, dtype: torch.dtype):
        """h -> list of (K, V), each [B, n_kv_heads, m, head_dim]."""
        B = h.shape[0]
        if self.training:
            with torch.no_grad():
                self.h_mean.mul_(0.99).add_(h.detach().mean(dim=0), alpha=0.01)
        h = h - self.h_mean
        slots = self.slot_norm(self.to_slots(h).view(B, self.m, -1))

        def to_rms(x, target):
            rms = x.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            return x / rms * target

        out = []
        for i in range(self.n_layers):
            k = to_rms(self.k_read[i](slots), self.k_target[i] * self.k_scale[i])
            v = to_rms(self.v_read[i](slots), self.v_target[i] * self.v_scale[i])
            k = k.view(B, self.m, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, self.m, self.n_kv_heads, self.head_dim).transpose(1, 2)
            out.append((k.to(dtype), v.to(dtype)))
        return out


class StateInjector(nn.Module):
    """Deliver h by modulating the residual stream, not by faking KV entries.

    Pushing h in as pseudo-KV puts it in a softmax competition it cannot win:
    give the synthetic K a real norm and it hijacks attention (-2797% recovery
    measured); shrink it to zero and it draws no weight, which zeroes the
    gradient into V, so it can never grow. Every initialisation collapsed to one
    of those two.

        hidden_l <- hidden_l + gate_l * up_l(down(h))       gate init 0

    Now d(loss)/d(gate) = d(loss)/d(hidden) . up(down(h)) -- no attention weight
    in the path. The slots start harmless AND keep a live gradient, which the
    KV route could not do at the same time. It is also closer to what an
    abstract state should be: something that conditions the computation, rather
    than something disguised as a few tokens.
    """

    def __init__(self, d_state: int, d_model: int, n_layers: int, rank: int = 128):
        super().__init__()
        self.n_layers = n_layers
        self.down = nn.Linear(d_state, rank)
        self.up = nn.ModuleList(
            [nn.Linear(rank, d_model, bias=False) for _ in range(n_layers)]
        )
        # Exactly one of the two factors may start at zero. Zeroing both
        # deadlocks them: d/d(gate) = up(z) = 0 and d/d(up) = gate*z = 0, so
        # neither can ever move. The gate carries the zero; up keeps its normal
        # init so the gate has a live gradient to ride.
        self.gate = nn.Parameter(torch.zeros(n_layers))

    def deltas(self, h: torch.Tensor, dtype: torch.dtype):
        z = torch.tanh(self.down(h.float()))
        return [(self.gate[i] * self.up[i](z)).to(dtype) for i in range(self.n_layers)]


def decoder_layers(model):
    """Reach the transformer blocks through any PEFT wrapping."""
    m = model
    if hasattr(m, "base_model"):
        m = m.base_model
    if hasattr(m, "model") and hasattr(m.model, "model"):
        m = m.model
    return m.model.layers


def attach_injector(model, holder: dict):
    """holder['deltas'] is a per-layer list, or None to pass through untouched."""
    handles = []
    for i, layer in enumerate(decoder_layers(model)):
        def hook(mod, args, out, idx=i):
            d = holder.get("deltas")
            if d is None:
                return out
            if isinstance(out, tuple):
                return (out[0] + d[idx].unsqueeze(1),) + out[1:]
            return out + d[idx].unsqueeze(1)
        handles.append(layer.register_forward_hook(hook))
    return handles
