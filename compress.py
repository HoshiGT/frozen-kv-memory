#!/usr/bin/env python3
"""Online KV bottleneck that actually mixes, instead of softly picking.

v0 failed because the K vectors in the cache already carry RoPE. Averaging
positions p and q blends two different rotations, so any diffuse softmax
produces a K that means nothing -- gradient descent's only escape was to go
sharp, i.e. to degrade into KV pruning, which free baselines already do better.

v1 makes mixing legal:

    K (rotated at its own position)
      -> un-rotate to phase 0        (exact: RoPE is an orthogonal rotation)
      -> mix in the aligned space    <- a slot can now superpose many tokens
      -> re-rotate at the slot's expected position

V never had RoPE, so it was always free to mix; only K was blocked.

The first n_sink entries bypass the whole thing and are kept verbatim. They are
the attention sink -- dropping them costs ~830 points of recovery, and that is
a trick worth zero parameters, so no capacity is spent relearning it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(pos: torch.Tensor, inv_freq: torch.Tensor):
    """pos: [...] -> cos, sin: [..., D]"""
    freqs = pos.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


class KVCompressor(nn.Module):
    def __init__(self, n_layers: int, head_dim: int, m: int, seg: int,
                 inv_freq: torch.Tensor, n_sink: int = 4,
                 temp_init: float = 1.0, stride_init: float = 8.0,
                 realign: bool = True, mix_mode: str = "softmax"):
        super().__init__()
        assert m > n_sink, "need slots beyond the sink"
        self.m, self.seg, self.n_sink = m, seg, n_sink
        self.realign = realign   # False = ablation: mix in rotated space (v0)
        # softmax weights live on a probability simplex -- non-negative, summing
        # to one -- which is the inductive bias of *choosing*, not of *summing*.
        # "linear" removes that cage so a slot can hold an unconstrained
        # combination; "sigmoid" keeps non-negativity but drops normalisation.
        self.mix_mode = mix_mode
        self.n_layers = n_layers
        self.m_learn = m - n_sink
        self.n_src = seg - n_sink

        inv_freq = inv_freq.detach().float().cpu()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = rope_cos_sin(torch.arange(seg), inv_freq)
        self.register_buffer("cos_src", cos[n_sink:], persistent=False)  # [n_src, D]
        self.register_buffer("sin_src", sin[n_sink:], persistent=False)
        self.register_buffer("src_pos", torch.arange(n_sink, seg).float(), persistent=False)

        self.score = nn.ModuleList(
            [nn.Linear(2 * head_dim, self.m_learn, bias=False) for _ in range(n_layers)]
        )
        self.k_out = nn.ModuleList(
            [nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_layers)]
        )
        self.v_out = nn.ModuleList(
            [nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_layers)]
        )
        for lin in list(self.k_out) + list(self.v_out):
            nn.init.zeros_(lin.weight)          # start as a pure residual
        for lin in self.score:
            lin.weight.data.mul_(0.05)          # let the positional prior lead

        # start from the strongest free baseline: evenly spaced picks.
        # softmax reaches "pick position j" with a large logit; the unnormalised
        # modes reach it with a coefficient of 1 after their own scaling, so the
        # two start from the same place and the comparison stays fair.
        if mix_mode != "softmax":
            stride_init = self.n_src ** 0.5
        bias = torch.zeros(n_layers, self.n_src, self.m_learn)
        idx = torch.linspace(0, self.n_src - 1, self.m_learn).long()
        for j, s in enumerate(idx.tolist()):
            bias[:, s, j] = stride_init
        self.pos_bias = nn.Parameter(bias)

        self.log_temp = nn.Parameter(torch.zeros(n_layers))
        # mixing shrinks vector norms; let each layer rescale
        self.k_scale = nn.Parameter(torch.ones(n_layers))
        self.v_scale = nn.Parameter(torch.ones(n_layers))

    def compress_layer(self, K: torch.Tensor, V: torch.Tensor, i: int):
        """K, V: [B, H, seg, D] -> [B, H, m, D], plus the mixing weights."""
        dt, ns = K.dtype, self.n_sink
        Ksink, Vsink = K[:, :, :ns], V[:, :, :ns]
        Kr, Vr = K[:, :, ns:].float(), V[:, :, ns:].float()

        # --- un-rotate to a common phase ---------------------------------
        Ku = (Kr * self.cos_src - rotate_half(Kr) * self.sin_src) if self.realign else Kr

        s = self.score[i](torch.cat([Ku, Vr], dim=-1)) + self.pos_bias[i]
        s = s / self.log_temp[i].exp().clamp(min=1e-2)
        if self.mix_mode == "softmax":
            w = s.softmax(dim=2)
        elif self.mix_mode == "sigmoid":
            w = s.sigmoid() / self.n_src ** 0.5
        elif self.mix_mode == "linear":
            w = s / self.n_src ** 0.5
        else:
            raise ValueError(self.mix_mode)

        # --- mix in the aligned space ------------------------------------
        Kc = torch.einsum("bhsm,bhsd->bhmd", w, Ku)
        Vc = torch.einsum("bhsm,bhsd->bhmd", w, Vr)
        Kc = Kc + self.k_out[i](Kc)
        Vc = Vc + self.v_out[i](Vc)

        # --- re-rotate at each slot's expected source position ------------
        if self.realign:
            p = torch.einsum("bhsm,s->bhm", w, self.src_pos)
            cos_p, sin_p = rope_cos_sin(p, self.inv_freq)
            Kc = Kc * cos_p + rotate_half(Kc) * sin_p

        Kc = (Kc * self.k_scale[i]).to(dt)
        Vc = (Vc * self.v_scale[i]).to(dt)
        return torch.cat([Ksink, Kc], dim=2), torch.cat([Vsink, Vc], dim=2), w

    @torch.no_grad()
    def selection_stats(self, w: torch.Tensor) -> dict:
        a = w.abs()
        tot = a.sum(dim=2, keepdim=True).clamp_min(1e-9)
        p = (a / tot).clamp_min(1e-9)          # normalised, so entropy is comparable
        return {
            "entropy": (-(p * p.log()).sum(dim=2)).mean().item(),
            "max_w": p.amax(dim=2).mean().item(),
        }
