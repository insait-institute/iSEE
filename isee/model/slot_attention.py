"""The corrector C_theta: two-stream slot attention (Sec. 3.1, Eq. 3; Supp. "Two streams").

A slot is s^k = [a^k ; p^k], with a^k in R^60 the appearance stream and p^k in R^4 the position stream. The attention
logit of patch n for slot k is the sum of an appearance term and a position term,

    M^k_n = q_a(a^k) . k_a(h_n) + q_p(p^k) . k_p(c_n),            A = softmax_K(M / sqrt(60 + 4)),

there is ONE softmax over slots and one attention map, and each stream is updated from its own grid by its own GRU:

    a^k <- GRU_a( sum_n Abar^k_n v_a(h_n), a^k ),                  p^k <- GRU_p( sum_n Abar^k_n v_p(c_n), p^k ),

where Abar is A renormalised over patches. Position can enter a slot only through c, which is read only into p.

Details the equation leaves out, all as in SlotContrast unless stated:
  * h is layer-normalised before k_a and v_a; c is used as is.
  * a is layer-normalised before q_a; p is RMS-normalised before q_p (it has no layer norm). The GRUs receive the
    normalised a and p as their previous state.
  * Abar = (A + 1e-8) / sum_n (A + 1e-8).
  * GRU_a is followed by a residual MLP on a (LayerNorm, 60 -> 256 -> 60); p has no MLP.
  * Two iterations per frame, three on the first frame.
  * Initialisation: q_p, k_p and v_p start as the identity (so the position logit starts as a spotlight
    c(p) . c_n), everything else at PyTorch's default.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C


def rms_norm(p: torch.Tensor) -> torch.Tensor:
    return p * torch.rsqrt(p.pow(2).mean(-1, keepdim=True) + C.RMS_EPS)


class ResidualMLP(nn.Module):
    """x + Linear(ReLU(Linear(LayerNorm(x))))."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class TwoStreamSlotAttention(nn.Module):
    def __init__(self):
        super().__init__()
        A, P, D, Cd = C.APPEARANCE_DIM, C.POSITION_DIM, C.FEATURE_DIM, C.POSITION_CODE_DIM
        self.norm_features = nn.LayerNorm(D)
        self.norm_appearance = nn.LayerNorm(A)
        self.q_a = nn.Linear(A, A, bias=False)
        self.k_a = nn.Linear(D, A, bias=False)
        self.v_a = nn.Linear(D, A, bias=False)
        self.q_p = nn.Linear(P, P, bias=False)
        self.k_p = nn.Linear(Cd, P, bias=False)
        self.v_p = nn.Linear(Cd, P, bias=False)
        self.gru_a = nn.GRUCell(A, A)
        self.gru_p = nn.GRUCell(P, P)
        self.mlp_a = ResidualMLP(A, C.APPEARANCE_MLP_HIDDEN)
        self.scale = (A + P) ** -0.5
        with torch.no_grad():
            for layer in (self.q_p, self.k_p, self.v_p):
                layer.weight.copy_(torch.eye(P))

    def iterate(self, slots, k_a, v_a, k_p, v_p) -> Tuple[torch.Tensor, torch.Tensor]:
        """One iteration. slots [B, K, 64] -> (slots [B, K, 64], A [B, K, N], the softmax over slots)."""
        a = self.norm_appearance(slots[..., :C.APPEARANCE_DIM])
        p = rms_norm(slots[..., C.APPEARANCE_DIM:])
        logits = torch.einsum("bsd, bfd -> bsf", self.q_a(a), k_a)
        logits = logits + torch.einsum("bsd, bfd -> bsf", self.q_p(p), k_p)
        attn = torch.softmax(logits * self.scale, dim=1)             # A: normalised over slots
        attn_bar = attn + C.ATTENTION_EPS                            # Abar: renormalised over patches
        attn_bar = attn_bar / attn_bar.sum(-1, keepdim=True)
        u_a = torch.einsum("bsf, bfd -> bsd", attn_bar, v_a)
        u_p = torch.einsum("bsf, bfd -> bsd", attn_bar, v_p)
        a = self.gru_a(u_a.flatten(0, 1), a.flatten(0, 1)).unflatten(0, a.shape[:2])
        p = self.gru_p(u_p.flatten(0, 1), p.flatten(0, 1)).unflatten(0, p.shape[:2])
        a = self.mlp_a(a)
        return torch.cat((a, p), dim=-1), attn

    def forward(self, h: torch.Tensor, c: torch.Tensor, slots: torch.Tensor, n_iters: int):
        """h [B, N, 64] appearance grid, c [B, N, 4] position grid, slots [B, K, 64] the predicted slots S^p_t.

        Returns (S^c_t [B, K, 64], A of the first iteration, A of the last iteration). The first iteration's attention
        is computed from the predicted slots alone; it is what TEN reads. The last iteration's attention is the slots'
        grouping of the frame."""
        f = self.norm_features(h)
        k_a, v_a, k_p, v_p = self.k_a(f), self.v_a(f), self.k_p(c), self.v_p(c)
        attn_first = None
        for i in range(n_iters):
            slots, attn = self.iterate(slots, k_a, v_a, k_p, v_p)
            if i == 0:
                attn_first = attn
        return slots, attn_first, attn
