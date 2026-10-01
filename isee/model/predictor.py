"""The predictor P_omega: one transformer per stream (Sec. 3.1, "Predictors, decoder and objective").

    a' = a + W_a( T_a(a) )                          the appearance predictor reads the appearance of all slots
    p' = p + W_p( T_p([p ; E(sg(a))]) )             the position predictor reads the position of all slots and an
                                                     8-dimensional projection E of their appearance, through a
                                                     stop-gradient sg

T_a and T_p are one pre-norm transformer block each, with four heads (width 60 and 4 + 8 = 12).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C


class SelfAttention(nn.Module):
    """Multi-head self-attention as in SlotContrast.

    Note: SlotContrast's attention MULTIPLIES the queries by sqrt(head_dim) (it divides by head_dim^-0.5) where the
    usual convention divides by it. The released checkpoints were trained this way, so it is kept."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        bound = math.sqrt(6.0 / (dim + dim))                    # Xavier-uniform for each of q, k, v
        nn.init.uniform_(self.qkv.weight, -bound, bound)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h, d = self.n_heads, self.head_dim
        q, k, v = (t.reshape(b, n, h, d).permute(0, 2, 1, 3).reshape(b * h, n, d)
                   for t in self.qkv(x).chunk(3, dim=-1))
        attn = torch.bmm(q / d ** -0.5, k.transpose(-2, -1)).softmax(dim=-1)
        out = (attn @ v).reshape(b, h, n, d).permute(0, 2, 1, 3).reshape(b, n, h * d)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)), then x + MLP(LN(x)) with hidden width 4 x dim."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class TwoStreamPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        A, P, E = C.APPEARANCE_DIM, C.POSITION_DIM, C.PREDICTOR_APPEARANCE_PROJ
        self.appearance = TransformerBlock(A, C.PREDICTOR_HEADS)
        self.appearance_out = nn.Linear(A, A)
        self.appearance_proj = nn.Linear(A, E)
        self.position = TransformerBlock(P + E, C.PREDICTOR_HEADS)
        self.position_out = nn.Linear(P + E, P)
        for layer in (self.appearance_out, self.position_out):      # the predictor starts as the identity
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """S^c_{t-1} [B, K, 64] -> S^p_t [B, K, 64]."""
        a, p = slots[..., :C.APPEARANCE_DIM], slots[..., C.APPEARANCE_DIM:]
        a_next = a + self.appearance_out(self.appearance(a))
        cond = self.appearance_proj(a.detach())
        p_next = p + self.position_out(self.position(torch.cat((p, cond), dim=-1)))
        return torch.cat((a_next, p_next), dim=-1)
