"""The decoder: reconstruct h_t in each object's own frame (Sec. 3.1; Supp. "Two streams", decoder equation).

    (mu^k, sigma^k) = h(p^k)                                     position and scale, from the position stream ALONE
    u^k_n           = (x_n - mu^k) / (delta sigma^k)             the patch coordinate relative to the object
    (h^k_n, alpha^k_n) = MLP([a^k ; gamma(u^k_n)])               feature and alpha logit of patch n for slot k
    h_hat_n         = sum_k softmax_k(alpha^k_n) h^k_n

h is a head with 64 hidden units, tanh for mu and a clamped exponential for sigma in [0.01, 4]; delta = 5 as in ISA;
gamma is an 8-band Fourier embedding; the MLP has three hidden layers of 1024. The painting MLP has no access to
absolute position. softmax_k(alpha) are the decoder's masks, the segmentation.

Implementation note: the first layer of the MLP on the concatenation [a ; gamma(u)] is factorised as
W_1 (a_pad + W_gamma gamma(u) + b_gamma), where a_pad is the slot with its position dimensions set to zero; this is
the same function class.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C
from .features import patch_coordinates


class MLP(nn.Module):
    """Linear layers with ReLU in between (no activation after the last)."""

    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class ObjectFrameDecoder(nn.Module):
    def __init__(self, grid_side: int):
        super().__init__()
        self.pose_head = MLP([C.POSITION_DIM, C.POSE_HEAD_HIDDEN, 4])            # h: p -> (mu, log sigma)
        n_gamma = 2 * (1 + 2 * C.FOURIER_BANDS)                                   # u, sin, cos per axis = 34
        self.gamma_proj = nn.Linear(n_gamma, C.SLOT_DIM)
        nn.init.zeros_(self.pose_head.layers[-1].weight)           # h starts at mu = 0, sigma = 1
        nn.init.zeros_(self.pose_head.layers[-1].bias)
        self.mlp = MLP([C.SLOT_DIM, *C.DECODER_HIDDEN, C.BACKBONE_DIM + 1])
        self.log_sigma_range = (math.log(C.SCALE_RANGE[0]), math.log(C.SCALE_RANGE[1]))
        self.register_buffer("grid", patch_coordinates(grid_side), persistent=False)          # x_n, [N, 2]
        self.register_buffer("freqs", math.pi * torch.pow(2.0, torch.arange(C.FOURIER_BANDS, dtype=torch.float32)),
                             persistent=False)
        appearance_mask = torch.zeros(C.SLOT_DIM)
        appearance_mask[:C.APPEARANCE_DIM] = 1.0
        self.register_buffer("appearance_mask", appearance_mask, persistent=False)

    def pose(self, p: torch.Tensor):
        """p [..., 4] -> (mu [..., 2] in (-1, 1), sigma [..., 2] in [0.01, 4]), in the coordinates of x_n."""
        raw = self.pose_head(p)
        mu = torch.tanh(raw[..., :2])
        sigma = torch.exp(raw[..., 2:4].clamp(*self.log_sigma_range))
        return mu, sigma

    def gamma(self, u: torch.Tensor) -> torch.Tensor:
        """[..., 2] -> [..., 34] = [u, sin(pi 2^j u_x), cos(pi 2^j u_x), sin(pi 2^j u_y), cos(pi 2^j u_y)]."""
        angle = u.unsqueeze(-1) * self.freqs.to(u.dtype)                           # [..., 2, bands]
        return torch.cat((u, torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1).flatten(-2)), dim=-1)

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """slots [B, K, 64] -> features [B, N, 384], masks [B, K, N], mu [B, K, 2], sigma [B, K, 2]."""
        B, K, D = slots.shape
        N = self.grid.shape[0]
        mu, sigma = self.pose(slots[..., C.APPEARANCE_DIM:])
        u = (self.grid.to(slots.dtype)[None, None] - mu[:, :, None, :]) \
            / (sigma * C.DELTA).clamp_min(C.SCALE_EPS)[:, :, None, :]              # [B, K, N, 2]
        appearance = (slots * self.appearance_mask.to(slots.dtype)).view(B, K, 1, D).expand(B, K, N, D)
        feats, alpha = self.mlp(appearance + self.gamma_proj(self.gamma(u))).split((C.BACKBONE_DIM, 1), dim=-1)
        masks = torch.softmax(alpha.squeeze(-1), dim=1)
        return {"features": torch.sum(feats * masks[..., None], dim=1), "masks": masks, "mu": mu, "sigma": sigma}
