"""The two input grids of the corrector (Sec. 3.1, "Inputs").

  appearance grid  h_t = g_psi(f(x_t)) in R^{N x 64}: the frozen DINOv2-NoPE features through a small MLP
  position grid    c   in R^{N x 4}:  a fixed Fourier code of each patch's image coordinate, the same on every frame
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C


class FeatureMLP(nn.Module):
    """g_psi: LayerNorm(384) -> Linear(384, 768) -> ReLU -> Linear(768, 64), as in SlotContrast."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(C.BACKBONE_DIM)
        self.fc1 = nn.Linear(C.BACKBONE_DIM, C.FEATURE_HIDDEN)
        self.fc2 = nn.Linear(C.FEATURE_HIDDEN, C.FEATURE_DIM)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(self.norm(f))))


def patch_coordinates(side: int) -> torch.Tensor:
    """x_n for the N = side^2 patches, row-major (the order of the backbone's tokens): [N, 2] as (x, y), where x is
    the column and y the row, both in [-1, 1] with -1 and 1 at the centres of the border patches."""
    ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, side), torch.linspace(-1.0, 1.0, side), indexing="ij")
    return torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=-1)


def position_code(side: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """c: [N, 4] = [sin(pi/2 x), cos(pi/2 x), sin(pi/2 y), cos(pi/2 y)] (Supp. "Two streams").

    A single band at pi/2 is injective on [-1, 1], so no two patches share a code."""
    x = patch_coordinates(side).to(dtype=dtype, device=device)
    freq = torch.tensor([math.pi * C.POSITION_CODE_FREQ], dtype=torch.float32).to(dtype=dtype, device=device)
    angle = x.unsqueeze(-1) * freq                                     # [N, 2, 1]
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1).flatten(-2)
