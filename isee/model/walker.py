"""The walker: where a held slot's object is (Sec. 3.3, Eq. 7).

A held slot keeps its object's appearance, but nothing in it says where the hidden object is. The walker follows the
content around the object from frame to frame (the occluder that covers it, the container that carries it) on the
patch grid, and the position it reads out is written into the held slot before the predictor, so the slot's next
attention is centred where the object is expected to be.

  P*(n)   every patch n has a position P*(n) in the coordinates of the position stream: GRU_p iterated 200 times
          from zero, driven by v_p(c_n) alone (h <- GRU_p(v_p(c_n), h)); interpolated bilinearly between patches.
  e_t     a normalised embedding of the appearance grid h_t: LayerNorm and a 1x1 conv to 32 channels, plus a
          zero-initialised residual from a 3x3 convolutional GRU run forward over the clip, then a 1x1 conv and l2
          normalisation.
  T_t     T_t(n -> n') = softmax over n' in W(n) of ( s <e_{t-1,n}, e_{t,n'}> + beta_{n'-n} ),  W(n) = 11 x 11 patches
  walk    pi_t = pi_{t-1} T_t                                                                               (Eq. 7)
  restart pi is reset to a point mass whenever the slot can be trusted: at step t, if the slot was not held on steps
          t, t-1 and t-2 (t and t-1 at t = 1), pi_{t-1} <- one-hot at n_a, the patch whose P* is nearest the slot's
          position p_{t-1}, the anchor p_a <- p_{t-1}, and the walk then steps to t. A held slot is walked from the
          anchor of the last restart before its hold.
  readout p_hat_t = p_a + P*(xbar_t) - P*(n_a), with xbar_t the pi_t-weighted mean position over the 3 x 3 patches
          around the mode of pi_t smoothed by a 3 x 3 box; a walk that does not move returns p_a exactly.

The walker steps on every second frame: step j is frame 2j. It reads h_t in half precision. Three walkers trained from
three seeds are read together: each walks on its own transitions, their distributions are averaged, and the average
is read out.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C


# ---------------------------------------------------------------------------------------------------------------
# P*: the position the corrector writes for a point object at each patch
# ---------------------------------------------------------------------------------------------------------------
def _cell_code_float64(side: int) -> torch.Tensor:
    """c_n in float64 (the code of `features.position_code`, at double precision)."""
    r, col = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    xy = torch.stack([col.reshape(-1), r.reshape(-1)], -1).to(torch.float64) / (side - 1) * 2 - 1
    a = torch.pi / 2 * xy
    return torch.stack([torch.sin(a[:, 0]), torch.cos(a[:, 0]), torch.sin(a[:, 1]), torch.cos(a[:, 1])], -1)


@torch.no_grad()
def position_table(gru_p: nn.GRUCell, v_p_weight: torch.Tensor, side: int, iters: int = C.PSTAR_ITERS) -> torch.Tensor:
    """P* [N, 4], computed in float64 on the CPU and returned in float32."""
    gru = nn.GRUCell(C.POSITION_DIM, C.POSITION_DIM).double()
    gru.load_state_dict({k: v.double().cpu() for k, v in gru_p.state_dict().items()})
    u = _cell_code_float64(side) @ v_p_weight.detach().double().cpu().T
    h = torch.zeros(side * side, C.POSITION_DIM, dtype=torch.float64)
    for _ in range(iters):
        h = gru(u, h)
    return h.float()


def nearest_patch(p: torch.Tensor, pstar: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """[..., 4] -> [...]: the patch whose P* is nearest."""
    flat = p.reshape(-1, p.shape[-1]).to(pstar.dtype)
    out = torch.empty(flat.shape[0], dtype=torch.long, device=p.device)
    for i in range(0, flat.shape[0], chunk):
        out[i:i + chunk] = torch.cdist(flat[i:i + chunk], pstar).argmin(-1)
    return out.reshape(p.shape[:-1])


def table_distance(p: torch.Tensor, pstar: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """[..., 4] -> [...]: the distance from p to the nearest P*(n)."""
    flat = p.reshape(-1, p.shape[-1]).to(pstar.dtype)
    out = torch.empty(flat.shape[0], dtype=pstar.dtype, device=p.device)
    for i in range(0, flat.shape[0], chunk):
        out[i:i + chunk] = torch.cdist(flat[i:i + chunk], pstar).min(-1).values
    return out.reshape(p.shape[:-1])


def table_at(pstar: torch.Tensor, xy: torch.Tensor, side: int) -> torch.Tensor:
    """P* bilinearly interpolated at continuous patch positions xy [..., 2] = (column, row)."""
    T = pstar.reshape(side, side, -1)
    x = xy[..., 0].clamp(0, side - 1)
    y = xy[..., 1].clamp(0, side - 1)
    x0 = x.floor().clamp(max=side - 2)
    y0 = y.floor().clamp(max=side - 2)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0, y0 = x0.long(), y0.long()
    v00, v01 = T[y0, x0], T[y0, x0 + 1]
    v10, v11 = T[y0 + 1, x0], T[y0 + 1, x0 + 1]
    return (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy


def mode_mean(pi: torch.Tensor, side: int) -> torch.Tensor:
    """xbar: pi [R, N] -> [R, 2] (column, row), the pi-weighted mean position over the 3 x 3 patches around the mode of
    pi smoothed by a 3 x 3 box (a point mass returns its own patch exactly)."""
    R, N = pi.shape
    img = pi.reshape(R, 1, side, side)
    mode = F.avg_pool2d(img, 3, 1, 1, count_include_pad=False).reshape(R, N).argmax(-1)
    local = F.unfold(img, 3, padding=1)                                             # [R, 9, N]
    w = local.gather(2, mode[:, None, None].expand(R, 9, 1))[..., 0]               # [R, 9]
    r, c = torch.meshgrid(torch.arange(side, device=pi.device), torch.arange(side, device=pi.device), indexing="ij")
    base = torch.stack([c.reshape(-1), r.reshape(-1)], -1).to(pi.dtype)[mode]     # [R, 2]
    offsets = torch.tensor([[dx, dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=pi.dtype, device=pi.device)
    pos = base[:, None, :] + offsets[None]                                          # [R, 9, 2]
    wsum = w.sum(-1, keepdim=True)
    mean = (w[..., None] * pos).sum(1) / wsum.clamp_min(1e-30)
    return torch.where(wsum > 1e-30, mean, base)


def readout(pi: torch.Tensor, p_a: torch.Tensor, n_a: torch.Tensor, pstar: torch.Tensor, side: int) -> torch.Tensor:
    """p_hat = p_a + P*(xbar) - P*(n_a): pi [R, N], p_a [R, 4], n_a [R] -> [R, 4]."""
    return p_a + table_at(pstar, mode_mean(pi, side), side) - pstar[n_a]


# ---------------------------------------------------------------------------------------------------------------
# one walker
# ---------------------------------------------------------------------------------------------------------------
class ConvGRUCell(nn.Module):
    def __init__(self, dim: int, k: int = 3):
        super().__init__()
        self.zr = nn.Conv2d(2 * dim, 2 * dim, k, padding=k // 2)
        self.hn = nn.Conv2d(2 * dim, dim, k, padding=k // 2)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.hn(torch.cat([x, r * h], 1)))
        return (1 - z) * n + z * h


class Walker(nn.Module):
    """One walker: the embedding e_t and the transition T_t (about 6 x 10^4 parameters), and the width sigma of its
    training likelihood."""

    def __init__(self, grid_side: int, feat_dim: int = C.FEATURE_DIM, dim: int = C.WALKER_EMBED_DIM,
                 win: int = C.WALKER_WINDOW):
        super().__init__()
        self.side, self.n, self.dim, self.win, self.pad = grid_side, grid_side * grid_side, dim, win, win // 2
        self.norm = nn.LayerNorm(feat_dim)
        self.inp = nn.Conv2d(feat_dim, dim, 1)
        self.mem = ConvGRUCell(dim)
        self.mem_out = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.mem_out.weight)
        nn.init.zeros_(self.mem_out.bias)
        self.key = nn.Conv2d(dim, dim, 1)
        nn.init.orthogonal_(self.inp.weight)
        nn.init.orthogonal_(self.key.weight)
        nn.init.zeros_(self.inp.bias)
        nn.init.zeros_(self.key.bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(C.WALKER_SCALE_INIT)))   # s = exp(log_scale)
        beta = torch.zeros(win * win)                                                 # the prior over offsets
        beta[(win * win) // 2] = C.WALKER_STAY_INIT
        self.beta = nn.Parameter(beta)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(C.WALKER_SIGMA_INIT)))
        ones = F.unfold(torch.ones(1, 1, grid_side, grid_side), win, padding=self.pad)
        self.register_buffer("valid", ones > 0.5, persistent=False)   # offsets that stay inside the grid

    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp_min(C.WALKER_SIGMA_MIN)

    def embed(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, T, N, 64] -> e [B, T, 32, S, S], causal (the memory runs forward over the clip)."""
        B, T = h.shape[:2]
        S, d = self.side, self.dim
        z = self.inp(self.norm(h.float()).reshape(B * T, S, S, -1).permute(0, 3, 1, 2)).reshape(B, T, d, S, S)
        mem = torch.zeros(B, d, S, S, device=z.device, dtype=z.dtype)
        states = []
        for t in range(T):
            mem = self.mem(z[:, t], mem)
            states.append(mem)
        z = z + self.mem_out(torch.stack(states, 1).reshape(B * T, d, S, S)).reshape(B, T, d, S, S)
        return F.normalize(self.key(z.reshape(B * T, d, S, S)), dim=1).reshape(B, T, d, S, S)

    def transition(self, e0: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
        """e0, e1 [B, 32, S, S] -> T [B, W*W, N]: for each source patch, a distribution over the window offsets."""
        B = e0.shape[0]
        k = F.unfold(e1, self.win, padding=self.pad).reshape(B, self.dim, -1, self.n)
        logits = self.log_scale.exp() * torch.einsum("bdn,bdwn->bwn", e0.reshape(B, self.dim, self.n), k) \
            + self.beta[None, :, None]
        return torch.softmax(logits.masked_fill(~self.valid, float("-inf")), dim=1)

    def step(self, pi: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """pi [R, N], trans [R, W*W, N] -> pi_t = pi_{t-1} T_t, [R, N]."""
        return F.fold(pi[:, None, :] * trans, (self.side, self.side), self.win,
                      padding=self.pad).reshape(pi.shape[0], self.n)


# ---------------------------------------------------------------------------------------------------------------
# the walk inside the model's recurrence
# ---------------------------------------------------------------------------------------------------------------
class OnlineWalk:
    """The ensemble's walk over one batch of clips, advanced frame by frame by the model (Supp. Algorithm 1).

    At frame t the predictor builds the slots of frame t+1, so the walk must say where a held object is at t+1 without
    reading any frame after it. On an even frame t the walk advances to step t/2 (with the restart rule) and its state
    is kept; on an odd frame it takes one provisional step to (t+1)/2, whose transition reads h_{t+1}, and discards
    it. The readout of either is written into the position stream of every slot that is held at t and whose anchor
    p_a lies within `dmax` of P* (a slot whose position is off the table was not anchored on an object)."""

    def __init__(self, walkers: Sequence[Walker], pstar: torch.Tensor, h: torch.Tensor, num_slots: int, dmax: float):
        self.walkers, self.pstar, self.dmax = list(walkers), pstar, dmax
        self.side = self.walkers[0].side
        B = h.shape[0]
        self.K = num_slots
        with torch.no_grad():
            steps = h[:, ::C.WALKER_STRIDE].half()
            self.e = [w.embed(steps) for w in self.walkers]                          # [B, S, 32, side, side] each
        self.n_steps = steps.shape[1]
        self.rows_b = torch.arange(B, device=h.device).repeat_interleave(num_slots)   # row = (clip, slot)
        R, N = B * num_slots, self.pstar.shape[0]
        self.pi = torch.zeros(len(self.walkers), R, N, device=h.device)
        self.p_a = torch.zeros(R, C.POSITION_DIM, device=h.device)
        self.n_a = torch.zeros(R, dtype=torch.long, device=h.device)
        self.updated: List[torch.Tensor] = []
        self.position: List[torch.Tensor] = []
        self._trans = {}

    def transitions(self, s: int) -> torch.Tensor:
        """[M, B, W*W, N]: every walker's transition from step s-1 to s."""
        if s not in self._trans:
            self._trans = {k: v for k, v in self._trans.items() if k >= s - 1}
            self._trans[s] = torch.stack([w.transition(e[:, s - 1], e[:, s]) for w, e in zip(self.walkers, self.e)])
        return self._trans[s]

    def step(self, pi: torch.Tensor, s: int) -> torch.Tensor:
        trans = self.transitions(s)
        return torch.stack([w.step(pi[m], trans[m][self.rows_b]) for m, w in enumerate(self.walkers)])

    def advance(self, s: int, updated: torch.Tensor, position: torch.Tensor):
        """Step s: updated [R] (not held), position [R, 4] (the committed position stream)."""
        self.updated.append(updated)
        self.position.append(position)
        if s == 0:
            n = nearest_patch(position, self.pstar)
            self.pi = F.one_hot(n, self.pstar.shape[0]).float()[None].expand(len(self.walkers), -1, -1).clone()
            self.p_a, self.n_a = position.clone(), n
            return
        trusted = updated & self.updated[s - 1] & (self.updated[s - 2] if s >= 2 else torch.ones_like(updated))
        n = nearest_patch(self.position[s - 1], self.pstar)
        self.pi = torch.where(trusted[None, :, None], F.one_hot(n, self.pstar.shape[0]).float()[None], self.pi)
        self.p_a = torch.where(trusted[:, None], self.position[s - 1], self.p_a)
        self.n_a = torch.where(trusted, n, self.n_a)
        self.pi = self.step(self.pi, s)

    @torch.no_grad()
    def __call__(self, t: int, committed: torch.Tensor, held: torch.Tensor):
        """Frame t. committed [B, K, 64], held [B, K] -> (the predictor's input [B, K, 64], the walked position of
        every slot at step t // 2 [B, K, 4], the slots whose position was written [B, K])."""
        B, K, D = committed.shape
        if t % 2 == 0:
            self.advance(t // 2, (~held).reshape(-1), committed[..., C.APPEARANCE_DIM:].reshape(B * K, -1).float())
            pi = self.pi
            self.walked = readout(pi.mean(0), self.p_a, self.n_a, self.pstar, self.side).reshape(B, K, -1)
        elif (t + 1) // 2 < self.n_steps:
            pi = self.step(self.pi, (t + 1) // 2)
        else:
            return committed, self.walked, torch.zeros_like(held)
        write = held.reshape(-1) & (table_distance(self.p_a, self.pstar) <= self.dmax)
        out = committed.reshape(B * K, D).clone()
        out[write, C.APPEARANCE_DIM:] = readout(pi.mean(0)[write], self.p_a[write], self.n_a[write], self.pstar,
                                                self.side).to(out.dtype)
        return out.reshape(B, K, D), self.walked, write.reshape(B, K)
