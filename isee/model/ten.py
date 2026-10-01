"""Temporal evidence normalisation, TEN (Sec. 3.2, Eqs. 4-6).

TEN decides, per slot and frame, whether the slot is HELD: while held, the appearance stream a^k keeps the value it
committed on the previous frame, so the slot keeps its object's appearance through an occlusion. TEN has no
parameters. Per slot k and frame t:

  evidence        e_t     = mean of the m = 4 largest values of A^k_t                                     (Eq. 4)
  local evidence  e_loc_t = max over 3 x 3 blocks of the block mean of A^k_t * W^k
  normalisation   ehat_loc_t = e_loc_t / r_loc_{t-1},   r_loc_t = max(r_loc_{t-1}, e_loc_t)            (Eq. 5)
                  ehat_rel_t = e_rel_t / r_rel_{t-1},   r_rel_t = max(r_rel_{t-1}, e_rel_t)
  onset           ehat_loc_t < tau  and  (min(1, e_loc_t / (kappa ebar_{t-1})))^8 < tau
  release         ehat_rel_t >= rho
  hold            u_t = onset (if u_{t-1} = 0),  u_t = [ehat_rel_t < rho] (if u_{t-1} = 1)                (Eq. 6)
                  a_t = u_t a_{t-1} + (1 - u_t) a~_t;  p_t = p~_t

The release evidence e_rel is the evidence e (`release: topk`) or the mean of the m largest values of A^k_t * W^k
(`release: window_topk`).

A^k_t is the corrector's attention (softmax over slots, before the renormalisation over patches) on its first
iteration, where it reflects the predicted slot rather than the updated one. W^k is the slot's final-iteration
attention on the last frame on which it was not held, scaled to a maximum of one and max-filtered over
`window_dilation` x `window_dilation` patches (no window on the first frame, which is never held). Blocks at the
border average over the patches inside the grid. ebar is the exponential average of e_loc at rate 0.5: a hold does
not start on a frame whose local evidence jumps well above its recent level. All references are initialised on the
first frame and updated on every frame, held or not.

With `track_scene`, the window of a held slot moves with the scene: every frame it is translated by the mean
displacement of the final-iteration attention centroids of the other slots that were not held on the previous frame
and cover less than 15 % of the frame, each weighted by exp(-d^2 / (2 * 4^2)), with d the distance in patches between
that slot's previous centroid and the window's centre.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import constants as C


@dataclass(frozen=True)
class HoldRule:
    tau: float = 0.5              # onset threshold on ehat_loc
    rho: float = 0.9              # release threshold on ehat_rel
    release: str = "topk"         # the release evidence: "topk" (e) or "window_topk"
    window_dilation: int = 5      # W^k: max filter size, in patches
    track_scene: bool = False     # the window of a held slot moves with the scene

    def __post_init__(self):
        if self.release not in ("topk", "window_topk"):
            raise ValueError(f"release must be 'topk' or 'window_topk', got {self.release!r}")


@dataclass
class TENState:
    r_rel: torch.Tensor       # [B, K] running maximum of e_rel
    r_loc: torch.Tensor       # [B, K] running maximum of e_loc
    ebar: torch.Tensor        # [B, K] exponential average of e_loc
    held: torch.Tensor        # [B, K] bool, u_t
    committed: torch.Tensor   # [B, K, 64] S^c_t, the state the predictor advances
    window_mask: torch.Tensor # [B, K, N] the slot's final-iteration attention on its last non-held frame
    centroid: torch.Tensor    # [B, K, 2] this frame's attention centroid of every slot (track_scene)
    window_centre: torch.Tensor  # [B, K, 2] the centroid of window_mask (track_scene)
    shift: torch.Tensor       # [B, K, 2] the window's accumulated translation while held (track_scene)


def _as_grid(x: torch.Tensor, side: int) -> torch.Tensor:
    B, K, N = x.shape
    return x.reshape(B * K, 1, side, side)


def topk_evidence(attn: torch.Tensor, m: int = C.TEN_TOP_M) -> torch.Tensor:
    """[B, K, N] -> [B, K], the mean of the m largest values of each slot."""
    return attn.topk(min(m, attn.shape[-1]), dim=-1).values.mean(-1)


def local_evidence(attn: torch.Tensor, window: Optional[torch.Tensor], side: int,
                   block: int = C.TEN_BLOCK) -> torch.Tensor:
    """e_loc: [B, K, N] -> [B, K], the largest block x block mean of attn * window."""
    B, K, N = attn.shape
    a = attn if window is None else attn * window
    pooled = F.avg_pool2d(_as_grid(a, side), block, stride=1, padding=block // 2, count_include_pad=False)
    return pooled.reshape(B, K, -1).amax(-1)


def window(mask: torch.Tensor, side: int, dilation: int) -> torch.Tensor:
    """W^k: a slot's attention [B, K, N] scaled to a maximum of one and max-filtered dilation x dilation."""
    B, K, N = mask.shape
    w = mask / mask.amax(-1, keepdim=True).clamp_min(1e-6)
    return F.max_pool2d(_as_grid(w, side), dilation, stride=1, padding=dilation // 2).reshape(B, K, N)


def centroids(mask: torch.Tensor, side: int) -> torch.Tensor:
    """[B, K, N] -> [B, K, 2] (column, row), the mass-weighted mean patch position."""
    ys, xs = torch.meshgrid(torch.arange(side, device=mask.device, dtype=mask.dtype),
                            torch.arange(side, device=mask.device, dtype=mask.dtype), indexing="ij")
    xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)
    return (mask @ xy) / mask.sum(-1, keepdim=True).clamp_min(1e-6)


def translate(x: torch.Tensor, shift: torch.Tensor, side: int) -> torch.Tensor:
    """x [B, K, N] on the grid translated by shift [B, K, 2] patches (column, row), bilinear, zero outside."""
    B, K, N = x.shape
    theta = torch.zeros(B * K, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, :, 2] = -shift.reshape(B * K, 2) * 2.0 / side
    grid = F.affine_grid(theta, (B * K, 1, side, side), align_corners=False)
    return F.grid_sample(_as_grid(x, side), grid, align_corners=False, padding_mode="zeros").reshape(B, K, N)


def scene_displacement(prev: torch.Tensor, now: torch.Tensor, open_prev: torch.Tensor, area: torch.Tensor,
                       centre: torch.Tensor) -> torch.Tensor:
    """[B, K, 2]: for each slot, the weighted mean displacement prev -> now of the other slots that were open on the
    previous frame and cover less than TEN_SCENE_MAX_AREA of the frame (zero if their total weight is <= 1e-3)."""
    d = now - prev
    ok = (open_prev & (area < C.TEN_SCENE_MAX_AREA)).to(now.dtype)
    dist2 = ((centre[:, :, None, :] - prev[:, None, :, :]) ** 2).sum(-1)                   # [B, K, J]
    w = torch.exp(-dist2 / (2 * C.TEN_SCENE_SIGMA ** 2)) * ok[:, None, :]
    w = w * (1 - torch.eye(w.shape[1], device=w.device, dtype=w.dtype))[None]
    total = w.sum(-1, keepdim=True)
    return (w @ d) / total.clamp_min(1e-6) * (total > 1e-3)


class TemporalEvidenceNormalisation:
    def __init__(self, grid_side: int, rule: HoldRule, held_dims: Tuple[int, int] = (0, C.APPEARANCE_DIM)):
        self.side, self.rule, self.held_dims = grid_side, rule, held_dims

    def _release_evidence(self, attn_first: torch.Tensor, W: Optional[torch.Tensor]) -> torch.Tensor:
        if self.rule.release == "topk" or W is None:
            return topk_evidence(attn_first)
        return topk_evidence(attn_first * W)

    @torch.no_grad()
    def __call__(self, updated: torch.Tensor, attn_first: torch.Tensor, attn_last: torch.Tensor,
                 state: Optional[TENState]):
        """One frame.

        updated     [B, K, 64]  the corrector's output S~_t
        attn_first  [B, K, N]   A_t of the corrector's first iteration
        attn_last   [B, K, N]   A_t of its last iteration
        state       TENState of frame t-1, or None on the first frame

        Returns (committed S^c_t, held u_t, e_loc_t, e_rel_t, new state)."""
        rule, side = self.rule, self.side
        attn_first, attn_last = attn_first.detach(), attn_last.detach()
        now = centroids(attn_last, side)
        if state is None:                                           # first frame: never held, references start here
            e_loc = local_evidence(attn_first, None, side)
            e_rel = self._release_evidence(attn_first, None)
            held = torch.zeros_like(e_loc, dtype=torch.bool)
            new = TENState(r_rel=e_rel.clone(), r_loc=e_loc.clone(), ebar=e_loc.clone(), held=held,
                           committed=updated, window_mask=attn_last, centroid=now, window_centre=now,
                           shift=torch.zeros_like(now))
            return updated, held, e_loc, e_rel, new

        W = window(state.window_mask, side, rule.window_dilation)
        shift = state.shift
        if rule.track_scene:
            area = attn_last.sum(-1) / attn_last.shape[-1]
            delta = scene_displacement(state.centroid, now, ~state.held, area, state.window_centre + state.shift)
            shift = torch.where(state.held[..., None], state.shift + delta, torch.zeros_like(delta))
            W = translate(W, shift, side)

        e_loc = local_evidence(attn_first, W, side)
        e_rel = self._release_evidence(attn_first, W)
        tiny = torch.finfo(e_loc.dtype).tiny
        ehat_loc = e_loc / state.r_loc.clamp_min(tiny)
        ehat_rel = e_rel / state.r_rel.clamp_min(tiny)
        guard = (e_loc / (C.TEN_GUARD_KAPPA * state.ebar).clamp_min(tiny)).clamp(max=1.0) ** C.TEN_GUARD_POWER
        onset = (ehat_loc < rule.tau) & (guard < rule.tau)
        held = torch.where(state.held, ehat_rel < rule.rho, onset)

        dims = torch.zeros(updated.shape[-1], dtype=torch.bool, device=updated.device)
        dims[self.held_dims[0]:self.held_dims[1]] = True
        committed = torch.where(held[..., None] & dims, state.committed, updated)

        new = TENState(r_rel=torch.maximum(state.r_rel, e_rel), r_loc=torch.maximum(state.r_loc, e_loc),
                       ebar=(1 - C.TEN_GUARD_EMA) * state.ebar + C.TEN_GUARD_EMA * e_loc, held=held,
                       committed=committed, window_mask=torch.where(held[..., None], state.window_mask, attn_last),
                       centroid=now, window_centre=torch.where(held[..., None], state.window_centre, now),
                       shift=shift)
        return committed, held, e_loc, e_rel, new
