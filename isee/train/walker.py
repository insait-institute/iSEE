"""Training the walker (Sec. 3.3, "Training"; Supp. "Walker").

The walker is trained after the encoder, which stays frozen, on the encoder's own outputs over training clips at the
walker's rate (step j = frame 2j): the appearance grid h, every slot's committed position p and TEN's held flags u.
No annotation enters.

Rows. One row per (clip, slot) that has a scored return, plus pseudo-hides:
  return    a step e at which the slot is not held after being held at e-1 (and not held at e+1), with a restart
            before it. The walker, restarted before the hold and stepped through it, must explain the slot's
            position one step after the return, y = p_{e+1}.
  pseudo    up to 8 per clip: a stretch of a slot that is not held for L + 5 steps, L ~ U{1..8} (one L per batch), with
            the L steps after the first three marked as held and the step after them scored as a return. Stretches
            whose anchor and target patches are at least two patches apart are drawn five times as often.
  Returns and pseudo-hides whose anchor position p_a or target y lies farther than `dmax` from P* are not scored.

Loss at a scored return (Eq. 8), with pi_{e-1} the walk just before the return and sigma a learned width:
    -log sum_n pi_{e-1}(n) N(y; p_a + P*(n) - P*(n_a), sigma^2 I)
weighted by 1 + (hidden frames), averaged over returns; the same for the pseudo-hides, and the two are added.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..model.walker import Walker, nearest_patch, table_distance

TARGET_LAG = 1      # the target is the position one step after the return


def restart_steps(updated: torch.Tensor) -> torch.Tensor:
    """[..., T] (1 = not held) -> 1 on the steps where the walk restarts: not held at t-2, t-1 and t (t-1, t at
    t = 1)."""
    trust = torch.zeros_like(updated)
    trust[..., 1] = updated[..., 1] * updated[..., 0]
    trust[..., 2:] = updated[..., 2:] * updated[..., 1:-1] * updated[..., :-2]
    return trust


def last_before(flag: torch.Tensor) -> torch.Tensor:
    """[..., T] -> the index of the last step strictly before t with flag 1 (-1 if none)."""
    T = flag.shape[-1]
    idx = torch.arange(T, device=flag.device).expand(flag.shape)
    last = torch.cummax(torch.where(flag > 0.5, idx, torch.full_like(idx, -1)), dim=-1).values
    return torch.cat([torch.full_like(last[..., :1], -1), last[..., :-1]], dim=-1)


def returns(updated: torch.Tensor) -> torch.Tensor:
    """[..., T] -> 1 at a scored return e: held at e-1, not held at e and e+1, and a restart before e."""
    T = updated.shape[-1]
    ret = torch.zeros_like(updated)
    lo, hi = 1, T - TARGET_LAG
    r = updated[..., lo:hi] * (1 - updated[..., lo - 1:hi - 1]) * updated[..., lo + 1:hi + 1]
    anchor = last_before(restart_steps(updated)) >= 0
    ret[..., lo:hi] = r * anchor[..., lo:hi].to(updated.dtype)
    return ret


def build_rows(position: torch.Tensor, updated: torch.Tensor, pstar: torch.Tensor, side: int, *, pseudo: int,
               pseudo_len=(1, 8), moved_boost: float = 4.0, dmax: float, gen: Optional[torch.Generator] = None) -> dict:
    """position [B, T, K, 4], updated [B, T, K] -> rows: b, k [R], updated [R, T], position [R, T, 4], score [R, T],
    pseudo [R] (bool)."""
    B, T, K = updated.shape
    dev = updated.device
    ret = returns(updated.transpose(1, 2)).transpose(1, 2)                                     # [B, T, K]
    b, k = (ret.sum(1) > 0).nonzero(as_tuple=True)
    rows = dict(b=b, k=k, updated=updated[b, :, k], position=position[b, :, k], score=ret[b, :, k],
                pseudo=torch.zeros(len(b), dtype=torch.bool, device=dev))
    if pseudo <= 0:
        return rows
    L = int(torch.randint(pseudo_len[0], pseudo_len[1] + 1, (1,), generator=gen).item())
    span = 3 + L + TARGET_LAG + 1                    # steps h-3 .. e+1 with e = h + L, all not held
    if span > T:
        return rows
    cs = torch.cat([torch.zeros(B, 1, K, device=dev), updated.cumsum(1)], 1)
    full = (cs[:, span:] - cs[:, :-span]) >= span - 0.5                                        # by start step s
    s_idx = torch.arange(T - span + 1, device=dev)
    dist = table_distance(position, pstar)                                                     # [B, T, K]
    full = full & (dist[:, s_idx + 2] <= dmax) & (dist[:, s_idx + 3 + L + TARGET_LAG] <= dmax)
    cand = full.nonzero(as_tuple=False)                                                        # [M, 3] (b, s, k)
    if len(cand) == 0:
        return rows
    cb, cs_, ck = cand.unbind(1)
    r, c = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    xy = torch.stack([c.reshape(-1), r.reshape(-1)], -1).float().to(dev)
    n_anchor = nearest_patch(position[cb, cs_ + 2, ck], pstar)
    n_target = nearest_patch(position[cb, cs_ + 3 + L + TARGET_LAG, ck], pstar)
    w = 1.0 + moved_boost * ((xy[n_anchor] - xy[n_target]).abs().amax(-1) >= 2).float()
    n = min(pseudo * B, len(cand))
    pick = torch.multinomial(w.cpu(), n, replacement=False, generator=gen).to(dev)
    pb, pk, ph = cb[pick], ck[pick], cs_[pick] + 3
    tt = torch.arange(T, device=dev)[None]
    pu = updated[pb, :, pk] * ~((tt >= ph[:, None]) & (tt < (ph + L)[:, None]))
    sc = torch.zeros_like(pu)
    sc[torch.arange(n, device=dev), ph + L] = 1.0
    return dict(b=torch.cat([rows["b"], pb]), k=torch.cat([rows["k"], pk]),
                updated=torch.cat([rows["updated"], pu]), position=torch.cat([rows["position"], position[pb, :, pk]]),
                score=torch.cat([rows["score"], sc]),
                pseudo=torch.cat([rows["pseudo"], torch.ones(n, dtype=torch.bool, device=dev)]))


def transitions(walker: Walker, h: torch.Tensor) -> torch.Tensor:
    """h [B, T, N, 64] -> [B, T, W*W, N]; entry t is the step t-1 -> t (entry 0 is unused). Each transition is
    recomputed in the backward pass instead of stored."""
    e = walker.embed(h)
    B, T = e.shape[:2]
    out = [torch.zeros(B, walker.win * walker.win, walker.n, device=e.device, dtype=e.dtype)]
    for t in range(1, T):
        out.append(checkpoint(walker.transition, e[:, t - 1], e[:, t], use_reentrant=False)
                   if torch.is_grad_enabled() else walker.transition(e[:, t - 1], e[:, t]))
    return torch.stack(out, 1)


def return_loss(walker: Walker, pstar: torch.Tensor, h: torch.Tensor, position: torch.Tensor, updated: torch.Tensor,
                *, dmax: float, pseudo: int = 0, pseudo_len=(1, 8), moved_boost: float = 4.0,
                gen: Optional[torch.Generator] = None) -> dict:
    """One batch of clips at the walker's rate: h [B, T, N, 64], position [B, T, K, 4], updated [B, T, K].
    Returns the weighted return NLL (`nll`), the pseudo-hide NLL (`nll_pseudo`) and their counts."""
    rows = build_rows(position, updated, pstar, walker.side, pseudo=pseudo, pseudo_len=pseudo_len,
                      moved_boost=moved_boost, dmax=dmax, gen=gen)
    zero = torch.zeros((), device=h.device)
    if len(rows["b"]) == 0:
        return {"nll": zero, "nll_pseudo": zero, "n_returns": 0.0, "n_pseudo": 0.0}
    trans = transitions(walker, h)
    upd, pos, score, rb, is_pseudo = rows["updated"], rows["position"], rows["score"], rows["b"], rows["pseudo"]
    R, T = upd.shape
    N = pstar.shape[0]
    trust = restart_steps(upd)
    prev_updated = last_before(upd)                                        # the last step before t not held
    cells = nearest_patch(pos, pstar)                                      # [R, T]
    scored = score > 0
    far = table_distance(pos, pstar) > dmax
    far_target = torch.zeros_like(far)
    far_target[:, :T - TARGET_LAG] = far[:, TARGET_LAG:]
    far_anchor = far.gather(1, (last_before(trust) - 1).clamp_min(0))
    scored = scored & ~far_target & ~far_anchor
    steps = set(torch.nonzero(scored.any(0)).flatten().tolist())
    sigma = walker.sigma()
    pi = F.one_hot(cells[:, 0], N).to(trans.dtype)
    p_a, n_a = pos[:, 0], cells[:, 0]
    ps = is_pseudo.to(trans.dtype)
    acc = {k: torch.zeros((), device=h.device) for k in ("real", "w_real", "pseudo", "w_pseudo", "n_real", "n_pseudo")}
    for t in range(1, T):
        if t in steps:
            m = scored[:, t].to(trans.dtype)
            y = pos[:, t + TARGET_LAG]
            hyp = (p_a - pstar[n_a])[:, None, :] + pstar[None]                             # [R, N, 4]
            d2 = (y[:, None, :] - hyp).pow(2).sum(-1)
            logp = torch.log(pi.clamp_min(1e-30)) - d2 / (2 * sigma ** 2) - 4 * torch.log(sigma) \
                - 2 * math.log(2 * math.pi)
            nll = -torch.logsumexp(logp, -1)
            hidden_frames = ((t - 1 - prev_updated[:, t]) * 2).to(trans.dtype)
            w = (1.0 + hidden_frames) * m
            acc["real"] = acc["real"] + (nll * w * (1 - ps)).sum()
            acc["w_real"] = acc["w_real"] + (w * (1 - ps)).sum()
            acc["pseudo"] = acc["pseudo"] + (nll * w * ps).sum()
            acc["w_pseudo"] = acc["w_pseudo"] + (w * ps).sum()
            acc["n_real"] = acc["n_real"] + (m * (1 - ps)).sum().detach()
            acc["n_pseudo"] = acc["n_pseudo"] + (m * ps).sum().detach()
        restart = trust[:, t] > 0.5
        pi = torch.where(restart[:, None], F.one_hot(cells[:, t - 1], N).to(pi.dtype), pi)
        p_a = torch.where(restart[:, None], pos[:, t - 1], p_a)
        n_a = torch.where(restart, cells[:, t - 1], n_a)
        pi = walker.step(pi, trans[:, t][rb])
    return {"nll": acc["real"] / acc["w_real"].clamp_min(1e-12),
            "nll_pseudo": acc["pseudo"] / acc["w_pseudo"].clamp_min(1e-12),
            "n_returns": float(acc["n_real"]), "n_pseudo": float(acc["n_pseudo"])}
