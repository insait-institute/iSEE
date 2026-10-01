"""Object discovery: video FG-ARI and video mBO of the decoder's masks (Sec. 4, Table 3).

Both are computed from a contingency table N[c, k], the number of pixels of ground-truth class c (0 = background)
that slot k owns, summed over all frames of a clip, so a slot must keep its object across time to score ("video"
metrics). The prediction of a pixel is the argmax over slots of the decoder's masks, bilinearly upsampled to the
ground-truth resolution.

  FG-ARI  the adjusted Rand index of the table without its background row
  mBO     for every ground-truth object with at least one pixel, the largest IoU with any slot, where a slot's area is
          counted over all pixels (background included); averaged over those objects
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def contingency(gt: np.ndarray, pred: np.ndarray, n_true: int, K: int) -> np.ndarray:
    """gt [T, H, W] int (0 = background), pred [T, H, W] int in [0, K) -> per-frame tables [T, n_true, K]."""
    T = gt.shape[0]
    flat = gt.reshape(T, -1).astype(np.int64) * K + pred.reshape(T, -1).astype(np.int64)
    out = np.zeros((T, n_true * K), np.int64)
    for t in range(T):
        out[t] = np.bincount(flat[t], minlength=n_true * K)
    return out.reshape(T, n_true, K)


def ari_from_table(N: np.ndarray) -> np.ndarray:
    """N [..., C, K] counts (background row already removed) -> [...] ARI (1 where it is undefined)."""
    N = N.astype(np.float64)
    A, B = N.sum(-1), N.sum(-2)
    n = A.sum(-1)
    rindex = (N * (N - 1)).sum((-2, -1))
    aindex = (A * (A - 1)).sum(-1)
    bindex = (B * (B - 1)).sum(-1)
    expected = aindex * bindex / np.clip(n * (n - 1), 1, None)
    denom = (aindex + bindex) / 2 - expected
    with np.errstate(invalid="ignore", divide="ignore"):
        ari = (rindex - expected) / denom
    return np.where(denom != 0.0, ari, 1.0)


def mbo_from_table(N: np.ndarray) -> np.ndarray:
    """N [..., 1 + C, K] counts, background row 0 included -> [...] mBO over the objects with pixels (NaN if none)."""
    N = N.astype(np.float64)
    slot_area = N.sum(-2)[..., None, :]                   # over all pixels, background included
    N = N[..., 1:, :]
    obj_area = N.sum(-1)[..., :, None]
    union = obj_area + slot_area - N
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, N / np.where(union > 0, union, 1), 0.0)
    best = iou.max(-1)
    active = obj_area[..., 0] > 0
    n_active = active.sum(-1)
    return np.where(n_active > 0, np.where(active, best, 0.0).sum(-1) / np.maximum(n_active, 1), np.nan)


def upsample_argmax(masks: np.ndarray, hw, chunk: int = 25) -> np.ndarray:
    """masks [T, K, g, g] -> [T, H, W] int: bilinear upsampling to hw, then the argmax over slots."""
    out = []
    for i in range(0, len(masks), chunk):
        x = torch.from_numpy(masks[i:i + chunk].astype(np.float32))
        out.append(F.interpolate(x, size=tuple(hw), mode="bilinear").argmax(1).numpy().astype(np.int16))
    return np.concatenate(out, 0)


def video_scores(gt: np.ndarray, pred: np.ndarray, K: int) -> dict:
    """gt [T, H, W] (0 = background), pred [T, H, W] -> video FG-ARI and video mBO of one clip."""
    N = contingency(gt, pred, int(gt.max()) + 1, K).sum(0)
    return {"fg_ari": float(ari_from_table(N[1:])), "mbo": float(mbo_from_table(N))}
