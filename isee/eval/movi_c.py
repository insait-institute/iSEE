"""Object discovery on MOVi-C (Sec. 4, Table 3): video FG-ARI and video mBO of the decoder's masks.

Evaluated on the 250 clips of the MOVi-C validation split (TFDS movi_c/256x256:1.0.0), all 24 frames of each. A clip
is resized to the model's input (bicubic, 256 -> 336 px: 24 x 24 patches) and run through the encoder with TEN in one
recurrent pass over its 24 frames (`ISEE.forward`, no walker). The decoder reads the corrector's
output S~_t; its masks softmax_k(alpha) [K = 11, 24 x 24] are upsampled bilinearly to 336 x 336 and every pixel goes
to its argmax slot. The ground-truth segmentation is resized to 336 x 336 (nearest-exact). Per clip, over all
24 x 336 x 336 pixels (`isee.eval.segmentation`):

  FG-ARI  the adjusted Rand index between the ground-truth objects and the slots, over the pixels of objects
          (background pixels excluded)
  mBO     for every ground-truth object with at least one pixel, the largest IoU with any slot over the whole clip,
          a slot's area counted over all pixels (background included); averaged over those objects

Reported: the mean of each over the clips (mBO over the clips with at least one object, which is every clip).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from ..data.movi_c import clip_paths, read_clip, resize_segmentations
from ..data.video import preprocess
from ..model.isee import ISEE
from .segmentation import upsample_argmax, video_scores

DEFINITION = {
    "split": "MOVi-C validation (TFDS movi_c/256x256:1.0.0), every clip, all 24 frames",
    "model": "encoder + TEN, one recurrent pass over the 24 frames of a clip, no walker",
    "prediction": "decoder masks softmax_k(alpha) of the 11 slots, bilinear 24x24 -> 336x336, argmax over slots",
    "ground_truth": "segmentation ids (0 = background), nearest-exact 256x256 -> 336x336",
    "fg_ari": "video FG-ARI, higher is better: ARI between ground-truth objects and slots over all 24x336x336 "
              "pixels of a clip, background pixels excluded; mean over clips",
    "mbo": "video mBO, higher is better: per ground-truth object with pixels, the max over slots of the IoU over "
           "all 24x336x336 pixels of a clip (slot area over all pixels); mean over objects, then over clips",
}


@torch.no_grad()
def decoder_masks(model: ISEE, videos: Sequence[np.ndarray], size: int, device,
                  preprocess_device="cpu") -> np.ndarray:
    """uint8 clips [T, H, W, 3] (same T) -> the decoder's masks [B, T, K, N] as float32."""
    x = torch.stack([preprocess(v, size, preprocess_device) for v in videos]).to(device)
    return model(model.encode(x))["masks"].float().cpu().numpy()


def clip_scores(masks: np.ndarray, segmentations: np.ndarray, grid_side: int) -> Dict[str, float]:
    """masks [T, K, N] of one clip, segmentations [T, S, S] at the model's input size -> fg_ari, mbo."""
    T, K, _ = masks.shape
    pred = upsample_argmax(masks.reshape(T, K, grid_side, grid_side), segmentations.shape[1:])
    return video_scores(segmentations, pred, K)


def evaluate(model: ISEE, root: str | Path, size: int, device, batch_size: int, preprocess_device="cpu",
             log: Callable[[str], None] = print) -> dict:
    """The validation clips under `root` -> {"fg_ari", "mbo", "clips", "per_clip": [{"clip", "fg_ari", "mbo"}]}."""
    paths = clip_paths(root, "validation")
    if not paths:
        raise FileNotFoundError(f"no clips under {Path(root) / 'validation'}")
    model.eval()
    rows: List[dict] = []
    for i in range(0, len(paths), batch_size):
        clips = [read_clip(p) for p in paths[i:i + batch_size]]
        masks = decoder_masks(model, [v for v, _ in clips], size, device, preprocess_device)
        for p, (_, seg), m in zip(paths[i:i + batch_size], clips, masks):
            rows.append({"clip": p.stem, **clip_scores(m, resize_segmentations(seg, size), model.grid_side)})
        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(paths):
            log(f"{len(rows)} / {len(paths)} clips: FG-ARI {np.mean([r['fg_ari'] for r in rows]):.4f} "
                f"mBO {np.nanmean([r['mbo'] for r in rows]):.4f}")
    return {"fg_ari": float(np.mean([r["fg_ari"] for r in rows])),
            "mbo": float(np.nanmean([r["mbo"] for r in rows])),
            "clips": len(rows), "per_clip": rows}
