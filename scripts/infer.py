#!/usr/bin/env python
"""Run iSEE on one clip.

    python scripts/infer.py configs/lacater_static.yaml --video path/to/clip.avi --out outputs/infer

Writes <out>/<clip>.npz (below) and <out>/<clip>.mp4 (the frame with a cross at the reported position of every held
slot | the decoder's segmentation).

  held               [T, K]        u_t, TEN's held indicator
  state              [T, K, 64]    S^c_t, the committed state (a held slot keeps its appearance stream [0:60])
  state_updated      [T, K, 64]    S~_t, the corrector's output (what the decoder reads)
  masks              [T, K, S, S]  the decoder's masks (the segmentation), float16
  grouping           [T, K, S, S]  the corrector's last-iteration attention, float16
  position           [T, K, 4]     the reported position: the walked position on held frames, p_t elsewhere
  center_xy          [T, K, 2]     the decoder's placement mu of `position`, in pixels of the model's input
  position_walk      [T, K, 4]     (configs with a walker) every slot's walked position
  written            [T, K]        (configs with a walker) the slots whose position was written before the predictor

A frame is stretched to the model's square input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isee.config import build_model, build_walkers, grid_side, load_config, set_precision  # noqa: E402
from isee.data.video import preprocess, read_video  # noqa: E402
from isee.model import constants as C  # noqa: E402
from isee.visualize import render, to_pixels, write_video  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/infer")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tf32", action="store_true", help="TF32 matrix products (the paper's evaluation setting)")
    a = ap.parse_args()
    set_precision(a.tf32)
    cfg = load_config(a.config)
    size, side = cfg["model"]["input_size"], grid_side(cfg)
    frames = read_video(a.video, a.max_frames)
    x = preprocess(frames, size, a.device)
    view = ((x * torch.tensor(C.IMAGENET_STD, device=x.device)[:, None, None]
             + torch.tensor(C.IMAGENET_MEAN, device=x.device)[:, None, None]).clamp(0, 1) * 255)
    view = view.round().byte().permute(0, 2, 3, 1).cpu().numpy()

    model = build_model(cfg, a.device).eval()
    walkers, dmax = build_walkers(cfg, a.device)
    h = model.encode(x[None])
    out = model(h, walkers, dmax)
    p = out["state_updated"][0, ..., C.APPEARANCE_DIM:]
    result = {k: out[k][0] for k in ("held", "state", "state_updated")}
    result["position"] = torch.where(out["held"][0, ..., None], out["position_walk"][0], p) if walkers else p
    if walkers:
        result["position_walk"], result["written"] = out["position_walk"][0], out["written"][0]
    mu, _ = model.decoder.pose(result["position"])
    result["center_xy"] = torch.from_numpy(to_pixels(mu.cpu().numpy(), side, size, size))
    for k in ("masks", "grouping"):
        result[k] = out[k][0].reshape(*out[k].shape[1:3], side, side).half()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.video).stem
    np.savez_compressed(out_dir / f"{stem}.npz", **{k: v.cpu().numpy() for k, v in result.items()})
    held = result["held"].cpu().numpy()
    print(f"{len(frames)} frames; held {held.mean():.1%} of slot-frames -> {out_dir / (stem + '.npz')}")
    viz = render(view, out["masks"][0].float().cpu().numpy(), held, result["center_xy"].numpy(), side)
    print(f"video -> {write_video(viz, out_dir / f'{stem}.mp4')}")


if __name__ == "__main__":
    main()
