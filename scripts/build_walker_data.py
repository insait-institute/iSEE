#!/usr/bin/env python
"""The walker's training data (Sec. 3.3, "Training"): the trained encoder with TEN over the first `clips` training
clips (sorted by name), whole 300-frame clips, read at the walker's rate (frames 0, 2, ..., 298).

    python scripts/build_walker_data.py configs/lacater_static.yaml

TEN uses the config's `walker_training.hold_rule`. Writes to `walker_training.data`:
    h.npy         float16 [clips, 150, N, 64]  the appearance grid h_t
    position.npy  float16 [clips, 150, K, 4]   every slot's committed position stream p_t
    held.npy      uint8   [clips, 150, K]      TEN's held flags u_t
    pstar.npy     float32 [N, 4]               the encoder's P*
    meta.json     the clips, in order
Clips are processed 8 at a time in one recurrent batch; rerunning resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isee.config import build_model, load_config, resolve, set_precision  # noqa: E402
from isee.data import lacater  # noqa: E402
from isee.data.video import preprocess  # noqa: E402
from isee.model import constants as C  # noqa: E402
from isee.model.ten import HoldRule, TemporalEvidenceNormalisation  # noqa: E402

BATCH_CLIPS = 8


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    w = cfg["walker_training"]
    set_precision(True)
    device = "cuda"
    model = build_model(cfg, device).eval()
    model.ten = TemporalEvidenceNormalisation(model.grid_side, HoldRule(**w["hold_rule"]))
    root = resolve(cfg["training"]["data"]["root"])
    clips = lacater.clip_names(root, "train")[:w["clips"]]
    out = resolve(w["data"])
    out.mkdir(parents=True, exist_ok=True)
    n, S, N, K = len(clips), lacater.N_FRAMES // C.WALKER_STRIDE, model.grid_side ** 2, model.num_slots
    mode = "r+" if (out / "done.npy").exists() else "w+"
    h_out = open_memmap(out / "h.npy", mode=mode, dtype=np.float16, shape=(n, S, N, C.FEATURE_DIM))
    p_out = open_memmap(out / "position.npy", mode=mode, dtype=np.float16, shape=(n, S, K, C.POSITION_DIM))
    u_out = open_memmap(out / "held.npy", mode=mode, dtype=np.uint8, shape=(n, S, K))
    done = np.load(out / "done.npy") if mode == "r+" else np.zeros(n, np.uint8)
    np.save(out / "pstar.npy", model.position_table().numpy())
    (out / "meta.json").write_text(json.dumps({"config": str(a.config), "hold_rule": w["hold_rule"],
                                               "frames": "0, 2, ..., 298", "clips": clips}, indent=1))
    for i in range(0, n, BATCH_CLIPS):
        if done[i:i + BATCH_CLIPS].all():
            continue
        h = []
        for clip in clips[i:i + BATCH_CLIPS]:
            frames = lacater.read_clip(root, "train", clip)
            x = preprocess(frames, cfg["model"]["input_size"], device)
            h.append(model.encode(x[None]))
        h = torch.cat(h)
        res = model(h, decode=False)
        steps = slice(0, None, C.WALKER_STRIDE)
        h_out[i:i + len(h)] = h[:, steps].half().cpu().numpy()
        p_out[i:i + len(h)] = res["state"][:, steps, :, C.APPEARANCE_DIM:].half().cpu().numpy()
        u_out[i:i + len(h)] = res["held"][:, steps].to(torch.uint8).cpu().numpy()
        for arr in (h_out, p_out, u_out):
            arr.flush()
        done[i:i + len(h)] = 1
        np.save(out / "done.npy", done)
        print(f"{int(done.sum())}/{n} clips; held {res['held'].float().mean():.3f} of slot-frames", flush=True)


if __name__ == "__main__":
    main()
