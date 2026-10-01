#!/usr/bin/env python
"""Evaluate iSEE on LA-CATER (Tables 1, 2 and 3 of the paper; definitions in isee/eval/lacater.py).

    python scripts/eval_lacater.py configs/lacater_static.yaml
    python scripts/eval_lacater.py configs/lacater_moving.yaml

Two stages, both driven by the config's `evaluation` block:
  run     (GPU) the model with TEN and the walkers over every clip of the eval set and over the first
          `discovery_clips` test clips, `batch_clips` clips per recurrent batch, frames resized on
          `preprocess_device`, TF32 as the config says. Per clip:
            <out>/permanence/<clip>.npz   position f16 [300, K, 4], held u8 [300, K], walked f32 [300, K, 4],
                                          grouping f16 [300, K, g, g]
            <out>/discovery/<clip>.npz    masks f16 [300, K, g, g] (the decoder's)
          A clip already written is not run again.
  score   (CPU) Tables 1-3 from those files and the ground truth -> <out>/results.json, printed as tables.
`--stage score` rescores without a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isee.config import build_model, build_walkers, grid_side, load_config, resolve, set_precision  # noqa: E402
from isee.data import lacater  # noqa: E402
from isee.data.video import preprocess  # noqa: E402
from isee.eval import lacater as E  # noqa: E402
from isee.model import constants as C  # noqa: E402


@torch.no_grad()
def run(cfg: dict, clips: list, out: Path, what: str):
    e, size, side = cfg["evaluation"], cfg["model"]["input_size"], grid_side(cfg)
    root = resolve(e["data"])
    todo = [c for c in clips if not (out / f"{c}.npz").exists()]
    if not todo:
        return
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model = build_model(cfg, device).eval()
    walkers, dmax = build_walkers(cfg, device)
    for i in range(0, len(clips), e["batch_clips"]):
        batch = clips[i:i + e["batch_clips"]]
        if all((out / f"{c}.npz").exists() for c in batch):
            continue
        h = []
        for clip in batch:
            frames = lacater.read_clip(root, "test", clip)
            x = preprocess(frames, size, e["preprocess_device"]).to(device)
            h.append(model.encode(x[None]))
        res = model(torch.cat(h), walkers=walkers, walk_dmax=dmax, decode=what == "discovery")
        T, K = res["held"].shape[1:]
        for b, clip in enumerate(batch):
            if what == "permanence":
                arrays = dict(position=res["state"][b, ..., C.APPEARANCE_DIM:].half(),
                              held=res["held"][b].to(torch.uint8), walked=res["position_walk"][b].float(),
                              grouping=res["grouping"][b].reshape(T, K, side, side).half())
            else:
                arrays = dict(masks=res["masks"][b].reshape(T, K, side, side).half())
            np.savez_compressed(out / f"{clip}.npz", **{k: v.cpu().numpy() for k, v in arrays.items()})
        print(f"[{what}] {min(i + len(batch), len(clips))}/{len(clips)} clips", flush=True)


def score(cfg: dict, eval_set: dict, discovery: list, out: Path) -> dict:
    e = cfg["evaluation"]
    masks = resolve(e["masks"])
    records = {}
    for c in eval_set["clips"]:
        with np.load(out / "permanence" / f"{c}.npz") as z:
            records[c] = E.clip_record({k: z[k] for k in z.files}, masks / f"{c}.npz")
    perm = E.permanence_table(records, eval_set["cases"], resolve(e["data"]))
    same = E.same_table(records, eval_set["cases"])
    per_clip = []
    for c in discovery:
        with np.load(out / "discovery" / f"{c}.npz") as z:
            dec = z["masks"]
        modal, _, _ = E.read_masks(masks / f"{c}.npz")
        per_clip.append({"clip": c, **E.discovery_scores(dec, modal)})
    disc = {k: 100.0 * float(np.mean([r[k] for r in per_clip])) for k in ("fg_ari", "mbo")}
    return {"camera": eval_set["camera"], "model": cfg["model"]["checkpoint"], "walkers": cfg["walker"]["checkpoints"],
            "permanence": perm, "same": same,
            "discovery": {**disc, "n_clips": len(per_clip), "per_clip": per_clip}}


def print_tables(r: dict):
    t = r["permanence"]["table"]
    print(f"\nTable 1: own-box mAP@[0.1:0.3] while hidden, {r['camera']} camera, "
          f"{r['permanence']['n_occlusions']} occlusions (higher is better)")
    print(f"{'':10s}" + "".join(f"{lab:>14s}" for lab in E.LABELS))
    for row in ("iSEE", "frozen"):
        print(f"{row:10s}" + "".join(f"{t[row][lab]['mAP']:9.1f} ({t[row][lab]['n']:3d})" for lab in E.LABELS))
    s = r["same"]["table"]
    print(f"\nTable 2: SAME (%) by occlusion length in frames, {r['same']['n_scored']} of "
          f"{r['same']['n_occlusions']} occlusions scored (higher is better)")
    print("".join(f"{b:>14s}" for b in s))
    print("".join(f"{s[b]['SAME']:9.1f} ({s[b]['n']:3d})" for b in s))
    d = r["discovery"]
    print(f"\nTable 3: object discovery on the first {d['n_clips']} test clips (higher is better)")
    print(f"video FG-ARI {d['fg_ari']:.1f}   video mBO {d['mbo']:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", help="default: outputs/<config name>")
    ap.add_argument("--stage", choices=("all", "run", "score"), default="all")
    a = ap.parse_args()
    cfg = load_config(a.config)
    e = cfg["evaluation"]
    out = Path(a.out or resolve("outputs") / Path(a.config).stem)
    eval_set = E.load_eval_set(resolve(e["occlusions"]))
    discovery = lacater.clip_names(resolve(e["data"]), "test")[:e["discovery_clips"]]
    if a.stage in ("all", "run"):
        set_precision(e["tf32"])
        run(cfg, eval_set["clips"], out / "permanence", "permanence")
        run(cfg, discovery, out / "discovery", "discovery")
    if a.stage in ("all", "score"):
        r = score(cfg, eval_set, discovery, out)
        (out / "results.json").write_text(json.dumps(r, indent=1))
        print_tables(r)
        print(f"\n-> {out / 'results.json'}")


if __name__ == "__main__":
    main()
