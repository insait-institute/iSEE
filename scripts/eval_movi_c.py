#!/usr/bin/env python
"""Evaluate iSEE on MOVi-C: object discovery (Table 3), video FG-ARI and video mBO of the decoder's masks on the
validation split (definitions in isee/eval/movi_c.py).

    python scripts/eval_movi_c.py configs/movi_c.yaml

Settings from the config's `evaluation` block (data root, clips per batch, where frames are resized, TF32). Prints both means (x 100) and writes
<out>/results.json: the means, every clip's values, the definitions and the settings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isee.config import build_model, load_config, resolve, set_precision  # noqa: E402
from isee.eval.movi_c import DEFINITION, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", help="default: outputs/<config name>")
    a = ap.parse_args()
    cfg = load_config(a.config)
    e = cfg["evaluation"]
    set_precision(e["tf32"])
    device = "cuda"
    root = resolve(e["data"])
    model = build_model(cfg, device)

    t0 = time.time()
    res = evaluate(model, root, cfg["model"]["input_size"], device, e["batch_clips"], e["preprocess_device"])
    print(f"\nTable 3: object discovery on MOVi-C validation, {res['clips']} clips (higher is better)")
    print(f"video FG-ARI {100 * res['fg_ari']:.1f}   video mBO {100 * res['mbo']:.1f}")

    out = Path(a.out or resolve("outputs") / Path(a.config).stem)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"config": str(a.config), "checkpoint": cfg["model"]["checkpoint"], "data": str(root),
                "batch_clips": e["batch_clips"], "preprocess_device": e["preprocess_device"], "tf32": e["tf32"],
                "gpu": torch.cuda.get_device_name(),
                "seconds": round(time.time() - t0, 1)}
    path = out / "results.json"
    path.write_text(json.dumps({"fg_ari": res["fg_ari"], "mbo": res["mbo"], "clips": res["clips"],
                                "definition": DEFINITION, "settings": settings, "per_clip": res["per_clip"]}, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
