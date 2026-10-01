#!/usr/bin/env python
"""Train the encoder of a config (Sec. 3.1; Supp. "Training").

    python scripts/train_encoder.py configs/lacater_static.yaml --out runs/lacater_static

Adam at `lr` with a linear warmup over `warmup` steps and then an exponential decay that halves the rate by the last
step; gradients clipped to global norm `grad_clip`; `precision: bf16` runs the forward pass and the losses under
bfloat16 autocast (float32 weights and optimiser state). Checkpoints `<out>/step=<n>.safetensors` every
`checkpoint_every` steps, and `<out>/latest.pt` to resume (rerunning the same command resumes).

With a `validation` block (MOVi-C), every `validation.every` steps the model is scored by video FG-ARI + video mBO of
the decoder's masks on the first `validation.clips` validation clips, and `<out>/selected.txt` names the checkpoint
used for evaluation: the last one saved at or before the best validation step.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isee.config import build_model, grid_side, load_config, resolve  # noqa: E402
from isee.data import lacater, movi_c  # noqa: E402
from isee.data.video import preprocess  # noqa: E402
from isee.eval.movi_c import clip_scores  # noqa: E402
from isee.train.encoder import learning_rate_factor, loss  # noqa: E402


def training_windows(cfg: dict):
    d, size, seed = cfg["training"]["data"], cfg["model"]["input_size"], cfg["training"]["seed"]
    if d["dataset"] == "lacater":
        return lacater.TrainingWindows(str(resolve(d["root"])), "train", d["chunk"], d["stride"],
                                       d["samples_per_clip"], size, seed)
    if d["dataset"] == "movi_c":
        return movi_c.TrainingWindows(str(resolve(d["root"])), d["chunk"], size, seed)
    raise ValueError(f"unknown dataset {d['dataset']!r}")


def features_of(model, x):
    """x [B, T, 3, S, S] -> the frozen backbone's features [B, T, N, 384], all frames of the batch in one pass."""
    with torch.no_grad():
        return model.backbone(x.flatten(0, 1)).unflatten(0, x.shape[:2])


def validation_clips(cfg: dict):
    """-> [(preprocessed frames [T, 3, S, S], segmentations [T, S, S])], the first `validation.clips` MOVi-C
    validation clips."""
    v, d, size = cfg["training"]["validation"], cfg["training"]["data"], cfg["model"]["input_size"]
    out = []
    for p in movi_c.clip_paths(resolve(d["root"]), "validation")[:v["clips"]]:
        video, seg = movi_c.read_clip(p)
        out.append((preprocess(video, size), movi_c.resize_segmentations(seg, size)))
    return out


@torch.no_grad()
def validate(model, cfg, clips, device, autocast) -> float:
    v = cfg["training"]["validation"]
    model.eval()
    scores = []
    for i in range(0, len(clips), v["batch_size"]):
        x = torch.stack([c[0] for c in clips[i:i + v["batch_size"]]]).to(device)
        with autocast():
            out = model.rollout(features_of(model, x))
        for b, (_, seg) in enumerate(clips[i:i + v["batch_size"]]):
            s = clip_scores(out["masks"][b].float().cpu().numpy(), seg, grid_side(cfg))
            scores.append(s["fg_ari"] + s["mbo"])
    model.train()
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    t = cfg["training"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    model = build_model(cfg, device, checkpoint="")
    model.train()
    params = model.trainable_parameters()
    opt = torch.optim.Adam(params, lr=t["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: learning_rate_factor(s, t["warmup"], t["steps"]))
    bf16 = t["precision"] == "bf16"

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)

    step, history = 0, []
    latest = out / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, history = int(ck["step"]), ck["validation"]
        torch.set_rng_state(ck["rng"])
        print(f"resumed from {latest} at step {step}", flush=True)

    data = training_windows(cfg)
    loader = torch.utils.data.DataLoader(data, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                                         num_workers=t["workers"], pin_memory=True, persistent_workers=True,
                                         prefetch_factor=6)
    val_clips = validation_clips(cfg) if "validation" in t else None
    print(f"{len(data)} training windows per epoch, batch {t['batch_size']}; "
          f"{sum(p.numel() for p in params):,} trainable parameters", flush=True)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    while step < t["steps"]:
        for x in loader:
            x = x.to(device, non_blocking=True)
            with autocast():
                f = features_of(model, x)
                losses = loss(model.rollout(f), f)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if step % t["log_every"] == 0:
                rec = {"step": step, **{k: float(v.detach()) for k, v in losses.items()},
                       "lr": sched.get_last_lr()[0], "hours": (time.time() - t0) / 3600}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if val_clips is not None and step % t["validation"]["every"] == 0:
                history.append((step, validate(model, cfg, val_clips, device, autocast)))
                print(json.dumps({"step": step, "validation": history[-1][1]}), flush=True)
                log.write(json.dumps({"step": step, "validation": history[-1][1]}) + "\n")
                log.flush()
            if step % t["checkpoint_every"] == 0 or step == t["steps"]:
                save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                          str(out / f"step={step}.safetensors"))
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(), "step": step, "validation": history,
                            "rng": torch.get_rng_state()}, latest)
            if step >= t["steps"]:
                break
    if val_clips is not None:
        best_step = max(history, key=lambda r: r[1])[0]
        chosen = (best_step // t["checkpoint_every"]) * t["checkpoint_every"]
        (out / "selected.txt").write_text(f"step={chosen}.safetensors\n")
        print(f"best validation at step {best_step}; selected step={chosen}.safetensors", flush=True)


if __name__ == "__main__":
    main()
