#!/usr/bin/env python
"""Train one walker on the data written by `build_walker_data.py` (Sec. 3.3; Supp. "Walker").

    python scripts/train_walker.py configs/lacater_static.yaml --seed 0 --out runs/walker_lacater_static_seed0

Adam at 1e-3, 100 warmup steps, then a cosine decay to 5 % of the rate over `schedule_steps`; training stops at
`steps`. Gradients are clipped to norm 1. The weights kept are an exponential moving average (0.998) of the trained
ones: `<out>/walker.safetensors` at the end, `<out>/latest.pt` (to resume) every `save_every` steps. The last
`val_clips` clips are held out: every `val_every` steps the averaged weights' return NLL on them is logged.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isee.config import load_config, resolve  # noqa: E402
from isee.model.walker import Walker  # noqa: E402
from isee.train.walker import return_loss  # noqa: E402


class WalkerData:
    """The walker's training clips: h [n, T, N, 64] (float16), position [n, T, K, 4] (float16), held [n, T, K]."""

    def __init__(self, directory: Path):
        self.meta = json.load(open(directory / "meta.json"))
        self.h = np.load(directory / "h.npy", mmap_mode="r")
        self.position = np.load(directory / "position.npy", mmap_mode="r")
        self.held = np.load(directory / "held.npy", mmap_mode="r")
        self.pstar = torch.from_numpy(np.load(directory / "pstar.npy"))

    def __len__(self):
        return self.h.shape[0]

    def get(self, ids):
        h = torch.from_numpy(np.stack([np.asarray(self.h[i]) for i in ids]))
        p = torch.from_numpy(np.stack([np.asarray(self.position[i], dtype=np.float32) for i in ids]))
        u = torch.from_numpy(np.stack([1.0 - np.asarray(self.held[i], dtype=np.float32) for i in ids]))
        return h, p, u


class Prefetch:
    """A background thread that keeps a few batches ready."""

    def __init__(self, fn, depth: int = 3):
        self.q = queue.Queue(maxsize=depth)
        self.fn = fn
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        k = 0
        while True:
            self.q.put(self.fn(k))
            k += 1

    def next(self):
        return self.q.get()


def learning_rate(t: dict, step: int) -> float:
    if step < t["warmup"]:
        return t["lr"] * (step + 1) / t["warmup"]
    p = min(1.0, (step - t["warmup"]) / max(1, t["schedule_steps"] - t["warmup"]))
    return t["lr"] * (t["lr_floor"] + (1 - t["lr_floor"]) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def validate(walker, data, ids, t, device):
    walker.eval()
    total, weight = 0.0, 0.0
    for i in range(0, len(ids), t["val_batch"]):
        h, p, u = data.get(ids[i:i + t["val_batch"]])
        out = return_loss(walker, data.pstar.to(device), h.to(device), p.to(device), u.to(device), dmax=t["dmax"])
        total += float(out["nll"]) * out["n_returns"]
        weight += out["n_returns"]
    walker.train()
    return total / max(weight, 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="default: the config's walker_training.data")
    a = ap.parse_args()
    cfg = load_config(a.config)
    t = cfg["walker_training"]
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = WalkerData(resolve(a.data or t["data"]))
    side = int(round(math.sqrt(data.h.shape[2])))

    torch.manual_seed(a.seed)
    walker = Walker(side).to(device)
    ema = copy.deepcopy(walker).requires_grad_(False)
    opt = torch.optim.Adam(walker.parameters(), lr=t["lr"])
    step = 0
    latest = out / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu")
        walker.load_state_dict(ck["walker"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])
        print(f"resumed from {latest} at step {step}", flush=True)
    pstar = data.pstar.to(device)
    train_ids = np.arange(0, len(data) - t["val_clips"])
    val_ids = np.arange(len(data) - t["val_clips"], len(data))
    print(f"{len(train_ids)} training clips, {len(val_ids)} held out; {sum(p.numel() for p in walker.parameters())} "
          f"parameters", flush=True)

    def batch(k):
        rng = np.random.default_rng([a.seed, k])
        return data.get(rng.choice(train_ids, t["batch_size"], replace=False))

    start = step
    prefetch = Prefetch(lambda k: batch(start + k))
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    walker.train()
    while step < t["steps"]:
        h, p, u = prefetch.next()
        gen = torch.Generator().manual_seed(a.seed * 1_000_003 + step)
        for g in opt.param_groups:
            g["lr"] = learning_rate(t, step)
        o = return_loss(walker, pstar, h.to(device, non_blocking=True), p.to(device), u.to(device), dmax=t["dmax"],
                        pseudo=t["pseudo_per_clip"], pseudo_len=tuple(t["pseudo_len"]), moved_boost=t["moved_boost"],
                        gen=gen)
        loss = o["nll"] + o["nll_pseudo"]
        if o["n_returns"] + o["n_pseudo"] > 0:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(walker.parameters(), t["grad_clip"]))
            if math.isfinite(norm):
                opt.step()
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), walker.parameters()):
                        pe.lerp_(pm, 1 - t["ema"])
        step += 1
        if step % t["log_every"] == 0:
            rec = {"step": step, "nll": float(o["nll"].detach()), "nll_pseudo": float(o["nll_pseudo"].detach()),
                   "returns": o["n_returns"], "pseudo": o["n_pseudo"], "sigma": float(walker.sigma().detach()),
                   "lr": learning_rate(t, step - 1), "minutes": (time.time() - t0) / 60}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % t["save_every"] == 0 or step == t["steps"]:
            torch.save({"walker": walker.state_dict(), "ema": ema.state_dict(), "optimizer": opt.state_dict(),
                        "step": step}, latest)
        if step % t["val_every"] == 0 or step == t["steps"]:
            rec = {"step": step, "val_nll": validate(ema, data, val_ids, t, device)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    save_file({k: v.float().contiguous() for k, v in ema.state_dict().items()}, str(out / "walker.safetensors"),
              metadata={"seed": str(a.seed), "steps": str(step), "weights": "EMA"})
    print(f"wrote {out / 'walker.safetensors'}", flush=True)


if __name__ == "__main__":
    main()
