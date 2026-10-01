"""Configs (`configs/*.yaml`) and building the models they describe."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import yaml
from safetensors.torch import load_file

from .model.backbone import fetch_backbone
from .model.isee import ISEE
from .model.ten import HoldRule
from .model.walker import Walker

REPO = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """A config file as a dict; relative paths inside it are relative to the repository root."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def grid_side(cfg: dict) -> int:
    return cfg["model"]["input_size"] // 14


def build_model(cfg: dict, device: str = "cpu", checkpoint: str | Path | None = None) -> ISEE:
    """The encoder with TEN, loaded from `checkpoint` (default: the config's `model.checkpoint`; pass "" for a freshly
    initialised model)."""
    m = cfg["model"]
    model = ISEE(m["num_slots"], grid_side(cfg), str(fetch_backbone(REPO / "checkpoints")), HoldRule(**cfg["ten"]))
    checkpoint = m["checkpoint"] if checkpoint is None else checkpoint
    if checkpoint:
        model.load_state_dict(load_file(str(resolve(checkpoint))))
    return model.to(device)


def build_walkers(cfg: dict, device: str = "cpu") -> Tuple[List[Walker], float]:
    """The walker ensemble of a config and its `dmax` ([] and 0 when the config has no walker)."""
    if "walker" not in cfg:
        return [], 0.0
    walkers = []
    for path in cfg["walker"]["checkpoints"]:
        w = Walker(grid_side(cfg))
        w.load_state_dict(load_file(str(resolve(path))))
        walkers.append(w.eval().requires_grad_(False).to(device))
    return walkers, float(cfg["walker"]["dmax"])


def set_precision(tf32: bool):
    """TF32 matrix products on GPU (the paper's evaluation ran with TF32 on)."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
