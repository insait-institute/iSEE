"""The frozen backbone f: DINOv2 NoPE (Sec. 3.1, "Inputs").

DINOv2 ViT-S/14 with registers whose learned positional embedding has been removed and whose
features have been re-distilled from the original DINOv2 on COCO images (DINO-SAW). We use the
authors' released checkpoint unchanged; it is downloaded from their Hugging Face repository and
verified by its sha256.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import timm
import torch
import torch.nn as nn

from . import constants as C

BACKBONE_FILE = "nope_coco_dv2_vits14_reg4.pth"
BACKBONE_URL = f"https://huggingface.co/rmdocherty/dino-saw/resolve/main/{BACKBONE_FILE}"
BACKBONE_SHA256 = "9a6e3e4a2359fc7227d70c6db1660566ec0c3ec0af995bb5cf058d94fad6f623"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_backbone(directory: str | os.PathLike = "checkpoints") -> Path:
    """Return the path of the DINOv2-NoPE checkpoint, downloading it on first use."""
    path = Path(directory) / BACKBONE_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BACKBONE_URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(BACKBONE_URL, tmp)
        tmp.rename(path)
    digest = _sha256(path)
    if digest != BACKBONE_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest} does not match the released DINO-SAW checkpoint")
    return path


class NoPEBackbone(nn.Module):
    """f: image [B, 3, H, W] (ImageNet-normalised, H and W multiples of 14) -> patch features [B, N, 384]."""

    def __init__(self, checkpoint: str | os.PathLike):
        super().__init__()
        vit = timm.create_model(C.BACKBONE_ARCH, pretrained=False, dynamic_img_size=True)
        vit.pos_embed = None  # NoPE: no positional embedding at all
        vit.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.vit = vit.eval().requires_grad_(False)
        self.num_prefix_tokens = vit.num_prefix_tokens  # [CLS] + 4 registers, dropped from the output

    def train(self, mode: bool = True):
        return super().train(False)  # frozen: always in eval mode

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.vit.forward_features(images)[:, self.num_prefix_tokens:]
