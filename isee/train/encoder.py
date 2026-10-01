"""Training the encoder (Sec. 3.1, "Objective"; Supp. "Training").

The encoder is trained on short windows of consecutive frames, with TEN and the walker switched off. Two losses:

  reconstruction  the mean squared error between the decoder's reconstruction and the frozen backbone's features
                  f(x_t), over all frames, patches and the 384 channels (weight 1)
  contrast        SlotContrast's slot-slot contrastive loss on the appearance stream (weight 0.5): the appearance
                  streams a^k_t are l2-normalised; for every slot j of every clip at frame t+1, a softmax at
                  temperature 0.1 over the similarities to all slots of all clips of the batch at frame t must pick
                  the same slot of the same clip; cross-entropy averaged over frames and slots
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model import constants as C

RECONSTRUCTION_WEIGHT = 1.0
CONTRAST_WEIGHT = 0.5
TEMPERATURE = 0.1


def reconstruction_loss(reconstruction: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """reconstruction, features [B, T, N, 384] -> the mean squared error."""
    return F.mse_loss(reconstruction, features.detach())


def contrastive_loss(slots: torch.Tensor) -> torch.Tensor:
    """slots [B, T, K, 64] -> the slot-slot contrastive loss on a^k = slots[..., :60]."""
    B, T, K, _ = slots.shape
    z = F.normalize(slots[..., :C.APPEARANCE_DIM], p=2, dim=-1)
    z = z.permute(1, 0, 2, 3).reshape(T, B * K, -1)                     # [T, B*K, 60], index b*K + k
    logits = (z[:-1] @ z[1:].transpose(-2, -1)) / TEMPERATURE            # [T-1, i (frame t), j (frame t+1)]
    target = torch.eye(B * K, device=slots.device).expand(T - 1, -1, -1)
    return F.cross_entropy(logits, target)


def loss(out: dict, features: torch.Tensor) -> dict:
    rec = reconstruction_loss(out["reconstruction"], features)
    con = contrastive_loss(out["slots"])
    return {"loss": RECONSTRUCTION_WEIGHT * rec + CONTRAST_WEIGHT * con, "reconstruction": rec, "contrast": con}


def learning_rate_factor(step: int, warmup: int, decay_steps: int, decay_rate: float = 0.5) -> float:
    """Linear warmup from 0 over `warmup` steps, then an exponential decay that halves over `decay_steps - warmup`."""
    if step < warmup:
        return step / warmup
    return decay_rate ** ((step - warmup) / (decay_steps - warmup))
