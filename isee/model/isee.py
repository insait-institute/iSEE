"""iSEE (Supp. Algorithm 1).

Training (`ISEE.rollout`): for every frame t of a clip,
    S^p_t         = P_omega(S~_{t-1})                             (S^p_1 = the learned initial slots)
    S~_t, A_t     = C_theta(h_t, c, S^p_t)                        two-stream slot attention
    h_hat_t       = Dec(S~_t)                                     reconstruction of the backbone features

Inference (`ISEE.forward`): the same recurrence, with TEN and the walker between the corrector and the predictor,
    S~_t, A_t     = C_theta(h_t, c, S^p_t)
    u_t, S^c_t    = TEN(S~_t, A_t, ...)                           held slots keep their appearance stream
    p_hat_t       = walk(h, S^c, u)                               the walked position of every slot
    S^p_{t+1}     = P_omega(S^c_t with p <- p_hat on held slots)
    masks_t       = Dec(S~_t)                                     the decoder reads the corrector's output
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from . import constants as C
from .backbone import NoPEBackbone
from .decoder import ObjectFrameDecoder
from .features import FeatureMLP, position_code
from .predictor import TwoStreamPredictor
from .slot_attention import TwoStreamSlotAttention
from .ten import HoldRule, TemporalEvidenceNormalisation
from .walker import OnlineWalk, Walker, position_table


class ISEE(nn.Module):
    def __init__(self, num_slots: int, grid_side: int, backbone_checkpoint: str, hold_rule: HoldRule = HoldRule()):
        super().__init__()
        self.num_slots, self.grid_side = num_slots, grid_side
        self.backbone = NoPEBackbone(backbone_checkpoint)
        self.feature_mlp = FeatureMLP()                                                  # g_psi
        self.initial_slots = nn.Parameter(torch.randn(1, num_slots, C.SLOT_DIM) * C.SLOT_DIM ** -0.5)  # S^p_1
        self.corrector = TwoStreamSlotAttention()                                        # C_theta
        self.predictor = TwoStreamPredictor()                                            # P_omega
        self.decoder = ObjectFrameDecoder(grid_side)
        self.ten = TemporalEvidenceNormalisation(grid_side, hold_rule)                   # no parameters

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("backbone.")}           # the backbone is not ours

    def load_state_dict(self, state_dict, strict: bool = True):
        missing, unexpected = super().load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith("backbone.")]
        if strict and (missing or unexpected):
            raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
        return missing, unexpected

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    # -- the two input grids ------------------------------------------------------------------------------------
    @torch.no_grad()
    def backbone_features(self, images: torch.Tensor, chunk: int = 32) -> torch.Tensor:
        """images [B, T, 3, H, W] (preprocessed) -> f(x) [B, T, N, 384]. The backbone is stateless, so the frames of a
        clip are batched, `chunk` at a time."""
        B, T = images.shape[:2]
        out = [torch.cat([self.backbone(images[b, i:i + chunk]) for i in range(0, T, chunk)]) for b in range(B)]
        return torch.stack(out, 0)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, chunk: int = 32) -> torch.Tensor:
        """images [B, T, 3, H, W] (preprocessed) -> the appearance grid h [B, T, N, 64]."""
        return self.feature_mlp(self.backbone_features(images, chunk))

    def position_table(self) -> torch.Tensor:
        """P*, from this encoder's GRU_p and v_p (Sec. 3.3)."""
        return position_table(self.corrector.gru_p, self.corrector.v_p.weight, self.grid_side)

    # -- training -----------------------------------------------------------------------------------------------
    def rollout(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """features [B, T, N, 384] (the backbone's) -> per frame, with gradients:

        slots          [B, T, K, 64]  S~_t, the corrector's output
        reconstruction [B, T, N, 384] the decoder's reconstruction of the backbone features
        masks          [B, T, K, N]   the decoder's masks"""
        B, T = features.shape[:2]
        h = self.feature_mlp(features)
        c = position_code(self.grid_side, h.dtype, h.device).expand(B, -1, -1)          # c, in h's precision
        slots = self.initial_slots.expand(B, -1, -1)
        states = []
        for t in range(T):
            slots, _, _ = self.corrector(h[:, t], c, slots, C.N_ITERS_FIRST if t == 0 else C.N_ITERS)
            states.append(slots)
            slots = self.predictor(slots)
        states = torch.stack(states, 1)
        dec = self.decoder(states.flatten(0, 1))
        return {"slots": states, "reconstruction": dec["features"].unflatten(0, (B, T)),
                "masks": dec["masks"].unflatten(0, (B, T))}

    # -- inference ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, h: torch.Tensor, walkers: Optional[Sequence[Walker]] = None, walk_dmax: float = 0.0,
                decode: bool = True) -> Dict[str, torch.Tensor]:
        """h [B, T, N, 64] -> per-frame outputs. With `walkers`, the walked position of held slots is written into
        the predictor's input (a slot whose anchor lies farther than `walk_dmax` from P* is not written).

        state            [B, T, K, 64]  S^c_t, the committed state (held slots keep a from the previous frame)
        state_updated    [B, T, K, 64]  S~_t, the corrector's output
        held             [B, T, K]      u_t
        evidence_local   [B, T, K]      e_loc_t, the onset evidence
        evidence_release [B, T, K]      e_rel_t, the release evidence
        grouping         [B, T, K, N]   the corrector's attention on its last iteration (softmax over slots)
        position_walk    [B, T, K, 4]   (with walkers) every slot's walked position; frame t reports step t // 2
        written          [B, T, K]      (with walkers) the slots whose position was written before the predictor
        masks            [B, T, K, N]   (with decode) the decoder's masks, softmax_k(alpha)
        mu, sigma        [B, T, K, 2]   (with decode) the decoder's position and scale of each slot"""
        B, T, N, _ = h.shape
        c = position_code(self.grid_side, h.dtype, h.device).expand(B, -1, -1)
        slots = self.initial_slots.expand(B, -1, -1)
        walk = OnlineWalk(walkers, self.position_table().to(h.device), h, self.num_slots, walk_dmax) \
            if walkers else None
        ten_state = None
        keys = ("state", "state_updated", "held", "evidence_local", "evidence_release", "grouping")
        out = {k: [] for k in keys + (("position_walk", "written") if walk else ())}
        for t in range(T):
            updated, attn_first, attn_last = self.corrector(h[:, t], c, slots, C.N_ITERS_FIRST if t == 0 else C.N_ITERS)
            committed, held, e_loc, e_rel, ten_state = self.ten(updated, attn_first, attn_last, ten_state)
            for k, v in zip(keys, (committed, updated, held, e_loc, e_rel, attn_last)):
                out[k].append(v)
            if walk:
                steered, walked, written = walk(t, committed, held)
                out["position_walk"].append(walked)
                out["written"].append(written)
                slots = self.predictor(steered)
            else:
                slots = self.predictor(committed)
        out = {k: torch.stack(v, 1) for k, v in out.items()}
        if decode:
            dec = [self.decoder(out["state_updated"][:, t]) for t in range(T)]
            for k in ("masks", "mu", "sigma"):
                out[k] = torch.stack([d[k] for d in dec], 1)
        return out
