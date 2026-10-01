"""LA-CATER evaluation: permanence of the snitch while it is hidden (Tables 1 and 2) and object discovery (Table 3).

Ground truth, per test clip:
    <data>/test_data/labels/<clip>_bb.json                     the snitch's amodal box on every frame
    <data>/test_data/containment_and_occlusions/*.txt          per-frame visibility and containment of the snitch
    <masks>/<clip>.npz                                          rendered masks (see `read_masks`)
The occlusions scored are listed in eval_sets/lacater_<camera>.json.

A model run provides, per clip and frame, every slot's committed position p_t, TEN's held flags u_t, the walked
position and the grouping masks (the corrector's attention on its last iteration, a softmax over slots on the patch
grid).

Table 1, own-box mAP while hidden (`permanence_table`)
    follow    The followed slot on frame t is the argmax over slots of the share of each slot's attention (its
              grouping mask normalised to sum 1 over the patches) that lies on the snitch's modal mask; only frames
              on which the snitch covers at least 20 pixels have one.
    anchor    t_a is the last frame before the occlusion on which at least half of the snitch is visible (modal
              pixels / pixels of the snitch rendered alone), k the slot followed at t_a.
    read-out  A slot's 4-d position is mapped to image pixels by ridge regression onto the snitch's amodal box
              centre (`PositionReadout`), fit on the even frames of the evaluated clips on which a slot follows the
              snitch; each clip is read by a fold that did not see it.
    position  On a hidden frame: the walked position of k if TEN holds k, otherwise k's committed position.
    box       The region of k in the grouping masks on t_a (bilinear to 320 x 240, argmax over slots), or on the
              latest of the 5 frames before t_a where it is not empty; the box keeps its size and is centred on the
              read-out position of every hidden frame. A slot without a region scores IoU 0.
    score     IoU with the amodal box. Per occlusion and label (occluded / contained / carried, from the dataset's
              flags), the mean over the thresholds 0.10, 0.15, ..., 0.30 of the share of frames with IoU above the
              threshold; a cell is the mean over the occlusions that have frames of that label, in percent.
    frozen    The same box and slot, at the read-out position of k on t_a, for the whole occlusion.

Table 2, SAME (`same_table`)
    For every run of frames on which the snitch has no rendered pixel: t_p is the last frame before it and t_q the
    first frame from its end on which at least half of the snitch is visible; the owner on a frame is the slot with
    the largest attention mass on the snitch (the grouping mask summed over the snitch's share of each patch). SAME
    is 1 if the owner at t_q is the owner at t_p. Each occlusion of the eval set takes the run that overlaps it most;
    cells group the occlusions by length (frames) and give the percentage with SAME = 1.

Table 3, object discovery (`discovery_scores`): video FG-ARI and video mBO (isee/eval/segmentation.py) of the decoder's
masks against the rendered modal masks, on the first 150 test clips (sorted by name), averaged over clips.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..data.lacater import FRAME_H, FRAME_W, N_FRAMES, SNITCH, snitch_boxes
from .segmentation import upsample_argmax, video_scores

MIN_FOLLOW_PX = 20
HALF_VISIBLE = 0.5
BOX_LOOKBACK = 5
IOU_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)
LABELS = {"occluded": 0, "contained": 1, "carried": 2}
LENGTH_BINS = (("5-25", 5, 25), ("25-50", 25, 50), ("50-100", 50, 100), (">=100", 100, None))
RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
RIDGE_FOLDS = 5


# ---------------------------------------------------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------------------------------------------------

def read_masks(path: str | Path):
    """-> modal uint8 [300, 240, 320] (0 = background, i = object i - 1 of `classes`), classes [n],
    full_px [300, n] (the pixels of each object rendered alone, occluders removed)."""
    with np.load(path) as z:
        return z["modal"], [str(c) for c in z["classes"]], z["full_px"].astype(np.int64)


def snitch_visibility(modal: np.ndarray, classes: Sequence[str], full_px: np.ndarray):
    """-> (the snitch's modal mask [T, H, W] bool, its visible pixels [T], its visible fraction [T])."""
    o = list(classes).index(SNITCH)
    mask = modal == o + 1
    px = mask.reshape(len(mask), -1).sum(1)
    full = full_px[:, o]
    frac = np.where(full > 0, px / np.maximum(full, 1), 0.0)
    return mask, px, frac


def _frame_list(path: Path) -> Dict[str, np.ndarray]:
    """`<clip>\\t<comma-separated frames>` per line -> {clip: bool [300]}."""
    out = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        clip, frames = line.split("\t")
        m = np.zeros(N_FRAMES, bool)
        if frames:
            m[np.array([int(f) for f in frames.split(",")])] = True
        out[clip] = m
    return out


def read_flags(root: str | Path, split: str = "test") -> Dict[str, Dict[str, np.ndarray]]:
    """The dataset's per-frame flags of the snitch, {clip: {visible, occluded, contained, carried}}:
    visible = visibility_rate_gt_0, occluded = not visible and not in any container, contained = in a container that
    does not move (containment_only_static), carried = in a container that moves (containment_with_move)."""
    d = Path(root) / f"{split}_data" / "containment_and_occlusions"
    visible = _frame_list(d / "visibility_rate_gt_0.txt")
    contained_any = _frame_list(d / "containment_annotations.txt")
    contained = _frame_list(d / "containment_only_static_annotations.txt")
    carried = _frame_list(d / "containment_with_move_annotations.txt")
    return {c: {"visible": visible[c], "occluded": ~visible[c] & ~contained_any[c], "contained": contained[c],
                "carried": carried[c]} for c in visible}


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """The maximal runs [start, end) of True."""
    out, s = [], None
    for t, m in enumerate(mask):
        if m and s is None:
            s = t
        elif not m and s is not None:
            out.append((s, t))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def half_visible_before(frac: np.ndarray, t: int) -> Optional[int]:
    """The last frame u < t with frac[u] >= 1/2."""
    idx = np.flatnonzero(frac[:t] >= HALF_VISIBLE)
    return int(idx[-1]) if len(idx) else None


def half_visible_after(frac: np.ndarray, t: int) -> Optional[int]:
    """The first frame u >= t with frac[u] >= 1/2."""
    idx = np.flatnonzero(frac[t:] >= HALF_VISIBLE)
    return int(t + idx[0]) if len(idx) else None


# ---------------------------------------------------------------------------------------------------------------------
# Which slot holds the snitch
# ---------------------------------------------------------------------------------------------------------------------

def _patch_index(H: int, W: int, g: int, rounded: bool) -> Tuple[np.ndarray, np.ndarray]:
    """(pixel -> patch index [H * W], pixels per patch [g * g]). Patch boundaries at floor(i * H / g) (`rounded`
    False: pixel y lies in patch y * g // H) or at round(i * H / g)."""
    if rounded:
        ys, xs = np.round(np.arange(g + 1) * H / g).astype(int), np.round(np.arange(g + 1) * W / g).astype(int)
        row, col = np.zeros(H, int), np.zeros(W, int)
        for k in range(g):
            row[ys[k]:ys[k + 1]], col[xs[k]:xs[k + 1]] = k, k
    else:
        row, col = np.arange(H) * g // H, np.arange(W) * g // W
    idx = (row[:, None] * g + col[None, :]).reshape(-1)
    return idx, np.bincount(idx, minlength=g * g).astype(np.float64)


def patch_coverage(mask: np.ndarray, g: int, rounded: bool) -> np.ndarray:
    """[H, W] bool -> [g * g] the share of each patch the mask covers."""
    idx, area = _patch_index(*mask.shape, g, rounded)
    return np.bincount(idx, weights=mask.reshape(-1).astype(np.float64), minlength=g * g) / np.maximum(area, 1.0)


def followed_slot(grouping: np.ndarray, snitch: np.ndarray, px: np.ndarray) -> np.ndarray:
    """grouping [T, K, g, g], snitch mask [T, H, W], its pixels [T] -> the followed slot [T] (-1: fewer than 20 px).
    Each slot's attention is normalised over the patches before it is weighed by the snitch's coverage."""
    T, K, g = grouping.shape[0], grouping.shape[1], grouping.shape[-1]
    A = grouping.reshape(T, K, -1).astype(np.float64)
    A = A / np.maximum(A.sum(-1, keepdims=True), 1e-12)
    out = np.full(T, -1)
    for t in np.flatnonzero(px >= MIN_FOLLOW_PX):
        out[t] = int(np.argmax(A[t] @ patch_coverage(snitch[t], g, rounded=False)))
    return out


def owning_slot(grouping: np.ndarray, snitch: np.ndarray) -> np.ndarray:
    """grouping [T, K, g, g], snitch mask [T, H, W] -> the slot with the largest attention mass on the snitch [T]
    (-1 where the snitch has no pixel)."""
    T, K, g = grouping.shape[0], grouping.shape[1], grouping.shape[-1]
    out = np.full(T, -1)
    for t in range(T):
        cover = patch_coverage(snitch[t], g, rounded=True)
        mass = grouping[t].reshape(K, -1).astype(np.float64) @ cover
        if mass.max() > 0:
            out[t] = int(mass.argmax())
    return out


# ---------------------------------------------------------------------------------------------------------------------
# From a slot's position to pixels
# ---------------------------------------------------------------------------------------------------------------------

def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression with an unpenalised intercept, [D + 1, 2]."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    reg = alpha * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)


def _apply(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


class PositionReadout:
    """Maps a slot's position [4] to the image [2] (x, y px). Positions are standardised over all rows; the clips are
    shuffled (RandomState(0)) into 5 folds; the penalty is the one with the lowest mean pixel error over the folds, each
    fold predicted by the model fit on the other four; clip c is then read by the model of the folds without c."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, clip: np.ndarray):
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        clips = np.unique(clip)
        np.random.RandomState(0).shuffle(clips)
        folds = np.array_split(clips, RIDGE_FOLDS)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self.cv_px = {}
        for a in RIDGE_ALPHAS:
            err = []
            for f in folds:
                test = np.isin(clip, f)
                err.append(np.linalg.norm(_apply(Xs[test], _ridge(Xs[~test], Y[~test], a)) - Y[test], axis=1).mean())
            self.cv_px[a] = float(np.mean(err))
        self.alpha = min(RIDGE_ALPHAS, key=lambda a: self.cv_px[a])
        self.models = {}
        for f in folds:
            W = _ridge(Xs[~np.isin(clip, f)], Y[~np.isin(clip, f)], self.alpha)
            for c in f:
                self.models[int(c)] = W
        self.n_rows = len(X)

    def __call__(self, position: np.ndarray, clip: int) -> np.ndarray:
        """position [..., 4] -> [..., 2] pixels."""
        p = np.asarray(position)
        return _apply(((p.reshape(-1, p.shape[-1]) - self.mu) / self.sd), self.models[int(clip)]).reshape(
            *p.shape[:-1], 2)


# ---------------------------------------------------------------------------------------------------------------------
# Boxes and scores
# ---------------------------------------------------------------------------------------------------------------------

def own_box(grouping: np.ndarray, anchor: int, k: int) -> Optional[np.ndarray]:
    """Slot k's region on `anchor` (or the latest of the 5 frames before it where it is not empty) -> x1, y1, x2, y2."""
    for t in range(anchor, max(anchor - BOX_LOOKBACK - 1, -1), -1):
        up = F.interpolate(torch.from_numpy(grouping[t].astype(np.float32))[None], size=(FRAME_H, FRAME_W),
                           mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        ys, xs = np.nonzero(up == k)
        if len(xs):
            return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)
    return None


def box_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """[T, 4] x1, y1, x2, y2 (inclusive pixel coordinates, so a box is x2 - x1 + 1 wide) -> IoU [T]."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    iw = np.maximum(np.minimum(pred[:, 2], gt[:, 2]) - np.maximum(pred[:, 0], gt[:, 0]) + 1, 0)
    ih = np.maximum(np.minimum(pred[:, 3], gt[:, 3]) - np.maximum(pred[:, 1], gt[:, 1]) + 1, 0)
    inter = iw * ih
    area = lambda b: (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    return inter / (area(pred) + area(gt) - inter)


def _map(iou: np.ndarray) -> float:
    return 100.0 * float(np.mean([np.mean(iou > thr) for thr in IOU_THRESHOLDS]))


# ---------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------

def clip_record(run: Dict[str, np.ndarray], masks_path: str | Path) -> dict:
    """One clip's model run (position, held, walked, grouping) and ground truth -> what the tables read."""
    modal, classes, full_px = read_masks(masks_path)
    snitch, px, frac = snitch_visibility(modal, classes, full_px)
    grouping = run["grouping"]
    return dict(position=run["position"].astype(np.float32), held=run["held"].astype(bool),
                walked=run["walked"].astype(np.float32), grouping=grouping, px=px, frac=frac,
                followed=followed_slot(grouping, snitch, px), owner=owning_slot(grouping, snitch))


def permanence_table(records: Dict[str, dict], cases: List[dict], root: str | Path) -> dict:
    """Table 1. records: {clip: clip_record(...)} for every clip of the eval set, cases: the eval set's occlusions."""
    flags = read_flags(root)
    names = sorted(records)
    boxes = {c: snitch_boxes(root, "test", c) for c in names}
    X, Y, C = [], [], []
    for i, c in enumerate(names):
        r = records[c]
        centre = np.stack([(boxes[c][:, 0] + boxes[c][:, 2]) / 2, (boxes[c][:, 1] + boxes[c][:, 3]) / 2], 1)
        for t in range(0, N_FRAMES, 2):
            if r["followed"][t] >= 0:
                X.append(r["position"][t, r["followed"][t]])
                Y.append(centre[t])
                C.append(i)
    readout = PositionReadout(np.array(X, np.float64), np.array(Y, np.float64), np.array(C, np.float64))
    clip_id = {c: i for i, c in enumerate(names)}

    rows = {"iSEE": [], "frozen": []}
    no_box = 0
    for case in cases:
        c, s, e = case["clip"], case["start"], case["end"]
        r, f = records[c], flags[c]
        anchor = half_visible_before(r["frac"], s)
        k = int(r["followed"][anchor])
        label = np.full(e - s, -1)
        label[f["occluded"][s:e]] = LABELS["occluded"]
        label[f["contained"][s:e] & ~f["carried"][s:e]] = LABELS["contained"]
        label[f["carried"][s:e]] = LABELS["carried"]
        t = np.arange(s, e)
        position = np.where(r["held"][t, k, None], r["walked"][t, k], r["position"][t, k])
        centres = {"iSEE": readout(position, clip_id[c]),
                   "frozen": np.repeat(readout(r["position"][anchor, k], clip_id[c])[None], e - s, 0)}
        box = own_box(r["grouping"], anchor, k)
        no_box += box is None
        gt = boxes[c][s:e].astype(np.float64)
        for row, ctr in centres.items():
            if box is None:
                iou = np.zeros(e - s)
            else:
                half = (box[2:] - box[:2]) / 2
                iou = box_iou(np.concatenate([ctr - half, ctr + half], 1), gt)
            cell = {lab: _map(iou[label == code]) for lab, code in LABELS.items() if (label == code).any()}
            cell["all hidden"] = _map(iou)
            rows[row].append(cell)
    table = {row: {lab: {"mAP": float(np.mean([x[lab] for x in per if lab in x])),
                         "n": sum(lab in x for x in per)} for lab in list(LABELS) + ["all hidden"]}
             for row, per in rows.items()}
    return {"table": table, "n_occlusions": len(cases), "n_without_box": int(no_box),
            "readout": {"alpha": readout.alpha, "cv_px": readout.cv_px[readout.alpha], "n_rows": readout.n_rows}}


def same_table(records: Dict[str, dict], cases: List[dict]) -> dict:
    """Table 2. records: {clip: clip_record(...)}, cases: the eval set's occlusions."""
    episodes = {}
    for c, r in records.items():
        out = []
        for gs, ge in runs(r["px"] == 0):
            if gs == 0 or ge >= N_FRAMES:
                continue
            tp, tq = half_visible_before(r["frac"], gs), half_visible_after(r["frac"], ge)
            if tp is None or tq is None or r["owner"][tp] < 0:
                continue
            out.append((gs, ge, bool(r["owner"][tq] == r["owner"][tp])))
        episodes[c] = out
    scored = []
    for case in cases:
        best, overlap = None, 0
        for gs, ge, same in episodes[case["clip"]]:
            o = min(ge, case["end"]) - max(gs, case["start"])
            if o > overlap:
                best, overlap = same, o
        if best is not None:
            scored.append((case["end"] - case["start"], best))
    cells = {}
    for name, lo, hi in LENGTH_BINS + (("all", 0, None),):
        v = [same for n, same in scored if n >= lo and (hi is None or n < hi)]
        cells[name] = {"SAME": round(100.0 * float(np.mean(v)), 1) if v else None, "n": len(v)}
    return {"table": cells, "n_occlusions": len(cases), "n_scored": len(scored)}


def discovery_scores(decoder_masks: np.ndarray, modal: np.ndarray) -> dict:
    """decoder masks [T, K, g, g], modal [T, 240, 320] -> video FG-ARI and video mBO of one clip."""
    pred = upsample_argmax(decoder_masks, modal.shape[1:])
    return video_scores(modal.astype(np.int16), pred, decoder_masks.shape[1])


def load_eval_set(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
