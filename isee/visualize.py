"""Two panels per frame: the frame with the reported position of every held slot, and the decoder's segmentation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
    [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40],
    [128, 0, 0], [170, 255, 195], [128, 128, 0]], dtype=np.uint8)


def to_pixels(mu: np.ndarray, side: int, height: int, width: int) -> np.ndarray:
    """mu [..., 2] in patch-centre coordinates ([-1, 1] at the centres of the border patches) -> pixels (x, y)."""
    col_row = (mu + 1.0) / 2.0 * (side - 1)
    return (col_row + 0.5) * np.array([width / side, height / side])


def _cross(img, x, y, color, r=7):
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    for d in range(-r, r + 1):
        for (px, py) in ((x + d, y + d), (x + d, y - d)):
            for ox in (-1, 0, 1):
                if 0 <= px + ox < w and 0 <= py < h:
                    img[py, px + ox] = color
    for t in range(0, 360, 10):                                     # a white ring, so the cross reads on any colour
        px, py = int(round(x + (r + 3) * np.cos(np.radians(t)))), int(round(y + (r + 3) * np.sin(np.radians(t))))
        if 0 <= px < w and 0 <= py < h:
            img[py, px] = 255


def render(frames: np.ndarray, masks: np.ndarray, held: np.ndarray, centers_px: np.ndarray, side: int) -> np.ndarray:
    """frames uint8 [T, H, W, 3], masks [T, K, N], held [T, K], centers_px [T, K, 2] -> uint8 [T, H, 2W, 3]."""
    T, H, W, _ = frames.shape
    rows = (np.arange(H) * side // H)[:, None]
    cols = (np.arange(W) * side // W)[None, :]
    out = np.empty((T, H, 2 * W, 3), dtype=np.uint8)
    for t in range(T):
        seg = PALETTE[masks[t].argmax(0).reshape(side, side)[rows, cols] % len(PALETTE)]
        left = frames[t].copy()
        for k in np.flatnonzero(held[t]):
            _cross(left, *centers_px[t, k], PALETTE[k % len(PALETTE)])
        out[t, :, :W] = left
        out[t, :, W:] = (0.45 * seg + 0.55 * frames[t]).astype(np.uint8)
    return out


def write_video(frames: np.ndarray, path: Path, fps: int = 12):
    """uint8 [T, H, W, 3] -> an .mp4 (PyAV), or a .gif if no H.264 / MPEG-4 encoder is available."""
    import av
    H, W = frames.shape[1:3]
    frames = frames[:, : H - H % 2, : W - W % 2]
    for codec in ("libx264", "mpeg4"):
        try:
            with av.open(str(path), "w") as container:
                stream = container.add_stream(codec, rate=fps)
                stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
                for f in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            return path
        except Exception:
            continue
    from PIL import Image
    gif = path.with_suffix(".gif")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    return gif
