#!/usr/bin/env python
"""Write MOVi-C as one file per clip, the layout `isee/data/movi_c.py` reads.

    python scripts/prepare_movi_c.py --out data/movi_c

Reads `movi_c/256x256:1.0.0` (Kubric) through TensorFlow Datasets, from the public bucket gs://kubric-public/tfds or
from a local copy of it (`--data-dir`), and writes every record of a split in the order of the release:

    <out>/train/<index>.npz          video uint8 [24, 256, 256, 3]                                      9737 clips
    <out>/validation/<index>.npz     video, and segmentations uint8 [24, 256, 256] (0 = background)       250 clips

<index> is the record's position in the split, six digits (000000, 000001, ...). Files are written under a temporary
name and renamed, and clips already written are skipped, so an interrupted run can be restarted. Needs numpy,
tensorflow and tensorflow-datasets only.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

DATASET = "movi_c/256x256:1.0.0"
PUBLIC_TFDS = "gs://kubric-public/tfds"
KEYS = {"train": ("video",), "validation": ("video", "segmentations")}


def records(split: str, data_dir: str):
    """-> (number of records, iterator over the records of `split` as dicts of numpy arrays), in release order."""
    import tensorflow_datasets as tfds
    builder = tfds.builder(DATASET, data_dir=data_dir)
    ds = builder.as_dataset(split=split, shuffle_files=False,
                            decoders=tfds.decode.PartialDecoding({k: True for k in KEYS[split]}))
    return builder.info.splits[split].num_examples, tfds.as_numpy(ds)


def write_clip(path: Path, record: dict):
    """One record -> <path>.npz: video [24, 256, 256, 3], and segmentations [24, 256, 256] when present."""
    arrays = {"video": record["video"]}
    if "segmentations" in record:
        arrays["segmentations"] = record["segmentations"][..., 0]
    for k, v in arrays.items():
        if v.dtype != np.uint8 or v.shape[:3] != (24, 256, 256):
            raise ValueError(f"{path.name}: {k} {v.dtype} {v.shape}")
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=PUBLIC_TFDS, help="the TFDS data directory holding movi_c/256x256/1.0.0")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"], choices=list(KEYS))
    a = ap.parse_args()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")                      # decoding only
    for split in a.splits:
        n, it = records(split, a.data_dir)
        out = Path(a.out) / split
        out.mkdir(parents=True, exist_ok=True)
        count = 0
        for i, record in enumerate(it):
            path = out / f"{i:06d}.npz"
            if not path.exists():
                write_clip(path, record)
            count += 1
            if count % 500 == 0:
                print(f"{split}: {count} / {n}", flush=True)
        if count != n:
            raise RuntimeError(f"{split}: read {count} records, the release has {n}")
        print(f"{split}: {count} clips -> {out}", flush=True)


if __name__ == "__main__":
    main()
