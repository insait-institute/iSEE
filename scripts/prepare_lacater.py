#!/usr/bin/env python
"""Arrange LA-CATER for this repository (layout in isee/data/lacater.py).

Static camera, from the zips of the LA-CATER release (test_data.zip, train_data_part_1.zip, train_data_part_2.zip):

    python scripts/prepare_lacater.py static --downloads <dir with the zips> --out data/la_cater

extracts test_data/{videos, labels, containment_and_occlusions} and every training video into train_data/videos.

Moving camera, from the LA-CATER-Moving release (la_cater_moving.tar.gz), read as a stream:

    python scripts/prepare_lacater.py moving --tarball la_cater_moving.tar.gz --out data/la_cater_moving
    pigz -dc la_cater_moving.tar.gz | python scripts/prepare_lacater.py moving --tarball - --out data/la_cater_moving

packs the 300 JPEG frames of every clip, byte for byte, into one MJPEG .avi (24 fps; frame k is annotation k) and
extracts labels and containment_and_occlusions of every split. A clip already packed is skipped, so an interrupted
run can be restarted.
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from fractions import Fraction
from pathlib import Path

N_FRAMES, RATE = 300, 24
MOVING_MEMBER = re.compile(r"^la_cater_moving/(\w+)_data/(frames/(CLEVR_new_\d+)/(\d{3})\.jpg|"
                           r"labels/[^/]+_bb\.json|containment_and_occlusions/[^/]+\.txt)$")


def extract(zip_path: Path, out: Path, rename):
    """Extract the members for which rename(name) gives a path (relative to out)."""
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            dst = None if info.is_dir() else rename(info.filename)
            if dst is None:
                continue
            path = out / dst
            if path.exists() and path.stat().st_size == info.file_size:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with z.open(info) as src, open(tmp, "wb") as f:
                while chunk := src.read(1 << 22):
                    f.write(chunk)
            tmp.replace(path)
            n += 1
    print(f"{zip_path.name}: {n} files extracted", flush=True)


def prepare_static(downloads: Path, out: Path):
    keep = re.compile(r"^test_data/(videos/[^/]+\.avi|labels/[^/]+_bb\.json|containment_and_occlusions/[^/]+\.txt)$")
    extract(downloads / "test_data.zip", out, lambda n: n if keep.match(n) else None)
    train = re.compile(r"^train_data_part_\d/videos_\d/([^/]+\.avi)$")
    for part in ("train_data_part_1.zip", "train_data_part_2.zip"):
        extract(downloads / part, out, lambda n: f"train_data/videos/{m.group(1)}" if (m := train.match(n)) else None)
    for split in ("train", "test"):
        print(f"{split}: {len(list((out / f'{split}_data' / 'videos').glob('*.avi')))} videos", flush=True)


def write_avi(path: Path, jpegs):
    """One MJPEG .avi from JPEG frames of 320 x 240, each frame's bytes one packet (no re-encoding)."""
    import av
    tmp = path.with_name(path.stem + ".tmp.avi")
    with av.open(str(tmp), "w", format="avi") as out:
        st = out.add_stream("mjpeg", rate=RATE)
        st.width, st.height, st.pix_fmt = 320, 240, "yuvj420p"
        for i, b in enumerate(jpegs):
            pkt = av.Packet(b)
            pkt.stream, pkt.time_base, pkt.pts, pkt.dts, pkt.is_keyframe = st, Fraction(1, RATE), i, i, True
            out.mux(pkt)
    tmp.replace(path)


def prepare_moving(tarball: str, out: Path):
    tar = (tarfile.open(fileobj=sys.stdin.buffer, mode="r|") if tarball == "-" else tarfile.open(tarball, mode="r|gz"))
    packed, skipped, refused = 0, 0, []
    clip, frames = None, {}

    def flush():
        nonlocal packed, skipped
        if clip is None:
            return
        path = out / f"{clip[0]}_data" / "videos" / f"{clip[1]}.avi"
        if path.exists():
            skipped += 1
        elif sorted(frames) != list(range(N_FRAMES)):
            refused.append((clip, len(frames)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_avi(path, [frames[i] for i in range(N_FRAMES)])
            packed += 1
            if packed % 500 == 0:
                print(f"{packed} clips packed", flush=True)

    with tar:
        for m in tar:
            match = MOVING_MEMBER.match(m.name) if m.isfile() else None
            if match is None:
                continue
            split, rel = match.group(1), match.group(2)
            if match.group(3) is None:                                   # labels and flags
                path = out / f"{split}_data" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(m).read())
                continue
            key = (split, match.group(3))
            if key != clip:
                flush()
                clip, frames = key, {}
            if not (out / f"{split}_data" / "videos" / f"{key[1]}.avi").exists():
                frames[int(match.group(4))] = tar.extractfile(m).read()
        flush()
    print(f"{packed} clips packed, {skipped} already present, {len(refused)} refused (not {N_FRAMES} frames): "
          f"{refused[:10]}", flush=True)
    if refused:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="camera", required=True)
    s = sub.add_parser("static")
    s.add_argument("--downloads", required=True, type=Path)
    s.add_argument("--out", default="data/la_cater", type=Path)
    m = sub.add_parser("moving")
    m.add_argument("--tarball", required=True, help="la_cater_moving.tar.gz, or - for an uncompressed tar on stdin")
    m.add_argument("--out", default="data/la_cater_moving", type=Path)
    a = ap.parse_args()
    if a.camera == "static":
        prepare_static(a.downloads, a.out)
    else:
        prepare_moving(a.tarball, a.out)


if __name__ == "__main__":
    main()
