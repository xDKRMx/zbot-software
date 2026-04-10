import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def _ensure_dirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _make_background(h: int, w: int, rng: random.Random) -> np.ndarray:
    base = rng.randint(40, 90)
    img = np.full((h, w), base, dtype=np.float32)

    # Low-frequency gradients
    gx = np.linspace(0, rng.uniform(-20, 20), w, dtype=np.float32)
    gy = np.linspace(0, rng.uniform(-20, 20), h, dtype=np.float32)
    img += gy[:, None]
    img += gx[None, :]

    # Add noise + blur to look thermal-ish
    noise = rng.uniform(2.0, 8.0) * np.random.randn(h, w).astype(np.float32)
    img += noise
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=rng.uniform(1.2, 2.8), sigmaY=rng.uniform(1.2, 2.8))

    return img


def _draw_hotspot(img: np.ndarray, rng: random.Random):
    h, w = img.shape[:2]

    # Hotspot center
    cx = rng.randint(int(0.15 * w), int(0.85 * w))
    cy = rng.randint(int(0.15 * h), int(0.85 * h))

    # Hotspot size
    rx = rng.randint(int(0.03 * w), int(0.12 * w))
    ry = rng.randint(int(0.03 * h), int(0.12 * h))

    # Temperature increase
    amp = rng.uniform(40.0, 110.0)

    # Create Gaussian blob
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.exp(-(((xx - cx) ** 2) / (2 * (rx**2) + 1e-6) + ((yy - cy) ** 2) / (2 * (ry**2) + 1e-6)))
    img += amp * blob

    # Bounding box (tight-ish)
    x1 = _clamp(int(cx - 2.0 * rx), 0, w - 1)
    y1 = _clamp(int(cy - 2.0 * ry), 0, h - 1)
    x2 = _clamp(int(cx + 2.0 * rx), 0, w - 1)
    y2 = _clamp(int(cy + 2.0 * ry), 0, h - 1)

    return x1, y1, x2, y2


def _to_yolo_label(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> str:
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def _render_thermal(img_f32: np.ndarray, use_colormap: bool) -> np.ndarray:
    img = img_f32.copy()
    img = np.clip(img, 0, 255)
    img_u8 = img.astype(np.uint8)

    if not use_colormap:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    cm = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
    return cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="dataset_hotspot", help="Output dataset folder")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--colormap", action="store_true", help="Save colorized thermal images (3ch).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)

    for split, n in [("train", args.train), ("val", args.val)]:
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        _ensure_dirs(img_dir)
        _ensure_dirs(lbl_dir)

        for i in range(n):
            h = w = args.imgsz
            img = _make_background(h, w, rng)

            # Some images without hotspot (hard negatives)
            has_hotspot = rng.random() > 0.18
            bbox = None
            if has_hotspot:
                bbox = _draw_hotspot(img, rng)

            # Normalize contrast a bit
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_bgr = _render_thermal(img, args.colormap)

            stem = f"{split}_{i:06d}"
            img_path = img_dir / f"{stem}.png"
            lbl_path = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_path), img_bgr)

            # YOLO label: write empty file if no object
            if bbox is None:
                lbl_path.write_text("")
            else:
                x1, y1, x2, y2 = bbox
                lbl_path.write_text(_to_yolo_label(x1, y1, x2, y2, w, h))

    # data.yaml
    yaml_path = out / "data.yaml"
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "hotspot"},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    print(f"OK: dataset generated at: {out}")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
