import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml


def _ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _heuristic_hotspots(img_bgr: np.ndarray, threshold: int, min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    warm = (h <= 35) | (h >= 160)
    mask = warm & (s >= 80) & (v >= threshold)
    mask_u8 = (mask.astype(np.uint8) * 255)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    confs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < float(min_area):
            continue
        x, y, w, hh = cv2.boundingRect(c)
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + hh)

        roi = v[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            continue
        score = float(np.clip((roi.mean() - threshold) / max(1.0, (255 - threshold)), 0.0, 1.0))

        boxes.append([x1, y1, x2, y2])
        confs.append(score)

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.array(boxes, dtype=np.float32), np.array(confs, dtype=np.float32)


def _yolo_lines(boxes_xyxy: np.ndarray, img_w: int, img_h: int) -> str:
    out = []
    for x1, y1, x2, y2 in boxes_xyxy.tolist():
        xc = ((x1 + x2) / 2.0) / float(img_w)
        yc = ((y1 + y2) / 2.0) / float(img_h)
        bw = (x2 - x1) / float(img_w)
        bh = (y2 - y1) / float(img_h)
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        out.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Folder with thermal building images")
    ap.add_argument("--out", type=str, default="dataset_pseudo_hotspot", help="Output YOLO dataset folder")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=int, default=210)
    ap.add_argument("--min_area", type=int, default=600)
    ap.add_argument("--imgsz", type=int, default=640, help="Resize images to this square size (0 = keep original)")
    ap.add_argument("--copy", action="store_true", help="Copy images instead of re-encoding (ignored if imgsz != 0)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    out = Path(args.out)
    img_train = out / "images" / "train"
    img_val = out / "images" / "val"
    lbl_train = out / "labels" / "train"
    lbl_val = out / "labels" / "val"
    for p in [img_train, img_val, lbl_train, lbl_val]:
        _ensure(p)

    imgs = _iter_images(in_dir)
    if args.limit and args.limit > 0:
        imgs = imgs[: args.limit]

    rng = random.Random(args.seed)
    rng.shuffle(imgs)

    n_val = int(len(imgs) * float(args.val_ratio))
    val_set = set(imgs[:n_val])

    kept = 0
    for p in imgs:
        split = "val" if p in val_set else "train"
        img_out_dir = img_val if split == "val" else img_train
        lbl_out_dir = lbl_val if split == "val" else lbl_train

        img = cv2.imread(str(p))
        if img is None:
            continue

        if args.imgsz and args.imgsz > 0:
            img = cv2.resize(img, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)

        h, w = img.shape[:2]
        boxes, _ = _heuristic_hotspots(img, threshold=int(args.threshold), min_area=int(args.min_area))

        stem = p.stem
        out_img = img_out_dir / f"{stem}.png"
        out_lbl = lbl_out_dir / f"{stem}.txt"

        if args.copy and args.imgsz == 0:
            shutil.copy2(str(p), str(out_img))
        else:
            cv2.imwrite(str(out_img), img)

        out_lbl.write_text(_yolo_lines(boxes, w, h), encoding="utf-8")
        kept += 1

    yaml_path = out / "data.yaml"
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "hotspot"},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    print(f"OK: wrote {kept} images")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
