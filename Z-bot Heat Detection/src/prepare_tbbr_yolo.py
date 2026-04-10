import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


def _ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm_thermal_to_u8(thermal: np.ndarray) -> np.ndarray:
    t = thermal.astype(np.float32)
    lo = np.percentile(t, 2.0)
    hi = np.percentile(t, 98.0)
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
    return (t * 255.0).astype(np.uint8)


def _yolo_line_from_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    xc = ((x1 + x2) / 2.0) / float(w)
    yc = ((y1 + y2) / 2.0) / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    # clamp
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def _load_coco(coco_json: Path) -> Tuple[List[dict], Dict[int, List[dict]]]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    anns = data.get("annotations", [])

    anns_by_image: Dict[int, List[dict]] = {}
    for a in anns:
        img_id = int(a["image_id"])
        anns_by_image.setdefault(img_id, []).append(a)

    return images, anns_by_image


def _resolve_npy(root_images_dir: Path, file_name: str) -> Path:
    # COCO uses relative paths like images/Flug1_100Media/DJI_0004_R.npy
    # We accept either already-rooted path or relative.
    p = Path(file_name)
    if p.is_absolute():
        return p

    # common cases
    cand1 = root_images_dir / p
    if cand1.exists():
        return cand1

    # sometimes COCO paths start with "images/" already
    if len(p.parts) > 0 and p.parts[0].lower() == "images":
        cand2 = root_images_dir / Path(*p.parts[1:])
        if cand2.exists():
            return cand2

    # fallback: try as-is under root
    cand3 = root_images_dir / file_name
    return cand3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbbr_root", type=str, required=True, help="Folder that contains train/ and/or test/ as in Zenodo usage")
    ap.add_argument("--coco", type=str, required=True, help="COCO json path (e.g. .../Flug1_100-104Media_coco.json)")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Output split name")
    ap.add_argument("--out", type=str, default="dataset_tbbr_yolo", help="Output YOLO dataset folder")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.0, help="If >0, create both train/val splits in one run")
    ap.add_argument("--imgsz", type=int, default=640, help="Resize output images to this square size")
    args = ap.parse_args()

    tbbr_root = Path(args.tbbr_root)
    coco_path = Path(args.coco)
    out = Path(args.out)

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO json not found: {coco_path}")

    # root_images_dir should point to the folder that contains the per-flight subfolders
    # In the recommended structure: <tbbr_root>/train/images/<FlightFolders>
    # We'll derive it from the COCO file location if possible.
    root_images_dir = (coco_path.parent / "images")
    if not root_images_dir.exists():
        # fallback: user provided tbbr_root directly
        cand = tbbr_root / "images"
        if cand.exists():
            root_images_dir = cand

    images, anns_by_image = _load_coco(coco_path)

    rng = random.Random(args.seed)
    idxs_all = list(range(len(images)))
    rng.shuffle(idxs_all)
    if args.limit and args.limit > 0:
        idxs_all = idxs_all[: args.limit]

    # If val_ratio > 0, we create a deterministic split from the same COCO source.
    if args.val_ratio and args.val_ratio > 0.0:
        vr = float(args.val_ratio)
        vr = max(0.0, min(vr, 0.95))
        n_val = int(round(len(idxs_all) * vr))
        idxs_val = idxs_all[:n_val]
        idxs_train = idxs_all[n_val:]
        split_to_idxs = {"train": idxs_train, "val": idxs_val}
    else:
        split_to_idxs = {args.split: idxs_all}

    kept = 0
    for split_name, idxs in split_to_idxs.items():
        img_dir = out / "images" / split_name
        lbl_dir = out / "labels" / split_name
        _ensure(img_dir)
        _ensure(lbl_dir)

        for k in idxs:
            im = images[k]
            img_id = int(im["id"])
            file_name = im.get("file_name")
            if not file_name:
                continue

            npy_path = _resolve_npy(root_images_dir, file_name)
            if not npy_path.exists():
                continue

            arr = np.load(str(npy_path))
            # shape (H, W, 5) in order [B,G,R,Thermal,Height]
            if arr.ndim != 3 or arr.shape[2] < 4:
                continue

            thermal = arr[:, :, 3]
            th_u8 = _norm_thermal_to_u8(thermal)
            th_bgr = cv2.cvtColor(th_u8, cv2.COLOR_GRAY2BGR)

            h0, w0 = th_bgr.shape[:2]
            if args.imgsz and args.imgsz > 0:
                th_bgr = cv2.resize(th_bgr, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)

            # scale bboxes from original size to resized
            h1, w1 = th_bgr.shape[:2]
            sx = w1 / float(w0)
            sy = h1 / float(h0)

            anns = anns_by_image.get(img_id, [])
            lines = []
            for a in anns:
                bbox = a.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, bw, bh = bbox
                x1 = float(x) * sx
                y1 = float(y) * sy
                x2 = float(x + bw) * sx
                y2 = float(y + bh) * sy
                if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                    continue
                lines.append(_yolo_line_from_xyxy(x1, y1, x2, y2, w1, h1))

            stem = npy_path.stem
            out_img = img_dir / f"{stem}.png"
            out_lbl = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(out_img), th_bgr)
            out_lbl.write_text("".join(lines), encoding="utf-8")
            kept += 1

    # Create/overwrite data.yaml at dataset root
    yaml_path = out / "data.yaml"
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "thermal_bridge"},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    print(f"OK: wrote {kept} images")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
