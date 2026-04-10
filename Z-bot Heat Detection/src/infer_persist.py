import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Optional dependency: allow heuristic/headless scripts to import Track/iou_xyxy
# without requiring ultralytics to be installed.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class Track:
    box: np.ndarray  # xyxy
    hits: int
    missed: int


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def draw_box(img: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = box.astype(int).tolist()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="runs/detect/hotspot_yolov8n/weights/best.pt")
    ap.add_argument("--source", type=str, default="0", help="0 for webcam, or path to video/image")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--persist_frames", type=int, default=3)
    ap.add_argument("--iou_match", type=float, default=0.35)
    ap.add_argument("--max_missed", type=int, default=8)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    if YOLO is None:
        raise ModuleNotFoundError(
            "ultralytics is not installed. Install it (pip install -r requirements.txt) "
            "or run heuristic mode via infer_rpi.py."
        )

    model = YOLO(str(weights))

    src = args.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    tracks: List[Track] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        r0 = results[0]

        dets: List[Tuple[np.ndarray, float]] = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, confs, clss):
                if k != 0:
                    continue
                dets.append((box.astype(np.float32), float(c)))

        # Track association (very simple)
        matched_det = set()
        for t in tracks:
            best_j = -1
            best_iou = 0.0
            for j, (dbox, _) in enumerate(dets):
                if j in matched_det:
                    continue
                v = iou_xyxy(t.box, dbox)
                if v > best_iou:
                    best_iou = v
                    best_j = j

            if best_j >= 0 and best_iou >= args.iou_match:
                dbox, _ = dets[best_j]
                t.box = 0.7 * t.box + 0.3 * dbox
                t.hits += 1
                t.missed = 0
                matched_det.add(best_j)
            else:
                t.missed += 1

        # New tracks
        for j, (dbox, _) in enumerate(dets):
            if j in matched_det:
                continue
            tracks.append(Track(box=dbox, hits=1, missed=0))

        # Prune
        tracks = [t for t in tracks if t.missed <= args.max_missed]

        # Draw
        alarm_on = False
        for t in tracks:
            stable = t.hits >= args.persist_frames
            if stable:
                alarm_on = True
            color = (0, 0, 255) if stable else (0, 255, 255)
            draw_box(frame, t.box, color, f"hotspot hits={t.hits}")

        if alarm_on:
            cv2.putText(frame, "HOTSPOT (PERSISTENT)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "no persistent hotspot", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        cv2.imshow("Z-BOT Heat Detection (demo)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
