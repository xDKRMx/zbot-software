import argparse
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from ultralytics import YOLO


def _bgr_to_tk(img_bgr: np.ndarray) -> tk.PhotoImage:
    # Convert BGR -> RGB -> PPM bytes for Tkinter PhotoImage (no Pillow dependency)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    header = f"P6 {w} {h} 255\n".encode("ascii")
    data = header + rgb.tobytes()
    return tk.PhotoImage(data=data, format="PPM")


def _draw_dets(img: np.ndarray, boxes_xyxy: np.ndarray, confs: np.ndarray) -> np.ndarray:
    out = img.copy()
    for box, c in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = box.astype(int).tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"hotspot {c:.2f}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def _heuristic_hotspots(img_bgr: np.ndarray, threshold: int, min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Typical false-color thermal palettes map "hot" to red/yellow/white.
    # Mask: high brightness and high saturation + warm hues.
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


class App:
    def __init__(self, root: tk.Tk, weights: Path, conf: float, imgsz: int):
        self.root = root
        self.root.title("Z-BOT Heat Detection - Image GUI")

        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz

        self.model = YOLO(str(weights))

        self._tk_img: Optional[tk.PhotoImage] = None
        self._last_path: Optional[Path] = None

        # UI
        self.top = tk.Frame(root)
        self.top.pack(fill="x", padx=10, pady=10)

        self.btn_open = tk.Button(self.top, text="Open Image", command=self.open_image)
        self.btn_open.pack(side="left")

        self.btn_run = tk.Button(self.top, text="Detect", command=self.run_detect, state="disabled")
        self.btn_run.pack(side="left", padx=8)

        self.lbl_info = tk.Label(self.top, text=f"weights={weights.name}  conf={conf}  imgsz={imgsz}")
        self.lbl_info.pack(side="left", padx=12)

        self.mode = tk.StringVar(value="auto")
        self.mode_menu = tk.OptionMenu(self.top, self.mode, "auto", "yolo", "heuristic")
        self.mode_menu.config(width=10)
        self.mode_menu.pack(side="left", padx=6)

        self.thr = tk.IntVar(value=210)
        self.scale_thr = tk.Scale(self.top, from_=120, to=250, orient="horizontal", variable=self.thr, length=180)
        self.scale_thr.pack(side="left", padx=6)

        self.min_area = tk.IntVar(value=600)
        self.scale_area = tk.Scale(self.top, from_=50, to=5000, orient="horizontal", variable=self.min_area, length=180)
        self.scale_area.pack(side="left", padx=6)

        self.canvas = tk.Label(root)
        self.canvas.pack(padx=10, pady=10)

        self.status = tk.Label(root, text="Ready", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 10))

    def set_status(self, text: str) -> None:
        self.status.config(text=text)
        self.root.update_idletasks()

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a thermal image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        img = cv2.imread(str(p))
        if img is None:
            messagebox.showerror("Error", f"Failed to read image: {p}")
            return

        self._last_path = p
        self.btn_run.config(state="normal")
        self.show_image(img)
        self.set_status(f"Loaded: {p}")

    def show_image(self, img_bgr: np.ndarray) -> None:
        # Fit to reasonable size for screen
        max_w, max_h = 1000, 700
        h, w = img_bgr.shape[:2]
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 1.0:
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        self._tk_img = _bgr_to_tk(img_bgr)
        self.canvas.config(image=self._tk_img)

    def run_detect(self) -> None:
        if self._last_path is None:
            return

        # Do inference in a thread to keep UI responsive
        def worker():
            self.set_status("Running detection...")
            img = cv2.imread(str(self._last_path))
            if img is None:
                self.set_status("Error: failed to read image")
                return

            mode = self.mode.get().strip().lower()

            xyxy = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)

            if mode in ("auto", "yolo"):
                results = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False)
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy0 = r0.boxes.xyxy.cpu().numpy()
                    confs0 = r0.boxes.conf.cpu().numpy()
                    clss0 = r0.boxes.cls.cpu().numpy().astype(int)
                    keep = clss0 == 0
                    xyxy = xyxy0[keep].astype(np.float32)
                    confs = confs0[keep].astype(np.float32)

            if (mode in ("auto", "heuristic") and len(xyxy) == 0) or mode == "heuristic":
                xyxy, confs = _heuristic_hotspots(img, threshold=int(self.thr.get()), min_area=int(self.min_area.get()))

            out = _draw_dets(img, xyxy, confs) if len(xyxy) else img
            self.root.after(0, lambda: self.show_image(out))
            self.set_status(f"Detections: {len(xyxy)}")

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="runs/detect/hotspot_yolov8n/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    root = tk.Tk()
    App(root, weights=weights, conf=args.conf, imgsz=args.imgsz)
    root.mainloop()


if __name__ == "__main__":
    main()
