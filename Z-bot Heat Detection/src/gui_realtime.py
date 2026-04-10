import argparse
import threading
from pathlib import Path
from typing import Optional, List, Tuple
import winsound  # For Windows beep sound

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

from ultralytics import YOLO

# Reuse track logic from infer_persist
from infer_persist import Track, iou_xyxy, draw_box
from gui_detect import _bgr_to_tk, _heuristic_hotspots

class RealTimeApp:
    def __init__(self, root: tk.Tk, weights: Path, conf: float, imgsz: int, source: str):
        self.root = root
        self.root.title("Z-BOT Real-Time Heat Detection")

        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        
        self.is_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Tracking config
        self.persist_frames = 3
        self.iou_match = 0.35
        self.max_missed = 8
        self.tracks: List[Track] = []

        try:
            self.model = YOLO(str(weights))
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {e}\nFalling back to heuristic mode if available.")
            self.model = None

        # Try opening camera to verify
        source_idx = int(source) if source.isdigit() else source
        self.cap_source = source_idx

        # UI Layout
        self.top = tk.Frame(root)
        self.top.pack(fill="x", padx=10, pady=10)

        self.btn_toggle = tk.Button(self.top, text="Start Camera", command=self.toggle_camera, bg="green", fg="white", font=("Arial", 12, "bold"))
        self.btn_toggle.pack(side="left")

        self.lbl_info = tk.Label(self.top, text=f"Source={source} Conf={conf}")
        self.lbl_info.pack(side="left", padx=12)

        self.mode = tk.StringVar(value="yolo")
        self.mode_menu = tk.OptionMenu(self.top, self.mode, "yolo", "heuristic")
        self.mode_menu.config(width=10)
        self.mode_menu.pack(side="left", padx=6)
        
        # Heuristic params (hidden by default, but available)
        self.thr = tk.IntVar(value=210)
        self.scale_thr = tk.Scale(self.top, from_=120, to=250, orient="horizontal", variable=self.thr, length=120, label="Heuristic Thr")
        self.scale_thr.pack(side="left", padx=6)

        self.canvas = tk.Label(root)
        self.canvas.pack(padx=10, pady=10)

        self.status = tk.Label(root, text="Ready. Press 'Start Camera'.", anchor="w", fg="blue", font=("Arial", 10))
        self.status.pack(fill="x", padx=10, pady=(0, 10))
        
        # To avoid blocking UI, run detection in a background thread
        self.thread = None

    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.cap_source)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera source: {self.cap_source}")
            return
            
        self.is_running = True
        self.btn_toggle.config(text="Stop Camera", bg="red")
        self.status.config(text="Camera running...", fg="green")
        self.tracks = [] # Reset tracks
        
        self.thread = threading.Thread(target=self._process_frames, daemon=True)
        self.thread.start()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_toggle.config(text="Start Camera", bg="green")
        self.status.config(text="Camera stopped.", fg="blue")
        self.canvas.config(image="")
        self.tracks = []

    def _play_beep(self):
        # Play a beep: Frequency 2000Hz, Duration 300ms
        winsound.Beep(2000, 300)

    def _process_frames(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            mode = self.mode.get()
            dets: List[Tuple[np.ndarray, float]] = []
            
            # --- DETECTION ---
            if mode == "yolo" and self.model:
                results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    confs = r0.boxes.conf.cpu().numpy()
                    clss = r0.boxes.cls.cpu().numpy().astype(int)
                    # We accept class 0. Since we don't know if weights are custom yet, 
                    # generic yolov8n class 0 is person, which is good for testing.
                    for box, c, k in zip(xyxy, confs, clss):
                        if k == 0:
                            dets.append((box.astype(np.float32), float(c)))
            elif mode == "heuristic":
                xyxy, confs = _heuristic_hotspots(frame, threshold=int(self.thr.get()), min_area=600)
                for box, c in zip(xyxy, confs):
                     dets.append((box.astype(np.float32), float(c)))

            # --- TRACKING ---
            matched_det = set()
            for t in self.tracks:
                best_j = -1
                best_iou = 0.0
                for j, (dbox, _) in enumerate(dets):
                    if j in matched_det:
                        continue
                    v = iou_xyxy(t.box, dbox)
                    if v > best_iou:
                        best_iou = v
                        best_j = j

                if best_j >= 0 and best_iou >= self.iou_match:
                    dbox, _ = dets[best_j]
                    t.box = 0.7 * t.box + 0.3 * dbox
                    t.hits += 1
                    t.missed = 0
                    matched_det.add(best_j)
                else:
                    t.missed += 1

            for j, (dbox, _) in enumerate(dets):
                if j in matched_det:
                    continue
                self.tracks.append(Track(box=dbox, hits=1, missed=0))

            self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

            # --- ALARM & DRAWING ---
            alarm_triggered = False
            for t in self.tracks:
                stable = t.hits >= self.persist_frames
                if stable:
                    alarm_triggered = True
                
                color = (0, 0, 255) if stable else (0, 255, 255)
                # BGR color mapping for OpenCV drawing
                draw_box(frame, t.box, color, f"HOTSPOT (hits={t.hits})" if stable else f"detecting {t.hits}")

            if alarm_triggered:
                cv2.putText(frame, "!!! ALARM: HOTSPOT DETECTED !!!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # Play beep in another thread to not freeze video
                threading.Thread(target=self._play_beep, daemon=True).start()
                
            # Resize frame for GUI to avoid taking whole screen
            max_w, max_h = 800, 600
            h, w = frame.shape[:2]
            scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
            if scale < 1.0:
                 frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Update GUI
            rgb_tk = _bgr_to_tk(frame)
            self.root.after(0, self._update_canvas, rgb_tk)

    def _update_canvas(self, tk_img):
        # We need to keep a reference to prevent garbage collection
        self._tk_img = tk_img
        self.canvas.config(image=self._tk_img)

def main() -> None:
    ap = argparse.ArgumentParser()
    # Defaulting to yolov8n.pt first since model might not be trained. 
    # If custom weights exist, use them.
    default_weights = "runs/detect/hotspot_yolov8n/weights/best.pt"
    if not Path(default_weights).exists():
        default_weights = "yolov8n.pt"
        print(f"Custom weights not found, using {default_weights} for testing.")

    ap.add_argument("--weights", type=str, default=default_weights)
    ap.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0) or video file")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--auto-start", action="store_true", help="Automatically start the camera on launch")
    args = ap.parse_args()

    weights = Path(args.weights)

    root = tk.Tk()
    app = RealTimeApp(root, weights=weights, conf=args.conf, imgsz=args.imgsz, source=args.source)
    
    if args.auto_start:
        app.start_camera()

    
    # Handle window close gracefully
    def on_closing():
        app.stop_camera()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
