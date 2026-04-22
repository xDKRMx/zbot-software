"""Z-BOT Thermal Panorama GUI - 100% z-bot-map architecture

Minimal GUI wrapper around ThermalMapper (port of LiveWebcamMapper).
NO complex threading, NO worker queues - just direct processing like live_webcam_map.py.
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    print("[ERROR] Install Pillow: pip install Pillow")
    exit(1)

from stitcher import ThermalMapper

# GUI Constants
PANEL_W = 480
PANEL_H = 360
UPDATE_MS = 100  # ~10 fps


class ThermalPanoramaGUI:
    """Simple GUI for z-bot-map thermal mapper."""
    
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        
        self.root.title("Z-BOT Thermal Panorama (z-bot-map)")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)
        
        # State
        self.running = False
        self.cap_rgb: Optional[cv2.VideoCapture] = None
        self.cap_thermal: Optional[cv2.VideoCapture] = None
        self.mapper: Optional[ThermalMapper] = None
        self.last_process_time = 0.0
        self.process_interval = 1.0 / args.capture_fps
        
        # Video mode
        self.video_file: Optional[str] = None
        self.is_video_mode = False
        
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start refresh loop
        self.root.after(UPDATE_MS, self._refresh_loop)
    
    def _build_ui(self):
        """Build UI (simple z-bot-map style)."""
        PAD = 8
        
        # Top bar
        top = tk.Frame(self.root, bg="#313244", pady=5)
        top.pack(fill="x")
        
        tk.Label(top, text="Z-BOT Thermal Panorama (z-bot-map)", 
                 font=("Helvetica", 11, "bold"),
                 bg="#313244", fg="#cdd6f4").pack(side="left", padx=10)
        
        # Buttons
        self.btn_start = tk.Button(
            top, text="▶ Start", width=8,
            bg="#a6e3a1", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", command=self._toggle)
        self.btn_start.pack(side="right", padx=4)
        
        self.btn_export = tk.Button(
            top, text="💾 Export", width=8,
            bg="#89b4fa", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", state="disabled", command=self._export)
        self.btn_export.pack(side="right", padx=4)
        
        self.btn_reset = tk.Button(
            top, text="🔄 Reset", width=8,
            bg="#f38ba8", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", state="disabled", command=self._reset)
        self.btn_reset.pack(side="right", padx=4)
        
        self.btn_import = tk.Button(
            top, text="📁 Import Video", width=12,
            bg="#fab387", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", command=self._import_video)
        self.btn_import.pack(side="right", padx=4)
        
        # Source selector
        src = tk.Frame(self.root, bg="#1e1e2e", pady=3)
        src.pack(fill="x")
        
        tk.Label(src, text="Heat source:", bg="#1e1e2e",
                 fg="#a6adc8", font=("Helvetica", 9)).pack(side="left", padx=10)
        
        self.source_var = tk.StringVar(value="rgb")
        tk.Radiobutton(src, text="RGB camera", variable=self.source_var,
                      value="rgb", bg="#1e1e2e", fg="#cdd6f4",
                      selectcolor="#313244", font=("Helvetica", 9)).pack(side="left", padx=5)
        tk.Radiobutton(src, text="Thermal/IR camera", variable=self.source_var,
                      value="thermal", bg="#1e1e2e", fg="#cdd6f4",
                      selectcolor="#313244", font=("Helvetica", 9)).pack(side="left")
        
        # Panels
        panels = tk.Frame(self.root, bg="#1e1e2e")
        panels.pack(pady=PAD)
        
        # Live Camera
        cam_frame = tk.Frame(panels, bg="#313244")
        cam_frame.pack(side="left", padx=PAD)
        tk.Label(cam_frame, text="Live Camera", bg="#313244", fg="#a6adc8",
                font=("Helvetica", 9)).pack()
        self.cam_label = tk.Label(cam_frame, bg="#11111b", width=PANEL_W, height=PANEL_H)
        self.cam_label.pack()
        
        # Thermal Heatmap
        map_frame = tk.Frame(panels, bg="#313244")
        map_frame.pack(side="left", padx=PAD)
        tk.Label(map_frame, text="Thermal Heat Map (z-bot-map memory canvas)",
                bg="#313244", fg="#a6adc8", font=("Helvetica", 9)).pack()
        self.map_label = tk.Label(map_frame, bg="#11111b", width=PANEL_W, height=PANEL_H)
        self.map_label.pack()
        
        # Status bar
        status_bar = tk.Frame(self.root, bg="#313244", pady=5)
        status_bar.pack(fill="x")
        
        self.status_var = tk.StringVar(value="Press ▶ Start to begin")
        tk.Label(status_bar, textvariable=self.status_var,
                bg="#313244", fg="#cdd6f4", font=("Helvetica", 9),
                anchor="w").pack(side="left", padx=10, fill="x", expand=True)
        
        self.stats_var = tk.StringVar(value="")
        tk.Label(status_bar, textvariable=self.stats_var,
                bg="#313244", fg="#a6e3a1", font=("Helvetica", 9),
                anchor="e").pack(side="right", padx=10)
        
        # Initial placeholders
        self._show_placeholder(self.cam_label, "Press ▶ Start")
        self._show_placeholder(self.map_label, "Waiting for frames…")
    
    def _show_placeholder(self, label: tk.Label, text: str):
        """Show placeholder text."""
        img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        img[:] = (17, 17, 27)
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(img, text, ((PANEL_W - tw) // 2, (PANEL_H + th) // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (88, 91, 112), 1)
        self._put_image(label, img)
    
    def _put_image(self, label: tk.Label, img_bgr: np.ndarray):
        """Display image in label."""
        h, w = img_bgr.shape[:2]
        scale = min(PANEL_W / max(w, 1), PANEL_H / max(h, 1))
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        canvas[:] = (17, 17, 27)
        y0, x0 = (PANEL_H - nh) // 2, (PANEL_W - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil)
        label.config(image=tk_img)
        label.image = tk_img
    
    def _toggle(self):
        """Toggle start/stop."""
        if self.running:
            self._stop()
        else:
            self._start()
    
    def _start(self):
        """Start capture (z-bot-map style - no threads!)."""
        if self.running:
            return
        
        # Open cameras
        if not self.is_video_mode:
            self.cap_rgb = cv2.VideoCapture(self.args.rgb)
            if not self.cap_rgb.isOpened():
                messagebox.showerror("Error", f"Cannot open RGB camera {self.args.rgb}")
                return
            
            if self.args.thermal >= 0:
                self.cap_thermal = cv2.VideoCapture(self.args.thermal)
                if not self.cap_thermal.isOpened():
                    messagebox.showwarning("Warning", f"Cannot open thermal camera {self.args.thermal}")
                    self.cap_thermal = None
        else:
            self.cap_rgb = cv2.VideoCapture(self.video_file)
            if not self.cap_rgb.isOpened():
                messagebox.showerror("Error", f"Cannot open video: {self.video_file}")
                return
        
        # Create z-bot-map mapper
        self.mapper = ThermalMapper(
            canvas_width=2200,
            canvas_height=1600,
            canvas_pad=120,
            max_canvas_mp=64.0,
            detector="orb",
            nfeatures=2000,
            memory_alpha=0.45,
            anchor_step_px=8.0,
            anchor_rotation_deg=1.0,
            lock_small_motion_updates=True,
        )
        
        self.running = True
        self.last_process_time = time.time()
        
        self.btn_start.config(text="⏹ Stop", bg="#f38ba8")
        self.btn_export.config(state="disabled")
        self.btn_reset.config(state="disabled")
        self.status_var.set("Running - z-bot-map memory canvas active")
    
    def _stop(self):
        """Stop capture."""
        self.running = False
        
        if self.cap_rgb:
            self.cap_rgb.release()
            self.cap_rgb = None
        if self.cap_thermal:
            self.cap_thermal.release()
            self.cap_thermal = None
        
        self.btn_start.config(text="▶ Start", bg="#a6e3a1")
        self.btn_export.config(state="normal")
        self.btn_reset.config(state="normal")
        self.status_var.set("Stopped - press Export to save or Reset to clear")
    
    def _reset(self):
        """Reset mapper."""
        if self.running:
            return
        
        if self.mapper:
            self.mapper.reset()
        
        self.video_file = None
        self.is_video_mode = False
        self.btn_import.config(bg="#fab387")
        
        self._show_placeholder(self.cam_label, "Press ▶ Start")
        self._show_placeholder(self.map_label, "Waiting for frames…")
        
        self.status_var.set("Reset complete - press Start")
        self.stats_var.set("")
    
    def _import_video(self):
        """Import video file."""
        if self.running:
            messagebox.showwarning("Warning", "Stop processing first")
            return
        
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if path:
            self.video_file = path
            self.is_video_mode = True
            self.btn_import.config(bg="#a6e3a1")
            self.status_var.set(f"Video loaded: {Path(path).name}")
    
    def _export(self):
        """Export thermal heatmap."""
        if not self.mapper:
            messagebox.showinfo("Export", "Nothing to export yet")
            return
        
        heatmap = self.mapper.get_thermal_heatmap()
        if heatmap.size == 0:
            messagebox.showwarning("Export", "No frames stitched yet")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")]
        )
        
        if path:
            cv2.imwrite(path, heatmap)
            messagebox.showinfo("Export", f"Saved:\n{path}")
            self.status_var.set(f"Exported → {Path(path).name}")
    
    def _refresh_loop(self):
        """Main loop - z-bot-map style (called from Tk event loop)."""
        if self.running and self.cap_rgb:
            # Read frame
            ret, frame_rgb = self.cap_rgb.read()
            if not ret:
                if self.is_video_mode:
                    self._stop()
                    self.status_var.set("Video complete - press Export to save")
                return
            
            # Read thermal (if available)
            frame_thermal = None
            if self.cap_thermal:
                ret_t, frame_thermal = self.cap_thermal.read()
                if not ret_t:
                    frame_thermal = None
            
            # Process at capture_fps rate
            now = time.time()
            if (now - self.last_process_time) >= self.process_interval:
                self.last_process_time = now
                
                # Prepare thermal data
                use_thermal = self.source_var.get() == "thermal"
                thermal_gray = None
                if use_thermal and frame_thermal is not None:
                    thermal_gray = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
                elif not use_thermal:
                    thermal_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
                
                # z-bot-map process
                processed_view = self.mapper.process(frame_rgb, thermal_gray, now)
                
                # Update displays
                self._put_image(self.cam_label, processed_view)
                
                heatmap = self.mapper.get_thermal_heatmap()
                if heatmap.size > 0:
                    self._put_image(self.map_label, heatmap)
                
                # Update stats
                self.stats_var.set(
                    f"Stitched: {self.mapper.mapped_frames}  |  "
                    f"Tracked: {self.mapper.tracked_frames}  |  "
                    f"Rejected: {self.mapper.rejected_frames}"
                )
                self.status_var.set(f"z-bot-map: {self.mapper.status}")
            else:
                # Just update camera view
                self._put_image(self.cam_label, frame_rgb)
        
        # Schedule next refresh
        self.root.after(UPDATE_MS, self._refresh_loop)
    
    def _on_close(self):
        """Handle window close."""
        if self.running:
            self._stop()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Z-BOT Thermal Panorama (z-bot-map)")
    parser.add_argument("--rgb", type=int, default=0, help="RGB camera index")
    parser.add_argument("--thermal", type=int, default=-1, help="Thermal camera index (-1=none)")
    parser.add_argument("--capture-fps", type=float, default=5.0, help="Stitch frames per second")
    parser.add_argument("--source", choices=["rgb", "thermal"], default="rgb", help="Heat source")
    
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ThermalPanoramaGUI(root, args)
    
    if args.source == "thermal":
        app.source_var.set("thermal")
    
    root.mainloop()


if __name__ == "__main__":
    main()
