"""Panoramic Heat Extraction GUI — Z-BOT

Left  : live camera feed (always visible after Start)
Right : real-time growing 2D thermal heat map (fixed size, content scales inside)

Usage:
    python gui.py                           # webcam only
    python gui.py --rgb 0 --thermal 1      # webcam + IR camera
    python gui.py --rgb 0 --thermal 1 --source thermal
    python gui.py --rgb 0 --capture-fps 1.0
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from stitcher import PanoramaConfig, PanoramaStitcher

# Try to import Pure Thermal bridge (for FLIR Lepton 2.5)
try:
    from purethermal_python import PureThermalCamera
    PURETHERMAL_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    PURETHERMAL_AVAILABLE = False
    print(f"[INFO] Pure Thermal bridge not available: {e}")

# Import Lepton thermal processor (for fixed-scale normalization)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _zbot_path = _Path(__file__).resolve().parent.parent / "zbot-eyes" / "src"
    if _zbot_path.exists() and str(_zbot_path) not in _sys.path:
        _sys.path.insert(0, str(_zbot_path))
    from net_inspector.lepton_processor import LeptonThermalProcessor
    LEPTON_PROCESSOR_AVAILABLE = True
except ImportError as e:
    LEPTON_PROCESSOR_AVAILABLE = False
    LeptonThermalProcessor = None
    print(f"[INFO] Lepton processor not available: {e}")


def discover_pure_thermal_camera() -> Optional[Tuple[int, int, str]]:
    """Scan all camera indices to find Pure Thermal board.
    
    Returns:
        (index, backend, backend_name) if found, None otherwise
    """
    print("\n[AUTO-DISCOVER] Quick scan for Pure Thermal camera (timeout: 10s)...")
    
    import time as _time
    start_time = _time.time()
    
    backends_to_try = []
    if sys.platform == "win32":
        # Only try MSMF - DSHOW is too slow
        backends_to_try = [(cv2.CAP_MSMF, "CAP_MSMF")]
    else:
        backends_to_try = [(cv2.CAP_V4L2, "CAP_V4L2")]
    
    # Scan only likely indices (0-3) to save time
    for idx in range(4):
        if (_time.time() - start_time) > 10.0:
            print("[AUTO-DISCOVER] ⏱️ Timeout reached (10s)")
            break
            
        for backend, backend_name in backends_to_try:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                
                # Configure for Y16
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                
                # Test read with timeout
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Check if it's Pure Thermal signature
                    is_thermal = (
                        frame.dtype == np.uint16 and 
                        (frame.shape == (120, 160) or frame.shape == (60, 80))
                    ) or (
                        len(frame.shape) == 2 and
                        (frame.shape == (120, 160) or frame.shape == (60, 80))
                    )
                    
                    if is_thermal:
                        print(f"[AUTO-DISCOVER] FOUND Pure Thermal: index={idx}, backend={backend_name}")
                        print(f"[AUTO-DISCOVER] Shape: {frame.shape}, dtype: {frame.dtype}")
                        cap.release()
                        return (idx, backend, backend_name)
                
                cap.release()
            except Exception:
                pass
    
    print("[AUTO-DISCOVER] Pure Thermal not found (OpenCV cannot access it)")
    return None

try:
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk
except ImportError:
    print("[ERROR] Install Pillow:  pip install Pillow")
    sys.exit(1)

from stitcher import PanoramaConfig, PanoramaStitcher

# ── Layout constants ─────────────────────────────────────────────────────────
PANEL_W = 480   # width of each panel (camera + heatmap)
PANEL_H = 360   # height of each panel
UPDATE_MS = 80  # GUI refresh ~12 fps


class PanoramaGUI:
    def __init__(self, root: tk.Tk, config: PanoramaConfig) -> None:
        self._cfg = config
        self._root = root
        self._root.title("Z-BOT Panoramic Heat Extraction")
        self._root.resizable(False, False)
        self._root.configure(bg="#1e1e2e")

        self._running = False
        self._cap_rgb: Optional[cv2.VideoCapture] = None
        self._cap_thermal: Optional[cv2.VideoCapture] = None
        self._stitcher: Optional[PanoramaStitcher] = None
        self._cam_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_thermal_viz: Optional[np.ndarray] = None
        self._latest_heatmap: Optional[np.ndarray] = None
        
        # Lepton thermal processor (fixed-scale normalization)
        if LEPTON_PROCESSOR_AVAILABLE:
            self._lepton_processor = LeptonThermalProcessor(
                min_raw=28815,  # ~15°C
                max_raw=33315,  # ~60°C
                colormap=cv2.COLORMAP_JET,
                apply_histogram_eq=False,
            )
        else:
            self._lepton_processor = None
        self._frame_lock = threading.Lock()
        self._last_cap_ts = 0.0

        self._build_ui()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        PAD = 8

        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self._root, bg="#313244", pady=5)
        top.pack(fill="x")

        tk.Label(top, text="Z-BOT  Panoramic Heat Extraction",
                 font=("Helvetica", 11, "bold"),
                 bg="#313244", fg="#cdd6f4").pack(side="left", padx=10)

        self._btn_start = tk.Button(
            top, text="▶ Start", width=9,
            bg="#a6e3a1", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", command=self._toggle)
        self._btn_start.pack(side="right", padx=4)

        self._btn_export = tk.Button(
            top, text="💾 Export", width=9,
            bg="#89b4fa", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", state="disabled", command=self._export)
        self._btn_export.pack(side="right", padx=4)

        self._btn_reset = tk.Button(
            top, text="🔄 Reset", width=9,
            bg="#f38ba8", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", state="disabled",
            command=self._reset_panorama)
        self._btn_reset.pack(side="right", padx=4)

        # ── Source selector ───────────────────────────────────────────────────
        src = tk.Frame(self._root, bg="#1e1e2e", pady=3)
        src.pack(fill="x")

        tk.Label(src, text="Heat source:", bg="#1e1e2e",
                 fg="#a6adc8", font=("Helvetica", 9)).pack(side="left", padx=10)

        self._source_var = tk.StringVar(value="rgb")
        # Thermal available if explicit index OR if thermal source requested
        has_thermal = (self._cfg.thermal_camera_idx >= 0 or 
                      self._cfg.use_thermal_as_source)

        for val, txt in [("rgb", "RGB camera"), ("thermal", "Thermal / IR camera")]:
            rb = tk.Radiobutton(
                src, text=txt, variable=self._source_var, value=val,
                bg="#1e1e2e", fg="#cdd6f4", selectcolor="#45475a",
                activebackground="#1e1e2e",
                state="normal" if (val == "rgb" or has_thermal) else "disabled")
            rb.pack(side="left", padx=4)

        if not has_thermal:
            tk.Label(src, text="(no IR camera configured)",
                     bg="#1e1e2e", fg="#585b70",
                     font=("Helvetica", 8, "italic")).pack(side="left")

        # ── Two panels side by side ───────────────────────────────────────────
        panels = tk.Frame(self._root, bg="#1e1e2e", padx=PAD, pady=PAD)
        panels.pack()

        # Left — live camera
        lf = tk.Frame(panels, bg="#1e1e2e")
        lf.pack(side="left", padx=(0, PAD))

        tk.Label(lf, text="Live Camera", bg="#1e1e2e",
                 fg="#89b4fa", font=("Helvetica", 9, "bold")).pack(anchor="w")

        self._cam_label = tk.Label(lf, bg="#11111b",
                                   width=PANEL_W, height=PANEL_H,
                                   relief="flat")
        self._cam_label.pack()
        # Show placeholder immediately
        self._show_placeholder(self._cam_label, "Press  ▶ Start")

        # Right — heat map
        rf = tk.Frame(panels, bg="#1e1e2e")
        rf.pack(side="left")

        tk.Label(rf, text="Thermal Heat Map  (builds as robot moves)",
                 bg="#1e1e2e", fg="#f38ba8",
                 font=("Helvetica", 9, "bold")).pack(anchor="w")

        self._map_label = tk.Label(rf, bg="#11111b",
                                   width=PANEL_W, height=PANEL_H,
                                   relief="flat")
        self._map_label.pack()
        self._show_placeholder(self._map_label, "Waiting for frames…")

        # ── Status bar ────────────────────────────────────────────────────────
        sb = tk.Frame(self._root, bg="#313244", pady=3)
        sb.pack(fill="x")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(sb, textvariable=self._status_var,
                 bg="#313244", fg="#a6adc8",
                 font=("Helvetica", 9), anchor="w").pack(side="left", padx=10)

        self._stats_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self._stats_var,
                 bg="#313244", fg="#a6e3a1",
                 font=("Helvetica", 9), anchor="e").pack(side="right", padx=10)

    def _show_placeholder(self, label: tk.Label, text: str) -> None:
        img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        img[:] = (17, 17, 27)  # #11111b
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(img, text,
                    ((PANEL_W - tw) // 2, (PANEL_H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (88, 91, 112), 1)
        self._put_image(label, img)

    # ── Start / Stop ─────────────────────────────────────────────────────────

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._btn_start.config(state="disabled")
            self._status_var.set("Opening camera…")
            threading.Thread(target=self._start, daemon=True).start()

    def _start(self) -> None:
        # Runs in background thread — NO direct Tkinter calls here
        
        # Open RGB camera only if NOT using thermal as source
        if not self._cfg.use_thermal_as_source:
            idx = self._cfg.rgb_camera_idx

            # Windows uses index directly, Linux uses /dev/videoX path
            if sys.platform == "win32":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(f"/dev/video{idx}")

            if not cap.isOpened():
                self._root.after(0, lambda: messagebox.showerror(
                    "Camera Error", f"Cannot open camera {idx}"))
                self._root.after(0, lambda: self._btn_start.config(
                    state="normal", text="▶ Start", bg="#a6e3a1"))
                self._root.after(0, lambda: self._status_var.set("Ready"))
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._cfg.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._cfg.height))
            self._cap_rgb = cap
        else:
            # Thermal-only mode - no RGB camera needed
            self._cap_rgb = None
            print("[INFO] Thermal-only mode - skipping RGB camera")

        # Open thermal camera (FLIR Lepton via Pure Thermal board)
        self._cap_thermal = None
        self._thermal_is_purethermal = False
        tidx = self._cfg.thermal_camera_idx
        
        # If thermal source selected, try to open thermal camera even without explicit index
        if tidx >= 0 or self._cfg.use_thermal_as_source:
            print(f"\n[THERMAL] Initializing FLIR Lepton 2.5 Pure Thermal...")
            
            # METHOD 1: Try Pure Thermal bridge DLL (Windows Media Foundation)
            if PURETHERMAL_AVAILABLE:
                print("[THERMAL] Attempting native Pure Thermal bridge...")
                try:
                    pt_cam = PureThermalCamera()
                    if pt_cam.is_connected():
                        self._cap_thermal = pt_cam
                        self._thermal_is_purethermal = True
                        print(f"[THERMAL] SUCCESS - Pure Thermal connected via bridge DLL")
                        print(f"[THERMAL] Resolution: {pt_cam.width}x{pt_cam.height} (FLIR Lepton 2.5)")
                        
                        # Perform FFC (Flat Field Correction) on startup
                        print(f"[THERMAL] Performing camera warmup and calibration...")
                        try:
                            if pt_cam.perform_ffc():
                                print(f"[THERMAL] Calibration complete\n")
                            else:
                                print(f"[THERMAL] WARNING: FFC not available (using older DLL)\n")
                        except Exception as ffc_err:
                            print(f"[THERMAL] WARNING: FFC failed: {ffc_err}\n")
                    else:
                        print(f"[THERMAL] WARNING: Bridge loaded but device not connected")
                except Exception as e:
                    print(f"[THERMAL] Bridge failed: {e}")
            else:
                print("[THERMAL] WARNING: Pure Thermal bridge DLL not available")
                print("[THERMAL] Please build PureThermalBridge.dll (see BUILD_PURETHERMAL.md)")
            
            # METHOD 2: Fallback to OpenCV (usually doesn't work for Pure Thermal)
            if self._cap_thermal is None:
                print("[THERMAL] Attempting OpenCV fallback (rarely works for Pure Thermal)...")
                discovery = discover_pure_thermal_camera()
                
                if discovery is not None:
                    found_idx, found_backend, backend_name = discovery
                    try:
                        tcap = cv2.VideoCapture(found_idx, found_backend)
                        if tcap.isOpened():
                            tcap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                            tcap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
                            tcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                            ret_test, frame_test = tcap.read()
                            if ret_test and frame_test is not None:
                                self._cap_thermal = tcap
                                self._thermal_is_purethermal = False
                                print(f"[THERMAL] OpenCV fallback success: {frame_test.shape}\n")
                            else:
                                tcap.release()
                    except Exception as e:
                        print(f"[THERMAL] OpenCV failed: {e}")
            
            # Final status
            if self._cap_thermal is None:
                print(f"[THERMAL] FAILED - Could not initialize Pure Thermal")
                print(f"[THERMAL] For exhibition: Build PureThermalBridge.dll (run purethermal_bridge\\build.bat)")
                self._root.after(0, lambda: messagebox.showerror(
                    "Pure Thermal REQUIRED", 
                    f"FLIR Lepton 2.5 not accessible!\n\n"
                    f"For exhibition, you MUST build PureThermalBridge.dll\n\n"
                    f"Steps:\n"
                    f"1. Open: purethermal_bridge\\build.bat\n"
                    f"2. Requires: Visual Studio with C++ tools\n"
                    f"3. See: BUILD_PURETHERMAL.md\n\n"
                    f"Falling back to RGB (not suitable for exhibition)."))
            else:
                print(f"[THERMAL] Ready for thermal capture\n")

        # Create stitcher
        self._stitcher = PanoramaStitcher(self._cfg, on_update=self._on_stitch_update)
        self._stitcher.start()

        self._running = True
        self._stop_evt.clear()
        self._last_cap_ts = 0.0
        self._seq = 0  # Frame sequence counter for debug logging

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        # Update UI from main thread
        self._root.after(0, lambda: self._btn_start.config(
            text="⏹ Stop", bg="#f38ba8", state="normal"))
        self._root.after(0, lambda: self._btn_export.config(state="disabled"))
        self._root.after(0, lambda: self._btn_reset.config(state="disabled"))
        self._root.after(0, lambda: self._status_var.set(
            "Running — move the robot across the surface"))
        self._root.after(UPDATE_MS, self._refresh_ui)

    def _stop(self) -> None:
        self._running = False
        self._stop_evt.set()
        # Release cameras in background to avoid blocking main thread
        def _cleanup():
            if self._cam_thread:
                self._cam_thread.join(timeout=3.0)
            if self._cap_rgb:
                self._cap_rgb.release()
                self._cap_rgb = None
            if self._cap_thermal:
                # Release thermal camera (works for both OpenCV and PureThermalCamera)
                if hasattr(self._cap_thermal, 'release'):
                    self._cap_thermal.release()
                self._cap_thermal = None
            if self._stitcher:
                self._stitcher.stop()
            self._root.after(0, lambda: self._btn_start.config(
                text="▶ Start", bg="#a6e3a1", state="normal"))
            self._root.after(0, lambda: self._btn_export.config(state="normal"))
            self._root.after(0, lambda: self._btn_reset.config(state="normal"))
            self._root.after(0, lambda: self._status_var.set(
                "Stopped — press Export to save"))
        threading.Thread(target=_cleanup, daemon=True).start()

    def _reset_panorama(self) -> None:
        if self._running:
            return
        self._stitcher = None
        with self._frame_lock:
            self._latest_heatmap = None
        self._show_placeholder(self._map_label, "Waiting for frames…")
        self._show_placeholder(self._cam_label, "Press  ▶ Start")
        self._stats_var.set("")
        self._status_var.set("Reset — press Start")
        self._btn_export.config(state="disabled")
        self._btn_reset.config(state="disabled")

    # ── Camera loop (background thread) ──────────────────────────────────────

    def _camera_loop(self) -> None:
        min_interval = 1.0 / max(self._cfg.capture_fps, 0.1)
        use_thermal = (self._source_var.get() == "thermal"
                       and self._cap_thermal is not None)

        while self._running and not self._stop_evt.is_set():
            # Read RGB frame only if RGB camera is available
            frame_rgb = None
            if self._cap_rgb is not None:
                ret, frame_rgb = self._cap_rgb.read()
                if not ret or frame_rgb is None:
                    time.sleep(0.05)
                    continue

            frame_thermal = None
            frame_thermal_display = None  # For live camera visualization
            if self._cap_thermal is not None:
                # Both PureThermalCamera and cv2.VideoCapture use .read() -> (bool, ndarray)
                ret_t, frame_thermal_raw = self._cap_thermal.read()
                if ret_t and frame_thermal_raw is not None:
                    # Process with Lepton fixed-scale normalization
                    if frame_thermal_raw.dtype == np.uint16:
                        # Use Lepton processor if available, otherwise fallback to adaptive
                        if self._lepton_processor is not None:
                            # Fixed-scale normalization (15-60°C range)
                            thermal_8bit, thermal_color = self._lepton_processor.process_frame(
                                frame_thermal_raw,
                                return_colorized=True
                            )
                            
                            # Debug: print temperature stats every 30 frames
                            if self._seq % 30 == 0:
                                stats = self._lepton_processor.get_temperature_stats(frame_thermal_raw)
                                print(f"[THERMAL] {stats['celsius_min']:.1f}°C - {stats['celsius_max']:.1f}°C "
                                      f"(raw: {stats['raw_min']}-{stats['raw_max']})")
                            
                            # For display: use colorized JET version
                            frame_thermal_display = thermal_color
                        else:
                            # Fallback: adaptive normalization (old method)
                            thermal_min = frame_thermal_raw.min()
                            thermal_max = frame_thermal_raw.max()
                            
                            if self._seq % 30 == 0:
                                print(f"[THERMAL DATA] min={thermal_min}, max={thermal_max}, range={thermal_max-thermal_min}")
                            
                            if thermal_max > thermal_min:
                                thermal_8bit = ((frame_thermal_raw - thermal_min) * (255.0 / (thermal_max - thermal_min))).astype(np.uint8)
                            else:
                                thermal_8bit = np.zeros_like(frame_thermal_raw, dtype=np.uint8)
                            
                            thermal_8bit = cv2.equalizeHist(thermal_8bit)
                            frame_thermal_display = cv2.applyColorMap(thermal_8bit, cv2.COLORMAP_JET)
                        
                        # For stitching: use grayscale BGR
                        frame_thermal = cv2.cvtColor(thermal_8bit, cv2.COLOR_GRAY2BGR)
                    else:
                        # Already 8-bit (shouldn't happen with Y16, but handle gracefully)
                        if len(frame_thermal_raw.shape) == 2:
                            frame_thermal = cv2.cvtColor(frame_thermal_raw, cv2.COLOR_GRAY2BGR)
                        else:
                            frame_thermal = frame_thermal_raw
                else:
                    frame_thermal = None

            with self._frame_lock:
                self._latest_rgb = frame_rgb.copy() if frame_rgb is not None else None
                if frame_thermal_display is not None:
                    self._latest_thermal_viz = frame_thermal_display.copy()
            
            self._seq += 1  # Increment frame counter

            # Feed stitcher at capture_fps rate
            now = time.time()
            if self._stitcher and (now - self._last_cap_ts) >= min_interval:
                self._last_cap_ts = now
                heat_src = (frame_thermal
                            if use_thermal and frame_thermal is not None
                            else frame_rgb)
                if heat_src is not None:
                    thermal_gray = cv2.cvtColor(heat_src, cv2.COLOR_BGR2GRAY)
                    rgb_for_stitcher = frame_rgb.copy() if frame_rgb is not None else heat_src.copy()
                    self._stitcher.feed_frame(rgb_for_stitcher, thermal_gray, now)

            time.sleep(0.01)

    # ── Stitch callback (worker thread → stores latest heatmap) ──────────────

    def _on_stitch_update(self, colored: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_heatmap = colored.copy()

    # ── GUI refresh (main thread) ─────────────────────────────────────────────

    def _refresh_ui(self) -> None:
        if not self._running:
            return

        with self._frame_lock:
            rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
            thermal_viz = self._latest_thermal_viz.copy() if hasattr(self, '_latest_thermal_viz') and self._latest_thermal_viz is not None else None
            hm = self._latest_heatmap.copy() if self._latest_heatmap is not None else None

        # Show thermal visualization if thermal mode, otherwise RGB
        use_thermal = self._source_var.get() == "thermal"
        if use_thermal and thermal_viz is not None:
            self._put_image(self._cam_label, thermal_viz)
        elif rgb is not None:
            self._put_image(self._cam_label, rgb)

        if hm is not None:
            # Add scale bar then fit into fixed panel
            with_bar = self._add_scale_bar(hm)
            self._put_image(self._map_label, with_bar)

        if self._stitcher:
            s = self._stitcher
            self._stats_var.set(
                f"Stitched: {s.frames_stitched}  |  "
                f"Skipped: {s.frames_skipped}  |  "
                f"Drift: {s.drift_corrections}"
            )

        self._root.after(UPDATE_MS, self._refresh_ui)

    def _put_image(self, label: tk.Label, img_bgr: np.ndarray) -> None:
        """Scale img_bgr to fit inside PANEL_W × PANEL_H, then display."""
        h, w = img_bgr.shape[:2]
        
        # For small thermal frames (160x120), FILL entire panel (no letterbox)
        if w < 200:  # Thermal camera resolution
            # Stretch to fill entire panel for maximum visibility
            resized = cv2.resize(img_bgr, (PANEL_W, PANEL_H), 
                                interpolation=cv2.INTER_LINEAR)  # Smooth upscale for thermal
        else:
            # For larger images, maintain aspect ratio with letterbox
            scale = min(PANEL_W / max(w, 1), PANEL_H / max(h, 1))
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        # Letterbox only for non-thermal (RGB) images
        if w >= 200:
            canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            canvas[:] = (17, 17, 27)
            y0 = (PANEL_H - nh) // 2
            x0 = (PANEL_W - nw) // 2
            canvas[y0:y0+nh, x0:x0+nw] = resized
        else:
            # Thermal: already full panel size
            canvas = resized

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil)
        label.config(image=tk_img, width=PANEL_W, height=PANEL_H)
        label.image = tk_img

    def _add_scale_bar(self, img: np.ndarray) -> np.ndarray:
        """Append a small 20px scale bar on the right."""
        h = img.shape[0]
        grad = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
        bar = cv2.applyColorMap(np.repeat(grad, 20, axis=1), cv2.COLORMAP_JET)
        return np.hstack([img, bar])

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self) -> None:
        if self._stitcher is None:
            messagebox.showinfo("Export", "Nothing to export yet.")
            return
        path = self._stitcher.export()
        if path:
            self._status_var.set(f"Exported → {path.name}")
            messagebox.showinfo("Export complete", f"Saved:\n{path}")
        else:
            messagebox.showwarning("Export", "No frames stitched yet.")

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._running:
            self._stop()
        self._root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Z-BOT Panoramic Heat Extraction GUI")
    parser.add_argument("--rgb", type=int, default=0,
                        help="RGB/webcam index (default: 0)")
    parser.add_argument("--thermal", type=int, default=-1,
                        help="Thermal/IR camera index (-1=none)")
    parser.add_argument("--source", choices=["rgb", "thermal"], default="rgb",
                        help="Heat source camera (default: rgb)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--capture-fps", type=float, default=5.0,
                        help="Stitch frames per second (default: 5.0)")
    parser.add_argument("--thermal-minv", type=int, default=40)
    parser.add_argument("--thermal-maxv", type=int, default=160)
    parser.add_argument("--dx", type=int, default=0)
    parser.add_argument("--dy", type=int, default=0)
    args = parser.parse_args()

    cfg = PanoramaConfig(
        rgb_camera_idx=args.rgb,
        thermal_camera_idx=args.thermal,
        use_thermal_as_source=(args.source == "thermal"),
        width=args.width,
        height=args.height,
        capture_fps=args.capture_fps,
        thermal_minv=args.thermal_minv,
        thermal_maxv=args.thermal_maxv,
        rgb_thermal_dx=args.dx,
        rgb_thermal_dy=args.dy,
    )

    root = tk.Tk()
    app = PanoramaGUI(root, cfg)
    if args.source == "thermal" and args.thermal >= 0:
        app._source_var.set("thermal")
    root.mainloop()


if __name__ == "__main__":
    main()
