"""Z-BOT Exhibition Dashboard — Single unified GUI for all subsystems.

Runs wall/net/debris/fire detection, thermal heat map panorama, and GLM
orchestrator under one roof. Single command launch for exhibition.

Usage:
    python dashboard.py                          # RGB only
    python dashboard.py --rgb 0 --thermal 1     # RGB + IR camera
    python dashboard.py --rgb 0 --thermal 1 --glm-interval 15
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ZBOT_SRC = _HERE / "src"
_PANO = _HERE.parent / "panoramic_heat_extraction"
_ZBOT_MAP = _HERE.parent / "z-bot-map"

for _p in [str(_ZBOT_SRC), str(_PANO), str(_ZBOT_MAP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk
except ImportError:
    print("[ERROR] pip install Pillow")
    sys.exit(1)

from net_inspector.config import AppConfig
from net_inspector.orchestrator import DetectionEvent, UnifiedOrchestrator
from net_inspector.unified_runner import (
    compute_debris_mask,
    compute_fire_mask,
    compute_heat_mask,
    compute_net_mask,
)

try:
    from stitcher_zbot import ThermalMapper
    _MAPPER_AVAILABLE = True
except Exception as _e:
    print(f"[DASHBOARD] ThermalMapper unavailable: {_e}")
    _MAPPER_AVAILABLE = False

# ── Colors (dark purple/violet AI theme) ─────────────────────────────────────
C_BG       = "#0a0008"   # near-black with purple tint
C_PANEL    = "#0f000d"   # panel background — deep purple-black
C_BORDER   = "#3d0050"   # purple border
C_ACCENT1  = "#c084fc"   # soft purple — titles (like reference)
C_ACCENT2  = "#f472b6"   # pink — alerts/thermal
C_ACCENT3  = "#a855f7"   # vivid purple — GLM
C_GREEN    = "#4ade80"   # soft green — OK states
C_YELLOW   = "#fbbf24"   # amber — warnings
C_TEXT     = "#e9d5ff"   # light lavender text
C_SUBTEXT  = "#6b21a8"   # dim purple text
C_BAR      = "#0d0015"   # top/status bar — darkest purple
C_GLOW     = "#7c3aed"   # purple glow for borders

PANEL_W, PANEL_H = 400, 300
WIN_W, WIN_H = 1280, 820
UPDATE_MS = 100


# ── Heat source record ────────────────────────────────────────────────────────
@dataclass
class HeatSourceRecord:
    """One detected heat source event."""
    id: int
    timestamp: str
    pixel_x: int
    pixel_y: int
    area_px: int
    confidence: float = 0.85
    # Canvas coords (set when mapped to heat map)
    canvas_x: Optional[int] = None
    canvas_y: Optional[int] = None
    # Expiry for pink circle overlay (seconds since epoch)
    expires_at: float = 0.0


# ── Shared state ──────────────────────────────────────────────────────────────
@dataclass
class DashboardState:
    rgb_display: Optional[np.ndarray] = None
    thermal_display: Optional[np.ndarray] = None
    heatmap: Optional[np.ndarray] = None
    net_ratio: float = 0.0
    debris_ratio: float = 0.0
    fire_ratio: float = 0.0
    hotspot_detected: bool = False
    system_state: str = "WALL"
    stitched: int = 0
    tracked: int = 0
    rejected: int = 0
    mapper_status: str = ""
    error: str = ""
    # Heat source tracking
    heat_sources: List[HeatSourceRecord] = field(default_factory=list)
    heat_source_counter: int = 0


# ── Dashboard ─────────────────────────────────────────────────────────────────
class ZBotDashboard:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self._root = root
        self._args = args
        self._config = AppConfig()

        self._state = DashboardState()
        self._lock = threading.Lock()
        self._log_q: queue.Queue[str] = queue.Queue(maxsize=200)
        self._glm_q: queue.Queue[str] = queue.Queue(maxsize=20)

        self._stop_evt = threading.Event()
        self._det_thread: Optional[threading.Thread] = None
        self._orchestrator: Optional[UnifiedOrchestrator] = None
        self._mapper: Optional[ThermalMapper] = None
        self._thermal_processing = False

        self._cap_rgb: Optional[cv2.VideoCapture] = None
        self._cap_thermal: Optional[cv2.VideoCapture] = None

        self._event_cooldown: dict[str, float] = {}
        self._cooldown_s = 3.0

        self._root.title("Z-BOT Exhibition Dashboard — Challenge Cup")
        self._root.configure(bg=C_BG)
        self._root.resizable(False, False)
        self._root.geometry(f"{WIN_W}x{WIN_H}")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from tkinter import ttk

        # ── Top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self._root, bg=C_BAR, pady=6)
        top.pack(fill="x")

        # Neon title
        title_f = tk.Frame(top, bg=C_BAR)
        title_f.pack(side="left", padx=12)
        tk.Label(title_f, text="⬡ Z-BOT",
                 font=("Courier", 14, "bold"),
                 bg=C_BAR, fg=C_ACCENT1).pack(side="left")
        tk.Label(title_f, text="  EXHIBITION DASHBOARD",
                 font=("Courier", 11),
                 bg=C_BAR, fg=C_TEXT).pack(side="left")
        tk.Label(title_f, text="  ◈ CHALLENGE CUP",
                 font=("Courier", 9),
                 bg=C_BAR, fg=C_SUBTEXT).pack(side="left")

        # Buttons
        for txt, bg, cmd in [
            ("▶ START", C_GREEN, self._toggle),
            ("💾 EXPORT", C_ACCENT1, self._export),
            ("↺ RESET", C_ACCENT2, self._reset),
        ]:
            b = tk.Button(top, text=txt, width=10,
                          bg=bg, fg=C_BG,
                          font=("Courier", 9, "bold"),
                          relief="flat", cursor="hand2",
                          activebackground=bg, activeforeground=C_BG,
                          command=cmd)
            b.pack(side="right", padx=3)
            if txt == "▶ START":
                self._btn_start = b

        # Camera source indicator
        src_f = tk.Frame(top, bg=C_BAR)
        src_f.pack(side="right", padx=16)
        tk.Label(src_f, text="RGB→WALL", bg=C_BAR,
                 fg=C_GREEN, font=("Courier", 8)).pack(side="left", padx=4)
        tk.Label(src_f, text="IR→HEAT+MAP", bg=C_BAR,
                 fg=C_ACCENT2, font=("Courier", 8)).pack(side="left", padx=4)

        # ── Top panels row ────────────────────────────────────────────────────
        top_row = tk.Frame(self._root, bg=C_BG)
        top_row.pack(pady=(4, 0))

        panel_defs = [
            ("◈ RGB  ·  WALL / NET DETECTION", "rgb", C_GREEN),
            ("◈ THERMAL  ·  HEAT DETECTION", "thermal", C_ACCENT2),
            ("◈ THERMAL HEAT MAP  ·  PANORAMA", "heatmap", C_ACCENT1),
        ]
        self._panel_labels: dict[str, tk.Label] = {}
        for title, key, color in panel_defs:
            f = tk.Frame(top_row, bg=C_BG, padx=3)
            f.pack(side="left")
            # Neon title bar
            title_bar = tk.Frame(f, bg=C_PANEL, pady=2)
            title_bar.pack(fill="x")
            tk.Label(title_bar, text=title, bg=C_PANEL, fg=color,
                     font=("Courier", 8, "bold")).pack(side="left", padx=6)
            # Panel with purple glow border effect
            border = tk.Frame(f, bg=C_GLOW, padx=1, pady=1)
            border.pack()
            lbl = tk.Label(border, bg=C_PANEL, width=PANEL_W, height=PANEL_H)
            lbl.pack()
            self._panel_labels[key] = lbl
            self._show_placeholder(lbl, "PRESS  ▶ START")

        # ── Bottom section ────────────────────────────────────────────────────
        bot = tk.Frame(self._root, bg=C_BG)
        bot.pack(fill="both", expand=True, padx=4, pady=(4, 0))

        # Left column: Log (top) + GLM (bottom)
        left_col = tk.Frame(bot, bg=C_BG)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 4))

        # Detection Log
        log_hdr = tk.Frame(left_col, bg=C_PANEL, pady=2)
        log_hdr.pack(fill="x")
        tk.Label(log_hdr, text="◈ DETECTION LOG", bg=C_PANEL,
                 fg=C_GREEN, font=("Courier", 8, "bold")).pack(side="left", padx=6)
        self._log_text = tk.Text(
            left_col, height=5, bg=C_PANEL, fg=C_GREEN,
            font=("Courier", 8), state="disabled", wrap="word",
            insertbackground=C_GREEN, selectbackground=C_ACCENT3)
        self._log_text.pack(fill="both", expand=True)

        # GLM Commentary
        glm_hdr = tk.Frame(left_col, bg=C_PANEL, pady=2)
        glm_hdr.pack(fill="x", pady=(3, 0))
        tk.Label(glm_hdr, text="◈ Z-BOT AI  ·  GLM COMMENTARY", bg=C_PANEL,
                 fg=C_ACCENT3, font=("Courier", 8, "bold")).pack(side="left", padx=6)
        self._glm_text = tk.Text(
            left_col, height=5, bg=C_PANEL, fg=C_TEXT,
            font=("Helvetica", 9), state="disabled", wrap="word",
            insertbackground=C_ACCENT3, selectbackground=C_ACCENT3)
        self._glm_text.pack(fill="both", expand=True)

        # Right column: Heat Sources table
        right_col = tk.Frame(bot, bg=C_BG, width=480)
        right_col.pack(side="left", fill="both")
        right_col.pack_propagate(False)

        heat_hdr = tk.Frame(right_col, bg=C_PANEL, pady=2)
        heat_hdr.pack(fill="x")
        tk.Label(heat_hdr, text="◈ HEAT SOURCES  ·  DETECTED POSITIONS",
                 bg=C_PANEL, fg=C_ACCENT2,
                 font=("Courier", 8, "bold")).pack(side="left", padx=6)
        self._heat_summary_var = tk.StringVar(value="No heat sources yet")
        tk.Label(heat_hdr, textvariable=self._heat_summary_var,
                 bg=C_PANEL, fg=C_YELLOW,
                 font=("Courier", 7)).pack(side="right", padx=6)

        cols = ("ID", "Time", "Px X", "Px Y", "Area", "Conf")
        self._heat_tree = ttk.Treeview(right_col, columns=cols,
                                       show="headings", height=9)
        col_widths = [35, 65, 55, 55, 65, 50]
        for col, w in zip(cols, col_widths):
            self._heat_tree.heading(col, text=col)
            self._heat_tree.column(col, width=w, anchor="center")
        self._heat_tree.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                         background=C_PANEL, foreground=C_TEXT,
                         fieldbackground=C_PANEL, rowheight=18,
                         font=("Courier", 8))
        style.configure("Treeview.Heading",
                         background=C_BORDER, foreground=C_ACCENT1,
                         font=("Courier", 8, "bold"))
        style.map("Treeview", background=[("selected", C_ACCENT3)])

        # ── Status bar ────────────────────────────────────────────────────────
        sb = tk.Frame(self._root, bg=C_BAR, pady=3)
        sb.pack(fill="x", side="bottom")

        # Animated state indicator
        self._state_dot = tk.Label(sb, text="●", bg=C_BAR,
                                   fg=C_SUBTEXT, font=("Courier", 10))
        self._state_dot.pack(side="left", padx=(8, 2))

        self._status_var = tk.StringVar(value="READY  ·  PRESS START")
        tk.Label(sb, textvariable=self._status_var,
                 bg=C_BAR, fg=C_TEXT,
                 font=("Courier", 9), anchor="w").pack(side="left")

        self._stats_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self._stats_var,
                 bg=C_BAR, fg=C_SUBTEXT,
                 font=("Courier", 8), anchor="e").pack(side="right", padx=10)

    def _show_placeholder(self, label: tk.Label, text: str) -> None:
        img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        img[:] = (13, 13, 26)  # C_PANEL
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(img, text, ((PANEL_W - tw) // 2, (PANEL_H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 100), 1)
        self._put_image(label, img)

    def _put_image(self, label: tk.Label, img_bgr: np.ndarray) -> None:
        h, w = img_bgr.shape[:2]
        scale = min(PANEL_W / max(w, 1), PANEL_H / max(h, 1))
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        canvas[:] = (13, 13, 26)
        y0, x0 = (PANEL_H - nh) // 2, (PANEL_W - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        tk_img = ImageTk.PhotoImage(pil)
        label.config(image=tk_img)
        label.image = tk_img

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def _toggle(self) -> None:
        if self._det_thread and self._det_thread.is_alive():
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self._stop_evt.clear()

        # Open RGB camera (webcam → wall/net detection)
        idx = self._args.rgb
        if sys.platform == "win32":
            self._cap_rgb = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            # Try /dev/videoX path first (better for PureThermal/UVC on Linux)
            self._cap_rgb = cv2.VideoCapture(f"/dev/video{idx}")
            if not self._cap_rgb.isOpened():
                self._cap_rgb = cv2.VideoCapture(idx)
        if not self._cap_rgb.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open RGB camera {idx}")
            return

        # Open thermal/IR camera (→ heat detection + heat map panorama)
        self._cap_thermal = None
        tidx = self._args.thermal
        if tidx >= 0:
            if sys.platform == "win32":
                self._cap_thermal = cv2.VideoCapture(tidx, cv2.CAP_DSHOW)
            else:
                self._cap_thermal = cv2.VideoCapture(f"/dev/video{tidx}")
                if not self._cap_thermal.isOpened():
                    self._cap_thermal = cv2.VideoCapture(tidx)
            if not self._cap_thermal.isOpened():
                self._cap_thermal = None
                self._log("⚠ Thermal/IR camera unavailable — using RGB fallback for heat")
            else:
                self._log(f"✓ Thermal/IR camera {tidx} opened")

        # Create thermal mapper
        if _MAPPER_AVAILABLE:
            self._mapper = ThermalMapper(
                canvas_width=2200, canvas_height=1600, canvas_pad=120,
                max_canvas_mp=64.0, detector="orb", nfeatures=3000,
                memory_alpha=0.45, anchor_step_px=6.0,
                anchor_rotation_deg=1.5, lock_small_motion_updates=True,
            )

        # Create orchestrator
        self._orchestrator = UnifiedOrchestrator(
            config=self._config,
            glm_interval_s=self._args.glm_interval,
            enable_audio_output=False,
        )
        self._orchestrator.start()

        # Start detection thread
        self._det_thread = threading.Thread(
            target=self._detection_loop, daemon=True)
        self._det_thread.start()

        self._btn_start.config(text="⏹ STOP", bg=C_ACCENT2)
        self._status_var.set("RUNNING  ·  Z-BOT SYSTEMS ACTIVE")
        self._root.after(UPDATE_MS, self._refresh_loop)

    def _stop(self) -> None:
        self._stop_evt.set()
        if self._det_thread:
            self._det_thread.join(timeout=3.0)
        if self._orchestrator:
            self._orchestrator.stop()
            self._orchestrator = None
        if self._cap_rgb:
            self._cap_rgb.release()
            self._cap_rgb = None
        if self._cap_thermal:
            self._cap_thermal.release()
            self._cap_thermal = None
        self._btn_start.config(text="▶ START", bg=C_GREEN)
        self._status_var.set("STOPPED")

    def _reset(self) -> None:
        self._stop()
        with self._lock:
            self._state = DashboardState()
        self._mapper = None
        for lbl in self._panel_labels.values():
            self._show_placeholder(lbl, "Press ▶ Start")
        self._clear_text(self._log_text)
        self._clear_text(self._glm_text)
        for row in self._heat_tree.get_children():
            self._heat_tree.delete(row)
        self._heat_summary_var.set("No heat sources detected yet")
        self._status_var.set("RESET  ·  PRESS START")
        self._stats_var.set("")

    def _export(self) -> None:
        with self._lock:
            hm = self._state.heatmap
        if hm is None or hm.size == 0:
            messagebox.showinfo("Export", "No heat map yet.")
            return
        out_dir = Path("outputs/dashboard")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"heatmap_{stamp}.png"
        cv2.imwrite(str(path), hm)
        messagebox.showinfo("Export", f"Saved:\n{path}")

    # ── Detection loop (daemon thread) ────────────────────────────────────────

    def _detection_loop(self) -> None:
        min_interval = 1.0 / max(self._args.fps, 0.5)
        thermal_interval = 1.0 / max(self._args.thermal_fps, 0.5)
        last_ts = 0.0
        last_thermal_ts = 0.0

        try:
            while not self._stop_evt.is_set():
                now = time.time()
                if (now - last_ts) < min_interval:
                    time.sleep(0.01)
                    continue
                last_ts = now

                ret, frame_rgb = self._cap_rgb.read()
                if not ret or frame_rgb is None:
                    time.sleep(0.1)
                    continue

                frame_thermal = None
                if self._cap_thermal:
                    ret_t, frame_thermal = self._cap_thermal.read()
                    if not ret_t:
                        frame_thermal = None

                # ── Detection ────────────────────────────────────────────────
                net_mask = compute_net_mask(frame_rgb, self._config)
                net_pixels = int(np.count_nonzero(net_mask))
                net_ratio = float(net_pixels) / float(net_mask.size)

                debris_mask = compute_debris_mask(frame_rgb, net_mask, self._config)
                debris_ratio = (float(np.count_nonzero(debris_mask)) /
                                float(debris_mask.size)) if net_pixels > 0 else 0.0

                fire_mask = compute_fire_mask(frame_rgb)
                fire_ratio = float(np.count_nonzero(fire_mask)) / float(fire_mask.size)

                heat_src = frame_thermal if frame_thermal is not None else frame_rgb
                heat_mask = compute_heat_mask(heat_src, threshold=self._args.heat_threshold)
                cnts, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_cnts = [c for c in cnts if cv2.contourArea(c) >= self._args.heat_min_area]
                hotspot = len(valid_cnts) > 0

                # Extract heat source positions (centroids)
                new_heat_sources: List[HeatSourceRecord] = []
                for c in valid_cnts:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        area = int(cv2.contourArea(c))
                        with self._lock:
                            self._state.heat_source_counter += 1
                            src_id = self._state.heat_source_counter
                        rec = HeatSourceRecord(
                            id=src_id,
                            timestamp=datetime.utcnow().strftime("%H:%M:%S"),
                            pixel_x=cx, pixel_y=cy, area_px=area,
                            expires_at=time.time() + 8.0,  # show circle for 8s
                        )
                        new_heat_sources.append(rec)

                system_state = "NET" if net_ratio > self._args.net_threshold else "WALL"
                if hotspot:
                    system_state = "HOTSPOT"

                # ── Build RGB display ─────────────────────────────────────────
                disp_rgb = frame_rgb.copy()
                if net_pixels > 0:
                    ov = np.zeros_like(disp_rgb)
                    ov[net_mask > 0] = [0, 255, 0]
                    disp_rgb = cv2.addWeighted(disp_rgb, 1.0, ov, 0.3, 0)
                if np.count_nonzero(debris_mask) > 0:
                    ov = np.zeros_like(disp_rgb)
                    ov[debris_mask > 0] = [0, 140, 255]
                    disp_rgb = cv2.addWeighted(disp_rgb, 1.0, ov, 0.3, 0)
                if fire_ratio > self._args.fire_threshold:
                    ov = np.zeros_like(disp_rgb)
                    ov[fire_mask > 0] = [0, 0, 255]
                    disp_rgb = cv2.addWeighted(disp_rgb, 1.0, ov, 0.3, 0)
                cv2.putText(disp_rgb, f"Net:{net_ratio*100:.1f}%  Debris:{debris_ratio*100:.1f}%",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # ── Build thermal display ─────────────────────────────────────
                if frame_thermal is not None:
                    # Auto-scale normalization: stretch actual frame min/max to 0-255
                    # This gives maximum contrast regardless of scene temperature range
                    gray_t = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
                    # Use center ROI to avoid black borders/UI overlays
                    h_t, w_t = gray_t.shape
                    roi = gray_t[int(h_t*0.05):int(h_t*0.90),
                                 int(w_t*0.05):int(w_t*0.95)]
                    min_val = int(roi.min())
                    max_val = int(roi.max())
                    denom = max(max_val - min_val, 1)
                    # Stretch to full 0-255 range
                    x = np.clip(gray_t.astype(np.int32) - min_val, 0, denom)
                    x = ((x * 255.0) / denom).astype(np.uint8)
                    disp_thermal = cv2.applyColorMap(x, cv2.COLORMAP_INFERNO)
                else:
                    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
                    disp_thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
                if hotspot and np.count_nonzero(heat_mask) > 0:
                    ov = np.zeros_like(disp_thermal)
                    ov[heat_mask > 0] = [255, 0, 255]
                    disp_thermal = cv2.addWeighted(disp_thermal, 1.0, ov, 0.4, 0)
                cv2.putText(disp_thermal, f"Hotspot: {'YES' if hotspot else 'NO'}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255) if hotspot else (0, 255, 0), 1)

                # ── Emit events ───────────────────────────────────────────────
                ts_iso = datetime.utcnow().isoformat() + "Z"
                if system_state == "NET" and self._should_emit("NET"):
                    self._emit(DetectionEvent(
                        timestamp=ts_iso, source="wall_net", event_type="NET",
                        confidence=min(1.0, net_ratio * 10),
                        frame_rgb=frame_rgb.copy(),
                        metadata={"net_coverage_percent": round(net_ratio * 100, 2)},
                    ))
                elif system_state == "WALL" and self._should_emit("WALL"):
                    self._emit(DetectionEvent(
                        timestamp=ts_iso, source="wall_net", event_type="WALL",
                        confidence=0.8, frame_rgb=frame_rgb.copy(),
                        metadata={"net_coverage_percent": round(net_ratio * 100, 2)},
                    ))
                if debris_ratio > self._args.debris_threshold and self._should_emit("DEBRIS"):
                    self._emit(DetectionEvent(
                        timestamp=ts_iso, source="debris", event_type="DEBRIS",
                        confidence=min(1.0, debris_ratio * 20),
                        frame_rgb=frame_rgb.copy(),
                        metadata={"debris_coverage_percent": round(debris_ratio * 100, 2)},
                    ))
                if hotspot and self._should_emit("HOTSPOT"):
                    positions = [{"id": r.id, "px": r.pixel_x, "py": r.pixel_y,
                                  "area": r.area_px} for r in new_heat_sources]
                    self._emit(DetectionEvent(
                        timestamp=ts_iso, source="heat", event_type="HOTSPOT",
                        confidence=0.85,
                        frame_rgb=frame_rgb.copy(),
                        frame_thermal=frame_thermal.copy() if frame_thermal is not None else None,
                        metadata={
                            "hotspot_count": len(valid_cnts),
                            "positions": positions,
                            "total_heat_sources_detected": 0,  # updated below
                        },
                    ))

                # ── Thermal mapper ────────────────────────────────────────────
                if self._mapper and (now - last_thermal_ts) >= thermal_interval:
                    last_thermal_ts = now
                    if not self._thermal_processing:
                        self._thermal_processing = True
                        # Use raw grayscale for mapper (before colormap)
                        if frame_thermal is not None:
                            thermal_gray = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
                        else:
                            thermal_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
                        threading.Thread(
                            target=self._thermal_worker,
                            args=(frame_rgb.copy(), thermal_gray, now),
                            daemon=True,
                        ).start()

                # ── Update shared state ───────────────────────────────────────
                with self._lock:
                    self._state.rgb_display = disp_rgb
                    self._state.thermal_display = disp_thermal
                    self._state.net_ratio = net_ratio
                    self._state.debris_ratio = debris_ratio
                    self._state.fire_ratio = fire_ratio
                    self._state.hotspot_detected = hotspot
                    self._state.system_state = system_state
                    # Add new heat sources, keep all (never expire from list)
                    if new_heat_sources:
                        self._state.heat_sources.extend(new_heat_sources)

        except Exception as exc:
            with self._lock:
                self._state.error = str(exc)
            print(f"[DASHBOARD] Detection loop error: {exc}")

    def _thermal_worker(self, frame_rgb: np.ndarray,
                        thermal_gray: np.ndarray, ts: float) -> None:
        try:
            self._mapper.process(frame_rgb, thermal_gray, ts)
            hm = self._mapper.get_thermal_heatmap()
            with self._lock:
                if hm is not None and hm.size > 0:
                    self._state.heatmap = hm
                self._state.stitched = self._mapper.mapped_frames
                self._state.tracked = self._mapper.tracked_frames
                self._state.rejected = self._mapper.rejected_frames
                self._state.mapper_status = self._mapper.status
        except Exception as exc:
            print(f"[DASHBOARD] Thermal worker error: {exc}")
        finally:
            self._thermal_processing = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _should_emit(self, event_type: str) -> bool:
        now = time.time()
        if (now - self._event_cooldown.get(event_type, 0.0)) >= self._cooldown_s:
            self._event_cooldown[event_type] = now
            return True
        return False

    def _emit(self, event: DetectionEvent) -> None:
        if self._orchestrator:
            self._orchestrator.submit_event(event)
        ts = datetime.utcnow().strftime("%H:%M:%S")
        meta_str = "  ".join(f"{k}={v}" for k, v in event.metadata.items())
        self._log(f"[{ts}] {event.event_type}: conf={event.confidence:.2f}  {meta_str}")

    def _log(self, line: str) -> None:
        try:
            self._log_q.put_nowait(line)
        except queue.Full:
            pass

    def _clear_text(self, widget: tk.Text) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.config(state="disabled")

    def _append_text(self, widget: tk.Text, text: str) -> None:
        widget.config(state="normal")
        widget.insert("end", text + "\n")
        # Trim to 50 lines
        lines = int(widget.index("end-1c").split(".")[0])
        if lines > 50:
            widget.delete("1.0", f"{lines - 50}.0")
        widget.see("end")
        widget.config(state="disabled")

    # ── Refresh loop (main thread) ────────────────────────────────────────────

    def _refresh_loop(self) -> None:
        if self._stop_evt.is_set():
            return

        with self._lock:
            s = self._state

        # Update panels
        if s.rgb_display is not None:
            self._put_image(self._panel_labels["rgb"], s.rgb_display)
        if s.thermal_display is not None:
            self._put_image(self._panel_labels["thermal"], s.thermal_display)
        if s.heatmap is not None and s.heatmap.size > 0:
            # Draw pink circles for active heat sources on heatmap copy
            hm_display = s.heatmap.copy()
            now_t = time.time()
            hm_h, hm_w = hm_display.shape[:2]
            for src in s.heat_sources:
                if src.expires_at > now_t:
                    # Scale pixel coords to heatmap display size
                    # (heatmap is cropped canvas, use raw pixel coords scaled)
                    sx = min(max(int(src.pixel_x * hm_w / max(640, 1)), 0), hm_w - 1)
                    sy = min(max(int(src.pixel_y * hm_h / max(480, 1)), 0), hm_h - 1)
                    radius = max(20, int((src.area_px ** 0.5) * 0.5))
                    # Pulsing effect: fade based on remaining time
                    remaining = src.expires_at - now_t
                    alpha = min(1.0, remaining / 2.0)
                    color_intensity = int(255 * alpha)
                    cv2.circle(hm_display, (sx, sy), radius,
                               (color_intensity, 0, color_intensity), 2)
                    cv2.circle(hm_display, (sx, sy), 4,
                               (255, 100, 255), -1)
                    cv2.putText(hm_display, f"#{src.id}",
                                (sx + radius + 3, sy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 150, 255), 1)
            self._put_image(self._panel_labels["heatmap"], hm_display)

        # Update Heat Sources tab
        self._update_heat_sources_tab(s)

        # Drain log queue
        while True:
            try:
                line = self._log_q.get_nowait()
                self._append_text(self._log_text, line)
            except queue.Empty:
                break

        # Poll GLM response
        if self._orchestrator:
            resp = self._orchestrator.get_latest_response()
            if resp:
                md = resp.get("markdown", "")
                if md:
                    self._append_text(self._glm_text, f"🤖 Z-BOT:\n{md}\n{'─'*40}")

        # Status bar with neon state indicator
        state_map = {
            "NET":     ("🟢 NET DETECTED",    C_GREEN),
            "WALL":    ("⬜ WALL",             C_SUBTEXT),
            "HOTSPOT": ("🔴 HOTSPOT ALERT",   C_ACCENT2),
        }
        state_text, state_color = state_map.get(s.system_state, (s.system_state, C_TEXT))
        if s.error:
            state_text, state_color = f"⚠ {s.error[:60]}", C_YELLOW
        self._status_var.set(state_text)
        self._state_dot.config(fg=state_color)
        self._stats_var.set(
            f"Stitched:{s.stitched}  Tracked:{s.tracked}  "
            f"Rejected:{s.rejected}  ·  {s.mapper_status[:50]}"
        )

        self._root.after(UPDATE_MS, self._refresh_loop)

    def _update_heat_sources_tab(self, s: DashboardState) -> None:
        """Update the Heat Sources treeview and summary."""
        total = len(s.heat_sources)
        active = sum(1 for src in s.heat_sources if src.expires_at > time.time())
        self._heat_summary_var.set(
            f"Total detected: {total}  |  Active (last 8s): {active}  |  "
            f"Latest pixel positions shown below"
        )
        # Refresh treeview — show last 50
        for row in self._heat_tree.get_children():
            self._heat_tree.delete(row)
        for src in s.heat_sources[-50:]:
            tag = "active" if src.expires_at > time.time() else "old"
            self._heat_tree.insert("", "end", values=(
                src.id, src.timestamp,
                src.pixel_x, src.pixel_y,
                src.area_px,
                f"{src.confidence:.2f}",
            ), tags=(tag,))
        self._heat_tree.tag_configure("active", foreground="#ff006e")
        self._heat_tree.tag_configure("old", foreground="#6060a0")
        # Scroll to bottom
        children = self._heat_tree.get_children()
        if children:
            self._heat_tree.see(children[-1])

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop()
        self._root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z-BOT Exhibition Dashboard")
    parser.add_argument("--rgb", type=int, default=1,
                        help="RGB/webcam camera index (default: 1)")
    parser.add_argument("--thermal", type=int, default=0,
                        help="Thermal/IR camera index (-1=none, default: 0)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Detection FPS (default: 5.0)")
    parser.add_argument("--thermal-fps", type=float, default=3.0,
                        help="Thermal mapper FPS (default: 3.0)")
    parser.add_argument("--glm-interval", type=float, default=10.0,
                        help="Seconds between GLM requests (default: 10.0)")
    parser.add_argument("--net-threshold", type=float, default=0.05)
    parser.add_argument("--debris-threshold", type=float, default=0.02)
    parser.add_argument("--fire-threshold", type=float, default=0.01)
    parser.add_argument("--heat-threshold", type=int, default=210)
    parser.add_argument("--heat-min-area", type=int, default=600)
    args = parser.parse_args()

    root = tk.Tk()
    app = ZBotDashboard(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
