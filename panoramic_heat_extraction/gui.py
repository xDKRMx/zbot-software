#!/usr/bin/env python3
"""Tk GUI wrapper for live_webcam_map.py with thermal support.

100% z-bot-map LiveWebcamMapper + Tk interface + thermal canvas.
NO algorithm changes - EXACT copy of working system.
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    print("[ERROR] Install Pillow: pip install Pillow")
    sys.exit(1)

# Add z-bot-map to path for imports
ZBOT_MAP_DIR = Path(__file__).resolve().parent.parent / "z-bot-map"
sys.path.insert(0, str(ZBOT_MAP_DIR))

from floor_video_map import (
    ProcessedFrame,
    clamp_memory_alpha,
    compute_frame_geometry,
    crop_to_valid_region,
    detect_and_describe,
    estimate_pair_transform,
    estimate_rotation_deg_from_H,
    make_detector,
    make_matcher,
    normalize_angle_deg,
    parse_crop,
    preprocess_frame,
    transform_points,
    update_memory_canvas,
    validate_transform,
)


def build_tracking_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build tracking args from z-bot-map."""
    return SimpleNamespace(
        model=args.model,
        ratio=args.ratio,
        ransac=args.ransac,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        use_ecc=args.use_ecc,
        disable_flow=args.disable_flow,
        min_ecc=args.min_ecc,
        max_translation_factor=args.max_translation_factor,
        max_rotation_jump_deg=args.max_rotation_jump_deg,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        min_determinant=args.min_determinant,
        max_determinant=args.max_determinant,
    )


def make_canvas_transform(canvas_shape: tuple[int, int], frame_size: tuple[int, int]) -> np.ndarray:
    """Create canvas transform (z-bot-map)."""
    canvas_h, canvas_w = canvas_shape
    frame_w, frame_h = frame_size
    offset_x = canvas_w * 0.5 - frame_w * 0.5
    offset_y = canvas_h * 0.5 - frame_h * 0.5
    return np.array(
        [[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class ThermalWebcamMapper:
    """EXACT copy of LiveWebcamMapper with thermal support.
    
    Algorithm: 100% z-bot-map (no changes!)
    Addition: thermal_canvas for heatmap
    """
    
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.crop = parse_crop(None)
        self.tracking_args = build_tracking_args(args)
        self.detector, detector_name = make_detector(args.detector, args.nfeatures)
        self.matcher = make_matcher(detector_name)
        self.memory_alpha = clamp_memory_alpha(args.memory_alpha)
        self.reset()

    def reset(self) -> None:
        """Reset mapper state (z-bot-map EXACT)."""
        # RGB canvas (z-bot-map)
        self.canvas = np.zeros(
            (self.args.canvas_height, self.args.canvas_width, 3),
            dtype=np.uint8,
        )
        self.valid_mask = np.zeros((self.args.canvas_height, self.args.canvas_width), dtype=np.uint8)
        
        # Thermal canvas (addition)
        self.thermal_canvas = np.zeros(
            (self.args.canvas_height, self.args.canvas_width),
            dtype=np.float32,
        )
        self.thermal_mask = np.zeros((self.args.canvas_height, self.args.canvas_width), dtype=np.uint8)
        
        # z-bot-map state (EXACT)
        self.T_world_to_canvas: np.ndarray | None = None
        self.previous_frame: ProcessedFrame | None = None
        self.anchor_H: np.ndarray | None = None
        self.anchor_center: tuple[float, float] | None = None
        self.frame_size: tuple[int, int] | None = None
        self.frame_index = 0
        self.mapped_frames = 0
        self.tracked_frames = 0
        self.rejected_frames = 0
        self.lost_streak = 0
        self.status = "waiting for frames"
        self.last_source_view: np.ndarray | None = None

    def maybe_expand_canvas(self, corners_canvas: np.ndarray) -> bool:
        """Expand canvas if needed (z-bot-map EXACT)."""
        pad = self.args.canvas_pad
        min_x = float(np.min(corners_canvas[:, 0]))
        min_y = float(np.min(corners_canvas[:, 1]))
        max_x = float(np.max(corners_canvas[:, 0]))
        max_y = float(np.max(corners_canvas[:, 1]))
        canvas_h, canvas_w = self.canvas.shape[:2]

        left = max(0, int(np.ceil(pad - min_x)))
        top = max(0, int(np.ceil(pad - min_y)))
        right = max(0, int(np.ceil(max_x + pad - canvas_w)))
        bottom = max(0, int(np.ceil(max_y + pad - canvas_h)))
        if not any((left, top, right, bottom)):
            return False

        new_w = canvas_w + left + right
        new_h = canvas_h + top + bottom
        if (new_w * new_h) / 1_000_000.0 > self.args.max_canvas_mp:
            self.status = "canvas limit reached; increase --max-canvas-mp or reset"
            return False

        # Expand RGB canvas (z-bot-map)
        expanded_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        expanded_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        expanded_canvas[top : top + canvas_h, left : left + canvas_w] = self.canvas
        expanded_mask[top : top + canvas_h, left : left + canvas_w] = self.valid_mask
        self.canvas = expanded_canvas
        self.valid_mask = expanded_mask
        
        # Expand thermal canvas (addition)
        expanded_thermal = np.zeros((new_h, new_w), dtype=np.float32)
        expanded_thermal_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        expanded_thermal[top : top + canvas_h, left : left + canvas_w] = self.thermal_canvas
        expanded_thermal_mask[top : top + canvas_h, left : left + canvas_w] = self.thermal_mask
        self.thermal_canvas = expanded_thermal
        self.thermal_mask = expanded_thermal_mask
        
        assert self.T_world_to_canvas is not None
        self.T_world_to_canvas[0, 2] += left
        self.T_world_to_canvas[1, 2] += top
        return True

    def paint(self, image_bgr: np.ndarray, thermal_gray: Optional[np.ndarray], 
              H_frame_to_world: np.ndarray) -> None:
        """Paint RGB + thermal to canvas (z-bot-map update_memory_canvas)."""
        if self.T_world_to_canvas is None:
            self.T_world_to_canvas = make_canvas_transform(
                self.canvas.shape[:2],
                (image_bgr.shape[1], image_bgr.shape[0]),
            )

        local_corners = np.array(
            [
                [0.0, 0.0],
                [float(image_bgr.shape[1]), 0.0],
                [float(image_bgr.shape[1]), float(image_bgr.shape[0])],
                [0.0, float(image_bgr.shape[0])],
            ],
            dtype=np.float64,
        )
        for _ in range(2):
            H_frame_to_canvas = self.T_world_to_canvas @ H_frame_to_world
            corners_canvas = transform_points(H_frame_to_canvas, local_corners)
            if not self.maybe_expand_canvas(corners_canvas):
                break

        H_frame_to_canvas = self.T_world_to_canvas @ H_frame_to_world
        canvas_h, canvas_w = self.canvas.shape[:2]
        
        # Warp RGB (z-bot-map EXACT)
        warped = cv2.warpPerspective(
            image_bgr,
            H_frame_to_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpPerspective(
            np.full(image_bgr.shape[:2], 255, dtype=np.uint8),
            H_frame_to_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        
        # z-bot-map MEMORY CANVAS (EXACT!)
        update_memory_canvas(self.canvas, self.valid_mask, warped, mask > 0, self.memory_alpha)
        
        # Thermal canvas (same logic)
        if thermal_gray is not None:
            warped_thermal = cv2.warpPerspective(
                thermal_gray.astype(np.float32),
                H_frame_to_canvas,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask_thermal = warped_thermal > 0
            update_memory_canvas(
                self.thermal_canvas.reshape(canvas_h, canvas_w, 1),
                self.thermal_mask,
                warped_thermal.reshape(canvas_h, canvas_w, 1),
                mask_thermal,
                self.memory_alpha
            )
        
        self.mapped_frames += 1

    def paint_pose_for_tracking(self, color: np.ndarray, thermal: Optional[np.ndarray],
                                H_frame_to_world: np.ndarray) -> str:
        """Paint with anchor locking (z-bot-map EXACT)."""
        H_to_paint = H_frame_to_world
        paint_mode = "raw"
        if self.args.lock_small_motion_updates and self.anchor_H is not None and self.anchor_center is not None:
            current_center, _ = compute_frame_geometry(H_frame_to_world, color.shape[1], color.shape[0])
            step = float(np.linalg.norm(np.asarray(current_center) - np.asarray(self.anchor_center)))
            rotation_delta = abs(
                normalize_angle_deg(
                    estimate_rotation_deg_from_H(H_frame_to_world)
                    - estimate_rotation_deg_from_H(self.anchor_H)
                )
            )
            if step < self.args.anchor_step_px and rotation_delta < self.args.anchor_rotation_deg:
                H_to_paint = self.anchor_H
                paint_mode = "locked"
            else:
                self.anchor_H = H_frame_to_world.copy()
                self.anchor_center = current_center
                paint_mode = "anchored"
        else:
            self.anchor_H = H_frame_to_world.copy()
            self.anchor_center, _ = compute_frame_geometry(H_frame_to_world, color.shape[1], color.shape[0])
        self.paint(color, thermal, H_to_paint)
        return paint_mode

    def process(self, frame_bgr: np.ndarray, thermal_gray: Optional[np.ndarray],
                timestamp_sec: float) -> np.ndarray:
        """Process frame (z-bot-map EXACT algorithm!)."""
        color, gray = preprocess_frame(frame_bgr, self.crop, self.args.max_dim)
        self.last_source_view = color
        self.frame_size = (color.shape[1], color.shape[0])
        keypoints, descriptors = detect_and_describe(gray, self.detector)

        if self.previous_frame is None:
            H_identity = np.eye(3, dtype=np.float64)
            self.previous_frame = ProcessedFrame(
                frame_index=self.frame_index,
                timestamp_sec=timestamp_sec,
                color=color,
                gray=gray,
                keypoints=keypoints,
                descriptors=descriptors,
                H_frame_to_world=H_identity,
                tracking_mode="bootstrap",
            )
            self.anchor_H = H_identity.copy()
            self.anchor_center, _ = compute_frame_geometry(H_identity, color.shape[1], color.shape[0])
            self.paint(color, thermal_gray, H_identity)
            self.status = f"initialized  keypoints={len(keypoints)}"
            return color

        estimate = estimate_pair_transform(
            self.previous_frame,
            gray,
            keypoints,
            descriptors,
            self.matcher,
            self.tracking_args,
        )
        verdict = (
            (False, estimate.reason)
            if estimate.H_current_to_reference is None
            else validate_transform(
                estimate.H_current_to_reference,
                estimate,
                color.shape,
                self.tracking_args,
            )
        )

        if verdict[0]:
            assert estimate.H_current_to_reference is not None
            H_frame_to_world = self.previous_frame.H_frame_to_world @ estimate.H_current_to_reference
            paint_mode = self.paint_pose_for_tracking(color, thermal_gray, H_frame_to_world)
            self.previous_frame = ProcessedFrame(
                frame_index=self.frame_index,
                timestamp_sec=timestamp_sec,
                color=color,
                gray=gray,
                keypoints=keypoints,
                descriptors=descriptors,
                H_frame_to_world=H_frame_to_world,
                tracking_mode="visual",
            )
            self.tracked_frames += 1
            self.lost_streak = 0
            source = "ECC" if estimate.used_ecc else "flow" if estimate.used_flow else "features"
            # FIX: use num_inliers, num_matches (not inlier_count, match_count!)
            self.status = (
                f"{source} ok  inliers={estimate.num_inliers}/{estimate.num_matches}  "
                f"ratio={estimate.inlier_ratio:.2f}  paint={paint_mode}"
            )
            return color

        self.rejected_frames += 1
        self.lost_streak += 1
        self.status = f"tracking rejected: {verdict[1]}"
        if self.lost_streak >= self.args.reacquire_after:
            H_reference = self.anchor_H.copy() if self.anchor_H is not None else np.eye(3, dtype=np.float64)
            self.previous_frame = ProcessedFrame(
                frame_index=self.frame_index,
                timestamp_sec=timestamp_sec,
                color=color,
                gray=gray,
                keypoints=keypoints,
                descriptors=descriptors,
                H_frame_to_world=H_reference,
                tracking_mode="reacquire",
            )
            self.lost_streak = 0
            self.status = "reacquiring from current frame"
        return color
    
    def get_thermal_heatmap(self) -> np.ndarray:
        """Get thermal heatmap with JET colormap."""
        if not np.any(self.thermal_mask):
            return np.zeros((self.args.canvas_height, self.args.canvas_width, 3), dtype=np.uint8)
        
        valid = self.thermal_canvas >= 1.0
        thermal_minv, thermal_maxv = 40, 160
        denom = max(thermal_maxv - thermal_minv, 1)
        stretched = ((np.clip(self.thermal_canvas, thermal_minv, thermal_maxv) - thermal_minv)
                     * (255.0 / denom)).astype(np.uint8)
        stretched[~valid] = 0
        colored = cv2.applyColorMap(stretched, cv2.COLORMAP_JET)
        colored[~valid, :] = 0  # FIX: proper 3D array indexing
        
        # Crop to valid region
        if np.any(self.thermal_mask):
            try:
                colored_cropped, _, _ = crop_to_valid_region(colored, self.thermal_mask)
                return colored_cropped
            except:
                return colored
        return colored


# GUI Constants
PANEL_W = 480
PANEL_H = 360
UPDATE_MS = 100


class ThermalPanoramaGUI:
    """Tk GUI wrapper for z-bot-map ThermalWebcamMapper."""
    
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        
        self.root.title("Z-BOT Thermal Panorama (100% z-bot-map)")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)
        
        self.running = False
        self.cap_rgb: Optional[cv2.VideoCapture] = None
        self.cap_thermal: Optional[cv2.VideoCapture] = None
        self.mapper: Optional[ThermalWebcamMapper] = None
        self.start_time = 0.0
        
        self.video_file: Optional[str] = None
        self.is_video_mode = False
        
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(UPDATE_MS, self._refresh_loop)
    
    def _build_ui(self):
        """Build UI."""
        # Top bar
        top = tk.Frame(self.root, bg="#313244", pady=5)
        top.pack(fill="x")
        
        tk.Label(top, text="Z-BOT Thermal Panorama (100% z-bot-map algorithm)", 
                 font=("Helvetica", 11, "bold"),
                 bg="#313244", fg="#cdd6f4").pack(side="left", padx=10)
        
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
        panels.pack(pady=8)
        
        cam_frame = tk.Frame(panels, bg="#313244")
        cam_frame.pack(side="left", padx=8)
        tk.Label(cam_frame, text="Live Camera", bg="#313244", fg="#a6adc8",
                font=("Helvetica", 9)).pack()
        self.cam_label = tk.Label(cam_frame, bg="#11111b", width=PANEL_W, height=PANEL_H)
        self.cam_label.pack()
        
        map_frame = tk.Frame(panels, bg="#313244")
        map_frame.pack(side="left", padx=8)
        tk.Label(map_frame, text="Thermal Heat Map (z-bot-map memory canvas)",
                bg="#313244", fg="#a6adc8", font=("Helvetica", 9)).pack()
        self.map_label = tk.Label(map_frame, bg="#11111b", width=PANEL_W, height=PANEL_H)
        self.map_label.pack()
        
        # Status
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
        if self.running:
            self._stop()
        else:
            self._start()
    
    def _start(self):
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
        self.mapper = ThermalWebcamMapper(self.args)
        self.running = True
        self.start_time = time.monotonic()
        
        self.btn_start.config(text="⏹ Stop", bg="#f38ba8")
        self.btn_export.config(state="disabled")
        self.btn_reset.config(state="disabled")
        self.status_var.set("Running - z-bot-map live memory canvas active")
    
    def _stop(self):
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
        if self.running:
            return
        
        if self.mapper:
            self.mapper.reset()
        
        self.video_file = None
        self.is_video_mode = False
        self.btn_import.config(bg="#fab387")
        
        self.status_var.set("Reset complete - press Start")
        self.stats_var.set("")
    
    def _import_video(self):
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
        """Main loop - z-bot-map style."""
        if self.running and self.cap_rgb:
            ret, frame_rgb = self.cap_rgb.read()
            if not ret:
                if self.is_video_mode:
                    self._stop()
                    self.status_var.set("Video complete - press Export to save")
                self.root.after(UPDATE_MS, self._refresh_loop)
                return
            
            # Read thermal
            frame_thermal = None
            if self.cap_thermal:
                ret_t, frame_thermal = self.cap_thermal.read()
                if not ret_t:
                    frame_thermal = None
            
            # Prepare thermal data
            use_thermal = self.source_var.get() == "thermal"
            thermal_gray = None
            if use_thermal and frame_thermal is not None:
                thermal_gray = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
            elif not use_thermal:
                thermal_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            
            # z-bot-map process (EXACT!)
            timestamp_sec = time.monotonic() - self.start_time
            source_view = self.mapper.process(frame_rgb, thermal_gray, timestamp_sec)
            
            # Update displays
            self._put_image(self.cam_label, source_view)
            
            heatmap = self.mapper.get_thermal_heatmap()
            if heatmap.size > 0:
                self._put_image(self.map_label, heatmap)
            
            # Update stats (z-bot-map style)
            self.stats_var.set(
                f"Stitched: {self.mapper.mapped_frames}  |  "
                f"Tracked: {self.mapper.tracked_frames}  |  "
                f"Rejected: {self.mapper.rejected_frames}"
            )
            self.status_var.set(f"z-bot-map: {self.mapper.status}")
            
            self.mapper.frame_index += 1
        
        self.root.after(UPDATE_MS, self._refresh_loop)
    
    def _on_close(self):
        if self.running:
            self._stop()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Z-BOT Thermal Panorama (100% z-bot-map)")
    parser.add_argument("--rgb", type=int, default=0, help="RGB camera index")
    parser.add_argument("--thermal", type=int, default=-1, help="Thermal camera index (-1=none)")
    parser.add_argument("--source", choices=["rgb", "thermal"], default="rgb", help="Heat source")
    
    # z-bot-map parameters
    parser.add_argument("--max-dim", type=int, default=640)
    parser.add_argument("--canvas-width", type=int, default=2200)
    parser.add_argument("--canvas-height", type=int, default=1600)
    parser.add_argument("--canvas-pad", type=int, default=120)
    parser.add_argument("--max-canvas-mp", type=float, default=64.0)
    parser.add_argument("--detector", choices=("sift", "orb"), default="orb")
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--model", choices=("translation", "partial-affine", "affine", "homography"),
                       default="partial-affine")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac", type=float, default=3.0)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.2)
    parser.add_argument("--memory-alpha", type=float, default=0.45)
    parser.add_argument("--anchor-step-px", type=float, default=8.0)
    parser.add_argument("--anchor-rotation-deg", type=float, default=1.0)
    parser.add_argument("--lock-small-motion-updates", action="store_true", default=True)
    parser.add_argument("--use-ecc", action="store_true")
    parser.add_argument("--disable-flow", action="store_true")
    parser.add_argument("--min-ecc", type=float, default=0.85)
    parser.add_argument("--max-translation-factor", type=float, default=1.5)
    parser.add_argument("--max-rotation-jump-deg", type=float, default=75.0)
    parser.add_argument("--min-scale", type=float, default=0.6)
    parser.add_argument("--max-scale", type=float, default=1.5)
    parser.add_argument("--min-determinant", type=float, default=0.2)
    parser.add_argument("--max-determinant", type=float, default=5.0)
    parser.add_argument("--reacquire-after", type=int, default=12)
    
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ThermalPanoramaGUI(root, args)
    
    if args.source == "thermal":
        app.source_var.set("thermal")
    
    root.mainloop()


if __name__ == "__main__":
    main()
