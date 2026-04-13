"""Standalone 2D Thermal Panorama Stitcher — no net_inspector dependency.

Drop this folder onto the RPi and run gui.py directly.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def timestamp_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def save_image(path: Path, img: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PanoramaConfig:
    """All tunable parameters in one place."""
    # Camera
    rgb_camera_idx: int = 0       # webcam index
    thermal_camera_idx: int = -1  # -1 = use RGB as thermal source
    use_thermal_as_source: bool = False  # if True, thermal cam drives heat map
    width: int = 640
    height: int = 480
    capture_fps: float = 5.0      # how many frames/sec to stitch

    # Stitching
    akaze_threshold: float = 0.001
    ransac_threshold: float = 5.0
    min_inliers: int = 10
    canvas_padding_factor: int = 5
    blend_width: int = 32
    bundle_adjust_interval: int = 20

    # Thermal colorization (matches run_thermal.py)
    thermal_minv: int = 40
    thermal_maxv: int = 160

    # RGB-to-thermal mounting offset (pixels)
    rgb_thermal_dx: int = 0
    rgb_thermal_dy: int = 0

    # Hybrid stitching — override + edge-catching
    overlap_margin_px: int = 20        # extra pixels beyond edge mask for vibration
    rotation_threshold_deg: float = 5.0  # above this → fallback to unvisited-only
    max_move_px: float = 400.0         # above this → reject frame (bad match/shake)
    min_move_px: float = 5.0           # below this → skip (robot not moving enough)

    # Output
    output_dir: Path = OUTPUT_DIR


# ---------------------------------------------------------------------------
# FramePair
# ---------------------------------------------------------------------------

@dataclass
class FramePair:
    seq_id: int
    timestamp: float
    rgb: np.ndarray
    thermal_gray: np.ndarray
    H_to_canvas: Optional[np.ndarray] = None
    inlier_count: int = 0
    low_confidence: bool = False


# ---------------------------------------------------------------------------
# FeatureAligner
# ---------------------------------------------------------------------------

class FeatureAligner:
    """Translation-only feature aligner.

    Extracts only (tx, ty) from AKAZE matches, discards rotation and scale.
    This prevents drift accumulation — correct for a robot that only
    translates on a flat wall surface.
    """

    def __init__(self, akaze_threshold: float = 0.001,
                 ransac_threshold: float = 5.0, min_inliers: int = 6) -> None:
        self._akaze = cv2.AKAZE_create(threshold=akaze_threshold)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ransac_threshold = ransac_threshold
        self._min_inliers = min_inliers

    def compute_homography(self, rgb_prev: np.ndarray,
                           rgb_curr: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        """Returns a 3×3 pure-translation matrix or None."""
        g1 = cv2.cvtColor(rgb_prev, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(rgb_curr, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self._akaze.detectAndCompute(g1, None)
        kp2, des2 = self._akaze.detectAndCompute(g2, None)

        if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
            return None, 0

        raw = self._matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]

        if len(good) < 3:
            return None, len(good)

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_threshold,
        )

        if M is None or mask is None:
            return None, 0

        inliers = int(mask.sum())
        if inliers < self._min_inliers:
            return None, inliers

        # Extract only translation — discard rotation and scale to prevent drift
        tx = float(M[0, 2])
        ty = float(M[1, 2])

        # Pure translation matrix
        H = np.array([[1.0, 0.0, tx],
                      [0.0, 1.0, ty],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return H, inliers


# ---------------------------------------------------------------------------
# CanvasManager
# ---------------------------------------------------------------------------

class CanvasManager:
    """Growing float32 thermal canvas with override-based pixel memory.

    Hybrid approach:
    - Override semantics: last write wins per pixel (no blending accumulation)
    - Edge-catching mask: only write pixels from the newly-revealed region
      based on translation vector (tx, ty) from AKAZE
    - Vibration robustness: overlap_margin_px + rotation fallback
    """

    def __init__(self, frame_h: int, frame_w: int, padding_factor: int = 3,
                 overlap_margin_px: int = 20,
                 rotation_threshold_deg: float = 5.0) -> None:
        ch, cw = frame_h * padding_factor, frame_w * padding_factor
        self._canvas = np.zeros((ch, cw), dtype=np.float32)
        self._visited = np.zeros((ch, cw), dtype=bool)   # True = written at least once
        self._ox = frame_w * (padding_factor // 2)
        self._oy = frame_h * (padding_factor // 2)
        self._H_offset = np.array([[1, 0, self._ox],
                                   [0, 1, self._oy],
                                   [0, 0, 1]], dtype=np.float64)
        self._margin = overlap_margin_px
        self._rot_threshold = rotation_threshold_deg
        self._frame_h = frame_h
        self._frame_w = frame_w

    @property
    def shape(self) -> tuple[int, int]:
        return self._canvas.shape

    @property
    def H_offset(self) -> np.ndarray:
        return self._H_offset.copy()

    def place_first(self, gray: np.ndarray) -> None:
        """Place the very first frame at the canvas centre."""
        h, w = gray.shape[:2]
        self._canvas[self._oy:self._oy+h, self._ox:self._ox+w] = gray.astype(np.float32)
        self._visited[self._oy:self._oy+h, self._ox:self._ox+w] = True

    def _compute_edge_mask(self, tx: float, ty: float) -> np.ndarray:
        """Build frame-space bool mask of newly revealed pixels.

        Based on movement direction:
          tx > 0 → robot moved right → left edge of frame is new
          tx < 0 → robot moved left  → right edge of frame is new
          ty > 0 → robot moved down  → top edge of frame is new
          ty < 0 → robot moved up    → bottom edge of frame is new
        """
        h, w = self._frame_h, self._frame_w
        m = self._margin
        mask = np.zeros((h, w), dtype=bool)

        if tx > 0:    # moved right → left strip is new
            mask[:, :min(int(abs(tx)) + m, w)] = True
        elif tx < 0:  # moved left → right strip is new
            mask[:, max(0, w - int(abs(tx)) - m):] = True

        if ty > 0:    # moved down → top strip is new
            mask[:min(int(abs(ty)) + m, h), :] = True
        elif ty < 0:  # moved up → bottom strip is new
            mask[max(0, h - int(abs(ty)) - m):, :] = True

        # Fallback: if mask covers < 1% of frame, write everything
        # (handles diagonal or very small movements)
        if mask.sum() < (h * w * 0.01):
            mask[:] = True

        return mask

    def warp_and_blend(self, gray: np.ndarray, H_acc: np.ndarray,
                       tx: float = 0.0, ty: float = 0.0,
                       angle_deg: float = 0.0) -> bool:
        """Warp thermal frame onto canvas.

        - Unvisited pixels: write directly (first visit)
        - Already visited pixels: soft blend (0.3 old + 0.7 new) for smooth seams
        """
        ch, cw = self._canvas.shape
        H_c = self._H_offset @ H_acc

        warped = cv2.warpPerspective(
            gray.astype(np.float32), H_c, (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        new_data = warped > 0

        if not new_data.any():
            return False

        # First visit: write directly
        first_visit = new_data & ~self._visited
        self._canvas[first_visit] = warped[first_visit]
        self._visited[first_visit] = True

        # Revisit: soft blend to smooth seams (70% new, 30% old)
        revisit = new_data & self._visited & ~first_visit
        if revisit.any():
            self._canvas[revisit] = (0.7 * warped[revisit] +
                                     0.3 * self._canvas[revisit])

        return True

    def expand_if_needed(self, margin: int = 50) -> None:
        """Expand canvas if content is within margin pixels of any edge."""
        rows = np.any(self._visited, axis=1)
        cols = np.any(self._visited, axis=0)
        if not rows.any():
            return
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        ch, cw = self._canvas.shape
        if not (rmin < margin or rmax > ch - margin or
                cmin < margin or cmax > cw - margin):
            return
        py, px = max(ch // 2, 200), max(cw // 2, 200)
        self._canvas = np.pad(self._canvas, ((py, py), (px, px)), constant_values=0)
        self._visited = np.pad(self._visited, ((py, py), (px, px)), constant_values=False)
        self._ox += px
        self._oy += py
        self._H_offset[0, 2] = self._ox
        self._H_offset[1, 2] = self._oy

    def get_cropped(self) -> np.ndarray:
        """Return canvas cropped to the visited bounding box."""
        if not self._visited.any():
            return self._canvas.copy()
        rows, cols = np.any(self._visited, axis=1), np.any(self._visited, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        return self._canvas[r0:r1+1, c0:c1+1].copy()


# ---------------------------------------------------------------------------
# HeatMapRenderer
# ---------------------------------------------------------------------------

class HeatMapRenderer:
    def __init__(self, minv: int = 40, maxv: int = 160) -> None:
        self._minv, self._maxv = minv, maxv

    def render(self, canvas_f32: np.ndarray) -> np.ndarray:
        valid = canvas_f32 >= 1.0
        denom = max(self._maxv - self._minv, 1)
        stretched = ((np.clip(canvas_f32, self._minv, self._maxv) - self._minv)
                     * (255.0 / denom)).astype(np.uint8)
        # Smooth to reduce stitching seams at frame boundaries
        stretched = cv2.GaussianBlur(stretched, (7, 7), 2)
        stretched[~valid] = 0
        colored = cv2.applyColorMap(stretched, cv2.COLORMAP_JET)
        colored[~valid] = [0, 0, 0]
        return colored

    def add_scale_bar(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        grad = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
        bar = cv2.applyColorMap(np.repeat(grad, 30, axis=1), cv2.COLORMAP_JET)
        labels = np.full((h, 60, 3), 255, dtype=np.uint8)
        cv2.putText(labels, str(self._maxv), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(labels, str(self._minv), (4, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return np.hstack([img, bar, labels])


# ---------------------------------------------------------------------------
# PanoramaStitcher
# ---------------------------------------------------------------------------

class PanoramaStitcher:
    """Thread-safe panorama stitcher. Feed frames, get preview, export."""

    def __init__(self, config: PanoramaConfig,
                 on_update: Optional[Callable[[np.ndarray], None]] = None) -> None:
        self._cfg = config
        self._aligner = FeatureAligner(config.akaze_threshold,
                                       config.ransac_threshold, config.min_inliers)
        self._renderer = HeatMapRenderer(config.thermal_minv, config.thermal_maxv)
        self._canvas_mgr: Optional[CanvasManager] = None
        self._on_update = on_update  # called with colorized preview after each stitch

        self._q: queue.Queue[FramePair] = queue.Queue(maxsize=500)
        self._seq = 0
        self._prev: Optional[FramePair] = None
        self._H_acc = np.eye(3, dtype=np.float64)
        self._stitched: list[FramePair] = []

        self.frames_stitched = 0
        self.frames_skipped = 0
        self._low_conf = 0
        self._total_inliers = 0
        self.drift_corrections = 0
        self._start_time = 0.0

        self._alive = False
        self._lock = threading.Lock()

    def start(self) -> None:
        self._alive = True
        self._start_time = time.time()
        threading.Thread(target=self._worker, daemon=True).start()

    def stop(self) -> None:
        self._alive = False

    def feed_frame(self, rgb: np.ndarray, thermal_gray: np.ndarray,
                   ts: Optional[float] = None) -> None:
        fp = FramePair(self._seq, ts or time.time(), rgb, thermal_gray)
        self._seq += 1
        try:
            self._q.put_nowait(fp)
        except queue.Full:
            pass  # drop silently in GUI mode

    def get_preview(self, max_w: int = 900) -> Optional[np.ndarray]:
        with self._lock:
            if self._canvas_mgr is None:
                return None
            cropped = self._canvas_mgr.get_cropped()
        colored = self._renderer.render(cropped)
        h, w = colored.shape[:2]
        if w > max_w:
            colored = cv2.resize(colored, (max_w, int(h * max_w / w)), cv2.INTER_AREA)
        return colored

    def export(self) -> Optional[Path]:
        with self._lock:
            if self._canvas_mgr is None:
                return None
            cropped = self._canvas_mgr.get_cropped()
        colored = self._renderer.render(cropped)
        with_bar = self._renderer.add_scale_bar(colored)

        pano_dir = self._cfg.output_dir / "panorama"
        ensure_dir(pano_dir)
        stamp = timestamp_id()
        out = pano_dir / f"heatmap_{stamp}.png"
        save_image(out, with_bar)

        # Preview
        h, w = with_bar.shape[:2]
        if w > 2048:
            prev = cv2.resize(with_bar, (2048, int(h * 2048 / w)), cv2.INTER_AREA)
        else:
            prev = with_bar
        save_image(pano_dir / f"heatmap_{stamp}_preview.jpg", prev)

        # Metadata
        dur = time.time() - self._start_time
        meta = {
            "total_frames": self._seq,
            "frames_stitched": self.frames_stitched,
            "frames_skipped": self.frames_skipped,
            "canvas_width": int(cropped.shape[1]),
            "canvas_height": int(cropped.shape[0]),
            "duration_s": round(dur, 2),
            "low_confidence_count": self._low_conf,
            "avg_inliers": round(self._total_inliers / max(self.frames_stitched, 1), 1),
            "drift_corrections": self.drift_corrections,
        }
        (pano_dir / f"heatmap_{stamp}_metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[PANORAMA] Saved → {out}")
        return out

    # -- worker --------------------------------------------------------------

    def _worker(self) -> None:
        try:
            while self._alive:
                try:
                    fp = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                with self._lock:
                    self._process(fp)
        except Exception as e:
            print(f"[PANORAMA] Worker error: {e}")
            self._alive = False

    def _process(self, fp: FramePair) -> None:
        import math
        thermal = fp.thermal_gray
        dx, dy = self._cfg.rgb_thermal_dx, self._cfg.rgb_thermal_dy
        if dx or dy:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            thermal = cv2.warpAffine(thermal, M, (thermal.shape[1], thermal.shape[0]))

        if self._canvas_mgr is None:
            h, w = thermal.shape[:2]
            self._canvas_mgr = CanvasManager(
                h, w,
                padding_factor=self._cfg.canvas_padding_factor,
                overlap_margin_px=self._cfg.overlap_margin_px,
                rotation_threshold_deg=self._cfg.rotation_threshold_deg,
            )
            self._canvas_mgr.place_first(thermal)
            fp.H_to_canvas = np.eye(3, dtype=np.float64)
            self._prev = fp
            self._stitched.append(fp)
            self.frames_stitched = 1
            self._notify_update()
            return

        H, inliers = self._aligner.compute_homography(self._prev.rgb, fp.rgb)

        if H is None and len(self._stitched) >= 2:
            alt = self._stitched[-2]
            H, inliers = self._aligner.compute_homography(alt.rgb, fp.rgb)
            if H is not None and alt.H_to_canvas is not None:
                # Recompute H_acc from alt's known canvas position
                alt_tx = float(H[0, 2])
                alt_ty = float(H[1, 2])
                self._H_acc = alt.H_to_canvas.copy()
                self._H_acc[0, 2] -= alt_tx
                self._H_acc[1, 2] -= alt_ty
                fp.low_confidence = True
                self._low_conf += 1

        if H is None:
            self.frames_skipped += 1
            return

        # Extract movement info from H (frame-space)
        # H maps prev→curr: positive tx means scene moved left (camera moved right)
        tx = float(H[0, 2])
        ty = float(H[1, 2])
        move = math.sqrt(tx**2 + ty**2)

        # Reject bad matches
        if move > self._cfg.max_move_px:
            self.frames_skipped += 1
            print(f"[PANORAMA] Frame {fp.seq_id} rejected: move={move:.1f}px > max={self._cfg.max_move_px}")
            return

        # Silent skip if robot hasn't moved enough
        if move < self._cfg.min_move_px:
            return

        if not fp.low_confidence:
            # H maps prev→curr: tx = how much features moved in curr relative to prev
            # Positive tx = features moved right = camera moved left
            # To place curr on canvas: shift canvas position by +tx (opposite direction)
            self._H_acc[0, 2] += tx
            self._H_acc[1, 2] += ty

        fp.H_to_canvas = self._H_acc.copy()
        fp.inlier_count = inliers
        self._total_inliers += inliers

        self._canvas_mgr.expand_if_needed()
        ok = self._canvas_mgr.warp_and_blend(thermal, self._H_acc)
        if ok:
            self.frames_stitched += 1
            self._stitched.append(fp)
            self._prev = fp
            self._notify_update()
            if (self.frames_stitched % self._cfg.bundle_adjust_interval == 0
                    and self.frames_stitched > self._cfg.bundle_adjust_interval):
                self._drift_correct()
        else:
            self.frames_skipped += 1

    def _drift_correct(self) -> None:
        if len(self._stitched) < 10:
            return
        curr = self._stitched[-1]
        for off in [5, 10, 15]:
            idx = len(self._stitched) - 1 - off
            if idx < 0:
                continue
            old = self._stitched[idx]
            H_c, inliers = self._aligner.compute_homography(old.rgb, curr.rgb)
            if H_c is not None and inliers > 15 and old.H_to_canvas is not None and curr.H_to_canvas is not None:
                H_exp = curr.H_to_canvas @ np.linalg.inv(old.H_to_canvas)
                H_meas = np.linalg.inv(H_c)
                corr = (H_meas + H_exp) / 2.0
                corr[2, :] = [0, 0, 1]
                self._H_acc = old.H_to_canvas @ corr
                self.drift_corrections += 1
                break

    def _notify_update(self) -> None:
        if self._on_update is None or self._canvas_mgr is None:
            return
        try:
            cropped = self._canvas_mgr.get_cropped()
            colored = self._renderer.render(cropped)
            self._on_update(colored)
        except Exception:
            pass
