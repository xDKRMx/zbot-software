"""2D Thermal Panorama Stitcher for Z-BOT.

Accumulates overlapping IR camera frames as the robot traverses a building
surface and stitches them into a single 2D thermal heat map. Uses RGB camera
features (AKAZE) for geometric alignment and applies transforms to thermal
frames. Runs on a separate daemon thread to avoid blocking detection pipeline.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from net_inspector.config import PanoramaConfig, OUTPUT_DIR
from net_inspector.utils.io import ensure_dir, save_image, timestamp_id


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FramePair:
    """One RGB + thermal frame pair with alignment metadata."""

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
    """AKAZE-based feature extraction, matching, and homography estimation."""

    def __init__(
        self,
        akaze_threshold: float = 0.001,
        ransac_threshold: float = 5.0,
        min_inliers: int = 10,
    ) -> None:
        self._akaze = cv2.AKAZE_create(threshold=akaze_threshold)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ransac_threshold = ransac_threshold
        self._min_inliers = min_inliers

    def compute_homography(
        self, rgb_prev: np.ndarray, rgb_curr: np.ndarray,
    ) -> tuple[Optional[np.ndarray], int]:
        """Compute homography from *rgb_prev* to *rgb_curr*.

        Returns (H, inlier_count).  H is None when matching fails.
        """
        gray_prev = cv2.cvtColor(rgb_prev, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(rgb_curr, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self._akaze.detectAndCompute(gray_prev, None)
        kp2, des2 = self._akaze.detectAndCompute(gray_curr, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, 0

        raw_matches = self._matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 4:
            return None, len(good)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self._ransac_threshold)

        if H is None or mask is None:
            return None, 0

        inlier_count = int(mask.sum())
        if inlier_count < self._min_inliers:
            return None, inlier_count

        return H, inlier_count


# ---------------------------------------------------------------------------
# CanvasManager
# ---------------------------------------------------------------------------

class CanvasManager:
    """Growing float32 thermal canvas — hybrid override + edge-catching.

    Override semantics: last write wins per pixel (no blending accumulation).
    Edge-catching mask: only write pixels from the newly-revealed region
    based on translation vector (tx, ty) from AKAZE.
    """

    def __init__(
        self,
        frame_h: int,
        frame_w: int,
        padding_factor: int = 3,
        overlap_margin_px: int = 20,
        rotation_threshold_deg: float = 5.0,
    ) -> None:
        ch = frame_h * padding_factor
        cw = frame_w * padding_factor
        self._canvas = np.zeros((ch, cw), dtype=np.float32)
        self._visited = np.zeros((ch, cw), dtype=bool)
        self._offset_x = frame_w * (padding_factor // 2)
        self._offset_y = frame_h * (padding_factor // 2)
        self._H_offset = np.array(
            [[1, 0, self._offset_x],
             [0, 1, self._offset_y],
             [0, 0, 1]], dtype=np.float64,
        )
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

    def place_first(self, thermal_gray: np.ndarray) -> None:
        """Place the very first frame at the canvas centre."""
        h, w = thermal_gray.shape[:2]
        y0 = self._offset_y
        x0 = self._offset_x
        self._canvas[y0:y0 + h, x0:x0 + w] = thermal_gray.astype(np.float32)
        self._visited[y0:y0 + h, x0:x0 + w] = True

    def _compute_edge_mask(self, tx: float, ty: float) -> np.ndarray:
        """Build frame-space bool mask of newly revealed pixels."""
        h, w = self._frame_h, self._frame_w
        m = self._margin
        mask = np.zeros((h, w), dtype=bool)

        if tx > 0:
            mask[:, :min(int(abs(tx)) + m, w)] = True
        elif tx < 0:
            mask[:, max(0, w - int(abs(tx)) - m):] = True

        if ty > 0:
            mask[:min(int(abs(ty)) + m, h), :] = True
        elif ty < 0:
            mask[max(0, h - int(abs(ty)) - m):, :] = True

        if mask.sum() < (h * w * 0.01):
            mask[:] = True
        return mask

    def warp_and_blend(self, thermal_gray: np.ndarray, H_accumulated: np.ndarray,
                       tx: float = 0.0, ty: float = 0.0,
                       angle_deg: float = 0.0) -> bool:
        """Warp thermal frame onto canvas — pure override semantics."""
        ch, cw = self._canvas.shape
        H_canvas = self._H_offset @ H_accumulated

        warped = cv2.warpPerspective(
            thermal_gray.astype(np.float32), H_canvas, (cw, ch),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        write_mask = warped > 0

        if not write_mask.any():
            return False

        self._canvas[write_mask] = warped[write_mask]
        self._visited[write_mask] = True
        return True

    def expand_if_needed(self, margin: int = 50) -> bool:
        """Expand canvas if content is within *margin* pixels of any edge."""
        rows = np.any(self._visited, axis=1)
        cols = np.any(self._visited, axis=0)
        if not rows.any():
            return False

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        ch, cw = self._canvas.shape

        need_expand = (rmin < margin or rmax > ch - margin or
                       cmin < margin or cmax > cw - margin)
        if not need_expand:
            return False

        pad_y = max(ch // 2, 200)
        pad_x = max(cw // 2, 200)

        self._canvas = np.pad(self._canvas, ((pad_y, pad_y), (pad_x, pad_x)),
                              mode='constant', constant_values=0)
        self._visited = np.pad(self._visited, ((pad_y, pad_y), (pad_x, pad_x)),
                               mode='constant', constant_values=False)

        self._offset_x += pad_x
        self._offset_y += pad_y
        self._H_offset[0, 2] = self._offset_x
        self._H_offset[1, 2] = self._offset_y
        return True

    def get_cropped(self) -> np.ndarray:
        """Return the canvas cropped to the visited bounding box."""
        if not self._visited.any():
            return self._canvas.copy()

        rows = np.any(self._visited, axis=1)
        cols = np.any(self._visited, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return self._canvas[rmin:rmax + 1, cmin:cmax + 1].copy()


# ---------------------------------------------------------------------------
# HeatMapRenderer
# ---------------------------------------------------------------------------

class HeatMapRenderer:
    """Colorize a float32 thermal canvas using the same MINV/MAXV + JET
    pipeline as ``run_thermal.py``."""

    def __init__(self, minv: int = 40, maxv: int = 160) -> None:
        self._minv = minv
        self._maxv = maxv

    def render(self, canvas_f32: np.ndarray) -> np.ndarray:
        """Return a BGR colorized heat map from a float32 thermal canvas."""
        valid_mask = canvas_f32 >= 1.0  # pixels that have data

        clipped = np.clip(canvas_f32, self._minv, self._maxv)
        denom = max(self._maxv - self._minv, 1)
        stretched = ((clipped - self._minv) * (255.0 / denom)).astype(np.uint8)

        colored = cv2.applyColorMap(stretched, cv2.COLORMAP_JET)
        colored[~valid_mask] = [0, 0, 0]
        return colored

    def add_scale_bar(self, img_bgr: np.ndarray) -> np.ndarray:
        """Append a vertical colour-scale bar on the right side."""
        h, w = img_bgr.shape[:2]
        bar_w = 30
        label_w = 60
        total_extra = bar_w + label_w

        # Build gradient strip
        gradient = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
        gradient_bgr = cv2.applyColorMap(
            np.repeat(gradient, bar_w, axis=1), cv2.COLORMAP_JET,
        )

        # Label area (white background)
        label_area = np.full((h, label_w, 3), 255, dtype=np.uint8)
        cv2.putText(label_area, str(self._maxv), (4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(label_area, str(self._minv), (4, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        bar_block = np.hstack([gradient_bgr, label_area])
        return np.hstack([img_bgr, bar_block])


# ---------------------------------------------------------------------------
# PanoramaStitcher  (main entry point)
# ---------------------------------------------------------------------------

class PanoramaStitcher:
    """Orchestrates frame accumulation, stitching, and export.

    Runs a daemon worker thread so the detection pipeline is never blocked.
    """

    def __init__(
        self,
        config: PanoramaConfig,
        on_event: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._cfg = config
        self._aligner = FeatureAligner(
            akaze_threshold=config.akaze_threshold,
            ransac_threshold=config.ransac_threshold,
            min_inliers=config.min_inliers,
        )
        self._renderer = HeatMapRenderer(config.thermal_minv, config.thermal_maxv)
        self._canvas_mgr: Optional[CanvasManager] = None
        self._on_event = on_event

        self._frame_queue: queue.Queue[FramePair] = queue.Queue(maxsize=config.max_frames)
        self._seq = 0
        self._prev: Optional[FramePair] = None
        self._H_acc = np.eye(3, dtype=np.float64)
        self._stitched: list[FramePair] = []

        # Stats
        self._frames_stitched = 0
        self._frames_skipped = 0
        self._low_conf_count = 0
        self._total_inliers = 0
        self._drift_corrections = 0
        self._start_time = 0.0

        self._alive = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._stitch_worker, daemon=True)
        self._thread.start()
        print("[PANORAMA] Stitcher thread started.")

    def stop(self) -> None:
        self._alive = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        print("[PANORAMA] Stitcher stopped.")

    def feed_frame(
        self, rgb: np.ndarray, thermal_gray: np.ndarray, timestamp: float,
    ) -> None:
        fp = FramePair(
            seq_id=self._seq, timestamp=timestamp,
            rgb=rgb, thermal_gray=thermal_gray,
        )
        self._seq += 1
        try:
            self._frame_queue.put_nowait(fp)
        except queue.Full:
            print("[PANORAMA] Warning: frame queue full, dropping frame.")

    def get_preview(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._canvas_mgr is None:
                return None
            cropped = self._canvas_mgr.get_cropped()
        colored = self._renderer.render(cropped)
        h, w = colored.shape[:2]
        if w > 800:
            scale = 800.0 / w
            colored = cv2.resize(colored, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
        return colored

    def export(self, output_dir: Optional[Path] = None) -> Optional[Path]:
        with self._lock:
            if self._canvas_mgr is None:
                print("[PANORAMA] Nothing to export — no frames stitched.")
                return None
            cropped = self._canvas_mgr.get_cropped()

        colored = self._renderer.render(cropped)
        with_bar = self._renderer.add_scale_bar(colored)

        if output_dir is None:
            output_dir = OUTPUT_DIR
        pano_dir = output_dir / "panorama"
        ensure_dir(pano_dir)

        stamp = timestamp_id()
        full_path = pano_dir / f"heatmap_{stamp}.png"
        save_image(full_path, with_bar)

        # Preview (max 2048px wide)
        h, w = with_bar.shape[:2]
        if w > 2048:
            scale = 2048.0 / w
            preview = cv2.resize(with_bar, (2048, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            preview = with_bar
        preview_path = pano_dir / f"heatmap_{stamp}_preview.jpg"
        save_image(preview_path, preview)

        # Metadata
        duration = time.time() - self._start_time if self._start_time else 0.0
        avg_inliers = (self._total_inliers / max(self._frames_stitched, 1))
        meta = {
            "total_frames": self._seq,
            "frames_stitched": self._frames_stitched,
            "frames_skipped": self._frames_skipped,
            "canvas_width": int(cropped.shape[1]),
            "canvas_height": int(cropped.shape[0]),
            "duration_s": round(duration, 2),
            "low_confidence_count": self._low_conf_count,
            "avg_inliers": round(avg_inliers, 1),
            "drift_corrections": self._drift_corrections,
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._start_time)),
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path = pano_dir / f"heatmap_{stamp}_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[PANORAMA] Exported → {full_path}")
        print(f"[PANORAMA]   stitched={self._frames_stitched}  skipped={self._frames_skipped}  "
              f"drift_corrections={self._drift_corrections}")
        return full_path

    # -- worker thread -------------------------------------------------------

    def _stitch_worker(self) -> None:
        """Background thread that consumes frame pairs and stitches them."""
        try:
            while self._alive:
                try:
                    fp: FramePair = self._frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                with self._lock:
                    self._process_frame(fp)

        except Exception as exc:
            print(f"[PANORAMA] Worker crashed: {exc}")
            self._alive = False

    def _process_frame(self, fp: FramePair) -> None:
        """Process a single frame pair (called under lock)."""
        import math
        thermal = fp.thermal_gray
        dx, dy = self._cfg.rgb_thermal_dx, self._cfg.rgb_thermal_dy
        if dx != 0 or dy != 0:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            thermal = cv2.warpAffine(thermal, M, (thermal.shape[1], thermal.shape[0]))

        # First frame — initialise canvas
        if self._canvas_mgr is None:
            h, w = thermal.shape[:2]
            self._canvas_mgr = CanvasManager(
                h, w,
                padding_factor=self._cfg.canvas_padding_factor,
                overlap_margin_px=self._cfg.overlap_margin_px,
                rotation_threshold_deg=self._cfg.rotation_threshold_deg,
            )
            self._canvas_mgr.place_first(thermal)
            self._prev = fp
            self._H_acc = np.eye(3, dtype=np.float64)
            fp.H_to_canvas = self._H_acc.copy()
            self._stitched.append(fp)
            self._frames_stitched = 1
            print("[PANORAMA] First frame placed on canvas.")
            return

        H, inliers = self._aligner.compute_homography(self._prev.rgb, fp.rgb)

        if H is None and len(self._stitched) >= 2:
            alt_prev = self._stitched[-2]
            H, inliers = self._aligner.compute_homography(alt_prev.rgb, fp.rgb)
            if H is not None and alt_prev.H_to_canvas is not None:
                self._H_acc = alt_prev.H_to_canvas @ H
                fp.low_confidence = True
                self._low_conf_count += 1

        if H is None:
            self._frames_skipped += 1
            print(f"[PANORAMA] Frame {fp.seq_id} skipped (inliers={inliers}).")
            return

        # Extract movement info from H (frame-space)
        tx = float(H[0, 2])
        ty = float(H[1, 2])
        angle_deg = math.degrees(math.atan2(float(H[1, 0]), float(H[0, 0])))
        move = math.sqrt(tx**2 + ty**2)

        # Reject bad matches
        if move > self._cfg.max_move_px:
            self._frames_skipped += 1
            print(f"[PANORAMA] Frame {fp.seq_id} rejected: move={move:.1f}px > max={self._cfg.max_move_px}")
            return

        # Silent skip if robot hasn't moved
        if move < self._cfg.min_move_px:
            return

        if not fp.low_confidence:
            self._H_acc = self._H_acc @ H

        fp.H_to_canvas = self._H_acc.copy()
        fp.inlier_count = inliers
        self._total_inliers += inliers

        self._canvas_mgr.expand_if_needed()
        ok = self._canvas_mgr.warp_and_blend(thermal, self._H_acc, tx, ty, angle_deg)
        if ok:
            self._frames_stitched += 1
            self._stitched.append(fp)
            self._prev = fp
            self._emit_update()
            if (self._frames_stitched % self._cfg.bundle_adjust_interval == 0
                    and self._frames_stitched > self._cfg.bundle_adjust_interval):
                self._drift_correct()
        else:
            self._frames_skipped += 1

    def _drift_correct(self) -> None:
        """Simple drift correction: cross-match current frame with older frames."""
        if len(self._stitched) < 10:
            return

        curr = self._stitched[-1]
        offsets = [5, 10, 15]
        for off in offsets:
            idx = len(self._stitched) - 1 - off
            if idx < 0:
                continue
            old = self._stitched[idx]
            H_cross, inliers = self._aligner.compute_homography(old.rgb, curr.rgb)
            if H_cross is not None and inliers > 15:
                # Compute expected H from chain
                if old.H_to_canvas is not None and curr.H_to_canvas is not None:
                    H_expected = curr.H_to_canvas @ np.linalg.inv(old.H_to_canvas)
                    H_measured = np.linalg.inv(H_cross)
                    # Blend 50% correction
                    correction = (H_measured + H_expected) / 2.0
                    correction[2, :] = [0, 0, 1]  # keep projective row clean
                    # Apply correction to accumulated H
                    self._H_acc = old.H_to_canvas @ correction
                    self._drift_corrections += 1
                    tx = abs(H_measured[0, 2] - H_expected[0, 2])
                    ty = abs(H_measured[1, 2] - H_expected[1, 2])
                    print(f"[PANORAMA] Drift correction applied (Δx={tx:.1f}px, Δy={ty:.1f}px)")
                break

    def _emit_update(self) -> None:
        """Emit a PANORAMA_UPDATE event via the callback."""
        if self._on_event is None or self._canvas_mgr is None:
            return
        try:
            from net_inspector.orchestrator import DetectionEvent
            from datetime import datetime, timezone

            ch, cw = self._canvas_mgr.shape
            self._on_event(DetectionEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="panorama",
                event_type="PANORAMA_UPDATE",
                confidence=1.0,
                metadata={
                    "frames_stitched": self._frames_stitched,
                    "canvas_w": cw,
                    "canvas_h": ch,
                },
            ))
        except Exception:
            pass  # non-critical
