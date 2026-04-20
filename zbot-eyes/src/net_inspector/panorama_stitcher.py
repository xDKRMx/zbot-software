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

# Relocalization constants
MIN_RELOCALIZE_MATCHES = 20      # Minimum feature matches to attempt relocalization
MIN_RELOCALIZE_INLIERS = 15      # Minimum inliers to accept relocalization
KEYFRAME_DISTANCE_THRESHOLD = 100.0  # px - add keyframe if moved this far
KEYFRAME_TIME_THRESHOLD = 2.0    # seconds - add keyframe if this much time passed
MAX_KEYFRAMES = 200              # Maximum keyframes to store


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
    H_to_global: Optional[np.ndarray] = None  # Renamed from H_to_canvas for clarity
    inlier_count: int = 0
    low_confidence: bool = False
    features: Optional[tuple] = None  # (keypoints, descriptors) for relocalization


@dataclass
class Keyframe:
    """Keyframe for relocalization - stores spatial position in global coordinates."""

    id: int
    seq_id: int
    timestamp: float
    rgb: np.ndarray              # RGB image for feature matching
    thermal: np.ndarray          # Thermal data
    H_to_global: np.ndarray      # Transform to global coordinate system
    keypoints: tuple             # cv2.KeyPoint list
    descriptors: np.ndarray      # AKAZE descriptors
    canvas_bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) on canvas


@dataclass
class KeyframeMatch:
    """Result of relocalization attempt."""

    kf_id: int
    H_match: np.ndarray
    inliers: int


# ---------------------------------------------------------------------------
# FeatureAligner
# ---------------------------------------------------------------------------

class FeatureAligner:
    """Translation-only feature aligner — extracts only (tx, ty), discards rotation/scale."""

    def __init__(
        self,
        akaze_threshold: float = 0.001,
        ransac_threshold: float = 5.0,
        min_inliers: int = 6,
    ) -> None:
        self._akaze = cv2.AKAZE_create(threshold=akaze_threshold)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ransac_threshold = ransac_threshold
        self._min_inliers = min_inliers

    def extract_features(self, rgb: np.ndarray) -> tuple[tuple, Optional[np.ndarray]]:
        """Extract AKAZE features from RGB image.
        
        Returns:
            (keypoints, descriptors) - keypoints as tuple, descriptors as ndarray or None
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kp, des = self._akaze.detectAndCompute(gray, None)
        return (tuple(kp) if kp else tuple(), des)
    
    def compute_homography(
        self, rgb_prev: np.ndarray, rgb_curr: np.ndarray,
    ) -> tuple[Optional[np.ndarray], int]:
        """Returns a 3×3 pure-translation matrix or None."""
        gray_prev = cv2.cvtColor(rgb_prev, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(rgb_curr, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self._akaze.detectAndCompute(gray_prev, None)
        kp2, des2 = self._akaze.detectAndCompute(gray_curr, None)

        if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
            return None, 0

        raw_matches = self._matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 3:
            return None, len(good)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_threshold,
        )

        if M is None or mask is None:
            return None, 0

        inlier_count = int(mask.sum())
        if inlier_count < self._min_inliers:
            return None, inlier_count

        # Extract only translation — discard rotation and scale to prevent drift
        tx = float(M[0, 2])
        ty = float(M[1, 2])

        H = np.array([[1.0, 0.0, tx],
                      [0.0, 1.0, ty],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
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
        """Warp thermal frame onto canvas with soft blending for smooth seams.
        
        - Unvisited pixels: write directly (first visit)
        - Already visited pixels: soft blend (70% new + 30% old) for smooth seams
        """
        ch, cw = self._canvas.shape
        H_canvas = self._H_offset @ H_accumulated

        warped = cv2.warpPerspective(
            thermal_gray.astype(np.float32), H_canvas, (cw, ch),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
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

        # Smooth to reduce stitching seams at frame boundaries
        stretched = cv2.GaussianBlur(stretched, (7, 7), 2)
        stretched[~valid_mask] = 0

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
        self._H_to_global = np.eye(3, dtype=np.float64)  # Renamed from _H_acc
        self._stitched: list[FramePair] = []
        
        # Keyframe database for relocalization
        self._keyframes: list[Keyframe] = []
        self._kf_id_counter = 0
        self._last_keyframe_pos = np.array([0.0, 0.0])  # Track last keyframe position
        self._last_keyframe_time = 0.0

        # Stats (public for GUI access)
        self.frames_stitched = 0
        self.frames_skipped = 0
        self.drift_corrections = 0
        self.relocalizations = 0  # Track successful relocalizations
        self._low_conf_count = 0
        self._total_inliers = 0
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
        avg_inliers = (self._total_inliers / max(self.frames_stitched, 1))
        meta = {
            "total_frames": self._seq,
            "frames_stitched": self.frames_stitched,
            "frames_skipped": self.frames_skipped,
            "canvas_width": int(cropped.shape[1]),
            "canvas_height": int(cropped.shape[0]),
            "duration_s": round(duration, 2),
            "low_confidence_count": self._low_conf_count,
            "avg_inliers": round(avg_inliers, 1),
            "drift_corrections": self.drift_corrections,
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._start_time)),
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path = pano_dir / f"heatmap_{stamp}_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[PANORAMA] Exported → {full_path}")
        print(f"[PANORAMA]   stitched={self.frames_stitched}  skipped={self.frames_skipped}  "
              f"drift_corrections={self.drift_corrections}")
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
            self._H_to_global = np.eye(3, dtype=np.float64)
            fp.H_to_global = self._H_to_global.copy()
            self._stitched.append(fp)
            self.frames_stitched = 1
            
            # Add first frame as keyframe
            self._add_keyframe(fp)
            
            print("[PANORAMA] First frame placed on canvas.")
            return

        H, inliers = self._aligner.compute_homography(self._prev.rgb, fp.rgb)

        # Normal tracking failed - try alternative previous frame
        if H is None and len(self._stitched) >= 2:
            alt_prev = self._stitched[-2]
            H, inliers = self._aligner.compute_homography(alt_prev.rgb, fp.rgb)
            if H is not None and alt_prev.H_to_global is not None:
                alt_tx = float(H[0, 2])
                alt_ty = float(H[1, 2])
                self._H_to_global = alt_prev.H_to_global.copy()
                self._H_to_global[0, 2] -= alt_tx
                self._H_to_global[1, 2] -= alt_ty
                fp.low_confidence = True
                self._low_conf_count += 1

        # Tracking still failed - try relocalization against keyframe database
        if H is None:
            relocalize_match = self._relocalize(fp)
            
            if relocalize_match is not None:
                # Found matching keyframe! Compute global position from it
                kf = self._keyframes[relocalize_match.kf_id]
                # H_match maps keyframe → current, so apply it to keyframe's global position
                fp.H_to_global = kf.H_to_global @ relocalize_match.H_match
                self._H_to_global = fp.H_to_global.copy()
                fp.inlier_count = relocalize_match.inliers
                self._total_inliers += relocalize_match.inliers
                self.relocalizations += 1
                
                print(f"[RELOCALIZE] Matched keyframe {kf.id} (inliers={relocalize_match.inliers})")
                
                # Warp thermal to global canvas
                self._canvas_mgr.expand_if_needed()
                ok = self._canvas_mgr.warp_and_blend(thermal, self._H_to_global)
                if ok:
                    self.frames_stitched += 1
                    self._stitched.append(fp)
                    self._prev = fp
                    self._emit_update()
                    
                    # Add as keyframe if criteria met
                    if self._should_add_keyframe(fp):
                        self._add_keyframe(fp)
                else:
                    self.frames_skipped += 1
                return
            else:
                # Truly lost - skip frame
                self.frames_skipped += 1
                print(f"[PANORAMA] Frame {fp.seq_id} skipped (tracking lost, relocalization failed)")
                return

        tx = float(H[0, 2])
        ty = float(H[1, 2])
        move = math.sqrt(tx**2 + ty**2)

        if move > self._cfg.max_move_px:
            self.frames_skipped += 1
            print(f"[PANORAMA] Frame {fp.seq_id} rejected: move={move:.1f}px > max={self._cfg.max_move_px}")
            return

        if move < self._cfg.min_move_px:
            return

        if not fp.low_confidence:
            self._H_to_global[0, 2] += tx
            self._H_to_global[1, 2] += ty

        fp.H_to_global = self._H_to_global.copy()
        fp.inlier_count = inliers
        self._total_inliers += inliers

        self._canvas_mgr.expand_if_needed()
        ok = self._canvas_mgr.warp_and_blend(thermal, self._H_to_global)
        if ok:
            self.frames_stitched += 1
            self._stitched.append(fp)
            self._prev = fp
            self._emit_update()
            
            # Add as keyframe if criteria met
            if self._should_add_keyframe(fp):
                self._add_keyframe(fp)
            
            if (self.frames_stitched % self._cfg.bundle_adjust_interval == 0
                    and self.frames_stitched > self._cfg.bundle_adjust_interval):
                self._drift_correct()
        else:
            self.frames_skipped += 1

    def _drift_correct(self) -> None:
        """Loop closure via keyframe database - correct drift when revisiting known areas."""
        if len(self._keyframes) < 3:
            return

        curr = self._stitched[-1]
        if curr.H_to_global is None:
            return
        
        # Try to match current frame with older keyframes (skip most recent ones)
        for i in range(len(self._keyframes) - 3):  # Skip last 2 keyframes
            kf = self._keyframes[i]
            try:
                H_cross, inliers = self._aligner.compute_homography(kf.rgb, curr.rgb)
                if H_cross is not None and inliers > 20:  # Higher threshold for drift correction
                    # Found loop closure!
                    # H_expected: where we think curr is relative to kf (from chain)
                    H_expected = curr.H_to_global @ np.linalg.inv(kf.H_to_global)
                    # H_measured: where curr actually is relative to kf (from features)
                    H_measured = np.linalg.inv(H_cross)
                    
                    # Compute correction
                    correction = (H_measured + H_expected) / 2.0
                    correction[2, :] = [0, 0, 1]  # keep projective row clean
                    
                    # Apply correction to accumulated H
                    self._H_to_global = kf.H_to_global @ correction
                    self.drift_corrections += 1
                    
                    tx = abs(H_measured[0, 2] - H_expected[0, 2])
                    ty = abs(H_measured[1, 2] - H_expected[1, 2])
                    print(f"[LOOP_CLOSURE] Drift corrected via keyframe {kf.id} (Δx={tx:.1f}px, Δy={ty:.1f}px, inliers={inliers})")
                    break
            except Exception:
                continue

    # -- Relocalization methods (Phase 2) ---------------------------------------

    def _relocalize(self, fp: FramePair) -> Optional[KeyframeMatch]:
        """Attempt to relocalize current frame against keyframe database.
        
        Returns best matching keyframe if found, None otherwise.
        """
        if not self._keyframes:
            return None
        
        # Extract features for current frame if not already done
        if fp.features is None:
            fp.features = self._aligner.extract_features(fp.rgb)
        
        kp_curr, des_curr = fp.features
        if des_curr is None or len(kp_curr) < MIN_RELOCALIZE_MATCHES:
            return None
        
        best_match = None
        best_inliers = 0
        
        # Match against all keyframes
        for kf in self._keyframes:
            try:
                matches = self._aligner._matcher.knnMatch(kf.descriptors, des_curr, k=2)
                good = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                
                if len(good) < MIN_RELOCALIZE_MATCHES:
                    continue
                
                # Compute homography from keyframe to current
                src_pts = np.float32([kf.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                M, mask = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self._aligner._ransac_threshold,
                )
                
                if M is not None and mask is not None:
                    inliers = int(mask.sum())
                    if inliers > best_inliers:
                        # Convert affine to homography
                        H = np.array([[M[0, 0], M[0, 1], M[0, 2]],
                                      [M[1, 0], M[1, 1], M[1, 2]],
                                      [0.0, 0.0, 1.0]], dtype=np.float64)
                        best_inliers = inliers
                        best_match = KeyframeMatch(kf.id, H, inliers)
            
            except Exception as e:
                # Silently skip problematic keyframes
                continue
        
        return best_match if best_inliers >= MIN_RELOCALIZE_INLIERS else None

    def _should_add_keyframe(self, fp: FramePair) -> bool:
        """Determine if current frame should be added as keyframe.
        
        Criteria:
        - Distance from last keyframe > threshold
        - Time since last keyframe > threshold
        - First frame always added
        """
        if not self._keyframes:
            return True
        
        # Distance check
        if fp.H_to_global is not None:
            curr_pos = np.array([fp.H_to_global[0, 2], fp.H_to_global[1, 2]])
            dist = np.linalg.norm(curr_pos - self._last_keyframe_pos)
            if dist > KEYFRAME_DISTANCE_THRESHOLD:
                return True
        
        # Time check
        if fp.timestamp - self._last_keyframe_time > KEYFRAME_TIME_THRESHOLD:
            return True
        
        return False

    def _add_keyframe(self, fp: FramePair) -> None:
        """Add current frame as keyframe to database."""
        if fp.H_to_global is None:
            return
        
        # Extract features if not already done
        if fp.features is None:
            fp.features = self._aligner.extract_features(fp.rgb)
        
        kp, des = fp.features
        if des is None or len(kp) == 0:
            return
        
        # Compute canvas bounding box (approximate)
        h, w = fp.thermal_gray.shape[:2]
        x1 = int(fp.H_to_global[0, 2])
        y1 = int(fp.H_to_global[1, 2])
        x2 = x1 + w
        y2 = y1 + h
        
        # Create keyframe
        kf = Keyframe(
            id=self._kf_id_counter,
            seq_id=fp.seq_id,
            timestamp=fp.timestamp,
            rgb=fp.rgb.copy(),
            thermal=fp.thermal_gray.copy(),
            H_to_global=fp.H_to_global.copy(),
            keypoints=kp,
            descriptors=des.copy(),
            canvas_bbox=(x1, y1, x2, y2),
        )
        
        self._keyframes.append(kf)
        self._kf_id_counter += 1
        self._last_keyframe_pos = np.array([fp.H_to_global[0, 2], fp.H_to_global[1, 2]])
        self._last_keyframe_time = fp.timestamp
        
        # Prune old keyframes if database too large
        if len(self._keyframes) > MAX_KEYFRAMES:
            # Remove oldest keyframes (keep first and recent ones)
            mid_start = 1
            mid_end = len(self._keyframes) - (MAX_KEYFRAMES // 2)
            if mid_end > mid_start:
                del self._keyframes[mid_start:mid_end]
        
        print(f"[PANORAMA] Added keyframe {kf.id} (total: {len(self._keyframes)})")

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
                    "frames_stitched": self.frames_stitched,
                    "canvas_w": cw,
                    "canvas_h": ch,
                },
            ))
        except Exception:
            pass  # non-critical
