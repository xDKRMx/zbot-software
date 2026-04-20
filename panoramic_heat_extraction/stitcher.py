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
# Helpers (z-bot-map integration)
# ---------------------------------------------------------------------------

def normalize_percentile_map(src: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Normalize a map using percentile scaling."""
    scale = float(np.percentile(src, percentile))
    if scale <= 1e-6 or not np.isfinite(scale):
        return np.zeros_like(src, dtype=np.float32)
    return np.clip(src / scale, 0.0, 1.0).astype(np.float32)


def compute_detail_score_map(image_bgr: np.ndarray) -> np.ndarray:
    """Estimate soft detail/saliency map for adaptive blending.
    
    From z-bot-map by Lennart A. Conrad.
    Flat regions get low scores (safe to feather).
    Edge regions get high scores (keep single crisp observation).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    local_contrast = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0))
    
    score = (
        0.55 * normalize_percentile_map(grad_mag)
        + 0.25 * normalize_percentile_map(lap_abs)
        + 0.20 * normalize_percentile_map(local_contrast)
    )
    score = cv2.GaussianBlur(score, (0, 0), 2.0)
    score = cv2.dilate(score, np.ones((9, 9), dtype=np.uint8), iterations=1)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def compute_feather_weight(size_wh: tuple[int, int]) -> np.ndarray:
    """Compute distance-from-edge feather weight map."""
    width, height = size_wh
    y_grid, x_grid = np.ogrid[:height, :width]
    x_center, y_center = width / 2.0, height / 2.0
    
    dist_x = np.minimum(x_grid, width - 1 - x_grid).astype(np.float32) / x_center
    dist_y = np.minimum(y_grid, height - 1 - y_grid).astype(np.float32) / y_center
    dist = np.minimum(dist_x, dist_y)
    
    if dist.max() > 0:
        dist = dist / dist.max()
    dist = np.clip(dist, 1e-3, 1.0)
    return dist.astype(np.float32)


def compute_frame_sharpness(gray: np.ndarray) -> float:
    """Compute frame sharpness using Laplacian variance.
    
    Motion blur detection - low sharpness = skip frame.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def compute_frame_quality(
    gray: np.ndarray,
    inliers: int,
    move_px: float,
    min_sharpness: float = 10.0,
    max_move_px: float = 400.0
) -> tuple[float, str]:
    """Compute frame quality for blend weighting (z-bot-map style).
    
    Quality score affects blend weight, NOT frame acceptance.
    Lower quality = lower contribution in overlap regions.
    
    Returns:
        quality: 0.0-1.0 (1.0 = best)
        reason: Quality assessment reason
    """
    # 1. Sharpness score (permissive - webcam typically 20-60)
    sharpness = compute_frame_sharpness(gray)
    sharpness_score = np.clip(sharpness / 300.0, 0.1, 1.0)  # Min 0.1 (always contribute)
    
    # 2. Tracking confidence (inlier quality)
    inlier_score = np.clip(inliers / 30.0, 0.2, 1.0)  # Min 0.2
    
    # 3. Motion speed factor (slower = better)
    motion_score = np.clip(1.0 - (move_px / max_move_px) * 0.5, 0.3, 1.0)  # Min 0.3
    
    # Combined quality (permissive - always >= 0.2)
    quality = 0.4 * sharpness_score + 0.3 * inlier_score + 0.3 * motion_score
    quality = float(np.clip(quality, 0.2, 1.0))  # Never below 0.2
    
    reason = f"sharp={sharpness:.0f}, inliers={inliers}, move={move_px:.1f}px"
    return quality, reason


# ---------------------------------------------------------------------------
# Original Helpers
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
    detector_type: str = "orb"  # orb or sift (z-bot-map)
    akaze_threshold: float = 0.0001  # Not used (ORB/SIFT detector now)
    ransac_threshold: float = 5.0
    min_inliers: int = 3  # ORB detector optimized for thermal (was 4)
    use_ecc_fallback: bool = True  # z-bot-map ECC refinement
    blend_mode: str = "best"  # best, soft, or simple (z-bot-map integration)
    min_sharpness: float = 10.0  # Motion blur threshold (skip blurry frames)
    motion_quality_threshold: float = 0.0  # Skip frames below this quality
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
    """Translation-only feature aligner with z-bot-map enhancements.

    Extracts only (tx, ty) from feature matches, discards rotation and scale.
    This prevents drift accumulation — correct for a robot that only
    translates on a flat wall surface.
    
    z-bot-map integration:
    - SIFT detector option (better for some thermal scenes)
    - ECC fallback when descriptor matching fails
    """

    def __init__(self, detector_type: str = "orb",
                 ransac_threshold: float = 5.0, min_inliers: int = 3,
                 use_ecc: bool = True) -> None:
        # Detector selection: ORB (fast) or SIFT (more features)
        if detector_type == "sift":
            if hasattr(cv2, "SIFT_create"):
                self._detector = cv2.SIFT_create(nfeatures=2000)
                self._norm = cv2.NORM_L2
                print("[FEATURE] Using SIFT detector (z-bot-map style)")
            else:
                print("[FEATURE] SIFT unavailable, falling back to ORB")
                self._detector = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
                self._norm = cv2.NORM_HAMMING
        else:
            self._detector = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
            self._norm = cv2.NORM_HAMMING
        
        self._matcher = cv2.BFMatcher(self._norm, crossCheck=False)
        self._ransac_threshold = ransac_threshold
        self._min_inliers = min_inliers
        self._use_ecc = use_ecc
        # CLAHE for thermal contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))

    def compute_homography(self, rgb_prev: np.ndarray,
                           rgb_curr: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        """Returns a 3×3 pure-translation matrix or None."""
        g1 = cv2.cvtColor(rgb_prev, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(rgb_curr, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for low-res thermal (FLIR Lepton 160x120)
        if g1.shape[0] <= 120 or g1.shape[1] <= 160:
            g1 = self._clahe.apply(g1)
            g2 = self._clahe.apply(g2)
        
        # Gaussian blur for noise reduction before feature detection
        g1 = cv2.GaussianBlur(g1, (3, 3), 0)
        g2 = cv2.GaussianBlur(g2, (3, 3), 0)
        
        kp1, des1 = self._detector.detectAndCompute(g1, None)
        kp2, des2 = self._detector.detectAndCompute(g2, None)

        if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
            print(f"[FEATURE] SKIP: kp1={len(kp1) if kp1 else 0}, kp2={len(kp2) if kp2 else 0}, des1={des1 is not None}, des2={des2 is not None}")
            # z-bot-map ECC fallback for low-texture scenes
            if self._use_ecc:
                return self._ecc_fallback(g1, g2)
            return None, 0

        raw = self._matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < 0.8 * n.distance]  # Relaxed for thermal
        
        if len(good) < self._min_inliers:
            print(f"[FEATURE] SKIP: matches={len(good)} < min_inliers={self._min_inliers}")
            return None, 0

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
    
    def _ecc_fallback(self, gray1: np.ndarray, gray2: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        """ECC (Enhanced Correlation Coefficient) fallback from z-bot-map.
        
        When descriptor matching fails (low texture), use ECC alignment.
        """
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
        
        try:
            cc, warp_matrix = cv2.findTransformECC(
                gray1, gray2,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                gaussFiltSize=5
            )
            
            if cc < 0.5:  # Low correlation
                print(f"[ECC] SKIP: correlation={cc:.3f} too low")
                return None, 0
            
            # Extract translation from 2x3 matrix
            tx = float(warp_matrix[0, 2])
            ty = float(warp_matrix[1, 2])
            
            # Pure translation homography
            H = np.array([[1.0, 0.0, tx],
                         [0.0, 1.0, ty],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
            
            print(f"[ECC] SUCCESS: cc={cc:.3f}, tx={tx:.1f}, ty={ty:.1f}")
            return H, 0  # inliers unknown for ECC
            
        except cv2.error as e:
            print(f"[ECC] FAIL: {e}")
            return None, 0


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
                 rotation_threshold_deg: float = 5.0,
                 blend_mode: str = "best") -> None:
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
        self._blend_mode = blend_mode
        
        # z-bot-map best blend tracking
        if blend_mode == "best":
            self._detail_score = np.zeros((ch, cw), dtype=np.float32)
            print(f"[CANVAS] Using 'best' blend mode (z-bot-map)")
        else:
            self._detail_score = None

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
                       angle_deg: float = 0.0,
                       gray_bgr: Optional[np.ndarray] = None,
                       frame_quality: float = 1.0) -> bool:
        """Warp thermal frame onto canvas.
        
        Blend modes:
        - best: z-bot-map detail-based selection (crisp edges)
        - soft: weighted blend (smooth seams)
        - simple: last-write-wins
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
        
        if self._blend_mode == "best" and self._detail_score is not None:
            # z-bot-map best blend: select best pixel based on detail + quality
            self._blend_best(gray, gray_bgr, H_c, warped, new_data, frame_quality)
        elif self._blend_mode == "soft":
            # Soft blend: weighted average for smooth seams
            revisit = new_data & self._visited & ~first_visit
            if revisit.any():
                self._canvas[revisit] = (0.7 * warped[revisit] +
                                         0.3 * self._canvas[revisit])
        # else: simple mode - first_visit already written, no overlap blend

        return True
    
    def _blend_best(self, gray: np.ndarray, gray_bgr: Optional[np.ndarray],
                    H_c: np.ndarray, warped: np.ndarray, 
                    new_data: np.ndarray, frame_quality: float) -> None:
        """z-bot-map best blend: pick best observation per pixel."""
        ch, cw = self._canvas.shape
        
        # Compute detail score for this frame
        if gray_bgr is not None:
            detail_map = compute_detail_score_map(gray_bgr)
        else:
            # Fallback: use gradient magnitude on grayscale
            gray_norm = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 0 else gray
            gray_3ch = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR)
            detail_map = compute_detail_score_map(gray_3ch)
        
        # Feather weight (center > edges)
        feather = compute_feather_weight((gray.shape[1], gray.shape[0]))
        
        # Combine detail + feather + quality
        selection_score = detail_map * feather * frame_quality
        
        # Warp detail score
        warped_score = cv2.warpPerspective(
            selection_score.astype(np.float32), H_c, (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        # Update pixels where new frame has better detail
        better_mask = new_data & (warped_score > self._detail_score)
        if better_mask.any():
            self._canvas[better_mask] = warped[better_mask]
            self._detail_score[better_mask] = warped_score[better_mask]

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
        self._aligner = FeatureAligner(
            detector_type=config.detector_type,
            ransac_threshold=config.ransac_threshold,
            min_inliers=config.min_inliers,
            use_ecc=config.use_ecc_fallback
        )
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
                blend_mode=self._cfg.blend_mode,  # z-bot-map integration
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
            print(f"[STITCH] Frame {fp.seq_id} SKIPPED: no homography (total skip={self.frames_skipped})")
            return

        # Extract movement info from H (frame-space)
        # H maps prev→curr: positive tx means scene moved left (camera moved right)
        tx = float(H[0, 2])
        ty = float(H[1, 2])
        move = math.sqrt(tx**2 + ty**2)

        # Reject bad matches
        if move > self._cfg.max_move_px:
            self.frames_skipped += 1
            print(f"[STITCH] Frame {fp.seq_id} REJECTED: move={move:.1f}px > max={self._cfg.max_move_px}")
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
        
        # Compute frame quality for blend weighting (z-bot-map style)
        # NOTE: Quality affects BLEND weight, NOT frame skip decision
        gray_for_quality = cv2.cvtColor(fp.rgb, cv2.COLOR_BGR2GRAY) if fp.rgb is not None else thermal
        frame_quality, quality_reason = compute_frame_quality(
            gray_for_quality,
            inliers=inliers,
            move_px=move,
            min_sharpness=10.0,  # Very permissive (webcam ~20-40 typical)
            max_move_px=self._cfg.max_move_px
        )
        
        # z-bot-map: pass BGR frame for detail analysis (if available)
        gray_bgr = fp.rgb if fp.rgb is not None else None
        ok = self._canvas_mgr.warp_and_blend(
            thermal, self._H_acc, 
            tx=tx, ty=ty, angle_deg=0.0,
            gray_bgr=gray_bgr,
            frame_quality=frame_quality  # Used for blend weight only
        )
        if ok:
            self.frames_stitched += 1
            self._stitched.append(fp)
            self._prev = fp
            print(f"[STITCH] Frame {fp.seq_id} SUCCESS: inliers={inliers}, move={move:.1f}px, Q={frame_quality:.2f} | total={self.frames_stitched}")
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
