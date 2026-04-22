"""Z-BOT-MAP Based Thermal Stitcher - 100% Port
Port of live_webcam_map.py with thermal camera support.
All stitching logic from z-bot-map (Lennart A. Conrad).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

# Add z-bot-map to path for imports
ZBOT_MAP_DIR = Path(__file__).resolve().parent.parent / "z-bot-map"
if str(ZBOT_MAP_DIR) not in sys.path:
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


class ThermalMapper:
    """Live thermal mapper - 100% z-bot-map logic + thermal support.
    
    Based on LiveWebcamMapper from live_webcam_map.py (Lennart A. Conrad).
    Extensions:
    - Dual canvas: RGB tracking + thermal heatmap
    - Thermal camera integration
    """
    
    def __init__(self, 
                 canvas_width: int = 2200,
                 canvas_height: int = 1600,
                 canvas_pad: int = 120,
                 max_canvas_mp: float = 64.0,
                 detector: str = "orb",
                 nfeatures: int = 2000,
                 memory_alpha: float = 0.45,
                 anchor_step_px: float = 8.0,
                 anchor_rotation_deg: float = 1.0,
                 lock_small_motion_updates: bool = True) -> None:
        
        # z-bot-map parameters
        self.args = SimpleNamespace(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            canvas_pad=canvas_pad,
            max_canvas_mp=max_canvas_mp,
            detector=detector,
            nfeatures=nfeatures,
            memory_alpha=clamp_memory_alpha(memory_alpha),
            anchor_step_px=anchor_step_px,
            anchor_rotation_deg=anchor_rotation_deg,
            lock_small_motion_updates=lock_small_motion_updates,
            # Default tracking args
            model="partial-affine",
            ratio=0.75,
            ransac=3.0,
            min_matches=8,
            min_inliers=6,
            min_inlier_ratio=0.2,
            use_ecc=False,
            disable_flow=True,
            min_ecc=0.85,
            max_translation_factor=2.5,   # increased: allows larger left/right moves
            max_rotation_jump_deg=75.0,
            min_scale=0.5,                # loosened: handles depth changes
            max_scale=2.0,                # loosened: handles depth changes
            min_determinant=0.1,
            max_determinant=10.0,
        )
        
        self.crop = parse_crop(None)
        self.tracking_args = self._build_tracking_args()
        self.detector, detector_name = make_detector(self.args.detector, self.args.nfeatures)
        self.matcher = make_matcher(detector_name)
        self.memory_alpha = self.args.memory_alpha
        
        # Thermal-specific
        self._thermal_minv = 40
        self._thermal_maxv = 160
        
        self.reset()
    
    def _build_tracking_args(self) -> SimpleNamespace:
        """Build tracking args from z-bot-map."""
        return SimpleNamespace(
            model=self.args.model,
            ratio=self.args.ratio,
            ransac=self.args.ransac,
            min_matches=self.args.min_matches,
            min_inliers=self.args.min_inliers,
            min_inlier_ratio=self.args.min_inlier_ratio,
            use_ecc=self.args.use_ecc,
            disable_flow=self.args.disable_flow,
            min_ecc=self.args.min_ecc,
            max_translation_factor=self.args.max_translation_factor,
            max_rotation_jump_deg=self.args.max_rotation_jump_deg,
            min_scale=self.args.min_scale,
            max_scale=self.args.max_scale,
            min_determinant=self.args.min_determinant,
            max_determinant=self.args.max_determinant,
        )
    
    def reset(self) -> None:
        """Reset mapper state (z-bot-map)."""
        # RGB tracking canvas (for feature detection)
        self.canvas_rgb = np.zeros(
            (self.args.canvas_height, self.args.canvas_width, 3),
            dtype=np.uint8,
        )
        self.valid_mask_rgb = np.zeros((self.args.canvas_height, self.args.canvas_width), dtype=np.uint8)
        
        # Thermal heatmap canvas (float32 for thermal values)
        self.canvas_thermal = np.zeros(
            (self.args.canvas_height, self.args.canvas_width),
            dtype=np.float32,
        )
        self.valid_mask_thermal = np.zeros((self.args.canvas_height, self.args.canvas_width), dtype=np.uint8)
        
        # z-bot-map state
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
        """Expand canvas if needed (z-bot-map)."""
        pad = self.args.canvas_pad
        min_x = float(np.min(corners_canvas[:, 0]))
        min_y = float(np.min(corners_canvas[:, 1]))
        max_x = float(np.max(corners_canvas[:, 0]))
        max_y = float(np.max(corners_canvas[:, 1]))
        canvas_h, canvas_w = self.canvas_rgb.shape[:2]

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

        # Expand RGB canvas
        expanded_canvas_rgb = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        expanded_mask_rgb = np.zeros((new_h, new_w), dtype=np.uint8)
        expanded_canvas_rgb[top : top + canvas_h, left : left + canvas_w] = self.canvas_rgb
        expanded_mask_rgb[top : top + canvas_h, left : left + canvas_w] = self.valid_mask_rgb
        self.canvas_rgb = expanded_canvas_rgb
        self.valid_mask_rgb = expanded_mask_rgb
        
        # Expand thermal canvas
        expanded_canvas_thermal = np.zeros((new_h, new_w), dtype=np.float32)
        expanded_mask_thermal = np.zeros((new_h, new_w), dtype=np.uint8)
        expanded_canvas_thermal[top : top + canvas_h, left : left + canvas_w] = self.canvas_thermal
        expanded_mask_thermal[top : top + canvas_h, left : left + canvas_w] = self.valid_mask_thermal
        self.canvas_thermal = expanded_canvas_thermal
        self.valid_mask_thermal = expanded_mask_thermal
        
        assert self.T_world_to_canvas is not None
        self.T_world_to_canvas[0, 2] += left
        self.T_world_to_canvas[1, 2] += top
        return True
    
    def paint(self, image_bgr: np.ndarray, thermal_gray: Optional[np.ndarray], 
              H_frame_to_world: np.ndarray) -> None:
        """Paint RGB + thermal to canvas (z-bot-map memory canvas)."""
        if self.T_world_to_canvas is None:
            self.T_world_to_canvas = self._make_canvas_transform(
                self.canvas_rgb.shape[:2],
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
        
        # Expand canvas if needed (z-bot-map)
        for _ in range(2):
            H_frame_to_canvas = self.T_world_to_canvas @ H_frame_to_world
            corners_canvas = transform_points(H_frame_to_canvas, local_corners)
            if not self.maybe_expand_canvas(corners_canvas):
                break

        H_frame_to_canvas = self.T_world_to_canvas @ H_frame_to_world
        canvas_h, canvas_w = self.canvas_rgb.shape[:2]
        
        # Warp RGB (for tracking)
        warped_rgb = cv2.warpPerspective(
            image_bgr,
            H_frame_to_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask_rgb = cv2.warpPerspective(
            np.full(image_bgr.shape[:2], 255, dtype=np.uint8),
            H_frame_to_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        
        # z-bot-map MEMORY CANVAS: pixels stay FIXED
        update_memory_canvas(self.canvas_rgb, self.valid_mask_rgb, warped_rgb, mask_rgb > 0, self.memory_alpha)
        
        # Warp thermal (if available)
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
            
            # Memory canvas for thermal (same logic)
            new_mask = mask_thermal & (self.valid_mask_thermal == 0)
            update_mask = mask_thermal & (self.valid_mask_thermal > 0)
            
            if np.any(new_mask):
                self.canvas_thermal[new_mask] = warped_thermal[new_mask]
            if np.any(update_mask) and self.memory_alpha > 0.0:
                old_pixels = self.canvas_thermal[update_mask].astype(np.float32)
                new_pixels = warped_thermal[update_mask].astype(np.float32)
                blended = old_pixels * (1.0 - self.memory_alpha) + new_pixels * self.memory_alpha
                self.canvas_thermal[update_mask] = blended
            self.valid_mask_thermal[mask_thermal] = 255
        
        self.mapped_frames += 1
    
    def paint_pose_for_tracking(self, color: np.ndarray, thermal: Optional[np.ndarray],
                                H_frame_to_world: np.ndarray) -> str:
        """Paint with anchor locking (z-bot-map)."""
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
        """Process frame (z-bot-map logic)."""
        color, gray = preprocess_frame(frame_bgr, self.crop, max_dim=640)
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
            self.status = (
                f"{source}  inliers={estimate.inlier_count}/{estimate.match_count}  "
                f"ratio={estimate.inlier_ratio:.2f}  paint={paint_mode}"
            )
        else:
            self.rejected_frames += 1
            self.lost_streak += 1
            self.status = f"rejected: {verdict[1]}"

        return color
    
    def get_thermal_heatmap(self) -> np.ndarray:
        """Get thermal heatmap (JET colormap)."""
        if not self.valid_mask_thermal.any():
            return np.zeros((self.args.canvas_height, self.args.canvas_width, 3), dtype=np.uint8)
        
        valid = self.canvas_thermal >= 1.0
        denom = max(self._thermal_maxv - self._thermal_minv, 1)
        stretched = ((np.clip(self.canvas_thermal, self._thermal_minv, self._thermal_maxv) - self._thermal_minv)
                     * (255.0 / denom)).astype(np.uint8)
        stretched = cv2.GaussianBlur(stretched, (7, 7), 2)
        stretched[~valid] = 0
        colored = cv2.applyColorMap(stretched, cv2.COLORMAP_JET)
        colored[~valid] = [0, 0, 0]
        
        # Crop to valid region
        if np.any(self.valid_mask_thermal):
            colored_cropped, _, _ = crop_to_valid_region(colored, self.valid_mask_thermal)
            return colored_cropped
        return colored
    
    def _make_canvas_transform(self, canvas_shape: tuple[int, int], 
                               frame_size: tuple[int, int]) -> np.ndarray:
        """Create canvas transform (z-bot-map)."""
        canvas_h, canvas_w = canvas_shape
        frame_w, frame_h = frame_size
        offset_x = canvas_w * 0.5 - frame_w * 0.5
        offset_y = canvas_h * 0.5 - frame_h * 0.5
        return np.array(
            [[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
