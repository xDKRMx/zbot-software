"""Unified runner that combines wall/net, debris, and heat detection with GLM orchestration.

This script runs all three detection systems simultaneously and feeds events to the
GLM orchestrator for conversational robot responses.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Try to import Pure Thermal bridge (for FLIR Lepton 2.5)
try:
    import sys
    from pathlib import Path
    # Add panoramic_heat_extraction to path for purethermal_python import
    _pano_path = Path(__file__).resolve().parent.parent.parent.parent / "panoramic_heat_extraction"
    if _pano_path.exists() and str(_pano_path) not in sys.path:
        sys.path.insert(0, str(_pano_path))
    from purethermal_python import PureThermalCamera
    PURETHERMAL_AVAILABLE = True
except (ImportError, FileNotFoundError) as _e:
    PURETHERMAL_AVAILABLE = False
    PureThermalCamera = None

from net_inspector.config import AppConfig
from net_inspector.lepton_processor import LeptonThermalProcessor
from net_inspector.orchestrator import DetectionEvent, UnifiedOrchestrator
from net_inspector.panorama_stitcher import PanoramaStitcher


def compute_net_mask(image_bgr: np.ndarray, config: AppConfig) -> np.ndarray:
    """Compute net mask using green HSV thresholds."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = (config.heuristic.green_h_min, config.heuristic.green_s_min, config.heuristic.green_v_min)
    upper = (config.heuristic.green_h_max, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def compute_debris_mask(image_bgr: np.ndarray, net_mask: np.ndarray, config: AppConfig) -> np.ndarray:
    """Compute debris mask within net regions."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        (config.heuristic.green_h_min, config.heuristic.green_s_min, config.heuristic.green_v_min),
        (config.heuristic.green_h_max, 255, 255),
    )
    
    contours, _ = cv2.findContours(net_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    net_hull_mask = np.zeros_like(net_mask)
    
    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
        if valid_contours:
            all_points = np.vstack(valid_contours)
            hull = cv2.convexHull(all_points)
            cv2.drawContours(net_hull_mask, [hull], -1, 255, thickness=-1)
    
    non_green = cv2.bitwise_not(green_mask)
    debris = cv2.bitwise_and(non_green, net_hull_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    debris = cv2.morphologyEx(debris, cv2.MORPH_OPEN, kernel, iterations=2)
    debris = cv2.morphologyEx(debris, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return debris


def compute_fire_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Compute fire mask."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    fire1 = cv2.inRange(hsv, (0, 140, 140), (25, 255, 255))
    fire2 = cv2.inRange(hsv, (160, 140, 140), (179, 255, 255))
    fire = cv2.bitwise_or(fire1, fire2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fire = cv2.morphologyEx(fire, cv2.MORPH_OPEN, kernel, iterations=1)
    return fire


def compute_heat_mask(image_bgr: np.ndarray, threshold: int = 210, is_thermal: bool = False) -> np.ndarray:
    """Compute heat/hotspot mask.
    
    Args:
        image_bgr: Input image (BGR format)
        threshold: For RGB mode: HSV V threshold. For thermal mode: brightness threshold.
        is_thermal: If True, use direct brightness thresholding (for FLIR Lepton).
                   If False, use HSV warm-color heuristic (for RGB camera).
    """
    if is_thermal:
        # Thermal camera: use direct brightness threshold on grayscale
        # frame_thermal is already normalized thermal data (0-255)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Higher values = hotter
        mask_u8 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    else:
        # RGB camera: use HSV warm-color heuristic
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        warm = (h <= 35) | (h >= 160)
        mask = warm & (s >= 80) & (v >= threshold)
        mask_u8 = (mask.astype(np.uint8) * 255)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)
    
    return mask_u8


class UnifiedDetectionRunner:
    """Runs all detection systems and feeds events to orchestrator."""
    
    def __init__(
        self,
        camera_idx: int = 0,
        thermal_camera_idx: Optional[int] = None,
        width: int = 640,
        height: int = 480,
        fps_limit: float = 5.0,
        glm_interval_s: float = 10.0,
        net_threshold: float = 0.05,
        debris_threshold: float = 0.02,
        fire_threshold: float = 0.01,
        heat_threshold: int = 210,
        heat_min_area: int = 600,
        show_display: bool = True,
        speak: bool = True,
        panorama_config=None,
    ):
        self.camera_idx = camera_idx
        self.thermal_camera_idx = thermal_camera_idx
        self.width = width
        self.height = height
        self.fps_limit = fps_limit
        self.net_threshold = net_threshold
        self.debris_threshold = debris_threshold
        self.fire_threshold = fire_threshold
        self.heat_threshold = heat_threshold
        self.heat_min_area = heat_min_area
        self.show_display = show_display
        
        self.config = AppConfig()
        self.orchestrator = UnifiedOrchestrator(
            config=self.config,
            glm_interval_s=glm_interval_s,
            enable_audio_output=bool(speak),
        )
        
        # Panorama stitcher (optional)
        self._stitcher: Optional[PanoramaStitcher] = None
        self._panorama_interval = 0.5
        self._panorama_live_preview = False
        self._panorama_duration = 0.0
        if panorama_config is not None and panorama_config.enabled:
            self._stitcher = PanoramaStitcher(
                config=panorama_config,
                on_event=self.orchestrator.submit_event,
            )
            self._panorama_interval = panorama_config.capture_interval_s
            self._panorama_live_preview = panorama_config.live_preview
            self._panorama_duration = panorama_config.duration_s
        
        # Lepton thermal processor (fixed-scale normalization)
        self._lepton_processor = LeptonThermalProcessor(
            min_raw=28815,  # ~15°C
            max_raw=33315,  # ~60°C
            colormap=cv2.COLORMAP_JET,
            apply_histogram_eq=False,  # Fixed scale doesn't need histogram EQ
        )
        
        self._running = False
        self._last_event_ts = {
            "NET": 0.0,
            "WALL": 0.0,
            "DEBRIS": 0.0,
            "FIRE": 0.0,
            "HOTSPOT": 0.0,
        }
        self._event_cooldown_s = 3.0  # Prevent spam of same event type
    
    def _should_emit_event(self, event_type: str) -> bool:
        """Check if enough time has passed since last event of this type."""
        now = time.time()
        if (now - self._last_event_ts.get(event_type, 0.0)) >= self._event_cooldown_s:
            self._last_event_ts[event_type] = now
            return True
        return False
    
    def run(self) -> None:
        """Main detection loop."""
        # Open thermal camera (FLIR Lepton via Pure Thermal board) FIRST
        # Pure Thermal boards often fail with DSHOW - try multiple strategies
        cap_thermal = None
        _thermal_is_purethermal_bridge = False
        
        if self.thermal_camera_idx is not None:
            print(f"\n[THERMAL] Attempting to open Pure Thermal camera...")
            
            # METHOD 1: Try Pure Thermal DLL bridge (Windows Media Foundation)
            if PURETHERMAL_AVAILABLE and sys.platform == "win32":
                print("[THERMAL] Trying Pure Thermal bridge DLL...")
                try:
                    pt_cam = PureThermalCamera()
                    if pt_cam.is_connected():
                        cap_thermal = pt_cam
                        _thermal_is_purethermal_bridge = True
                        print(f"[THERMAL] ✅ SUCCESS - Pure Thermal via DLL bridge: {pt_cam.width}x{pt_cam.height}\n")
                        # Skip OpenCV attempts - DLL bridge works!
                    else:
                        print("[THERMAL] ⚠️  Bridge loaded but device not connected")
                except Exception as e:
                    print(f"[THERMAL] ❌ Bridge failed: {e}")
            
            # METHOD 2: Try OpenCV (fallback) - only if DLL didn't work
            if cap_thermal is None:
                print(f"[THERMAL] Trying OpenCV at index {self.thermal_camera_idx}...")
                
                # FLIR Lepton outputs Y16 format (16-bit grayscale), not RGB
                # Pure Thermal board: 160x120 (Lepton 2.5) or 80x60 (Lepton 2.0)
                
                backends_to_try = []
                if sys.platform == "win32":
                    # On Windows: MSMF works better than DSHOW for UVC devices
                    backends_to_try = [
                        (cv2.CAP_MSMF, "CAP_MSMF"),
                        (cv2.CAP_ANY, "CAP_ANY"),
                        (cv2.CAP_DSHOW, "CAP_DSHOW"),  # Last resort
                    ]
                else:
                    backends_to_try = [
                        (cv2.CAP_V4L2, "CAP_V4L2"),
                        (cv2.CAP_ANY, "CAP_ANY"),
                    ]
                
                # Try each backend until one works
                for backend, backend_name in backends_to_try:
                    print(f"[THERMAL] Trying {backend_name}...")
                    try:
                        if sys.platform == "win32":
                            tcap = cv2.VideoCapture(self.thermal_camera_idx, backend)
                        else:
                            tcap = cv2.VideoCapture(self.thermal_camera_idx, backend)
                        
                        if tcap.isOpened():
                            print(f"[THERMAL] {backend_name} opened successfully!")
                            
                            # Configure for Y16 format
                            tcap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                            tcap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
                            tcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                            tcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1','6',' '))
                            
                            # Test read to verify format
                            ret_test, frame_test = tcap.read()
                            if ret_test and frame_test is not None:
                                print(f"[THERMAL] Test read: shape={frame_test.shape} dtype={frame_test.dtype}")
                                
                                # Validate thermal camera format
                                # Accept various Lepton resolutions (some drivers crop/scale)
                                is_valid_thermal = False
                                if len(frame_test.shape) == 2:
                                    # Grayscale - check if small thermal camera resolution
                                    h, w = frame_test.shape
                                    # Accept: 60x80 (Lepton 2.0), 63x80 (cropped), 120x160 (Lepton 2.5), 122x164 (raw)
                                    is_valid_thermal = (50 <= h <= 130 and 70 <= w <= 170)
                                elif len(frame_test.shape) == 3 and frame_test.shape[2] == 1:
                                    # Single-channel image
                                    h, w = frame_test.shape[:2]
                                    is_valid_thermal = (50 <= h <= 130 and 70 <= w <= 170)
                                
                                if is_valid_thermal:
                                    # Success - Y16 or grayscale format
                                    cap_thermal = tcap
                                    print(f"[THERMAL] ✅ SUCCESS via {backend_name}: {frame_test.shape} {frame_test.dtype}")
                                    break
                                else:
                                    print(f"[THERMAL] ❌ Invalid thermal format: {frame_test.shape} (not thermal camera size)")
                                    tcap.release()
                            else:
                                print(f"[THERMAL] ❌ Test read failed")
                                tcap.release()
                        else:
                            print(f"[THERMAL] ❌ {backend_name} failed to open")
                            if tcap:
                                tcap.release()
                    except Exception as e:
                        print(f"[THERMAL] ❌ {backend_name} exception: {e}")
                
                # Final check (still inside 'if cap_thermal is None' block)
                if cap_thermal is None:
                    print(f"[THERMAL] ❌ FAILED: Could not open thermal camera with any backend")
                    print(f"[THERMAL] Tried: {', '.join([b[1] for b in backends_to_try])}")
            
        # Final thermal status
        if cap_thermal is not None:
            print(f"[THERMAL] Ready to capture thermal frames.\n")
        elif PURETHERMAL_AVAILABLE:
            print(f"[THERMAL] Note: Pure Thermal DLL bridge available but device not found.")
            print(f"[THERMAL] Continuing without thermal camera.")
        
        # Open RGB camera (optional if thermal camera is available)
        cap_rgb = None
        print(f"\n[RGB] Attempting to open RGB camera at index {self.camera_idx}...")
        try:
            if sys.platform == "win32":
                cap_rgb = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
            else:
                cap_rgb = cv2.VideoCapture(self.camera_idx)
            
            if cap_rgb and cap_rgb.isOpened():
                cap_rgb.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                cap_rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
                print(f"[RGB] ✅ SUCCESS - RGB camera opened: {self.width}x{self.height}")
            else:
                cap_rgb = None
                print(f"[RGB] ❌ Failed to open RGB camera")
        except Exception as e:
            cap_rgb = None
            print(f"[RGB] ❌ Exception: {e}")
        
        # Check if we have at least one camera
        if cap_rgb is None and cap_thermal is None:
            print(f"\n[ERROR] No cameras available (RGB and thermal both failed)")
            print(f"[ERROR] Cannot continue without at least one camera source")
            return
        
        if cap_rgb is None:
            print(f"[INFO] ⚠️  Thermal-only mode (no RGB overlay)")
        if cap_thermal is None:
            print(f"[INFO] ⚠️  RGB-only mode (no thermal data)")
        
        # Start orchestrator
        self.orchestrator.start()
        if self._stitcher is not None:
            self._stitcher.start()
        self._running = True
        
        print("[INFO] Unified detection started. Press 'q' to quit.")
        
        min_frame_time = 1.0 / self.fps_limit
        _last_panorama_ts = 0.0
        _panorama_start = time.time()
        _pano_preview_counter = 0
        
        try:
            while self._running:
                loop_start = time.time()
                
                # Read RGB frame (if available)
                frame_rgb = None
                if cap_rgb is not None and cap_rgb.isOpened():
                    ret_rgb, frame_rgb = cap_rgb.read()
                    if not ret_rgb or frame_rgb is None:
                        # Try to reopen camera
                        print("[WARNING] Failed to grab RGB frame, attempting to reopen...")
                        cap_rgb.release()
                        time.sleep(0.5)
                        if sys.platform == "win32":
                            cap_rgb = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
                        else:
                            cap_rgb = cv2.VideoCapture(self.camera_idx)
                        continue
                
                # If no RGB, use thermal as main frame for detection
                if frame_rgb is None and cap_thermal is None:
                    print("[ERROR] No RGB and no thermal frames available")
                    time.sleep(1.0)
                    continue
                
                # Read thermal frame if available (Y16 format from FLIR Lepton)
                frame_thermal = None
                frame_thermal_colorized = None  # For display
                if cap_thermal is not None:
                    # Both PureThermalCamera (DLL) and cv2.VideoCapture use .read() interface
                    ret_thermal, frame_thermal_raw = cap_thermal.read()
                    if ret_thermal and frame_thermal_raw is not None:
                        # Process with Lepton fixed-scale normalization
                        if frame_thermal_raw.dtype == np.uint16:
                            # Use fixed-scale Lepton processor
                            thermal_8bit, thermal_color = self._lepton_processor.process_frame(
                                frame_thermal_raw,
                                return_colorized=True
                            )
                            
                            # Store colorized version for display
                            frame_thermal_colorized = thermal_color
                            
                            # Convert grayscale to BGR for detection pipeline
                            frame_thermal = cv2.cvtColor(thermal_8bit, cv2.COLOR_GRAY2BGR)
                        else:
                            # Already 8-bit (shouldn't happen with Y16, but handle gracefully)
                            if len(frame_thermal_raw.shape) == 2:
                                frame_thermal = cv2.cvtColor(frame_thermal_raw, cv2.COLOR_GRAY2BGR)
                            else:
                                frame_thermal = frame_thermal_raw
                    else:
                        frame_thermal = None
                
                # --- PANORAMA CAPTURE ---
                if self._stitcher is not None and (frame_rgb is not None or frame_thermal is not None):
                    now_pano = time.time()
                    if (now_pano - _last_panorama_ts) >= self._panorama_interval:
                        _last_panorama_ts = now_pano
                        # Use thermal or RGB, whichever is available
                        source_frame = frame_thermal if frame_thermal is not None else frame_rgb
                        thermal_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
                        # For panorama, use RGB if available, otherwise thermal
                        rgb_frame = frame_rgb if frame_rgb is not None else frame_thermal
                        self._stitcher.feed_frame(rgb_frame.copy(), thermal_gray, now_pano)
                    # Duration auto-stop
                    if self._panorama_duration > 0 and (now_pano - _panorama_start) >= self._panorama_duration:
                        print("[PANORAMA] Duration reached, exporting...")
                        self._stitcher.export()
                        self._stitcher.stop()
                        self._stitcher = None
                
                # --- WALL/NET DETECTION --- (only if RGB available)
                net_mask = None
                net_pixels = 0
                net_ratio = 0.0
                system_state = "UNKNOWN"
                
                if frame_rgb is not None:
                    net_mask = compute_net_mask(frame_rgb, self.config)
                    net_pixels = np.count_nonzero(net_mask)
                    net_ratio = float(net_pixels) / float(net_mask.size)
                    system_state = "NET" if net_ratio > self.net_threshold else "WALL"
                else:
                    # Thermal-only mode: skip net/wall detection
                    net_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                    system_state = "THERMAL_ONLY"
                
                if system_state == "NET" and self._should_emit_event("NET"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="wall_net",
                        event_type="NET",
                        confidence=min(1.0, net_ratio * 10.0),
                        frame_rgb=frame_rgb.copy() if frame_rgb is not None else None,
                        metadata={"net_coverage_percent": round(net_ratio * 100, 2)},
                    ))
                elif system_state == "WALL" and self._should_emit_event("WALL"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="wall_net",
                        event_type="WALL",
                        confidence=0.8,
                        frame_rgb=frame_rgb.copy() if frame_rgb is not None else None,
                        metadata={"net_coverage_percent": round(net_ratio * 100, 2)},
                    ))
                
                # --- DEBRIS DETECTION --- (only if RGB available)
                debris_mask = None
                debris_pixels = 0
                debris_ratio = 0.0
                
                if frame_rgb is not None and net_mask is not None:
                    debris_mask = compute_debris_mask(frame_rgb, net_mask, self.config)
                    debris_pixels = np.count_nonzero(debris_mask)
                    debris_ratio = float(debris_pixels) / float(debris_mask.size) if net_pixels > 0 else 0.0
                else:
                    debris_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                
                if debris_ratio > self.debris_threshold and self._should_emit_event("DEBRIS"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="debris",
                        event_type="DEBRIS",
                        confidence=min(1.0, debris_ratio * 20.0),
                        frame_rgb=frame_rgb.copy() if frame_rgb is not None else None,
                        metadata={"debris_coverage_percent": round(debris_ratio * 100, 2)},
                    ))
                
                # --- FIRE DETECTION --- (only if RGB available)
                fire_mask = None
                fire_pixels = 0
                fire_ratio = 0.0
                
                if frame_rgb is not None:
                    fire_mask = compute_fire_mask(frame_rgb)
                    fire_pixels = np.count_nonzero(fire_mask)
                    fire_ratio = float(fire_pixels) / float(fire_mask.size)
                else:
                    fire_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                
                if fire_ratio > self.fire_threshold and self._should_emit_event("FIRE"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="fire",
                        event_type="FIRE",
                        confidence=min(1.0, fire_ratio * 50.0),
                        frame_rgb=frame_rgb.copy() if frame_rgb is not None else None,
                        metadata={"fire_coverage_percent": round(fire_ratio * 100, 2)},
                    ))
                
                # --- HEAT DETECTION (on thermal camera if available, else RGB) ---
                has_thermal = frame_thermal is not None
                heat_source_frame = frame_thermal if has_thermal else frame_rgb
                heat_mask = compute_heat_mask(
                    heat_source_frame, 
                    threshold=self.heat_threshold,
                    is_thermal=has_thermal
                )
                
                # Find contours for hotspots
                cnts, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hotspot_detected = False
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area >= self.heat_min_area:
                        hotspot_detected = True
                        break
                
                if hotspot_detected and self._should_emit_event("HOTSPOT"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="heat",
                        event_type="HOTSPOT",
                        confidence=0.85,
                        frame_rgb=frame_rgb.copy() if frame_rgb is not None else None,
                        frame_thermal=frame_thermal.copy() if frame_thermal is not None else None,
                        metadata={
                            "hotspot_count": len([c for c in cnts if cv2.contourArea(c) >= self.heat_min_area]),
                            "thermal_camera": frame_thermal is not None,
                        },
                    ))
                
                # --- DISPLAY ---
                if self.show_display and (frame_rgb is not None or frame_thermal_colorized is not None):
                    # Use thermal colorized if no RGB, otherwise RGB
                    if frame_rgb is not None:
                        display = frame_rgb.copy()
                    else:
                        display = frame_thermal_colorized.copy()
                    
                    # Resize masks to match display dimensions (thermal 160x120 vs RGB 640x480)
                    dh, dw = display.shape[:2]
                    
                    # Overlay masks with proper blending
                    if net_mask is not None and net_mask.shape[:2] != (dh, dw):
                        net_mask_resized = cv2.resize(net_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
                    else:
                        net_mask_resized = net_mask
                    
                    green_overlay = np.zeros_like(display)
                    if net_mask_resized is not None:
                        green_overlay[net_mask_resized > 0] = [0, 255, 0]
                    display = cv2.addWeighted(display, 1.0, green_overlay, 0.3, 0)
                    
                    if debris_mask is not None and debris_mask.shape[:2] != (dh, dw):
                        debris_mask_resized = cv2.resize(debris_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
                    else:
                        debris_mask_resized = debris_mask
                    
                    debris_overlay = np.zeros_like(display)
                    if debris_mask_resized is not None:
                        debris_overlay[debris_mask_resized > 0] = [0, 140, 255]
                    display = cv2.addWeighted(display, 1.0, debris_overlay, 0.3, 0)
                    
                    if fire_mask is not None and fire_mask.shape[:2] != (dh, dw):
                        fire_mask_resized = cv2.resize(fire_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
                    else:
                        fire_mask_resized = fire_mask
                    
                    fire_overlay = np.zeros_like(display)
                    if fire_mask_resized is not None:
                        fire_overlay[fire_mask_resized > 0] = [0, 0, 255]
                    display = cv2.addWeighted(display, 1.0, fire_overlay, 0.3, 0)
                    
                    if heat_mask is not None and heat_mask.shape[:2] != (dh, dw):
                        heat_mask_resized = cv2.resize(heat_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
                    else:
                        heat_mask_resized = heat_mask
                    
                    heat_overlay = np.zeros_like(display)
                    if heat_mask_resized is not None:
                        heat_overlay[heat_mask_resized > 0] = [255, 0, 255]
                    display = cv2.addWeighted(display, 1.0, heat_overlay, 0.3, 0)
                    
                    # HUD
                    cv2.putText(display, f"State: {system_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display, f"Net: {net_ratio*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Debris: {debris_ratio*100:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                    cv2.putText(display, f"Fire: {fire_ratio*100:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(display, f"Hotspot: {'YES' if hotspot_detected else 'NO'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    # Check for GLM response
                    response = self.orchestrator.get_latest_response()
                    if response:
                        # Display snippet of markdown
                        md_snippet = response["markdown"][:80] + "..." if len(response["markdown"]) > 80 else response["markdown"]
                        cv2.putText(display, f"GLM: {md_snippet}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    cv2.imshow("Z-BOT Unified Detection", display)
                    
                    # Show thermal if available
                    if frame_thermal_colorized is not None:
                        thermal_display = frame_thermal_colorized.copy()
                        # Overlay heat mask on thermal display
                        if heat_mask is not None:
                            if heat_mask.shape[:2] != thermal_display.shape[:2]:
                                heat_mask_th = cv2.resize(heat_mask, (thermal_display.shape[1], thermal_display.shape[0]), interpolation=cv2.INTER_NEAREST)
                            else:
                                heat_mask_th = heat_mask
                            heat_overlay_th = np.zeros_like(thermal_display)
                            heat_overlay_th[heat_mask_th > 0] = [255, 0, 255]
                            thermal_display = cv2.addWeighted(thermal_display, 1.0, heat_overlay_th, 0.3, 0)
                        cv2.imshow("Z-BOT Thermal Camera", thermal_display)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] User requested quit.")
                        break
                
                # Panorama live preview
                if self._stitcher is not None and self._panorama_live_preview:
                    _pano_preview_counter += 1
                    if _pano_preview_counter % 5 == 0:
                        pano_preview = self._stitcher.get_preview()
                        if pano_preview is not None:
                            cv2.imshow("Z-BOT Thermal Panorama", pano_preview)
                
                # Throttle to FPS limit
                process_time = time.time() - loop_start
                sleep_time = min_frame_time - process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user.")
        finally:
            self._running = False
            # Export panorama before shutdown
            if self._stitcher is not None:
                try:
                    self._stitcher.export()
                    # Emit PANORAMA_COMPLETE
                    from datetime import datetime as _dt, timezone as _tz
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=_dt.now(_tz.utc).isoformat(),
                        source="panorama",
                        event_type="PANORAMA_COMPLETE",
                        confidence=1.0,
                        metadata={"frames_stitched": self._stitcher._frames_stitched},
                    ))
                except Exception as exc:
                    print(f"[PANORAMA] Export error: {exc}")
                self._stitcher.stop()
            self.orchestrator.stop()
            if cap_rgb is not None:
                cap_rgb.release()
            if cap_thermal is not None:
                cap_thermal.release()
            if self.show_display:
                cv2.destroyAllWindows()
            print("[INFO] Unified detection stopped.")


def main():
    parser = argparse.ArgumentParser(description="Z-BOT Unified Detection with GLM Orchestration")
    parser.add_argument("--camera", type=int, default=0, help="RGB camera index")
    parser.add_argument("--thermal-camera", type=int, default=None, help="Thermal/infrared camera index (optional)")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=float, default=5.0, help="Maximum FPS to process")
    parser.add_argument("--glm-interval", type=float, default=10.0, help="Minimum seconds between GLM requests")
    parser.add_argument("--net-threshold", type=float, default=0.05, help="Net coverage threshold (0-1)")
    parser.add_argument("--debris-threshold", type=float, default=0.02, help="Debris coverage threshold (0-1)")
    parser.add_argument("--fire-threshold", type=float, default=0.01, help="Fire coverage threshold (0-1)")
    parser.add_argument("--heat-threshold", type=int, default=210, help="Heat detection HSV value threshold")
    parser.add_argument("--heat-min-area", type=int, default=600, help="Minimum hotspot area in pixels")
    parser.add_argument("--no-display", action="store_true", help="Disable visual display (headless mode)")
    parser.add_argument("--no-speak", action="store_true", help="Disable built-in TTS speech output")
    parser.add_argument("--panorama", action="store_true", help="Enable thermal panorama stitching")
    parser.add_argument("--panorama-interval", type=float, default=0.5, help="Seconds between panorama frame captures")
    parser.add_argument("--panorama-max-frames", type=int, default=500, help="Max frames in panorama buffer")
    parser.add_argument("--panorama-live-preview", action="store_true", help="Show live panorama preview window")
    parser.add_argument("--panorama-duration", type=float, default=0.0, help="Auto-stop panorama after N seconds (0=unlimited)")
    parser.add_argument("--panorama-rgb-thermal-dx", type=int, default=0, help="RGB-thermal pixel offset X")
    parser.add_argument("--panorama-rgb-thermal-dy", type=int, default=0, help="RGB-thermal pixel offset Y")
    
    args = parser.parse_args()
    
    # Build panorama config if enabled
    pano_cfg = None
    if args.panorama:
        from net_inspector.config import PanoramaConfig
        pano_cfg = PanoramaConfig(
            enabled=True,
            capture_interval_s=args.panorama_interval,
            max_frames=args.panorama_max_frames,
            live_preview=args.panorama_live_preview,
            duration_s=args.panorama_duration,
            rgb_thermal_dx=args.panorama_rgb_thermal_dx,
            rgb_thermal_dy=args.panorama_rgb_thermal_dy,
        )
    
    runner = UnifiedDetectionRunner(
        camera_idx=args.camera,
        thermal_camera_idx=args.thermal_camera,
        width=args.width,
        height=args.height,
        fps_limit=args.fps,
        glm_interval_s=args.glm_interval,
        net_threshold=args.net_threshold,
        debris_threshold=args.debris_threshold,
        fire_threshold=args.fire_threshold,
        heat_threshold=args.heat_threshold,
        heat_min_area=args.heat_min_area,
        show_display=not args.no_display,
        speak=not args.no_speak,
        panorama_config=pano_cfg,
    )
    
    runner.run()


if __name__ == "__main__":
    main()
