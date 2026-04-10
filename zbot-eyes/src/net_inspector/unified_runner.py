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

from net_inspector.config import AppConfig
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


def compute_heat_mask(image_bgr: np.ndarray, threshold: int = 210) -> np.ndarray:
    """Compute heat/hotspot mask using heuristic (warm colors)."""
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
        # Open RGB camera
        if sys.platform == "win32":
            cap_rgb = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        else:
            cap_rgb = cv2.VideoCapture(self.camera_idx)
        
        if not cap_rgb.isOpened():
            print(f"[ERROR] Cannot open RGB camera {self.camera_idx}")
            return
        
        cap_rgb.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap_rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        
        # Open thermal camera if specified
        cap_thermal = None
        if self.thermal_camera_idx is not None:
            if sys.platform == "win32":
                cap_thermal = cv2.VideoCapture(self.thermal_camera_idx, cv2.CAP_DSHOW)
            else:
                cap_thermal = cv2.VideoCapture(self.thermal_camera_idx)
            
            if cap_thermal.isOpened():
                cap_thermal.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                cap_thermal.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
                print(f"[INFO] Thermal camera {self.thermal_camera_idx} opened successfully.")
            else:
                print(f"[WARNING] Cannot open thermal camera {self.thermal_camera_idx}, continuing without it.")
                cap_thermal = None
        
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
                
                # Read RGB frame
                ret_rgb, frame_rgb = cap_rgb.read()
                if not ret_rgb or frame_rgb is None:
                    print("[WARNING] Failed to grab RGB frame.")
                    time.sleep(1.0)
                    continue
                
                # Read thermal frame if available
                frame_thermal = None
                if cap_thermal is not None:
                    ret_thermal, frame_thermal = cap_thermal.read()
                    if not ret_thermal:
                        frame_thermal = None
                
                # --- PANORAMA CAPTURE ---
                if self._stitcher is not None:
                    now_pano = time.time()
                    if (now_pano - _last_panorama_ts) >= self._panorama_interval:
                        _last_panorama_ts = now_pano
                        thermal_gray = cv2.cvtColor(
                            frame_thermal if frame_thermal is not None else frame_rgb,
                            cv2.COLOR_BGR2GRAY,
                        )
                        self._stitcher.feed_frame(frame_rgb.copy(), thermal_gray, now_pano)
                    # Duration auto-stop
                    if self._panorama_duration > 0 and (now_pano - _panorama_start) >= self._panorama_duration:
                        print("[PANORAMA] Duration reached, exporting...")
                        self._stitcher.export()
                        self._stitcher.stop()
                        self._stitcher = None
                
                # --- WALL/NET DETECTION ---
                net_mask = compute_net_mask(frame_rgb, self.config)
                net_pixels = np.count_nonzero(net_mask)
                net_ratio = float(net_pixels) / float(net_mask.size)
                
                system_state = "NET" if net_ratio > self.net_threshold else "WALL"
                
                if system_state == "NET" and self._should_emit_event("NET"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="wall_net",
                        event_type="NET",
                        confidence=min(1.0, net_ratio * 10.0),
                        frame_rgb=frame_rgb.copy(),
                        metadata={"net_coverage_percent": round(net_ratio * 100, 2)},
                    ))
                elif system_state == "WALL" and self._should_emit_event("WALL"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="wall_net",
                        event_type="WALL",
                        confidence=0.8,
                        frame_rgb=frame_rgb.copy(),
                        metadata={"net_coverage_percent": round(net_ratio * 100, 2)},
                    ))
                
                # --- DEBRIS DETECTION ---
                debris_mask = compute_debris_mask(frame_rgb, net_mask, self.config)
                debris_pixels = np.count_nonzero(debris_mask)
                debris_ratio = float(debris_pixels) / float(debris_mask.size) if net_pixels > 0 else 0.0
                
                if debris_ratio > self.debris_threshold and self._should_emit_event("DEBRIS"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="debris",
                        event_type="DEBRIS",
                        confidence=min(1.0, debris_ratio * 20.0),
                        frame_rgb=frame_rgb.copy(),
                        metadata={"debris_coverage_percent": round(debris_ratio * 100, 2)},
                    ))
                
                # --- FIRE DETECTION ---
                fire_mask = compute_fire_mask(frame_rgb)
                fire_pixels = np.count_nonzero(fire_mask)
                fire_ratio = float(fire_pixels) / float(fire_mask.size)
                
                if fire_ratio > self.fire_threshold and self._should_emit_event("FIRE"):
                    self.orchestrator.submit_event(DetectionEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="fire",
                        event_type="FIRE",
                        confidence=min(1.0, fire_ratio * 50.0),
                        frame_rgb=frame_rgb.copy(),
                        metadata={"fire_coverage_percent": round(fire_ratio * 100, 2)},
                    ))
                
                # --- HEAT DETECTION (on thermal camera if available, else RGB) ---
                heat_source_frame = frame_thermal if frame_thermal is not None else frame_rgb
                heat_mask = compute_heat_mask(heat_source_frame, threshold=self.heat_threshold)
                
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
                        frame_rgb=frame_rgb.copy(),
                        frame_thermal=frame_thermal.copy() if frame_thermal is not None else None,
                        metadata={
                            "hotspot_count": len([c for c in cnts if cv2.contourArea(c) >= self.heat_min_area]),
                            "thermal_camera": frame_thermal is not None,
                        },
                    ))
                
                # --- DISPLAY ---
                if self.show_display:
                    display = frame_rgb.copy()
                    
                    # Overlay masks with proper blending
                    green_overlay = np.zeros_like(display)
                    green_overlay[net_mask > 0] = [0, 255, 0]
                    display = cv2.addWeighted(display, 1.0, green_overlay, 0.3, 0)
                    
                    debris_overlay = np.zeros_like(display)
                    debris_overlay[debris_mask > 0] = [0, 140, 255]
                    display = cv2.addWeighted(display, 1.0, debris_overlay, 0.3, 0)
                    
                    fire_overlay = np.zeros_like(display)
                    fire_overlay[fire_mask > 0] = [0, 0, 255]
                    display = cv2.addWeighted(display, 1.0, fire_overlay, 0.3, 0)
                    
                    heat_overlay = np.zeros_like(display)
                    heat_overlay[heat_mask > 0] = [255, 0, 255]
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
                    if frame_thermal is not None:
                        thermal_display = frame_thermal.copy()
                        thermal_display[heat_mask > 0] = cv2.addWeighted(thermal_display[heat_mask > 0], 0.7, np.array([255, 0, 255], dtype=np.uint8), 0.3, 0)
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
