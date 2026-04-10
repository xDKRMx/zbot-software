"""Thermal Camera Process (Heat / Hotspot).
Connects to Infrared camera, emits MQTT events, and saves latest frame to disk.
"""

import argparse
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import json

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import cv2
import numpy as np

from net_inspector.mqtt_client import MessagePublisher

def process_thermal_frame(image_bgr: np.ndarray, min_area: int) -> tuple[np.ndarray, bool, int]:
    """
    1. Uses the raw grayscale output.
    2. Applies FIXED MINV/MAXV scaling to prevent breathing/auto-leveling based on environment.
    3. Applies JET colormap to match user's custom script.
    4. Detects relative hotspots against the room's temperature using light blur.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Analyze center area to drop UI overlays (black bars at bottom) from min/max calculations
    h, w = gray.shape
    roi = gray[int(h*0.05):int(h*0.90), int(w*0.05):int(w*0.95)]
    min_val, max_val, _, _ = cv2.minMaxLoc(roi)
    
    # User's Fixed Display Range Logic (adapted for 8-bit space)
    # Tune these once to lock the temperature-to-color mapping!
    MINV = 40  # Cold threshold (Fixed Blue)
    MAXV = 160 # Hot threshold (Fixed Red)
    
    # Clip and stretch mathematically exactly like the lepton script
    x = np.clip(gray, MINV, MAXV)
    x = ((x - MINV) * (255.0 / (MAXV - MINV))).astype(np.uint8)
    
    thermal_color = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    
    hotspot_count = 0
    # Threshold for detection: 10 raw pixel difference between room minimum and hottest object
    if (max_val - min_val) > 10:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh_level = max_val - ((max_val - min_val) * 0.15)  # Top 15% hottest things
        _, mask_u8 = cv2.threshold(blurred, thresh_level, 255, cv2.THRESH_BINARY)
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
        
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_hotspots = [c for c in cnts if cv2.contourArea(c) >= min_area]
        hotspot_count = len(valid_hotspots)
        
        if hotspot_count > 0:
            # Draw contours directly in black
            cv2.drawContours(thermal_color, valid_hotspots, -1, (0, 0, 0), 1)
            
    return thermal_color, hotspot_count > 0, hotspot_count

def run_thermal(
    camera_idx: int = 1,
    broker_ip: str = "127.0.0.1",
    broker_port: int = 1883,
    topic: str = "zbot/vision",
    width: int = 640,
    height: int = 480,
    fps_limit: float = 2.0,
    heat_threshold: int = 180,
    heat_min_area: int = 50,
    show_gui: bool = False
):
    print(f"[THERMAL] Starting IR Hotspot Detection on camera {camera_idx}")
    publisher = MessagePublisher(broker_ip=broker_ip, broker_port=broker_port, topic=topic)
    
    shared_dir = Path(__file__).resolve().parent.parent / "outputs" / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    latest_img_path = shared_dir / "latest_thermal.jpg"
    
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_idx)
        
    if not cap.isOpened():
        print(f"[ERROR] Cannot open Thermal camera {camera_idx}")
        sys.exit(1)
        
    last_event_ts = {"HOTSPOT": 0.0}
    event_cooldown_s = 3.0
    
    def emit_event(source: str, event_type: str, confidence: float, meta: dict):
        now = time.time()
        if (now - last_event_ts.get(event_type, 0.0)) >= event_cooldown_s:
            last_event_ts[event_type] = now
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "event_type": event_type,
                "confidence": confidence,
                "metadata": meta
            }
            publisher.publish(payload)
            print(f"[THERMAL] Emitted event: {event_type} (conf: {confidence:.2f})")
    
    min_frame_time = 1.0 / fps_limit
    print("[THERMAL] Loop running. Press 'q' to quit (if display enabled).")
    
    last_process_time = 0.0
    
    try:
        while True:
            # Fast grab to flush Linux USB hardware buffer (prevents controller stall/brownouts)
            ret = cap.grab()
            if not ret:
                time.sleep(0.5)
                continue
                
            now = time.time()
            if (now - last_process_time) < min_frame_time:
                continue # Skip processing this frame to save CPU / battery power
                
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue
                
            last_process_time = now
                
            # --- DETECTION LOGIC ---
            colorized_frame, hotspot_detected, hotspot_count = process_thermal_frame(frame, min_area=heat_min_area)
            
            # Write the COLORIZED thermal frame to disk 
            temp_path = str(latest_img_path).replace(".jpg", "_tmp.jpg")
            cv2.imwrite(temp_path, colorized_frame)
            try:
                os.replace(temp_path, str(latest_img_path))
            except PermissionError:
                pass  # Ignore if GLM Orchestrator is currently reading the file this millisecond
            
            if hotspot_detected:
                emit_event(
                    "heat", 
                    "HOTSPOT", 
                    0.85, 
                    {"hotspot_count": hotspot_count, "thermal_camera": True}
                )
            
            # --- DISPLAY ---
            if sys.platform == "win32" or show_gui:
                # Resize using INTER_CUBIC to get a smooth and high-resolution heatmap look
                display_resized = cv2.resize(colorized_frame, (1024, 768), interpolation=cv2.INTER_CUBIC)
                
                status_color = (0, 255, 0) if hotspot_detected else (0, 0, 255)
                cv2.putText(display_resized, f"Hotspot: {'YES' if hotspot_detected else 'NO'}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
                
                cv2.namedWindow("Thermal Camera", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Thermal Camera", 1024, 768)
                cv2.imshow("Thermal Camera", display_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("[THERMAL] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        publisher.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1, help="Thermal Camera index")
    parser.add_argument("--fps", type=float, default=2.0, help="Max FPS")
    parser.add_argument("--show", action="store_true", help="Show GUI")
    args = parser.parse_args()
    
    run_thermal(camera_idx=args.camera, fps_limit=args.fps, show_gui=args.show)
