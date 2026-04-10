"""Webcam Detection Process (Wall/Net, Debris, Fire).
Connects to RGB camera, emits MQTT events, and saves latest frame to disk.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import json

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import cv2
import numpy as np

from net_inspector.config import AppConfig
from net_inspector.mqtt_client import MessagePublisher

def compute_net_mask(image_bgr: np.ndarray, config: AppConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = (config.heuristic.green_h_min, config.heuristic.green_s_min, config.heuristic.green_v_min)
    upper = (config.heuristic.green_h_max, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def compute_debris_mask(image_bgr: np.ndarray, net_mask: np.ndarray, config: AppConfig) -> np.ndarray:
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
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    fire1 = cv2.inRange(hsv, (0, 140, 140), (25, 255, 255))
    fire2 = cv2.inRange(hsv, (160, 140, 140), (179, 255, 255))
    fire = cv2.bitwise_or(fire1, fire2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fire = cv2.morphologyEx(fire, cv2.MORPH_OPEN, kernel, iterations=1)
    return fire

def run_webcam(
    camera_idx: int = 0,
    broker_ip: str = "127.0.0.1",
    broker_port: int = 1883,
    topic: str = "zbot/vision",
    width: int = 640,
    height: int = 480,
    fps_limit: float = 5.0,
    net_threshold: float = 0.05,
    debris_threshold: float = 0.02,
    fire_threshold: float = 0.01,
    show_gui: bool = False
):
    print(f"[WEBCAM] Starting RGB Wall/Net Detection on camera {camera_idx}")
    config = AppConfig()
    publisher = MessagePublisher(broker_ip=broker_ip, broker_port=broker_port, topic=topic)
    
    shared_dir = Path(__file__).resolve().parent.parent / "outputs" / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    latest_img_path = shared_dir / "latest_rgb.jpg"
    
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_idx)
        
    if not cap.isOpened():
        print(f"[ERROR] Cannot open RGB camera {camera_idx}")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    
    last_event_ts = {"NET": 0.0, "WALL": 0.0, "DEBRIS": 0.0, "FIRE": 0.0}
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
            print(f"[WEBCAM] Emitted event: {event_type} (conf: {confidence:.2f})")
    
    min_frame_time = 1.0 / fps_limit
    print("[WEBCAM] Loop running. Press 'q' to quit (if display enabled).")
    
    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(1.0)
                continue
                
            # Write latest frame to disk for the orchestrator
            cv2.imwrite(str(latest_img_path), frame)
            
            # --- DETECTION LOGIC ---
            net_mask = compute_net_mask(frame, config)
            net_pixels = np.count_nonzero(net_mask)
            net_ratio = float(net_pixels) / float(net_mask.size)
            
            system_state = "NET" if net_ratio > net_threshold else "WALL"
            meta_net = {"net_coverage_percent": round(net_ratio * 100, 2)}
            
            if system_state == "NET":
                emit_event("wall_net", "NET", min(1.0, net_ratio * 10.0), meta_net)
            else:
                emit_event("wall_net", "WALL", 0.8, meta_net)
                
            debris_mask = compute_debris_mask(frame, net_mask, config)
            debris_pixels = np.count_nonzero(debris_mask)
            debris_ratio = float(debris_pixels) / float(debris_mask.size) if net_pixels > 0 else 0.0
            
            if debris_ratio > debris_threshold:
                emit_event("debris", "DEBRIS", min(1.0, debris_ratio * 20.0), {"debris_coverage_percent": round(debris_ratio * 100, 2)})
                
            fire_mask = compute_fire_mask(frame)
            fire_pixels = np.count_nonzero(fire_mask)
            fire_ratio = float(fire_pixels) / float(fire_mask.size)
            
            if fire_ratio > fire_threshold:
                emit_event("fire", "FIRE", min(1.0, fire_ratio * 50.0), {"fire_coverage_percent": round(fire_ratio * 100, 2)})
            
            # --- DISPLAY ---
            if sys.platform == "win32" or show_gui:
                display = frame.copy()
                display[net_mask > 0] = [0, 255, 0]
                display[debris_mask > 0] = [0, 140, 255]
                cv2.putText(display, f"State: {system_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"Net: {net_ratio*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Debris: {debris_ratio*100:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                
                cv2.namedWindow("Webcam (Wall/Net)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Webcam (Wall/Net)", 1024, 768)
                cv2.imshow("Webcam (Wall/Net)", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            process_time = time.time() - loop_start
            sleep_time = min_frame_time - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("[WEBCAM] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        publisher.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=float, default=5.0, help="Max FPS")
    parser.add_argument("--show", action="store_true", help="Show GUI")
    args = parser.parse_args()
    
    run_webcam(camera_idx=args.camera, fps_limit=args.fps, show_gui=args.show)
