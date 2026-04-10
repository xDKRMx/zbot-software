"""Headless execution script for Net Inspector.

This script runs on a Raspberry Pi without a GUI. It captures frames from a camera,
runs segmentation, detects debris/fire, and publishes JSON payloads via MQTT and stdout.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import urllib.request

# Automatically add the 'src' directory to Python's path so it can find 'net_inspector'
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import cv2
import numpy as np

# Adjust imports according to the package structure
import net_inspector.config as config_module
from net_inspector.segmenter import Segmenter
from net_inspector.mqtt_client import MessagePublisher

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_net_mask(image_bgr: np.ndarray, config: config_module.AppConfig) -> np.ndarray:
    """Compute net mask using green HSV thresholds."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = (config.heuristic.green_h_min, config.heuristic.green_s_min, config.heuristic.green_v_min)
    upper = (config.heuristic.green_h_max, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def compute_debris_mask(image_bgr: np.ndarray, net_mask: np.ndarray, config: config_module.AppConfig) -> np.ndarray:
    """Compute debris mask within net regions.
    Instead of relying on the raw net_mask, we find the convex hull of the net to define the 
    'expected net area', and then flag any non-green objects inside that area as debris.
    """
    # 1. Basic green detection
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        (config.heuristic.green_h_min, config.heuristic.green_s_min, config.heuristic.green_v_min),
        (config.heuristic.green_h_max, 255, 255),
    )
    
    # 2. Find the total "working area" of the net by bridging gaps (Convex Hull)
    contours, _ = cv2.findContours(net_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    net_hull_mask = np.zeros_like(net_mask)
    
    if contours:
        # Filter tiny specs
        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
        if valid_contours:
            # Create a convex hull bringing all major net parts together
            all_points = np.vstack(valid_contours)
            hull = cv2.convexHull(all_points)
            cv2.drawContours(net_hull_mask, [hull], -1, 255, thickness=-1)
    
    # 3. Debris is anything INSIDE the hull that is NOT GREEN
    non_green = cv2.bitwise_not(green_mask)
    debris = cv2.bitwise_and(non_green, net_hull_mask)
    
    # 4. Clean up the debris mask to avoid noise triggering it
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

def download_weights_if_missing():
    """Download default deeplabv3_resnet50.pth if not found."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    weights_path = models_dir / "deeplabv3_resnet50.pth"
    
    if not weights_path.exists():
        logger.warning(f"Weights not found at {weights_path}")
        logger.info("Downloading PyTorch COCO weights (approx 233MB)... Please wait.")
        models_dir.mkdir(parents=True, exist_ok=True)
        url = "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth"
        urllib.request.urlretrieve(url, str(weights_path))
        logger.info("Download complete!")

def run_headless(
    camera_idx: int = 0,
    broker_ip: str = "127.0.0.1",
    broker_port: int = 1883,
    topic: str = "zbot/vision",
    width: int = 640,
    height: int = 480,
    fps_limit: float = 5.0,
    mock_speed_y: float = 0.5, # Default climb speed: 0.5 meters/second (Y-axis)
    show_gui: bool = False     # Toggle to show display on any OS
):
    """Main headless loop."""
    config = config_module.AppConfig()
    
    # Auto-try downloading weights if they don't exist
    download_weights_if_missing()
    
    logger.info("Initializing Segmenter...")
    segmenter = Segmenter()
    if not segmenter.available():
        logger.error("Segmentation weights STILL not found after attempted download!")
        sys.exit(1)
        
    logger.info(f"Opening camera index {camera_idx}...")
    
    # Use DirectShow on Windows to avoid MSMF errors
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_idx)
        
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_idx}")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    
    publisher = MessagePublisher(broker_ip=broker_ip, broker_port=broker_port, topic=topic)
    
    logger.info("Starting headless detection loop. Press Ctrl+C to stop.")
    
    # --- Mock Localization State ---
    pos_x = 0.0
    pos_y = 0.0
    last_frame_time = time.time()
    logger.info(f"Initialized Mock Localization. Climbing speed Y={mock_speed_y}m/s")
    
    try:
        min_frame_time = 1.0 / fps_limit
        
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to grab frame. Retrying in 1s...")
                time.sleep(1.0)
                continue
                
            # Run Heavy Segmentation if needed (optional for future enhancements)
            # mask = segmenter.segment(frame)
            
            # --- Detect Scaffolding Net ---
            net_mask = compute_net_mask(frame, config)
            net_pixels = np.count_nonzero(net_mask)
            net_ratio = float(net_pixels) / float(net_mask.size)
            
            # --- Detect Debris inside the Net ---
            debris_mask = compute_debris_mask(frame, net_mask, config)
            debris_pixels = np.count_nonzero(debris_mask)
            debris_ratio = float(debris_pixels) / float(debris_mask.size) if net_pixels > 0 else 0.0
            
            # --- Detect Fire ---
            fire_mask = compute_fire_mask(frame)
            fire_pixels = np.count_nonzero(fire_mask)
            fire_ratio = float(fire_pixels) / float(fire_mask.size)
            
            # System State Decision Rules
            # If we see enough scaffolding net, we are facing the NET. Otherwise, an obstacle/WALL.
            # We set NET threshold very low (e.g., 5%) because a close wall might occlude it, but if any net is visible it's likely a net.
            NET_COVERAGE_THRESHOLD = 0.05
            system_state = "NET" if net_ratio > NET_COVERAGE_THRESHOLD else "WALL"
            
            # Sub-alert rules: Debris blocks the path, Fire is an emergency.
            alert_state = "NONE"
            if fire_ratio > 0.01:
                alert_state = "FIRE"
            elif debris_ratio > 0.02:
                alert_state = "DEBRIS"
            
            # --- Update Mock Localization ---
            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time
            
            # Simulate vertical climbing ONLY if we are facing a NET and there are no blocking alerts
            if system_state == "NET" and alert_state == "NONE":
                pos_y += mock_speed_y * dt
            
            # Generate Alert Metrics Payload
            payload = {
                "timestamp": current_time,
                "system_state": system_state,
                "alert": alert_state,
                "position": {
                    "x": float(round(pos_x, 3)),
                    "y": float(round(pos_y, 3))
                },
                "metrics": {
                    "net_coverage_percent": round(net_ratio * 100, 2),
                    "debris_coverage_percent": round(debris_ratio * 100, 2),
                    "fire_coverage_percent": round(fire_ratio * 100, 2)
                }
            }
            
            # Publish
            publisher.publish(payload)
            
            # Show visually (Only for testing on Windows OR if --show is passed on Linux)
            if sys.platform == "win32" or show_gui:
                # Draw net contours to show what is seen
                display_frame = frame.copy()
                display_frame[net_mask > 0] = [0, 255, 0] # Color net green
                display_frame[debris_mask > 0] = [0, 140, 255] # Color debris orange
                display_frame[fire_mask > 0] = [0, 0, 255] # Color fire red
                
                # Draw HUD (Heads-up display)
                cv2.putText(display_frame, f"State: {system_state} | Alert: {alert_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Map Pos: X={payload['position']['x']}m, Y={payload['position']['y']}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)
                cv2.putText(display_frame, f"Net: {payload['metrics']['net_coverage_percent']}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Debris: {payload['metrics']['debris_coverage_percent']}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                cv2.imshow("ZBot Headless Test View", display_frame)
                
                # Check for 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user frame window.")
                    break
            
            # Rest to meet fps_limit
            process_time = time.time() - loop_start
            sleep_time = min_frame_time - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        publisher.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZBot Headless Vision Processor")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--broker", type=str, default="127.0.0.1", help="MQTT Broker IP")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker Port")
    parser.add_argument("--topic", type=str, default="zbot/vision", help="MQTT Topic")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=float, default=5.0, help="Maximum FPS to process")
    parser.add_argument("--show", action="store_true", help="Force display the visual GUI on any OS")
    
    args = parser.parse_args()
    run_headless(
        camera_idx=args.camera,
        broker_ip=args.broker,
        broker_port=args.port,
        topic=args.topic,
        width=args.width,
        height=args.height,
        fps_limit=args.fps,
        show_gui=args.show
    )
