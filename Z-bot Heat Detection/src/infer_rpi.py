import argparse
import json
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import cv2
import numpy as np

try:
    import requests
except ImportError:
    requests = None


def _utc_iso() -> str:
    return datetime.now().astimezone().isoformat()

# We'll import YOLO here, but handle the case when weights are missing
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Import our tracking and detection logic
from infer_persist import Track, iou_xyxy


def _heuristic_hotspots(img_bgr: np.ndarray, threshold: int, min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    warm = (h <= 35) | (h >= 160)
    mask = warm & (s >= 80) & (v >= threshold)
    mask_u8 = (mask.astype(np.uint8) * 255)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    confs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < float(min_area):
            continue
        x, y, w, hh = cv2.boundingRect(c)
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + hh)

        roi = v[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            continue
        score = float(np.clip((roi.mean() - threshold) / max(1.0, (255 - threshold)), 0.0, 1.0))

        boxes.append([x1, y1, x2, y2])
        confs.append(score)

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.array(boxes, dtype=np.float32), np.array(confs, dtype=np.float32)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _xyxy_to_xywh_norm(box_xyxy: np.ndarray, w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    xc = ((x1 + x2) / 2.0) / float(w)
    yc = ((y1 + y2) / 2.0) / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return [_clip01(xc), _clip01(yc), _clip01(bw), _clip01(bh)]


def _draw_xyxy(img: np.ndarray, box_xyxy: np.ndarray, color: Tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _get_robot_position(api_url: str, timeout: float = 2.0) -> Optional[Dict[str, float]]:
    """Fetch robot position from API. Expected JSON: {"x": float, "y": float, "heading": float}"""
    if not requests:
        return None
    try:
        resp = requests.get(api_url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            # Validate required fields
            if "x" in data and "y" in data and "heading" in data:
                return {
                    "x": float(data["x"]),
                    "y": float(data["y"]),
                    "heading": float(data["heading"]),
                }
    except Exception:
        pass
    return None


def _compute_hotspot_world_coords(
    bbox_xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
    robot_x: float,
    robot_y: float,
    robot_heading_deg: float,
    camera_fov_deg: float,
    hotspot_distance_m: float,
) -> Tuple[float, float]:
    """Compute global (X,Y) of hotspot using robot position + camera geometry.
    
    Args:
        bbox_xyxy: bounding box [x1,y1,x2,y2] in pixels
        frame_w, frame_h: frame dimensions
        robot_x, robot_y: robot global position (meters or lat/lon)
        robot_heading_deg: robot heading (0=North/+Y, 90=East/+X, clockwise)
        camera_fov_deg: horizontal field of view (degrees)
        hotspot_distance_m: estimated distance to hotspot (meters)
    
    Returns:
        (hotspot_x, hotspot_y) in global coordinates
    """
    # Bbox center in normalized coordinates (0-1)
    x1, y1, x2, y2 = bbox_xyxy
    cx_norm = ((x1 + x2) / 2.0) / float(frame_w)
    
    # Angle offset from camera center (negative = left, positive = right)
    # cx_norm=0.5 → center, cx_norm=0 → left edge, cx_norm=1 → right edge
    angle_offset_deg = (cx_norm - 0.5) * camera_fov_deg
    
    # Hotspot bearing in global frame (robot heading + camera offset)
    hotspot_bearing_deg = robot_heading_deg + angle_offset_deg
    hotspot_bearing_rad = math.radians(hotspot_bearing_deg)
    
    # Compute global coordinates (assuming standard coordinate system: +X=East, +Y=North)
    # heading=0 → North (+Y), heading=90 → East (+X)
    hotspot_x = robot_x + hotspot_distance_m * math.sin(hotspot_bearing_rad)
    hotspot_y = robot_y + hotspot_distance_m * math.cos(hotspot_bearing_rad)
    
    return (float(hotspot_x), float(hotspot_y))

class HeadlessDetector:
    def __init__(
        self,
        weights: Path,
        conf: float,
        imgsz: int,
        source: str,
        mode: str,
        heuristic_thr: int,
        alarm_conf: float,
        show: bool,
        robot_api_url: Optional[str] = None,
        camera_fov: float = 62.0,
        hotspot_distance: float = 2.0,
        yolo_class: int = 0,
        all_classes: bool = False,
        save_dir: Optional[Path] = None,
        save_every_s: float = 2.0,
        save_crops: bool = True,
        save_meta: bool = True,
        heartbeat: bool = False,
        heartbeat_every_s: float = 5.0,
    ):
        self.conf = conf
        self.imgsz = imgsz
        self.mode = mode.lower()
        self.heuristic_thr = heuristic_thr
        self.alarm_conf = float(alarm_conf)
        self.show = bool(show)
        self.robot_api_url = robot_api_url
        self.camera_fov = float(camera_fov)
        self.hotspot_distance = float(hotspot_distance)

        self.yolo_class = int(yolo_class)
        self.all_classes = bool(all_classes)

        # Auto-enable saving to 'detections/' if no save_dir specified
        if save_dir is None:
            save_dir = Path("detections")
        self.save_dir = save_dir
        self.save_every_s = float(save_every_s)
        self.save_crops = bool(save_crops)
        self.save_meta = bool(save_meta)
        self._last_save_ts = 0.0

        self.heartbeat = bool(heartbeat)
        self.heartbeat_every_s = float(heartbeat_every_s)
        self._last_heartbeat_ts = 0.0
        
        self.persist_frames = 3
        self.iou_match = 0.35
        self.max_missed = 8
        self.tracks: List[Track] = []

        if self.mode in ("auto", "yolo") and YOLO is not None:
            try:
                self.model = YOLO(str(weights))
            except Exception as e:
                self._log_event("SYSTEM_ERROR", f"Failed to load YOLO model: {e}")
                self.model = None
                self.mode = "heuristic"
        else:
            self.model = None

        self.cap_source = int(source) if source.isdigit() else source
        
    def _log_event(self, status: str, message: str = "", details: dict = None):
        """Outputs a structured JSON log that the Raspberry Pi / Robot controller can parse."""
        log = {
            "timestamp": _utc_iso(),
            "status": status
        }
        if message:
            log["message"] = message
        if details:
            log.update(details)
        print(json.dumps(log), flush=True)

    def run(self):
        # Use DirectShow backend on Windows to avoid MSMF issues
        if isinstance(self.cap_source, int):
            cap = cv2.VideoCapture(self.cap_source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.cap_source)
        
        if not cap.isOpened():
            self._log_event("SYSTEM_ERROR", f"Cannot open camera source: {self.cap_source}")
            return

        self._log_event("SYSTEM_START", f"Headless detection started on source {self.cap_source} in mode '{self.mode}'")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self._log_event("CAMERA_ERROR", "Failed to grab frame.")
                    time.sleep(1.0)
                    continue

                fh, fw = frame.shape[:2]
                dets: List[Tuple[np.ndarray, float, int]] = []

                # --- DETECTION ---
                if self.mode in ("auto", "yolo") and self.model:
                    results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
                    r0 = results[0]
                    if r0.boxes is not None and len(r0.boxes) > 0:
                        xyxy = r0.boxes.xyxy.cpu().numpy()
                        confs = r0.boxes.conf.cpu().numpy()
                        clss = r0.boxes.cls.cpu().numpy().astype(int)
                        # By default we filter class 0, but can be overridden via flags.
                        for box, c, k in zip(xyxy, confs, clss):
                            if self.all_classes or int(k) == self.yolo_class:
                                dets.append((box.astype(np.float32), float(c), int(k)))
                
                if self.mode == "heuristic" or (self.mode == "auto" and not dets):
                    xyxy, confs = _heuristic_hotspots(frame, threshold=self.heuristic_thr, min_area=600)
                    for box, c in zip(xyxy, confs):
                         dets.append((box.astype(np.float32), float(c), -1))

                # --- TRACKING ---
                matched_det = set()
                for t in self.tracks:
                    best_j = -1
                    best_iou = 0.0
                    for j, (dbox, dscore, _) in enumerate(dets):
                        if j in matched_det:
                            continue
                        v = iou_xyxy(t.box, dbox)
                        if v > best_iou:
                            best_iou = v
                            best_j = j

                    if best_j >= 0 and best_iou >= self.iou_match:
                        dbox, dscore, dcls = dets[best_j]
                        t.box = 0.7 * t.box + 0.3 * dbox
                        t.hits += 1
                        t.missed = 0
                        t.last_score = float(dscore)
                        t.last_cls = int(dcls)
                        matched_det.add(best_j)
                    else:
                        t.missed += 1

                for j, (dbox, dscore, dcls) in enumerate(dets):
                    if j in matched_det:
                        continue
                    nt = Track(box=dbox, hits=1, missed=0)
                    try:
                        nt.last_score = float(dets[j][1])
                    except Exception:
                        nt.last_score = 0.0
                    try:
                        nt.last_cls = int(dcls)
                    except Exception:
                        nt.last_cls = -1
                    self.tracks.append(nt)

                self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

                if self.heartbeat:
                    now_ts = time.time()
                    if (now_ts - self._last_heartbeat_ts) >= max(0.2, self.heartbeat_every_s):
                        self._log_event(
                            "SCANNING",
                            details={
                                "mode": self.mode,
                                "active_tracks": int(len(self.tracks)),
                                "frame_w": int(fw),
                                "frame_h": int(fh),
                            },
                        )
                        self._last_heartbeat_ts = now_ts

                # --- REPORTING ---
                alarm_triggered = False
                active_hits = 0

                stable_tracks = [
                    t
                    for t in self.tracks
                    if t.hits >= self.persist_frames and float(getattr(t, "last_score", 0.0)) >= self.alarm_conf
                ]
                if stable_tracks:
                    alarm_triggered = True
                    active_hits = max((t.hits for t in stable_tracks), default=0)

                if alarm_triggered:
                    # Fetch robot position if API is configured
                    robot_pos = None
                    if self.robot_api_url:
                        robot_pos = _get_robot_position(self.robot_api_url)
                    
                    boxes_out = []
                    for t in stable_tracks:
                        best_score = float(getattr(t, "last_score", 0.0))
                        xyxy_px = [float(v) for v in t.box.tolist()]
                        xywh_norm = _xyxy_to_xywh_norm(t.box, fw, fh)
                        
                        box_data = {
                            "xyxy_px": xyxy_px,
                            "xywh_norm": xywh_norm,
                            "score": float(best_score),
                            "hits": int(t.hits),
                        }
                        
                        # Compute world coordinates if robot position is available
                        if robot_pos:
                            world_x, world_y = _compute_hotspot_world_coords(
                                t.box, fw, fh,
                                robot_pos["x"], robot_pos["y"], robot_pos["heading"],
                                self.camera_fov, self.hotspot_distance
                            )
                            box_data["world_xy"] = [world_x, world_y]
                        
                        boxes_out.append(box_data)

                    if self.save_dir is not None:
                        now_ts = time.time()
                        if (now_ts - self._last_save_ts) >= max(0.0, self.save_every_s):
                            self.save_dir.mkdir(parents=True, exist_ok=True)
                            stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
                            base = f"alarm_{stamp}"

                            full_path = self.save_dir / f"{base}_full.png"
                            cv2.imwrite(str(full_path), frame)

                            if self.save_crops:
                                for i, t in enumerate(stable_tracks):
                                    x1, y1, x2, y2 = [int(v) for v in t.box.tolist()]
                                    x1 = max(0, min(x1, fw - 1))
                                    y1 = max(0, min(y1, fh - 1))
                                    x2 = max(0, min(x2, fw - 1))
                                    y2 = max(0, min(y2, fh - 1))
                                    if x2 <= x1 or y2 <= y1:
                                        continue
                                    crop = frame[y1:y2, x1:x2]
                                    crop_path = self.save_dir / f"{base}_crop_{i}.png"
                                    cv2.imwrite(str(crop_path), crop)

                            if self.save_meta:
                                meta = {
                                    "timestamp": _utc_iso(),
                                    "frame_w": int(fw),
                                    "frame_h": int(fh),
                                    "mode": self.mode,
                                    "alarm_conf": float(self.alarm_conf),
                                    "boxes": boxes_out,
                                }
                                if robot_pos:
                                    meta["robot_position"] = {
                                        "x": robot_pos["x"],
                                        "y": robot_pos["y"],
                                        "heading": robot_pos["heading"],
                                    }
                                meta_path = self.save_dir / f"{base}.json"
                                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

                            self._last_save_ts = now_ts

                    log_details = {
                        "target": "PERSISTENT_HOTSPOT",
                        "hits": int(active_hits),
                        "active_tracks": int(len(self.tracks)),
                        "frame_w": int(fw),
                        "frame_h": int(fh),
                        "boxes": boxes_out,
                        "mode": self.mode,
                    }
                    
                    # Include robot position in log if available
                    if robot_pos:
                        log_details["robot_position"] = {
                            "x": robot_pos["x"],
                            "y": robot_pos["y"],
                            "heading": robot_pos["heading"],
                        }
                    
                    self._log_event(
                        status="ALARM", 
                        message="Persistent hotspot detected.",
                        details=log_details,
                    )
                else:
                    # Optional: uncomment to print heartbeat
                    # self._log_event("SCANNING", details={"active_tracks": len(self.tracks)})
                    pass

                if self.show:
                    disp = frame.copy()
                    for t in self.tracks:
                        stable = t.hits >= self.persist_frames
                        color = (0, 0, 255) if stable else (0, 255, 255)
                        cls_txt = getattr(t, "last_cls", -1)
                        scr_txt = getattr(t, "last_score", 0.0)
                        _draw_xyxy(disp, t.box, color, f"cls={cls_txt} score={scr_txt:.2f} hits={t.hits}")
                    cv2.imshow("Z-BOT Headless Preview", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        self._log_event("SYSTEM_STOP", "User requested stop. Stopping camera.")
                        break
                
                # Sleep tiny amount to yield CPU if needed, though reading frames usually throttles
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            self._log_event("SYSTEM_STOP", "Process interrupted by user. Stopping camera.")
        finally:
            cap.release()
            if self.show:
                cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser(description="Headless Heat Detection Script for Raspberry Pi")
    default_weights = "runs/detect/hotspot_yolov8n/weights/best.pt"
    if not Path(default_weights).exists():
        default_weights = "yolov8n.pt"

    ap.add_argument("--weights", type=str, default=default_weights)
    ap.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0) or video file")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--mode", type=str, default="heuristic", choices=["yolo", "heuristic", "auto"])
    ap.add_argument("--yolo-class", type=int, default=0, help="YOLO class id to accept (ignored if --all-classes)")
    ap.add_argument("--all-classes", action="store_true", help="Accept detections from all YOLO classes")
    ap.add_argument("--thr", type=int, default=210, help="Heuristic threshold")
    ap.add_argument("--alarm-conf", type=float, default=0.5, help="Alarm score threshold for stable tracks")
    ap.add_argument("--show", action="store_true", help="Show OpenCV preview window (for local testing)")
    ap.add_argument("--robot-api-url", type=str, default=None, help="Robot localization API endpoint (e.g. http://192.168.1.100:5000/robot/position)")
    ap.add_argument("--camera-fov", type=float, default=62.0, help="Camera horizontal field of view in degrees (default: 62)")
    ap.add_argument("--hotspot-distance", type=float, default=2.0, help="Estimated distance to hotspot in meters (default: 2.0)")
    ap.add_argument("--save-dir", type=str, default=None, help="If set, save alarm screenshots into this folder")
    ap.add_argument("--save-every", type=float, default=2.0, help="Min seconds between saved alarm samples")
    ap.add_argument("--save-crops", action="store_true", help="Also save per-box crops when saving screenshots")
    ap.add_argument("--save-meta", action="store_true", help="Also save a JSON metadata file alongside screenshots")
    ap.add_argument("--heartbeat", action="store_true", help="Print periodic SCANNING logs (useful in headless mode)")
    ap.add_argument("--heartbeat-every", type=float, default=5.0, help="Seconds between SCANNING logs")
    args = ap.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else None

    detector = HeadlessDetector(
        weights=Path(args.weights),
        conf=args.conf,
        imgsz=args.imgsz,
        source=args.source,
        mode=args.mode,
        heuristic_thr=args.thr,
        alarm_conf=args.alarm_conf,
        show=args.show,
        robot_api_url=args.robot_api_url,
        camera_fov=args.camera_fov,
        hotspot_distance=args.hotspot_distance,
        yolo_class=args.yolo_class,
        all_classes=args.all_classes,
        save_dir=save_dir,
        save_every_s=args.save_every,
        save_crops=args.save_crops,
        save_meta=args.save_meta,
        heartbeat=args.heartbeat,
        heartbeat_every_s=args.heartbeat_every,
    )
    detector.run()

if __name__ == "__main__":
    main()
