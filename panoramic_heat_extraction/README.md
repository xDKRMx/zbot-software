# Z-BOT Panoramic Heat Extraction

Standalone module — no other Z-BOT code needed. Drop this folder on the RPi and run.

## Install

```bash
pip install -r requirements.txt
```

## Run (GUI)

```bash
# Webcam only (RGB used as heat source)
python gui.py

# Webcam + IR camera, IR as heat source
python gui.py --rgb 0 --thermal 1 --source thermal

# Webcam only, slower capture for RPi
python gui.py --rgb 0 --capture-fps 1.0

# Custom temperature range
python gui.py --rgb 0 --thermal 1 --source thermal --thermal-minv 30 --thermal-maxv 180
```

## GUI Layout

```
┌─────────────────────────────────────────────────────────┐
│  Z-BOT Panoramic Heat Extraction    [Start] [Export] [Reset] │
│  Heat source: ● RGB camera  ○ Thermal / IR camera       │
├──────────────────────┬──────────────────────────────────┤
│   Live Camera        │   Thermal Heat Map (growing)     │
│   (640×480)          │   (updates every stitch)         │
│                      │                                  │
│   [real-time feed]   │   [2D heat mosaic + scale bar]   │
├──────────────────────┴──────────────────────────────────┤
│  Status bar          │  Stitched: N | Skipped: N | ...  │
└─────────────────────────────────────────────────────────┘
```

## How it works

1. Robot moves across the building surface
2. RGB frames are used for AKAZE feature matching → homography
3. Same transform applied to thermal/IR frames
4. Frames warped onto a growing float32 canvas with distance-weighted blending
5. Canvas colorized with JET colormap (MINV=cold/blue → MAXV=hot/red)
6. Press **Export** to save full-resolution PNG + metadata JSON

## Output

```
outputs/panorama/
  heatmap_YYYYMMDD_HHMMSS_us.png          # full resolution with scale bar
  heatmap_YYYYMMDD_HHMMSS_us_preview.jpg  # max 2048px wide
  heatmap_YYYYMMDD_HHMMSS_us_metadata.json
```

## RPi tips

- Use `--capture-fps 1.0` to reduce CPU load
- Use `--width 320 --height 240` for faster processing
- Make sure both cameras are listed: `v4l2-ctl --list-devices`


python3 -c "
import cv2
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f'video{i}: OK, frame={ret}, shape={frame.shape if ret else None}')
        cap.release()
    else:
        print(f'video{i}: FAILED')
"



python3 -c "
import cv2
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
print('opened:', cap.isOpened())
ret, frame = cap.read()
print('read:', ret, frame.shape if ret else None)
cap.release()
"
