# Z-BOT Unified Detection System with GLM Orchestration

## 🎯 Vision

Integrate **all detection systems** (Wall/Net, Debris, Heat) under a **single GLM-powered conversational orchestrator** to create a "talking robot" that impresses Challenge Cup judges with real-time, intelligent commentary.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  UNIFIED DETECTION RUNNER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Wall/Net     │  │ Debris       │  │ Heat         │          │
│  │ Detection    │  │ Detection    │  │ Detection    │          │
│  │ (HSV Green)  │  │ (Non-green)  │  │ (Warm HSV)   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │ Event Aggregator│                            │
│                  │ • SS spam filter│                            │
│                  │ • Event cooldown│                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │ GLM Orchestrator│                            │
│                  │ • Multi-modal   │                            │
│                  │ • Conversational│                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │  ChatGLM Vision │                            │
│                  │  API (glm-4v)   │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │ Response Output │                            │
│                  │ • JSON + MD     │                            │
│                  │ • Audio (TTS)   │                            │
│                  │ • Display       │                            │
│                  └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Components

### 1. **Detection Systems**

#### Wall/Net Detection
- **Source**: `unified_runner.py` → `compute_net_mask()`
- **Method**: HSV green color detection
- **Thresholds**: 
  - NET: `net_coverage > 5%`
  - WALL: `net_coverage < 5%`
- **Output**: `DetectionEvent(source="wall_net", event_type="NET"|"WALL")`

#### Debris Detection
- **Source**: `unified_runner.py` → `compute_debris_mask()`
- **Method**: Non-green objects within net convex hull
- **Threshold**: `debris_coverage > 2%`
- **Output**: `DetectionEvent(source="debris", event_type="DEBRIS")`

#### Fire Detection
- **Source**: `unified_runner.py` → `compute_fire_mask()`
- **Method**: Red/orange HSV detection
- **Threshold**: `fire_coverage > 1%`
- **Output**: `DetectionEvent(source="fire", event_type="FIRE")`

#### Heat Detection
- **Source**: `unified_runner.py` → `compute_heat_mask()`
- **Method**: Warm color HSV + contour area
- **Threshold**: `hotspot_area > 600px`
- **Output**: `DetectionEvent(source="heat", event_type="HOTSPOT")`
- **Special**: Supports **thermal camera** (optional second camera)

### 2. **Event Aggregator** (`orchestrator.py`)

**Purpose**: Prevent screenshot spam and batch events

**Features**:
- **Event cooldown**: 3 seconds per event type
- **GLM request interval**: Configurable (default 10s)
- **Queue management**: Max 100 events
- **Frame selection**: Prioritizes thermal > RGB

**Anti-Spam Logic**:
```python
# Per-event-type cooldown
_last_event_ts = {"NET": 0.0, "WALL": 0.0, "DEBRIS": 0.0, ...}
_event_cooldown_s = 3.0

# GLM request throttling
min_interval_s = 10.0  # Configurable
```

### 3. **GLM Orchestrator** (`orchestrator.py`)

**Purpose**: Convert detection events → conversational responses

**Workflow**:
1. Aggregate events from queue
2. Build multi-modal prompt (text + image)
3. Send to ChatGLM Vision API
4. Save response (JSON + Markdown + Frame)
5. Queue for audio/display output

**Prompt Format**:
```
You are Z-BOT, a friendly wall-climbing inspection robot.
Analyze the current situation and provide a brief, conversational status update.

Current detections:
- NET (source: wall_net, confidence: 0.85)
  * net_coverage_percent: 42.3
- DEBRIS (source: debris, confidence: 0.65)
  * debris_coverage_percent: 3.2

Robot position: X=2.5m, Y=1.8m, Heading=45°

Provide a conversational response in this format:
## Status
## What I See
## Recommendation

Keep it brief, friendly, and informative. Add personality to engage judges!
```

**Response Example**:
```markdown
## Status
Hey there! I'm currently climbing at position (2.5m, 1.8m) and facing northeast.

## What I See
I've detected a safety net ahead covering about 42% of my view. However, I'm also 
spotting some debris caught in the net - looks like about 3% coverage. Not ideal!

## Recommendation
I recommend we pause here and alert the maintenance team about the debris. 
Safety first! Should I continue monitoring or move to a different section?
```

### 4. **Unified Runner** (`unified_runner.py`)

**Purpose**: Main entry point that runs all systems

**Features**:
- Dual camera support (RGB + thermal)
- Real-time visual display with overlays
- HUD showing all detection metrics
- GLM response preview
- Headless mode for Raspberry Pi

## 🚀 Usage

### Basic Command (Webcam Only)
```bash
python -m net_inspector.unified_runner --camera 0
```

### With Thermal Camera
```bash
python -m net_inspector.unified_runner --camera 0 --thermal-camera 1
```

### Full Configuration
```bash
python -m net_inspector.unified_runner \
  --camera 0 \
  --thermal-camera 1 \
  --width 640 \
  --height 480 \
  --fps 5.0 \
  --glm-interval 10.0 \
  --net-threshold 0.05 \
  --debris-threshold 0.02 \
  --fire-threshold 0.01 \
  --heat-threshold 210 \
  --heat-min-area 600
```

### Headless Mode (Raspberry Pi)
```bash
python -m net_inspector.unified_runner \
  --camera 0 \
  --thermal-camera 1 \
  --no-display \
  --glm-interval 15.0
```

## 🔑 Setup Requirements

### 1. Install Dependencies
```bash
cd zbot-eyes
pip install -r requirements.txt
pip install -e .
```

### 2. ChatGLM API Key
**Option A**: Environment variable
```bash
export CHATGLM_API_KEY="your_api_key_here"
```

**Option B**: File
```bash
echo "your_api_key_here" > secrets/chatglm_api_key.txt
```

### 3. Verify Setup
```bash
# Test GLM connection
python -m net_inspector.gui
# Click "Analyze current frame" in ChatGLM panel
```

## 📊 Output Structure

### Detection Events (In-Memory)
```json
{
  "timestamp": "2026-02-28T15:30:45.123456Z",
  "source": "wall_net",
  "event_type": "NET",
  "confidence": 0.85,
  "metadata": {
    "net_coverage_percent": 42.3
  }
}
```

### GLM Response (Saved to Disk)
```
outputs/orchestrator/response_20260228_153045_123456/
├── response.json      # Full response data + events
├── response.md        # Markdown conversational output
└── frame.jpg          # Associated frame (RGB or thermal)
```

**response.json**:
```json
{
  "timestamp": "2026-02-28T15:30:45.123456Z",
  "events": [
    {
      "timestamp": "2026-02-28T15:30:42.000000Z",
      "source": "wall_net",
      "event_type": "NET",
      "confidence": 0.85,
      "metadata": {"net_coverage_percent": 42.3}
    },
    {
      "timestamp": "2026-02-28T15:30:44.000000Z",
      "source": "debris",
      "event_type": "DEBRIS",
      "confidence": 0.65,
      "metadata": {"debris_coverage_percent": 3.2}
    }
  ],
  "markdown": "## Status\nHey there! I'm currently climbing...",
  "robot_position": null
}
```

## 🎤 Audio/Display Integration (Future)

### Text-to-Speech (TTS)
The orchestrator has a placeholder for TTS integration:

```python
def _trigger_audio_output(self, markdown: str) -> None:
    # TODO: Integrate with TTS system
    # Options:
    # 1. pyttsx3 (offline, cross-platform)
    # 2. gTTS (Google TTS, requires internet)
    # 3. Robot's built-in audio system
    pass
```

**Recommended for Challenge Cup**:
```bash
pip install pyttsx3
```

```python
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed
engine.setProperty('volume', 0.9)  # Volume
engine.say(markdown_response)
engine.runAndWait()
```

### Display Output
For judges, display the markdown on a small LCD screen or tablet mounted on the robot.

**Options**:
1. **Raspberry Pi Official Display** (7-inch touchscreen)
2. **HDMI Monitor** connected to Pi
3. **Web interface** accessible via WiFi

## 🔧 Integration with Existing Systems

### Heat Detection Project
The heat detection logic from `Z-bot Heat Detection/src/infer_rpi.py` has been **integrated** into `unified_runner.py`:

- ✅ Heuristic hotspot detection (warm HSV colors)
- ✅ Thermal camera support (optional)
- ✅ Contour-based area filtering
- ✅ Event emission on hotspot detection

**Migration Path**:
If you want to use the full YOLOv8 heat detection model:
1. Copy trained weights to `zbot-eyes/models/heat_yolov8.pt`
2. Modify `unified_runner.py` to load YOLO model
3. Replace `compute_heat_mask()` with YOLO inference

### Wall/Net Detection (run_headless.py)
The wall/net detection from `scripts/run_headless.py` has been **integrated** into `unified_runner.py`:

- ✅ Green HSV net detection
- ✅ NET vs WALL state logic
- ✅ Debris detection (convex hull method)
- ✅ Fire detection (red/orange HSV)

## 📈 Performance Tuning

### Raspberry Pi Optimization
```bash
# Lower resolution for faster processing
--width 320 --height 240

# Reduce FPS
--fps 3.0

# Increase GLM interval (less frequent requests)
--glm-interval 20.0

# Increase event cooldown (less spam)
# Edit in unified_runner.py:
self._event_cooldown_s = 5.0
```

### GPU Acceleration (Optional)
If using Raspberry Pi 4/5 with GPU:
```bash
# Install OpenCV with GPU support
pip install opencv-contrib-python
```

## 🎓 Challenge Cup Demo Script

**Scenario**: Robot climbs wall, encounters debris, detects hotspot

**Expected GLM Responses**:

1. **Initial Climb** (NET detected)
   > "Hey judges! I'm Z-BOT and I'm starting my climb. I can see the safety net ahead - looks good with about 45% coverage. All systems nominal!"

2. **Debris Encounter** (DEBRIS detected)
   > "Uh oh! I've spotted some debris caught in the net at my 2 o'clock position. It's covering about 3% of the net area. I'm marking this location for the maintenance crew."

3. **Hotspot Alert** (HOTSPOT detected)
   > "Alert! My thermal sensors are picking up a hotspot at approximately 3 meters ahead. Temperature signature suggests potential electrical issue. Recommend immediate inspection!"

4. **Wall Obstacle** (WALL detected)
   > "I've encountered a solid wall section - no net visible. I'm adjusting my path to find the next net section. Stand by!"

## 🐛 Troubleshooting

### GLM Not Responding
```bash
# Check API key
cat secrets/chatglm_api_key.txt

# Test connection
curl -X POST https://open.bigmodel.cn/api/paas/v4/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4v-flash","messages":[{"role":"user","content":"test"}]}'
```

### Camera Not Opening
```bash
# List available cameras (Linux)
v4l2-ctl --list-devices

# Test camera (Windows)
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### High CPU Usage
- Reduce resolution: `--width 320 --height 240`
- Lower FPS: `--fps 2.0`
- Disable display: `--no-display`

## 📝 Next Steps

1. ✅ **Core system implemented** (orchestrator + unified runner)
2. ⏳ **TTS integration** (add pyttsx3 for audio output)
3. ⏳ **Robot position API** (integrate with robot's localization system)
4. ⏳ **YOLOv8 heat model** (replace heuristic with trained model)
5. ⏳ **Web dashboard** (real-time monitoring interface)
6. ⏳ **Challenge Cup demo** (prepare scripted scenarios)

## 🎉 Summary

**You now have a unified system that**:
- ✅ Runs **all 3 detection systems** simultaneously
- ✅ Prevents **screenshot spam** with intelligent throttling
- ✅ Sends **multi-modal content** (image + logs) to GLM
- ✅ Generates **conversational responses** for judges
- ✅ Supports **dual cameras** (RGB + thermal)
- ✅ Works **headless** on Raspberry Pi
- ✅ Saves **all responses** to disk for review

**The 90-degree goal is scored!** 🎯⚽
