# Robot CV Demo Refactor Plan

## Goal

Refactor the current Raspberry Pi robot software so that:

1. **Robot core starts automatically on boot** with no monitor, keyboard, mouse, or login required.
2. **Computer vision processing runs headless** as boot-time `systemd` services.
3. **Remote desktop (`xrdp`) is used only for maintenance and viewing**, not as a dependency for robot startup.
4. When I connect remotely later, I can **open one GUI viewer/dashboard** that shows the robot's live camera/computer-vision outputs on my laptop screen.

## Current situation

### Old behavior
The old setup used:
- Ubuntu Desktop auto-login
- GNOME autostart
- `/home/nawat/Desktop/Robot/WallNet Detection/launch_terminals.sh`
- that script opened 4 GNOME terminals
- each terminal launched one robot-related Python program

Programs that were launched:
- `run_thermal67.py --camera 1 --fps 5.0 --show`
- `run_thermal67.py --camera 0 --fps 5.0 --show`
- `run_webcam.py --camera 1 --fps 5.0 --show`
- `run_orchestrator.py --glm-interval 10.0`

This old method depended on a logged-in GUI session and was not good for demo/presentation boot behavior.

### New direction
We migrated toward:
- `systemd` services for boot-time startup
- `xrdp` for remote maintenance/debugging
- optional GUI viewer as a separate program

### Important discovery
`xrdp` is a **separate desktop session**.

So we should **not** try to make the boot services open GUI windows and expect those same windows to appear in the xrdp desktop.

Instead:
- boot services do the real CV work **headlessly**
- viewer is a **separate GUI app** launched after remote login

## Final architecture we want

### A. Boot-time robot core
At boot, these should run as `systemd` services:
- thermal camera service 1
- thermal camera service 0
- webcam service 1
- orchestrator service

These services should:
- start automatically at boot
- not depend on GUI login
- not use `gnome-terminal`
- not use `cv2.imshow()` or local windows as part of required boot logic
- ideally restart if they fail
- write logs to journal/systemd logs
- optionally publish their latest processed frame to a shared folder for the viewer

### B. Remote GUI viewer
After I connect through `xrdp`, I want to manually run **one viewer program** that shows:
- thermal camera 1 live output
- thermal camera 0 live output
- webcam live output
- optional overlays / status / detections / orchestrator info

This viewer should:
- run only when I want to see visuals
- display already-produced frames/results from the boot services
- not directly own the cameras if the services already own them
- work fine inside the xrdp desktop

## What needs to be changed in the codebase

### 1. Split processing from display
Current code likely mixes these together.

We want this separation:

#### Processing side
`run_thermal67.py`, `run_webcam.py`, `run_orchestrator.py`

These should:
- run headless
- read cameras / do CV
- generate outputs
- save latest displayable frames somewhere accessible

#### Display side
A new `robot_viewer.py` should:
- read latest output frames/results
- display them in windows/dashboard
- be launched manually after xrdp login

### 2. Remove GUI dependency from boot services
For the versions of the scripts used in `systemd`:
- remove `--show` from service launches, or make `--show` optional and disabled in service mode
- do not rely on `cv2.imshow()`
- do not require a graphical session
- do not launch through `gnome-terminal`

The services must be safe to run with **no logged-in desktop**.

### 3. Publish latest frames from each running service
Each camera-processing service should periodically save its latest displayable frame to a shared directory.

Recommended shared directory:
- `/home/nawat/robot_frames/`

Suggested outputs:
- `/home/nawat/robot_frames/thermal_cam0.jpg`
- `/home/nawat/robot_frames/thermal_cam1.jpg`
- `/home/nawat/robot_frames/webcam_cam1.jpg`

The save should be atomic:
- write temp file first
- rename into place

That avoids the viewer reading a half-written image.

Suggested helper pattern inside the processing scripts:

```python
from pathlib import Path
import os
import cv2

FRAME_DIR = Path("/home/nawat/robot_frames")
FRAME_DIR.mkdir(parents=True, exist_ok=True)

def publish_frame(frame, name: str):
    tmp = FRAME_DIR / f".{name}.jpg"
    out = FRAME_DIR / f"{name}.jpg"
    cv2.imwrite(str(tmp), frame)
    os.replace(tmp, out)
```

Then call `publish_frame(...)` inside each script’s processing loop after producing a displayable frame.

### 4. Thermal scripts may need display conversion
If thermal frames are not already normal 8-bit display images, convert them before publishing.

For example:
- normalize raw 16-bit thermal data
- convert to 8-bit
- apply a color map if desired
- publish that display image

The viewer should consume a normal display-ready image, not raw thermal sensor format.

### 5. Add a separate viewer program
Create a new script such as:
- `/home/nawat/robot_service/robot_viewer.py`

Responsibilities:
- read the latest published image files
- lay them out in one window
- show placeholders when a feed is missing
- refresh continuously
- quit cleanly with a key like `q`

This viewer should be for display only.

It should **not** try to open `/dev/video*` directly if the services already own the cameras.

### 6. Service launch commands should be headless
Current `systemd` approach uses wrapper scripts.

That is good.

Keep using wrapper shell scripts for service startup, because paths contain spaces.

Service wrappers should launch headless versions of the scripts, for example:
- thermal cam 1
- thermal cam 0
- webcam cam 1
- orchestrator

No `gnome-terminal`, no display injection, no GUI required.

### 7. Camera selection should be robust
When real cameras are connected, camera numbering by index (`0`, `1`) may be unstable.

Potential improvement:
- allow scripts to take explicit device paths like `/dev/video0`, `/dev/video2`, etc.
- prefer explicit device paths in services once device mapping is known

This is especially important for boot reliability.

## Current service migration status

### Already done
- `xrdp` works
- old `robot-demo.desktop` 4-terminal autostart was disabled
- `systemd` service wrappers were created
- `systemd` unit files were created
- services were enabled and can launch through `systemd`
- `203/EXEC` was fixed by using wrapper scripts

### Known remaining issues
- camera services may fail when no cameras are connected
- camera index/device mapping still needs verification when hardware is attached
- orchestrator had a permission issue with an output directory
- leftover GNOME autostart test files may still exist and should be disabled if not needed

## Desired workflow after refactor

### Demo / presentation workflow
1. Power on Raspberry Pi
2. No login needed
3. `systemd` starts robot core automatically
4. Robot begins computer-vision processing headlessly
5. I connect from laptop using `xrdp`
6. I manually launch `robot_viewer.py`
7. Viewer shows live robot CV on my laptop screen

### Maintenance workflow
From xrdp terminal:
- check service status
- restart/stop services
- inspect logs
- edit code
- relaunch viewer if needed

## What the next LLM should change in the code

The next LLM should adjust the current codebase so that:

### Processing scripts
- `run_thermal67.py`
- `run_webcam.py`
- `run_orchestrator.py`

are compatible with headless `systemd` execution.

That means:
- no required GUI display dependency
- no required `imshow`
- optional debug display should be separated from core logic
- publish latest processed frames to `/home/nawat/robot_frames/`

### New viewer script
Create a new GUI script:
- `robot_viewer.py`

It should:
- display all published feeds in one dashboard window
- refresh continuously
- work inside xrdp
- handle missing frames gracefully

### Service behavior
Keep services headless and boot-safe.

Do not make the services depend on:
- GNOME login
- local monitor
- `xrdp`
- GUI windows

## Recommended implementation order

1. Refactor `run_thermal67.py` and `run_webcam.py` so they can run headless
2. Add frame publishing to those scripts
3. Make sure orchestrator works headlessly too
4. Fix output directory permissions for orchestrator
5. Create `robot_viewer.py`
6. Test:
   - boot without login
   - services start automatically
   - connect with xrdp
   - launch viewer
   - verify live feeds appear

## Acceptance criteria

The refactor is successful if:
1. Powering on the Pi starts robot CV automatically with no login
2. No GNOME terminals are required
3. Services run under `systemd`
4. I can connect later via `xrdp`
5. I can launch one viewer app and see live CV outputs
6. Viewer does not break camera access or conflict with services
7. Rebooting the Pi reproduces the setup reliably

## Important constraints

- Boot-time robot behavior must not depend on GUI login
- `xrdp` is only for remote viewing/debugging
- Viewer must be separate from core processing
- Services should own the actual camera capture
- Viewer should consume published frames/results, not steal camera devices
- Headless startup is more important than fancy windows

## Short version

The plan is:
- keep robot core as boot-time headless `systemd` services
- remove GUI dependency from those core scripts
- make the camera services publish latest processed frames
- create one separate GUI viewer app for xrdp
- use xrdp only to launch the viewer and manage the system
