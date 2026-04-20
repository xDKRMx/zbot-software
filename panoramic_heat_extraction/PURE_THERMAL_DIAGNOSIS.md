# Pure Thermal Camera Diagnosis & Fix

## Current Issue
Pure Thermal FLIR Lepton 2.5 **cannot be accessed via OpenCV VideoCapture** on Windows.

**Error:** `-1072875772` (Windows Media Foundation failure)  
**Root Cause:** Pure Thermal board is NOT enumerated as standard UVC camera by Windows.

---

## ✅ IMMEDIATE WORKAROUND - Test Heat Mapping NOW

**Use RGB camera as thermal source** to test panoramic heat mapping:

```bash
cd "c:\WallNet Detection\panoramic_heat_extraction"
python gui.py --rgb 0 --thermal -1 --source rgb
```

This will:
- ✅ Open RGB camera
- ✅ Generate heat map from RGB grayscale
- ✅ Test panoramic stitching with relocalization
- ✅ Show real-time heat map building

**You can verify the entire panorama system works while we fix Pure Thermal access.**

---

## Pure Thermal Requirements (For Real Thermal Data)

### 1. Check Device Manager

**Open Device Manager (Win+X → Device Manager):**

Pure Thermal should appear as:
- ✅ **"Pure Thermal"** or **"FLIR Lepton"** under **"Cameras"** or **"Imaging Devices"**

If you see:
- ❌ **"Unknown Device"** → Drivers NOT installed
- ❌ **Nothing** → USB connection issue
- ❌ **"USB Composite Device"** → Generic driver (won't work with OpenCV)

### 2. Install Pure Thermal Drivers

**GroupGets Pure Thermal Board needs specific drivers:**

Download from: https://groupgets.com/manufacturers/getlab/products/purethermal-2-flir-lepton-smart-i-o-module

**Driver Installation:**
1. Download PureThermal SDK
2. Run driver installer for Windows
3. Reboot system
4. Verify in Device Manager

### 3. Python Library Requirements

**OpenCV CANNOT directly access Pure Thermal**. Need specialized library:

```bash
# Install uvctypes (Pure Thermal Python library)
pip install uvctypes

# Or use libuvc-python
pip install pyuvc
```

**Alternative: PureThermal Python SDK**
```bash
git clone https://github.com/groupgets/purethermal1-uvc-capture
cd purethermal1-uvc-capture/python
pip install -r requirements.txt
```

### 4. Test Pure Thermal Access

**After installing drivers and SDK, test with:**

```python
from uvctypes import *
import cv2
import numpy as np

def find_purethermal():
    ctx = uvc_init()
    dev_list = uvc_get_device_list(ctx)
    
    dev = uvc_find_device(ctx, PT_USB_VID, PT_USB_PID_LEPTON)
    if dev:
        print("✅ Pure Thermal found!")
        print(f"   Vendor: {uvc_get_vendor(dev)}")
        print(f"   Product: {uvc_get_product(dev)}")
    else:
        print("❌ Pure Thermal not found")
    
    uvc_unref_device_list(dev_list)
    uvc_exit(ctx)

find_purethermal()
```

---

## Integration Steps (After Pure Thermal Works)

Once Pure Thermal is accessible via uvctypes/pyuvc:

1. **Modify `gui.py`** to use uvctypes instead of `cv2.VideoCapture`
2. **Capture frames** via UVC library
3. **Convert** raw thermal data to numpy array
4. **Feed** to existing panorama stitcher

**Example thermal capture:**
```python
import ctypes
from uvctypes import *

def capture_thermal_frame():
    ctx = uvc_init()
    dev = uvc_find_device(ctx, PT_USB_VID, PT_USB_PID_LEPTON)
    devh = uvc_open(dev)
    
    frame_ptr = POINTER(uvc_frame)()
    res = uvc_stream_get_frame(devh, byref(frame_ptr), 1000000)  # 1 sec timeout
    
    if res == 0:
        # Convert to numpy
        data = ctypes.string_at(frame_ptr.contents.data, 
                                frame_ptr.contents.data_bytes)
        thermal = np.frombuffer(data, dtype=np.uint16).reshape(120, 160)
        return thermal
    
    uvc_close(devh)
    uvc_unref_device(dev)
    uvc_exit(ctx)
    return None
```

---

## Quick Diagnostic Checklist

Run these checks:

```powershell
# 1. Check USB devices
Get-PnPDevice | Where-Object {$_.FriendlyName -like "*thermal*" -or $_.FriendlyName -like "*lepton*"}

# 2. Check if libuvc DLL exists
dir "C:\Windows\System32\*uvc*.dll"

# 3. List all video devices
python -c "import cv2; print([cv2.VideoCapture(i).getBackendName() for i in range(5)])"
```

---

## Summary

**Current State:**
- ❌ Pure Thermal NOT accessible via standard OpenCV
- ✅ RGB camera works
- ✅ Panoramic heat mapping algorithm works (tested with RGB)

**Next Steps:**
1. **Test NOW:** Use RGB as thermal source (`--thermal -1 --source rgb`)
2. **Install:** Pure Thermal drivers + uvctypes library
3. **Verify:** Device appears in Device Manager
4. **Integrate:** Replace OpenCV capture with uvctypes in code

**Timeline:**
- Immediate: RGB-based heat mapping (works now)
- Short-term: Pure Thermal driver installation (user action required)
- Medium-term: Code integration with uvctypes library (1-2 hours dev work)
