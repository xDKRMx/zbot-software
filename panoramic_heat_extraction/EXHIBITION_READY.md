# 🎯 EXHIBITION SETUP - FLIR Lepton 2.5 Pure Thermal

## 🚀 Quick Start (Final Steps)

### Step 1: Build Pure Thermal Bridge DLL

```powershell
cd "c:\WallNet Detection\panoramic_heat_extraction\purethermal_bridge"
.\build.bat
```

**What this does:**
- Compiles C++ bridge using Windows Media Foundation
- Creates `PureThermalBridge.dll`
- Copies DLL to `panoramic_heat_extraction` folder

**Requirements:**
- Visual Studio 2019 or 2022 with C++ tools
- Windows 10 SDK (already installed with VS)

**Expected output:**
```
========================================
BUILD SUCCESS!
========================================
DLL copied to: c:\WallNet Detection\panoramic_heat_extraction\PureThermalBridge.dll
```

---

### Step 2: Test Pure Thermal Camera

```powershell
cd "c:\WallNet Detection\panoramic_heat_extraction"
python purethermal_python.py
```

**Expected output:**
```
Testing Pure Thermal Camera...
[PureThermal] Connected: 160x120
Frame captured: (120, 160) dtype=uint16
  Min: 28453
  Max: 32167
  Mean: 30234.5

✅ SUCCESS - Pure Thermal works!
```

---

### Step 3: Run Exhibition Demo

```powershell
cd "c:\WallNet Detection\panoramic_heat_extraction"
python gui.py --thermal 0
```

**What you'll see:**

**Console:**
```
[THERMAL] Initializing FLIR Lepton 2.5 Pure Thermal...
[THERMAL] Attempting native Pure Thermal bridge...
[PureThermal] Connected: 160x120
[THERMAL] ✅ SUCCESS - Pure Thermal connected via bridge DLL
[THERMAL] Resolution: 160x120 (FLIR Lepton 2.5)
[THERMAL] ✅ Ready for thermal capture
```

**GUI:**
- **Left Panel:** Live RGB camera (for navigation)
- **Right Panel:** Real-time thermal heat map from FLIR Lepton 2.5
- **Heat Map:** Shows actual temperature data (yellow/red = hot, blue = cold)

---

## 🎪 Exhibition Demo Flow

1. **Start Application:**
   ```
   python gui.py --thermal 0
   ```

2. **Select Thermal Source:**
   - Click "Thermal / IR camera" radio button
   - Click "▶ Start"

3. **Demonstrate Capabilities:**
   - Move robot across surface (wall, floor, equipment)
   - Heat map builds in real-time showing thermal patterns
   - Thermal data **persists** when revisiting areas (relocalization)
   - Show temperature hotspots in equipment
   - Demonstrate heat loss detection in walls

4. **Export Results:**
   - Click "⏹ Stop"
   - Click "💾 Export"
   - Saves panoramic thermal map as PNG

---

## 🔧 Technical Details

### Pure Thermal Bridge Architecture

```
FLIR Lepton 2.5 Hardware
         ↓
Pure Thermal USB Board
         ↓
Windows Media Foundation (MF)
         ↓
PureThermalBridge.dll (C++)
         ↓
purethermal_python.py (Python ctypes)
         ↓
gui.py (Panorama application)
```

### Data Flow

1. **Capture:** FLIR Lepton 2.5 → Y16 format (160×120, 16-bit unsigned)
2. **Transfer:** USB UVC → MediaFoundation → C++ DLL
3. **Expose:** C API → Python ctypes
4. **Process:** Normalize uint16 → uint8 grayscale
5. **Stitch:** Panorama stitcher with AKAZE features
6. **Render:** Apply JET colormap (thermal visualization)

### Thermal Data

- **Format:** Y16 (16-bit raw thermal)
- **Resolution:** 160×120 pixels (FLIR Lepton 2.5)
- **Frame Rate:** ~9 FPS
- **Temperature Range:** Kelvin × 100 (27315 = 0°C, 32315 = 50°C)
- **Precision:** 0.01°C per unit

---

## 🐛 Troubleshooting

### "Pure Thermal bridge not available"

**Cause:** DLL not built or not found

**Fix:**
```powershell
cd purethermal_bridge
.\build.bat
```

### "Pure Thermal device not found"

**Cause:** Device not connected or drivers missing

**Fix:**
1. Check USB connection (try different port)
2. Open Device Manager → Check under "Cameras"
3. Should see "Pure Thermal" or "FLIR Lepton"
4. If "Unknown Device" → Install drivers from GroupGets

### "Build failed" in build.bat

**Cause:** Visual Studio not installed or wrong version

**Fix:**
1. Install Visual Studio 2022 Community (free)
2. Select "Desktop development with C++"
3. Install Windows 10 SDK component
4. Re-run build.bat

### Thermal data looks wrong (all one color)

**Cause:** Static scene with low thermal variation

**Fix:**
- Point at heat source (hand, laptop, radiator)
- Ensure AGC is enabled on Lepton (automatic)
- Check temperature range is reasonable

---

## 📊 Performance Metrics

**Expected Performance:**
- **Capture Rate:** 9 FPS (Lepton limit)
- **Processing:** ~12-15 FPS (GUI refresh)
- **Stitching:** Real-time (< 100ms per frame)
- **Memory:** ~200MB (for 2000×2000 canvas)
- **CPU:** 15-25% (Intel i5 or better)

**Optimization:**
- Use `--capture-fps 5.0` to reduce CPU load
- Thermal camera runs at native 9 FPS regardless

---

## 🎬 Exhibition Script

**Opening:**
"This is Z-BOT's panoramic thermal mapping system using a FLIR Lepton 2.5 thermal camera. Watch as we scan this wall..."

**During Scan:**
"The yellow and red areas show heat signatures. Blue areas are cooler. Notice how the thermal data persists - when we return to an area we scanned earlier, the thermal information is still there. This is our relocalization system preventing data loss."

**Closing:**
"The entire thermal panorama is built in real-time. We can export this as a high-resolution thermal map for analysis, perfect for detecting heat loss, electrical faults, or equipment issues in network infrastructure."

---

## ✅ Pre-Exhibition Checklist

- [ ] Visual Studio installed with C++ tools
- [ ] PureThermalBridge.dll built successfully
- [ ] Pure Thermal device appears in Device Manager
- [ ] Test script (`purethermal_python.py`) runs successfully
- [ ] GUI launches and shows thermal data
- [ ] Heat map builds correctly when moving camera
- [ ] Export function saves PNG file
- [ ] RGB camera works for navigation view
- [ ] Tested with actual thermal sources (hand, laptop, etc.)
- [ ] Prepared demo surface with thermal variation

---

## 🎯 Success Criteria

**You're ready for exhibition when:**
1. ✅ `python purethermal_python.py` shows "SUCCESS - Pure Thermal works!"
2. ✅ `python gui.py --thermal 0` connects to FLIR Lepton
3. ✅ Heat map shows actual thermal data (not grayscale RGB)
4. ✅ Thermal data persists when revisiting areas
5. ✅ Export creates usable thermal panorama PNG

**This is production-ready for your exhibition!** 🚀
