# ⚡ QUICK START - Test Panorama Heat Mapping NOW

## 🚀 Run This Command Immediately:

```bash
cd "c:\WallNet Detection\panoramic_heat_extraction"
python gui.py --rgb 0
```

**This will:**
- ✅ Open your RGB camera (webcam)
- ✅ Show live camera feed on left panel
- ✅ Build panoramic heat map on right panel in REAL-TIME
- ✅ Test the entire relocalization system (position persistence)
- ✅ Prove the heat mapping algorithm works

**Expected behavior:**
1. Auto-scan tries to find Pure Thermal (10 seconds max)
2. Falls back to RGB camera
3. GUI opens with live feed + heat map
4. As you move camera, heat map grows
5. Heat data PERSISTS when you return to previous positions

---

## ❌ Pure Thermal Setup is SEPARATE

**Pure Thermal requires complex setup:**
- Native C++ library (libuvc.dll)
- Windows 10 SDK + Visual Studio
- MediaFoundation compilation

**This is NOT needed to test panorama!**

---

## 📊 What You'll See

**Left Panel:** Live RGB camera  
**Right Panel:** Growing thermal heat map (yellow/red colors)

**Move the camera** across a surface (wall, desk, floor) and watch the heat map build up. The new relocalization system ensures thermal data doesn't disappear when you revisit areas.

---

## ✅ Success Criteria

If you see:
- Live camera feed updating
- Heat map growing as you move
- No crashes or freezes

**Then the panoramic heat mapping system WORKS!** ✅

Pure Thermal integration is a separate hardware driver issue, not a software bug.

---

## 🔧 Pure Thermal (Optional - For Later)

**For real thermal data from FLIR Lepton:**

### Option 1: Use MediaFoundation C++ Example (Recommended for Windows)
```
cd "c:\WallNet Detection\panoramic_heat_extraction\purethermal1-uvc-capture\mediafoundation\PureThermal"
# Open .sln file in Visual Studio
# Compile and run
```

### Option 2: Build libuvc.dll from source
```
# Requires CMake, Visual Studio Build Tools, libusb
git clone https://github.com/groupgets/libuvc
cd libuvc
mkdir build && cd build
cmake -G "Visual Studio 16 2019" ..
cmake --build . --config Release
# Copy libuvc.dll to System32 or panoramic_heat_extraction folder
```

### Option 3: Find pre-built libuvc.dll binary
Search for "libuvc.dll windows binary" or check GroupGets forums.

**All of this is OPTIONAL and can be done later.**

---

## 🎯 Bottom Line

**RIGHT NOW:**
```bash
python gui.py --rgb 0
```

**Test the panorama system works. Pure Thermal is a bonus feature for later.**
