# 🔨 Build Pure Thermal Bridge for FLIR Lepton 2.5

## Overview

This creates a C++ DLL that Python can use to capture thermal frames from Pure Thermal board.

**What it does:**
- Uses Windows Media Foundation (already on Windows 10/11)
- Captures Y16 raw thermal data (160×120 uint16)
- Exposes simple C API for Python ctypes
- **No libuvc.dll needed** - native Windows solution

---

## ⚙️ Prerequisites

### 1. Install Visual Studio 2019 or 2022

Download: https://visualstudio.microsoft.com/downloads/

**Required components:**
- Desktop development with C++
- Windows 10 SDK (10.0.19041.0 or later)
- MSBuild tools

### 2. Verify Installation

```powershell
# Check if MSBuild is available
where msbuild

# Should show path like:
# C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe
```

---

## 🏗️ Build Steps

### Option A: Visual Studio GUI (Recommended)

1. **Open Solution:**
   ```
   Navigate to: c:\WallNet Detection\panoramic_heat_extraction\purethermal1-uvc-capture\mediafoundation\PureThermal
   Double-click: PureThermal.sln
   ```

2. **Add Bridge Files:**
   - Right-click solution → Add → Existing Item
   - Add: `c:\WallNet Detection\panoramic_heat_extraction\purethermal_bridge\*.h`
   - Add: `c:\WallNet Detection\panoramic_heat_extraction\purethermal_bridge\*.cpp`

3. **Change Output Type:**
   - Right-click project → Properties
   - Configuration Type: Dynamic Library (.dll)
   - Apply

4. **Build:**
   - Build → Build Solution (or press F7)
   - **Output:** `x64\Release\PureThermalBridge.dll`

5. **Copy DLL:**
   ```powershell
   copy "x64\Release\PureThermalBridge.dll" "c:\WallNet Detection\panoramic_heat_extraction\"
   ```

### Option B: Command Line (Faster)

```powershell
# Navigate to bridge directory
cd "c:\WallNet Detection\panoramic_heat_extraction\purethermal_bridge"

# Build using MSBuild
msbuild PureThermalBridge.sln /p:Configuration=Release /p:Platform=x64

# Copy output
copy "x64\Release\PureThermalBridge.dll" ".."
```

---

## ✅ Test Build

After building, test the DLL:

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

## 🔧 Alternative: Pre-Build Script

Create `build_bridge.bat`:

```batch
@echo off
echo Building PureThermal Bridge...

set MSBUILD="C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
set PROJECT="c:\WallNet Detection\panoramic_heat_extraction\purethermal_bridge\PureThermalBridge.sln"

%MSBUILD% %PROJECT% /p:Configuration=Release /p:Platform=x64

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    copy "x64\Release\PureThermalBridge.dll" "c:\WallNet Detection\panoramic_heat_extraction\"
    echo DLL copied to panoramic_heat_extraction folder
) else (
    echo Build failed!
    pause
)
```

Run: `build_bridge.bat`

---

## 📦 Creating .vcxproj File

If you need a proper Visual Studio project:

**Create:** `purethermal_bridge\PureThermalBridge.vcxproj`

```xml
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{YOUR-GUID-HERE}</ProjectGuid>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Label="Configuration">
    <ConfigurationType>DynamicLibrary</ConfigurationType>
    <PlatformToolset>v142</PlatformToolset>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ItemGroup>
    <ClCompile Include="PureThermalBridge.cpp" />
    <ClCompile Include="pch.cpp">
      <PrecompiledHeader>Create</PrecompiledHeader>
    </ClCompile>
    <ClCompile Include="..\purethermal1-uvc-capture\mediafoundation\PureThermal\Device.cpp" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="PureThermalBridge.h" />
    <ClInclude Include="pch.h" />
    <ClInclude Include="..\purethermal1-uvc-capture\mediafoundation\PureThermal\Device.h" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
</Project>
```

---

## 🚨 Troubleshooting

### Error: "Cannot find Device.h"

**Fix:** Copy Device.cpp and Device.h to purethermal_bridge folder:
```powershell
copy "purethermal1-uvc-capture\mediafoundation\PureThermal\Device.*" "purethermal_bridge\"
```

### Error: "mfapi.h not found"

**Fix:** Install Windows 10 SDK via Visual Studio Installer

### Error: "Pure Thermal device not found"

**Check:**
1. Pure Thermal is connected via USB
2. Device appears in Device Manager under "Cameras"
3. Try different USB port
4. Restart device

---

## 📝 Next Steps

After successful build:

1. **Test DLL:** `python purethermal_python.py`
2. **Run GUI:** `python gui.py --thermal 0`
3. **Exhibition ready!** 🎉
