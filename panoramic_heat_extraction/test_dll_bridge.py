"""Test Pure Thermal DLL bridge directly."""

import sys
from pathlib import Path

print("=" * 70)
print("Pure Thermal DLL Bridge Test")
print("=" * 70)

# Check if DLL exists
dll_path = Path("PureThermalBridge.dll")
if not dll_path.exists():
    print(f"\n❌ ERROR: PureThermalBridge.dll not found in current directory")
    print(f"   Expected: {dll_path.absolute()}")
    print(f"\n   Please build the DLL using:")
    print(f"   cd purethermal_bridge")
    print(f"   build.bat")
    sys.exit(1)

print(f"\n✅ DLL found: {dll_path.absolute()}")

# Try to load the Python wrapper
print("\n[1/2] Loading Python wrapper (purethermal_python.py)...")
try:
    from purethermal_python import PureThermalCamera
    print("✅ Python wrapper loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load wrapper: {e}")
    sys.exit(1)

# Try to connect to Pure Thermal
print("\n[2/2] Attempting to connect to Pure Thermal camera...")
try:
    pt_cam = PureThermalCamera()
    print("   - DLL initialized")
    
    if pt_cam.is_connected():
        print(f"\n✅✅✅ SUCCESS! Pure Thermal is connected!")
        print(f"   Resolution: {pt_cam.width}x{pt_cam.height}")
        print(f"   Format: FLIR Lepton 2.5 (Y16)")
        
        # Try to capture a frame
        print("\n[TEST] Capturing test frame...")
        ret, frame = pt_cam.read()
        if ret and frame is not None:
            print(f"✅ Frame captured successfully!")
            print(f"   Shape: {frame.shape}")
            print(f"   Dtype: {frame.dtype}")
            print(f"   Min: {frame.min()}, Max: {frame.max()}")
            
            # Temperature stats
            print(f"\n[THERMAL DATA]")
            print(f"   Raw min:  {frame.min()}")
            print(f"   Raw max:  {frame.max()}")
            print(f"   Raw mean: {frame.mean():.1f}")
            
            # Rough conversion (Lepton raw to Celsius)
            # Raw = (Kelvin * 100) / 100 = Kelvin
            # Celsius = Kelvin - 273.15
            celsius_min = (frame.min() / 100.0) - 273.15
            celsius_max = (frame.max() / 100.0) - 273.15
            celsius_mean = (frame.mean() / 100.0) - 273.15
            
            print(f"   Temp min:  {celsius_min:.1f}°C")
            print(f"   Temp max:  {celsius_max:.1f}°C")
            print(f"   Temp mean: {celsius_mean:.1f}°C")
            
            print("\n" + "=" * 70)
            print("✅ ALL TESTS PASSED - Pure Thermal is working correctly!")
            print("=" * 70)
            print("\nYou can now use the panoramic GUI or unified_runner.")
        else:
            print(f"❌ Failed to capture frame")
            print(f"   The camera is connected but cannot read data.")
    else:
        print(f"\n❌ Pure Thermal NOT connected")
        print(f"\nPossible reasons:")
        print(f"1. USB cable not plugged in")
        print(f"2. Pure Thermal board not powered")
        print(f"3. Device in use by another application")
        print(f"4. Windows driver issue")
        print(f"\nTroubleshooting:")
        print(f"- Check Device Manager → Cameras → 'PureThermal'")
        print(f"- Unplug and replug USB cable")
        print(f"- Close any other apps using the camera")
        print(f"- Restart the Pure Thermal board")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest complete.")
