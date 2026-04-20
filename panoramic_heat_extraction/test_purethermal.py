"""Test Pure Thermal camera detection using uvctypes library."""

import sys

print("=" * 60)
print("Pure Thermal Detection Test")
print("=" * 60)

# Step 1: Check if libuvc.dll is available
print("\n[1/3] Checking for libuvc.dll...")
try:
    from ctypes import cdll
    libuvc = cdll.LoadLibrary("libuvc")
    print("✅ libuvc.dll found!")
except OSError as e:
    print(f"❌ libuvc.dll NOT found")
    print(f"   Error: {e}")
    print("\n" + "=" * 60)
    print("SOLUTION: Install libuvc.dll")
    print("=" * 60)
    print("\nDownload from: https://github.com/groupgets/libuvc/releases")
    print("\nSteps:")
    print("1. Download libuvc.dll for Windows")
    print("2. Place in C:\\Windows\\System32\\")
    print("   OR in same folder as this script")
    print("\nAlternative: Build from source")
    print("  git clone https://github.com/groupgets/libuvc")
    print("  cd libuvc")
    print("  mkdir build && cd build")
    print("  cmake ..")
    print("  cmake --build .")
    sys.exit(1)

# Step 2: Import uvctypes
print("\n[2/3] Loading uvctypes module...")
try:
    import uvctypes
    from uvctypes import *
    print("✅ uvctypes loaded successfully")
except Exception as e:
    print(f"❌ Failed to load uvctypes: {e}")
    sys.exit(1)

# Step 3: Try to find Pure Thermal device
print("\n[3/3] Scanning for Pure Thermal device...")
try:
    ctx = uvc_init()
    print("   - UVC context initialized")
    
    # Pure Thermal VID/PID
    PT_USB_VID = 0x1e4e  # GroupGets vendor ID
    PT_USB_PID = 0x0100  # PureThermal product ID (may vary)
    
    dev = uvc_find_device(ctx, PT_USB_VID, PT_USB_PID)
    
    if dev:
        print("✅ Pure Thermal FOUND!")
        print(f"   Vendor ID: 0x{PT_USB_VID:04x}")
        print(f"   Product ID: 0x{PT_USB_PID:04x}")
        
        # Try to open it
        devh = uvc_open(dev)
        if devh:
            print("✅ Successfully opened Pure Thermal")
            
            # Get format
            ctrl = uvc_stream_ctrl()
            res = uvc_get_stream_ctrl_format_size(
                devh, byref(ctrl),
                UVC_FRAME_FORMAT_Y16,  # 16-bit thermal
                160, 120, 9  # Lepton 2.5 resolution, 9 FPS
            )
            
            if res == 0:
                print("✅ Y16 format supported (160x120 @ 9fps)")
                print("\n" + "=" * 60)
                print("SUCCESS! Pure Thermal is ready to use.")
                print("=" * 60)
            else:
                print(f"⚠️  Could not configure format (error: {res})")
            
            uvc_close(devh)
        
        uvc_unref_device(dev)
    else:
        print("❌ Pure Thermal NOT FOUND")
        print("\nPossible reasons:")
        print("1. Device not connected via USB")
        print("2. Wrong VID/PID (try different product ID)")
        print("3. Drivers not installed")
        print("\nTry running: python uvc-deviceinfo.py")
        print("(from purethermal1-uvc-capture/python/)")
    
    uvc_exit(ctx)
    
except Exception as e:
    print(f"❌ Error during detection: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest complete.")
