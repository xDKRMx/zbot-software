"""Python interface to PureThermal C++ bridge DLL."""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional

class PureThermalCamera:
    """Python wrapper for FLIR Lepton 2.5 via PureThermal board."""
    
    def __init__(self, dll_path: str = "PureThermalBridge.dll"):
        """Initialize connection to Pure Thermal DLL.
        
        Args:
            dll_path: Path to PureThermalBridge.dll (default: same directory)
        """
        self._dll = None
        self._width = 160
        self._height = 120
        self._connected = False
        
        # Try FIXED DLL first (7:44 PM build with FFC frame copy fix)
        base_dir = Path(__file__).parent
        fixed_dll = base_dir / "PureThermalBridge_FIXED.dll"
        
        if fixed_dll.exists():
            dll_file = fixed_dll
            print(f"[PureThermal] Using FIXED DLL (FFC bug fix): {fixed_dll.name}")
        else:
            # Fall back to specified DLL path
            if Path(dll_path).is_absolute():
                dll_file = Path(dll_path)
            else:
                # Try current directory first
                dll_file = Path(__file__).parent / dll_path
                if not dll_file.exists():
                    # Try relative to cwd
                    dll_file = Path(dll_path).absolute()
        
        if not dll_file.exists():
            raise FileNotFoundError(
                f"PureThermalBridge.dll not found at {dll_path}\n"
                f"Please compile the C++ bridge first (see BUILD_INSTRUCTIONS.md)"
            )
        
        try:
            self._dll = ctypes.CDLL(str(dll_file))
        except Exception as e:
            raise RuntimeError(f"Failed to load PureThermalBridge.dll: {e}")
        
        # Define function signatures
        self._dll.PT_Initialize.argtypes = []
        self._dll.PT_Initialize.restype = ctypes.c_int
        
        self._dll.PT_IsConnected.argtypes = []
        self._dll.PT_IsConnected.restype = ctypes.c_int
        
        self._dll.PT_GetFrameWidth.argtypes = []
        self._dll.PT_GetFrameWidth.restype = ctypes.c_int
        
        self._dll.PT_GetFrameHeight.argtypes = []
        self._dll.PT_GetFrameHeight.restype = ctypes.c_int
        
        self._dll.PT_CaptureFrame.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.c_int]
        self._dll.PT_CaptureFrame.restype = ctypes.c_int
        
        self._dll.PT_Shutdown.argtypes = []
        self._dll.PT_Shutdown.restype = None
        
        self._dll.PT_GetLastError.argtypes = []
        self._dll.PT_GetLastError.restype = ctypes.c_char_p
        
        # FFC (Flat Field Correction) - optional, may not exist in older DLL
        self._has_ffc = False
        try:
            self._dll.PT_PerformFFC.argtypes = []
            self._dll.PT_PerformFFC.restype = None
            self._has_ffc = True
        except AttributeError:
            # FFC not available in this DLL version
            pass
        
        # Initialize device
        if not self._dll.PT_Initialize():
            error = self.get_last_error()
            raise RuntimeError(f"Failed to initialize Pure Thermal: {error}")
        
        self._connected = True
        self._width = self._dll.PT_GetFrameWidth()
        self._height = self._dll.PT_GetFrameHeight()
        
        print(f"[PureThermal] Connected: {self._width}x{self._height}")
    
    def is_connected(self) -> bool:
        """Check if Pure Thermal is connected."""
        if not self._dll:
            return False
        return self._dll.PT_IsConnected() == 1
    
    def get_last_error(self) -> str:
        """Get last error message from the DLL."""
        if not self._dll:
            return "DLL not loaded"
        error = self._dll.PT_GetLastError()
        return error.decode('utf-8') if error else "No error"
    
    def perform_ffc(self) -> bool:
        """Trigger Flat Field Correction (FFC) - Lepton camera calibration.
        
        FFC should be performed:
        - On startup (camera warmup)
        - When thermal drift is suspected
        - Periodically during long capture sessions
        
        Returns:
            True if FFC was performed, False if not available
        """
        if not self._dll or not self._connected:
            raise RuntimeError("Device not connected")
        
        if not self._has_ffc:
            print("[PureThermal] ⚠️  FFC not available in this DLL version")
            return False
        
        print("[PureThermal] Triggering FFC (Flat Field Correction)...")
        self._dll.PT_PerformFFC()
        # FFC takes ~1-2 seconds on Lepton
        import time
        time.sleep(2.0)
        print("[PureThermal] FFC complete")
        return True
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Capture a frame from Pure Thermal.
        
        Returns:
            (success, frame) where frame is 160x120 uint16 numpy array
        """
        if not self._connected or not self.is_connected():
            return (False, None)
        
        # Allocate buffer for frame
        buffer_size = self._width * self._height
        buffer = (ctypes.c_uint16 * buffer_size)()
        
        # Capture frame
        result = self._dll.PT_CaptureFrame(buffer, buffer_size)
        if result <= 0:
            error = self.get_last_error()
            print(f"[PureThermal] Capture failed: {error}")
            return (False, None)
        
        # Convert to numpy array
        frame = np.frombuffer(buffer, dtype=np.uint16).reshape((self._height, self._width))
        return (True, frame.copy())
    
    def release(self):
        """Release Pure Thermal device."""
        if self._dll and self._connected:
            self._dll.PT_Shutdown()
            self._connected = False
            print("[PureThermal] Disconnected")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.release()
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height


# Test function
if __name__ == "__main__":
    print("Testing Pure Thermal Camera...")
    
    try:
        camera = PureThermalCamera()
        print(f"Camera: {camera.width}x{camera.height}")
        
        # Capture test frame
        success, frame = camera.read()
        if success:
            print(f"Frame captured: {frame.shape} dtype={frame.dtype}")
            print(f"  Min: {frame.min()}")
            print(f"  Max: {frame.max()}")
            print(f"  Mean: {frame.mean():.1f}")
            print("\n✅ SUCCESS - Pure Thermal works!")
        else:
            print("❌ FAILED - Could not capture frame")
        
        camera.release()
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
