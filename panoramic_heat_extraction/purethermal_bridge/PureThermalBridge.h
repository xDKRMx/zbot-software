#pragma once

#ifdef PURETHERMAL_BRIDGE_EXPORTS
#define PURETHERMAL_API __declspec(dllexport)
#else
#define PURETHERMAL_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Simple C API for Python ctypes
PURETHERMAL_API int PT_Initialize();
PURETHERMAL_API int PT_IsConnected();
PURETHERMAL_API int PT_GetFrameWidth();
PURETHERMAL_API int PT_GetFrameHeight();
PURETHERMAL_API int PT_CaptureFrame(unsigned short* buffer, int bufferSize);
PURETHERMAL_API void PT_Shutdown();
PURETHERMAL_API const char* PT_GetLastError();

// Perform Flat Field Correction (FFC) - Lepton camera calibration
PURETHERMAL_API void PT_PerformFFC();

#ifdef __cplusplus
}
#endif
