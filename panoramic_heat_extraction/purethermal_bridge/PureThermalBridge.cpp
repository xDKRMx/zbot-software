#include "pch.h"
#include "PureThermalBridge.h"
#include "../purethermal1-uvc-capture/mediafoundation/PureThermal/Device.h"
#include <memory>
#include <string>

static std::unique_ptr<Device> g_device;
static std::string g_lastError;

extern "C" {

PURETHERMAL_API int PT_Initialize()
{
    try
    {
        auto hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (FAILED(hr))
        {
            g_lastError = "CoInitializeEx failed";
            return 0;
        }

        hr = MFStartup(MF_VERSION);
        if (FAILED(hr))
        {
            CoUninitialize();
            g_lastError = "MFStartup failed";
            return 0;
        }

        g_device = std::make_unique<Device>();
        if (!(*g_device))
        {
            g_lastError = "Pure Thermal device not found";
            MFShutdown();
            CoUninitialize();
            g_device.reset();
            return 0;
        }

        g_lastError = "Success";
        return 1;
    }
    catch (const std::exception& e)
    {
        g_lastError = e.what();
        return 0;
    }
}

PURETHERMAL_API int PT_IsConnected()
{
    return (g_device && (*g_device)) ? 1 : 0;
}

PURETHERMAL_API int PT_GetFrameWidth()
{
    return 160; // FLIR Lepton 2.5
}

PURETHERMAL_API int PT_GetFrameHeight()
{
    return 120; // FLIR Lepton 2.5
}

PURETHERMAL_API int PT_CaptureFrame(unsigned short* buffer, int bufferSize)
{
    if (!g_device || !(*g_device))
    {
        g_lastError = "Device not initialized";
        return 0;
    }

    try
    {
        auto frame = g_device->GetFrame();
        if (frame.data.empty())
        {
            g_lastError = "Failed to capture frame";
            return 0;
        }

        const int expectedSize = static_cast<int>(frame.width * frame.height);
        if (bufferSize < expectedSize)
        {
            g_lastError = "Buffer too small";
            return 0;
        }

        // Copy frame data to buffer
        std::memcpy(buffer, frame.data.data(), expectedSize * sizeof(unsigned short));

        return expectedSize;
    }
    catch (const std::exception& e)
    {
        g_lastError = e.what();
        return 0;
    }
}

PURETHERMAL_API void PT_Shutdown()
{
    if (g_device)
    {
        g_device.reset();
    }
    MFShutdown();
    CoUninitialize();
}

PURETHERMAL_API const char* PT_GetLastError()
{
    return g_lastError.c_str();
}

PURETHERMAL_API void PT_PerformFFC()
{
    if (g_device && (*g_device))
    {
        try
        {
            g_device->PerformFFC();
            g_lastError = "FFC triggered successfully";
        }
        catch (const std::exception& e)
        {
            g_lastError = std::string("FFC failed: ") + e.what();
        }
    }
    else
    {
        g_lastError = "Cannot perform FFC: device not initialized";
    }
}

} // extern "C"
