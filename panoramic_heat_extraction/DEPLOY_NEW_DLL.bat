@echo off
echo ========================================
echo DEPLOYING NEW PURE THERMAL DLL FIX
echo ========================================
echo.

echo Killing all Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 >nul

echo Deleting old DLL...
del /F /Q "PureThermalBridge.dll" 2>nul
timeout /t 1 >nul

echo Copying new DLL (7:44 PM - FFC fix)...
copy /Y "purethermal_bridge\x64\Release\PureThermalBridge.dll" "PureThermalBridge.dll"

if exist "PureThermalBridge.dll" (
    echo.
    echo ========================================
    echo ✅ NEW DLL DEPLOYED SUCCESSFULLY
    echo ========================================
    echo.
    echo Starting GUI with thermal camera...
    echo.
    python gui.py --thermal 0 --source thermal
) else (
    echo.
    echo ❌ DLL COPY FAILED
    pause
)
