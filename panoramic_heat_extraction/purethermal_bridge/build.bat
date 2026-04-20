@echo off
echo ========================================
echo Building PureThermal Bridge DLL
echo ========================================
echo.

REM Find MSBuild
set MSBUILD=""
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD="C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
) else if exist "C:\Program Files\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD="C:\Program Files\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
) else (
    echo ERROR: MSBuild not found!
    echo Please install Visual Studio 2019 or 2022 with C++ tools
    pause
    exit /b 1
)

echo Found MSBuild: %MSBUILD%
echo.

REM Build the project
echo Building Release x64...
%MSBUILD% PureThermalBridge.vcxproj /p:Configuration=Release /p:Platform=x64 /v:m

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo BUILD SUCCESS!
    echo ========================================
    echo.
    
    REM Copy DLL to parent directory
    if exist "x64\Release\PureThermalBridge.dll" (
        copy /Y "x64\Release\PureThermalBridge.dll" ".."
        echo DLL copied to: %~dp0..\PureThermalBridge.dll
        echo.
        echo Next step: python purethermal_python.py
    ) else (
        echo WARNING: DLL not found at expected location
    )
) else (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo Check errors above
)

echo.
pause
