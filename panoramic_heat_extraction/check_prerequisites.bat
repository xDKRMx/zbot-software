@echo off
echo ================================================
echo Pure Thermal - Onkoşul Kontrol / Prerequisite Check
echo ================================================
echo.

REM Check 1: Visual Studio / MSBuild
echo [1/4] Visual Studio kontrolu...
set MSBUILD_FOUND=0

if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD_FOUND=1
    echo ✅ Visual Studio 2022 Community bulundu
    echo    Konum: C:\Program Files\Microsoft Visual Studio\2022\Community
) else if exist "C:\Program Files\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD_FOUND=1
    echo ✅ Visual Studio 2019 Community bulundu
    echo    Konum: C:\Program Files\Microsoft Visual Studio\2019\Community
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD_FOUND=1
    echo ✅ Visual Studio 2019 Community bulundu (x86)
) else (
    echo ❌ Visual Studio BULUNAMADI!
    echo.
    echo İndirmek için:
    echo https://visualstudio.microsoft.com/downloads/
    echo.
    echo "Community 2022" sürümünü indir (ÜCRETSİZ)
    echo Kurulumda "Desktop development with C++" seçeneğini işaretle
)
echo.

REM Check 2: Python
echo [2/4] Python kontrolu...
python --version 2>NUL
if %ERRORLEVEL% EQU 0 (
    echo ✅ Python kurulu
    python --version
) else (
    echo ❌ Python BULUNAMADI!
    echo Python 3.8 veya üstü gerekli
)
echo.

REM Check 3: Required files
echo [3/4] Gerekli dosyalar kontrolu...
set FILES_OK=1

if exist "purethermal_bridge\PureThermalBridge.cpp" (
    echo ✅ PureThermalBridge.cpp mevcut
) else (
    echo ❌ PureThermalBridge.cpp BULUNAMADI!
    set FILES_OK=0
)

if exist "purethermal_bridge\build.bat" (
    echo ✅ build.bat mevcut
) else (
    echo ❌ build.bat BULUNAMADI!
    set FILES_OK=0
)

if exist "purethermal_python.py" (
    echo ✅ purethermal_python.py mevcut
) else (
    echo ❌ purethermal_python.py BULUNAMADI!
    set FILES_OK=0
)
echo.

REM Check 4: Pure Thermal device
echo [4/4] Pure Thermal cihaz kontrolu...
echo Cihaz Manager'da kontrol etmek için:
echo Win+X tuşlarına basın → Device Manager → Cameras
echo "Pure Thermal" veya "FLIR Lepton" görünmeli
echo.

REM Summary
echo ================================================
echo ÖZET / SUMMARY
echo ================================================

if %MSBUILD_FOUND% EQU 1 if %FILES_OK% EQU 1 (
    echo.
    echo ✅ ÖNKOŞULLAr HAZIR!
    echo.
    echo Sıradaki adım: DLL derlemek
    echo.
    echo Komut:
    echo   cd purethermal_bridge
    echo   .\build.bat
    echo.
) else (
    echo.
    echo ❌ EKSIK ÖNKOŞULLAR VAR
    echo.
    if %MSBUILD_FOUND% EQU 0 (
        echo 1. Visual Studio 2022 Community indir ve kur:
        echo    https://visualstudio.microsoft.com/downloads/
        echo.
        echo    Kurulumda şunu seç: "Desktop development with C++"
        echo.
    )
    if %FILES_OK% EQU 0 (
        echo 2. Gerekli dosyalar eksik
        echo    Tüm dosyaların doğru klasörde olduğundan emin ol
        echo.
    )
)

echo ================================================
echo.
pause
