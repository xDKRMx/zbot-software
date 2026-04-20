# Force deploy new DLL
Write-Host "Killing all Python processes..." -ForegroundColor Yellow
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "Deleting old DLL..." -ForegroundColor Yellow
Remove-Item "PureThermalBridge.dll" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Write-Host "Copying new DLL..." -ForegroundColor Cyan
Copy-Item "purethermal_bridge\x64\Release\PureThermalBridge.dll" "PureThermalBridge.dll" -Force

if (Test-Path "PureThermalBridge.dll") {
    $dll = Get-Item "PureThermalBridge.dll"
    Write-Host "✅ NEW DLL DEPLOYED: $($dll.LastWriteTime)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Starting GUI..." -ForegroundColor Cyan
    python gui.py --thermal 0 --source thermal
} else {
    Write-Host "❌ DLL copy failed" -ForegroundColor Red
}
