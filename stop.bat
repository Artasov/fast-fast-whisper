@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PID_FILE=.fast-fast-whisper.pid"

if not exist "%PID_FILE%" (
    echo [INFO] PID file not found. Server is not running.
    exit /b 0
)

set "TARGET_PID="
for /f %%P in ('powershell -NoProfile -Command "(Get-Content -Path '%PID_FILE%' -Raw) -replace '[^0-9]',''"') do set "TARGET_PID=%%P"

if not defined TARGET_PID (
    echo [WARN] PID file does not contain a valid number. Removing it.
    del "%PID_FILE%"
    exit /b 1
)

powershell -NoProfile -Command "if (Get-Process -Id %TARGET_PID% -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Process %TARGET_PID% not found. Removing stale PID file.
    del "%PID_FILE%"
    exit /b 0
)

echo [INFO] Stopping process %TARGET_PID%...
taskkill /PID %TARGET_PID% /T /F >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to stop process %TARGET_PID%.
    exit /b 1
)

del "%PID_FILE%"
echo [OK] Process %TARGET_PID% stopped successfully.
exit /b 0
