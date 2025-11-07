@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_DIR=%CD%"
set "PYTHONHOME="
set "PYTHONPATH="

set "PY_VERSION=3.12.5"
set "PY_ROOT=%PROJECT_DIR%\python_portable"
set "PY_EXE=%PY_ROOT%\python.exe"
set "VENV_DIR=%PROJECT_DIR%\.venv-win"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "DEFAULT_PORT=8868"
set "APP_PORT="
if defined FAST_FAST_WHISPER_PORT set "APP_PORT=%FAST_FAST_WHISPER_PORT%"
if not defined APP_PORT if defined PORT set "APP_PORT=%PORT%"
if not defined APP_PORT set "APP_PORT=%DEFAULT_PORT%"
set "PID_FILE=%PROJECT_DIR%\.fast-fast-whisper.pid"
set "LOG_FILE=%PROJECT_DIR%\fast-fast-whisper.log"
set "PAUSE_SECONDS=5"
set "PS_CAPTURE=%TEMP%\ffw_start_ps.out"
set "EXIT_CODE=0"

call :check_existing_instance
if errorlevel 1 (
    set "EXIT_CODE=1"
    goto final_exit
)

echo ---------------------------------
echo [INFO] Start installation

echo [INFO] Enabling PowerShell user scripts (RemoteSigned) for CurrentUser...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force" >nul 2>nul

if exist "%PY_EXE%" goto venv

echo ---------------------------------
echo [INFO] Portable Python not found — downloading...
echo ---------------------------------

set "PKG_URL=https://www.nuget.org/api/v2/package/python/%PY_VERSION%"
set "TMPNUP=%TEMP%\python_portable_%PY_VERSION%.nupkg"
set "TMPZIP=%TEMP%\python_portable_%PY_VERSION%.zip"

where curl >nul 2>nul
if "%ERRORLEVEL%"=="0" (
    curl -L -o "%TMPNUP%" "%PKG_URL%"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%PKG_URL%' -OutFile '%TMPNUP%'"
)

if not exist "%TMPNUP%" (
    echo ---------------------------------
    echo [ERROR] Failed to download file %TMPNUP%
    echo ---------------------------------

    pause
    set "EXIT_CODE=1"
    goto final_exit
)

echo ---------------------------------
echo [INFO] Renaming "%TMPNUP%" → "%TMPZIP%"

ren "%TMPNUP%" "python_portable_%PY_VERSION%.zip"
if not exist "%TMPZIP%" (
    REM file might have been renamed elsewhere, check current Temp folder

    echo ---------------------------------
    echo [WARN] Expected file %TMPZIP% in TEMP, but it's not there
    echo ---------------------------------

    dir /b "%TEMP%\*.zip"
    pause
    set "EXIT_CODE=1"
    goto final_exit
)

echo ---------------------------------
echo [INFO] Extracting archive %TMPZIP%

powershell -NoProfile -Command "Expand-Archive -Path '%TMPZIP%' -DestinationPath 'python_extracted'"
if not exist "python_extracted\tools\python.exe" (
    echo [ERROR] tools\python.exe not found in extracted archive
    pause
    set "EXIT_CODE=1"
    goto final_exit
)

echo ---------------------------------
echo [INFO] Copying tools → %PY_ROOT%
echo ---------------------------------

xcopy /e /i /y "python_extracted\tools" "%PY_ROOT%"
rd /s /q "python_extracted"
del /q "%TMPZIP%"

if not exist "%PY_EXE%" (
    echo ---------------------------------
    echo [ERROR] after copying %PY_EXE% is missing
    echo ---------------------------------
    pause
    set "EXIT_CODE=1"
    goto final_exit
)

echo ---------------------------------
echo [OK] Portable Python ready: %PY_EXE%

:venv
if not exist "%VENV_PY%" (
    echo ---------------------------------
    echo [INFO] Creating venv...
    echo ---------------------------------
    "%PY_EXE%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ---------------------------------
        echo [ERROR] Failed to create venv
        echo ---------------------------------
        pause
        set "EXIT_CODE=1"
        goto final_exit
    )
) else (
    echo ---------------------------------
    echo [OK] venv already exists
    echo ---------------------------------
)

echo [INFO] Updating pip...
"%VENV_PY%" -m pip install --upgrade pip

if exist "requirements.txt" (
    echo ---------------------------------
    echo [INFO] Installing dependencies...
    echo ---------------------------------
    "%VENV_PY%" -m pip install -r requirements.txt
)
echo ---------------------------------
echo [RUN] Starting uvicorn on port %APP_PORT% in background (logs: %LOG_FILE%)
echo [INFO] Override port via FAST_FAST_WHISPER_PORT or PORT variables
echo ---------------------------------


powershell -NoProfile -Command "New-Item -Path '%LOG_FILE%' -ItemType File -Force | Out-Null" >nul 2>&1
if exist "%PS_CAPTURE%" del "%PS_CAPTURE%" >nul 2>&1

powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; $log = Join-Path (Resolve-Path .) '%LOG_FILE%'; $pwd = Resolve-Path .; $python = '%VENV_PY%'; $pidFile = '%PID_FILE%'; $args = @('-m','uvicorn','main:app','--host','0.0.0.0','--port','%APP_PORT%'); $cmdLine = '"' + $python + '" ' + ($args -join ' '); $full = '/c \"' + $cmdLine + ' >> \"' + $log + '\" 2>&1\"'; $p = Start-Process -FilePath 'cmd.exe' -ArgumentList $full -WorkingDirectory $pwd -WindowStyle Hidden -PassThru; Set-Content -Path $pidFile -Value ($p.Id.ToString()) -Encoding ascii; $p.Id" >"%PS_CAPTURE%" 2>&1

set "SERVER_PID="
if exist "%PS_CAPTURE%" (
    for /f "usebackq delims=" %%P in ("%PS_CAPTURE%") do (
        set "SERVER_PID=%%P"
        goto :after_pid_read
    )
)
:after_pid_read

if not defined SERVER_PID (
    echo ---------------------------------
    echo [ERROR] Failed to start uvicorn in background (no PID returned)
    echo ---------------------------------
    if exist "%PS_CAPTURE%" type "%PS_CAPTURE%"
    del "%PS_CAPTURE%" >nul 2>&1
    set "EXIT_CODE=1"
    goto final_exit
)

del "%PS_CAPTURE%" >nul 2>&1

if exist "%PID_FILE%" (
    echo [INFO] PID file written: %PID_FILE% (PID !SERVER_PID!)
) else (
    echo ---------------------------------
    echo [ERROR] Failed to write PID file at %PID_FILE%
    echo [DEBUG] Current directory is:
    cd
    echo [DEBUG] Retrying with PowerShell...
    powershell -NoProfile -Command "[IO.File]::WriteAllText('%PID_FILE%', '!SERVER_PID!'.Trim())"
    if exist "%PID_FILE%" (
        echo [INFO] PID file created by PowerShell fallback.
    ) else (
        echo [ERROR] PowerShell fallback also failed.
    )
)

powershell -NoProfile -Command "Start-Sleep -Seconds 1; if (Get-Process -Id !SERVER_PID! -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo ---------------------------------
    echo [ERROR] Process !SERVER_PID! exited immediately. Check %LOG_FILE% for details.
    echo ---------------------------------
    if exist "%PID_FILE%" del "%PID_FILE%"
    if exist "%LOG_FILE%" (
    echo [INFO] Last 40 log lines:
        powershell -NoProfile -Command "Get-Content -Path '%LOG_FILE%' -Tail 40"
    )
    set "EXIT_CODE=1"
    goto final_exit
)

echo ---------------------------------
echo [OK] Uvicorn running in background (PID !SERVER_PID!)
echo [INFO] Logs: %LOG_FILE%
echo [INFO] Stop via stop.bat
echo ---------------------------------

set "EXIT_CODE=0"
goto final_exit

:check_existing_instance
if not exist "%PID_FILE%" exit /b 0
set "EXISTING_PID="
set /p EXISTING_PID=<"%PID_FILE%"
if not defined EXISTING_PID (
    del "%PID_FILE%"
    exit /b 0
)
powershell -NoProfile -Command "if (Get-Process -Id %EXISTING_PID% -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo ---------------------------------
    echo [ERROR] fast-fast-whisper already running (PID %EXISTING_PID%). Use stop.bat to stop it before starting a new instance.
    echo ---------------------------------
    exit /b 1
)
echo [WARN] Removing stale PID file (%PID_FILE%)
del "%PID_FILE%"
exit /b 0

:final_exit
if "%EXIT_CODE%"=="" set "EXIT_CODE=0"
if exist "%PS_CAPTURE%" del "%PS_CAPTURE%" >nul 2>&1
if defined PAUSE_SECONDS (
    echo ---------------------------------
    echo [INFO] Closing window in %PAUSE_SECONDS%s...
    timeout /t %PAUSE_SECONDS% >nul
)
exit /b %EXIT_CODE%
