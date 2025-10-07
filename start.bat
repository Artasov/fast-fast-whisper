@echo off
setlocal EnableExtensions
setlocal DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "PY_VERSION=3.12.5"
set "PY_ROOT=python_portable"
set "PY_EXE=%PY_ROOT%\python.exe"
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo ---------------------------------
echo [INFO] Start installation
echo ---------------------------------

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
    exit /b 1
)

echo ---------------------------------
echo [INFO] Renaming "%TMPNUP%" → "%TMPZIP%"
echo ---------------------------------

ren "%TMPNUP%" "python_portable_%PY_VERSION%.zip"
if not exist "%TMPZIP%" (
    REM file might have been renamed elsewhere, check current Temp folder

    echo ---------------------------------
    echo [WARN] Expected file %TMPZIP% in TEMP, but it's not there
    echo ---------------------------------

    dir /b "%TEMP%\*.zip"
    pause
    exit /b 1
)

echo ---------------------------------
echo [INFO] Extracting archive %TMPZIP%
echo ---------------------------------

powershell -NoProfile -Command "Expand-Archive -Path '%TMPZIP%' -DestinationPath 'python_extracted'"
if not exist "python_extracted\tools\python.exe" (
    echo [ERROR] tools\python.exe not found in extracted archive
    pause
    exit /b 1
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
    exit /b 1
)

echo ---------------------------------
echo [OK] Portable Python ready: %PY_EXE%
echo ---------------------------------

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
        exit /b 1
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
echo [RUN] Starting uvicorn...
echo ---------------------------------
"%VENV_PY%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

exit /b %ERRORLEVEL%
