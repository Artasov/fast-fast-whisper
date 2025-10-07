@echo off
REM ============================================================
REM  start.bat — автозапуск локального Python, venv и сервера
REM  Действия:
REM   1) Проверяем локальный Python в .\python\python.exe, иначе качаем и ставим тихо.
REM   2) Проверяем .venv, иначе создаем.
REM   3) Устанавливаем зависимости из requirements.txt.
REM   4) Запускаем uvicorn main:app --reload на 0.0.0.0:8000
REM ============================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

REM --- Всегда работаем из папки скрипта
cd /d "%~dp0"

REM --- Конфиг
set PY_VERSION=3.12.5
set PY_DIR=python
set PY_EXE=%CD%\%PY_DIR%\python.exe
set PY_WIN_X64_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe
set PY_INSTALLER=%TEMP%\python-%PY_VERSION%-amd64.exe
set VENV_DIR=.venv
set VENV_PY=%CD%\%VENV_DIR%\Scripts\python.exe

echo [INFO] Репозиторий: %CD%
echo [INFO] Целевая версия Python: %PY_VERSION%
echo [INFO] Проверяю локальный интерпретатор: "%PY_EXE%"

REM --- 1) Python: если нет, скачиваем и ставим в .\python
if not exist "%PY_EXE%" (
    echo [INFO] Локальный Python не найден. Скачиваю установщик...
    REM Пытаемся использовать curl, если его нет — используем PowerShell
    where curl >nul 2>nul
    if %errorlevel%==0 (
        curl -L -o "%PY_INSTALLER%" "%PY_WIN_X64_URL%"
    ) else (
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
          "Invoke-WebRequest -Uri '%PY_WIN_X64_URL%' -OutFile '%PY_INSTALLER%'; if (!(Test-Path '%PY_INSTALLER%')) { exit 1 }"
        if not exist "%PY_INSTALLER%" (
            echo [ERROR] Не удалось скачать установщик Python.
            exit /b 1
        )
    )

    echo [INFO] Ставлю Python тихо в "%CD%\%PY_DIR%" (без добавления в PATH)...
    "%PY_INSTALLER%" ^
        /quiet ^
        InstallAllUsers=0 ^
        TargetDir="%CD%\%PY_DIR%" ^
        Include_pip=1 ^
        PrependPath=0 ^
        Include_launcher=0 ^
        Shortcuts=0 ^
        Include_debug=0 ^
        Include_test=0 ^
        SimpleInstall=1

    set INSTALL_RC=%ERRORLEVEL%
    del /q "%PY_INSTALLER%" 2>nul

    if not exist "%PY_EXE%" (
        echo [ERROR] Python не установился в "%CD%\%PY_DIR%". Код: %INSTALL_RC%
        exit /b 1
    )
    echo [OK] Python установлен: "%PY_EXE%"
) else (
    echo [OK] Найден локальный Python.
)

REM --- 2) VENV: если нет, создаем
if not exist "%VENV_PY%" (
    echo [INFO] Создаю виртуальное окружение: "%VENV_DIR%"...
    "%PY_EXE%" -m venv "%VENV_DIR%"
    if not exist "%VENV_PY%" (
        echo [ERROR] Не удалось создать venv по пути "%VENV_DIR%".
        exit /b 1
    )
) else (
    echo [OK] Найдено виртуальное окружение: "%VENV_DIR%".
)

REM --- 3) Устанавливаем/обновляем зависимости
echo [INFO] Обновляю pip...
"%VENV_PY%" -m pip install --upgrade pip

if exist "%CD%\requirements.txt" (
    echo [INFO] Устанавливаю зависимости из requirements.txt...
    "%VENV_PY%" -m pip install -r requirements.txt
) else (
    echo [WARN] requirements.txt не найден — пропускаю установку зависимостей.
)

REM --- 4) Запускаем сервер (без активации, напрямую через venv-питон)
echo [RUN] Запускаю Uvicorn на 0.0.0.0:8000 с --reload...
"%VENV_PY%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM --- Если uvicorn упал — сообщим код возврата
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
    echo [ERROR] Uvicorn завершился с кодом %RC%.
)
exit /b %RC%
