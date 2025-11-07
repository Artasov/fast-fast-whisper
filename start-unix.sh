#!/usr/bin/env bash

set -euo pipefail

REQUIRED_MAJOR=3
REQUIRED_MINOR=12
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python"
DEFAULT_PORT=8868
PORT="${FAST_FAST_WHISPER_PORT:-${PORT:-$DEFAULT_PORT}}"

log() {
    printf '[INFO] %s\n' "$1"
}

log_error() {
    printf '[ERROR] %s\n' "$1" >&2
}

check_python_version() {
    local exe="$1"
    "$exe" - <<PY
import sys
min_version = (${REQUIRED_MAJOR}, ${REQUIRED_MINOR})
current = (sys.version_info.major, sys.version_info.minor)
sys.exit(0 if current >= min_version else 1)
PY
}

ensure_python() {
    local candidate

    if [ -n "${PYTHON_BIN:-}" ]; then
        if check_python_version "$PYTHON_BIN"; then
            log "Using user-provided PYTHON_BIN=$PYTHON_BIN"
            return
        fi
        log_error "PYTHON_BIN=$PYTHON_BIN does not meet >= ${REQUIRED_MAJOR}.${REQUIRED_MINOR} requirement"
        exit 1
    fi

    local candidates=(python3.12 python3 python)
    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if check_python_version "$candidate"; then
                PYTHON_BIN="$(command -v "$candidate")"
                log "Using Python interpreter: $PYTHON_BIN"
                return
            fi
            log "Skipping $candidate (version too old)"
        fi
    done

    log_error "Python >= ${REQUIRED_MAJOR}.${REQUIRED_MINOR} is required. Set PYTHON_BIN to an explicit interpreter if available."
    exit 1
}

create_venv() {
    if [ ! -x "$VENV_PY" ]; then
        log "Creating virtual environment in $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        log "Reusing existing virtual environment in $VENV_DIR"
    fi
}

install_dependencies() {
    log "Upgrading pip"
    "$VENV_PY" -m pip install --upgrade pip >/dev/null

    if [ -f requirements.txt ]; then
        log "Installing dependencies from requirements.txt"
        "$VENV_PY" -m pip install -r requirements.txt
    else
        log "requirements.txt not found, skipping dependency installation"
    fi
}

run_server() {
    log "Starting uvicorn on 0.0.0.0:$PORT (set FAST_FAST_WHISPER_PORT or PORT to override)"
    exec "$VENV_PY" -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
}

ensure_python
create_venv
install_dependencies
run_server
