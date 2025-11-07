#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PID_FILE=".fast-fast-whisper.pid"

log() {
    printf '[INFO] %s\n' "$1"
}

log_error() {
    printf '[ERROR] %s\n' "$1" >&2
}

pid_is_running() {
    local pid="$1"
    if [ -z "$pid" ]; then
        return 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

if [ ! -f "$PID_FILE" ]; then
    log "PID file not found. Server is not running."
    exit 0
fi

PID="$(tr -cd '0-9' <"$PID_FILE" 2>/dev/null || true)"

if [ -z "$PID" ]; then
    log_error "PID file is empty. Removing it."
    rm -f "$PID_FILE"
    exit 1
fi

if ! pid_is_running "$PID"; then
    log "Process $PID is not running. Removing stale PID file."
    rm -f "$PID_FILE"
    exit 0
fi

log "Stopping process $PID..."
kill "$PID" 2>/dev/null || true

for _ in $(seq 1 20); do
    if ! pid_is_running "$PID"; then
        rm -f "$PID_FILE"
        log "Process $PID stopped successfully."
        exit 0
    fi
    sleep 0.5
done

log "Process $PID did not exit after 10 seconds. Sending SIGKILL..."
kill -9 "$PID" 2>/dev/null || true

if pid_is_running "$PID"; then
    log_error "Failed to stop process $PID. Check manually."
    exit 1
fi

rm -f "$PID_FILE"
log "Process $PID forcefully stopped."
