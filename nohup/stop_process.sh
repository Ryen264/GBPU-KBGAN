#!/bin/bash

# Stop background KBGAN run started by nohup/run_process.sh.
# Usage: ./nohup/stop_process.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/nohup"
PID_FILE="$LOG_DIR/training.pid"
FORCE_KILL="${1:-}"

stop_by_pid() {
    local pid="$1"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo "Process is not running (PID: $pid)"
        return 0
    fi

    if [[ "$FORCE_KILL" == "--force" ]]; then
        echo "Force killing PID: $pid"
        kill -9 "$pid"
        sleep 1
    else
        echo "Sending SIGTERM to PID: $pid"
        kill -TERM "$pid"

        for _ in {1..10}; do
            if ! ps -p "$pid" > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done

        if ps -p "$pid" > /dev/null 2>&1; then
            echo "PID $pid still alive, sending SIGKILL"
            kill -9 "$pid"
            sleep 1
        fi
    fi

    if ps -p "$pid" > /dev/null 2>&1; then
        echo "ERROR: Failed to stop PID $pid"
        return 1
    fi

    echo "Stopped PID: $pid"
    return 0
}

stopped_any=0

if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]]; then
        if stop_by_pid "$pid"; then
            stopped_any=1
        fi
    fi
    rm -f "$PID_FILE"
fi

# Fallback: find processes that match this repository's main.py
mapfile -t fallback_pids < <(pgrep -f "$ROOT_DIR/main.py" || true)
for p in "${fallback_pids[@]}"; do
    if stop_by_pid "$p"; then
        stopped_any=1
    fi
done

if [[ "$stopped_any" -eq 1 ]]; then
    echo "Background training process stopped."
    exit 0
fi

echo "No matching background training process found."
exit 1
