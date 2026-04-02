#!/bin/bash

# Check status and recent logs for a background KBGAN run.
# Usage: ./nohup/check_process.sh [lines]

set -euo pipefail

lines="${1:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/nohup"
PID_FILE="$LOG_DIR/training.pid"

latest_log=""
if [[ -L "$LOG_DIR/latest.log" || -f "$LOG_DIR/latest.log" ]]; then
    latest_log="$LOG_DIR/latest.log"
else
    latest_log="$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1 || true)"
fi

if [[ ! -f "$PID_FILE" ]]; then
    echo "No active PID file found at $PID_FILE"
    if [[ -n "$latest_log" ]]; then
        echo "Showing latest log tail ($lines lines): $latest_log"
        echo "========================================"
        tail -n "$lines" "$latest_log"
        echo "========================================"
    else
        echo "No nohup log files found yet in $LOG_DIR"
    fi
    exit 1
fi

pid="$(cat "$PID_FILE")"

if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
    echo "Training is RUNNING (PID: $pid)"
    ps -p "$pid" -o etime=,pid=,cmd=
    echo ""
    if [[ -n "$latest_log" ]]; then
        echo "Showing last $lines lines from: $latest_log"
        echo "========================================"
        tail -n "$lines" "$latest_log"
        echo "========================================"
        echo "Follow live: tail -f $latest_log"
    else
        echo "No log file found in $LOG_DIR"
    fi
else
    echo "Training is NOT running (stale PID: $pid)"
    if [[ -n "$latest_log" ]]; then
        echo "Showing latest log tail (50 lines): $latest_log"
        echo "========================================"
        tail -n 50 "$latest_log"
        echo "========================================"
    fi
    rm -f "$PID_FILE"
    echo "Removed stale PID file: $PID_FILE"
    exit 1
fi
