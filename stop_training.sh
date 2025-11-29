#!/bin/bash

# Helper script to stop training process

LOG_DIR="./logs"
PID_FILE="$LOG_DIR/training.pid"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "No training process found. PID file does not exist: $PID_FILE"
    exit 1
fi

# Read PID
PID=$(cat "$PID_FILE")

# Function to stop by PID
stop_by_pid() {
    local pid=$1
    if ps -p $pid > /dev/null 2>&1; then
        echo "Stopping training process (PID: $pid)..."
        kill $pid
        sleep 2
        if ps -p $pid > /dev/null 2>&1; then
            echo "Process still running, forcing termination..."
            kill -9 $pid
            sleep 1
        fi
        if ps -p $pid > /dev/null 2>&1; then
            echo "ERROR: Failed to stop process $pid"
            return 1
        else
            echo "Training process stopped successfully"
            return 0
        fi
    else
        echo "Training process is not running (PID: $pid)"
        return 0
    fi
}

# Try stopping the stored PID first
if stop_by_pid $PID; then
    rm -f "$PID_FILE"
    exit 0
fi

# If PID method failed, try to find a matching process and kill it (best-effort)
echo "Attempting fallback kill by command match..."
PIDS=$(pgrep -f "main.py --mode full" || true)
if [ -n "$PIDS" ]; then
    echo "Found processes: $PIDS"
    for p in $PIDS; do
        stop_by_pid $p || true
    done
    rm -f "$PID_FILE"
    exit 0
else
    echo "No fallback processes found matching 'main.py --mode full'"
    exit 1
fi
