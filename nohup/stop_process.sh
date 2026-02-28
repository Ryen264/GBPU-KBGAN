#!/bin/bash

# Helper script to stop training process
# Usage: ./stop_process.sh [--force]
# Use --force for immediate kill (SIGKILL) instead of graceful shutdown

# Ensure we're in the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="./logs"
PID_FILE="$LOG_DIR/training.pid"
FORCE_KILL=${1:-}

# Function to stop by PID
stop_by_pid() {
    local pid=$1
    local force=$2
    
    if ps -p $pid > /dev/null 2>&1; then
        if [ "$force" = "--force" ]; then
            echo "Force terminating process (PID: $pid)..."
            kill -9 $pid
            sleep 1
        else
            echo "Stopping training process gracefully (PID: $pid)..."
            kill -TERM $pid
            sleep 2
        fi
        
        # Check if process is still running
        if ps -p $pid > /dev/null 2>&1; then
            if [ "$force" != "--force" ]; then
                echo "Process still running after SIGTERM, forcing termination..."
                kill -9 $pid
                sleep 1
            fi
            
            # Final check
            if ps -p $pid > /dev/null 2>&1; then
                echo "✗ ERROR: Failed to stop process $pid"
                return 1
            fi
        fi
        
        echo "✓ Training process stopped successfully"
        return 0
    else
        echo "✓ Process is not running (PID: $pid)"
        return 0
    fi
}

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "⚠ No training process found. PID file does not exist: $PID_FILE"
    echo ""
    echo "Attempting fallback kill by command match..."
    # Try to find any main.py process
    PIDS=$(pgrep -f "main.py" || true)
    if [ -n "$PIDS" ]; then
        echo "Found processes: $PIDS"
        for p in $PIDS; do
            stop_by_pid $p $FORCE_KILL || true
        done
        exit 0
    else
        echo "No processes found matching 'main.py'"
        exit 1
    fi
fi

# Read PID
PID=$(cat "$PID_FILE")

# Try stopping the stored PID
if stop_by_pid $PID $FORCE_KILL; then
    rm -f "$PID_FILE"
    echo ""
    echo "PID file removed: $PID_FILE"
    exit 0
fi

# If PID method failed, try to find a matching process and kill it (best-effort)
echo ""
echo "Attempting fallback kill by command match..."
PIDS=$(pgrep -f "main.py" || true)
if [ -n "$PIDS" ]; then
    echo "Found processes: $PIDS"
    for p in $PIDS; do
        stop_by_pid $p $FORCE_KILL || true
    done
    rm -f "$PID_FILE"
    exit 0
else
    echo "No fallback processes found matching 'main.py'"
    rm -f "$PID_FILE"
    exit 1
fi
