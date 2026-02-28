#!/bin/bash

# Helper script to monitor training progress
# Usage: ./check_process.sh [lines]
# Example: ./check_process.sh 50  # Show last 50 lines

# Number of lines to display (default: 30, or 50 if process is stopped)
LINES=${1:-30}

# Ensure we're in the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="./logs"
PID_FILE="$LOG_DIR/training.pid"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "⚠ No training process found. PID file does not exist: $PID_FILE"
    echo ""
    echo "Available log files:"
    if [ -d "$LOG_DIR" ]; then
        ls -lht "$LOG_DIR"/training_*.log 2>/dev/null | head -10 || echo "No log files found"
    else
        echo "Log directory does not exist yet"
    fi
    exit 1
fi

# Read PID
PID=$(cat "$PID_FILE")

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "✓ Training is RUNNING (PID: $PID)"
    echo ""
    
    # Get process info
    PS_INFO=$(ps -p $PID -o etime=,pid=,cmd= 2>/dev/null || echo "")
    if [ -n "$PS_INFO" ]; then
        echo "Process info: $PS_INFO"
        echo ""
    fi
    
    # Find the latest log file (training_*.log)
    LATEST_LOG=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        echo "Showing last $LINES lines of: $LATEST_LOG"
        echo "========================================"
        tail -$LINES "$LATEST_LOG"
        echo "========================================"
        echo ""
        echo "To follow live: tail -f $LATEST_LOG"
        echo "To stop:        ./stop_process.sh"
    else
        echo "No log file found"
    fi
else
    echo "⚠ Training process is NOT RUNNING (PID was: $PID)"
    echo ""
    
    # Find the latest log file (training_*.log)
    LATEST_LOG=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        echo "Latest log file: $LATEST_LOG"
        LINES_TO_SHOW=${1:-50}  # Show more lines if process has stopped
        echo "Showing last $LINES_TO_SHOW lines:"
        echo "========================================"
        tail -$LINES_TO_SHOW "$LATEST_LOG"
        echo "========================================"
    else
        echo "No log file found"
    fi
    
    # Remove stale PID file
    echo ""
    echo "Cleaning up stale PID file: $PID_FILE"
    rm -f "$PID_FILE"
    
    echo ""
    echo "To start a new training: ./run_process.sh"
fi
