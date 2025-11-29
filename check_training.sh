#!/bin/bash

# Helper script to monitor training progress

LOG_DIR="./logs"
PID_FILE="$LOG_DIR/training.pid"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "No training process found. PID file does not exist: $PID_FILE"
    echo ""
    echo "Available log files:"
    ls -lht "$LOG_DIR"/*.log 2>/dev/null || echo "No log files found"
    exit 1
fi

# Read PID
PID=$(cat "$PID_FILE")

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "Training is RUNNING (PID: $PID)"
    echo ""
    
    # Find the latest log file (training_*.log)
    LATEST_LOG=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        echo "Showing last 30 lines of: $LATEST_LOG"
        echo "========================================"
        tail -30 "$LATEST_LOG"
        echo "========================================"
        echo ""
        echo "To follow live: tail -f $LATEST_LOG"
    else
        echo "No log file found"
    fi
else
    echo "Training process is NOT RUNNING (PID: $PID)"
    echo ""
    
    # Find the latest log file (training_*.log)
    LATEST_LOG=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        echo "Latest log file: $LATEST_LOG"
        echo "Showing last 50 lines:"
        echo "========================================"
        tail -50 "$LATEST_LOG"
        echo "========================================"
    else
        echo "No log file found"
    fi
    
    # Remove stale PID file
    rm -f "$PID_FILE"
fi
