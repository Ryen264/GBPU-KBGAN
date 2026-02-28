#!/bin/bash

# KBGAN Training Script with nohup
# This script runs the training process in the background and saves logs
# Usage: ./run_process.sh [mode] [extra_args]
# Example: ./run_process.sh full-train "--override KBGAN.n_epoch=1000"

set -e  # Exit on error

# Ensure we're in the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verify virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found. Create it first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Set up log directory and filename
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "Starting KBGAN training..."
echo "Log file: $LOG_FILE"
echo "Process will run in background with nohup"
echo ""

# Build command
PYTHON="./.venv/bin/python"
# MODE_ARG: default is full-train
MODE_ARG=${1:-mode=full-train}
# Additional arguments can be passed via positional args
EXTRA_ARGS=${2:-}
CMD="$PYTHON main.py $MODE_ARG $EXTRA_ARGS"

echo "Running command: $CMD"
echo ""

# Run training with nohup in background
nohup $CMD > "$LOG_FILE" 2>&1 &

# Save the process ID
PID=$!
echo $PID > "$LOG_DIR/training.pid"

echo "✓ Training started with PID: $PID"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "  ./check_process.sh"
echo ""
echo "Stop training:"
echo "  ./stop_process.sh"
echo ""
echo "PID saved to: $LOG_DIR/training.pid"
