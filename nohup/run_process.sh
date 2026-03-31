#!/bin/bash

# KBGAN Training Script with nohup
# This script runs the training process in the background and saves logs
# Usage: ./run_process.sh [config_path] [mode] [extra_args]
# Examples:
#   ./run_process.sh                                    # Use defaults
#   ./run_process.sh config/config_wn18rr.yaml          # Specific config with default mode
#   ./run_process.sh config/config_wn18rr.yaml mode=full-train
#   ./run_process.sh config/config_wn18rr.yaml mode=test-only
#   ./run_process.sh config/config_wn18rr.yaml mode=full-train "--override KBGAN.n_epoch=1000"

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

# Parse arguments intelligently
# Supports (config path first is preferred):
#   ./run_process.sh
#   ./run_process.sh config/config_wn18rr.yaml
#   ./run_process.sh config/config_wn18rr.yaml mode=full-train
#   ./run_process.sh config/config_wn18rr.yaml mode=full-train "--override KBGAN.n_epoch=1000"
# Backward compatible with mode first:
#   ./run_process.sh mode=full-train
#   ./run_process.sh full-train config/config_wn18rr.yaml

CMD="$PYTHON main.py"
EXTRA_ARGS=""

# Check first argument
if [ -n "$1" ]; then
    if [[ "$1" == *.yaml ]]; then
        # First arg is config path
        CMD="$CMD $1"
        shift
    else
        # First arg might be mode or other parameter
        CMD="$CMD $1"
        shift
    fi
fi

# Check if next arg is a config path
if [ -n "$1" ] && [[ "$1" == *.yaml ]]; then
    CMD="$CMD $1"
    shift
fi

# Collect remaining arguments as config overrides
while [ -n "$1" ]; do
    EXTRA_ARGS="$EXTRA_ARGS $1"
    shift
done

CMD="$CMD $EXTRA_ARGS"
CMD=${CMD%% }  # Remove trailing spaces

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
