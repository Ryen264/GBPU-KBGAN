#!/bin/bash

# KBGAN Training Script with nohup
# This script runs the training process in the background and saves logs

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
PYTHON=${PYTHON:-python3}
# You can pass additional args to the training command via the EXTRA_ARGS env var.
# Example: EXTRA_ARGS="--config-file ./config/config_fb15k237.yaml --override KBGAN.n_epoch=1000"
CMD="$PYTHON main.py --mode full $EXTRA_ARGS"

# Run training with nohup using setsid to detach fully
nohup setsid bash -lc "$CMD" > "$LOG_FILE" 2>&1 &

# Save the process ID
PID=$!
echo $PID > "$LOG_DIR/training.pid"

echo "Training started with PID: $PID"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To check if running: ps -p $PID"
echo "To stop training: kill $PID"
echo ""
echo "PID saved to: $LOG_DIR/training.pid"
