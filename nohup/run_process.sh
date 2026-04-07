#!/bin/bash

# KBGAN training launcher using nohup.
# Usage:
#   ./nohup/run_process.sh [config_path.yaml] [mode=...] [extra_args]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python virtual environment not found at $PYTHON_BIN"
    echo "Create it first from project root:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

LOG_DIR="$ROOT_DIR/logs/nohup"
PID_FILE="$LOG_DIR/training.pid"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE")"
    if [[ -n "$existing_pid" ]] && ps -p "$existing_pid" > /dev/null 2>&1; then
        echo "A training process is already running (PID: $existing_pid)."
        echo "Stop it first with: ./nohup/stop_process.sh"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

config_path=""
mode_arg=""
extra_args=()

for arg in "$@"; do
    if [[ -z "$config_path" && "$arg" == *.yaml ]]; then
        config_path="$arg"
    elif [[ -z "$mode_arg" && "$arg" == mode=* ]]; then
        mode_arg="$arg"
    else
        extra_args+=("$arg")
    fi
done

cmd=("$PYTHON_BIN" -u "$ROOT_DIR/main.py")
if [[ -n "$config_path" ]]; then
    cmd+=("$config_path")
fi
if [[ -n "$mode_arg" ]]; then
    cmd+=("$mode_arg")
fi
if [[ ${#extra_args[@]} -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
fi

timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file="$LOG_DIR/training_${timestamp}.log"

echo "Starting KBGAN in background..."
echo "Project root: $ROOT_DIR"
echo "Log file: $log_file"
echo "Command: ${cmd[*]}"

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

echo "$pid" > "$PID_FILE"
ln -sfn "$log_file" "$LOG_DIR/latest.log"

echo ""
echo "Training started with PID: $pid"
echo "Check status: ./nohup/check_process.sh"
echo "Follow logs:  tail -f $LOG_DIR/latest.log"
echo "Stop:         ./nohup/stop_process.sh"
