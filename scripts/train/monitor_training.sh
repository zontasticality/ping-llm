#!/bin/bash
# Monitor training progress via TensorBoard
#
# Usage:
#   # On the cluster (background):
#   ./scripts/train/monitor_training.sh start
#
#   # On your local machine (SSH tunnel):
#   ssh -L 6006:localhost:6006 <cluster-host>
#   # Then open: http://localhost:6006

set -euo pipefail

TENSORBOARD_DIR="${TENSORBOARD_DIR:-outputs/latency_network}"
PORT="${PORT:-6006}"

case "${1:-help}" in
  start)
    echo "Starting TensorBoard on port $PORT..."
    echo "Monitoring: $TENSORBOARD_DIR"

    # Check if TensorBoard is installed
    if ! command -v tensorboard &> /dev/null; then
        echo "ERROR: TensorBoard not found. Install with:"
        echo "  pip install tensorboard"
        exit 1
    fi

    # Kill any existing TensorBoard on this port
    pkill -f "tensorboard.*--port $PORT" || true

    # Start TensorBoard in background
    nohup tensorboard --logdir="$TENSORBOARD_DIR" --port=$PORT --bind_all > tensorboard.log 2>&1 &
    TB_PID=$!

    echo "TensorBoard started (PID: $TB_PID)"
    echo ""
    echo "To view locally on cluster:"
    echo "  http://localhost:$PORT"
    echo ""
    echo "To view from your laptop:"
    echo "  1. Run on your laptop: ssh -L 6006:localhost:6006 $(whoami)@$(hostname)"
    echo "  2. Open browser: http://localhost:6006"
    echo ""
    echo "Logs: tensorboard.log"
    ;;

  stop)
    echo "Stopping TensorBoard..."
    pkill -f "tensorboard.*--port $PORT" || echo "No TensorBoard process found"
    ;;

  status)
    if pgrep -f "tensorboard.*--port $PORT" > /dev/null; then
        echo "TensorBoard is running on port $PORT"
        echo "PID: $(pgrep -f "tensorboard.*--port $PORT")"
    else
        echo "TensorBoard is not running"
    fi
    ;;

  *)
    echo "Usage: $0 {start|stop|status}"
    echo ""
    echo "Commands:"
    echo "  start   - Start TensorBoard server"
    echo "  stop    - Stop TensorBoard server"
    echo "  status  - Check if TensorBoard is running"
    echo ""
    echo "Environment variables:"
    echo "  TENSORBOARD_DIR - Directory to monitor (default: outputs/latency_network)"
    echo "  PORT - Port to run on (default: 6006)"
    exit 1
    ;;
esac
