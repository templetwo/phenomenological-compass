#!/bin/bash
# start_compass.sh — Launch Phenomenological Compass UI in proxy mode
#
# On MacBook: connects to Mac Studio for inference
# On Mac Studio: loads models directly
#
# Usage:
#   ./start_compass.sh              # proxy mode (MacBook → Mac Studio)
#   ./start_compass.sh --local      # local mode (Mac Studio, loads models)
#   ./start_compass.sh --port 9000  # custom port

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Defaults
PORT="${PORT:-8420}"
UPSTREAM="${COMPASS_UPSTREAM:-http://192.168.1.195:8420}"
MODE="proxy"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --local) MODE="local"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --upstream) UPSTREAM="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --quiet fastapi uvicorn httpx
    echo "Dependencies installed."
else
    source "$VENV_DIR/bin/activate"
fi

cd "$SCRIPT_DIR"

if [ "$MODE" = "proxy" ]; then
    echo ""
    echo "  Phenomenological Compass — Proxy Mode"
    echo "  ──────────────────────────────────────"
    echo "  UI:       http://localhost:$PORT"
    echo "  Upstream: $UPSTREAM"
    echo ""
    COMPASS_UPSTREAM="$UPSTREAM" PORT="$PORT" python3 compass_server.py
else
    echo ""
    echo "  Phenomenological Compass — Local Mode"
    echo "  ──────────────────────────────────────"
    echo "  UI:  http://localhost:$PORT"
    echo "  API: http://localhost:$PORT/api/health"
    echo ""
    PORT="$PORT" python3 compass_server.py
fi
