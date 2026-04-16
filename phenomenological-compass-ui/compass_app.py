#!/usr/bin/env python3
"""
compass_app.py — Phenomenological Compass as a native macOS app
================================================================
Self-contained: MLX compass + Ollama action model + PyWebView window.
No browser. No terminal. No visible server. Just a window.

Usage:
    python3 compass_app.py              # default: gemma-e2b action model
    python3 compass_app.py --action qwen  # use Qwen 9B via MLX instead

Requirements:
    pip install pywebview fastapi uvicorn mlx mlx-lm requests
    ollama pull gemma4:e2b
"""

import os
import sys
import threading
import time
import signal

# Project root
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(APP_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Force local mode — clear any stale proxy env var so the app
# runs inference locally instead of forwarding to the Mac Studio
os.environ.pop("COMPASS_UPSTREAM", None)


def start_server(port=8420):
    """Start the FastAPI compass server in-process."""
    import uvicorn
    # Import after path setup so pipeline.py is found
    from compass_server import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def wait_for_server(port=8420, timeout=120):
    """Block until the server responds or timeout."""
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/api/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phenomenological Compass App")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--action", default=None,
                        help="Action model key (gemma-e2b, qwen, etc.)")
    args = parser.parse_args()

    # If action model override, set it before importing compass_server
    if args.action:
        os.environ["COMPASS_ACTION"] = args.action

    port = args.port

    # Start server in background thread
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    print("Starting Phenomenological Compass...")
    print(f"  Loading models (this takes ~15-30s on first launch)...")

    # Wait for server to be ready
    if not wait_for_server(port, timeout=120):
        print("ERROR: Server failed to start within 120s")
        sys.exit(1)

    print(f"  Server ready on port {port}")
    print(f"  Opening compass window...")

    # Launch native window
    import webview
    window = webview.create_window(
        "Phenomenological Compass",
        f"http://127.0.0.1:{port}",
        width=1200,
        height=800,
        min_size=(800, 600),
        text_select=True,
    )

    # webview.start() blocks until the window is closed
    webview.start()

    print("Window closed. Shutting down...")
    os._exit(0)


if __name__ == "__main__":
    main()
