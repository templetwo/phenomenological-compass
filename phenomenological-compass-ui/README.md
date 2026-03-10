# Phenomenological Compass — Web UI

Local web interface for the compass pipeline with session memory.

## Setup (Mac Studio)

```bash
cd ~/phenomenological-compass

# Copy these files into your project
cp compass_server.py .
mkdir -p ui && cp ui/index.html ui/

# Install server dependencies
source .venv/bin/activate
pip install fastapi uvicorn

# Launch
HF_HOME=~/.cache/huggingface_local python3 compass_server.py
```

Open **http://localhost:8420** in your browser.

## Features

- **Three signal modes** — OPEN (green), PAUSE (gold), WITNESS (purple) with visual indicators
- **Compass reading display** — SHAPE, TONE, SIGNAL, and translation shown in collapsible cards
- **Session memory** — conversations persist to `sessions/*.json`, survive server restarts
- **Context threading** — last 5 exchanges fed as context to the pipeline
- **Compare mode** — see compass-routed vs raw Qwen side by side
- **Three input modes** — Compass (routed), Compare (side-by-side), Raw (direct to Qwen)

## Architecture

```
Browser (localhost:8420)
    ↓ POST /api/infer
FastAPI Server (compass_server.py)
    ↓
Compass (Ministral-3B + LoRA) → SHAPE/TONE/SIGNAL
    ↓
Qwen3.5-9B-abliterated → Final Response
    ↓
Session stored to sessions/*.json
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve the UI |
| GET | `/api/health` | Pipeline status |
| GET | `/api/sessions` | List all sessions |
| POST | `/api/sessions` | Create new session |
| DELETE | `/api/sessions/:id` | Delete session |
| GET | `/api/sessions/:id/messages` | Get session history |
| POST | `/api/infer` | Run inference |

## Port

Default: **8420**. Change in `compass_server.py` at the bottom.
