#!/usr/bin/env python3
"""
compass_server.py — Web API for the Phenomenological Compass pipeline

Wraps pipeline.py as a FastAPI server with session memory and streaming support.

Usage:
    cd ~/phenomenological-compass
    source .venv/bin/activate
    pip install fastapi uvicorn
    HF_HOME=~/.cache/huggingface_local python3 compass_server.py

Then open http://localhost:8420 in your browser.
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Phenomenological Compass", version="0.8")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session Storage ──────────────────────────────────────────────────────────

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

sessions = {}  # session_id -> { messages: [], created: str, title: str }


def load_sessions():
    """Load all saved sessions from disk."""
    global sessions
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            sessions[f.stem] = data
        except Exception:
            pass


def save_session(session_id: str):
    """Persist a session to disk."""
    if session_id in sessions:
        path = SESSIONS_DIR / f"{session_id}.json"
        path.write_text(json.dumps(sessions[session_id], indent=2))


def get_or_create_session(session_id: Optional[str] = None) -> str:
    if session_id and session_id in sessions:
        return session_id
    new_id = str(uuid.uuid4())[:8]
    sessions[new_id] = {
        "messages": [],
        "created": datetime.now().isoformat(),
        "title": "New Session",
    }
    save_session(new_id)
    return new_id


# ── Pipeline Singleton ───────────────────────────────────────────────────────

pipeline_instance = None


def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        print("Loading compass pipeline (this takes ~30s on first load)...")
        from pipeline import Pipeline
        pipeline_instance = Pipeline()
        print("Pipeline ready.")
    return pipeline_instance


# ── API Models ───────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    mode: str = "routed"  # "routed", "raw", "compare"


class SessionRenameRequest(BaseModel):
    title: str


# ── API Routes ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_sessions()


@app.get("/")
async def serve_ui():
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>Compass UI not found. Place index.html in ./ui/</h1>")


@app.get("/api/health")
async def health():
    return {
        "status": "ready" if pipeline_instance else "loading",
        "version": "0.8",
        "sessions": len(sessions),
    }


@app.get("/api/sessions")
async def list_sessions():
    return {
        sid: {
            "title": s["title"],
            "created": s["created"],
            "message_count": len(s["messages"]),
            "last_message": s["messages"][-1]["timestamp"] if s["messages"] else s["created"],
        }
        for sid, s in sorted(sessions.items(), key=lambda x: x[1]["created"], reverse=True)
    }


@app.post("/api/sessions")
async def create_session():
    sid = get_or_create_session()
    return {"session_id": sid}


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: SessionRenameRequest):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sessions[session_id]["title"] = req.title
    save_session(session_id)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    del sessions[session_id]
    path = SESSIONS_DIR / f"{session_id}.json"
    path.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return {"messages": sessions[session_id]["messages"]}


@app.post("/api/infer")
async def infer(req: InferenceRequest):
    pipe = get_pipeline()
    sid = get_or_create_session(req.session_id)

    t0 = time.time()

    # Build context from session history (last 5 exchanges for context window)
    history = sessions[sid]["messages"][-10:]  # last 5 Q&A pairs
    context_summary = ""
    if history:
        context_lines = []
        for msg in history:
            if msg["role"] == "user":
                context_lines.append(f"Previous question: {msg['content'][:100]}")
            elif msg["role"] == "compass":
                context_lines.append(f"Previous signal: {msg.get('signal', 'unknown')}")
        context_summary = "\n".join(context_lines[-6:])

    # Run pipeline
    if req.mode == "raw":
        response_text, elapsed, thinking = pipe.raw(req.question)
        result = {
            "signal": None,
            "compass_response": None,
            "action_response": response_text,
            "thinking": thinking,
            "t_compass": 0,
            "t_action": elapsed,
        }
    elif req.mode == "compare":
        routed = pipe.run(req.question)
        raw_text, raw_elapsed, raw_thinking = pipe.raw(req.question)
        result = {
            **routed,
            "raw_response": raw_text,
            "raw_thinking": raw_thinking,
            "t_raw": raw_elapsed,
        }
    else:
        result = pipe.run(req.question)

    total_time = time.time() - t0

    # Parse compass reading
    compass_text = result.get("compass_response", "") or ""
    shape = tone = signal = translation = ""
    for section in ["SHAPE:", "TONE:", "SIGNAL:", "FRAMING:", "APPROACH:", "THRESHOLD:"]:
        pass  # We'll send the raw compass text and parse on frontend

    # Store in session
    user_msg = {
        "role": "user",
        "content": req.question,
        "timestamp": datetime.now().isoformat(),
    }
    compass_msg = {
        "role": "compass",
        "content": compass_text,
        "signal": result.get("signal", ""),
        "timestamp": datetime.now().isoformat(),
        "t_compass": result.get("t_compass", 0),
    }
    action_msg = {
        "role": "assistant",
        "content": result.get("action_response", ""),
        "timestamp": datetime.now().isoformat(),
        "t_action": result.get("t_action", 0),
    }

    sessions[sid]["messages"].extend([user_msg, compass_msg, action_msg])

    # Auto-title from first question
    if len(sessions[sid]["messages"]) <= 3:
        sessions[sid]["title"] = req.question[:60] + ("..." if len(req.question) > 60 else "")

    save_session(sid)

    response = {
        "session_id": sid,
        "signal": result.get("signal", ""),
        "compass_reading": compass_text,
        "action_response": result.get("action_response", ""),
        "thinking": result.get("thinking", ""),
        "t_compass": round(result.get("t_compass", 0), 1),
        "t_action": round(result.get("t_action", 0), 1),
        "t_total": round(total_time, 1),
    }

    if req.mode == "compare":
        response["raw_response"] = result.get("raw_response", "")
        response["raw_thinking"] = result.get("raw_thinking", "")
        response["t_raw"] = round(result.get("t_raw", 0), 1)

    return response


# ── Serve Static UI ──────────────────────────────────────────────────────────

ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n  🧭 Phenomenological Compass Server")
    print("  ──────────────────────────────────")
    print("  UI:  http://localhost:8420")
    print("  API: http://localhost:8420/api/health")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8420)
