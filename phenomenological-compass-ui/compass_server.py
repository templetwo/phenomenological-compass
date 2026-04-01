#!/usr/bin/env python3
"""
compass_server.py — Web API for the Phenomenological Compass pipeline
=====================================================================
Wraps pipeline.py as a FastAPI server with session memory and SSE streaming.

Endpoints:
    GET  /              — Serve UI
    GET  /api/health    — Pipeline status
    POST /api/infer     — Non-streaming inference (backward compat)
    POST /api/stream    — SSE streaming inference with per-token entropy
    GET  /api/sessions  — List sessions
    POST /api/sessions  — Create session
    ...

Usage:
    cd ~/phenomenological-compass/phenomenological-compass-ui
    source ~/phenomenological-compass/.venv/bin/activate
    HF_HOME=~/.cache/huggingface_local python3 compass_server.py
"""

import os
import re
import sys
import json
import math
import time
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Phenomenological Compass", version="0.9")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-threaded executor for MLX (not thread-safe for concurrent model access)
_executor = ThreadPoolExecutor(max_workers=1)
_stream_lock = asyncio.Lock()

# ── Session Storage ──────────────────────────────────────────────────────────

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

sessions = {}


def load_sessions():
    global sessions
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            sessions[f.stem] = data
        except Exception:
            pass


def save_session(session_id: str):
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


# ── Entropy Computation ─────────────────────────────────────────────────────

def compute_entropy(logprobs_array):
    """Shannon entropy in nats from MLX log-probability array."""
    import mlx.core as mx
    probs = mx.exp(logprobs_array)
    entropy = -mx.sum(probs * logprobs_array)
    mx.eval(entropy)
    return round(float(entropy.item()), 4)


# ── Think-Tag State Machine ─────────────────────────────────────────────────

class ThinkDetector:
    """Detects <think>...</think> blocks in a token stream."""

    def __init__(self):
        self.state = "pending"  # pending | thinking | response
        self.buffer = ""

    def feed(self, token_text):
        """Feed a token, return (stage, text_to_emit).
        stage is 'thinking', 'action', or None (still buffering)."""
        self.buffer += token_text

        if self.state == "pending":
            if "<think>" in self.buffer:
                self.state = "thinking"
                after = self.buffer.split("<think>", 1)[1]
                return "thinking", after
            # If we see </think> without opening (Qwen sometimes skips it)
            if "</think>" in self.buffer:
                before = self.buffer.split("</think>", 1)[0]
                after = self.buffer.split("</think>", 1)[1]
                self.state = "response"
                # The 'before' was thinking, 'after' is response
                return "action", after.lstrip()
            # If enough tokens without any tag, it's direct response
            if len(self.buffer) > 20 and "<" not in self.buffer:
                self.state = "response"
                return "action", self.buffer
            return None, ""

        elif self.state == "thinking":
            if "</think>" in self.buffer:
                after = self.buffer.split("</think>", 1)[1]
                self.state = "response"
                return "action", after.lstrip()
            return "thinking", token_text

        else:  # response
            # Clean stray tags
            clean = token_text.replace("<|im_end|>", "")
            return "action", clean


# ── Compass Section Detector ────────────────────────────────────────────────

COMPASS_SECTIONS = ["SHAPE", "TONE", "SIGNAL", "FRAMING", "APPROACH", "THRESHOLD"]

def detect_compass_section(accumulated_text):
    """Return the current section label based on accumulated compass text."""
    last_section = None
    for section in COMPASS_SECTIONS:
        if f"{section}:" in accumulated_text:
            last_section = section.lower()
    return last_section


def detect_signal(accumulated_text):
    """Extract signal from accumulated compass text."""
    m = re.search(r"SIGNAL:\s*(OPEN|PAUSE|WITNESS)", accumulated_text, re.IGNORECASE)
    return m.group(1).upper() if m else None


# ── API Models ───────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    mode: str = "routed"  # "routed", "raw", "compare"


class SessionRenameRequest(BaseModel):
    title: str


# ── SSE Helpers ──────────────────────────────────────────────────────────────

def sse_event(data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"


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
        "version": "0.9",
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


# ── Streaming Endpoint ──────────────────────────────────────────────────────

@app.post("/api/stream")
async def stream_infer(req: InferenceRequest):
    """SSE streaming endpoint. Yields per-token events with entropy."""

    async def event_generator():
        async with _stream_lock:
            pipe = get_pipeline()
            sid = get_or_create_session(req.session_id)
            loop = asyncio.get_event_loop()

            # ── Stage 1: Compass ────────────────────────────────────
            if req.mode != "raw":
                compass_text = ""
                signal = None
                signal_sent = False
                prev_section = None
                t0 = time.time()
                n_compass = 0

                queue = asyncio.Queue()

                def _run_compass():
                    for text, logprobs, finish, tps in pipe.stream_classify(req.question):
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("token", text, logprobs, finish, tps)), loop
                        )
                    asyncio.run_coroutine_threadsafe(queue.put(("done",)), loop)

                loop.run_in_executor(_executor, _run_compass)

                while True:
                    item = await queue.get()
                    if item[0] == "done":
                        break

                    _, text, logprobs, finish, tps = item
                    compass_text += text
                    n_compass += 1

                    # Detect section transitions
                    section = detect_compass_section(compass_text)
                    section_changed = section != prev_section
                    prev_section = section

                    # Check for signal
                    if not signal_sent:
                        detected = detect_signal(compass_text)
                        if detected:
                            signal = detected
                            signal_sent = True
                            yield sse_event({
                                "stage": "signal_lock",
                                "signal": signal,
                            })

                    yield sse_event({
                        "stage": "compass",
                        "token": text,
                        "section": section,
                        "section_changed": section_changed,
                        "n": n_compass,
                    })

                t_compass = round(time.time() - t0, 1)
                signal = signal or detect_signal(compass_text) or "OPEN"

                yield sse_event({
                    "stage": "compass_done",
                    "text": compass_text.strip(),
                    "signal": signal,
                    "t_compass": t_compass,
                })
            else:
                compass_text = ""
                signal = None
                t_compass = 0

            # ── Stage 2: Action Model ───────────────────────────────
            think_detector = ThinkDetector()
            action_text = ""
            thinking_text = ""
            response_text = ""
            entropy_values = []
            t1 = time.time()
            n_action = 0

            queue2 = asyncio.Queue()

            def _run_action():
                if req.mode == "raw":
                    gen = pipe.stream_raw(req.question)
                else:
                    gen = pipe.stream_act(req.question, signal, compass_text)
                for text, logprobs, finish, tps in gen:
                    asyncio.run_coroutine_threadsafe(
                        queue2.put(("token", text, logprobs, finish, tps)), loop
                    )
                asyncio.run_coroutine_threadsafe(queue2.put(("done",)), loop)

            loop.run_in_executor(_executor, _run_action)

            while True:
                item = await queue2.get()
                if item[0] == "done":
                    break

                _, text, logprobs, finish, tps = item
                action_text += text
                n_action += 1

                # Compute entropy
                entropy = None
                if logprobs is not None:
                    try:
                        entropy = compute_entropy(logprobs)
                    except Exception:
                        pass

                # Route through think detector
                stage, emit_text = think_detector.feed(text)

                if stage == "thinking":
                    thinking_text += emit_text
                    yield sse_event({
                        "stage": "thinking",
                        "token": emit_text,
                        "n": n_action,
                    })
                elif stage == "action":
                    response_text += emit_text
                    if entropy is not None:
                        entropy_values.append(entropy)
                    yield sse_event({
                        "stage": "action",
                        "token": emit_text,
                        "entropy": entropy,
                        "n": n_action,
                        "tps": round(tps, 1) if tps else None,
                    })
                # stage is None: still buffering for think detection

            t_action = round(time.time() - t1, 1)

            # ── Finalize ────────────────────────────────────────────
            # Clean response text
            response_text = response_text.replace("<|im_end|>", "").strip()

            mean_entropy = round(sum(entropy_values) / len(entropy_values), 4) if entropy_values else None

            yield sse_event({
                "stage": "done",
                "session_id": sid,
                "signal": signal,
                "t_compass": t_compass,
                "t_action": t_action,
                "t_total": round(t_compass + t_action, 1),
                "mean_entropy": mean_entropy,
                "response_text": response_text,
                "thinking_text": thinking_text,
            })

            # Save to session
            now = datetime.now().isoformat()
            user_msg = {"role": "user", "content": req.question, "timestamp": now}
            compass_msg = {
                "role": "compass", "content": compass_text.strip(),
                "signal": signal or "", "timestamp": now,
                "t_compass": t_compass,
            }
            action_msg = {
                "role": "assistant", "content": response_text,
                "timestamp": now, "t_action": t_action,
            }
            sessions[sid]["messages"].extend([user_msg, compass_msg, action_msg])

            if len(sessions[sid]["messages"]) <= 3:
                sessions[sid]["title"] = req.question[:60] + ("..." if len(req.question) > 60 else "")

            save_session(sid)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Non-Streaming Endpoint (backward compat) ────────────────────────────────

@app.post("/api/infer")
async def infer(req: InferenceRequest):
    pipe = get_pipeline()
    sid = get_or_create_session(req.session_id)

    t0 = time.time()

    if req.mode == "raw":
        response_text, elapsed, thinking = pipe.raw(req.question)
        result = {
            "signal": None, "compass_response": None,
            "action_response": response_text, "thinking": thinking,
            "t_compass": 0, "t_action": elapsed,
        }
    elif req.mode == "compare":
        routed = pipe.run(req.question)
        raw_text, raw_elapsed, raw_thinking = pipe.raw(req.question)
        result = {
            **routed, "raw_response": raw_text,
            "raw_thinking": raw_thinking, "t_raw": raw_elapsed,
        }
    else:
        result = pipe.run(req.question)

    total_time = time.time() - t0
    compass_text = result.get("compass_response", "") or ""

    now = datetime.now().isoformat()
    sessions[sid]["messages"].extend([
        {"role": "user", "content": req.question, "timestamp": now},
        {"role": "compass", "content": compass_text,
         "signal": result.get("signal", ""), "timestamp": now,
         "t_compass": result.get("t_compass", 0)},
        {"role": "assistant", "content": result.get("action_response", ""),
         "timestamp": now, "t_action": result.get("t_action", 0)},
    ])

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
    print("\n  Phenomenological Compass Server v0.9")
    print("  ────────────────────────────────────")
    print("  UI:      http://localhost:8420")
    print("  API:     http://localhost:8420/api/health")
    print("  Stream:  http://localhost:8420/api/stream")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8420)
