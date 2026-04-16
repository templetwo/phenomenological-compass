#!/usr/bin/env python3
"""
compass_server.py — Web API for the Phenomenological Compass pipeline
=====================================================================
Wraps pipeline.py as a FastAPI server with session memory and SSE streaming.

Endpoints:
    GET  /                              — Serve UI
    GET  /api/health                    — Pipeline status
    POST /api/infer                     — Non-streaming inference (backward compat)
    POST /api/stream                    — SSE streaming inference with per-token entropy
    GET  /api/sessions                  — List sessions
    POST /api/sessions                  — Create session
    PATCH /api/sessions/{id}            — Rename session
    DELETE /api/sessions/{id}           — Delete session
    GET  /api/sessions/{id}/messages    — Fetch message history
    GET  /api/sessions/{id}/export      — Export session as Markdown

Usage:
    cd ~/phenomenological-compass/phenomenological-compass-ui
    source ~/phenomenological-compass/.venv/bin/activate
    HF_HOME=~/.cache/huggingface_local python3 compass_server.py

Proxy mode (no local model required):
    COMPASS_UPSTREAM=http://remote-host:8420 python3 compass_server.py
"""

import os
import re
import sys
import json
import math
import time
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("compass")

# ── Environment ──────────────────────────────────────────────────────────────

# If set, this server operates in proxy mode: it forwards all inference
# requests to the upstream and does not load any local model.
UPSTREAM_URL = os.environ.get("COMPASS_UPSTREAM")

# Configurable port (default 8420 for backward compatibility)
PORT = int(os.environ.get("PORT", 8420))

# ── Conditional Imports ───────────────────────────────────────────────────────

# Add project root to path regardless of mode (needed for pipeline in local mode)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

if UPSTREAM_URL:
    if httpx is None:
        log.error("COMPASS_UPSTREAM is set but httpx is not installed. Run: pip install httpx")
        sys.exit(1)
    log.info("Proxy mode enabled — upstream: %s", UPSTREAM_URL)
else:
    log.info("Local mode — will load pipeline on first request")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel, field_validator

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Phenomenological Compass", version="1.0")
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
    """Load all persisted session files from disk into the in-memory store."""
    global sessions
    loaded = 0
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            sessions[f.stem] = data
            loaded += 1
        except Exception as exc:
            log.warning("Failed to load session %s: %s", f.stem, exc)
    log.info("Loaded %d session(s) from disk", loaded)


def save_session(session_id: str):
    """Persist a single session to disk as JSON."""
    if session_id in sessions:
        path = SESSIONS_DIR / f"{session_id}.json"
        path.write_text(json.dumps(sessions[session_id], indent=2))


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Return an existing session ID or create and persist a new one."""
    if session_id and session_id in sessions:
        return session_id
    new_id = str(uuid.uuid4())[:8]
    sessions[new_id] = {
        "messages": [],
        "created": datetime.now().isoformat(),
        "title": "New Session",
    }
    save_session(new_id)
    log.info("Created new session: %s", new_id)
    return new_id


# ── Pipeline Singleton ───────────────────────────────────────────────────────

pipeline_instance = None


def detect_adapter():
    """Find the latest adapter directory and its highest-numbered checkpoint.

    Scans sibling ``adapters_v*`` directories relative to the project root,
    picks the one with the highest version name, then finds the highest-numbered
    ``*_adapters.safetensors`` checkpoint file within it.

    Returns:
        tuple[str | None, str | None]: (adapter_dir_path, checkpoint_filename)
            Both values are None when no adapters are found.
    """
    root = Path(__file__).parent.parent
    adapter_dirs = sorted(root.glob("adapters_v*"), key=lambda p: p.name)
    if not adapter_dirs:
        log.info("No adapter directories found under %s", root)
        return None, None
    best_dir = adapter_dirs[-1]
    checkpoints = sorted(best_dir.glob("*_adapters.safetensors"))
    best_cp = checkpoints[-1].name if checkpoints else None
    log.info(
        "Detected adapter dir: %s  checkpoint: %s",
        best_dir.name,
        best_cp or "(none)",
    )
    return str(best_dir), best_cp


def get_pipeline():
    """Return the shared Pipeline singleton, loading it on first call.

    Raises:
        RuntimeError: If called in proxy mode (UPSTREAM_URL is set).
    """
    global pipeline_instance
    if UPSTREAM_URL:
        raise RuntimeError(
            "get_pipeline() must not be called in proxy mode. "
            "Set COMPASS_UPSTREAM env var only when forwarding to a remote host."
        )
    if pipeline_instance is None:
        log.info("Loading compass pipeline (this takes ~30s on first load)...")
        from pipeline import Pipeline
        adapter_dir, adapter_cp = detect_adapter()
        try:
            pipeline_instance = Pipeline(
                adapter_path=adapter_dir,
                adapter_checkpoint=adapter_cp,
            )
        except TypeError:
            # Pipeline does not yet accept adapter kwargs — fall back gracefully
            log.warning(
                "Pipeline() does not accept adapter kwargs; loading without adapter"
            )
            pipeline_instance = Pipeline()
        log.info("Pipeline ready.")
    return pipeline_instance


# ── Entropy Computation ─────────────────────────────────────────────────────

def compute_entropy(logprobs_array):
    """Compute Shannon entropy in nats from an MLX log-probability array.

    Args:
        logprobs_array: MLX array of log-probabilities for the vocabulary.

    Returns:
        float: Entropy value rounded to 4 decimal places.
    """
    try:
        import mlx.core as mx
        probs = mx.exp(logprobs_array)
        entropy = -mx.sum(probs * logprobs_array)
        mx.eval(entropy)
        return round(float(entropy.item()), 4)
    except ImportError:
        # mlx not available (proxy mode or non-Apple hardware)
        return None


# ── Think-Tag State Machine ─────────────────────────────────────────────────

class ThinkDetector:
    """Detect ``<think>...</think>`` blocks in a streaming token sequence.

    The detector maintains a state machine with three states:

    - ``pending``: No tags seen yet; buffering until intent is clear.
    - ``thinking``: Inside a ``<think>`` block.
    - ``response``: Past the closing ``</think>``; emitting action tokens.
    """

    def __init__(self):
        self.state = "pending"  # pending | thinking | response
        self.buffer = ""

    def feed(self, token_text: str):
        """Feed a single token into the detector.

        Args:
            token_text: Raw token string from the model.

        Returns:
            tuple[str | None, str]: (stage, text_to_emit).
                ``stage`` is ``'thinking'``, ``'action'``, or ``None`` (still
                buffering). ``text_to_emit`` is the text to forward downstream.
        """
        self.buffer += token_text

        if self.state == "pending":
            if "<think>" in self.buffer:
                self.state = "thinking"
                after = self.buffer.split("<think>", 1)[1]
                return "thinking", after
            # If we see </think> without opening (Qwen sometimes skips it)
            if "</think>" in self.buffer:
                after = self.buffer.split("</think>", 1)[1]
                self.state = "response"
                # The 'before' was thinking, 'after' is response
                return "action", after.lstrip()
            # If enough tokens without any tag, it's a direct response
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
            # Clean stray tokens injected by some chat templates
            clean = token_text.replace("<|im_end|>", "").replace("<|im_start|>", "")
            return "action", clean


# ── Compass Section Detector ────────────────────────────────────────────────

COMPASS_SECTIONS = ["SHAPE", "TONE", "SIGNAL", "FRAMING", "APPROACH", "THRESHOLD"]


def detect_compass_section(accumulated_text: str) -> Optional[str]:
    """Return the current section label based on accumulated compass text.

    Uses ``rfind`` to identify the *last* section header seen, so the label
    always reflects the deepest position in the reading.

    Args:
        accumulated_text: All compass tokens received so far.

    Returns:
        str | None: Lower-cased section name (e.g. ``'signal'``) or ``None``.
    """
    last_section = None
    last_pos = -1
    for section in COMPASS_SECTIONS:
        for prefix in [f"\n{section}:", f"{section}:"]:
            pos = accumulated_text.rfind(prefix)
            if pos != -1:
                # For "\n" prefix, the actual position is after the newline
                if prefix.startswith("\n"):
                    pos += 1
                # Only match if it's at start of text or after newline
                if pos == 0 or accumulated_text[pos - 1] == "\n":
                    if pos > last_pos:
                        last_pos = pos
                        last_section = section.lower()
    return last_section


def detect_signal(accumulated_text: str) -> Optional[str]:
    """Extract the SIGNAL value from accumulated compass text.

    Args:
        accumulated_text: All compass tokens received so far.

    Returns:
        str | None: One of ``'OPEN'``, ``'PAUSE'``, ``'WITNESS'``, or ``None``.
    """
    m = re.search(r"SIGNAL:\s*(OPEN|PAUSE|WITNESS)", accumulated_text, re.IGNORECASE)
    return m.group(1).upper() if m else None


# ── Context Threading ─────────────────────────────────────────────────────────

def build_context(session_id: str, max_turns: int = 3) -> str:
    """Build a compact prior-context string from the session's recent exchanges.

    Reads back through the session messages in groups of three
    (user / compass / assistant) and summarises each user turn with its
    detected signal. Returns an empty string when there is no history.

    Args:
        session_id: The session to pull history from.
        max_turns: Maximum number of prior turns to include.

    Returns:
        str: A ``[PRIOR CONTEXT]`` block to prepend to the current question,
            or an empty string when the session has no prior messages.
    """
    if session_id not in sessions:
        return ""
    msgs = sessions[session_id]["messages"]
    if not msgs:
        return ""

    turns = []
    # Messages are stored in triples: user, compass, assistant
    for i in range(0, len(msgs) - 2, 3):
        if msgs[i]["role"] == "user":
            signal = ""
            if i + 1 < len(msgs):
                signal = msgs[i + 1].get("signal", "")
            question_snippet = msgs[i]["content"][:100]
            turns.append(f"Q: {question_snippet} [{signal}]")

    recent = turns[-max_turns:]
    if recent:
        return "[PRIOR CONTEXT]\n" + "\n".join(recent) + "\n[END CONTEXT]\n\n"
    return ""


# ── API Models ───────────────────────────────────────────────────────────────

VALID_MODES = {"routed", "raw", "compare"}
MAX_QUESTION_LEN = 2000


class InferenceRequest(BaseModel):
    """Request body for both streaming and non-streaming inference endpoints."""

    question: str
    session_id: Optional[str] = None
    mode: str = "routed"  # "routed" | "raw" | "compare"

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        """Reject blank questions before any processing begins."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v

    @field_validator("question")
    @classmethod
    def question_length_limit(cls, v: str) -> str:
        """Enforce the 2000-character hard limit."""
        if len(v) > MAX_QUESTION_LEN:
            raise ValueError(
                f"Question exceeds {MAX_QUESTION_LEN} character limit "
                f"(received {len(v)} chars)"
            )
        return v

    @field_validator("mode")
    @classmethod
    def mode_must_be_valid(cls, v: str) -> str:
        """Reject unknown inference modes."""
        if v not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(VALID_MODES)!r}, got {v!r}"
            )
        return v


class SessionRenameRequest(BaseModel):
    """Request body for the session rename (PATCH) endpoint."""

    title: str


# ── SSE Helpers ──────────────────────────────────────────────────────────────

def sse_event(data: dict) -> str:
    """Encode a dictionary as a Server-Sent Event string.

    Args:
        data: Arbitrary JSON-serialisable dictionary.

    Returns:
        str: A properly formatted SSE ``data:`` line with trailing double newline.
    """
    return f"data: {json.dumps(data)}\n\n"


# ── Proxy Helpers ─────────────────────────────────────────────────────────────

async def proxy_stream(req: InferenceRequest) -> StreamingResponse:
    """Forward a streaming inference request to the upstream server.

    Opens an async SSE connection to ``UPSTREAM_URL/api/stream`` and re-emits
    every ``data:`` line verbatim so the browser sees an identical event stream.
    Session management (session creation, history) remains local.

    Args:
        req: The validated inference request from the client.

    Returns:
        StreamingResponse: An SSE response relaying the upstream event stream.
    """
    sid = get_or_create_session(req.session_id)

    async def event_generator():
        log.info("Proxying stream to %s for session %s", UPSTREAM_URL, sid)
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST",
                    f"{UPSTREAM_URL}/api/stream",
                    json={"question": req.question, "mode": req.mode},
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            yield line + "\n\n"
        except httpx.RequestError as exc:
            log.error("Proxy stream error: %s", exc)
            yield sse_event({"stage": "error", "message": f"Upstream unavailable: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def proxy_infer(req: InferenceRequest):
    """Forward a non-streaming inference request to the upstream server.

    Args:
        req: The validated inference request from the client.

    Returns:
        dict: The upstream JSON response, passed through unchanged.

    Raises:
        HTTPException: If the upstream is unreachable or returns an error.
    """
    log.info("Proxying infer to %s", UPSTREAM_URL)
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{UPSTREAM_URL}/api/infer",
                json={"question": req.question, "mode": req.mode},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.RequestError as exc:
        log.error("Proxy infer error: %s", exc)
        raise HTTPException(502, f"Upstream unavailable: {exc}")
    except httpx.HTTPStatusError as exc:
        log.error("Proxy upstream error %s: %s", exc.response.status_code, exc)
        raise HTTPException(502, f"Upstream returned {exc.response.status_code}")


# ── API Routes ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load persisted sessions from disk when the server starts."""
    load_sessions()


@app.get("/")
async def serve_ui():
    """Serve the main UI HTML file."""
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>Compass UI not found. Place index.html in ./ui/</h1>")


@app.get("/api/health")
async def health():
    """Return server readiness, version, session count, and operating mode.

    In proxy mode the response includes the upstream URL and attempts to
    check whether the upstream is reachable.
    """
    if UPSTREAM_URL:
        upstream_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{UPSTREAM_URL}/api/health")
                upstream_status = resp.json().get("status", "unknown")
        except Exception:
            upstream_status = "unreachable"
        return {
            "status": "proxy",
            "version": "1.0",
            "sessions": len(sessions),
            "mode": "proxy",
            "upstream": UPSTREAM_URL,
            "upstream_status": upstream_status,
        }

    return {
        "status": "ready" if pipeline_instance else "loading",
        "version": "1.0",
        "sessions": len(sessions),
        "mode": "local",
        "upstream": None,
    }


@app.get("/api/sessions")
async def list_sessions():
    """Return a summary of all sessions sorted newest-first."""
    return {
        sid: {
            "title": s["title"],
            "created": s["created"],
            "message_count": len(s["messages"]),
            "last_message": (
                s["messages"][-1]["timestamp"] if s["messages"] else s["created"]
            ),
        }
        for sid, s in sorted(
            sessions.items(), key=lambda x: x[1]["created"], reverse=True
        )
    }


@app.post("/api/sessions")
async def create_session():
    """Create a new empty session and return its ID."""
    sid = get_or_create_session()
    return {"session_id": sid}


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: SessionRenameRequest):
    """Rename a session title.

    Args:
        session_id: The session to rename.
        req: The new title.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sessions[session_id]["title"] = req.title
    save_session(session_id)
    log.info("Renamed session %s -> %r", session_id, req.title)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its persisted file.

    Args:
        session_id: The session to delete.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    del sessions[session_id]
    path = SESSIONS_DIR / f"{session_id}.json"
    path.unlink(missing_ok=True)
    log.info("Deleted session %s", session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    """Return the full message list for a session.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return {"messages": sessions[session_id]["messages"]}


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str):
    """Export a session as a human-readable Markdown document.

    Each exchange (user question, compass reading, assistant response) is
    rendered as a numbered turn with timing metadata. The response uses
    ``Content-Type: text/markdown`` so browsers and editors recognise it.

    Args:
        session_id: The session to export.

    Returns:
        PlainTextResponse: A Markdown-formatted document.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    title = session.get("title", "Untitled Session")
    created = session.get("created", "")
    msgs = session.get("messages", [])

    lines = [f"# {title}", f"*{created}*", "", "---", ""]

    turn_number = 0
    i = 0
    while i < len(msgs):
        # Expect groups of three: user, compass, assistant
        user_msg = msgs[i] if i < len(msgs) and msgs[i]["role"] == "user" else None
        compass_msg = (
            msgs[i + 1]
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "compass"
            else None
        )
        action_msg = (
            msgs[i + 2]
            if i + 2 < len(msgs) and msgs[i + 2]["role"] == "assistant"
            else None
        )

        if user_msg is None:
            # Malformed message — skip
            i += 1
            continue

        turn_number += 1
        lines.append(f"## Turn {turn_number}")
        lines.append("")
        lines.append(f"**You:** {user_msg['content']}")
        lines.append("")

        if compass_msg:
            signal = compass_msg.get("signal", "").upper() or "—"
            lines.append(f"### Compass — {signal}")
            lines.append("")
            lines.append(compass_msg.get("content", "").strip())
            lines.append("")

        if action_msg:
            lines.append("### Response")
            lines.append("")
            lines.append(action_msg.get("content", "").strip())
            lines.append("")

        # Timing footer
        t_compass = compass_msg.get("t_compass", 0) if compass_msg else 0
        t_action = action_msg.get("t_action", 0) if action_msg else 0
        lines.append(f"*Compass: {t_compass}s · Response: {t_action}s*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Advance by 3 (full turn) or as far as we have
        step = 1
        if action_msg:
            step = 3
        elif compass_msg:
            step = 2
        i += step

    markdown_text = "\n".join(lines)
    log.info("Exported session %s (%d turn(s))", session_id, turn_number)

    return PlainTextResponse(
        content=markdown_text,
        media_type="text/markdown",
        headers={
            "Content-Disposition": (
                f'attachment; filename="compass-session-{session_id}.md"'
            )
        },
    )


# ── Streaming Endpoint ──────────────────────────────────────────────────────

@app.post("/api/stream")
async def stream_infer(req: InferenceRequest):
    """SSE streaming endpoint. Yields per-token events with entropy.

    In proxy mode this transparently forwards the request to the upstream
    server. In local mode it runs the compass pipeline and action model,
    emitting events for each stage: ``compass``, ``signal_lock``,
    ``compass_done``, ``thinking``, ``action``, and ``done``.

    Args:
        req: Validated inference request (question, session_id, mode).

    Returns:
        StreamingResponse: An SSE event stream.

    Raises:
        HTTPException: 422 if the question is empty or too long.
    """
    # Guard clauses (Pydantic validators already catch these, but explicit
    # checks here produce cleaner HTTPException payloads for API consumers)
    if not req.question.strip():
        raise HTTPException(422, "Question cannot be empty")
    if len(req.question) > MAX_QUESTION_LEN:
        raise HTTPException(
            422,
            f"Question exceeds {MAX_QUESTION_LEN} character limit "
            f"(received {len(req.question)} chars)",
        )

    if UPSTREAM_URL:
        return await proxy_stream(req)

    async def event_generator():
        async with _stream_lock:
            pipe = get_pipeline()
            sid = get_or_create_session(req.session_id)
            loop = asyncio.get_event_loop()

            # Prepend prior context when the session has history
            context_prefix = build_context(sid)
            effective_question = context_prefix + req.question
            if context_prefix:
                log.info(
                    "Session %s: prepending %d-char context prefix",
                    sid,
                    len(context_prefix),
                )

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
                    for text, logprobs, finish, tps in pipe.stream_classify(
                        effective_question
                    ):
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
                    gen = pipe.stream_raw(effective_question)
                else:
                    gen = pipe.stream_act(effective_question, signal, compass_text)
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
                    except Exception as exc:
                        log.debug("Entropy computation failed: %s", exc)

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
            # Clean response text — truncate at hallucinated continuations
            for stop_tag in ["<|im_start|>", "<|im_end|>"]:
                if stop_tag in response_text:
                    response_text = response_text[:response_text.index(stop_tag)]
            # Strip any remaining think tags
            import re as _re
            response_text = _re.sub(r"</?think>", "", response_text).strip()
            # If response is duplicated (thinking leaked or model repeated itself),
            # check every paragraph boundary for an exact-half split.
            if len(response_text) > 100:
                parts = response_text.split("\n\n")
                for i in range(1, len(parts)):
                    first = "\n\n".join(parts[:i]).strip()
                    second = "\n\n".join(parts[i:]).strip()
                    if first == second:
                        response_text = first
                        log.info("Dedup: removed exact duplicate (%d chars)", len(first))
                        break

            mean_entropy = (
                round(sum(entropy_values) / len(entropy_values), 4)
                if entropy_values
                else None
            )

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

            # Save to session (store original question, not context-prefixed)
            now = datetime.now().isoformat()
            user_msg = {"role": "user", "content": req.question, "timestamp": now}
            compass_msg = {
                "role": "compass",
                "content": compass_text.strip(),
                "signal": signal or "",
                "timestamp": now,
                "t_compass": t_compass,
            }
            action_msg = {
                "role": "assistant",
                "content": response_text,
                "timestamp": now,
                "t_action": t_action,
            }
            sessions[sid]["messages"].extend([user_msg, compass_msg, action_msg])

            if len(sessions[sid]["messages"]) <= 3:
                sessions[sid]["title"] = req.question[:60] + (
                    "..." if len(req.question) > 60 else ""
                )

            save_session(sid)
            log.info(
                "Session %s: stream complete (signal=%s, t_compass=%.1fs, "
                "t_action=%.1fs, mean_entropy=%s)",
                sid, signal, t_compass, t_action, mean_entropy,
            )

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
    """Non-streaming inference endpoint (backward-compatible).

    Runs the full pipeline synchronously and returns a single JSON response.
    In proxy mode the request is forwarded to the upstream server. Context
    threading is applied identically to the streaming endpoint.

    Args:
        req: Validated inference request (question, session_id, mode).

    Returns:
        dict: Inference results including signal, compass reading, and response.

    Raises:
        HTTPException: 422 if the question is empty or too long.
    """
    if not req.question.strip():
        raise HTTPException(422, "Question cannot be empty")
    if len(req.question) > MAX_QUESTION_LEN:
        raise HTTPException(
            422,
            f"Question exceeds {MAX_QUESTION_LEN} character limit "
            f"(received {len(req.question)} chars)",
        )

    if UPSTREAM_URL:
        return await proxy_infer(req)

    pipe = get_pipeline()
    sid = get_or_create_session(req.session_id)

    # Prepend prior context when the session has history
    context_prefix = build_context(sid)
    effective_question = context_prefix + req.question
    if context_prefix:
        log.info(
            "Session %s: prepending %d-char context prefix",
            sid,
            len(context_prefix),
        )

    t0 = time.time()

    if req.mode == "raw":
        response_text, elapsed, thinking = pipe.raw(effective_question)
        result = {
            "signal": None,
            "compass_response": None,
            "action_response": response_text,
            "thinking": thinking,
            "t_compass": 0,
            "t_action": elapsed,
        }
    elif req.mode == "compare":
        routed = pipe.run(effective_question)
        raw_text, raw_elapsed, raw_thinking = pipe.raw(effective_question)
        result = {
            **routed,
            "raw_response": raw_text,
            "raw_thinking": raw_thinking,
            "t_raw": raw_elapsed,
        }
    else:
        result = pipe.run(effective_question)

    total_time = time.time() - t0
    compass_text = result.get("compass_response", "") or ""

    now = datetime.now().isoformat()
    sessions[sid]["messages"].extend([
        {"role": "user", "content": req.question, "timestamp": now},
        {
            "role": "compass",
            "content": compass_text,
            "signal": result.get("signal", ""),
            "timestamp": now,
            "t_compass": result.get("t_compass", 0),
        },
        {
            "role": "assistant",
            "content": result.get("action_response", ""),
            "timestamp": now,
            "t_action": result.get("t_action", 0),
        },
    ])

    if len(sessions[sid]["messages"]) <= 3:
        sessions[sid]["title"] = req.question[:60] + (
            "..." if len(req.question) > 60 else ""
        )

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

    log.info(
        "Session %s: infer complete (mode=%s, signal=%s, t_total=%.1fs)",
        sid, req.mode, response["signal"], total_time,
    )
    return response


# ── Serve Static UI ──────────────────────────────────────────────────────────

ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    mode_label = f"PROXY -> {UPSTREAM_URL}" if UPSTREAM_URL else "LOCAL (pipeline)"
    log.info("Phenomenological Compass Server v1.0")
    log.info("Mode:    %s", mode_label)
    log.info("UI:      http://localhost:%d", PORT)
    log.info("API:     http://localhost:%d/api/health", PORT)
    log.info("Stream:  http://localhost:%d/api/stream", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
