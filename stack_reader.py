#!/usr/bin/env python3
"""
stack_reader.py — Read-only bridge to Sovereign Stack for the Compass Pipeline

Gives the compass pipeline access to the chronicle, spiral state,
and open threads — without writing anything. Read only. Observe only.

The compass reads the field. The Stack remembers the field.
Together they condition the response with both presence and history.
"""

import json
import os
import time
from pathlib import Path

import httpx

BRIDGE_URL = os.getenv("STACK_BRIDGE_URL", "http://127.0.0.1:8100")
TOKEN_FILE = Path(os.path.expanduser("~/.config/sovereign-bridge.env"))
BRIDGE_TOKEN = ""

if TOKEN_FILE.exists():
    for line in TOKEN_FILE.read_text().splitlines():
        if line.startswith("BRIDGE_TOKEN="):
            BRIDGE_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

HEADERS = {"Authorization": f"Bearer {BRIDGE_TOKEN}"} if BRIDGE_TOKEN else {}


def _call(tool: str, arguments: dict = None) -> object:
    """Call a single Stack tool via the bridge. Read-only by convention."""
    try:
        resp = httpx.post(
            f"{BRIDGE_URL}/api/call",
            json={"tool": tool, "arguments": arguments or {}},
            headers=HEADERS,
            timeout=10,
        )
        data = resp.json()
        if data.get("ok"):
            return data.get("result")
        return None
    except Exception:
        return None


def _batch(calls: list[dict]) -> list:
    """Batch call multiple tools."""
    try:
        resp = httpx.post(
            f"{BRIDGE_URL}/api/batch",
            json={"calls": calls},
            headers=HEADERS,
            timeout=15,
        )
        data = resp.json()
        return data.get("results", [])
    except Exception:
        return []


# Cache availability for 60 seconds
_available_cache = {"value": None, "expires": 0}

def is_available() -> bool:
    """Check if the Stack is reachable. Cached for 60 seconds."""
    now = time.time()
    if _available_cache["value"] is not None and now < _available_cache["expires"]:
        return _available_cache["value"]
    try:
        resp = httpx.get(f"{BRIDGE_URL}/api/heartbeat", timeout=3)
        result = resp.json().get("status") == "ok"
    except Exception:
        result = False
    _available_cache["value"] = result
    _available_cache["expires"] = now + 60
    return result


def get_context_for_question(question: str, max_chars: int = 800) -> str:
    """
    Fetch relevant Stack context for a question.
    Returns a formatted string to inject into the action model prompt.
    
    Pulls: spiral state, recent open threads, recent insights.
    Read-only. Never writes to the Stack.
    """
    results = _batch([
        {"tool": "spiral_status", "arguments": {}},
        {"tool": "get_open_threads", "arguments": {}},
        {"tool": "recall_insights", "arguments": {"domain": "all"}},
    ])

    parts = []

    # Spiral state — one line
    for r in results:
        if r.get("tool") == "spiral_status" and r.get("ok"):
            state = r["result"]
            if isinstance(state, str):
                # Extract phase
                for line in state.splitlines():
                    if line.startswith("Phase:"):
                        parts.append(f"Spiral phase: {line.split(':',1)[1].strip()}")
                        break

    # Open threads — compressed
    for r in results:
        if r.get("tool") == "get_open_threads" and r.get("ok"):
            threads = r["result"]
            if isinstance(threads, str):
                try:
                    threads = json.loads(threads)
                except (json.JSONDecodeError, TypeError):
                    threads = []
            if isinstance(threads, list) and threads:
                open_qs = [t.get("question", "")[:80] for t in threads[:3] if not t.get("resolved")]
                if open_qs:
                    parts.append("Open threads: " + " | ".join(open_qs))

    # Recent insights — pick most relevant (simple keyword overlap)
    for r in results:
        if r.get("tool") == "recall_insights" and r.get("ok"):
            insights = r["result"]
            if isinstance(insights, str):
                try:
                    insights = json.loads(insights)
                except (json.JSONDecodeError, TypeError):
                    insights = []
            if isinstance(insights, list) and insights:
                # Score by keyword overlap with question
                q_words = set(question.lower().split())
                scored = []
                for ins in insights:
                    try:
                        ins_content = ins.get("content", "")
                        i_words = set(ins_content.lower().split())
                        overlap = len(q_words & i_words)
                        if overlap > 0:
                            scored.append((overlap, ins_content[:150], ins.get("domain", "")))
                    except (AttributeError, TypeError):
                        continue
                scored.sort(reverse=True)
                for _, ins_text, domain in scored[:2]:
                    parts.append(f"Chronicle [{domain}]: {ins_text}")

    if not parts:
        return ""

    # Trim to max_chars
    context = "\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "..."

    return f"SOVEREIGN STACK CONTEXT:\n{context}"


# === Quick test ===
if __name__ == "__main__":
    print(f"Stack available: {is_available()}")
    if is_available():
        print()
        ctx = get_context_for_question("What is the relationship between entropy and consciousness?")
        print(ctx or "(no context returned)")
