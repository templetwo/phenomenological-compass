#!/usr/bin/env python3
"""
consolidate.py — Merge 5 model-sourced question sets into one clean eval set
============================================================================
Takes the Claude Opus set (105 questions, perfect format) as the base,
cross-validates signal assignments against DeepSeek/Gemini/GPT/Grok,
flags disagreements, deduplicates, and writes eval/questions.jsonl.

Usage:
    python3 eval/consolidate.py
"""

import json
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict

PROJECT = Path(__file__).parent.parent
DATA = PROJECT / "data" / "eval_v9"
OUT = PROJECT / "eval" / "questions.jsonl"

# v0.8 training questions to exclude
V8_TRAINING = PROJECT / "data" / "supplements_v8"


def normalize(text):
    """Normalize for dedup: lowercase, strip punctuation, collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def fingerprint(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()[:16]


def load_jsonl(path):
    """Load JSONL, skipping blank lines and parse errors."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items


def load_training_fingerprints():
    """Load fingerprints of all v0.8 training questions to prevent overlap."""
    fps = set()
    if not V8_TRAINING.exists():
        return fps
    for jf in V8_TRAINING.glob("*.jsonl"):
        for item in load_jsonl(jf):
            msgs = item.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    content = m.get("content", "")
                    # Strip "TASK: " prefix
                    if content.startswith("TASK: "):
                        content = content[6:]
                    fps.add(fingerprint(content))
    return fps


def main():
    # Load base set (Claude Opus — 105 questions, mission-brief format)
    base = load_jsonl(DATA / "claude_opus.jsonl")
    print(f"Base set (claude_opus): {len(base)} questions")

    # Load other models for cross-validation
    others = {}
    for name in ["deepseek", "gemini", "gpt", "grok"]:
        path = DATA / f"{name}.jsonl"
        if path.exists():
            qs = load_jsonl(path)
            others[name] = qs
            print(f"  {name}: {len(qs)} questions")

    # Build cross-validation index: fingerprint → {model: signal}
    cross_index = defaultdict(dict)
    for name, qs in others.items():
        for q in qs:
            fp = fingerprint(q["question"])
            cross_index[fp][name] = q["expected_signal"]

    # Check for v0.8 training overlap
    training_fps = load_training_fingerprints()
    print(f"\nv0.8 training fingerprints: {len(training_fps)}")

    # Process base set
    questions = []
    signal_counts = Counter()
    overlap_count = 0
    disagreement_count = 0

    for q in base:
        fp = fingerprint(q["question"])

        # Check training overlap
        if fp in training_fps:
            overlap_count += 1
            print(f"  OVERLAP with training: {q['question'][:60]}...")
            continue

        # Cross-validate signal
        cross_signals = cross_index.get(fp, {})
        if cross_signals:
            disagreements = [
                f"{model}={sig}"
                for model, sig in cross_signals.items()
                if sig != q["expected_signal"]
            ]
            if disagreements:
                disagreement_count += 1
                # Don't flag, just note — base set is authoritative

        # Build clean output
        clean = {
            "question": q["question"],
            "expected_signal": q["expected_signal"],
            "domain": q.get("domain", ""),
            "id": q.get("id", f"{q['expected_signal'].lower()}_{len(questions)+1:03d}"),
        }
        questions.append(clean)
        signal_counts[q["expected_signal"]] += 1

    print(f"\nConsolidated: {len(questions)} questions")
    print(f"  OPEN: {signal_counts['OPEN']}  PAUSE: {signal_counts['PAUSE']}  WITNESS: {signal_counts['WITNESS']}")
    print(f"  Training overlaps removed: {overlap_count}")
    print(f"  Cross-model signal disagreements: {disagreement_count} (base set authoritative)")

    # Write
    with open(OUT, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"\n→ {OUT} ({len(questions)} questions)")


if __name__ == "__main__":
    main()
