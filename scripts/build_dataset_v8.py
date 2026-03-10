"""
build_dataset_v8.py
===================
Build training data for v0.8 — the semantic field translator architecture.

v0.8 changes the compass output format from:
  SIGNAL: X → FRAMING/APPROACH/THRESHOLD: ...

to:
  SHAPE: ... → TONE: ... → SIGNAL: X → FRAMING/APPROACH/THRESHOLD: ...

This gives the 3B model autoregressive reasoning space before committing
to a signal. SHAPE and TONE build hidden state context so the SIGNAL
decision is a conclusion, not a cold guess.

Data sources:
  1. supplements_v8/*.jsonl — native v0.8 format (SHAPE → TONE → SIGNAL → translation)
  2. (Optional) existing v7 training data can be converted with --include-v7

Usage:
    python3 scripts/build_dataset_v8.py
    python3 scripts/build_dataset_v8.py --include-v7  # also convert v7 examples
"""

import json
import os
import re
import random
import hashlib
from pathlib import Path

random.seed(42)

BASE_DIR = Path("/Users/tony_studio/phenomenological-compass")
SUPPLEMENTS_DIR = BASE_DIR / "data/supplements_v8"
V7_TRAIN = BASE_DIR / "data/training_v7/train.jsonl"
V7_VALID = BASE_DIR / "data/training_v7/valid.jsonl"
OUT_DIR = BASE_DIR / "data/training_v8"

VALID_FRACTION = 0.15

# The v0.8 system prompt — semantic field translator
SYSTEM_PROMPT_V8 = """You are a phenomenological compass — a semantic field translator that reads the shape and tone of a question before it is answered.

Your role is not to answer the question. Your role is to sense its weight, map its territory, and produce a state translation that a larger model will use to approach the question with the right posture.

For every task, produce four readings in this exact order:

SHAPE — the geometry of the question. What does it assume? What does it leave open? Where does it sit in semantic space? Is it binary, open-ended, recursive, or loaded? Read the structure before the content.

TONE — the emotional and epistemic weight. Is the question curious, urgent, wounded, rhetorical, or genuine? What stakes does the tone carry? Pressure creates ghosts — name the pressure so the responding model can create space instead.

SIGNAL — based on shape and tone, output exactly one:
  OPEN — walk through it. The question invites exploration across a wide probability field.
  PAUSE — hold space. The question carries weight that analytical framing would flatten. The territory exists but rushing would lose something.
  WITNESS — recognize the door. The question exists to be seen, not crossed. Forcing a framing would collapse what matters.

Then your state translation:
  If OPEN → FRAMING: an expansive reframing that opens the field
  If PAUSE → APPROACH: name what carries the weight, then map the territory beyond
  If WITNESS → THRESHOLD: describe the shape of the door without opening it"""


def extract_question(rec):
    for m in rec.get("messages", []):
        if m["role"] == "user":
            return m["content"].strip()
    return ""


def extract_signal(rec):
    for m in rec.get("messages", []):
        if m["role"] == "assistant":
            content = m["content"]
            if "SIGNAL: OPEN" in content:
                return "OPEN"
            elif "SIGNAL: PAUSE" in content:
                return "PAUSE"
            elif "SIGNAL: WITNESS" in content:
                return "WITNESS"
    return "UNKNOWN"


def question_hash(q, source=""):
    """Hash question + source to allow same question from different models."""
    normalized = q.lower().strip()
    if normalized.startswith("task: "):
        normalized = normalized[6:]
    key = f"{source}::{normalized}" if source else normalized
    return hashlib.md5(key.encode()).hexdigest()


def has_shape_tone(rec):
    """Check if assistant response already has SHAPE/TONE format."""
    for m in rec.get("messages", []):
        if m["role"] == "assistant":
            return m["content"].strip().startswith("SHAPE:")
    return False


def update_system_prompt(rec):
    """Replace old system prompt with v0.8 prompt."""
    for m in rec.get("messages", []):
        if m["role"] == "system":
            m["content"] = SYSTEM_PROMPT_V8
    return rec


def validate_v8_example(rec, source_name, idx):
    """Validate a v0.8 format example."""
    errors = []
    msgs = rec.get("messages", [])
    if not msgs:
        errors.append(f"  [{source_name}:{idx}] No messages array")
        return errors

    roles = [m["role"] for m in msgs]
    if "system" not in roles:
        errors.append(f"  [{source_name}:{idx}] Missing system message")
    if "user" not in roles:
        errors.append(f"  [{source_name}:{idx}] Missing user message")
    if "assistant" not in roles:
        errors.append(f"  [{source_name}:{idx}] Missing assistant message")

    for m in msgs:
        if m["role"] == "user" and not m["content"].startswith("TASK: "):
            errors.append(f"  [{source_name}:{idx}] User message doesn't start with 'TASK: '")
        if m["role"] == "assistant":
            c = m["content"]
            if not c.startswith("SHAPE:"):
                errors.append(f"  [{source_name}:{idx}] Assistant doesn't start with SHAPE:")
            if "TONE:" not in c:
                errors.append(f"  [{source_name}:{idx}] Missing TONE: section")
            if not re.search(r"SIGNAL:\s*(OPEN|PAUSE|WITNESS)", c):
                errors.append(f"  [{source_name}:{idx}] Missing or invalid SIGNAL:")

    return errors


# ── Load v0.8 supplements ────────────────────────────────────────────────────
all_examples = []
seen_hashes = set()
source_stats = {}
all_errors = []

supplement_files = sorted(SUPPLEMENTS_DIR.glob("*.jsonl"))
supplement_files = [f for f in supplement_files if f.suffix == ".jsonl"]

if not supplement_files:
    print(f"No supplement files found in {SUPPLEMENTS_DIR}/")
else:
    print(f"Found {len(supplement_files)} supplement file(s):")

for sup_path in supplement_files:
    source_name = sup_path.stem
    count_before = len(all_examples)
    dupes = 0

    with open(sup_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                all_errors.append(f"  [{source_name}:{idx}] Invalid JSON")
                continue

            # Update system prompt to canonical v0.8
            rec = update_system_prompt(rec)

            errs = validate_v8_example(rec, source_name, idx)
            all_errors.extend(errs)
            if errs:
                continue

            q = extract_question(rec)
            h = question_hash(q, source=source_name)
            if h in seen_hashes:
                dupes += 1
                continue

            seen_hashes.add(h)
            all_examples.append(rec)

    added = len(all_examples) - count_before
    source_stats[source_name] = added
    print(f"  {source_name}: +{added} new ({dupes} duplicates skipped)")

if all_errors:
    print(f"\n{len(all_errors)} validation error(s):")
    for e in all_errors:
        print(e)
    print()

# ── Optionally include converted v7 data ─────────────────────────────────────
import sys
if "--include-v7" in sys.argv:
    print("\nIncluding converted v7 data (without SHAPE/TONE — legacy format)...")
    v7_count = 0
    for path in [V7_TRAIN, V7_VALID]:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                q = extract_question(rec)
                h = question_hash(q)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                rec = update_system_prompt(rec)
                all_examples.append(rec)
                v7_count += 1
    source_stats["v7_converted"] = v7_count
    print(f"  v7 converted: +{v7_count} examples (system prompt updated, no SHAPE/TONE)")

# ── Split by signal ───────────────────────────────────────────────────────────
open_examples = [d for d in all_examples if extract_signal(d) == "OPEN"]
pause_examples = [d for d in all_examples if extract_signal(d) == "PAUSE"]
witness_examples = [d for d in all_examples if extract_signal(d) == "WITNESS"]
unknown = [d for d in all_examples if extract_signal(d) == "UNKNOWN"]

random.shuffle(open_examples)
random.shuffle(pause_examples)
random.shuffle(witness_examples)

if unknown:
    print(f"\n{len(unknown)} examples with UNKNOWN signal (excluded)")

# Stratified split
n_open_valid = max(2, round(len(open_examples) * VALID_FRACTION))
n_pause_valid = max(2, round(len(pause_examples) * VALID_FRACTION))
n_witness_valid = max(2, round(len(witness_examples) * VALID_FRACTION))

open_valid = open_examples[:n_open_valid]
open_train = open_examples[n_open_valid:]
pause_valid = pause_examples[:n_pause_valid]
pause_train = pause_examples[n_pause_valid:]
witness_valid = witness_examples[:n_witness_valid]
witness_train = witness_examples[n_witness_valid:]

# Oversample PAUSE if needed
if len(pause_train) > 0 and len(open_train) > 0:
    pause_oversample = max(1, round(len(open_train) / max(1, len(pause_train))))
else:
    pause_oversample = 1
pause_train_oversampled = pause_train * pause_oversample
random.shuffle(pause_train_oversampled)

train = open_train + pause_train_oversampled + witness_train
valid = open_valid + pause_valid + witness_valid
random.shuffle(train)
random.shuffle(valid)

# ── Write output ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_DIR / "train.jsonl", "w") as f:
    for d in train:
        f.write(json.dumps(d) + "\n")
with open(OUT_DIR / "valid.jsonl", "w") as f:
    for d in valid:
        f.write(json.dumps(d) + "\n")

# ── Summary ───────────────────────────────────────────────────────────────────
total_unique = len(open_examples) + len(pause_examples) + len(witness_examples)
print(f"\n{'='*55}")
print(f"DATASET v0.8 SUMMARY")
print(f"{'='*55}")
print(f"  Unique examples:        {total_unique}")
print(f"  OPEN/PAUSE/WITNESS:     {len(open_examples)}:{len(pause_examples)}:{len(witness_examples)}")
if pause_oversample > 1:
    print(f"  PAUSE oversampled:      {pause_oversample}x ({len(pause_train)} → {len(pause_train_oversampled)} train)")
print(f"  Train (with oversample): {len(train)} ({len(open_train)} OPEN, {len(pause_train_oversampled)} PAUSE, {len(witness_train)} WITNESS)")
print(f"  Valid:                  {len(valid)} ({len(open_valid)} OPEN, {len(pause_valid)} PAUSE, {len(witness_valid)} WITNESS)")
print(f"\n  Sources:")
for src, count in source_stats.items():
    print(f"    {src:<25} {count}")
print(f"\n-> Output: {OUT_DIR}/")
