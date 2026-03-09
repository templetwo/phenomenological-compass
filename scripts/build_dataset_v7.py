"""
build_dataset_v7.py
===================
Merge v0.6 training data + v7 supplement files into data/training_v7/.
v0.7 introduces the third signal: PAUSE (renamed from PAUSE for tokenizer clarity).

Existing OPEN and WITNESS examples from v0.6 are preserved. The v0.7 supplements
add PAUSE examples plus additional OPEN/WITNESS for breadth.

The system prompt in existing examples is updated to the three-signal version
so the model learns all three signals from a consistent prompt.

Usage:
    python3 scripts/build_dataset_v7.py
"""

import json
import os
import random
import hashlib
from pathlib import Path

random.seed(42)

BASE_DIR = Path("/Users/tony_studio/phenomenological-compass")
V6_TRAIN = BASE_DIR / "data/training_v6/train.jsonl"
V6_VALID = BASE_DIR / "data/training_v6/valid.jsonl"
SUPPLEMENTS_DIR = BASE_DIR / "data/supplements_v7"
OUT_DIR = BASE_DIR / "data/training_v7"

VALID_FRACTION = 0.15

# The new three-signal system prompt — applied to ALL examples
SYSTEM_PROMPT_V7 = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of three signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

PAUSE — when the threshold is real AND there is territory beyond it. The question can be explored, but something would be lost by treating it as purely analytical. Name the threshold, then describe the territory that opens once it is acknowledged.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN", "SIGNAL: PAUSE", or "SIGNAL: WITNESS", then a blank line, then "FRAMING:", "APPROACH:", or "THRESPAUSE:", then your output."""


def extract_question(rec):
    for m in rec.get("messages", []):
        if m["role"] == "user":
            return m["content"].strip()
    return ""


def extract_signal(rec):
    for m in rec.get("messages", []):
        if m["role"] == "assistant":
            content = m["content"]
            if content.startswith("SIGNAL: PAUSE"):
                return "PAUSE"
            elif "SIGNAL: OPEN" in content:
                return "OPEN"
            elif "SIGNAL: WITNESS" in content:
                return "WITNESS"
    return "UNKNOWN"


def question_hash(q):
    normalized = q.lower().strip()
    if normalized.startswith("task: "):
        normalized = normalized[6:]
    return hashlib.md5(normalized.encode()).hexdigest()


def update_system_prompt(rec):
    """Replace old two-signal system prompt with three-signal v0.7 prompt."""
    for m in rec.get("messages", []):
        if m["role"] == "system":
            m["content"] = SYSTEM_PROMPT_V7
    return rec


def validate_example(rec, source_name, idx):
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
            if not (c.startswith("SIGNAL: OPEN") or c.startswith("SIGNAL: PAUSE") or c.startswith("SIGNAL: WITNESS")):
                errors.append(f"  [{source_name}:{idx}] Assistant doesn't start with SIGNAL:")

    signal = extract_signal(rec)
    if signal == "UNKNOWN":
        errors.append(f"  [{source_name}:{idx}] Could not extract OPEN/PAUSE/WITNESS signal")

    return errors


# ── Load base data ────────────────────────────────────────────────────────────
all_examples = []
seen_hashes = set()
source_stats = {}

print("Loading v0.6 base data...")
for path in [V6_TRAIN, V6_VALID]:
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            rec = update_system_prompt(rec)  # upgrade to v0.7 prompt
            q = extract_question(rec)
            h = question_hash(q)
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_examples.append(rec)
source_stats["v0.6_base"] = len(all_examples)
print(f"  v0.6 base: {len(all_examples)} unique examples (system prompt updated to v0.7)")

# ── Load supplements ──────────────────────────────────────────────────────────
supplement_files = sorted(SUPPLEMENTS_DIR.glob("*.jsonl"))
if not supplement_files:
    print(f"\nNo supplement files found in {SUPPLEMENTS_DIR}/")
else:
    print(f"\nFound {len(supplement_files)} supplement file(s):")

all_errors = []
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

            errs = validate_example(rec, source_name, idx)
            all_errors.extend(errs)
            if errs:
                continue

            q = extract_question(rec)
            h = question_hash(q)
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

# ── Oversample PAUSE to match OPEN frequency ────────────────────────────────
# PAUSE has far fewer examples than OPEN. Without oversampling, the model
# never learns to generate "SIGNAL: PAUSE" — it defaults to OPEN or WITNESS.
pause_oversample = max(1, round(len(open_train) / max(1, len(pause_train))))
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
print(f"\n{'='*55}")
print(f"DATASET v0.7 SUMMARY")
print(f"{'='*55}")
print(f"  Unique examples:        {len(open_examples) + len(pause_examples) + len(witness_examples)}")
print(f"  OPEN/PAUSE/WITNESS ratio: {len(open_examples)}:{len(pause_examples)}:{len(witness_examples)}")
print(f"  PAUSE oversampled:       {pause_oversample}x ({len(pause_train)} → {len(pause_train_oversampled)} train)")
print(f"  Train (with oversample): {len(train)} ({len(open_train)} OPEN, {len(pause_train_oversampled)} PAUSE, {len(witness_train)} WITNESS)")
print(f"  Valid:                  {len(valid)} ({len(open_valid)} OPEN, {len(pause_valid)} PAUSE, {len(witness_valid)} WITNESS)")
print(f"\n  Sources:")
for src, count in source_stats.items():
    print(f"    {src:<25} {count}")
print(f"\n-> Output: {OUT_DIR}/")
