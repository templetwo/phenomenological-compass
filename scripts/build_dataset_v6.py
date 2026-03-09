"""
build_dataset_v6.py
===================
Merge v0.5 training data + v6 supplement files into data/training_v6/.
Deduplicates by question text, stratifies by signal, reports source stats.

Usage:
    python3 scripts/build_dataset_v6.py

Inputs:
    data/training_v5/train.jsonl          (v0.5 base - 121 examples)
    data/training_v5/valid.jsonl          (v0.5 base - 21 examples)
    data/supplements_v6/*.jsonl           (new broadening examples)

Outputs:
    data/training_v6/train.jsonl
    data/training_v6/valid.jsonl
"""

import json
import os
import random
import hashlib
from pathlib import Path

random.seed(42)

BASE_DIR = Path("/Users/tony_studio/phenomenological-compass")
V5_TRAIN = BASE_DIR / "data/training_v5/train.jsonl"
V5_VALID = BASE_DIR / "data/training_v5/valid.jsonl"
SUPPLEMENTS_DIR = BASE_DIR / "data/supplements_v6"
OUT_DIR = BASE_DIR / "data/training_v6"

VALID_FRACTION = 0.15  # 15% held out for validation


def extract_question(rec):
    """Get the user question text for deduplication."""
    for m in rec.get("messages", []):
        if m["role"] == "user":
            return m["content"].strip()
    return ""


def extract_signal(rec):
    """Get OPEN or WITNESS from assistant response."""
    for m in rec.get("messages", []):
        if m["role"] == "assistant":
            content = m["content"]
            if "SIGNAL: OPEN" in content:
                return "OPEN"
            elif "SIGNAL: WITNESS" in content:
                return "WITNESS"
    return "UNKNOWN"


def question_hash(q):
    """Normalize and hash question for dedup."""
    normalized = q.lower().strip()
    if normalized.startswith("task: "):
        normalized = normalized[6:]
    return hashlib.md5(normalized.encode()).hexdigest()


def validate_example(rec, source_name, idx):
    """Check that an example has the right structure."""
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
            if not (c.startswith("SIGNAL: OPEN") or c.startswith("SIGNAL: WITNESS")):
                errors.append(f"  [{source_name}:{idx}] Assistant doesn't start with SIGNAL:")

    signal = extract_signal(rec)
    if signal == "UNKNOWN":
        errors.append(f"  [{source_name}:{idx}] Could not extract OPEN/WITNESS signal")

    return errors


# ── Load base data ────────────────────────────────────────────────────────────
all_examples = []
seen_hashes = set()
source_stats = {}

print("Loading v0.5 base data...")
for path in [V5_TRAIN, V5_VALID]:
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            q = extract_question(rec)
            h = question_hash(q)
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_examples.append(rec)
source_stats["v0.5_base"] = len(all_examples)
print(f"  v0.5 base: {len(all_examples)} unique examples")

# ── Load supplements ──────────────────────────────────────────────────────────
supplement_files = sorted(SUPPLEMENTS_DIR.glob("*.jsonl"))
if not supplement_files:
    print(f"\nNo supplement files found in {SUPPLEMENTS_DIR}/")
    print("Place .jsonl files there and re-run.")
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

            # Validate
            errs = validate_example(rec, source_name, idx)
            all_errors.extend(errs)
            if errs:
                continue

            # Dedup
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

# ── Report errors ─────────────────────────────────────────────────────────────
if all_errors:
    print(f"\n{len(all_errors)} validation error(s):")
    for e in all_errors:
        print(e)
    print()

# ── Split by signal ───────────────────────────────────────────────────────────
open_examples = [d for d in all_examples if extract_signal(d) == "OPEN"]
witness_examples = [d for d in all_examples if extract_signal(d) == "WITNESS"]
unknown = [d for d in all_examples if extract_signal(d) == "UNKNOWN"]

random.shuffle(open_examples)
random.shuffle(witness_examples)

if unknown:
    print(f"\n{len(unknown)} examples with UNKNOWN signal (excluded)")

# Stratified split
n_open_valid = max(2, round(len(open_examples) * VALID_FRACTION))
n_witness_valid = max(2, round(len(witness_examples) * VALID_FRACTION))

open_valid = open_examples[:n_open_valid]
open_train = open_examples[n_open_valid:]
witness_valid = witness_examples[:n_witness_valid]
witness_train = witness_examples[n_witness_valid:]

train = open_train + witness_train
valid = open_valid + witness_valid
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
print(f"DATASET v0.6 SUMMARY")
print(f"{'='*55}")
print(f"  Total unique examples:  {len(train) + len(valid)}")
print(f"  Train:                  {len(train)} ({len(open_train)} OPEN, {len(witness_train)} WITNESS)")
print(f"  Valid:                  {len(valid)} ({len(open_valid)} OPEN, {len(witness_valid)} WITNESS)")
print(f"  OPEN/WITNESS ratio:     {len(open_examples)}:{len(witness_examples)}")
print(f"\n  Sources:")
for src, count in source_stats.items():
    print(f"    {src:<25} {count}")
print(f"\n-> Output: {OUT_DIR}/")
