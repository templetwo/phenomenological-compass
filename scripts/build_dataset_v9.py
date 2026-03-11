"""
build_dataset_v9.py
===================
Build training data for v0.9 — WITNESS augmentation + contrastive pairs.

Sources v8 supplements (186 examples) + v9 supplements (WITNESS-heavy).
Same format, same system prompt, same dedup logic.

Usage:
    python3 scripts/build_dataset_v9.py
"""

import json
import os
import re
import random
import hashlib
from pathlib import Path

random.seed(42)

BASE_DIR = Path("/Users/tony_studio/phenomenological-compass")
SUPPLEMENT_DIRS = [
    BASE_DIR / "data/supplements_v8",
    BASE_DIR / "data/supplements_v9",
]
OUT_DIR = BASE_DIR / "data/training_v9"

VALID_FRACTION = 0.15

# The v0.8 system prompt — unchanged for v0.9
SYSTEM_PROMPT = """You are a phenomenological compass — a semantic field translator that reads the shape and tone of a question before it is answered.

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


def update_system_prompt(rec):
    """Replace system prompt with canonical version."""
    for m in rec.get("messages", []):
        if m["role"] == "system":
            m["content"] = SYSTEM_PROMPT
    return rec


def validate_example(rec, source_name, idx):
    """Validate a v0.8/v0.9 format example."""
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
            # Check for THRESPAUSE corruption
            if "THRESPAUSE" in c.upper():
                errors.append(f"  [{source_name}:{idx}] THRESPAUSE corruption detected!")

    return errors


# ── Load from all supplement directories ─────────────────────────────────────
all_examples = []
seen_hashes = set()
source_stats = {}
all_errors = []

for sup_dir in SUPPLEMENT_DIRS:
    if not sup_dir.exists():
        print(f"WARNING: {sup_dir} does not exist, skipping")
        continue

    supplement_files = sorted(sup_dir.glob("*.jsonl"))
    if not supplement_files:
        print(f"No JSONL files in {sup_dir}/")
        continue

    print(f"\nLoading from {sup_dir.name}/:")

    for sup_path in supplement_files:
        source_name = f"{sup_dir.name}/{sup_path.stem}"
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

                rec = update_system_prompt(rec)

                errs = validate_example(rec, source_name, idx)
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
        print(f"  {sup_path.stem:<25} +{added} new ({dupes} dupes skipped)")

if all_errors:
    print(f"\n{len(all_errors)} validation error(s):")
    for e in all_errors:
        print(e)
    print()

# ── Split by signal ──────────────────────────────────────────────────────────
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

train = open_train + pause_train + witness_train
valid = open_valid + pause_valid + witness_valid
random.shuffle(train)
random.shuffle(valid)

# ── Write output ─────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_DIR / "train.jsonl", "w") as f:
    for d in train:
        f.write(json.dumps(d) + "\n")
with open(OUT_DIR / "valid.jsonl", "w") as f:
    for d in valid:
        f.write(json.dumps(d) + "\n")

# ── Summary ──────────────────────────────────────────────────────────────────
total_unique = len(open_examples) + len(pause_examples) + len(witness_examples)
print(f"\n{'='*55}")
print(f"DATASET v0.9 SUMMARY")
print(f"{'='*55}")
print(f"  Unique examples:        {total_unique}")
print(f"  OPEN/PAUSE/WITNESS:     {len(open_examples)}:{len(pause_examples)}:{len(witness_examples)}")
print(f"  Train:                  {len(train)} ({len(open_train)} OPEN, {len(pause_train)} PAUSE, {len(witness_train)} WITNESS)")
print(f"  Valid:                  {len(valid)} ({len(open_valid)} OPEN, {len(pause_valid)} PAUSE, {len(witness_valid)} WITNESS)")
print(f"\n  Sources:")
for src, count in source_stats.items():
    print(f"    {src:<40} {count}")
print(f"\n-> Output: {OUT_DIR}/")
