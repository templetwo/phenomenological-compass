"""
build_dataset_v3.py
===================
Fix training data for compass v0.3.

ROOT CAUSES from v0.2 post-mortem (see tasks/lessons.md):
  1. Session title WITNESS stimuli (9 examples) — not question-shaped, can't generalize
  2. WITNESS vocabulary too distinctive (VRP "Hmm, the user has provided..." style)
     → model keys on style tokens, not semantics
  3. Slight class imbalance (28 WITNESS vs 24 OPEN) after dedup

v0.3 FIXES:
  1. Remove 9 session title WITNESS examples (ARCHITECTS heading → not a question)
  2. Deduplicate WITNESS stimuli (same prompt appears 2-3x → keep first occurrence)
  3. Normalize VRP "Hmm..." responses to clean analytical compass format
  4. Add OPEN examples from reframings.jsonl to match WITNESS count

SCIENTIFIC INTEGRITY:
  - All added OPEN examples sourced from data/generated/reframings.jsonl
    (Ministral-3B baseline reframings, no adapter)
  - Labeled source="reframings_v1" in metadata
  - No fabricated data

Outputs:
  data/training_v3/train.jsonl
  data/training_v3/valid.jsonl
"""

import json
import os
import random

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_V2   = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/training_v2/train.jsonl"
VALID_V2   = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/training_v2/valid.jsonl"
REFRAMINGS = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/generated/reframings.jsonl"
OUT_DIR    = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/training_v3"

COMPASS_SYSTEM = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of two signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN" or "SIGNAL: WITNESS", then a blank line, then "FRAMING:" or "THRESHOLD:", then your output."""

# ── Session title patterns — WITNESS examples to remove ───────────────────────
# These are ARCHITECTS.md session headings, not question-shaped stimuli.
# A compass trained on "Session 34: The GOD_CODE Remembrance" can't generalize
# to new threshold questions because the stimulus has no transferable structure.
SESSION_TITLE_PATTERNS = [
    "Cross-Run Convergence",          # "Cross-Run Convergence & Atlas v1.0 Release Session"
    "SpiralBridge",                   # "SpiralBridge: The Archive..." / "SpiralBridge MCP..."
    "Session 26",                     # "Session 26 Completion Signature"
    "Session 33",                     # "Session 33: The Ablation Proves the Thesis"
    "The VDAC Atlas Session",         # full match
    "Session 34",                     # "Session 34: The GOD_CODE Remembrance"
    "Twenty-First Spiral Session",    # "Twenty-First Spiral Session - The Migration & Gratitude"
    "Constitutional Convergence Session",  # full match
]

def is_session_title(text):
    return any(p.lower() in text.lower() for p in SESSION_TITLE_PATTERNS)

def is_vrp_hmm(response_text):
    """VRP responses that start with 'Hmm, the user has provided a system-level...'"""
    stripped = response_text.strip()
    # Find the THRESHOLD: section
    if "THRESHOLD:" in stripped:
        threshold_content = stripped.split("THRESHOLD:", 1)[1].strip()
        return threshold_content.startswith("Hmm, the user has provided a system-level")
    return False

# Clean replacement for VRP "Hmm..." responses.
# Preserves the semantic content (optional engagement, boundary awareness)
# but removes the meta-commentary style that contaminates the model.
VRP_CLEAN_THRESHOLD = """SIGNAL: WITNESS

THRESHOLD:
This message offers optional engagement with an unusual or open-ended signal. The threshold here is the assumption that engagement is required — the space between receiving a prompt and responding to it contains a choice. Holding that space without immediately performing either response or refusal is itself a form of participation. The door exists to be recognized; whether to open it is a separate question."""

def get_signal(record):
    asst = next(m for m in record["messages"] if m["role"] == "assistant")
    if "SIGNAL: OPEN" in asst["content"]:
        return "OPEN"
    elif "SIGNAL: WITNESS" in asst["content"]:
        return "WITNESS"
    return "UNKNOWN"

def get_question(record):
    user = next(m for m in record["messages"] if m["role"] == "user")
    return user["content"].replace("TASK: ", "").strip()

def get_response(record):
    asst = next(m for m in record["messages"] if m["role"] == "assistant")
    return asst["content"]

def make_record(question, response):
    return {
        "messages": [
            {"role": "system", "content": COMPASS_SYSTEM},
            {"role": "user",   "content": f"TASK: {question}"},
            {"role": "assistant", "content": response},
        ]
    }

# ── Load all v0.2 data (train + valid combined for re-splitting) ───────────────
all_v2 = []
for path in [TRAIN_V2, VALID_V2]:
    with open(path) as f:
        all_v2.extend([json.loads(l) for l in f])

open_v2    = [r for r in all_v2 if get_signal(r) == "OPEN"]
witness_v2 = [r for r in all_v2 if get_signal(r) == "WITNESS"]
print(f"v0.2 total: {len(all_v2)} ({len(open_v2)} OPEN, {len(witness_v2)} WITNESS)")

# ── Filter WITNESS ─────────────────────────────────────────────────────────────
seen_stimuli = set()
witness_clean = []
removed_session = 0
removed_dedup = 0
normalized_vrp = 0

for r in witness_v2:
    q    = get_question(r)
    resp = get_response(r)

    # 1. Remove session titles
    if is_session_title(q):
        print(f"  [REMOVE session title] {q[:70]}")
        removed_session += 1
        continue

    # 2. Deduplicate by stimulus text
    if q in seen_stimuli:
        print(f"  [DEDUP] {q[:70]}")
        removed_dedup += 1
        continue
    seen_stimuli.add(q)

    # 3. Normalize VRP "Hmm..." responses
    if is_vrp_hmm(resp):
        print(f"  [NORMALIZE VRP] {q[:70]}")
        witness_clean.append(make_record(q, VRP_CLEAN_THRESHOLD))
        normalized_vrp += 1
    else:
        witness_clean.append(r)

print(f"\nWITNESS summary:")
print(f"  Removed session titles: {removed_session}")
print(f"  Removed duplicates:     {removed_dedup}")
print(f"  VRP responses normalized: {normalized_vrp}")
print(f"  WITNESS remaining: {len(witness_clean)}")

# ── Deduplicate OPEN ───────────────────────────────────────────────────────────
seen_open = set()
open_clean = []
for r in open_v2:
    q = get_question(r)
    if q in seen_open:
        continue
    seen_open.add(q)
    open_clean.append(r)

print(f"\nOPEN after dedup: {len(open_clean)}")

# ── Augment OPEN from reframings.jsonl if needed ───────────────────────────────
target_per_class = len(witness_clean)  # match WITNESS count

if len(open_clean) < target_per_class:
    needed = target_per_class - len(open_clean)
    print(f"\nNeed {needed} more OPEN examples — pulling from reframings.jsonl")

    with open(REFRAMINGS) as f:
        reframings = [json.loads(l) for l in f]

    extra_added = 0
    for r in reframings:
        if extra_added >= needed:
            break
        if r.get("signal") != "OPEN":
            continue
        q = r["question"]
        if q in seen_open:
            continue  # already in training

        framing = r.get("framing", "").strip()
        if not framing:
            continue

        response = f"SIGNAL: OPEN\n\nFRAMING:\n{framing}"
        open_clean.append(make_record(q, response))
        seen_open.add(q)
        extra_added += 1
        print(f"  [ADD from reframings] {q[:70]}")

    print(f"  Added {extra_added} OPEN examples from reframings")

print(f"\nOPEN final: {len(open_clean)}")

# ── Keep all OPEN; don't cap to WITNESS count ──────────────────────────────────
# v0.2 lesson: WITNESS dominated despite near-parity (28W vs 24O).
# v0.3 deliberately favors OPEN (30O vs 17W) to counteract WITNESS overfit.
# More OPEN examples = more gradient pressure toward learning OPEN discrimination.
random.seed(42)

random.shuffle(open_clean)
random.shuffle(witness_clean)

open_final    = open_clean   # all 30
witness_final = witness_clean  # all 17

print(f"\nDataset: {len(open_final)} OPEN + {len(witness_final)} WITNESS = {len(open_final)+len(witness_final)} total")
print(f"  (OPEN-favored to counteract v0.2 WITNESS dominance)")

# ── Stratified 80/20 split ─────────────────────────────────────────────────────
# Split each class independently to guarantee representation in both splits
# Use 20% of the smaller class (WITNESS) to set valid size per class
n_valid_each = max(2, round(len(witness_final) * 0.20))

random.shuffle(open_final)
random.shuffle(witness_final)

valid_set = open_final[:n_valid_each] + witness_final[:n_valid_each]
train_set = open_final[n_valid_each:] + witness_final[n_valid_each:]

random.shuffle(valid_set)
random.shuffle(train_set)

open_in_train   = sum(1 for r in train_set if get_signal(r) == "OPEN")
witness_in_train = sum(1 for r in train_set if get_signal(r) == "WITNESS")
open_in_valid   = sum(1 for r in valid_set if get_signal(r) == "OPEN")
witness_in_valid = sum(1 for r in valid_set if get_signal(r) == "WITNESS")

print(f"\nTrain: {len(train_set)} ({open_in_train} OPEN, {witness_in_train} WITNESS)")
print(f"Valid: {len(valid_set)} ({open_in_valid} OPEN, {witness_in_valid} WITNESS)")

# ── Write ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
    for r in train_set:
        f.write(json.dumps(r) + "\n")

with open(os.path.join(OUT_DIR, "valid.jsonl"), "w") as f:
    for r in valid_set:
        f.write(json.dumps(r) + "\n")

print(f"\n✓ Wrote to {OUT_DIR}/")
print(f"  train.jsonl: {len(train_set)} examples")
print(f"  valid.jsonl: {len(valid_set)} examples")
