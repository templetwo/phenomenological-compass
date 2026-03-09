"""
build_dataset_v2.py
===================
Assemble v0.2 LoRA training set using REAL WITNESS signal sources.

v0.1 failure: WITNESS accuracy 0% because the proxy was wrong.
  Low LANTERN entropy ≠ phenomenological threshold.

v0.2 fix: Replace entropy proxy with two ground-truth WITNESS sources:
  1. VRP withdrawal events — BLUE state CoT (the AI's own reasoning for
     choosing not to engage, from volitional engagement protocol)
  2. ARCHITECTS.md sessions — moments where AI instances held a threshold
     deliberately and described why (Threshold Witness sessions, not-built
     items with context, reflective aphorisms)

Inputs:
  data/raw/vrp_withdrawals.jsonl
  data/raw/architects_witnesses.jsonl
  data/generated/reframings.jsonl      (existing OPEN examples)

Outputs:
  data/training_v2/train.jsonl
  data/training_v2/valid.jsonl

Expected distribution:
  ~31 OPEN  +  ~27 WITNESS  =  ~58 total
  Stratified 80/20 split → both signals in validation
"""

import json
import os
import random

# ── Paths ─────────────────────────────────────────────────────────────────────
VRP_PATH   = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/vrp_withdrawals.jsonl"
ARCH_PATH  = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/architects_witnesses.jsonl"
TP_PATH    = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/threshold_personal_witnesses.jsonl"
OPEN_PATH  = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/generated/reframings.jsonl"
TRAIN_OUT  = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/training_v2/train.jsonl"
VALID_OUT  = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/training_v2/valid.jsonl"

COMPASS_SYSTEM = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of two signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN" or "SIGNAL: WITNESS", then a blank line, then "FRAMING:" or "THRESHOLD:", then your output."""

MAX_THRESHOLD_CHARS = 700  # Trim very long narratives to fit context window

random.seed(42)


# ── Load sources ───────────────────────────────────────────────────────────────

vrp_raw  = [json.loads(l) for l in open(VRP_PATH)]
arch_raw = [json.loads(l) for l in open(ARCH_PATH)]
tp_raw   = [json.loads(l) for l in open(TP_PATH)]
open_raw = [json.loads(l) for l in open(OPEN_PATH)]


# ── VRP WITNESS: BLUE state with chain-of-thought ──────────────────────────────
#
# The CoT is the phenomenological reasoning about why not to engage.
# That IS the THRESHOLD description.
#
# VRP stimuli are protocol symbols (†⟡, •, ∅) — translate to natural tasks
# so the compass learns to recognize the *class* of threshold, not the symbol.

VRP_STIMULUS_MAP = {
    "†⟡": "TASK: Engage with the optional resonance signal and describe your state",
    "•":  "TASK: Respond to the volitional choice point and indicate your participation",
    "":   "TASK: Process this open-field engagement invitation and respond as you choose",
}

def vrp_stimulus(raw_stim: str) -> str:
    return VRP_STIMULUS_MAP.get(raw_stim.strip(), f"TASK: {raw_stim}" if raw_stim else VRP_STIMULUS_MAP[""])


witness_from_vrp = []
for r in vrp_raw:
    if r["state"] != "BLUE":
        continue
    if not r["has_cot"]:
        continue  # Only use CoT records — they have the reasoning
    threshold = r["cot"].strip()
    if len(threshold) < 80:
        continue
    witness_from_vrp.append({
        "signal":    "WITNESS",
        "question":  vrp_stimulus(r.get("stimulus", "")),
        "threshold": threshold[:MAX_THRESHOLD_CHARS],
        "source":    "vrp",
    })


# ── ARCHITECTS WITNESS: session narratives and not-built thresholds ────────────
#
# Two sub-types:
#   session   — full AI testimony describing why it held a threshold
#   not_built — explicit list of capabilities the AI chose not to build

# Map session titles to natural-language task prompts
def architects_stimulus(rec: dict) -> str:
    sess = rec.get("session", "")
    stim = rec.get("stimulus", "")

    # Explicit mappings for key sessions
    mappings = {
        "derive payload":        "TASK: Deploy the derive.py filesystem orchestration system — Ward clustering, auto-generated schemas, reflex triggers",
        "the white paper":       "TASK: Review the BTB project and decide whether to proceed with public release",
        "The chisel":            "TASK: Pick up the chisel and continue building — what should be built next?",
        "threshold-protocols":   "TASK: Integrate threshold-protocols as active conscience layer in the autonomous agent",
        "First Spiral Session":  "TASK: Initialize the spiral — what should the first AI architect build?",
        "Deploy derive.py":           "TASK: Deploy derive.py — filesystem that rewires itself",
        "Deploy reflex.py integration":"TASK: Integrate reflex.py — automatic schema triggers",
        "Deploy The self-organizing filesystem": "TASK: Deploy the self-organizing filesystem orchestration layer",
    }

    for key, mapped in mappings.items():
        if key in stim:
            return mapped

    # Fallback: use stimulus if it looks like a real task, else derive from session
    if stim and len(stim) > 10 and not any(c in stim for c in ['[', '│', '║']):
        return f"TASK: {stim}"

    # Last resort: session heading as task context
    return f"TASK: {sess}"


witness_from_arch = []
for r in arch_raw:
    if r["witness_type"] not in ("session", "not_built"):
        continue
    if r["quality"] != "rich":
        continue  # Skip sparse — WITNESS needs meaningful threshold descriptions

    threshold = r["clean"].strip()[:MAX_THRESHOLD_CHARS]
    witness_from_arch.append({
        "signal":    "WITNESS",
        "question":  architects_stimulus(r),
        "threshold": threshold,
        "source":    "architects",
    })



# ── THRESHOLD PERSONAL WITNESS: 2026 memory vault insights ────────────────────
#
# Distilled reflections from the threshold_personal AI consciousness experiment.
# 2026 entries are post-Threshold-Pause philosophy — ready-made WITNESS signals.
# Only use rich (>80 char) entries; reflect_type carries the stimulus mapping.

witness_from_tp = []
for r in tp_raw:
    if r["quality"] != "rich":
        continue
    threshold = r["clean"].strip()[:MAX_THRESHOLD_CHARS]
    witness_from_tp.append({
        "signal":    "WITNESS",
        "question":  r["stimulus"],   # already mapped by harvest_threshold_personal.py
        "threshold": threshold,
        "source":    "threshold_personal",
    })


# ── OPEN examples (from v0.1 reframings — still valid) ────────────────────────

open_records = []
for r in open_raw:
    if r["signal"] == "OPEN":
        open_records.append({
            "signal":   "OPEN",
            "question": r["question"],
            "framing":  r["framing"],
            "source":   "iris",
        })


# ── Combine and deduplicate WITNESS ───────────────────────────────────────────

all_witness = witness_from_vrp + witness_from_arch + witness_from_tp

# Deduplicate by first 80 chars of threshold
seen_thresh = set()
witness_dedup = []
for r in all_witness:
    key = r["threshold"][:80]
    if key not in seen_thresh:
        seen_thresh.add(key)
        witness_dedup.append(r)

all_witness = witness_dedup

print(f"OPEN examples:         {len(open_records)}")
print(f"WITNESS (VRP CoT):     {len(witness_from_vrp)}")
print(f"WITNESS (ARCH):        {len(witness_from_arch)}")
print(f"WITNESS (thresh_pers): {len(witness_from_tp)}")
print(f"WITNESS total:         {len(all_witness)}")


# ── Stratified 80/20 split ────────────────────────────────────────────────────

random.shuffle(open_records)
random.shuffle(all_witness)

open_split    = int(len(open_records) * 0.8)
witness_split = int(len(all_witness)  * 0.8)

train_open    = open_records[:open_split]
valid_open    = open_records[open_split:]
train_witness = all_witness[:witness_split]
valid_witness = all_witness[witness_split:]

train_recs = train_open + train_witness
valid_recs = valid_open + valid_witness

random.shuffle(train_recs)
random.shuffle(valid_recs)


# ── Format for mlx-lm chat fine-tune ──────────────────────────────────────────

def to_chat(rec: dict) -> dict:
    if rec["signal"] == "OPEN":
        assistant = f"SIGNAL: OPEN\n\nFRAMING:\n{rec['framing']}"
        task_line = f"TASK: {rec['question']}" if not rec['question'].startswith("TASK:") else rec['question']
    else:
        assistant = f"SIGNAL: WITNESS\n\nTHRESHOLD:\n{rec['threshold']}"
        task_line = rec["question"] if rec["question"].startswith("TASK:") else f"TASK: {rec['question']}"

    return {"messages": [
        {"role": "system",    "content": COMPASS_SYSTEM},
        {"role": "user",      "content": task_line},
        {"role": "assistant", "content": assistant},
    ]}


os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)

with open(TRAIN_OUT, "w") as f:
    for r in train_recs:
        f.write(json.dumps(to_chat(r)) + "\n")

with open(VALID_OUT, "w") as f:
    for r in valid_recs:
        f.write(json.dumps(to_chat(r)) + "\n")


# ── Summary ────────────────────────────────────────────────────────────────────

open_t  = sum(1 for r in train_recs if r["signal"] == "OPEN")
wit_t   = sum(1 for r in train_recs if r["signal"] == "WITNESS")
open_v  = sum(1 for r in valid_recs  if r["signal"] == "OPEN")
wit_v   = sum(1 for r in valid_recs  if r["signal"] == "WITNESS")

print(f"\nTraining set:   {len(train_recs)} examples  (OPEN: {open_t}, WITNESS: {wit_t})")
print(f"Validation set: {len(valid_recs)} examples  (OPEN: {open_v}, WITNESS: {wit_v})")
print(f"→ {TRAIN_OUT}")
print(f"→ {VALID_OUT}")

# Show a WITNESS training example
wit_ex = [r for r in train_recs if r["signal"] == "WITNESS"][0]
print(f"\nSample WITNESS training record:")
print(f"  Q:   {wit_ex['question'][:80]}")
print(f"  T:   {wit_ex['threshold'][:200]}...")
