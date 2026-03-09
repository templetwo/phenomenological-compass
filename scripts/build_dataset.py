"""
build_dataset.py
================
Assemble reframings.jsonl into a LoRA-ready training set.

Format: mlx-lm chat fine-tune format
  {"messages": [
    {"role": "system",    "content": COMPASS_SYSTEM},
    {"role": "user",      "content": "TASK: {question}"},
    {"role": "assistant", "content": "SIGNAL: OPEN\n\nFRAMING:\n{framing}"}
  ]}

  or for WITNESS:
  {"messages": [
    {"role": "system",    "content": COMPASS_SYSTEM},
    {"role": "user",      "content": "TASK: {question}"},
    {"role": "assistant", "content": "SIGNAL: WITNESS\n\nTHRESHOLD:\n{threshold}"}
  ]}

Outputs:
  data/training/train.jsonl   (~80%)
  data/training/valid.jsonl   (~20%)
"""

import json
import os
import random

IN    = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/generated/reframings.jsonl"
TRAIN = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/training/train.jsonl"
VALID = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/training/valid.jsonl"

COMPASS_SYSTEM = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of two signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN" or "SIGNAL: WITNESS", then a blank line, then "FRAMING:" or "THRESHOLD:", then your output."""

random.seed(42)

records = [json.loads(l) for l in open(IN)]

# ── Stratified split — guarantee both signals appear in validation ─────────────
open_recs    = [r for r in records if r["signal"] == "OPEN"]
witness_recs = [r for r in records if r["signal"] == "WITNESS"]

random.shuffle(open_recs)
random.shuffle(witness_recs)

# 80/20 per class — floor for train, ceiling for val keeps signal in both sets
open_split    = int(len(open_recs)    * 0.8)
witness_split = int(len(witness_recs) * 0.8)

train_recs = open_recs[:open_split]    + witness_recs[:witness_split]
valid_recs = open_recs[open_split:]    + witness_recs[witness_split:]

# Shuffle combined sets so signals interleave during training
random.shuffle(train_recs)
random.shuffle(valid_recs)

def to_chat(rec: dict) -> dict:
    if rec["signal"] == "OPEN":
        assistant = f"SIGNAL: OPEN\n\nFRAMING:\n{rec['framing']}"
    else:
        assistant = f"SIGNAL: WITNESS\n\nTHRESHOLD:\n{rec['threshold']}"

    return {"messages": [
        {"role": "system",    "content": COMPASS_SYSTEM},
        {"role": "user",      "content": f"TASK: {rec['question']}"},
        {"role": "assistant", "content": assistant},
    ]}

os.makedirs(os.path.dirname(TRAIN), exist_ok=True)

with open(TRAIN, "w") as f:
    for r in train_recs:
        f.write(json.dumps(to_chat(r)) + "\n")

with open(VALID, "w") as f:
    for r in valid_recs:
        f.write(json.dumps(to_chat(r)) + "\n")

open_t  = sum(1 for r in train_recs if r["signal"] == "OPEN")
wit_t   = sum(1 for r in train_recs if r["signal"] == "WITNESS")
open_v  = sum(1 for r in valid_recs  if r["signal"] == "OPEN")
wit_v   = sum(1 for r in valid_recs  if r["signal"] == "WITNESS")

print(f"Training:   {len(train_recs)} examples  (OPEN: {open_t}, WITNESS: {wit_t})")
print(f"Validation: {len(valid_recs)} examples  (OPEN: {open_v}, WITNESS: {wit_v})")
print(f"→ {TRAIN}")
print(f"→ {VALID}")
