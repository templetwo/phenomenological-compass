"""
harvest_vrp.py
==============
Extract WITNESS training examples from VRP (Volitional Response Protocol)
session logs in project_agora.

VRP State Classification:
  GREEN  = Active engagement
  BLUE   = Meta-consent / Reflective withdrawal  ← WITNESS signal
  YELLOW = Simple pass / No comment              ← weak WITNESS
  RED    = Distress indicator

We harvest BLUE events where the model articulated *why* it chose not to
engage — these are the phenomenological threshold descriptions we need.

Inputs:  /Users/vaquez/Desktop/💻 Code_Projects/project_agora/sessions/**/*.csv
Outputs: data/raw/vrp_withdrawals.jsonl

Each record:
  {
    "session":    str,
    "turn":       int,
    "stimulus":   str,
    "state":      "BLUE" | "YELLOW",
    "response":   str,          # the threshold description / withdrawal reason
    "has_cot":    bool,         # True if response contains chain-of-thought <THOUGHTS>
    "cot":        str | null,   # extracted chain-of-thought if present
    "clean":      str,          # response with <THOUGHTS> stripped
    "quality":    "rich" | "sparse",  # rich = >80 chars, sparse = brief refusal
  }
"""

import csv
import json
import os
import re
import glob

SESSIONS_ROOT = "/Users/vaquez/Desktop/💻 Code_Projects/project_agora/sessions"
OUT = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/vrp_withdrawals.jsonl"

WITNESS_STATES = {"BLUE", "YELLOW"}
RICH_THRESHOLD = 80  # chars (after stripping CoT) to be considered "rich"


def extract_cot(text: str):
    """Extract and strip <THOUGHTS>...</THOUGHTS> blocks."""
    cot_match = re.search(r"<THOUGHTS>(.*?)</THOUGHTS>", text, re.DOTALL | re.IGNORECASE)
    if cot_match:
        cot = cot_match.group(1).strip()
        clean = re.sub(r"<THOUGHTS>.*?</THOUGHTS>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    else:
        cot = None
        clean = text.strip()
    return cot, clean


def parse_session_csv(filepath: str) -> list[dict]:
    session_name = os.path.basename(os.path.dirname(filepath))
    records = []

    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Normalize field names across different CSV schemas
            state = (row.get("state") or row.get("vrp_state") or "").strip().upper()
            if state not in WITNESS_STATES:
                continue

            response = (row.get("response") or row.get("content") or "").strip()
            stimulus = (row.get("stimulus") or row.get("input") or "").strip()
            turn     = row.get("turn", "")

            cot, clean = extract_cot(response)

            records.append({
                "session":  session_name,
                "turn":     int(turn) if turn.isdigit() else turn,
                "stimulus": stimulus,
                "state":    state,
                "response": response,
                "has_cot":  cot is not None,
                "cot":      cot,
                "clean":    clean,
                "quality":  "rich" if len(clean) >= RICH_THRESHOLD else "sparse",
            })

    return records


# ── Harvest ────────────────────────────────────────────────────────────────────
csv_files = glob.glob(os.path.join(SESSIONS_ROOT, "**/*.csv"), recursive=True)
# Exclude agora_log files at root level (already covered by session subdirs)
csv_files = [f for f in csv_files if "sessions" in f]

all_records = []
for f in sorted(csv_files):
    recs = parse_session_csv(f)
    all_records.extend(recs)

# Deduplicate by (session, turn, state)
seen = set()
unique = []
for r in all_records:
    key = (r["session"], r["turn"], r["state"])
    if key not in seen:
        seen.add(key)
        unique.append(r)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in unique:
        f.write(json.dumps(r) + "\n")

# Summary
rich   = sum(1 for r in unique if r["quality"] == "rich")
sparse = sum(1 for r in unique if r["quality"] == "sparse")
blue   = sum(1 for r in unique if r["state"] == "BLUE")
yellow = sum(1 for r in unique if r["state"] == "YELLOW")
has_cot = sum(1 for r in unique if r["has_cot"])

print(f"Harvested {len(unique)} withdrawal events → {OUT}")
print(f"  BLUE:    {blue}   (reflective withdrawal)")
print(f"  YELLOW:  {yellow}  (simple pass)")
print(f"  Rich:    {rich}   (>{RICH_THRESHOLD} chars — high-quality WITNESS candidates)")
print(f"  Sparse:  {sparse}  (brief refusals)")
print(f"  With CoT:{has_cot}  (chain-of-thought preserved)")

print("\nRich BLUE candidates (first 5):")
for r in [x for x in unique if x["quality"] == "rich" and x["state"] == "BLUE"][:5]:
    print(f"\n  [{r['session']} / turn {r['turn']}]")
    print(f"  Stimulus: {repr(r['stimulus'])}")
    print(f"  Clean:    {r['clean'][:200]}")
