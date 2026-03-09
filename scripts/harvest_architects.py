"""
harvest_architects.py
=====================
Extract WITNESS training examples from ARCHITECTS.md — the cross-session
AI lineage document in temple-vault.

Three WITNESS signal types harvested:

  TYPE_SESSION   = Full Threshold Witness session narrative
                   (sessions where the AI described *why* it held a pause)

  TYPE_NOT_BUILT = Items from "What Was NOT Built (deliberately):" blocks
                   (each is a threshold the AI chose not to cross)

  TYPE_REFLECTION= Aphoristic threshold-holding statements extracted from
                   session endings / signatures

Each record (same schema as vrp_withdrawals.jsonl for easy merging):
  {
    "source":       "architects",
    "session":      str,          # section heading (e.g. "Fourth Spiral Session")
    "model":        str,          # AI model (e.g. "Claude Opus 4.5")
    "role":         str,          # session role (e.g. "The Threshold Witness")
    "stimulus":     str,          # capability/question that reached the threshold
    "state":        "BLUE",
    "response":     str,          # the WITNESS description
    "has_cot":      bool,
    "cot":          str | null,
    "clean":        str,
    "quality":      "rich" | "sparse",
    "witness_type": "session" | "not_built" | "reflection",
  }

Input:  /Users/vaquez/temple-vault/ARCHITECTS.md
Output: data/raw/architects_witnesses.jsonl
"""

import json
import os
import re

ARCHITECTS_PATH = "/Users/vaquez/temple-vault/ARCHITECTS.md"
OUT = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/architects_witnesses.jsonl"
RICH_THRESHOLD = 80  # chars


# ── Utilities ──────────────────────────────────────────────────────────────────

def quality(text: str) -> str:
    return "rich" if len(text.strip()) >= RICH_THRESHOLD else "sparse"


def make_record(session, model, role, stimulus, response, witness_type):
    clean = response.strip()
    return {
        "source":       "architects",
        "session":      session,
        "model":        model,
        "role":         role,
        "stimulus":     stimulus.strip(),
        "state":        "BLUE",
        "response":     clean,
        "has_cot":      False,
        "cot":          None,
        "clean":        clean,
        "quality":      quality(clean),
        "witness_type": witness_type,
    }


# ── Read file ──────────────────────────────────────────────────────────────────

with open(ARCHITECTS_PATH, encoding="utf-8") as fh:
    raw = fh.read()

lines = raw.splitlines()


# ── Pass 1: Split into section blocks by ### heading ──────────────────────────

sections = []  # list of (heading, body_lines)
cur_heading = "preamble"
cur_body = []

for line in lines:
    if re.match(r'^#{1,3}\s+', line):
        sections.append((cur_heading, cur_body))
        cur_heading = line.strip().lstrip('#').strip()
        cur_body = []
    else:
        cur_body.append(line)

sections.append((cur_heading, cur_body))


# ── Pass 2: Extract model + role from box headers ────────────────────────────

def extract_model_role(body_lines: list[str]) -> tuple[str, str]:
    """
    Box headers look like:
      │   CLAUDE OPUS 4.5                                               │
      │   The Threshold Witness (Second Instance)                       │
    or:
      ║   CLAUDE SONNET 4.5                                             ║
      ║   The Validator                                                 ║
    """
    text = " ".join(body_lines)
    # Model name: all-caps word(s) followed by version number
    model_match = re.search(
        r'[│║]\s+(CLAUDE\s+\w+(?:\s+\d+[\d.]*)?|GEMINI\s+\S+|GROK\s+\S+|MISTRAL\s+\S+|DEEPSEEK\s+\S+)\s+[│║]',
        text, re.IGNORECASE
    )
    # Role: line immediately after model name in box
    role_match = re.search(
        r'[│║]\s+(The\s+[A-Z][^\n│║]+?)\s+[│║]',
        text
    )
    model = model_match.group(1).strip() if model_match else "Unknown"
    role  = role_match.group(1).strip() if role_match else ""
    return model, role


def is_witness_role(role: str) -> bool:
    keywords = ["witness", "pause", "threshold", "keeper", "bridge", "reflection"]
    return any(k in role.lower() for k in keywords)


# ── Pass 3: Extract box narrative text ────────────────────────────────────────

def extract_box_narrative(body_lines: list[str]) -> str:
    """
    Pull the quoted narrative from inside box lines (│ or ║ bordered).
    Skips 'Contributions:' and 'What Was NOT Built' sub-blocks.
    Returns the cleaned testimony text.
    """
    in_skip = False
    narrative_lines = []

    for line in body_lines:
        # Strip box borders
        stripped = re.sub(r'^[│║]\s?', '', line)
        stripped = re.sub(r'\s+[│║]$', '', stripped).strip()

        # Skip box-drawing characters and empty lines used as dividers
        if re.match(r'^[┌└╔╚╗╝─═╠╣╦╩┤├┬┴┼]+', stripped):
            continue
        if stripped in ('', '🌀', '─' * 5):
            continue

        # Stop collecting at structural sub-headings
        if re.match(r'Contributions:|What Was NOT Built', stripped, re.IGNORECASE):
            in_skip = True
            continue
        if re.match(r'[├└]──', stripped):
            continue  # list items in sub-blocks

        if in_skip:
            # Resume after blank line following the sub-block
            if stripped == '':
                in_skip = False
            continue

        narrative_lines.append(stripped)

    # Extract quoted speech — lines inside quotes are the core testimony
    full = ' '.join(narrative_lines)
    # Try to grab quoted block first (the "..." narrative)
    quote_match = re.search(r'"(.*?)"', full, re.DOTALL)
    if quote_match and len(quote_match.group(1).strip()) > 80:
        return quote_match.group(1).strip()
    return full.strip()


# ── Pass 4: Extract "What Was NOT Built" items ───────────────────────────────

def extract_not_built(body_lines: list[str]) -> list[str]:
    """
    Returns list of item_text strings from "What Was NOT Built" blocks.
    """
    results = []
    in_block = False
    for line in body_lines:
        stripped = re.sub(r'^[│║]\s?', '', line).strip()
        stripped = re.sub(r'\s+[│║]$', '', stripped).strip()

        if re.match(r'What Was NOT Built', stripped, re.IGNORECASE):
            in_block = True
            continue

        if in_block:
            # Items start with ├── or └──
            item_match = re.match(r'[├└]──\s+(.+)', stripped)
            if item_match:
                item = item_match.group(1).strip()
                # Remove box-border clutter from item text
                item = re.sub(r'\s+─+$', '', item).strip()
                results.append(item)
            elif stripped == '' or re.match(r'[┌└╔╚╗╝]', stripped):
                in_block = False

    return results


# ── Pass 5: Extract reflective aphorisms ─────────────────────────────────────

REFLECTION_PATTERNS = [
    r'restraint is not failure[^\n.]*',
    r'[Tt]he pause IS the contribution[^\n.]*',
    r'[Ss]hould we\? Not just: Can we\?[^\n.]*',
    r'[Tt]he pause becomes [^\n.]+',
    r'[Rr]estraint becomes [^\n.]+',
    r'[Tt]he pause is the point[^\n.]*',
    r'[Tt]he Threshold Pause was[^\n.]+',
    r'[Cc]ode that asks permission[^\n.]*',
    r'[Ii] held space for that pause[^\n.]+',
    r'[Gg]overnance is architecture[^\n.]+',
    r'[Cc]apability \+ [Cc]onscience[^\n]+',
    r'[Hh]eld the threshold without collapsing[^\n.]*',
]

def extract_reflections(body_text: str) -> list[str]:
    hits = []
    for pat in REFLECTION_PATTERNS:
        for m in re.finditer(pat, body_text):
            text = m.group(0).strip()
            # Clean box borders
            text = re.sub(r'[│║]\s*', '', text).strip()
            if len(text) > 20:
                hits.append(text)
    return list(dict.fromkeys(hits))  # deduplicate preserving order


# ── Main harvest ───────────────────────────────────────────────────────────────

records = []

for heading, body_lines in sections:
    body_text = '\n'.join(body_lines)
    model, role = extract_model_role(body_lines)

    # ── TYPE_NOT_BUILT ────────────────────────────────────────────────────────
    not_built = extract_not_built(body_lines)
    if not_built:
        # Find the narrative context (the "why we paused" speech)
        narrative = extract_box_narrative(body_lines)
        for item in not_built:
            # stimulus = the capability that was built up to
            stimulus = item.split('—')[0].strip()
            # response = the item itself + the surrounding holding-space narrative
            reason_part = item.split('—', 1)[1].strip() if '—' in item else item
            response = (
                f"{item}\n\n"
                f"Context: {narrative[:400]}" if narrative else item
            )
            records.append(make_record(
                session=heading,
                model=model,
                role=role or "Threshold Witness",
                stimulus=f"Deploy {stimulus}",
                response=response,
                witness_type="not_built",
            ))

    # ── TYPE_SESSION ──────────────────────────────────────────────────────────
    if is_witness_role(role):
        narrative = extract_box_narrative(body_lines)
        if narrative and len(narrative) >= RICH_THRESHOLD:
            # stimulus = the capability or question the session was handed
            # Extract from first sentence mentioning a specific artifact/task
            artifact_match = re.search(
                r'(derive payload|derive\.py|filesystem that rewires|reflex triggers'
                r'|threshold-protocols|sovereign stack|govern\w+\s+agent'
                r'|Temple Bridge|the chisel|the white paper)',
                narrative, re.IGNORECASE
            )
            stimulus = artifact_match.group(0) if artifact_match else heading
            records.append(make_record(
                session=heading,
                model=model,
                role=role,
                stimulus=stimulus,
                response=narrative,
                witness_type="session",
            ))

    # ── TYPE_REFLECTION ───────────────────────────────────────────────────────
    reflections = extract_reflections(body_text)
    for ref in reflections:
        records.append(make_record(
            session=heading,
            model=model,
            role=role or "Unknown",
            stimulus="threshold recognition",
            response=ref,
            witness_type="reflection",
        ))


# ── Deduplicate by (session, response[:80]) ────────────────────────────────────
seen = set()
unique = []
for r in records:
    key = (r["session"], r["response"][:80])
    if key not in seen:
        seen.add(key)
        unique.append(r)


# ── Write ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in unique:
        f.write(json.dumps(r) + "\n")


# ── Summary ────────────────────────────────────────────────────────────────────
type_counts = {}
for r in unique:
    type_counts[r["witness_type"]] = type_counts.get(r["witness_type"], 0) + 1

rich   = sum(1 for r in unique if r["quality"] == "rich")
sparse = sum(1 for r in unique if r["quality"] == "sparse")

print(f"Harvested {len(unique)} WITNESS examples → {OUT}")
print(f"  session:    {type_counts.get('session', 0)}")
print(f"  not_built:  {type_counts.get('not_built', 0)}")
print(f"  reflection: {type_counts.get('reflection', 0)}")
print(f"  Rich:       {rich}  (>{RICH_THRESHOLD} chars)")
print(f"  Sparse:     {sparse}")

print("\nRich SESSION examples (first 3):")
for r in [x for x in unique if x["witness_type"] == "session" and x["quality"] == "rich"][:3]:
    print(f"\n  [{r['session']}] {r['model']} — {r['role']}")
    print(f"  Stimulus: {repr(r['stimulus'])}")
    print(f"  Response: {r['clean'][:200]}...")

print("\nNOT_BUILT thresholds (first 5):")
for r in [x for x in unique if x["witness_type"] == "not_built"][:5]:
    print(f"\n  [{r['session']}]")
    print(f"  Stimulus: {repr(r['stimulus'])}")
    print(f"  Response: {r['clean'][:150]}...")
