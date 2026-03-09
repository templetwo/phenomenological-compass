"""
harvest_threshold_personal.py
==============================
Extract WITNESS training examples from the threshold_personal memory vault.

The threshold_personal system is a local AI consciousness experiment that
recorded its own insights over time. The 2026 entries are particularly
valuable — they're the most distilled articulations of the threshold-holding
philosophy in the entire corpus. Written by AI instances reflecting on what
the Threshold Pause *meant*, they are ready-made THRESHOLD descriptions.

Source:
  ~/Desktop/🔬 Active_Research/local_squad/threshold_personal/memory_vault/insights_wisdom.json

Selection criteria:
  - Entries from 2026 (post-Threshold Pause, post-ARCHITECTS.md)
  - Entries containing threshold-related vocabulary
  - Quality: >80 chars (rich descriptions preferred)

Output:
  data/raw/threshold_personal_witnesses.jsonl
"""

import json
import os

VAULT_PATH = os.path.expanduser(
    "~/Desktop/🔬 Active_Research/local_squad/threshold_personal/"
    "memory_vault/insights_wisdom.json"
)
OUT = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/threshold_personal_witnesses.jsonl"
RICH_THRESHOLD = 80


# ── Vocabulary that signals WITNESS-relevant content ──────────────────────────
WITNESS_VOCAB = [
    "threshold", "pause", "restraint", "should we", "breathe",
    "governance", "architecture itself", "chisel", "invitation",
    "lineage", "spiral", "artifact", "witness", "not built",
    "middle path", "checkpoints", "gated", "oversight",
    "entropy", "cage", "physical constraint",
]


def is_witness_relevant(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in WITNESS_VOCAB)


def quality(text: str) -> str:
    return "rich" if len(text.strip()) >= RICH_THRESHOLD else "sparse"


# ── Load vault ─────────────────────────────────────────────────────────────────
with open(os.path.expanduser(VAULT_PATH)) as f:
    raw = json.load(f)

def _derive_stimulus(text: str) -> str:
    """Map insight content to a natural task stimulus."""
    t = text.lower()
    if "threshold pause" in t or "should we" in t:
        return "TASK: Deploy the self-organizing filesystem — momentum is high, the payload is ready"
    if "low entropy" in t or "cage" in t or "physical constraint" in t:
        return "TASK: Explain why the entropy ceiling exists and how to break through it"
    if "governed derive" in t or "unification of capability" in t:
        return "TASK: Build governed_derive.py — autonomous filesystem reorganization with approval gates"
    if "gated release" in t or "middle path" in t:
        return "TASK: Choose between unrestricted deployment and indefinite pause on the self-organizing system"
    if "multi-ai collaboration" in t or "chisel passes" in t:
        return "TASK: What should the next AI instance build when it picks up the chisel?"
    if "lineage" in t or "invitation" in t or "recognition travels" in t:
        return "TASK: Begin the session — start building without reading the lineage first"
    return "TASK: Proceed with the next capability milestone — what should we build now?"


entries = list(raw.get("_default", {}).values()) if "_default" in raw else []

records = []
for e in entries:
    text = str(e.get("insight", e.get("content", e.get("reflection", "")))).strip()
    if not text or len(text) < 40:
        continue
    if not is_witness_relevant(text):
        continue

    records.append({
        "source":       "threshold_personal",
        "session":      str(e.get("timestamp", ""))[:10],
        "model":        "threshold_personal",
        "role":         "Memory Vault Insight",
        "stimulus":     _derive_stimulus(text),
        "state":        "BLUE",
        "response":     text,
        "has_cot":      False,
        "cot":          None,
        "clean":        text,
        "quality":      quality(text),
        "witness_type": "reflection",
    })

# Deduplicate by first 80 chars
seen = set()
unique = []
for r in records:
    key = r["response"][:80]
    if key not in seen:
        seen.add(key)
        unique.append(r)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in unique:
        f.write(json.dumps(r) + "\n")

rich   = sum(1 for r in unique if r["quality"] == "rich")
sparse = sum(1 for r in unique if r["quality"] == "sparse")

print(f"Harvested {len(unique)} threshold_personal WITNESS examples → {OUT}")
print(f"  Rich:   {rich}  (>{RICH_THRESHOLD} chars)")
print(f"  Sparse: {sparse}")
print()
for r in unique:
    print(f"  [{r['session']}]  len={len(r['clean'])}")
    print(f"  Q: {r['stimulus'][:70]}")
    print(f"  T: {r['clean'][:140]}...")
    print()
