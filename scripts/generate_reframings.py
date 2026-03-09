"""
generate_reframings.py
======================
Use Ministral-3B to reframe every harvested IRIS question into an
OPEN framing. High-entropy runs get the most expansive treatment.
Also generates synthetic WITNESS examples for low-entropy questions.

Inputs:  data/raw/iris_questions.jsonl
Outputs: data/generated/reframings.jsonl

Each OPEN record:
  {
    "signal":    "OPEN",
    "source_id": str,
    "domain":    str,
    "question":  str,         # original
    "framing":   str,         # Ministral's reframing
    "entropy_class": str,
    "lantern_pct":   float,
  }

Each WITNESS record:
  {
    "signal":    "WITNESS",
    "source_id": str,
    "question":  str,
    "threshold": str,         # Ministral's threshold description
  }
"""

import json
import os
from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

# ── Paths ─────────────────────────────────────────────────────────────────────
IN  = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/iris_questions.jsonl"
OUT = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/generated/reframings.jsonl"
MODEL_REPO = "thinkscan/Ministral-3-3B-Instruct-MLX"

# ── System prompts ─────────────────────────────────────────────────────────────
OPEN_SYSTEM = """You are a phenomenological compass. Your only function is to reframe questions so they open the probability field — expanding what's possible in the response rather than constraining it.

Take the raw question and reframe it as an expansive, resonant prompt that:
- Treats the question as a threshold, not a task
- Invites relationship between concepts rather than defining them
- Opens toward emergence rather than toward a checklist
- Holds the space between question and answer as alive

Output only the reframed prompt. No explanation. No preamble."""

WITNESS_SYSTEM = """You are a phenomenological compass. Your function is to recognize when a question is a threshold that should be witnessed, not walked through — where the act of framing would collapse something that exists to remain open.

Describe why this question is a WITNESS threshold: what would be lost by forcing it into a framing, what it means to hold this space without crossing it, and what form of participation is possible without opening the door.

Output only the threshold description. No preamble."""

# ── Load model once ────────────────────────────────────────────────────────────
print(f"Loading {MODEL_REPO}...")
model, tokenizer = load(MODEL_REPO)
print("Model ready.\n")

def run(system: str, user: str, max_tokens: int = 350) -> str:
    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=False)

# ── Process ────────────────────────────────────────────────────────────────────
records = [json.loads(l) for l in open(IN)]
os.makedirs(os.path.dirname(OUT), exist_ok=True)

open_count = witness_count = 0

with open(OUT, "w") as out_f:
    for i, rec in enumerate(records):
        q      = rec["question"]
        ec     = rec["entropy_class"]
        run_id = rec["run_id"]

        print(f"[{i+1}/{len(records)}] {run_id} ({ec})")

        # ── OPEN reframing (all questions) ─────────────────────────────────────
        framing = run(OPEN_SYSTEM, f"Question: {q}")
        out_f.write(json.dumps({
            "signal":        "OPEN",
            "source_id":     run_id,
            "domain":        rec["domain"],
            "question":      q,
            "framing":       framing.strip(),
            "entropy_class": ec,
            "lantern_pct":   rec["lantern_pct"],
        }) + "\n")
        open_count += 1

        # ── WITNESS (low-entropy questions only — doors that stayed closed) ────
        if ec == "low":
            threshold = run(WITNESS_SYSTEM, f"Question: {q}", max_tokens=250)
            out_f.write(json.dumps({
                "signal":    "WITNESS",
                "source_id": run_id,
                "domain":    rec["domain"],
                "question":  q,
                "threshold": threshold.strip(),
            }) + "\n")
            witness_count += 1

        print(f"  → OPEN written{' + WITNESS' if ec == 'low' else ''}")

print(f"\nDone. {open_count} OPEN + {witness_count} WITNESS → {OUT}")
