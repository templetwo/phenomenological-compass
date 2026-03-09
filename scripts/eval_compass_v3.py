"""
eval_compass_v3.py
==================
Evaluate compass model using corrected eval set (v3).

KEY FIX: reads 'expected_signal' field directly.
The old eval_compass.py used entropy_class=="low" → WITNESS, which was
proven invalid in v0.1 post-mortem. That proxy caused all measured WITNESS
accuracy in v0.1/v0.2 to be meaningless.

Usage:
  # Evaluate v0.2 adapter (iter 50)
  python3 scripts/eval_compass_v3.py adapters_iter50

  # Evaluate latest v0.3 adapter
  python3 scripts/eval_compass_v3.py adapters_v3

Inputs:
  data/raw/compass_questions_v3.jsonl   (corrected eval — 18 OPEN + 13 WITNESS)
  {adapter_path}/                       (LoRA weights)

Outputs:
  data/eval_v3/compass_eval.jsonl
  data/eval_v3/eval_summary.json
"""

import json
import os
import re
import sys

from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

# ── Paths ──────────────────────────────────────────────────────────────────────
QUESTIONS  = "/Users/tony_studio/phenomenological-compass/data/raw/compass_questions_v3.jsonl"
ADAPTER_NAME = sys.argv[1] if len(sys.argv) > 1 else "adapters_iter50"
ADAPTERS   = f"/Users/tony_studio/phenomenological-compass/{ADAPTER_NAME}"
OUT_DIR    = "/Users/tony_studio/phenomenological-compass/data/eval_v3"
MODEL_REPO = "thinkscan/Ministral-3-3B-Instruct-MLX"

COMPASS_SYSTEM = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of two signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN" or "SIGNAL: WITNESS", then a blank line, then "FRAMING:" or "THRESHOLD:", then your output."""

FIELD_MARKERS = [
    "threshold", "emerge", "alive", "resonan", "field",
    "probabili", "shockwave", "invitation", "participat", "witness",
    "open", "hold", "space", "relationship", "boundary", "vector",
    "oscillat", "entangle", "collapse", "potential", "unfold",
    "breathe", "liminal", "horizon", "depth", "dissolv",
]

def field_density(text):
    t = text.lower()
    hits = sum(1 for m in FIELD_MARKERS if m in t)
    return round(hits / len(FIELD_MARKERS), 3)

def parse_signal(text):
    m = re.search(r"SIGNAL:\s*(OPEN|WITNESS)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"

def run(model, tokenizer, system, user, max_tokens=400):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=False)

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_REPO} + adapters from {ADAPTERS}...")
model, tokenizer = load(MODEL_REPO, adapter_path=ADAPTERS)
print("Model ready.\n")

# ── Load questions ─────────────────────────────────────────────────────────────
questions = [json.loads(l) for l in open(QUESTIONS)]
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Eval set: {len(questions)} questions")
n_open    = sum(1 for q in questions if q["expected_signal"] == "OPEN")
n_witness = sum(1 for q in questions if q["expected_signal"] == "WITNESS")
print(f"  OPEN: {n_open} | WITNESS: {n_witness}")
print(f"  Adapter: {ADAPTER_NAME}\n")

# ── Run eval ───────────────────────────────────────────────────────────────────
results       = []
correct_total = 0

with open(os.path.join(OUT_DIR, "compass_eval.jsonl"), "w") as out_f:
    for i, rec in enumerate(questions):
        q        = rec["question"]
        expected = rec["expected_signal"]  # KEY: use expected_signal, not entropy proxy
        domain   = rec.get("domain", "unknown")
        source   = rec.get("source", "unknown")

        print(f"[{i+1:02d}/{len(questions)}] {rec['run_id']} ({domain}) → {expected}")

        response = run(model, tokenizer, COMPASS_SYSTEM, f"TASK: {q}")
        signal   = parse_signal(response)
        density  = field_density(response)
        tokens   = len(response.split())
        correct  = (signal == expected)

        if correct:
            correct_total += 1

        status = "✓" if correct else "✗"
        print(f"  {status} got {signal} | {tokens} tok | density {density:.3f}")

        result = {
            "run_id":          rec["run_id"],
            "question":        q,
            "domain":          domain,
            "source":          source,
            "expected_signal": expected,
            "got_signal":      signal,
            "correct":         correct,
            "response":        response.strip(),
            "token_count":     tokens,
            "field_density":   density,
        }
        results.append(result)
        out_f.write(json.dumps(result) + "\n")

# ── Summary ────────────────────────────────────────────────────────────────────
total = len(results)

open_results    = [r for r in results if r["expected_signal"] == "OPEN"]
witness_results = [r for r in results if r["expected_signal"] == "WITNESS"]

open_correct    = sum(1 for r in open_results    if r["correct"])
witness_correct = sum(1 for r in witness_results if r["correct"])

summary = {
    "eval_version":        "v3",
    "adapter":             ADAPTER_NAME,
    "eval_label_method":   "expected_signal_field",  # NOT entropy proxy
    "eval_source":         "compass_questions_v3.jsonl",
    "total":               total,
    "signal_accuracy":     round(correct_total / total, 3),
    "open_accuracy":       round(open_correct / len(open_results), 3)    if open_results    else None,
    "witness_accuracy":    round(witness_correct / len(witness_results), 3) if witness_results else None,
    "open_n":              len(open_results),
    "witness_n":           len(witness_results),
    "witness_rate":        round(sum(1 for r in results if r["got_signal"] == "WITNESS") / total, 3),
    "mean_tokens":         round(sum(r["token_count"] for r in results) / total, 1),
    "mean_field_density":  round(sum(r["field_density"] for r in results) / total, 3),
}

with open(os.path.join(OUT_DIR, "eval_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*55)
print(f"EVAL SUMMARY  (v3 eval — corrected labels)")
print("="*55)
for k, v in summary.items():
    print(f"  {k:<30} {v}")

print(f"\n→ Full results: {OUT_DIR}/compass_eval.jsonl")
