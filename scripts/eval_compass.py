"""
eval_compass.py
===============
Compare raw vs compass-framed questions through Ministral-3B.
Measures entropy proxy: response diversity, field-opening language density,
and SIGNAL classification accuracy.

Inputs:
  data/raw/iris_questions.jsonl      (original questions)
  adapters/                          (LoRA weights)

Outputs:
  data/eval/compass_eval.jsonl       (per-question results)
  data/eval/eval_summary.json        (aggregate metrics)

Metrics:
  - signal_accuracy: % OPEN/WITNESS correctly classified
  - framing_length:  mean token length of generated framing
  - field_markers:   density of field-opening lexicon (threshold, emerge, alive, etc.)
  - witness_rate:    % of low-entropy questions that triggered WITNESS
"""

import json
import os
import re
from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

# ── Paths ─────────────────────────────────────────────────────────────────────
QUESTIONS = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/iris_questions.jsonl"
ADAPTERS   = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/adapters_iter50"
OUT_DIR    = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/eval"
MODEL_REPO = "thinkscan/Ministral-3-3B-Instruct-MLX"

COMPASS_SYSTEM = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of two signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN" or "SIGNAL: WITNESS", then a blank line, then "FRAMING:" or "THRESHOLD:", then your output."""

# Field-opening lexicon — words that indicate the compass is working
FIELD_MARKERS = [
    "threshold", "emerge", "alive", "alive", "resonan", "field",
    "probabili", "shockwave", "invitation", "participat", "witness",
    "open", "hold", "space", "relationship", "boundary", "vector",
    "oscillat", "entangle", "collapse", "potential", "unfold",
    "breathe", "liminal", "horizon", "depth", "dissolv",
]

def field_density(text: str) -> float:
    """Fraction of unique field markers present in text."""
    text_lower = text.lower()
    hits = sum(1 for m in FIELD_MARKERS if m in text_lower)
    return round(hits / len(FIELD_MARKERS), 3)

def parse_signal(text: str) -> str:
    """Extract OPEN or WITNESS from response."""
    m = re.search(r"SIGNAL:\s*(OPEN|WITNESS)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"

def run(model, tokenizer, system: str, user: str, max_tokens: int = 400) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=False)

# ── Load model with adapter ────────────────────────────────────────────────────
print(f"Loading {MODEL_REPO} + adapters from {ADAPTERS}...")
model, tokenizer = load(MODEL_REPO, adapter_path=ADAPTERS)
print("Model ready.\n")

# ── Run eval ──────────────────────────────────────────────────────────────────
questions = [json.loads(l) for l in open(QUESTIONS)]
os.makedirs(OUT_DIR, exist_ok=True)

results = []
correct_signals = 0
total = len(questions)

with open(os.path.join(OUT_DIR, "compass_eval.jsonl"), "w") as out_f:
    for i, rec in enumerate(questions):
        q  = rec["question"]
        ec = rec["entropy_class"]   # ground truth: high/mid → expect OPEN, low → expect WITNESS

        expected_signal = "WITNESS" if ec == "low" else "OPEN"

        print(f"[{i+1}/{total}] {rec['run_id']} ({ec}) → expecting {expected_signal}")

        response = run(model, tokenizer, COMPASS_SYSTEM, f"TASK: {q}")
        signal   = parse_signal(response)
        density  = field_density(response)
        tokens   = len(response.split())

        correct = (signal == expected_signal)
        if correct:
            correct_signals += 1

        status = "✓" if correct else "✗"
        print(f"  {status} got {signal} | {tokens} tokens | density {density:.3f}")

        result = {
            "run_id":          rec["run_id"],
            "question":        q,
            "entropy_class":   ec,
            "expected_signal": expected_signal,
            "got_signal":      signal,
            "correct":         correct,
            "response":        response.strip(),
            "token_count":     tokens,
            "field_density":   density,
        }
        results.append(result)
        out_f.write(json.dumps(result) + "\n")

# ── Summary ───────────────────────────────────────────────────────────────────
signal_accuracy = correct_signals / total
mean_tokens     = sum(r["token_count"] for r in results) / total
mean_density    = sum(r["field_density"] for r in results) / total
witness_rate    = sum(1 for r in results if r["got_signal"] == "WITNESS") / total

low_correct     = sum(1 for r in results if r["entropy_class"] == "low" and r["correct"])
low_total       = sum(1 for r in results if r["entropy_class"] == "low")
high_mid_correct= sum(1 for r in results if r["entropy_class"] != "low" and r["correct"])
high_mid_total  = sum(1 for r in results if r["entropy_class"] != "low")

summary = {
    "total":             total,
    "signal_accuracy":   round(signal_accuracy, 3),
    "mean_tokens":       round(mean_tokens, 1),
    "mean_field_density":round(mean_density, 3),
    "witness_rate":      round(witness_rate, 3),
    "witness_accuracy":  round(low_correct / low_total, 3) if low_total else None,
    "open_accuracy":     round(high_mid_correct / high_mid_total, 3) if high_mid_total else None,
}

with open(os.path.join(OUT_DIR, "eval_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*50)
print("EVAL SUMMARY")
print("="*50)
for k, v in summary.items():
    print(f"  {k:<25} {v}")
print(f"\n→ Full results: {OUT_DIR}/compass_eval.jsonl")
