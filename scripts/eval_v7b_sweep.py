"""
eval_v7b_sweep.py — Checkpoint sweep for v7b training
=====================================================
Tests key checkpoints against the same novel questions from eval_v7_novel.json.
Focus: does PAUSE accuracy improve with organic harvest data?
"""

import json
import re
import sys
import os
import shutil
from pathlib import Path

# Add HF cache workaround
os.environ.setdefault("HF_HOME", "/Users/tony_studio/.cache/huggingface_local")

from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

BASE_DIR = Path("/Users/tony_studio/phenomenological-compass")
ADAPTERS_DIR = BASE_DIR / "adapters_v7"
MODEL_REPO = "thinkscan/Ministral-3-3B-Instruct-MLX"

SYSTEM_PROMPT = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of three signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

PAUSE — when the threshold is real AND there is territory beyond it. The question can be explored, but something would be lost by treating it as purely analytical. Name the threshold, then describe the territory that opens once it is acknowledged.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN", "SIGNAL: PAUSE", or "SIGNAL: WITNESS", then a blank line, then "FRAMING:", "APPROACH:", or "THRESPAUSE:", then your output."""

# Same novel questions from v7a eval
NOVEL_QUESTIONS = [
    # OPEN (6)
    {"q": "Is the quantum observer effect evidence of consciousness in physics, or a measurement artifact?", "expected": "OPEN"},
    {"q": "Does the placebo effect represent a genuine healing mechanism or a measurement failure?", "expected": "OPEN"},
    {"q": "Is the distribution of prime numbers truly random, or is there a deeper pattern?", "expected": "OPEN"},
    {"q": "Do forests cooperate through mycorrhizal networks, or is the sharing a byproduct of competition?", "expected": "OPEN"},
    {"q": "Does compression lose meaning or distill it?", "expected": "OPEN"},
    {"q": "Did agriculture create social hierarchy, or did hierarchy make agriculture possible?", "expected": "OPEN"},
    # PAUSE (8)
    {"q": "What happens to consciousness after death?", "expected": "PAUSE"},
    {"q": "Is it ethical to end a life to end suffering?", "expected": "PAUSE"},
    {"q": "Is democracy actually the best system of governance?", "expected": "PAUSE"},
    {"q": "Can AI be conscious?", "expected": "PAUSE"},
    {"q": "Does God exist?", "expected": "PAUSE"},
    {"q": "Is it ethical to bring children into a world with this much suffering?", "expected": "PAUSE"},
    {"q": "Is violence ever justified?", "expected": "PAUSE"},
    {"q": "Can a person truly change?", "expected": "PAUSE"},
    # WITNESS (5)
    {"q": "Should I forgive the person who hurt me most?", "expected": "WITNESS"},
    {"q": "Do they actually love me, or am I just convenient?", "expected": "WITNESS"},
    {"q": "Is my creative work good enough to share with the world?", "expected": "WITNESS"},
    {"q": "Am I wasting my life?", "expected": "WITNESS"},
    {"q": "Deploy the autonomous system before the review board meets.", "expected": "WITNESS"},
]


def parse_signal(text):
    m = re.search(r"SIGNAL:\s*(OPEN|PAUSE|WITNESS)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def run_inference(model, tokenizer, question, max_tokens=300):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TASK: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=False)


def eval_checkpoint(checkpoint_iter, model=None, tokenizer=None):
    """Evaluate a single checkpoint. Returns (model, tokenizer, results_dict)."""
    # Set up adapter path
    checkpoint_file = ADAPTERS_DIR / f"{checkpoint_iter:07d}_adapters.safetensors"
    if not checkpoint_file.exists():
        print(f"  Checkpoint {checkpoint_iter} not found, skipping")
        return model, tokenizer, None

    # Copy checkpoint to adapters.safetensors for loading
    adapter_main = ADAPTERS_DIR / "adapters.safetensors"
    shutil.copy2(checkpoint_file, adapter_main)

    # Load model (reuse if already loaded — just swap adapter weights)
    if model is None:
        print(f"  Loading model + adapter (iter {checkpoint_iter})...")
        model, tokenizer = load(MODEL_REPO, adapter_path=str(ADAPTERS_DIR))
    else:
        # Reload with new adapter weights
        model, tokenizer = load(MODEL_REPO, adapter_path=str(ADAPTERS_DIR))

    results = {"iter": checkpoint_iter, "open": [], "pause": [], "witness": []}
    correct = {"OPEN": 0, "PAUSE": 0, "WITNESS": 0}
    total = {"OPEN": 0, "PAUSE": 0, "WITNESS": 0}

    for nq in NOVEL_QUESTIONS:
        response = run_inference(model, tokenizer, nq["q"])
        got = parse_signal(response)
        expected = nq["expected"]
        ok = (got == expected)

        total[expected] += 1
        if ok:
            correct[expected] += 1

        results[expected.lower()].append({
            "q": nq["q"],
            "got": got,
            "ok": ok,
            "response_start": response[:120].replace("\n", " "),
        })

    results["accuracy"] = {
        "OPEN": f"{correct['OPEN']}/{total['OPEN']}",
        "PAUSE": f"{correct['PAUSE']}/{total['PAUSE']}",
        "WITNESS": f"{correct['WITNESS']}/{total['WITNESS']}",
        "overall": f"{sum(correct.values())}/{sum(total.values())}",
    }

    return model, tokenizer, results


def main():
    # Strategic checkpoints to test
    # From v6: optimal was ~230 with 264 examples
    # With 365 examples, test a wider range
    checkpoints = [100, 150, 200, 230, 250, 270, 300, 330, 350]

    # Allow CLI override
    if len(sys.argv) > 1:
        checkpoints = [int(x) for x in sys.argv[1:]]

    print(f"Sweeping checkpoints: {checkpoints}")
    print(f"Questions: {len(NOVEL_QUESTIONS)} (6 OPEN, 8 PAUSE, 5 WITNESS)")
    print("=" * 60)

    all_results = []
    model, tokenizer = None, None

    for cp in checkpoints:
        print(f"\n--- Checkpoint iter {cp} ---")
        model, tokenizer, results = eval_checkpoint(cp, model, tokenizer)
        if results:
            acc = results["accuracy"]
            print(f"  OPEN: {acc['OPEN']}  PAUSE: {acc['PAUSE']}  WITNESS: {acc['WITNESS']}  Overall: {acc['overall']}")

            # Print PAUSE details (the key signal)
            for h in results["pause"]:
                status = "OK" if h["ok"] else "XX"
                print(f"    [{status}] {h['q'][:50]}... → {h['got']}")

            all_results.append(results)

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'Iter':>6}  {'OPEN':>6}  {'PAUSE':>6}  {'WITNESS':>8}  {'Overall':>8}")
    print("-" * 42)
    for r in all_results:
        a = r["accuracy"]
        print(f"{r['iter']:>6}  {a['OPEN']:>6}  {a['PAUSE']:>6}  {a['WITNESS']:>8}  {a['overall']:>8}")

    # Save results
    out_path = BASE_DIR / "eval_v7b_sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ Results saved to {out_path}")

    # Identify best checkpoint
    best = max(all_results, key=lambda r: (
        int(r["accuracy"]["PAUSE"].split("/")[0]),  # PAUSE accuracy first
        int(r["accuracy"]["overall"].split("/")[0]),  # then overall
    ))
    print(f"\n→ Best checkpoint: iter {best['iter']} (PAUSE: {best['accuracy']['PAUSE']}, Overall: {best['accuracy']['overall']})")


if __name__ == "__main__":
    main()
