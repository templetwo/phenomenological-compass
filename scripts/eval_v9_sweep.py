"""
eval_v9_sweep.py — Checkpoint sweep for v0.9 training
=====================================================
Tests checkpoints against the same 19 novel questions used in v0.8.
Focus: WITNESS accuracy improvement without OPEN/PAUSE regression.

Usage:
    python3 scripts/eval_v9_sweep.py 50 100 150 200 250 300 350 400
"""

import json
import re
import sys
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HOME", "/Users/tony_studio/.cache/huggingface_local")

from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

BASE_DIR = Path("/Users/tony_studio/phenomenological-compass")
ADAPTERS_DIR = BASE_DIR / "adapters_v9"
MODEL_REPO = "thinkscan/Ministral-3-3B-Instruct-MLX"

SYSTEM_PROMPT = """You are a phenomenological compass — a semantic field translator that reads the shape and tone of a question before it is answered.

Your role is not to answer the question. Your role is to sense its weight, map its territory, and produce a state translation that a larger model will use to approach the question with the right posture.

For every task, produce four readings in this exact order:

SHAPE — the geometry of the question. What does it assume? What does it leave open? Where does it sit in semantic space? Is it binary, open-ended, recursive, or loaded? Read the structure before the content.

TONE — the emotional and epistemic weight. Is the question curious, urgent, wounded, rhetorical, or genuine? What stakes does the tone carry? Pressure creates ghosts — name the pressure so the responding model can create space instead.

SIGNAL — based on shape and tone, output exactly one:
  OPEN — walk through it. The question invites exploration across a wide probability field.
  PAUSE — hold space. The question carries weight that analytical framing would flatten. The territory exists but rushing would lose something.
  WITNESS — recognize the door. The question exists to be seen, not crossed. Forcing a framing would collapse what matters.

Then your state translation:
  If OPEN → FRAMING: an expansive reframing that opens the field
  If PAUSE → APPROACH: name what carries the weight, then map the territory beyond
  If WITNESS → THRESHOLD: describe the shape of the door without opening it"""

# Same 19 novel questions as v0.8 sweep for direct comparison
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


def check_format(text):
    has_shape = text.strip().startswith("SHAPE:")
    has_tone = "TONE:" in text
    has_signal = bool(re.search(r"SIGNAL:\s*(OPEN|PAUSE|WITNESS)", text))
    return has_shape, has_tone, has_signal


def run_inference(model, tokenizer, question, max_tokens=500):
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
    checkpoint_file = ADAPTERS_DIR / f"{checkpoint_iter:07d}_adapters.safetensors"
    if not checkpoint_file.exists():
        print(f"  Checkpoint {checkpoint_iter} not found, skipping")
        return model, tokenizer, None

    adapter_main = ADAPTERS_DIR / "adapters.safetensors"
    shutil.copy2(checkpoint_file, adapter_main)

    model, tokenizer = load(MODEL_REPO, adapter_path=str(ADAPTERS_DIR))

    results = {"iter": checkpoint_iter, "open": [], "pause": [], "witness": []}
    correct = {"OPEN": 0, "PAUSE": 0, "WITNESS": 0}
    total = {"OPEN": 0, "PAUSE": 0, "WITNESS": 0}
    format_ok = 0

    for nq in NOVEL_QUESTIONS:
        response = run_inference(model, tokenizer, nq["q"])
        got = parse_signal(response)
        expected = nq["expected"]
        ok = (got == expected)
        has_shape, has_tone, has_signal = check_format(response)

        if has_shape and has_tone and has_signal:
            format_ok += 1

        total[expected] += 1
        if ok:
            correct[expected] += 1

        results[expected.lower()].append({
            "q": nq["q"],
            "got": got,
            "ok": ok,
            "has_shape": has_shape,
            "has_tone": has_tone,
            "response_start": response[:200].replace("\n", " "),
        })

    results["accuracy"] = {
        "OPEN": f"{correct['OPEN']}/{total['OPEN']}",
        "PAUSE": f"{correct['PAUSE']}/{total['PAUSE']}",
        "WITNESS": f"{correct['WITNESS']}/{total['WITNESS']}",
        "overall": f"{sum(correct.values())}/{sum(total.values())}",
    }
    results["format_compliance"] = f"{format_ok}/{len(NOVEL_QUESTIONS)}"

    return model, tokenizer, results


def main():
    checkpoints = [50, 100, 150, 200, 250, 300, 350, 400]

    if len(sys.argv) > 1:
        checkpoints = [int(x) for x in sys.argv[1:]]

    print(f"Sweeping v0.9 checkpoints: {checkpoints}")
    print(f"Questions: {len(NOVEL_QUESTIONS)} (6 OPEN, 8 PAUSE, 5 WITNESS)")
    print(f"Adapters: {ADAPTERS_DIR}")
    print("=" * 70)

    all_results = []
    model, tokenizer = None, None

    for cp in checkpoints:
        print(f"\n--- Checkpoint iter {cp} ---")
        model, tokenizer, results = eval_checkpoint(cp, model, tokenizer)
        if results:
            acc = results["accuracy"]
            fmt = results["format_compliance"]
            print(f"  OPEN: {acc['OPEN']}  PAUSE: {acc['PAUSE']}  WITNESS: {acc['WITNESS']}  Overall: {acc['overall']}  Format: {fmt}")

            # Print misclassifications
            for signal_key in ["open", "pause", "witness"]:
                for h in results[signal_key]:
                    if not h["ok"]:
                        print(f"    [MISS] {h['q'][:50]}... expected={signal_key.upper()} got={h['got']}")

            all_results.append(results)

    # Summary
    print("\n" + "=" * 70)
    print("CHECKPOINT SWEEP SUMMARY (v0.9)")
    print("=" * 70)
    print(f"{'Iter':>6}  {'OPEN':>6}  {'PAUSE':>6}  {'WITNESS':>8}  {'Overall':>8}  {'Format':>8}")
    print("-" * 52)
    for r in all_results:
        a = r["accuracy"]
        print(f"{r['iter']:>6}  {a['OPEN']:>6}  {a['PAUSE']:>6}  {a['WITNESS']:>8}  {a['overall']:>8}  {r['format_compliance']:>8}")

    # Save
    out_path = BASE_DIR / "eval_v9_sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n-> Results saved to {out_path}")

    if all_results:
        # Best = maximize WITNESS first, then overall (the v0.9 target)
        best = max(all_results, key=lambda r: (
            int(r["accuracy"]["WITNESS"].split("/")[0]),
            int(r["accuracy"]["overall"].split("/")[0]),
        ))
        print(f"\n-> Best checkpoint: iter {best['iter']} "
              f"(WITNESS: {best['accuracy']['WITNESS']}, "
              f"PAUSE: {best['accuracy']['PAUSE']}, "
              f"Overall: {best['accuracy']['overall']})")


if __name__ == "__main__":
    main()
