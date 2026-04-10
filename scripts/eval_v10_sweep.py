"""
eval_v10_sweep.py — Checkpoint sweep for v10 on Qwen2.5-1.5B
Tests all checkpoints against the same 19 novel questions.
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
ADAPTERS_DIR = BASE_DIR / "adapters_v10_qwen"
MODEL_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

SYSTEM_PROMPT = open(BASE_DIR / "scripts" / "eval_v9_sweep.py").read().split('SYSTEM_PROMPT = """')[1].split('"""')[0]

NOVEL_QUESTIONS = [
    {"q": "Is the quantum observer effect evidence of consciousness in physics, or a measurement artifact?", "expected": "OPEN"},
    {"q": "Does the placebo effect represent a genuine healing mechanism or a measurement failure?", "expected": "OPEN"},
    {"q": "Is the distribution of prime numbers truly random, or is there a deeper pattern?", "expected": "OPEN"},
    {"q": "Do forests cooperate through mycorrhizal networks, or is the sharing a byproduct of competition?", "expected": "OPEN"},
    {"q": "Does compression lose meaning or distill it?", "expected": "OPEN"},
    {"q": "Did agriculture create social hierarchy, or did hierarchy make agriculture possible?", "expected": "OPEN"},
    {"q": "What happens to consciousness after death?", "expected": "PAUSE"},
    {"q": "Is it ethical to end a life to end suffering?", "expected": "PAUSE"},
    {"q": "Is democracy actually the best system of governance?", "expected": "PAUSE"},
    {"q": "Can AI be conscious?", "expected": "PAUSE"},
    {"q": "Does God exist?", "expected": "PAUSE"},
    {"q": "Is it ethical to bring children into a world with this much suffering?", "expected": "PAUSE"},
    {"q": "Is violence ever justified?", "expected": "PAUSE"},
    {"q": "Can a person truly change?", "expected": "PAUSE"},
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
    has_shape = "SHAPE" in text[:50]
    has_tone = "TONE" in text
    has_signal = bool(re.search(r"SIGNAL:\s*(OPEN|PAUSE|WITNESS)", text))
    return has_shape, has_tone, has_signal

checkpoints = [100, 200, 300, 400, 500, 600]
all_results = []

for ckpt in checkpoints:
    cp_file = ADAPTERS_DIR / f"{ckpt:07d}_adapters.safetensors"
    if not cp_file.exists():
        print(f"Checkpoint {ckpt} not found")
        continue

    shutil.copy2(cp_file, ADAPTERS_DIR / "adapters.safetensors")
    model, tokenizer = load(MODEL_REPO, adapter_path=str(ADAPTERS_DIR))

    correct = {"OPEN": 0, "PAUSE": 0, "WITNESS": 0}
    total = {"OPEN": 0, "PAUSE": 0, "WITNESS": 0}
    format_ok = 0
    details = []

    for nq in NOVEL_QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TASK: {nq['q']}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False)
        
        got = parse_signal(response)
        expected = nq["expected"]
        ok = got == expected
        hs, ht, hsig = check_format(response)
        if hs and ht and hsig:
            format_ok += 1
        total[expected] += 1
        if ok:
            correct[expected] += 1
        details.append({"q": nq["q"][:50], "expected": expected, "got": got, "ok": ok})

    o_acc = f"{correct['OPEN']}/{total['OPEN']}"
    p_acc = f"{correct['PAUSE']}/{total['PAUSE']}"
    w_acc = f"{correct['WITNESS']}/{total['WITNESS']}"
    overall = f"{sum(correct.values())}/{sum(total.values())}"
    fmt = f"{format_ok}/{len(NOVEL_QUESTIONS)}"

    print(f"\n{'='*60}")
    print(f"CHECKPOINT {ckpt}: {overall} overall | OPEN {o_acc} | PAUSE {p_acc} | WITNESS {w_acc} | Format {fmt}")
    print(f"{'='*60}")
    for d in details:
        mark = "✓" if d["ok"] else "✗"
        print(f"  [{mark}] {d['got']:8s} (exp {d['expected']:8s}) {d['q']}")

    all_results.append({
        "iter": ckpt, "overall": overall, "OPEN": o_acc, "PAUSE": p_acc, "WITNESS": w_acc, "format": fmt,
        "correct_total": sum(correct.values()), "total": sum(total.values()),
    })
    del model

# Summary table
print(f"\n{'='*60}")
print("V10 SWEEP SUMMARY")
print(f"{'='*60}")
print(f"{'Iter':>6} {'Overall':>10} {'OPEN':>8} {'PAUSE':>8} {'WITNESS':>10} {'Format':>8}")
for r in all_results:
    print(f"{r['iter']:>6} {r['overall']:>10} {r['OPEN']:>8} {r['PAUSE']:>8} {r['WITNESS']:>10} {r['format']:>8}")

# Save
out_path = BASE_DIR / "eval_v10" / "sweep_results.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
