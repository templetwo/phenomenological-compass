#!/usr/bin/env python3
"""
entropy_profile.py — Measure probability field restructuring under compass conditioning
========================================================================================
For each eval question under compass-routed AND raw conditions:
  1. Forward-pass the action model token by token
  2. At each step, compute Shannon entropy of the logit distribution
  3. Record the full entropy trace

Produces signal-specific entropy signatures proving the compass restructures
the action model's probability field.

Usage:
    source ~/phenomenological-compass/.venv/bin/activate
    cd ~/phenomenological-compass
    HF_HOME=~/.cache/huggingface_local python3 eval_v9/entropy_profile.py

    # Subset
    HF_HOME=~/.cache/huggingface_local python3 eval_v9/entropy_profile.py --limit 10
"""

import json
import os
import sys
import time
import argparse
import math
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

QUESTIONS_PATH = PROJECT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT / "eval_v9" / "results"
PROFILES_PATH = RESULTS_DIR / "entropy_profiles.jsonl"
SUMMARY_PATH = RESULTS_DIR / "entropy_summary.json"

MAX_TOKENS = 300  # enough for signature, faster than full 800


def get_token_entropies(model, tokenizer, prompt_text, max_tokens=MAX_TOKENS):
    """Generate tokens one by one, recording entropy at each step."""
    tokens = tokenizer.encode(prompt_text)
    entropies = []

    for step in range(max_tokens):
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        # Get logits for last token position
        last_logits = logits[0, -1]
        # Softmax to get probabilities
        probs = mx.softmax(last_logits, axis=-1)
        # Shannon entropy
        log_probs = mx.log(probs + 1e-10)
        entropy = -mx.sum(probs * log_probs).item()
        entropies.append(entropy)

        # Greedy next token
        next_token = mx.argmax(last_logits).item()
        tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        # Evaluate to prevent graph buildup
        mx.eval(last_logits)

    return entropies


def build_routed_prompt(tokenizer, question, compass_reading, signal):
    """Build the action model prompt for compass-routed condition."""
    from pipeline import OPEN_SYSTEM, PAUSE_SYSTEM, WITNESS_SYSTEM

    system_prompts = {"OPEN": OPEN_SYSTEM, "PAUSE": PAUSE_SYSTEM, "WITNESS": WITNESS_SYSTEM}
    system = system_prompts.get(signal, OPEN_SYSTEM)

    user_msg = f"COMPASS READING:\n{compass_reading}\n\nORIGINAL QUESTION:\n{question}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_raw_prompt(tokenizer, question):
    """Build the action model prompt for raw (no compass) condition."""
    from pipeline import RAW_SYSTEM

    messages = [
        {"role": "system", "content": RAW_SYSTEM},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_stats(entropies):
    """Compute summary statistics for an entropy trace."""
    if not entropies:
        return {}
    arr = np.array(entropies)
    # Linear regression slope
    x = np.arange(len(arr))
    if len(arr) > 1:
        slope = np.polyfit(x, arr, 1)[0]
    else:
        slope = 0.0

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "variance": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "slope": float(slope),
        "n_tokens": len(arr),
    }


def jsd(p_trace, q_trace):
    """Jensen-Shannon Divergence between two entropy traces (of their distributions)."""
    # Bin the entropy values and compute JSD of the binned distributions
    min_val = min(min(p_trace), min(q_trace))
    max_val = max(max(p_trace), max(q_trace))
    bins = np.linspace(min_val, max_val, 50)

    p_hist, _ = np.histogram(p_trace, bins=bins, density=True)
    q_hist, _ = np.histogram(q_trace, bins=bins, density=True)

    # Normalize
    p_hist = p_hist / (p_hist.sum() + 1e-10)
    q_hist = q_hist / (q_hist.sum() + 1e-10)

    # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = (P+Q)/2
    m = 0.5 * (p_hist + q_hist)
    kl_pm = np.sum(p_hist * np.log((p_hist + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q_hist * np.log((q_hist + 1e-10) / (m + 1e-10)))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def main():
    parser = argparse.ArgumentParser(description="Entropy profiling")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    if args.limit:
        questions = questions[:args.limit]

    # We need compass readings — load from v0.9 eval responses
    responses_path = PROJECT / "eval" / "results_v9" / "responses.jsonl"
    if not responses_path.exists():
        print("ERROR: Need eval/results/responses.jsonl for compass readings")
        print("Run eval first: python3 eval/run_eval.py")
        sys.exit(1)

    responses = {}
    with open(responses_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                responses[r["id"]] = r

    # Resume
    done_ids = set()
    if args.resume and PROFILES_PATH.exists():
        with open(PROFILES_PATH) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Resuming — {len(done_ids)} already profiled")

    remaining = [q for q in questions if q["id"] not in done_ids and q["id"] in responses]

    print(f"Entropy profiling: {len(remaining)} questions")
    print(f"  Max tokens per trace: {MAX_TOKENS}")
    print(f"  Output: {PROFILES_PATH}\n")

    # Load action model only (we use pre-computed compass readings)
    from pipeline import ACTION_MODELS, DEFAULT_ACTION
    action_config = ACTION_MODELS[DEFAULT_ACTION]
    print(f"Loading action model ({action_config['name']})...")
    action_model, action_tokenizer = load(action_config["repo"])
    print("  Ready.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and done_ids else "w"

    with open(PROFILES_PATH, mode) as f:
        for i, q in enumerate(remaining):
            q_id = q["id"]
            resp = responses[q_id]

            print(f"  [{i+1}/{len(remaining)}] {resp['expected_signal']} {q_id} — {q['question'][:50]}...")

            # Routed condition: use existing compass reading
            routed_prompt = build_routed_prompt(
                action_tokenizer,
                q["question"],
                resp["compass_reading"],
                resp["compass_signal"],
            )

            t0 = time.time()
            routed_entropies = get_token_entropies(action_model, action_tokenizer, routed_prompt)
            t_routed = time.time() - t0

            # Raw condition
            raw_prompt = build_raw_prompt(action_tokenizer, q["question"])

            t1 = time.time()
            raw_entropies = get_token_entropies(action_model, action_tokenizer, raw_prompt)
            t_raw = time.time() - t1

            routed_stats = compute_stats(routed_entropies)
            raw_stats = compute_stats(raw_entropies)

            # JSD between the two entropy traces
            divergence = jsd(routed_entropies, raw_entropies) if routed_entropies and raw_entropies else 0.0

            record = {
                "id": q_id,
                "expected_signal": resp["expected_signal"],
                "compass_signal": resp["compass_signal"],
                "routed_entropies": routed_entropies,
                "raw_entropies": raw_entropies,
                "routed_stats": routed_stats,
                "raw_stats": raw_stats,
                "jsd": divergence,
                "t_routed": round(t_routed, 1),
                "t_raw": round(t_raw, 1),
                "timestamp": datetime.now().isoformat(),
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            print(f"           routed: mean={routed_stats['mean']:.2f} slope={routed_stats['slope']:.4f} "
                  f"| raw: mean={raw_stats['mean']:.2f} slope={raw_stats['slope']:.4f} "
                  f"| JSD={divergence:.4f} ({t_routed+t_raw:.0f}s)")

    # ── Generate summary ─────────────────────────────────────────────────────
    print(f"\nGenerating summary...")

    all_profiles = []
    with open(PROFILES_PATH) as f:
        for line in f:
            if line.strip():
                all_profiles.append(json.loads(line))

    summary = {}
    for signal in ["OPEN", "PAUSE", "WITNESS"]:
        signal_profiles = [p for p in all_profiles if p["expected_signal"] == signal]
        if not signal_profiles:
            continue

        routed_means = [p["routed_stats"]["mean"] for p in signal_profiles]
        raw_means = [p["raw_stats"]["mean"] for p in signal_profiles]
        routed_slopes = [p["routed_stats"]["slope"] for p in signal_profiles]
        raw_slopes = [p["raw_stats"]["slope"] for p in signal_profiles]
        jsds = [p["jsd"] for p in signal_profiles]

        summary[signal] = {
            "n": len(signal_profiles),
            "routed_mean_entropy": float(np.mean(routed_means)),
            "raw_mean_entropy": float(np.mean(raw_means)),
            "entropy_delta": float(np.mean(routed_means) - np.mean(raw_means)),
            "routed_mean_slope": float(np.mean(routed_slopes)),
            "raw_mean_slope": float(np.mean(raw_slopes)),
            "mean_jsd": float(np.mean(jsds)),
        }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ENTROPY PROFILING COMPLETE")
    print(f"{'='*70}")
    print(f"  {'Signal':>8}  {'Routed H':>10}  {'Raw H':>10}  {'ΔH':>8}  {'Routed slope':>14}  {'Raw slope':>12}  {'JSD':>8}")
    print(f"  {'-'*66}")
    for sig in ["OPEN", "PAUSE", "WITNESS"]:
        if sig in summary:
            s = summary[sig]
            print(f"  {sig:>8}  {s['routed_mean_entropy']:>10.3f}  {s['raw_mean_entropy']:>10.3f}  "
                  f"{s['entropy_delta']:>+8.3f}  {s['routed_mean_slope']:>14.5f}  "
                  f"{s['raw_mean_slope']:>12.5f}  {s['mean_jsd']:>8.4f}")
    print(f"{'='*70}")
    print(f"  Profiles: {PROFILES_PATH}")
    print(f"  Summary:  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
