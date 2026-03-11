#!/usr/bin/env python3
"""
ablation.py — Four-condition ablation study
============================================
Proves the compass actually changes responses, not just adds latency.

Conditions:
  full   — Normal pipeline: compass → action model
  raw    — No compass: question → generic prompt → action model
  oracle — Human-annotated signal injected → action model (no compass reading)
  random — Shuffled signal → action model (no compass reading)

Usage:
    source ~/phenomenological-compass/.venv/bin/activate
    cd ~/phenomenological-compass
    HF_HOME=~/.cache/huggingface_local python3 eval_v9/ablation.py

    # Resume from partial run
    HF_HOME=~/.cache/huggingface_local python3 eval_v9/ablation.py --resume

    # Test with first N
    HF_HOME=~/.cache/huggingface_local python3 eval_v9/ablation.py --limit 5
"""

import json
import os
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

QUESTIONS_PATH = PROJECT / "eval" / "questions.jsonl"
RESULTS_PATH = PROJECT / "eval_v9" / "results" / "ablation_responses.jsonl"

ACTION_MAX_TOKENS = 800
SIGNALS = ["OPEN", "PAUSE", "WITNESS"]


def load_questions(path, limit=None):
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit:
        questions = questions[:limit]
    return questions


def random_signal(correct_signal):
    """Return a random wrong signal."""
    others = [s for s in SIGNALS if s != correct_signal]
    return random.choice(others)


def main():
    parser = argparse.ArgumentParser(description="Ablation study — 4 conditions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = load_questions(QUESTIONS_PATH, args.limit)
    if not questions:
        print("No questions found. Run: python3 eval/consolidate.py")
        return

    # Resume support
    done_ids = set()
    if args.resume and RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Resuming — {len(done_ids)} already completed")

    remaining = [q for q in questions if q["id"] not in done_ids]

    print(f"Ablation study: {len(remaining)} questions, 4 conditions each")
    print(f"  Conditions: full, raw, oracle, random")
    print(f"  Output: {RESULTS_PATH}\n")

    # Load pipeline with v0.9 compass
    from pipeline import Pipeline
    pipe = Pipeline(load_compass=True, load_action=True,
                    adapter_path="adapters_v9",
                    adapter_checkpoint="0000300_adapters.safetensors")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and done_ids else "w"

    random.seed(42)

    with open(RESULTS_PATH, mode) as f:
        for i, q in enumerate(remaining):
            q_text = q["question"]
            expected = q["expected_signal"]

            print(f"  [{i+1}/{len(remaining)}] {expected} {q['id']} — {q_text[:50]}...")

            # Condition 1: full pipeline
            t0 = time.time()
            full_result = pipe.run(q_text, max_tokens=ACTION_MAX_TOKENS)
            t_full = time.time() - t0

            # Condition 2: raw (no compass)
            t0 = time.time()
            raw_text, _, raw_thinking = pipe.raw(q_text, max_tokens=ACTION_MAX_TOKENS)
            t_raw = time.time() - t0

            # Condition 3: oracle (correct signal forced, no compass reading)
            t0 = time.time()
            oracle_result = pipe.run_with_signal(q_text, expected, max_tokens=ACTION_MAX_TOKENS)
            t_oracle = time.time() - t0

            # Condition 4: random (wrong signal forced, no compass reading)
            wrong_signal = random_signal(expected)
            t0 = time.time()
            random_result = pipe.run_with_signal(q_text, wrong_signal, max_tokens=ACTION_MAX_TOKENS)
            t_random = time.time() - t0

            record = {
                "id": q["id"],
                "question": q_text,
                "expected_signal": expected,
                "domain": q.get("domain", ""),
                # Full pipeline
                "full_signal": full_result["signal"],
                "full_signal_correct": full_result["signal"] == expected,
                "full_response": full_result["action_response"],
                "full_compass_reading": full_result["compass_response"],
                "t_full": round(t_full, 1),
                # Raw
                "raw_response": raw_text,
                "t_raw": round(t_raw, 1),
                # Oracle
                "oracle_signal": expected,
                "oracle_response": oracle_result["action_response"],
                "t_oracle": round(t_oracle, 1),
                # Random
                "random_signal": wrong_signal,
                "random_response": random_result["action_response"],
                "t_random": round(t_random, 1),
                # Meta
                "timestamp": datetime.now().isoformat(),
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            total_t = t_full + t_raw + t_oracle + t_random
            print(f"           full={t_full:.0f}s raw={t_raw:.0f}s oracle={t_oracle:.0f}s random={t_random:.0f}s (total={total_t:.0f}s)")

    print(f"\n{'='*60}")
    print(f"  ABLATION COMPLETE — {len(remaining)} questions × 4 conditions")
    print(f"  Results: {RESULTS_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
