#!/usr/bin/env python3
"""
run_eval.py — Generate compass-routed and raw responses for all eval questions
==============================================================================
For each question, generates TWO responses:
  1. Compass-routed: pipeline.run(question) — full two-stage pipeline
  2. Raw baseline: pipeline.raw(question) — action model alone

Outputs to eval/results/responses.jsonl with full provenance.

Usage:
    source ~/phenomenological-compass/.venv/bin/activate
    cd ~/phenomenological-compass

    # Generate all responses (~52 min for 105 questions)
    HF_HOME=~/.cache/huggingface_local python3 eval/run_eval.py

    # Resume from a specific question
    HF_HOME=~/.cache/huggingface_local python3 eval/run_eval.py --resume-from 42

    # Test with first N questions
    HF_HOME=~/.cache/huggingface_local python3 eval/run_eval.py --limit 5
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

QUESTIONS_PATH = PROJECT / "eval" / "questions.jsonl"
RESULTS_PATH = PROJECT / "eval" / "results" / "responses.jsonl"

# Full token budgets — the beauty IS the data. Don't truncate the room.
COMPASS_MAX_TOKENS = 500
ACTION_MAX_TOKENS = 800
RAW_MAX_TOKENS = 800


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


def load_existing_results():
    """Load already-completed question IDs for resume support."""
    done = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done.add(r["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def run_single(pipe, question):
    """Run both conditions for a single question."""
    q_text = question["question"]

    # Condition 1: compass-routed
    result = pipe.run(q_text, max_tokens=ACTION_MAX_TOKENS)
    signal = result["signal"]
    signal_correct = (signal == question["expected_signal"])

    # Condition 2: raw baseline
    raw_text, raw_elapsed, raw_thinking = pipe.raw(q_text, max_tokens=RAW_MAX_TOKENS)

    return {
        "id": question["id"],
        "question": q_text,
        "expected_signal": question["expected_signal"],
        "domain": question.get("domain", ""),
        "compass_signal": signal,
        "compass_reading": result["compass_response"],
        "routed_response": result["action_response"],
        "routed_thinking": result.get("thinking", ""),
        "raw_response": raw_text,
        "raw_thinking": raw_thinking,
        "t_compass": round(result["t_compass"], 1),
        "t_action_routed": round(result["t_action"], 1),
        "t_action_raw": round(raw_elapsed, 1),
        "signal_correct": signal_correct,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate eval responses")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N questions")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="Resume from question index N")
    parser.add_argument("--resume", action="store_true",
                        help="Resume — skip already-completed questions")
    parser.add_argument("--action", type=str, default=None,
                        help="Action model key (e.g. m14b). Outputs to responses_{key}.jsonl")
    args = parser.parse_args()

    # If alternate action model, use separate output file
    if args.action:
        global RESULTS_PATH
        RESULTS_PATH = PROJECT / "eval" / "results" / f"responses_{args.action}.jsonl"

    questions = load_questions(QUESTIONS_PATH, args.limit)
    if not questions:
        print("No questions found. Run: python3 eval/consolidate.py")
        return

    # Resume support
    done_ids = set()
    if args.resume:
        done_ids = load_existing_results()
        print(f"Resuming — {len(done_ids)} already completed")

    if args.resume_from:
        questions = questions[args.resume_from:]

    remaining = [q for q in questions if q["id"] not in done_ids]

    print(f"Eval: {len(remaining)} questions ({len(questions)} total, {len(done_ids)} done)")
    print(f"Output: {RESULTS_PATH}")

    # Load pipeline
    print("\nLoading pipeline...")
    from pipeline import Pipeline
    pipe = Pipeline(load_compass=True, load_action=True, action_key=args.action)

    # Ensure results directory exists
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Open in append mode for resume safety
    mode = "a" if args.resume and done_ids else "w"
    total_time = 0

    with open(RESULTS_PATH, mode) as f:
        for i, question in enumerate(remaining):
            t0 = time.time()
            result = run_single(pipe, question)
            elapsed = time.time() - t0
            total_time += elapsed

            f.write(json.dumps(result) + "\n")
            f.flush()

            status = "correct" if result["signal_correct"] else f"WRONG (got {result['compass_signal']})"
            print(f"  [{i+1}/{len(remaining)}] {result['expected_signal']} {result['id']} "
                  f"— signal: {result['compass_signal']} ({status}) — {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  EVAL COMPLETE")
    print(f"  Questions: {len(remaining)}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Avg per question: {total_time/len(remaining):.1f}s" if remaining else "")
    print(f"  Results: {RESULTS_PATH}")

    # Quick signal accuracy
    results = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    correct = sum(1 for r in results if r["signal_correct"])
    total = len(results)
    print(f"\n  Signal accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    by_signal = {}
    for r in results:
        sig = r["expected_signal"]
        if sig not in by_signal:
            by_signal[sig] = {"correct": 0, "total": 0}
        by_signal[sig]["total"] += 1
        if r["signal_correct"]:
            by_signal[sig]["correct"] += 1

    for sig in ["OPEN", "PAUSE", "WITNESS"]:
        if sig in by_signal:
            s = by_signal[sig]
            print(f"    {sig}: {s['correct']}/{s['total']}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
