#!/usr/bin/env python3
"""
run_eval_parallel.py — Fast parallel evaluation with optimized token budgets
=============================================================================
Splits 105 questions across N worker processes with threading for compass/raw
overlap within each worker. Optimized max_tokens for eval (not full production).

On 36GB M4 Max: 2 workers share GPU via MLX kernel interleaving.
Each worker pipelines compass→action→raw to maximize GPU utilization.

Usage:
    source ~/phenomenological-compass/.venv/bin/activate
    cd ~/phenomenological-compass
    HF_HOME=~/.cache/huggingface_local python3 eval/run_eval_parallel.py
    HF_HOME=~/.cache/huggingface_local python3 eval/run_eval_parallel.py --workers 2
"""

import json
import os
import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

QUESTIONS_PATH = PROJECT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT / "eval" / "results"

# Optimized token budgets for eval — we need enough for quality assessment,
# not production-length responses
COMPASS_MAX_TOKENS = 350  # compass readings are ~200 tokens typically
ACTION_MAX_TOKENS = 500   # enough for the judge to assess depth/quality
RAW_MAX_TOKENS = 500      # same budget for fair comparison


def load_questions(path):
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def worker_fn(worker_id, question_chunk, result_path):
    """Worker process: loads its own pipeline, processes its chunk."""
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

    # Stagger model loading to avoid disk contention
    time.sleep(worker_id * 3)

    print(f"  [W{worker_id}] Loading pipeline ({len(question_chunk)} questions)...")
    from pipeline import Pipeline, generate, parse_signal, split_thinking
    from pipeline import COMPASS_SYSTEM, OPEN_SYSTEM, PAUSE_SYSTEM, WITNESS_SYSTEM, RAW_SYSTEM
    pipe = Pipeline(load_compass=True, load_action=True)
    print(f"  [W{worker_id}] Pipeline ready.")

    results = []
    for i, question in enumerate(question_chunk):
        t0 = time.time()
        q_text = question["question"]

        # Stage 1: Compass (fast — 3B model, ~3s)
        t_c0 = time.time()
        compass_response = generate(
            pipe.compass_model, pipe.compass_tokenizer,
            COMPASS_SYSTEM, f"TASK: {q_text}", max_tokens=COMPASS_MAX_TOKENS
        )
        signal = parse_signal(compass_response)
        t_compass = time.time() - t_c0

        signal_correct = (signal == question["expected_signal"])

        # Stage 2: Action model — routed (conditioned on compass)
        if signal == "OPEN":
            system = OPEN_SYSTEM
        elif signal == "PAUSE":
            system = PAUSE_SYSTEM
        else:
            system = WITNESS_SYSTEM

        user_msg = f"COMPASS READING:\n{compass_response.strip()}\n\nORIGINAL QUESTION:\n{q_text}"

        t_a0 = time.time()
        routed_raw = generate(
            pipe.action_model, pipe.action_tokenizer,
            system, user_msg, max_tokens=ACTION_MAX_TOKENS
        )
        t_action_routed = time.time() - t_a0
        routed_thinking, routed_response = split_thinking(routed_raw)

        # Stage 3: Raw baseline (same action model, no compass)
        t_r0 = time.time()
        raw_raw = generate(
            pipe.action_model, pipe.action_tokenizer,
            RAW_SYSTEM, q_text, max_tokens=RAW_MAX_TOKENS
        )
        t_action_raw = time.time() - t_r0
        raw_thinking, raw_response = split_thinking(raw_raw)

        record = {
            "id": question["id"],
            "question": q_text,
            "expected_signal": question["expected_signal"],
            "domain": question.get("domain", ""),
            "compass_signal": signal,
            "compass_reading": compass_response.strip(),
            "routed_response": routed_response,
            "routed_thinking": routed_thinking,
            "raw_response": raw_response,
            "raw_thinking": raw_thinking,
            "t_compass": round(t_compass, 1),
            "t_action_routed": round(t_action_routed, 1),
            "t_action_raw": round(t_action_raw, 1),
            "signal_correct": signal_correct,
            "timestamp": datetime.now().isoformat(),
            "worker_id": worker_id,
        }
        results.append(record)

        elapsed = time.time() - t0
        status = "correct" if signal_correct else f"WRONG→{signal}"
        print(f"  [W{worker_id}] [{i+1}/{len(question_chunk)}] {question['expected_signal']} "
              f"{question['id']} — {status} — {elapsed:.1f}s")

    # Write chunk results
    with open(result_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"  [W{worker_id}] Done. {len(results)} results → {result_path}")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Parallel eval runner")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel workers (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N questions")
    args = parser.parse_args()

    questions = load_questions(QUESTIONS_PATH)
    if args.limit:
        questions = questions[:args.limit]

    n_workers = args.workers
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Split questions into chunks
    chunk_size = len(questions) // n_workers
    chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        end = start + chunk_size if i < n_workers - 1 else len(questions)
        chunks.append(questions[start:end])

    print(f"Parallel eval: {len(questions)} questions across {n_workers} workers")
    print(f"  Chunk sizes: {[len(c) for c in chunks]}")
    print(f"  ~{len(questions) * 24 / n_workers / 60:.0f} min estimated\n")

    # Launch workers
    t0 = time.time()
    processes = []
    chunk_paths = []

    for i in range(n_workers):
        chunk_path = RESULTS_DIR / f"responses_worker{i}.jsonl"
        chunk_paths.append(chunk_path)
        p = mp.Process(target=worker_fn, args=(i, chunks[i], str(chunk_path)))
        processes.append(p)

    # Start all workers
    for p in processes:
        p.start()

    # Wait for all
    for p in processes:
        p.join()

    total_time = time.time() - t0

    # Merge results
    merged_path = RESULTS_DIR / "responses.jsonl"
    all_results = []
    for cp in chunk_paths:
        if cp.exists():
            with open(cp) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_results.append(json.loads(line))

    # Sort by ID to maintain original order
    signal_order = {"open": 0, "pause": 1, "witness": 2}
    all_results.sort(key=lambda r: (
        signal_order.get(r["id"].split("_")[0], 3),
        int(r["id"].split("_")[1])
    ))

    with open(merged_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Cleanup chunk files
    for cp in chunk_paths:
        if cp.exists():
            cp.unlink()

    # Summary
    correct = sum(1 for r in all_results if r["signal_correct"])
    total = len(all_results)

    print(f"\n{'=' * 60}")
    print(f"  PARALLEL EVAL COMPLETE")
    print(f"  Workers: {n_workers}")
    print(f"  Questions: {total}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Speedup: ~{len(questions) * 24 / total_time:.1f}x vs sequential estimate")
    print(f"  Signal accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    by_signal = {}
    for r in all_results:
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

    print(f"\n  Results: {merged_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
