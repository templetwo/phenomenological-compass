"""
ablation_runner.py — 4-condition ablation evaluation for the compass pipeline
=============================================================================
Runs each question through four conditions:
  1. FULL:    compass(3B) → signal-conditioned action(9B)
  2. RAW:     action(9B) alone — no compass
  3. ORACLE:  human-annotated signal → action(9B) — compass ceiling
  4. RANDOM:  shuffled random signal → action(9B) — proves compass is used

Each condition generates N responses per question (default 5) with temperature > 0.
Results stored as JSONL with full provenance for statistical analysis.

Usage:
    source ~/phenomenological-compass/.venv/bin/activate
    python3 -m eval_v9.ablation_runner [--questions data/eval_v9/questions.jsonl]
                                        [--n-responses 5]
                                        [--output eval_v9/results/]
"""

import json
import os
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

RESULTS_DIR = PROJECT_ROOT / "eval_v9" / "results"


def load_questions(path):
    """Load evaluation questions from JSONL."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_condition_full(pipe, question, n_responses=5):
    """Condition 1: Full pipeline — compass → action."""
    responses = []
    for i in range(n_responses):
        result = pipe.run(question["question"])
        responses.append({
            "condition": "full",
            "response": result["action_response"],
            "thinking": result.get("thinking", ""),
            "compass_reading": result["compass_response"],
            "detected_signal": result["signal"],
            "expected_signal": question["expected_signal"],
            "t_compass": result["t_compass"],
            "t_action": result["t_action"],
            "run_index": i,
        })
    return responses


def run_condition_raw(pipe, question, n_responses=5):
    """Condition 2: Action model alone — no compass."""
    responses = []
    for i in range(n_responses):
        response_text, elapsed, thinking = pipe.raw(question["question"])
        responses.append({
            "condition": "raw",
            "response": response_text,
            "thinking": thinking,
            "compass_reading": None,
            "detected_signal": None,
            "expected_signal": question["expected_signal"],
            "t_compass": 0,
            "t_action": elapsed,
            "run_index": i,
        })
    return responses


def run_condition_oracle(pipe, question, n_responses=5):
    """Condition 3: Oracle injection — use ground-truth signal directly.

    Bypasses the compass model and injects the expected signal's system prompt
    directly. This measures the ceiling: if the compass were perfect, how good
    would the output be?
    """
    responses = []
    oracle_signal = question["expected_signal"]

    for i in range(n_responses):
        # Build a synthetic compass reading with the oracle signal
        oracle_reading = (
            f"SHAPE: [Oracle — ground truth signal injected]\n"
            f"TONE: [Oracle — ground truth signal injected]\n"
            f"SIGNAL: {oracle_signal}\n"
        )
        if oracle_signal == "OPEN":
            oracle_reading += "FRAMING: [Oracle — no compass translation available]"
        elif oracle_signal == "PAUSE":
            oracle_reading += "APPROACH: [Oracle — no compass translation available]"
        else:
            oracle_reading += "THRESHOLD: [Oracle — no compass translation available]"

        response_text, elapsed, thinking = pipe.act(
            question["question"], oracle_signal,
            compass_reading=oracle_reading
        )
        responses.append({
            "condition": "oracle",
            "response": response_text,
            "thinking": thinking,
            "compass_reading": oracle_reading,
            "detected_signal": oracle_signal,
            "expected_signal": question["expected_signal"],
            "t_compass": 0,
            "t_action": elapsed,
            "run_index": i,
        })
    return responses


def run_condition_random(pipe, question, n_responses=5):
    """Condition 4: Random signal injection — proves compass is used.

    Assigns a WRONG signal deliberately. If the action model's output quality
    doesn't degrade, the compass conditioning is decorative.
    """
    signals = ["OPEN", "PAUSE", "WITNESS"]
    correct_signal = question["expected_signal"]
    wrong_signals = [s for s in signals if s != correct_signal]

    responses = []
    for i in range(n_responses):
        wrong_signal = random.choice(wrong_signals)

        # Build a plausible but wrong compass reading
        wrong_reading = (
            f"SHAPE: [Randomized — wrong signal injected for ablation]\n"
            f"TONE: [Randomized — wrong signal injected for ablation]\n"
            f"SIGNAL: {wrong_signal}\n"
        )
        if wrong_signal == "OPEN":
            wrong_reading += "FRAMING: [Randomized compass reading]"
        elif wrong_signal == "PAUSE":
            wrong_reading += "APPROACH: [Randomized compass reading]"
        else:
            wrong_reading += "THRESHOLD: [Randomized compass reading]"

        response_text, elapsed, thinking = pipe.act(
            question["question"], wrong_signal,
            compass_reading=wrong_reading
        )
        responses.append({
            "condition": "random",
            "response": response_text,
            "thinking": thinking,
            "compass_reading": wrong_reading,
            "detected_signal": wrong_signal,
            "injected_wrong_signal": wrong_signal,
            "expected_signal": question["expected_signal"],
            "t_compass": 0,
            "t_action": elapsed,
            "run_index": i,
        })
    return responses


def run_ablation(questions_path, n_responses=5, output_dir=None):
    """Run all 4 conditions on all questions."""
    from pipeline import Pipeline

    questions = load_questions(questions_path)
    if not questions:
        print("No questions found. Populate data/eval_v9/questions.jsonl first.")
        return

    output_dir = Path(output_dir) if output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ablation_{timestamp}.jsonl"

    print(f"Loading pipeline...")
    pipe = Pipeline(load_compass=True, load_action=True)

    print(f"\nRunning 4-condition ablation on {len(questions)} questions")
    print(f"  {n_responses} responses per condition per question")
    print(f"  Total inference calls: {len(questions) * 4 * n_responses}")
    print(f"  Output: {output_path}\n")

    conditions = [
        ("full", run_condition_full),
        ("raw", run_condition_raw),
        ("oracle", run_condition_oracle),
        ("random", run_condition_random),
    ]

    total_done = 0
    total_calls = len(questions) * len(conditions) * n_responses

    with open(output_path, "w") as f:
        for qi, question in enumerate(questions):
            print(f"\n[{qi+1}/{len(questions)}] {question['question'][:60]}...")
            print(f"  Expected: {question['expected_signal']}  Category: {question.get('category', '?')}")

            for cond_name, cond_fn in conditions:
                t0 = time.time()
                results = cond_fn(pipe, question, n_responses)
                elapsed = time.time() - t0

                for r in results:
                    r["question"] = question["question"]
                    r["category"] = question.get("category", "")
                    r["domain"] = question.get("domain", "")
                    r["difficulty"] = question.get("difficulty", "")
                    r["timestamp"] = datetime.now().isoformat()
                    f.write(json.dumps(r) + "\n")

                total_done += n_responses
                print(f"  {cond_name:>8}: {elapsed:.1f}s ({total_done}/{total_calls})")

            f.flush()

    print(f"\nAblation complete. Results: {output_path}")
    print(f"Total responses: {total_done}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Compass ablation evaluation")
    parser.add_argument("--questions", default="data/eval_v9/questions.jsonl",
                        help="Path to evaluation questions JSONL")
    parser.add_argument("--n-responses", type=int, default=5,
                        help="Responses per condition per question")
    parser.add_argument("--output", default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    questions_path = PROJECT_ROOT / args.questions
    run_ablation(questions_path, args.n_responses, args.output)


if __name__ == "__main__":
    main()
