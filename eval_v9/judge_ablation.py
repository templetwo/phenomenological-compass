#!/usr/bin/env python3
"""
judge_ablation.py — LLM judge for ablation study pairwise comparisons
=====================================================================
Judges 6 pairwise comparisons from 4 conditions:
  full vs raw, full vs random, full vs oracle,
  oracle vs raw, oracle vs random, raw vs random

Uses Claude Sonnet, position-debiased (AB+BA), 3x self-consistency.

Usage:
    ANTHROPIC_API_KEY=... python3 eval_v9/judge_ablation.py
    ANTHROPIC_API_KEY=... python3 eval_v9/judge_ablation.py --resume
    ANTHROPIC_API_KEY=... python3 eval_v9/judge_ablation.py --limit 5
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

from eval.rubrics import RUBRIC_TEXT, DIMENSION_NAMES
from eval.judge import call_judge, parse_judge_response, extract_scores, build_judge_prompt

ABLATION_PATH = PROJECT / "eval_v9" / "results" / "ablation_responses.jsonl"
JUDGMENTS_PATH = PROJECT / "eval_v9" / "results" / "ablation_judgments.jsonl"

# Pairwise comparisons to judge
PAIRS = [
    ("full", "raw"),
    ("full", "random"),
    ("full", "oracle"),
    ("oracle", "raw"),
    ("oracle", "random"),
    ("raw", "random"),
]

N_CONSISTENCY_RUNS = 3
API_DELAY = 1.0


def judge_pair_responses(question, response_a, response_b, domain):
    """Judge a pair with position debiasing and self-consistency."""
    results = {"ab": [], "ba": []}

    # AB ordering
    prompt_ab = build_judge_prompt(question, response_a, response_b, domain)
    for _ in range(N_CONSISTENCY_RUNS):
        try:
            raw = call_judge(prompt_ab)
            parsed = parse_judge_response(raw)
            if parsed:
                results["ab"].append({
                    "a_scores": extract_scores(parsed, "response_a"),
                    "b_scores": extract_scores(parsed, "response_b"),
                    "preference": parsed.get("preference", "TIE"),
                })
        except Exception as e:
            print(f"      [error] AB: {e}")
        time.sleep(API_DELAY)

    # BA ordering
    prompt_ba = build_judge_prompt(question, response_b, response_a, domain)
    for _ in range(N_CONSISTENCY_RUNS):
        try:
            raw = call_judge(prompt_ba)
            parsed = parse_judge_response(raw)
            if parsed:
                results["ba"].append({
                    "a_scores": extract_scores(parsed, "response_a"),
                    "b_scores": extract_scores(parsed, "response_b"),
                    "preference": parsed.get("preference", "TIE"),
                })
        except Exception as e:
            print(f"      [error] BA: {e}")
        time.sleep(API_DELAY)

    # Aggregate: in AB, A=response_a, B=response_b
    # In BA, A=response_b, B=response_a
    a_wins_ab = sum(1 for r in results["ab"] if r["preference"] == "A")
    a_wins_ba = sum(1 for r in results["ba"] if r["preference"] == "B")  # B in BA = A original

    b_wins_ab = sum(1 for r in results["ab"] if r["preference"] == "B")
    b_wins_ba = sum(1 for r in results["ba"] if r["preference"] == "A")  # A in BA = B original

    total = len(results["ab"]) + len(results["ba"])
    a_total = a_wins_ab + a_wins_ba
    b_total = b_wins_ab + b_wins_ba

    if a_total > b_total:
        winner = "A"
    elif b_total > a_total:
        winner = "B"
    else:
        winner = "TIE"

    return {
        "winner": winner,
        "a_wins": a_total,
        "b_wins": b_total,
        "ties": total - a_total - b_total,
        "total_runs": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    if not ABLATION_PATH.exists():
        print(f"ERROR: No ablation responses at {ABLATION_PATH}")
        print("Run: python3 eval_v9/ablation.py")
        sys.exit(1)

    # Load ablation responses
    records = []
    with open(ABLATION_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    # Resume support
    done_keys = set()
    if args.resume and JUDGMENTS_PATH.exists():
        with open(JUDGMENTS_PATH) as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    done_keys.add(f"{j['id']}_{j['pair']}")

    total_comparisons = len(records) * len(PAIRS)
    remaining = sum(1 for r in records for p in PAIRS
                    if f"{r['id']}_{p[0]}_vs_{p[1]}" not in done_keys)

    print(f"Ablation judging: {remaining} comparisons remaining")
    print(f"  {len(records)} questions × {len(PAIRS)} pairs = {total_comparisons} total")
    print(f"  API calls: ~{remaining * 2 * N_CONSISTENCY_RUNS}")
    print(f"  Output: {JUDGMENTS_PATH}\n")

    mode = "a" if args.resume and done_keys else "w"
    JUDGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(JUDGMENTS_PATH, mode) as f:
        for i, rec in enumerate(records):
            for pair in PAIRS:
                cond_a, cond_b = pair
                pair_key = f"{rec['id']}_{cond_a}_vs_{cond_b}"
                if pair_key in done_keys:
                    continue

                resp_a = rec[f"{cond_a}_response"]
                resp_b = rec[f"{cond_b}_response"]

                print(f"  [{i+1}/{len(records)}] {rec['id']} {cond_a} vs {cond_b}", end="", flush=True)

                result = judge_pair_responses(
                    rec["question"], resp_a, resp_b, rec.get("domain", "")
                )

                judgment = {
                    "id": rec["id"],
                    "pair": f"{cond_a}_vs_{cond_b}",
                    "expected_signal": rec["expected_signal"],
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    **result,
                    "timestamp": datetime.now().isoformat(),
                }

                f.write(json.dumps(judgment) + "\n")
                f.flush()

                winner_label = {"A": cond_a, "B": cond_b, "TIE": "tie"}[result["winner"]]
                print(f" → {winner_label} ({result['a_wins']}/{result['b_wins']}/{result['ties']})")

    # Summary
    all_judgments = []
    with open(JUDGMENTS_PATH) as f:
        for line in f:
            if line.strip():
                all_judgments.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"  ABLATION JUDGING COMPLETE")
    print(f"{'='*60}")

    for pair in PAIRS:
        pair_str = f"{pair[0]}_vs_{pair[1]}"
        pair_judgments = [j for j in all_judgments if j["pair"] == pair_str]
        a_wins = sum(1 for j in pair_judgments if j["winner"] == "A")
        b_wins = sum(1 for j in pair_judgments if j["winner"] == "B")
        ties = sum(1 for j in pair_judgments if j["winner"] == "TIE")
        total = len(pair_judgments)
        if total > 0:
            print(f"  {pair[0]:>8} vs {pair[1]:<8}: "
                  f"{pair[0]} wins {a_wins}/{total} ({100*a_wins/total:.0f}%), "
                  f"{pair[1]} wins {b_wins}/{total} ({100*b_wins/total:.0f}%), "
                  f"ties {ties}/{total}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
