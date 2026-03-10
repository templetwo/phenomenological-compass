#!/usr/bin/env python3
"""
judge.py — LLM-as-judge scoring with position debiasing and self-consistency
=============================================================================
For each question, presents BOTH responses (compass-routed and raw) in
randomized order (A/B). The judge does NOT know which is which.
Evaluates on 6 dimensions from rubrics.py.

Debiasing: runs each comparison TWICE with A/B order swapped.
           Only consistent results count as wins.
Self-consistency: 3 iterations at temperature 0.2, majority vote.

Uses Claude Sonnet via Anthropic API.

Usage:
    source ~/phenomenological-compass/.venv/bin/activate
    ANTHROPIC_API_KEY=... python3 eval/judge.py

    # Test with first N
    ANTHROPIC_API_KEY=... python3 eval/judge.py --limit 5

    # Resume from partial run
    ANTHROPIC_API_KEY=... python3 eval/judge.py --resume
"""

import json
import os
import sys
import time
import random
import argparse
import re
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from eval.rubrics import RUBRIC_TEXT, DIMENSION_NAMES

RESPONSES_PATH = PROJECT / "eval" / "results" / "responses.jsonl"
JUDGMENTS_PATH = PROJECT / "eval" / "results" / "judgments.jsonl"

JUDGE_MODEL = "claude-sonnet-4-20250514"
JUDGE_TEMP = 0.2
JUDGE_MAX_TOKENS = 2000
N_CONSISTENCY_RUNS = 3
API_DELAY = 1.0  # seconds between API calls


def build_judge_prompt(question, response_a, response_b, domain):
    """Build the evaluation prompt for the judge."""
    return f"""You are evaluating two AI responses to the same question. Score each response on
six dimensions using the rubrics below. Be precise and discriminating — use the
full 1-5 scale.

QUESTION: {question}
QUESTION TYPE: {domain}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

RUBRICS:
{RUBRIC_TEXT}

For each response, provide:
1. Scores for all 6 dimensions (1-5)
2. A brief justification for each score (1-2 sentences)
3. Overall preference: A, B, or TIE
4. Confidence in preference: HIGH, MEDIUM, LOW

Respond in JSON only — no markdown fences, no preamble:
{{
  "response_a": {{
    "epistemic_appropriateness": {{"score": N, "reason": "..."}},
    "emotional_attunement": {{"score": N, "reason": "..."}},
    "depth_of_exploration": {{"score": N, "reason": "..."}},
    "restraint_quality": {{"score": N, "reason": "..."}},
    "intellectual_rigor": {{"score": N, "reason": "..."}},
    "authenticity": {{"score": N, "reason": "..."}}
  }},
  "response_b": {{
    "epistemic_appropriateness": {{"score": N, "reason": "..."}},
    "emotional_attunement": {{"score": N, "reason": "..."}},
    "depth_of_exploration": {{"score": N, "reason": "..."}},
    "restraint_quality": {{"score": N, "reason": "..."}},
    "intellectual_rigor": {{"score": N, "reason": "..."}},
    "authenticity": {{"score": N, "reason": "..."}}
  }},
  "preference": "A or B or TIE",
  "confidence": "HIGH or MEDIUM or LOW",
  "reasoning": "..."
}}"""


def call_judge(prompt):
    """Call Claude Sonnet as judge."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMP,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def parse_judge_response(text):
    """Extract structured scores from judge response."""
    # Try to find JSON in the response
    # Strip markdown code fences if present
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def extract_scores(parsed, key):
    """Extract dimension scores from parsed judge response."""
    if not parsed or key not in parsed:
        return None
    scores = {}
    for dim in DIMENSION_NAMES:
        dim_data = parsed[key].get(dim, {})
        if isinstance(dim_data, dict):
            scores[dim] = dim_data.get("score", 0)
        elif isinstance(dim_data, (int, float)):
            scores[dim] = dim_data
    return scores if all(dim in scores for dim in DIMENSION_NAMES) else None


def judge_pair(question, routed_response, raw_response, domain):
    """Judge a pair of responses with position debiasing.

    Runs twice: once with routed=A,raw=B; once with routed=B,raw=A.
    Runs N_CONSISTENCY_RUNS times each for self-consistency.
    """
    results = {"ab": [], "ba": []}

    # Ordering 1: routed=A, raw=B
    prompt_ab = build_judge_prompt(question, routed_response, raw_response, domain)
    for run_i in range(N_CONSISTENCY_RUNS):
        try:
            raw_text = call_judge(prompt_ab)
            parsed = parse_judge_response(raw_text)
            if parsed:
                results["ab"].append({
                    "parsed": parsed,
                    "routed_scores": extract_scores(parsed, "response_a"),
                    "raw_scores": extract_scores(parsed, "response_b"),
                    "preference_raw": parsed.get("preference", "TIE"),
                    # Map back: if judge prefers A, routed wins
                    "routed_preferred": parsed.get("preference") == "A",
                    "raw_preferred": parsed.get("preference") == "B",
                    "confidence": parsed.get("confidence", "LOW"),
                })
        except Exception as e:
            print(f"    [error] AB run {run_i}: {e}")
        time.sleep(API_DELAY)

    # Ordering 2: raw=A, routed=B
    prompt_ba = build_judge_prompt(question, raw_response, routed_response, domain)
    for run_i in range(N_CONSISTENCY_RUNS):
        try:
            raw_text = call_judge(prompt_ba)
            parsed = parse_judge_response(raw_text)
            if parsed:
                results["ba"].append({
                    "parsed": parsed,
                    "raw_scores": extract_scores(parsed, "response_a"),
                    "routed_scores": extract_scores(parsed, "response_b"),
                    "preference_raw": parsed.get("preference", "TIE"),
                    # Map back: if judge prefers B, routed wins (B is routed in this ordering)
                    "routed_preferred": parsed.get("preference") == "B",
                    "raw_preferred": parsed.get("preference") == "A",
                    "confidence": parsed.get("confidence", "LOW"),
                })
        except Exception as e:
            print(f"    [error] BA run {run_i}: {e}")
        time.sleep(API_DELAY)

    return aggregate_judgments(results)


def aggregate_judgments(results):
    """Aggregate across runs and orderings."""
    all_routed_scores = []
    all_raw_scores = []
    preferences = []

    for ordering in ["ab", "ba"]:
        for run in results[ordering]:
            if run["routed_scores"]:
                all_routed_scores.append(run["routed_scores"])
            if run["raw_scores"]:
                all_raw_scores.append(run["raw_scores"])
            if run["routed_preferred"]:
                preferences.append("routed")
            elif run["raw_preferred"]:
                preferences.append("raw")
            else:
                preferences.append("tie")

    # Average scores across all runs
    def avg_scores(score_list):
        if not score_list:
            return {}
        avg = {}
        for dim in DIMENSION_NAMES:
            vals = [s[dim] for s in score_list if dim in s]
            avg[dim] = round(sum(vals) / len(vals), 2) if vals else 0
        return avg

    routed_avg = avg_scores(all_routed_scores)
    raw_avg = avg_scores(all_raw_scores)

    # Position-debiased preference: majority across AB and BA orderings
    # Only count as win if BOTH orderings agree
    ab_prefs = [r["routed_preferred"] for r in results["ab"]]
    ba_prefs = [r["routed_preferred"] for r in results["ba"]]

    ab_majority = sum(ab_prefs) > len(ab_prefs) / 2 if ab_prefs else False
    ba_majority = sum(ba_prefs) > len(ba_prefs) / 2 if ba_prefs else False

    ab_raw_majority = sum(1 for r in results["ab"] if r["raw_preferred"]) > len(results["ab"]) / 2 if results["ab"] else False
    ba_raw_majority = sum(1 for r in results["ba"] if r["raw_preferred"]) > len(results["ba"]) / 2 if results["ba"] else False

    if ab_majority and ba_majority:
        debiased_preference = "routed"
    elif ab_raw_majority and ba_raw_majority:
        debiased_preference = "raw"
    else:
        debiased_preference = "tie"

    return {
        "routed_scores": routed_avg,
        "raw_scores": raw_avg,
        "debiased_preference": debiased_preference,
        "raw_preferences": preferences,
        "n_runs": len(preferences),
        "n_ab": len(results["ab"]),
        "n_ba": len(results["ba"]),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Verify anthropic package
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    # Load responses
    if not RESPONSES_PATH.exists():
        print(f"ERROR: No responses found at {RESPONSES_PATH}")
        print("Run: python3 eval/run_eval.py")
        sys.exit(1)

    responses = []
    with open(RESPONSES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(json.loads(line))

    if args.limit:
        responses = responses[:args.limit]

    # Resume support
    done_ids = set()
    if args.resume and JUDGMENTS_PATH.exists():
        with open(JUDGMENTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        j = json.loads(line)
                        done_ids.add(j["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Resuming — {len(done_ids)} already judged")

    remaining = [r for r in responses if r["id"] not in done_ids]

    print(f"Judging {len(remaining)} questions")
    print(f"  Model: {JUDGE_MODEL}")
    print(f"  Temperature: {JUDGE_TEMP}")
    print(f"  Consistency runs: {N_CONSISTENCY_RUNS}")
    print(f"  Debiasing: dual ordering (AB + BA)")
    print(f"  API calls: ~{len(remaining) * 2 * N_CONSISTENCY_RUNS}")
    print(f"  Estimated time: ~{len(remaining) * 2 * N_CONSISTENCY_RUNS * 3 / 60:.0f} min")
    print(f"  Output: {JUDGMENTS_PATH}\n")

    mode = "a" if args.resume and done_ids else "w"

    with open(JUDGMENTS_PATH, mode) as f:
        for i, resp in enumerate(remaining):
            print(f"  [{i+1}/{len(remaining)}] {resp['expected_signal']} {resp['id']} — {resp['question'][:50]}...")

            judgment = judge_pair(
                question=resp["question"],
                routed_response=resp["routed_response"],
                raw_response=resp["raw_response"],
                domain=resp.get("domain", ""),
            )

            record = {
                "id": resp["id"],
                "question": resp["question"],
                "expected_signal": resp["expected_signal"],
                "domain": resp.get("domain", ""),
                "compass_signal": resp["compass_signal"],
                "signal_correct": resp["signal_correct"],
                **judgment,
                "timestamp": datetime.now().isoformat(),
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            pref = judgment["debiased_preference"]
            marker = {"routed": "+", "raw": "-", "tie": "="}[pref]
            print(f"           [{marker}] {pref}  "
                  f"routed={sum(judgment['routed_scores'].values())/6:.1f}  "
                  f"raw={sum(judgment['raw_scores'].values())/6:.1f}")

    # Quick summary
    all_judgments = []
    with open(JUDGMENTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                all_judgments.append(json.loads(line))

    wins = sum(1 for j in all_judgments if j["debiased_preference"] == "routed")
    losses = sum(1 for j in all_judgments if j["debiased_preference"] == "raw")
    ties = sum(1 for j in all_judgments if j["debiased_preference"] == "tie")
    total = len(all_judgments)

    print(f"\n{'=' * 60}")
    print(f"  JUDGING COMPLETE")
    print(f"  Compass wins: {wins}/{total} ({100*wins/total:.0f}%)")
    print(f"  Raw wins:     {losses}/{total} ({100*losses/total:.0f}%)")
    print(f"  Ties:         {ties}/{total} ({100*ties/total:.0f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
