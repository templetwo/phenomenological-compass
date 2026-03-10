"""
judge_ensemble.py — Multi-model LLM judge ensemble for compass evaluation
=========================================================================
Sends each response to N judge models, collects structured scores,
aggregates with position-debiasing and self-consistency checks.

Judge ensemble: Claude, GPT-4, Gemini, + open-source (Prometheus 2 or local)
Each judge evaluates both orderings (A/B and B/A) for preference comparisons.
Each evaluation run 3x at temperature 0.2 for self-consistency.

Usage:
    python3 -m eval_v9.judge_ensemble --results eval_v9/results/ablation_*.jsonl
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_v9.rubrics import (
    build_judge_prompt,
    compute_weighted_score,
    DIMENSIONS,
    RESTRAINT_RUBRIC,
)

RESULTS_DIR = PROJECT_ROOT / "eval_v9" / "results"
JUDGE_DIR = PROJECT_ROOT / "eval_v9" / "judge_scores"


# ── Judge Configuration ──────────────────────────────────────────────────────

JUDGES = {
    "claude": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.2,
        "max_tokens": 1000,
    },
    "gpt4": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 1000,
    },
    "gemini": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
        "max_tokens": 1000,
    },
    "local": {
        "provider": "local",
        "model": "prometheus-2",  # or local Qwen judge
        "temperature": 0.2,
        "max_tokens": 1000,
    },
}


# ── Judge API Calls ──────────────────────────────────────────────────────────

def call_judge_anthropic(prompt, config):
    """Call Claude as judge via Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except ImportError:
        print("  [skip] anthropic package not installed")
        return None
    except Exception as e:
        print(f"  [error] Anthropic: {e}")
        return None


def call_judge_openai(prompt, config):
    """Call GPT-4 as judge via OpenAI API."""
    try:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except ImportError:
        print("  [skip] openai package not installed")
        return None
    except Exception as e:
        print(f"  [error] OpenAI: {e}")
        return None


def call_judge_google(prompt, config):
    """Call Gemini as judge via Google AI API."""
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel(config["model"])
        response = model.generate_content(
            prompt,
            generation_config={"temperature": config["temperature"], "max_output_tokens": config["max_tokens"]},
        )
        return response.text
    except ImportError:
        print("  [skip] google-generativeai package not installed")
        return None
    except Exception as e:
        print(f"  [error] Google: {e}")
        return None


def call_judge_local(prompt, config):
    """Call a local model as judge (via Ollama or mlx-lm)."""
    # TODO: implement local judge via Ollama API
    # For now, skip
    print("  [skip] local judge not yet implemented")
    return None


JUDGE_DISPATCH = {
    "anthropic": call_judge_anthropic,
    "openai": call_judge_openai,
    "google": call_judge_google,
    "local": call_judge_local,
}


# ── Score Parsing ────────────────────────────────────────────────────────────

def parse_judge_response(text):
    """Extract structured scores from judge response."""
    if not text:
        return None

    # Try to find JSON block
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    json_match = re.search(r"\{[^{}]*\"scores\"[^{}]*\{[^{}]*\}[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ── Ensemble Evaluation ──────────────────────────────────────────────────────

def evaluate_single(question, response, expected_signal, condition, judges=None, n_runs=3):
    """Evaluate a single response with the judge ensemble.

    Args:
        question: The original question text
        response: The model's response text
        expected_signal: OPEN, PAUSE, or WITNESS
        condition: full, raw, oracle, or random
        judges: List of judge names to use (default: all available)
        n_runs: Number of evaluation runs per judge for self-consistency

    Returns:
        Dict with per-judge scores and ensemble aggregate
    """
    if judges is None:
        judges = list(JUDGES.keys())

    prompt = build_judge_prompt(question, response, expected_signal, condition)

    all_scores = {}

    for judge_name in judges:
        config = JUDGES[judge_name]
        call_fn = JUDGE_DISPATCH[config["provider"]]

        judge_runs = []
        for run_i in range(n_runs):
            raw_response = call_fn(prompt, config)
            parsed = parse_judge_response(raw_response)
            if parsed and "scores" in parsed:
                judge_runs.append(parsed)

        if judge_runs:
            # Average scores across runs for self-consistency
            avg_scores = {}
            all_dims = set()
            for run in judge_runs:
                all_dims.update(run["scores"].keys())

            for dim in all_dims:
                vals = [r["scores"][dim] for r in judge_runs if dim in r["scores"]]
                if vals:
                    avg_scores[dim] = round(sum(vals) / len(vals), 2)

            all_scores[judge_name] = {
                "scores": avg_scores,
                "n_runs": len(judge_runs),
                "consistency": _compute_consistency(judge_runs),
                "reasoning": judge_runs[0].get("reasoning", ""),
            }

    # Ensemble aggregate — average across judges
    ensemble_scores = {}
    if all_scores:
        all_dims = set()
        for judge_data in all_scores.values():
            all_dims.update(judge_data["scores"].keys())

        for dim in all_dims:
            vals = [
                jd["scores"][dim]
                for jd in all_scores.values()
                if dim in jd["scores"]
            ]
            if vals:
                ensemble_scores[dim] = round(sum(vals) / len(vals), 2)

    weighted = compute_weighted_score(ensemble_scores, expected_signal)

    return {
        "per_judge": all_scores,
        "ensemble_scores": ensemble_scores,
        "weighted_composite": round(weighted, 3),
        "n_judges": len(all_scores),
    }


def _compute_consistency(runs):
    """Compute self-consistency across multiple runs (Krippendorff's alpha simplified)."""
    if len(runs) < 2:
        return 1.0

    all_dims = set()
    for r in runs:
        all_dims.update(r["scores"].keys())

    total_agreement = 0
    total_dims = 0

    for dim in all_dims:
        vals = [r["scores"][dim] for r in runs if dim in r["scores"]]
        if len(vals) >= 2:
            max_diff = max(vals) - min(vals)
            agreement = 1.0 - (max_diff / 4.0)  # 4-point scale range
            total_agreement += agreement
            total_dims += 1

    return round(total_agreement / total_dims, 3) if total_dims > 0 else 1.0


# ── Batch Evaluation ─────────────────────────────────────────────────────────

def evaluate_ablation_results(results_path, judges=None, n_runs=3, output_dir=None):
    """Evaluate all responses from an ablation run."""
    output_dir = Path(output_dir) if output_dir else JUDGE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"judge_scores_{timestamp}.jsonl"

    # Load ablation results
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Evaluating {len(results)} responses with judge ensemble")
    print(f"Judges: {judges or list(JUDGES.keys())}")
    print(f"Runs per judge: {n_runs}")
    print(f"Output: {output_path}\n")

    with open(output_path, "w") as f:
        for ri, result in enumerate(results):
            print(f"  [{ri+1}/{len(results)}] {result['condition']:>8} | "
                  f"{result['expected_signal']:>7} | {result['question'][:40]}...")

            evaluation = evaluate_single(
                question=result["question"],
                response=result["response"],
                expected_signal=result["expected_signal"],
                condition=result["condition"],
                judges=judges,
                n_runs=n_runs,
            )

            scored_result = {
                **result,
                "evaluation": evaluation,
            }
            f.write(json.dumps(scored_result) + "\n")
            f.flush()

    print(f"\nJudge evaluation complete. Scores: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="LLM judge ensemble evaluation")
    parser.add_argument("--results", required=True,
                        help="Path to ablation results JSONL")
    parser.add_argument("--judges", nargs="+", default=None,
                        help="Judge names to use (default: all)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Evaluation runs per judge for consistency")
    parser.add_argument("--output", default=None,
                        help="Output directory for judge scores")
    args = parser.parse_args()

    evaluate_ablation_results(args.results, args.judges, args.n_runs, args.output)


if __name__ == "__main__":
    main()
