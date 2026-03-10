"""
statistics.py — Paired-difference statistics for compass evaluation
===================================================================
Implements Anthropic's evaluation framework:
  - Paired-difference analysis (per-question score differences)
  - Bootstrap confidence intervals
  - Permutation tests
  - Cohen's d effect sizes
  - Win/loss/tie rates with magnitude analysis

Usage:
    python3 -m eval_v9.statistics --scores eval_v9/judge_scores/judge_scores_*.jsonl
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_scored_results(path):
    """Load judge-scored results from JSONL."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def group_by_question(results):
    """Group results by question, then by condition."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        q = r["question"]
        c = r["condition"]
        grouped[q][c].append(r)
    return grouped


# ── Paired-Difference Analysis ───────────────────────────────────────────────

def paired_differences(grouped, condition_a="full", condition_b="raw", metric="weighted_composite"):
    """Compute per-question score differences between two conditions.

    Returns array of (mean_a - mean_b) for each question where both conditions exist.
    """
    diffs = []
    questions = []

    for question, conditions in grouped.items():
        if condition_a in conditions and condition_b in conditions:
            scores_a = [
                r["evaluation"][metric]
                for r in conditions[condition_a]
                if "evaluation" in r and metric in r["evaluation"]
            ]
            scores_b = [
                r["evaluation"][metric]
                for r in conditions[condition_b]
                if "evaluation" in r and metric in r["evaluation"]
            ]

            if scores_a and scores_b:
                mean_a = np.mean(scores_a)
                mean_b = np.mean(scores_b)
                diffs.append(mean_a - mean_b)
                questions.append(question)

    return np.array(diffs), questions


def cohens_d(diffs):
    """Compute Cohen's d effect size from paired differences."""
    if len(diffs) < 2:
        return 0.0
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def bootstrap_ci(diffs, n_bootstrap=10000, confidence=0.95):
    """Bootstrap confidence interval for the mean difference."""
    if len(diffs) < 2:
        return (0.0, 0.0)

    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return (lower, upper)


def permutation_test(diffs, n_permutations=10000):
    """Permutation test for whether mean difference != 0.

    Under the null hypothesis, each difference is equally likely to be
    positive or negative. We randomly flip signs and compute the
    proportion of permuted means as extreme as the observed mean.
    """
    if len(diffs) < 2:
        return 1.0

    observed_mean = np.mean(diffs)
    rng = np.random.default_rng(42)

    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_mean = np.mean(diffs * signs)
        if abs(perm_mean) >= abs(observed_mean):
            count_extreme += 1

    return float(count_extreme / n_permutations)


def win_loss_tie(diffs, threshold=0.1):
    """Compute win/loss/tie rates with magnitude analysis."""
    wins = diffs > threshold
    losses = diffs < -threshold
    ties = ~wins & ~losses

    result = {
        "wins": int(np.sum(wins)),
        "losses": int(np.sum(losses)),
        "ties": int(np.sum(ties)),
        "total": len(diffs),
        "win_rate": float(np.mean(wins)),
        "loss_rate": float(np.mean(losses)),
        "tie_rate": float(np.mean(ties)),
    }

    if np.sum(wins) > 0:
        result["mean_win_magnitude"] = float(np.mean(diffs[wins]))
    if np.sum(losses) > 0:
        result["mean_loss_magnitude"] = float(np.mean(diffs[losses]))

    return result


# ── Signal-Stratified Analysis ───────────────────────────────────────────────

def stratified_analysis(results, condition_a="full", condition_b="raw", metric="weighted_composite"):
    """Run paired analysis stratified by expected signal."""
    # Group by signal
    by_signal = defaultdict(list)
    for r in results:
        sig = r.get("expected_signal", "UNKNOWN")
        by_signal[sig].append(r)

    report = {}
    for signal, signal_results in by_signal.items():
        grouped = group_by_question(signal_results)
        diffs, questions = paired_differences(grouped, condition_a, condition_b, metric)

        if len(diffs) == 0:
            continue

        report[signal] = {
            "n_questions": len(diffs),
            "mean_difference": float(np.mean(diffs)),
            "std_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0,
            "se_mean": float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0,
            "cohens_d": cohens_d(diffs),
            "bootstrap_95ci": bootstrap_ci(diffs),
            "permutation_p": permutation_test(diffs),
            "win_loss_tie": win_loss_tie(diffs),
        }

    return report


# ── Dimension-Level Analysis ─────────────────────────────────────────────────

def dimension_analysis(results, condition_a="full", condition_b="raw"):
    """Analyze improvement per scoring dimension."""
    dimensions = [
        "epistemic_appropriateness", "emotional_attunement", "philosophical_depth",
        "signal_calibration", "factual_accuracy", "helpfulness", "restraint_quality",
    ]

    grouped = group_by_question(results)
    report = {}

    for dim in dimensions:
        diffs = []
        for question, conditions in grouped.items():
            if condition_a in conditions and condition_b in conditions:
                scores_a = [
                    r["evaluation"]["ensemble_scores"].get(dim)
                    for r in conditions[condition_a]
                    if "evaluation" in r and "ensemble_scores" in r["evaluation"]
                    and dim in r["evaluation"]["ensemble_scores"]
                ]
                scores_b = [
                    r["evaluation"]["ensemble_scores"].get(dim)
                    for r in conditions[condition_b]
                    if "evaluation" in r and "ensemble_scores" in r["evaluation"]
                    and dim in r["evaluation"]["ensemble_scores"]
                ]

                scores_a = [s for s in scores_a if s is not None]
                scores_b = [s for s in scores_b if s is not None]

                if scores_a and scores_b:
                    diffs.append(np.mean(scores_a) - np.mean(scores_b))

        if diffs:
            diffs = np.array(diffs)
            report[dim] = {
                "n_questions": len(diffs),
                "mean_improvement": float(np.mean(diffs)),
                "cohens_d": cohens_d(diffs),
                "bootstrap_95ci": bootstrap_ci(diffs),
                "p_value": permutation_test(diffs),
            }

    return report


# ── Full Report ──────────────────────────────────────────────────────────────

def generate_report(scored_path):
    """Generate the full statistical report."""
    results = load_scored_results(scored_path)
    grouped = group_by_question(results)

    print("=" * 70)
    print("  PHENOMENOLOGICAL COMPASS — STATISTICAL EVALUATION REPORT")
    print("=" * 70)

    # --- Full vs Raw (the main comparison) ---
    comparisons = [
        ("full", "raw", "Full Pipeline vs Raw (Main Effect)"),
        ("full", "random", "Full Pipeline vs Random Compass (Compass Utility)"),
        ("oracle", "full", "Oracle vs Full (Compass Accuracy Ceiling)"),
        ("oracle", "raw", "Oracle vs Raw (Maximum Possible Improvement)"),
    ]

    for ca, cb, label in comparisons:
        diffs, questions = paired_differences(grouped, ca, cb)
        if len(diffs) == 0:
            print(f"\n  {label}: No data")
            continue

        d = cohens_d(diffs)
        ci = bootstrap_ci(diffs)
        p = permutation_test(diffs)
        wlt = win_loss_tie(diffs)

        print(f"\n── {label} {'─' * max(0, 50 - len(label))}")
        print(f"  N questions:     {len(diffs)}")
        print(f"  Mean difference: {np.mean(diffs):+.3f} (SE: {np.std(diffs, ddof=1)/np.sqrt(len(diffs)):.3f})")
        print(f"  Cohen's d:       {d:.3f} ({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'})")
        print(f"  95% CI:          [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        print(f"  Permutation p:   {p:.4f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")
        print(f"  Wins/Losses/Ties:{wlt['wins']}/{wlt['losses']}/{wlt['ties']} "
              f"(win rate: {wlt['win_rate']:.1%})")

    # --- Signal-stratified ---
    print(f"\n{'=' * 70}")
    print("  SIGNAL-STRATIFIED ANALYSIS (Full vs Raw)")
    print(f"{'=' * 70}")

    strat = stratified_analysis(results)
    for signal, data in sorted(strat.items()):
        d = data["cohens_d"]
        print(f"\n  {signal} (n={data['n_questions']})")
        print(f"    Mean diff: {data['mean_difference']:+.3f}  Cohen's d: {d:.3f}  "
              f"p={data['permutation_p']:.4f}")
        ci = data["bootstrap_95ci"]
        print(f"    95% CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        wlt = data["win_loss_tie"]
        print(f"    W/L/T: {wlt['wins']}/{wlt['losses']}/{wlt['ties']}")

    # --- Dimension-level ---
    print(f"\n{'=' * 70}")
    print("  DIMENSION-LEVEL ANALYSIS (Full vs Raw)")
    print(f"{'=' * 70}")

    dims = dimension_analysis(results)
    for dim, data in sorted(dims.items(), key=lambda x: -abs(x[1]["mean_improvement"])):
        sig = "***" if data["p_value"] < 0.001 else "**" if data["p_value"] < 0.01 else "*" if data["p_value"] < 0.05 else "ns"
        print(f"  {dim:30s}  Δ={data['mean_improvement']:+.3f}  d={data['cohens_d']:.3f}  p={data['p_value']:.4f} {sig}")

    print(f"\n{'=' * 70}\n")

    return {
        "comparisons": {
            f"{ca}_vs_{cb}": {
                "n": len(paired_differences(grouped, ca, cb)[0]),
                "cohens_d": cohens_d(paired_differences(grouped, ca, cb)[0]),
            }
            for ca, cb, _ in comparisons
        },
        "stratified": strat,
        "dimensions": dims,
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of compass evaluation")
    parser.add_argument("--scores", required=True, help="Path to judge-scored JSONL")
    parser.add_argument("--output", default=None, help="Save report as JSON")
    args = parser.parse_args()

    report = generate_report(args.scores)

    if args.output:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=convert)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
