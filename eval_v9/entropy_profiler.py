"""
entropy_profiler.py — Token-level entropy analysis for compass evaluation
=========================================================================
Measures the entropy signature of compass-conditioned vs raw responses:
  - Mean token entropy (overall uncertainty)
  - Forking token analysis (where high-entropy decisions happen)
  - Lexical diversity (MTLD, Distinct-n)
  - Jensen-Shannon Divergence between condition distributions

The hypothesis: compass-routed responses should show STRUCTURED entropy —
high at meaningful semantic choice points, low at syntactic/formulaic tokens.
Raw responses should show either flat entropy (uncertain hedging) or
uniformly low entropy (template-following).

Usage:
    python3 -m eval_v9.entropy_profiler --results eval_v9/results/ablation_*.jsonl
"""

import json
import math
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Lexical Diversity Metrics ────────────────────────────────────────────────

def distinct_n(text, n=2):
    """Compute Distinct-n: ratio of unique n-grams to total n-grams.

    Higher = more diverse, less templated language.
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def mtld(text, threshold=0.72):
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010).

    Length-resistant measure of vocabulary richness. Computes the mean
    length of sequential word runs that maintain a TTR above threshold.

    Higher MTLD = richer vocabulary.
    """
    tokens = text.lower().split()
    if len(tokens) < 10:
        return 0.0

    def _mtld_forward(tokens, threshold):
        factors = 0
        current_start = 0
        for i in range(1, len(tokens) + 1):
            segment = tokens[current_start:i]
            ttr = len(set(segment)) / len(segment)
            if ttr <= threshold:
                factors += 1
                current_start = i
        # Partial factor for remaining tokens
        if current_start < len(tokens):
            remaining = tokens[current_start:]
            ttr = len(set(remaining)) / len(remaining)
            if ttr < 1.0:
                factors += (1.0 - ttr) / (1.0 - threshold)
        return len(tokens) / factors if factors > 0 else len(tokens)

    # Average forward and backward passes
    forward = _mtld_forward(tokens, threshold)
    backward = _mtld_forward(tokens[::-1], threshold)
    return (forward + backward) / 2


def type_token_ratio(text):
    """Simple type-token ratio."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ── Token Entropy (from logits) ──────────────────────────────────────────────

def compute_token_entropy_from_logits(logits):
    """Compute per-token entropy from logit arrays.

    Args:
        logits: List of arrays, each shape (vocab_size,) — one per generated token

    Returns:
        List of entropy values, one per token
    """
    entropies = []
    for token_logits in logits:
        # Softmax
        logits_shifted = token_logits - np.max(token_logits)
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
        # Entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        entropies.append(float(entropy))
    return entropies


def classify_forking_tokens(entropies, high_threshold=2.0, low_threshold=0.1):
    """Classify tokens into forking (high entropy) and determined (low entropy).

    Forking tokens are where the model makes meaningful semantic decisions.
    Determined tokens are syntactically forced.
    """
    high = sum(1 for e in entropies if e >= high_threshold)
    low = sum(1 for e in entropies if e <= low_threshold)
    mid = len(entropies) - high - low

    return {
        "n_tokens": len(entropies),
        "n_forking": high,
        "n_determined": low,
        "n_mid": mid,
        "forking_ratio": high / len(entropies) if entropies else 0,
        "determined_ratio": low / len(entropies) if entropies else 0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0,
        "std_entropy": float(np.std(entropies)) if entropies else 0,
        "max_entropy": float(np.max(entropies)) if entropies else 0,
        "entropy_at_forking": float(np.mean([e for e in entropies if e >= high_threshold])) if high > 0 else 0,
    }


# ── Jensen-Shannon Divergence ────────────────────────────────────────────────

def jensen_shannon_divergence(p, q, n_bins=50):
    """Compute JSD between two entropy distributions.

    JSD is symmetric and bounded [0, 1] (using log base 2).
    """
    # Histogram both distributions with same bins
    all_vals = np.concatenate([p, q])
    bin_edges = np.linspace(np.min(all_vals), np.max(all_vals), n_bins + 1)

    p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

    # Normalize to proper distributions
    p_hist = p_hist / (np.sum(p_hist) + 1e-12)
    q_hist = q_hist / (np.sum(q_hist) + 1e-12)

    # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
    m = 0.5 * (p_hist + q_hist)

    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return float(0.5 * kl(p_hist, m) + 0.5 * kl(q_hist, m))


# ── Text-Level Analysis (no logits needed) ───────────────────────────────────

def analyze_response_text(text):
    """Compute text-level metrics that don't require logits.

    These are always available — they work on the response text alone.
    """
    return {
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "mtld": mtld(text),
        "ttr": type_token_ratio(text),
        "word_count": len(text.split()),
        "unique_words": len(set(text.lower().split())),
    }


# ── Batch Analysis ───────────────────────────────────────────────────────────

def analyze_ablation_results(results_path, output_path=None):
    """Analyze entropy/diversity metrics across ablation conditions."""
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Analyzing {len(results)} responses from {results_path}")

    # Compute text metrics for each response
    by_condition = defaultdict(list)
    by_signal_condition = defaultdict(lambda: defaultdict(list))

    for r in results:
        metrics = analyze_response_text(r["response"])
        r["text_metrics"] = metrics
        by_condition[r["condition"]].append(metrics)
        by_signal_condition[r["expected_signal"]][r["condition"]].append(metrics)

    # Aggregate by condition
    print(f"\n{'=' * 70}")
    print("  LEXICAL DIVERSITY BY CONDITION")
    print(f"{'=' * 70}")
    print(f"  {'Condition':>10}  {'MTLD':>7}  {'D-1':>6}  {'D-2':>6}  {'D-3':>6}  {'TTR':>6}  {'Words':>6}")
    print(f"  {'-' * 55}")

    condition_summaries = {}
    for cond in ["full", "raw", "oracle", "random"]:
        if cond not in by_condition:
            continue
        metrics_list = by_condition[cond]
        summary = {
            k: float(np.mean([m[k] for m in metrics_list]))
            for k in metrics_list[0].keys()
        }
        condition_summaries[cond] = summary
        print(f"  {cond:>10}  {summary['mtld']:7.1f}  {summary['distinct_1']:.4f}  "
              f"{summary['distinct_2']:.4f}  {summary['distinct_3']:.4f}  "
              f"{summary['ttr']:.4f}  {summary['word_count']:6.0f}")

    # Signal-stratified
    print(f"\n{'=' * 70}")
    print("  LEXICAL DIVERSITY BY SIGNAL × CONDITION")
    print(f"{'=' * 70}")

    for signal in ["OPEN", "PAUSE", "WITNESS"]:
        if signal not in by_signal_condition:
            continue
        print(f"\n  {signal}:")
        print(f"  {'Condition':>10}  {'MTLD':>7}  {'D-2':>6}  {'Words':>6}")
        for cond in ["full", "raw", "oracle", "random"]:
            if cond not in by_signal_condition[signal]:
                continue
            metrics_list = by_signal_condition[signal][cond]
            mtld_mean = float(np.mean([m["mtld"] for m in metrics_list]))
            d2_mean = float(np.mean([m["distinct_2"] for m in metrics_list]))
            words_mean = float(np.mean([m["word_count"] for m in metrics_list]))
            print(f"  {cond:>10}  {mtld_mean:7.1f}  {d2_mean:.4f}  {words_mean:6.0f}")

    # Statistical comparison: full vs raw
    if "full" in by_condition and "raw" in by_condition:
        print(f"\n{'=' * 70}")
        print("  FULL vs RAW: DIVERSITY DIFFERENCES")
        print(f"{'=' * 70}")

        for metric_name in ["mtld", "distinct_2", "distinct_3"]:
            full_vals = np.array([m[metric_name] for m in by_condition["full"]])
            raw_vals = np.array([m[metric_name] for m in by_condition["raw"]])
            diff = np.mean(full_vals) - np.mean(raw_vals)
            # Simple t-test approximation
            se = np.sqrt(np.var(full_vals)/len(full_vals) + np.var(raw_vals)/len(raw_vals))
            t_stat = diff / se if se > 0 else 0
            print(f"  {metric_name:>12}: full={np.mean(full_vals):.4f}  raw={np.mean(raw_vals):.4f}  "
                  f"Δ={diff:+.4f}  t={t_stat:.2f}")

    # Save enriched results
    if output_path:
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nEnriched results saved to {output_path}")

    return condition_summaries


def main():
    parser = argparse.ArgumentParser(description="Entropy and diversity profiling")
    parser.add_argument("--results", required=True, help="Ablation results JSONL")
    parser.add_argument("--output", default=None, help="Save enriched results")
    args = parser.parse_args()

    analyze_ablation_results(args.results, args.output)


if __name__ == "__main__":
    main()
