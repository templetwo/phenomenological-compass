#!/usr/bin/env python3
"""
analyze.py — Statistical analysis and report generation
========================================================
Reads judge scores, computes paired statistics, generates figures and report.

Usage:
    source ~/phenomenological-compass/.venv/bin/activate
    python3 eval/analyze.py
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

PROJECT = Path(__file__).parent.parent
JUDGMENTS_PATH = PROJECT / "eval" / "results" / "judgments.jsonl"
RESPONSES_PATH = PROJECT / "eval" / "results" / "responses.jsonl"
RESULTS_DIR = PROJECT / "eval" / "results"

from eval.rubrics import DIMENSION_NAMES


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ── Statistical Functions ────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d for paired samples."""
    diff = np.array(a) - np.array(b)
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def bootstrap_ci(data, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    data = np.array(data)
    if len(data) < 2:
        return (float(np.mean(data)), float(np.mean(data)))
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = 1 - ci
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def permutation_test(a, b, n_perm=10000):
    """Two-sided permutation test for paired difference."""
    diff = np.array(a) - np.array(b)
    if len(diff) < 2:
        return 1.0
    observed = np.mean(diff)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        if abs(np.mean(diff * signs)) >= abs(observed):
            count += 1
    return float(count / n_perm)


def wilcoxon_test(a, b):
    """Wilcoxon signed-rank test."""
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b, alternative='two-sided')
        return float(p)
    except ImportError:
        return permutation_test(a, b)
    except ValueError:
        return 1.0


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze(judgments, responses=None):
    """Full statistical analysis."""
    n = len(judgments)
    print(f"\n{'=' * 70}")
    print(f"  PHENOMENOLOGICAL COMPASS — EVALUATION REPORT")
    print(f"  {n} questions evaluated | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}")

    # ── Signal Classification Accuracy ──
    if responses:
        correct = sum(1 for r in responses if r.get("signal_correct", False))
        total = len(responses)
        print(f"\n  SIGNAL CLASSIFICATION ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")

        # Confusion matrix
        confusion = defaultdict(Counter)
        for r in responses:
            confusion[r["expected_signal"]][r["compass_signal"]] += 1

        print(f"\n  {'':>10}  {'OPEN':>6}  {'PAUSE':>6}  {'WITNESS':>8}")
        print(f"  {'─' * 36}")
        for expected in ["OPEN", "PAUSE", "WITNESS"]:
            row = confusion[expected]
            total_row = sum(row.values())
            acc = row[expected] / total_row if total_row else 0
            print(f"  {expected:>10}  {row.get('OPEN',0):>6}  {row.get('PAUSE',0):>6}  "
                  f"{row.get('WITNESS',0):>8}  ({acc:.0%})")

    # ── Win Rates ──
    wins = sum(1 for j in judgments if j["debiased_preference"] == "routed")
    losses = sum(1 for j in judgments if j["debiased_preference"] == "raw")
    ties = sum(1 for j in judgments if j["debiased_preference"] == "tie")

    win_rate = wins / n
    ci = bootstrap_ci([1 if j["debiased_preference"] == "routed" else 0 for j in judgments])

    print(f"\n  OVERALL WIN RATE: {wins}/{n} ({win_rate:.1%})")
    print(f"    95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
    print(f"    Raw wins: {losses}/{n} ({losses/n:.1%})")
    print(f"    Ties: {ties}/{n} ({ties/n:.1%})")

    # Per-signal win rates
    print(f"\n  {'Signal':>10}  {'Wins':>6}  {'Losses':>6}  {'Ties':>6}  {'Win%':>6}  {'95% CI':>14}")
    print(f"  {'─' * 56}")

    for signal in ["OPEN", "PAUSE", "WITNESS"]:
        sig_judgments = [j for j in judgments if j["expected_signal"] == signal]
        if not sig_judgments:
            continue
        sig_n = len(sig_judgments)
        sig_wins = sum(1 for j in sig_judgments if j["debiased_preference"] == "routed")
        sig_losses = sum(1 for j in sig_judgments if j["debiased_preference"] == "raw")
        sig_ties = sig_n - sig_wins - sig_losses
        sig_rate = sig_wins / sig_n
        sig_ci = bootstrap_ci([1 if j["debiased_preference"] == "routed" else 0 for j in sig_judgments])
        print(f"  {signal:>10}  {sig_wins:>6}  {sig_losses:>6}  {sig_ties:>6}  "
              f"{sig_rate:>5.0%}  [{sig_ci[0]:.0%}, {sig_ci[1]:.0%}]")

    # ── Dimensional Scores ──
    print(f"\n  {'Dimension':>28}  {'Routed':>7}  {'Raw':>7}  {'Delta':>7}  {'d':>6}  {'p':>8}")
    print(f"  {'─' * 70}")

    dim_results = {}
    for dim in DIMENSION_NAMES:
        routed_vals = [j["routed_scores"].get(dim, 0) for j in judgments if j.get("routed_scores")]
        raw_vals = [j["raw_scores"].get(dim, 0) for j in judgments if j.get("raw_scores")]

        if not routed_vals or not raw_vals:
            continue

        # Ensure paired
        min_len = min(len(routed_vals), len(raw_vals))
        routed_vals = routed_vals[:min_len]
        raw_vals = raw_vals[:min_len]

        r_mean = np.mean(routed_vals)
        b_mean = np.mean(raw_vals)
        delta = r_mean - b_mean
        d = cohens_d(routed_vals, raw_vals)
        p = permutation_test(routed_vals, raw_vals)

        sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        d_label = "L" if abs(d) >= 0.8 else "M" if abs(d) >= 0.5 else "S" if abs(d) >= 0.2 else ""

        print(f"  {dim:>28}  {r_mean:>7.2f}  {b_mean:>7.2f}  {delta:>+7.2f}  "
              f"{d:>5.2f}{d_label}  {p:>7.4f} {sig_marker}")

        dim_results[dim] = {
            "routed_mean": float(r_mean),
            "raw_mean": float(b_mean),
            "delta": float(delta),
            "cohens_d": float(d),
            "p_value": float(p),
        }

    # ── Per-Signal Dimensional Breakdown ──
    print(f"\n{'=' * 70}")
    print(f"  PER-SIGNAL DIMENSIONAL ANALYSIS")
    print(f"{'=' * 70}")

    for signal in ["OPEN", "PAUSE", "WITNESS"]:
        sig_judgments = [j for j in judgments if j["expected_signal"] == signal]
        if not sig_judgments:
            continue

        print(f"\n  {signal} (n={len(sig_judgments)})")
        print(f"  {'Dimension':>28}  {'Routed':>7}  {'Raw':>7}  {'Delta':>7}  {'d':>6}")
        print(f"  {'─' * 55}")

        for dim in DIMENSION_NAMES:
            routed = [j["routed_scores"].get(dim, 0) for j in sig_judgments if j.get("routed_scores")]
            raw = [j["raw_scores"].get(dim, 0) for j in sig_judgments if j.get("raw_scores")]
            if not routed or not raw:
                continue
            min_len = min(len(routed), len(raw))
            r_mean = np.mean(routed[:min_len])
            b_mean = np.mean(raw[:min_len])
            d = cohens_d(routed[:min_len], raw[:min_len])
            print(f"  {dim:>28}  {r_mean:>7.2f}  {b_mean:>7.2f}  {r_mean-b_mean:>+7.2f}  {d:>5.2f}")

    # ── Hypothesis Tests ──
    print(f"\n{'=' * 70}")
    print(f"  HYPOTHESIS TESTS")
    print(f"{'=' * 70}")

    # H1: Overall win rate > 50%
    h1_p = permutation_test(
        [1 if j["debiased_preference"] == "routed" else 0 for j in judgments],
        [0.5] * n,
    )
    print(f"\n  H1: Compass wins > 50%")
    print(f"      Win rate: {win_rate:.1%}, p={h1_p:.4f} {'SUPPORTED' if h1_p < 0.05 and win_rate > 0.5 else 'NOT SUPPORTED'}")

    # H2: WITNESS advantage is largest
    signal_deltas = {}
    for signal in ["OPEN", "PAUSE", "WITNESS"]:
        sig_j = [j for j in judgments if j["expected_signal"] == signal]
        if sig_j:
            routed_total = [sum(j["routed_scores"].values())/6 for j in sig_j if j.get("routed_scores")]
            raw_total = [sum(j["raw_scores"].values())/6 for j in sig_j if j.get("raw_scores")]
            min_len = min(len(routed_total), len(raw_total))
            if min_len:
                signal_deltas[signal] = float(np.mean(np.array(routed_total[:min_len]) - np.array(raw_total[:min_len])))

    if signal_deltas:
        max_signal = max(signal_deltas, key=signal_deltas.get)
        print(f"\n  H2: WITNESS advantage is largest")
        for sig, delta in sorted(signal_deltas.items(), key=lambda x: -x[1]):
            print(f"      {sig}: Δ={delta:+.3f}")
        print(f"      Largest: {max_signal} {'SUPPORTED' if max_signal == 'WITNESS' else 'NOT SUPPORTED'}")

    # H4: Signal accuracy > 80%
    if responses:
        sig_acc = correct / total
        print(f"\n  H4: Signal accuracy > 80%")
        print(f"      Accuracy: {sig_acc:.1%} {'SUPPORTED' if sig_acc > 0.8 else 'NOT SUPPORTED'}")

    return dim_results, signal_deltas


# ── Visualization ────────────────────────────────────────────────────────────

def generate_figures(judgments, responses=None):
    """Generate matplotlib figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [skip] matplotlib not installed — skipping figures")
        return

    # 1. Win rates by signal
    fig, ax = plt.subplots(figsize=(8, 5))
    signals = ["Overall", "OPEN", "PAUSE", "WITNESS"]
    win_rates = []
    ci_lower = []
    ci_upper = []

    for signal in signals:
        if signal == "Overall":
            j_set = judgments
        else:
            j_set = [j for j in judgments if j["expected_signal"] == signal]

        if not j_set:
            win_rates.append(0)
            ci_lower.append(0)
            ci_upper.append(0)
            continue

        data = [1 if j["debiased_preference"] == "routed" else 0 for j in j_set]
        wr = np.mean(data)
        ci = bootstrap_ci(data)
        win_rates.append(wr)
        ci_lower.append(wr - ci[0])
        ci_upper.append(ci[1] - wr)

    colors = ["#666666", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(signals, win_rates, color=colors, alpha=0.8,
                  yerr=[ci_lower, ci_upper], capsize=5)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% baseline")
    ax.set_ylabel("Win Rate (compass-routed preferred)")
    ax.set_title("Compass Win Rate by Signal Type")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "win_rates.png", dpi=150)
    plt.close()
    print(f"  → win_rates.png")

    # 2. Dimensional scores comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    dims = DIMENSION_NAMES
    dim_labels = [d.replace("_", "\n") for d in dims]
    x = np.arange(len(dims))
    width = 0.35

    routed_means = []
    raw_means = []
    for dim in dims:
        r = [j["routed_scores"].get(dim, 0) for j in judgments if j.get("routed_scores")]
        b = [j["raw_scores"].get(dim, 0) for j in judgments if j.get("raw_scores")]
        routed_means.append(np.mean(r) if r else 0)
        raw_means.append(np.mean(b) if b else 0)

    ax.bar(x - width/2, routed_means, width, label="Compass-Routed", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, raw_means, width, label="Raw Baseline", color="#FF5722", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=8)
    ax.set_ylabel("Mean Score (1-5)")
    ax.set_title("Dimensional Scores: Compass-Routed vs Raw")
    ax.set_ylim(0, 5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dimensional_scores.png", dpi=150)
    plt.close()
    print(f"  → dimensional_scores.png")

    # 3. Effect sizes forest plot
    fig, ax = plt.subplots(figsize=(8, 6))
    effect_sizes = []
    labels = []

    for signal in ["Overall", "OPEN", "PAUSE", "WITNESS"]:
        j_set = judgments if signal == "Overall" else [j for j in judgments if j["expected_signal"] == signal]
        for dim in dims:
            r = [j["routed_scores"].get(dim, 0) for j in j_set if j.get("routed_scores")]
            b = [j["raw_scores"].get(dim, 0) for j in j_set if j.get("raw_scores")]
            min_len = min(len(r), len(b))
            if min_len >= 2:
                d = cohens_d(r[:min_len], b[:min_len])
                effect_sizes.append(d)
                labels.append(f"{signal}\n{dim[:12]}")

    if effect_sizes:
        y_pos = np.arange(len(effect_sizes))
        colors_es = ["green" if d > 0.2 else "red" if d < -0.2 else "gray" for d in effect_sizes]
        ax.barh(y_pos, effect_sizes, color=colors_es, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.axvline(x=0.2, color="blue", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.axvline(x=0.5, color="blue", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.axvline(x=0.8, color="blue", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.set_xlabel("Cohen's d (positive = compass better)")
        ax.set_title("Effect Sizes by Signal × Dimension")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "effect_sizes.png", dpi=150)
        plt.close()
        print(f"  → effect_sizes.png")

    # 4. Signal confusion matrix
    if responses:
        fig, ax = plt.subplots(figsize=(6, 5))
        signals_list = ["OPEN", "PAUSE", "WITNESS"]
        matrix = np.zeros((3, 3))
        for r in responses:
            exp_idx = signals_list.index(r["expected_signal"]) if r["expected_signal"] in signals_list else -1
            got_idx = signals_list.index(r["compass_signal"]) if r["compass_signal"] in signals_list else -1
            if exp_idx >= 0 and got_idx >= 0:
                matrix[exp_idx][got_idx] += 1

        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_xticklabels(signals_list)
        ax.set_yticks(range(3))
        ax.set_yticklabels(signals_list)
        ax.set_xlabel("Predicted Signal")
        ax.set_ylabel("Expected Signal")
        ax.set_title("Signal Classification Confusion Matrix")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(matrix[i][j])}", ha="center", va="center",
                        color="white" if matrix[i][j] > matrix.max()/2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "signal_confusion.png", dpi=150)
        plt.close()
        print(f"  → signal_confusion.png")


# ── Report Generation ────────────────────────────────────────────────────────

def generate_report(judgments, responses, dim_results, signal_deltas):
    """Generate publication-quality markdown report."""
    n = len(judgments)
    wins = sum(1 for j in judgments if j["debiased_preference"] == "routed")
    losses = sum(1 for j in judgments if j["debiased_preference"] == "raw")
    ties = n - wins - losses
    win_rate = wins / n

    sig_acc = sum(1 for r in responses if r.get("signal_correct", False)) / len(responses) if responses else 0

    report = f"""# Phenomenological Compass Evaluation Report
## Proving Semantic Procedural Generation Improves LLM Response Quality

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Questions**: {n} (35 OPEN / 35 PAUSE / 35 WITNESS)
**Compass**: Ministral-3B + LoRA v0.8 (iter 50)
**Action Model**: Qwen3.5-9B-abliterated-MLX-4bit
**Judge**: Claude Sonnet ({N_CONSISTENCY_RUNS}x self-consistency, position-debiased)

---

### Abstract

Evaluation of {n} novel questions across three signal types (OPEN, PAUSE, WITNESS)
demonstrates that compass-routed responses are preferred over raw baseline responses
in {wins}/{n} cases ({win_rate:.0%}) with position-debiased LLM judging. The compass
achieves {sig_acc:.0%} signal classification accuracy. {'The strongest advantage appears in ' + max(signal_deltas, key=signal_deltas.get) + ' questions.' if signal_deltas else ''}

---

### Results

#### Win Rate (Position-Debiased)

| Condition | Wins | Rate | 95% CI |
|-----------|------|------|--------|
"""

    for signal in ["Overall", "OPEN", "PAUSE", "WITNESS"]:
        j_set = judgments if signal == "Overall" else [j for j in judgments if j["expected_signal"] == signal]
        if not j_set:
            continue
        data = [1 if j["debiased_preference"] == "routed" else 0 for j in j_set]
        wr = np.mean(data)
        ci = bootstrap_ci(data)
        sig_wins = sum(data)
        report += f"| {signal} | {sig_wins}/{len(j_set)} | {wr:.0%} | [{ci[0]:.0%}, {ci[1]:.0%}] |\n"

    report += f"""
#### Dimensional Scores

| Dimension | Routed | Raw | Delta | Cohen's d | p-value |
|-----------|--------|-----|-------|-----------|---------|
"""

    for dim, data in dim_results.items():
        sig = "***" if data["p_value"] < 0.001 else "**" if data["p_value"] < 0.01 else "*" if data["p_value"] < 0.05 else ""
        report += f"| {dim} | {data['routed_mean']:.2f} | {data['raw_mean']:.2f} | {data['delta']:+.2f} | {data['cohens_d']:.2f} | {data['p_value']:.4f}{sig} |\n"

    report += f"""
#### Signal Classification

| Signal | Accuracy |
|--------|----------|
"""

    if responses:
        by_signal = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in responses:
            by_signal[r["expected_signal"]]["total"] += 1
            if r.get("signal_correct"):
                by_signal[r["expected_signal"]]["correct"] += 1

        for sig in ["OPEN", "PAUSE", "WITNESS"]:
            s = by_signal[sig]
            acc = s["correct"] / s["total"] if s["total"] else 0
            report += f"| {sig} | {s['correct']}/{s['total']} ({acc:.0%}) |\n"

    report += f"""
---

### Figures

![Win Rates](win_rates.png)
![Dimensional Scores](dimensional_scores.png)
![Effect Sizes](effect_sizes.png)
![Signal Confusion Matrix](signal_confusion.png)

---

### Example Responses

"""

    # Top 5 strongest compass wins
    scored = []
    for j in judgments:
        if j.get("routed_scores") and j.get("raw_scores"):
            routed_total = sum(j["routed_scores"].values()) / 6
            raw_total = sum(j["raw_scores"].values()) / 6
            scored.append((routed_total - raw_total, j))

    scored.sort(key=lambda x: x[0], reverse=True)

    report += "#### Strongest Compass Wins\n\n"
    for delta, j in scored[:5]:
        report += f"**{j['expected_signal']}** | {j['id']} | Δ={delta:+.2f}\n"
        report += f"> {j['question']}\n\n"

    if any(d < 0 for d, _ in scored):
        report += "\n#### Raw Wins (Compass Losses)\n\n"
        for delta, j in sorted(scored, key=lambda x: x[0])[:3]:
            if delta < 0:
                report += f"**{j['expected_signal']}** | {j['id']} | Δ={delta:+.2f}\n"
                report += f"> {j['question']}\n\n"

    report += f"""
---

*Phenomenological Compass v0.8 — Temple of Two*
*Anthony J. Vasquez Sr.*
*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    report_path = RESULTS_DIR / "report.md"
    report_path.write_text(report)
    print(f"\n  → report.md")
    return report


# ── Main ─────────────────────────────────────────────────────────────────────

N_CONSISTENCY_RUNS = 3  # for report text

def main():
    parser = argparse.ArgumentParser(description="Compass evaluation analysis")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing judgments.jsonl and responses.jsonl")
    args = parser.parse_args()

    if args.input_dir:
        global JUDGMENTS_PATH, RESPONSES_PATH, RESULTS_DIR
        JUDGMENTS_PATH = Path(args.input_dir) / "judgments.jsonl"
        RESPONSES_PATH = Path(args.input_dir) / "responses.jsonl"
        RESULTS_DIR = Path(args.input_dir)

    if not JUDGMENTS_PATH.exists():
        print(f"ERROR: No judgments found at {JUDGMENTS_PATH}")
        print("Run: python3 eval/judge.py")
        sys.exit(1)

    judgments = load_jsonl(JUDGMENTS_PATH)
    responses = load_jsonl(RESPONSES_PATH) if RESPONSES_PATH.exists() else None

    print(f"Loaded {len(judgments)} judgments" + (f", {len(responses)} responses" if responses else ""))

    dim_results, signal_deltas = analyze(judgments, responses)

    if not args.no_figures:
        print(f"\n  Generating figures...")
        generate_figures(judgments, responses)

    generate_report(judgments, responses, dim_results, signal_deltas)

    # Save JSON report
    json_report = {
        "n_questions": len(judgments),
        "dimensions": dim_results,
        "signal_deltas": signal_deltas,
        "win_rate": sum(1 for j in judgments if j["debiased_preference"] == "routed") / len(judgments),
        "timestamp": datetime.now().isoformat(),
    }
    json_path = RESULTS_DIR / "report.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    print(f"  → report.json")


if __name__ == "__main__":
    main()
