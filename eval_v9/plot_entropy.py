#!/usr/bin/env python3
"""
plot_entropy.py — Generate entropy profiling figures
=====================================================
Reads entropy_profiles.jsonl and produces:
  - entropy_by_signal.png (violin plots)
  - entropy_trajectory.png (mean curves over token position)
  - jsd_heatmap.png (JSD between routed vs raw per signal)

Usage:
    python3 eval_v9/plot_entropy.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT / "eval_v9" / "results"
PROFILES_PATH = RESULTS_DIR / "entropy_profiles.jsonl"

SIGNAL_COLORS = {"OPEN": "#4CAF50", "PAUSE": "#FF9800", "WITNESS": "#9C27B0"}


def load_profiles():
    profiles = []
    with open(PROFILES_PATH) as f:
        for line in f:
            if line.strip():
                profiles.append(json.loads(line))
    return profiles


def plot_violin(profiles):
    """Violin plots of entropy distributions per signal, routed vs raw."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Entropy Distributions by Signal: Compass-Routed vs Raw", fontsize=14, fontweight="bold")

    for idx, signal in enumerate(["OPEN", "PAUSE", "WITNESS"]):
        ax = axes[idx]
        signal_profiles = [p for p in profiles if p["expected_signal"] == signal]

        routed_all = []
        raw_all = []
        for p in signal_profiles:
            routed_all.extend(p["routed_entropies"])
            raw_all.extend(p["raw_entropies"])

        if routed_all and raw_all:
            parts = ax.violinplot([routed_all, raw_all], positions=[1, 2], showmeans=True, showmedians=True)
            for pc, color in zip(parts["bodies"], [SIGNAL_COLORS[signal], "#888888"]):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

        ax.set_title(signal, fontsize=12, fontweight="bold", color=SIGNAL_COLORS[signal])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Routed", "Raw"])
        ax.set_ylabel("Shannon Entropy (nats)" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "entropy_by_signal.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")


def plot_trajectory(profiles):
    """Mean entropy curves over token position."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Entropy Trajectory by Signal (Mean over Questions)", fontsize=14, fontweight="bold")

    for idx, signal in enumerate(["OPEN", "PAUSE", "WITNESS"]):
        ax = axes[idx]
        signal_profiles = [p for p in profiles if p["expected_signal"] == signal]

        # Pad traces to same length for averaging
        max_len = max(
            max((len(p["routed_entropies"]) for p in signal_profiles), default=0),
            max((len(p["raw_entropies"]) for p in signal_profiles), default=0),
        )
        if max_len == 0:
            continue

        routed_matrix = np.full((len(signal_profiles), max_len), np.nan)
        raw_matrix = np.full((len(signal_profiles), max_len), np.nan)

        for i, p in enumerate(signal_profiles):
            r = p["routed_entropies"]
            routed_matrix[i, :len(r)] = r
            w = p["raw_entropies"]
            raw_matrix[i, :len(w)] = w

        x = np.arange(max_len)
        routed_mean = np.nanmean(routed_matrix, axis=0)
        raw_mean = np.nanmean(raw_matrix, axis=0)
        routed_std = np.nanstd(routed_matrix, axis=0)
        raw_std = np.nanstd(raw_matrix, axis=0)

        ax.plot(x, routed_mean, color=SIGNAL_COLORS[signal], label="Routed", linewidth=2)
        ax.fill_between(x, routed_mean - routed_std, routed_mean + routed_std,
                         color=SIGNAL_COLORS[signal], alpha=0.15)
        ax.plot(x, raw_mean, color="#888888", label="Raw", linewidth=2, linestyle="--")
        ax.fill_between(x, raw_mean - raw_std, raw_mean + raw_std,
                         color="#888888", alpha=0.1)

        ax.set_title(signal, fontsize=12, fontweight="bold", color=SIGNAL_COLORS[signal])
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Shannon Entropy" if idx == 0 else "")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "entropy_trajectory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")


def plot_jsd_heatmap(profiles):
    """JSD between routed vs raw, per question, grouped by signal."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Jensen-Shannon Divergence: Routed vs Raw", fontsize=14, fontweight="bold")

    for signal in ["OPEN", "PAUSE", "WITNESS"]:
        signal_profiles = [p for p in profiles if p["expected_signal"] == signal]
        jsds = [p["jsd"] for p in signal_profiles]
        ids = [p["id"] for p in signal_profiles]

        x = range(len(jsds))
        ax.bar([f"{signal}\n{i}" for i in range(len(jsds))],
               jsds, color=SIGNAL_COLORS[signal], alpha=0.7, label=signal)

    # Simpler: box plot by signal
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.suptitle("JSD Distribution by Signal", fontsize=14, fontweight="bold")

    data = []
    labels = []
    colors = []
    for signal in ["OPEN", "PAUSE", "WITNESS"]:
        signal_profiles = [p for p in profiles if p["expected_signal"] == signal]
        jsds = [p["jsd"] for p in signal_profiles]
        if jsds:
            data.append(jsds)
            labels.append(signal)
            colors.append(SIGNAL_COLORS[signal])

    bp = ax2.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel("Jensen-Shannon Divergence")
    ax2.grid(axis="y", alpha=0.3)

    plt.close(fig)  # Close the bar chart, keep box plot

    out = RESULTS_DIR / "jsd_heatmap.png"
    fig2.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  → {out.name}")


def main():
    if not PROFILES_PATH.exists():
        print(f"ERROR: No profiles at {PROFILES_PATH}")
        print("Run: python3 eval_v9/entropy_profile.py")
        return

    profiles = load_profiles()
    print(f"Loaded {len(profiles)} entropy profiles")
    print("Generating figures...")

    plot_violin(profiles)
    plot_trajectory(profiles)
    plot_jsd_heatmap(profiles)

    print("\nDone.")


if __name__ == "__main__":
    main()
