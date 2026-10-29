"""
Script to generate all audited _v4 figures:
1. results/ptm_survival_v4.png:
   - Y-axis capped at 100 (ticks up to 100, headroom for labels up to 110)
   - Shows span-level and word-level survival for dates and currency (and all 5 categories)
   - Labels n under each kind
   - Verified cause caption
2. results/pareto_v4.png:
   - Two-panel slide-readable layout:
     Panel A: Accuracy vs. Simulated Cost (with matched-cost random baseline and 95% band)
     Panel B: End-to-End Latency vs. Simulated Cost (highlighting ~7s CPU compression latency)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Global slide-readable font sizes (>=14pt)
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_ptm_survival_v4():
    """
    ptm_survival_v4.png:
    - y-axis capped at 100 with ticks [0, 20, 40, 60, 80, 100] and ylim=(0, 112) for label headroom
    - shows span-level and word-level survival for dates and currency
    - includes number, percent, negation
    - prints n under each category
    - verified notes only
    """
    categories = [
        "Number\n(n=1,667 spans)",
        "Date\n(n=354 spans\n900 words)",
        "Currency\n(n=14 spans\n26 words)",
        "Percent\n(n=6 spans)",
        "Negation\n(n=69 spans)",
    ]

    vanilla_span = [95.98, 57.91, 35.71, 16.67, 73.91]
    ptm_span = [100.0, 66.10, 85.71, 100.0, 100.0]
    ptm_word = [100.0, 100.0, 100.0, 100.0, 100.0]  # 100% of real currency/date words

    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    x = np.arange(len(categories))
    width = 0.26

    rects1 = ax.bar(
        x - width,
        vanilla_span,
        width,
        label="Vanilla LLMLingua-2 (Span-Level)",
        color="#95a5a6",
        edgecolor="black",
        linewidth=1.2,
    )
    rects2 = ax.bar(
        x,
        ptm_span,
        width,
        label="PTM (Exact Substring Span-Level)",
        color="#27ae60",
        edgecolor="black",
        linewidth=1.2,
    )
    rects3 = ax.bar(
        x + width,
        ptm_word,
        width,
        label="PTM (Constituent Word & Digit Level)",
        color="#2ecc71",
        edgecolor="black",
        linewidth=1.2,
        hatch="//",
    )

    # Label bars
    for rect, val in zip(rects1, vanilla_span):
        ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    for rect, val in zip(rects2, ptm_span):
        ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    for rect, val in zip(rects3, ptm_word):
        ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    # Formatting: Y-axis capped strictly at 100
    ax.set_title("Entity Survival: Span-Level vs. Word-Level Analysis (N=100 Test Contexts)", pad=15)
    ax.set_ylabel("Survival Rate (%)", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, framealpha=0.95, facecolor="white", edgecolor="#cccccc")

    # Verified caption below chart
    fig.text(
        0.5,
        0.01,
        "Verified Findings: Percent (n=6) and currency (n=14) have small sample sizes.\n"
        "Date exact span-level survival is 66.1% because LLMLingua-2 prunes punctuation commas;\n"
        "word-level audit proves 100% of constituent date words and digits (900/900) survived under PTM.\n"
        "Vanilla word-level test strings were not serialized to disk (NOT AVAILABLE).",
        ha="center",
        fontsize=11,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cbd5e1", alpha=0.95),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    out_path = RESULTS_DIR / "ptm_survival_v4.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


def generate_pareto_v4():
    """
    pareto_v4.png:
    Two-panel figure:
    - Panel A: Accuracy vs. Simulated Cost (with matched-cost random routing baseline & 95% band)
    - Panel B: End-to-End Latency vs. Simulated Cost (showing LLM latency vs CPU compression overhead)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8.5), dpi=300)

    # Base points
    cost_small = 0.0417
    acc_small = 38.0
    cost_large = 0.2294
    acc_large = 48.0

    # Random baseline curve
    p_vals = np.linspace(0.0, 1.0, 101)
    rand_costs = (1 - p_vals) * cost_small + p_vals * cost_large
    rand_accs = (1 - p_vals) * acc_small + p_vals * acc_large

    se_acc = np.sqrt(rand_accs * (100 - rand_accs) / 100.0)
    band_low = np.maximum(0, rand_accs - 1.96 * se_acc)
    band_high = np.minimum(100, rand_accs + 1.96 * se_acc)

    # --- PANEL A: Accuracy vs. Cost ---
    ax1.plot(rand_costs, rand_accs, linestyle="--", color="#495057", linewidth=2.2,
             label="Matched-cost random routing", zorder=3)
    ax1.fill_between(rand_costs, band_low, band_high, color="#6c757d", alpha=0.18,
                     label="random routing at matched cost (95% CI band)", zorder=2)

    # Always Small and Large with CIs
    ax1.errorbar(cost_small, acc_small, xerr=[[cost_small - 0.0399], [0.0433 - cost_small]],
                 yerr=[[acc_small - 29.0], [48.0 - acc_small]], fmt="o", color="#1f77b4",
                 markersize=10, capsize=5, capthick=1.5, elinewidth=1.8, label="Always-SMALL (allam-2-7b, 95% CI)", zorder=5)

    ax1.errorbar(cost_large, acc_large, xerr=[[cost_large - 0.2198], [0.2384 - cost_large]],
                 yerr=[[acc_large - 38.0], [58.0 - acc_large]], fmt="s", color="#d62728",
                 markersize=10, capsize=5, capthick=1.5, elinewidth=1.8, label="Always-LARGE (gemini-3.5-flash-lite, 95% CI)", zorder=5)

    # Routers
    routers = [
        ("Heuristic Router (Score < 0.70)", 0.1713, 46.0, "^", "#e67e22", 0.676, 0.676),
        ("Router: Query-Only (@ 30%)", 0.0898, 42.0, "D", "#27ae60", 0.373, 0.373),
        ("Router: Telemetry-Only (@ 30%)", 0.0968, 41.0, "p", "#8e44ad", 0.389, 0.389),
        ("Router: Both (@ 30%)", 0.0889, 44.0, "h", "#d35400", 0.369, 0.369),
    ]

    for name, c, a, marker, color, llm_lat, e2e_lat in routers:
        ax1.scatter(c, a, marker=marker, s=120, color=color, edgecolor="black", linewidth=1.2, label=name, zorder=6)

    # PTM variants (hollow markers)
    ax1.scatter(0.0271, 41.0, marker="o", s=120, facecolors="none", edgecolors="#1f77b4", linewidth=2.5,
                label="Always-SMALL (PTM-Compressed)", zorder=6)
    ax1.scatter(0.1469, 46.0, marker="s", s=120, facecolors="none", edgecolors="#d62728", linewidth=2.5,
                label="Always-LARGE (PTM-Compressed)", zorder=6)
    ax1.scatter(0.0619, 42.0, marker="*", s=230, facecolors="none", edgecolors="#e377c2", linewidth=2.5,
                label="OptiRAG (Telemetry @ 30% + PTM)", zorder=7)

    ax1.text(0.03, 0.95, "No router is distinguishable from random routing\n(all paired CIs include 0)",
             transform=ax1.transAxes, fontsize=12, fontweight="bold", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffeeba", alpha=0.95))

    ax1.set_title("Panel A: Accuracy vs. Simulated Cost (N=100)", pad=12)
    ax1.set_xlabel("Simulated cost per 1,000 queries (USD, list price)", labelpad=10)
    ax1.set_ylabel("Accuracy (%)", labelpad=10)
    ax1.set_xlim(0.015, 0.25)
    ax1.set_ylim(26.0, 60.0)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # --- PANEL B: End-to-End Latency vs. Simulated Cost ---
    # Uncompressed systems
    ax2.scatter(cost_small, 0.173, marker="o", s=120, color="#1f77b4", edgecolor="black", label="Always-SMALL (LLM only: 0.17s)", zorder=5)
    ax2.scatter(cost_large, 0.919, marker="s", s=120, color="#d62728", edgecolor="black", label="Always-LARGE (LLM only: 0.92s)", zorder=5)
    for name, c, a, marker, color, llm_lat, e2e_lat in routers:
        ax2.scatter(c, e2e_lat, marker=marker, s=110, color=color, edgecolor="black", label=f"{name} ({e2e_lat:.2f}s)", zorder=5)

    # Compressed systems (with 6.87s CPU compression latency)
    ax2.scatter(0.0271, 7.016, marker="o", s=130, facecolors="none", edgecolors="#1f77b4", linewidth=2.5,
                label="Always-SMALL + PTM (E2E: 7.02s)", zorder=6)
    ax2.scatter(0.1469, 7.791, marker="s", s=130, facecolors="none", edgecolors="#d62728", linewidth=2.5,
                label="Always-LARGE + PTM (E2E: 7.79s)", zorder=6)
    ax2.scatter(0.0619, 7.224, marker="*", s=240, facecolors="none", edgecolors="#e377c2", linewidth=2.5,
                label="OptiRAG + PTM (E2E: 7.22s)", zorder=7)

    # Connect LLM latency to E2E latency with dashed arrows for compressed systems
    ax2.annotate("", xy=(0.0271, 6.8), xytext=(0.0271, 0.35),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5, ls="--"))
    ax2.annotate("", xy=(0.1469, 7.6), xytext=(0.1469, 1.1),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5, ls="--"))

    ax2.text(0.04, 0.55, "+6.87s CPU Compression Overhead\n(Compression reduces tokens & cost,\nNOT end-to-end latency on CPU)",
             transform=ax2.transAxes, fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8d7da", edgecolor="#f5c6cb", alpha=0.95))

    ax2.set_title("Panel B: End-to-End Latency vs. Cost", pad=12)
    ax2.set_xlabel("Simulated cost per 1,000 queries (USD, list price)", labelpad=10)
    ax2.set_ylabel("End-to-End Latency (seconds, log scale)", labelpad=10)
    ax2.set_yscale("log")
    ax2.set_ylim(0.1, 12.0)
    ax2.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    ax2.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Combined legend below plots
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(handles1, labels1, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, framealpha=0.95, facecolor="white", edgecolor="#cccccc", fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    out_path = RESULTS_DIR / "pareto_v4.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_ptm_survival_v4()
    generate_pareto_v4()
