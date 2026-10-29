"""
OptiRAG-lite Audit Figure Generation Script.
Generates:
1. results/pareto_v2.png:
   - Always-SMALL and Always-LARGE with 95% bootstrap CI error bars
   - Matched-cost random routing baseline curve (with 95% simulation envelope)
   - Learned routers (Query-Only, Telemetry-Only, Both) under train-calibrated thresholds
   - Heuristic baseline router
   - PTM-compressed variants (Always-Small PTM, Always-Large PTM, OptiRAG)
   - Visual indicator showing all routers fall within random routing confidence envelope
   - Slide-readable fonts (>=14pt) and exact axis labels
2. results/retrieval_accuracy.png:
   - Grouped bar chart showing accuracy of Small and Large when retrieval is complete vs incomplete
   - Side-by-side for Train (N=200) and Test (N=100)
   - Slide-readable fonts (>=14pt) and clear annotations
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set slide-readable fonts
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


def generate_pareto_v2():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # 1. Base endpoints
    cost_small = 0.0417
    cost_small_err = [[cost_small - 0.0399], [0.0433 - cost_small]]
    acc_small = 38.0
    acc_small_err = [[acc_small - 29.0], [48.0 - acc_small]]

    cost_large = 0.2294
    cost_large_err = [[cost_large - 0.2198], [0.2384 - cost_large]]
    acc_large = 48.0
    acc_large_err = [[acc_large - 38.0], [58.0 - acc_large]]

    # 2. Matched-Cost Random Baseline Curve
    p_vals = np.linspace(0.0, 1.0, 101)
    rand_costs = (1 - p_vals) * cost_small + p_vals * cost_large
    rand_accs = (1 - p_vals) * acc_small + p_vals * acc_large

    # Analytical 95% confidence band for random routing on N=100
    # SE(acc) = sqrt(acc * (100 - acc) / 100)
    se_acc = np.sqrt(rand_accs * (100 - rand_accs) / 100.0)
    band_low = np.maximum(0, rand_accs - 1.96 * se_acc)
    band_high = np.minimum(100, rand_accs + 1.96 * se_acc)

    ax.plot(rand_costs, rand_accs, linestyle="--", color="#6c757d", linewidth=2.5,
            label="Matched-Cost Random Routing (Coin Flip)")
    ax.fill_between(rand_costs, band_low, band_high, color="#6c757d", alpha=0.15,
                    label="Random Routing 95% Confidence Band (N=100)")

    # 3. Always Small and Always Large points with error bars
    ax.errorbar(cost_small, acc_small, xerr=cost_small_err, yerr=acc_small_err,
                fmt="o", color="#1f77b4", markersize=10, capsize=5, capthick=2, elinewidth=2,
                label="Always-SMALL (allam-2-7b, 95% CI)", zorder=5)

    ax.errorbar(cost_large, acc_large, xerr=cost_large_err, yerr=acc_large_err,
                fmt="s", color="#d62728", markersize=10, capsize=5, capthick=2, elinewidth=2,
                label="Always-LARGE (gemini-3.5-flash-lite, 95% CI)", zorder=5)

    # 4. Uncompressed Routers (marked with symbol denoting not significantly beating random)
    routers = [
        ("Heuristic Router (Score < 0.70)", 0.1713, 46.0, "^", "#ff7f0e"),
        ("Router: Query-Only (@ 30%)", 0.0898, 42.0, "D", "#2ca02c"),
        ("Router: Telemetry-Only (@ 30%)", 0.0968, 41.0, "p", "#9467bd"),
        ("Router: Both (@ 30%)", 0.0889, 44.0, "h", "#8c564b"),
    ]

    for name, c, a, marker, color in routers:
        ax.scatter(c, a, marker=marker, s=120, color=color, edgecolor="black", linewidth=1.5,
                   label=f"{name} *", zorder=6)

    # 5. PTM Compressed Variants
    ax.scatter(0.0271, 41.0, marker="o", s=110, facecolors="none", edgecolors="#1f77b4", linewidth=2.5,
               label="Always-SMALL (PTM-Compressed)", zorder=6)
    ax.scatter(0.1469, 46.0, marker="s", s=110, facecolors="none", edgecolors="#d62728", linewidth=2.5,
               label="Always-LARGE (PTM-Compressed)", zorder=6)
    ax.scatter(0.0619, 42.0, marker="*", s=220, color="#e377c2", edgecolor="black", linewidth=1.5,
               label="OptiRAG (Telemetry Router @ 30% + PTM) *", zorder=7)

    # Annotations
    ax.annotate("All routers (*) fall within\nrandom baseline 95% CI\n(No router statistically beats coin flip)",
                xy=(0.0968, 41.0), xytext=(0.05, 52.0),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=8),
                fontsize=13, fontweight="bold", backgroundcolor="#ffffff",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffeeba"))

    ax.annotate("PTM saves 36% cost\nwith +3% observed acc",
                xy=(0.0271, 41.0), xytext=(0.015, 33.0),
                arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1, headwidth=6),
                fontsize=11, backgroundcolor="#ffffff",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", edgecolor="#bce8f1"))

    # Labels and formatting
    ax.set_xlabel("Simulated cost per 1,000 queries (USD, list price)", labelpad=10)
    ax.set_ylabel("Accuracy (%)", labelpad=10)
    ax.set_title("OptiRAG-lite: Pareto Frontier & Matched-Cost Random Baseline Audit\n(N=100 Test Queries, Train-Calibrated Thresholds)", pad=15)
    ax.set_xlim(0.0, 0.26)
    ax.set_ylim(25.0, 62.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95, facecolor="white")

    plt.tight_layout()
    out_path = RESULTS_DIR / "pareto_v2.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


def generate_retrieval_accuracy_figure():
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # Data from audit
    # Train: Complete (n=99) vs Incomplete (n=101)
    # Test: Complete (n=49) vs Incomplete (n=51)
    categories = [
        "Train: Full Coverage\n(gold_all_in_top5 = True, n=99)",
        "Train: Incomplete\n(gold_all_in_top5 = False, n=101)",
        "Test: Full Coverage\n(gold_all_in_top5 = True, n=49)",
        "Test: Incomplete\n(gold_all_in_top5 = False, n=51)",
    ]

    small_accs = [71.7, 22.8, 57.1, 19.6]
    large_accs = [87.9, 30.7, 85.7, 11.8]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, small_accs, width, label="SMALL Tier (allam-2-7b)", color="#4a90e2", edgecolor="black", linewidth=1.2)
    rects2 = ax.bar(x + width/2, large_accs, width, label="LARGE Tier (gemini-3.5-flash-lite)", color="#e74c3c", edgecolor="black", linewidth=1.2)

    # Add direct percentage labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    # Annotation highlighting the core empirical driver
    ax.annotate("Complete retrieval enables ~86% Large accuracy;\nIncomplete collapses Large to 12% on test\n(Oracle Retrieval AUROC = 0.870 for Large Correctness)",
                xy=(2.18, 85.7), xytext=(1.2, 94.0),
                arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=8),
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor="#e74c3c"))

    ax.set_ylabel("Accuracy (%)", labelpad=10)
    ax.set_title("The Dominant Driver of RAG Success: Retrieval Completeness\nAccuracy of Small vs Large Tiers by Gold Supporting Document Recall", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="white")

    plt.tight_layout()
    out_path = RESULTS_DIR / "retrieval_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_pareto_v2()
    generate_retrieval_accuracy_figure()
