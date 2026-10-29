"""
Script to generate all audited _v3 figures:
1. results/retrieval_accuracy_v3.png
2. results/pareto_v3.png
3. results/ptm_survival_v3.png
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
    "legend.fontsize": 13,
    "figure.titlesize": 17,
})

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_retrieval_accuracy_v3():
    with open(RESULTS_DIR / "extra_stats_v3.json", "r", encoding="utf-8") as f:
        stats = json.load(f)["retrieval_subgroups"]

    categories = [
        "Train, all gold\nretrieved (n=99)",
        "Train, incomplete\nretrieval (n=101)",
        "Test, all gold\nretrieved (n=49)",
        "Test, incomplete\nretrieval (n=51)",
    ]

    keys = ["train_complete_n99", "train_incomplete_n101", "test_complete_n49", "test_incomplete_n51"]

    small_means = [stats[k]["small_mean"] for k in keys]
    small_err_low = [stats[k]["small_mean"] - stats[k]["small_ci"][0] for k in keys]
    small_err_high = [stats[k]["small_ci"][1] - stats[k]["small_mean"] for k in keys]
    small_yerr = [small_err_low, small_err_high]

    large_means = [stats[k]["large_mean"] for k in keys]
    large_err_low = [stats[k]["large_mean"] - stats[k]["large_ci"][0] for k in keys]
    large_err_high = [stats[k]["large_ci"][1] - stats[k]["large_mean"] for k in keys]
    large_yerr = [large_err_low, large_err_high]

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        small_means,
        width,
        yerr=small_yerr,
        error_kw=dict(capsize=5, capthick=1.5, elinewidth=1.5),
        label="SMALL (allam-2-7b)",
        color="#2b5c8f",
        edgecolor="black",
        linewidth=1.2,
    )
    rects2 = ax.bar(
        x + width / 2,
        large_means,
        width,
        yerr=large_yerr,
        error_kw=dict(capsize=5, capthick=1.5, elinewidth=1.5),
        label="LARGE (gemini-3.5-flash-lite)",
        color="#c0392b",
        edgecolor="black",
        linewidth=1.2,
    )

    # Label values comfortably above the error bar caps (zero collision)
    for rect, val, err_h in zip(rects1, small_means, small_err_high):
        ax.annotate(
            f"{val:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, val + err_h + 1.5),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    for rect, val, err_h in zip(rects2, large_means, large_err_high):
        ax.annotate(
            f"{val:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, val + err_h + 1.5),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title("Accuracy depends on whether retrieval found all gold paragraphs", pad=15)
    ax.set_ylabel("Accuracy (%)", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Legend in upper left with clean border
    ax.legend(loc="upper left", framealpha=0.95, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    out_path = RESULTS_DIR / "retrieval_accuracy_v3.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


def generate_pareto_v3():
    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)

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

    se_acc = np.sqrt(rand_accs * (100 - rand_accs) / 100.0)
    band_low = np.maximum(0, rand_accs - 1.96 * se_acc)
    band_high = np.minimum(100, rand_accs + 1.96 * se_acc)

    ax.plot(
        rand_costs,
        rand_accs,
        linestyle="--",
        color="#495057",
        linewidth=2.2,
        label="Matched-cost random routing",
        zorder=3,
    )
    ax.fill_between(
        rand_costs,
        band_low,
        band_high,
        color="#6c757d",
        alpha=0.18,
        label="random routing at matched cost (95% CI band, N=100)",
        zorder=2,
    )

    # 3. Always Small and Always Large points with error bars
    ax.errorbar(
        cost_small,
        acc_small,
        xerr=cost_small_err,
        yerr=acc_small_err,
        fmt="o",
        color="#1f77b4",
        markersize=10,
        capsize=5,
        capthick=1.5,
        elinewidth=1.8,
        label="Always-SMALL (allam-2-7b, 95% CI)",
        zorder=5,
    )

    ax.errorbar(
        cost_large,
        acc_large,
        xerr=cost_large_err,
        yerr=acc_large_err,
        fmt="s",
        color="#d62728",
        markersize=10,
        capsize=5,
        capthick=1.5,
        elinewidth=1.8,
        label="Always-LARGE (gemini-3.5-flash-lite, 95% CI)",
        zorder=5,
    )

    # 4. Uncompressed Routers
    routers = [
        ("Heuristic Router (Score < 0.70)", 0.1713, 46.0, "^", "#e67e22"),
        ("Router: Query-Only (@ 30%)", 0.0898, 42.0, "D", "#27ae60"),
        ("Router: Telemetry-Only (@ 30%)", 0.0968, 41.0, "p", "#8e44ad"),
        ("Router: Both (@ 30%)", 0.0889, 44.0, "h", "#d35400"),
    ]

    for name, c, a, marker, color in routers:
        ax.scatter(
            c,
            a,
            marker=marker,
            s=130,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            label=name,
            zorder=6,
        )

    # 5. PTM Compressed Variants (Hollow Markers)
    ax.scatter(
        0.0271,
        41.0,
        marker="o",
        s=120,
        facecolors="none",
        edgecolors="#1f77b4",
        linewidth=2.5,
        label="Always-SMALL (PTM-Compressed)",
        zorder=6,
    )
    ax.scatter(
        0.1469,
        46.0,
        marker="s",
        s=120,
        facecolors="none",
        edgecolors="#d62728",
        linewidth=2.5,
        label="Always-LARGE (PTM-Compressed)",
        zorder=6,
    )
    ax.scatter(
        0.0619,
        42.0,
        marker="*",
        s=230,
        facecolors="none",
        edgecolors="#e377c2",
        linewidth=2.5,
        label="OptiRAG (Telemetry Router @ 30% + PTM)",
        zorder=7,
    )

    # Note in plot away from error bars (upper left)
    ax.text(
        0.02,
        0.95,
        "No router is distinguishable from random routing\n(all paired CIs include 0)",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff3cd", edgecolor="#ffeeba", alpha=0.95),
    )

    # Labels and title
    ax.set_title("Accuracy vs. simulated cost (N=100 test queries)", pad=15)
    ax.set_xlabel("Simulated cost per 1,000 queries (USD, list price)", labelpad=10)
    ax.set_ylabel("Accuracy (%)", labelpad=10)
    ax.set_xlim(0.015, 0.25)
    ax.set_ylim(26.0, 60.0)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Put legend below the plot (outside)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=12,
    )

    plt.tight_layout()
    out_path = RESULTS_DIR / "pareto_v3.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


def generate_ptm_survival_v3():
    kinds = ["Number\n(n=1,667)", "Date\n(n=354)", "Currency\n(n=14)", "Percent\n(n=6)", "Negation\n(n=69)"]
    vanilla_rates = [95.98, 57.91, 35.71, 16.67, 73.91]
    ptm_rates = [100.0, 66.10, 85.71, 100.0, 100.0]

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    x = np.arange(len(kinds))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        vanilla_rates,
        width,
        label="Vanilla LLMLingua-2 (rate=0.5)",
        color="#95a5a6",
        edgecolor="black",
        linewidth=1.2,
    )
    rects2 = ax.bar(
        x + width / 2,
        ptm_rates,
        width,
        label="PTM (Protected Token Mask)",
        color="#27ae60",
        edgecolor="black",
        linewidth=1.2,
    )

    # Label percentages on bars
    for rect, val in zip(rects1, vanilla_rates):
        ax.annotate(
            f"{val:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    for rect, val in zip(rects2, ptm_rates):
        ax.annotate(
            f"{val:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title("Entity token survival: Vanilla LLMLingua-2 vs. PTM (N=100 test contexts)", pad=15)
    ax.set_ylabel("Survival Rate (%)", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, fontsize=13)
    ax.set_ylim(0, 130)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, framealpha=0.95, facecolor="white", edgecolor="#cccccc")

    # Clean note below the plot, completely clear of all bars
    fig.text(
        0.5,
        0.01,
        "Note: percent (n=6) and currency (n=14) are small.\nDates show 66.1% due to punctuation commas stripped by tokenizer; all constituent day/month/year words are 100% retained.",
        ha="center",
        fontsize=12,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cbd5e1", alpha=0.95),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    out_path = RESULTS_DIR / "ptm_survival_v3.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_retrieval_accuracy_v3()
    generate_pareto_v3()
    generate_ptm_survival_v3()
