"""
OptiRAG-lite Full Pipeline Evaluation & Pareto Analysis (eval.py).

Evaluates all systems on the 100 test queries (ZERO new API calls):
1. Always-SMALL (Groq allam-2-7b)
2. Always-LARGE (Gemini 3.5 Flash-Lite)
3. Heuristic Router
4. Learned Routers (query-only, telemetry-only, both) across thresholds [0, 1]
5. Random Routing baseline at matched cost
6. Compressed Context variants (Always-Small Compressed, Always-Large Compressed, OptiRAG)

Outputs generated:
- results/summary.csv: Full benchmark table with accuracy, F1, cost/1k, latency, 95% bootstrap CIs
- results/ablation.csv: Router threshold sweep data
- results/pareto.png: Professional publication-grade Pareto frontier plot
- results/entity_survival.png: Bar chart of PTM entity survival rates by category
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import config
import metrics


def compute_query_cost(in_tokens: int, out_tokens: int, tier: str) -> float:
    """Calculates simulated dollar cost for a single query using official list prices."""
    cfg = config.PRICES_PER_MILLION[tier]
    p_cost = (in_tokens / 1e6) * cfg["prompt"]
    c_cost = (out_tokens / 1e6) * cfg["completion"]
    return float(p_cost + c_cost)


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes empirical bootstrap confidence interval for the sample mean."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return low, high


def load_test_records() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Loads merged test labels, and optional compressed test labels."""
    merged_path = config.RESULTS_DIR / "labels_test.jsonl"
    if not merged_path.is_file():
        raise FileNotFoundError(f"Missing merged test labels: {merged_path}")

    uncompressed_records = []
    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                uncompressed_records.append(json.loads(line))

    # Optional Small Compressed
    small_comp_path = config.RESULTS_DIR / "labels_small_test_compressed.jsonl"
    small_comp = {}
    if small_comp_path.is_file():
        with open(small_comp_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    small_comp[rec["question_id"]] = rec

    # Optional Large Compressed
    large_comp_path = config.RESULTS_DIR / "labels_large_test_compressed.jsonl"
    large_comp = {}
    if large_comp_path.is_file():
        with open(large_comp_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    large_comp[rec["question_id"]] = rec

    return uncompressed_records, small_comp, large_comp


def load_router_probabilities() -> Dict[str, np.ndarray]:
    """Loads predicted probabilities from results/router_test_probs.csv."""
    probs_path = config.RESULTS_DIR / "router_test_probs.csv"
    if not probs_path.is_file():
        raise FileNotFoundError(f"Missing router test probabilities: {probs_path}")

    with open(probs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    probs_dict = {}
    for col in columns:
        if col != "question_id":
            probs_dict[col] = np.array([float(r[col]) for r in rows], dtype=np.float32)
    return probs_dict


def evaluate_system(
    name: str,
    decisions: np.ndarray,  # 0 = Small, 1 = Large
    records: List[Dict[str, Any]],
    small_comp: Dict[str, Dict[str, Any]],
    large_comp: Dict[str, Dict[str, Any]],
    use_compression: bool = False,
) -> Dict[str, Any]:
    """Evaluates a routed configuration across all 100 test items."""
    n = len(records)
    correct_flags = []
    f1_scores = []
    costs = []
    latencies = []
    routed_large_count = 0

    for idx, r in enumerate(records):
        qid = r["question_id"]
        gold = r["gold_answer"]
        route = decisions[idx]  # 0 or 1

        if route == 1:
            routed_large_count += 1
            if use_compression and qid in large_comp:
                ans = large_comp[qid]["answer"]
                in_tok = large_comp[qid]["in_tokens"]
                out_tok = large_comp[qid]["out_tokens"]
                lat = large_comp[qid]["latency_s"]
            else:
                ans = r["large_answer"]
                in_tok = r["large_in_tokens"]
                out_tok = r["large_out_tokens"]
                lat = r["large_latency_s"]
            tier = "large"
        else:
            if use_compression and qid in small_comp:
                ans = small_comp[qid]["answer"]
                in_tok = small_comp[qid]["in_tokens"]
                out_tok = small_comp[qid]["out_tokens"]
                lat = small_comp[qid]["latency_s"]
            else:
                ans = r["small_answer"]
                in_tok = r["small_in_tokens"]
                out_tok = r["small_out_tokens"]
                lat = r["small_latency_s"]
            tier = "small"

        corr = metrics.is_correct(ans, gold)
        f1 = metrics.compute_f1(ans, gold)
        q_cost = compute_query_cost(in_tok, out_tok, tier)

        correct_flags.append(float(corr))
        f1_scores.append(float(f1))
        costs.append(q_cost)
        latencies.append(lat)

    correct_arr = np.array(correct_flags)
    cost_arr = np.array(costs)
    f1_arr = np.array(f1_scores)
    lat_arr = np.array(latencies)

    mean_acc = float(np.mean(correct_arr))
    acc_ci_low, acc_ci_high = bootstrap_ci(correct_arr)

    # Cost per 1,000 queries
    cost_per_1k = float(np.mean(cost_arr) * 1000.0)
    cost_1k_low, cost_1k_high = bootstrap_ci(cost_arr * 1000.0)

    mean_f1 = float(np.mean(f1_arr))
    mean_lat = float(np.mean(lat_arr))
    large_route_pct = float(routed_large_count / n * 100.0)

    return {
        "system": name,
        "use_compression": use_compression,
        "accuracy": round(mean_acc * 100.0, 2),
        "accuracy_ci_low": round(acc_ci_low * 100.0, 2),
        "accuracy_ci_high": round(acc_ci_high * 100.0, 2),
        "f1": round(mean_f1, 4),
        "cost_per_1k": round(cost_per_1k, 4),
        "cost_per_1k_ci_low": round(cost_1k_low, 4),
        "cost_per_1k_ci_high": round(cost_1k_high, 4),
        "mean_latency_s": round(mean_lat, 3),
        "large_route_pct": round(large_route_pct, 1),
    }


def main():
    print("=" * 80)
    print("STEP 5: OPTIRAG-LITE BENCHMARK EVALUATION & PARETO SWEEP")
    print("=" * 80)

    records, small_comp, large_comp = load_test_records()
    probs = load_router_probabilities()
    n = len(records)
    print(f"Loaded {n} test records.")
    print(f"Compressed answers available: Small={len(small_comp)} | Large={len(large_comp)}")

    # 1. Evaluate Standard Baseline Systems
    summary_rows = []

    # Always Small
    always_small_decisions = np.zeros(n, dtype=int)
    res_small = evaluate_system("Always-SMALL", always_small_decisions, records, small_comp, large_comp)
    summary_rows.append(res_small)

    # Always Large
    always_large_decisions = np.ones(n, dtype=int)
    res_large = evaluate_system("Always-LARGE", always_large_decisions, records, small_comp, large_comp)
    summary_rows.append(res_large)

    # Heuristic Router
    h_probs = probs.get("prob_heuristic", np.zeros(n))
    h_decisions = (h_probs >= 0.5).astype(int)
    res_heur = evaluate_system("Heuristic Router", h_decisions, records, small_comp, large_comp)
    summary_rows.append(res_heur)

    # Random Routing matched to Heuristic cost
    h_large_pct = res_heur["large_route_pct"] / 100.0
    rng = np.random.RandomState(config.SEED)
    rand_decisions = (rng.rand(n) < h_large_pct).astype(int)
    res_rand = evaluate_system(f"Random Routing ({h_large_pct*100:.0f}% Large)", rand_decisions, records, small_comp, large_comp)
    summary_rows.append(res_rand)

    # Learned Routers at default threshold 0.5 (Uncompressed)
    router_cols = [
        ("prob_query_only_logistic", "Router: Query-Only (LogReg)"),
        ("prob_telemetry_only_logistic", "Router: Telemetry-Only (LogReg)"),
        ("prob_both_logistic", "Router: Both (LogReg)"),
    ]
    if "prob_telemetry_only_mlp" in probs:
        router_cols.append(("prob_telemetry_only_mlp", "Router: Telemetry-Only (MLP)"))

    for col_key, sys_name in router_cols:
        if col_key in probs:
            p_arr = probs[col_key]
            dec = (p_arr >= 0.5).astype(int)
            res = evaluate_system(sys_name, dec, records, small_comp, large_comp, use_compression=False)
            summary_rows.append(res)

    # Compressed Systems (if available)
    has_compressed = len(small_comp) == n
    if has_compressed:
        print("\n[Compressed Test Data Detected] Evaluating compressed pipelines...")
        # Always Small Compressed
        res_small_comp = evaluate_system("Always-SMALL (PTM-Compressed)", always_small_decisions, records, small_comp, large_comp, use_compression=True)
        summary_rows.append(res_small_comp)

        # OptiRAG Pipeline = Best Router (Telemetry-Only) + Compression
        if "prob_telemetry_only_logistic" in probs:
            optirag_dec = (probs["prob_telemetry_only_logistic"] >= 0.5).astype(int)
            res_optirag = evaluate_system("OptiRAG (Telemetry Router + PTM)", optirag_dec, records, small_comp, large_comp, use_compression=True)
            summary_rows.append(res_optirag)

    # 2. Save summary.csv
    summary_csv = config.RESULTS_DIR / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary benchmark table: {summary_csv}")

    # 3. Pareto Sweep across Thresholds [0.0 to 1.0]
    print("\nExecuting Pareto threshold sweeps...")
    thresholds = np.linspace(0.0, 1.0, 41)
    ablation_rows = []
    sweep_results = {}

    sweep_configs = [
        ("prob_query_only_logistic", "Query-Only (LogReg)", "blue", "--"),
        ("prob_telemetry_only_logistic", "Telemetry-Only (LogReg)", "green", "-"),
        ("prob_both_logistic", "Both (LogReg)", "purple", "-."),
    ]
    if "prob_telemetry_only_mlp" in probs:
        sweep_configs.append(("prob_telemetry_only_mlp", "Telemetry-Only (MLP)", "darkgreen", ":"))

    for col_key, label, color, style in sweep_configs:
        if col_key not in probs:
            continue
        p_arr = probs[col_key]
        points = []
        for th in thresholds:
            dec = (p_arr >= th).astype(int)
            eval_res = evaluate_system(f"{label} @ {th:.2f}", dec, records, small_comp, large_comp, use_compression=False)
            eval_res["threshold"] = round(th, 3)
            eval_res["router_variant"] = label
            ablation_rows.append(eval_res)
            points.append((eval_res["cost_per_1k"], eval_res["accuracy"]))
        sweep_results[label] = {"points": points, "color": color, "style": style}

    # Save ablation.csv
    ablation_csv = config.RESULTS_DIR / "ablation.csv"
    with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(ablation_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"Saved ablation sweep data: {ablation_csv}")

    # 4. Generate Publication-Grade Pareto Frontier Plot (results/pareto.png)
    print("\nGenerating Pareto plot (results/pareto.png)...")
    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot baseline reference line (Random Routing frontier connecting Small to Large)
    c_small, a_small = res_small["cost_per_1k"], res_small["accuracy"]
    c_large, a_large = res_large["cost_per_1k"], res_large["accuracy"]
    plt.plot([c_small, c_large], [a_small, a_large], "k--", alpha=0.5, label="Random Routing Frontier", linewidth=1.5)

    # Plot Baselines
    plt.scatter([c_small], [a_small], color="gray", s=120, zorder=5, label=f"Always-SMALL ({a_small}%, ${c_small:.3f})")
    plt.scatter([c_large], [a_large], color="red", s=120, zorder=5, label=f"Always-LARGE ({a_large}%, ${c_large:.3f})")
    plt.scatter([res_heur["cost_per_1k"]], [res_heur["accuracy"]], color="orange", marker="s", s=120, zorder=5,
                label=f"Heuristic ({res_heur['accuracy']}%, ${res_heur['cost_per_1k']:.3f})")

    # Plot Router Sweep Curves
    for label, info in sweep_results.items():
        pts = sorted(info["points"], key=lambda x: x[0])
        x_vals = [p[0] for p in pts]
        y_vals = [p[1] for p in pts]
        plt.plot(x_vals, y_vals, label=label, color=info["color"], linestyle=info["style"], linewidth=2.0)

    # Plot OptiRAG system if available
    for row in summary_rows:
        if "OptiRAG" in row["system"]:
            plt.scatter([row["cost_per_1k"]], [row["accuracy"]], color="darkviolet", marker="*", s=220, zorder=6,
                        label=f"OptiRAG (Router+PTM) ({row['accuracy']}%, ${row['cost_per_1k']:.3f})")

    plt.title("OptiRAG-lite: Accuracy vs. Simulated Cost per 1k Queries", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Simulated Cost per 1,000 Queries ($)", fontsize=11, labelpad=8)
    plt.ylabel("Accuracy on Test Split (%)", fontsize=11, labelpad=8)
    plt.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_pareto = config.RESULTS_DIR / "pareto.png"
    plt.savefig(out_pareto, dpi=300)
    plt.close()
    print(f"Saved Pareto frontier plot: {out_pareto}")

    # 5. Generate Entity Survival Bar Chart (results/entity_survival.png)
    survival_csv = config.RESULTS_DIR / "ptm_survival.csv"
    if survival_csv.is_file():
        print("\nGenerating Entity Survival bar chart (results/entity_survival.png)...")
        surv_data = {}
        with open(survival_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                surv_data[r["variant"]] = r

        categories = ["Overall", "Number", "Date", "Currency", "Negation", "Gold Answer"]
        var_keys = ["none", "ptm", "ptm_ent"]
        colors = ["#4a90e2", "#50e3c2", "#b8e986"]

        x = np.arange(len(categories))
        width = 0.25

        plt.figure(figsize=(10, 5.5), dpi=300)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        for i, (var, col) in enumerate(zip(var_keys, colors)):
            if var in surv_data:
                row = surv_data[var]
                vals = [
                    float(row["entity_survival_rate"]) * 100,
                    float(row["number_survival_rate"]) * 100,
                    float(row["date_survival_rate"]) * 100,
                    float(row["currency_survival_rate"]) * 100,
                    float(row["negation_survival_rate"]) * 100,
                    float(row["gold_survival_rate"]) * 100,
                ]
                offset = (i - 1) * width
                plt.bar(x + offset, vals, width, label=f"Variant: {var.upper()}", color=col, edgecolor="black", alpha=0.85)

        plt.title("Entity & Gold Retention Across Context Compression Variants (N=100)", fontsize=13, fontweight="bold", pad=12)
        plt.ylabel("Retention Rate (%)", fontsize=11)
        plt.xticks(x, categories, fontsize=10)
        plt.ylim(0, 110)
        plt.legend(loc="lower left", fontsize=9.5)
        plt.grid(True, linestyle=":", alpha=0.6, axis="y")
        plt.tight_layout()

        out_ent = config.RESULTS_DIR / "entity_survival.png"
        plt.savefig(out_ent, dpi=300)
        plt.close()
        print(f"Saved Entity Survival plot: {out_ent}")

    # Print summary table to console
    print("\n" + "=" * 90)
    print(f"{'System':<34} | {'Acc (%)':<9} | {'95% CI':<14} | {'F1':<7} | {'Cost/1k ($)':<12} | {'Latency':<8}")
    print("-" * 90)
    for r in summary_rows:
        ci_str = f"[{r['accuracy_ci_low']:.1f}, {r['accuracy_ci_high']:.1f}]"
        print(f"{r['system']:<34} | {r['accuracy']:<9.1f} | {ci_str:<14} | {r['f1']:<7.3f} | ${r['cost_per_1k']:<11.4f} | {r['mean_latency_s']:<8.2f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
