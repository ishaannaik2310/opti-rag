"""
OptiRAG-lite Router Training & Evaluation Pipeline.

Trains and evaluates routers across all feature variants ('query-only', 'telemetry-only', 'both')
and models (Heuristic, Logistic Regression with 5-fold CV, and PyTorch MLP).

STRICT RULES:
- Only train split (results/labels_train.jsonl) is used for training, feature scaling, and tuning C.
- Test split (results/labels_test.jsonl) is used strictly once for final evaluation.
- Saves predicted P(hard) for test queries to results/router_test_probs.csv.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_num_threads(2)

import config
import features
from retrieve import Retriever
from router import HeuristicRouter, LogisticRouter, MLPRouter, tune_logistic_cv


def load_labeled_split(split: str) -> List[Dict[str, Any]]:
    path = config.RESULTS_DIR / f"labels_{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing labeled split file: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_features_for_split(
    records: List[Dict[str, Any]],
    variant: str,
    retriever: Retriever,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for r in records:
        q = r["question"]
        chunks = r["retrieved_chunks"]  # list of [title, score]
        feat = features.extract_features(q, chunks, variant=variant, retriever=retriever)
        X_list.append(feat)
        y_list.append(int(r["hard"]))
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


def main():
    print("=" * 80)
    print("STEP 4: ROUTER TRAINING & EVALUATION (STRICT TRAIN-ONLY TUNING)")
    print("=" * 80)

    train_records = load_labeled_split("train")
    test_records = load_labeled_split("test")

    print(f"Loaded Train: {len(train_records)} | Loaded Test: {len(test_records)}")

    y_train = np.array([int(r["hard"]) for r in train_records], dtype=int)
    y_test = np.array([int(r["hard"]) for r in test_records], dtype=int)

    n_pos_train = int(np.sum(y_train == 1))
    n_pos_test = int(np.sum(y_test == 1))
    print(f"Train hard=1 count: {n_pos_train}/{len(y_train)} ({n_pos_train/len(y_train)*100:.1f}%)")
    print(f"Test hard=1 count:  {n_pos_test}/{len(y_test)} ({n_pos_test/len(y_test)*100:.1f}%)")

    allow_mlp = n_pos_train >= 15
    if not allow_mlp:
        print("[NOTE] Train has < 15 hard=1 labels. Skipping PyTorch MLP and using Logistic Regression only.")

    retriever = Retriever()

    feature_variants = ["query-only", "telemetry-only", "both"]
    models = ["logistic"]
    if allow_mlp:
        models.append("mlp")

    results_table = []
    test_probs_dict = {
        "question_id": [r["question_id"] for r in test_records],
        "hard_gold": y_test.tolist(),
    }

    # 1. Baseline: Heuristic Router
    print("\n--- BASELINE: HEURISTIC ROUTER ---")
    h_router = HeuristicRouter(score_threshold=0.70)
    train_queries = [r["question"] for r in train_records]
    train_scores = [float(r["retrieved_chunks"][0][1]) if r["retrieved_chunks"] else 0.0 for r in train_records]
    test_queries = [r["question"] for r in test_records]
    test_scores = [float(r["retrieved_chunks"][0][1]) if r["retrieved_chunks"] else 0.0 for r in test_records]

    h_train_probs = h_router.predict_proba(train_queries, train_scores)
    h_test_probs = h_router.predict_proba(test_queries, test_scores)

    h_train_auc = float(roc_auc_score(y_train, h_train_probs))
    h_train_pr = float(average_precision_score(y_train, h_train_probs))
    h_test_auc = float(roc_auc_score(y_test, h_test_probs))
    h_test_pr = float(average_precision_score(y_test, h_test_probs))

    print(f"Heuristic Train: AUROC = {h_train_auc:.4f} | PR-AUC = {h_train_pr:.4f}")
    print(f"Heuristic Test : AUROC = {h_test_auc:.4f} | PR-AUC = {h_test_pr:.4f}")

    results_table.append({
        "variant": "heuristic",
        "model": "rule-based",
        "cv_auroc_mean": h_train_auc,
        "cv_auroc_std": 0.0,
        "cv_prauc_mean": h_train_pr,
        "cv_prauc_std": 0.0,
        "test_auroc": h_test_auc,
        "test_prauc": h_test_pr,
    })
    test_probs_dict["prob_heuristic"] = h_test_probs.tolist()

    # 2. Extract and cache features per variant
    feature_data = {}
    for var in feature_variants:
        print(f"\nExtracting features for variant: '{var}'...")
        X_tr, _ = extract_features_for_split(train_records, var, retriever)
        X_te, _ = extract_features_for_split(test_records, var, retriever)
        print(f"  Train shape: {X_tr.shape} | Test shape: {X_te.shape}")
        feature_data[var] = (X_tr, X_te)

    # 3. Train & Evaluate Models across Feature Variants
    for var in feature_variants:
        X_tr, X_te = feature_data[var]
        dim = X_tr.shape[1]

        # --- Logistic Regression ---
        print(f"\n=== VARIANT: {var.upper()} | MODEL: LOGISTIC REGRESSION (dim={dim}) ===")
        best_c, cv_info = tune_logistic_cv(X_tr, y_train, n_splits=5)
        cv_best = cv_info[best_c]
        print(f"  Best C (by 5-fold CV): {best_c}")
        print(f"  5-Fold CV AUROC: {cv_best['mean_auc']:.4f} +/- {cv_best['std_auc']:.4f}")
        print(f"  5-Fold CV PR-AUC: {cv_best['mean_pr_auc']:.4f} +/- {cv_best['std_pr_auc']:.4f}")

        # Fit final model on all train data with best C
        clf = LogisticRouter(C=best_c).fit(X_tr, y_train)
        test_probs = clf.predict_proba(X_te)
        test_auc = float(roc_auc_score(y_test, test_probs))
        test_pr = float(average_precision_score(y_test, test_probs))
        print(f"  Test AUROC: {test_auc:.4f} | Test PR-AUC: {test_pr:.4f}")

        results_table.append({
            "variant": var,
            "model": "logistic",
            "best_param": f"C={best_c}",
            "cv_auroc_mean": cv_best["mean_auc"],
            "cv_auroc_std": cv_best["std_auc"],
            "cv_prauc_mean": cv_best["mean_pr_auc"],
            "cv_prauc_std": cv_best["std_pr_auc"],
            "test_auroc": test_auc,
            "test_prauc": test_pr,
        })
        col_name = f"prob_{var.replace('-', '_')}_logistic"
        test_probs_dict[col_name] = test_probs.tolist()

        # --- PyTorch MLP ---
        if allow_mlp:
            print(f"\n=== VARIANT: {var.upper()} | MODEL: PYTORCH MLP (dim={dim}) ===")
            mlp = MLPRouter(in_features=dim, hidden_dim=16, epochs=60).fit(X_tr, y_train)
            mlp_test_probs = mlp.predict_proba(X_te)
            mlp_test_auc = float(roc_auc_score(y_test, mlp_test_probs))
            mlp_test_pr = float(average_precision_score(y_test, mlp_test_probs))
            print(f"  Test AUROC: {mlp_test_auc:.4f} | Test PR-AUC: {mlp_test_pr:.4f}")

            results_table.append({
                "variant": var,
                "model": "mlp",
                "best_param": "hidden=16,drop=0.3",
                "cv_auroc_mean": 0.0,
                "cv_auroc_std": 0.0,
                "cv_prauc_mean": 0.0,
                "cv_prauc_std": 0.0,
                "test_auroc": mlp_test_auc,
                "test_prauc": mlp_test_pr,
            })
            col_name_mlp = f"prob_{var.replace('-', '_')}_mlp"
            test_probs_dict[col_name_mlp] = mlp_test_probs.tolist()

    # 4. Save results/router_test_probs.csv
    out_probs_csv = config.RESULTS_DIR / "router_test_probs.csv"
    with open(out_probs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = list(test_probs_dict.keys())
        writer.writerow(headers)
        n_rows = len(test_probs_dict["question_id"])
        for i in range(n_rows):
            row = [test_probs_dict[h][i] for h in headers]
            writer.writerow(row)
    print(f"\nSaved test probabilities to: {out_probs_csv}")

    # 5. Print Final Summary Table
    print("\n" + "=" * 80)
    print("ROUTER PERFORMANCE SUMMARY:")
    print("=" * 80)
    print(f"{'Variant':<16} | {'Model':<10} | {'CV AUROC (Train)':<20} | {'CV PR-AUC (Train)':<20} | {'Test AUROC':<10} | {'Test PR-AUC':<10}")
    print("-" * 80)
    for r in results_table:
        cv_auc_str = f"{r['cv_auroc_mean']:.3f} +/- {r['cv_auroc_std']:.3f}" if r["cv_auroc_mean"] > 0 else "N/A"
        cv_pr_str = f"{r['cv_prauc_mean']:.3f} +/- {r['cv_prauc_std']:.3f}" if r["cv_prauc_mean"] > 0 else "N/A"
        print(f"{r['variant']:<16} | {r['model']:<10} | {cv_auc_str:<20} | {cv_pr_str:<20} | {r['test_auroc']:<10.3f} | {r['test_prauc']:<10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
