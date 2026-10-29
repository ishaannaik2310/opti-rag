"""
OptiRAG-lite Compressed Test Context Generator & Survival Analysis.

1. Evaluates all 3 compression variants ('none', 'ptm', 'ptm_ent') across all 100 test contexts (CPU only).
2. Computes overall and per-kind entity survival rates + gold-answer survival.
3. Saves full comparison table to results/ptm_survival.csv.
4. Generates results/compressed_test.jsonl using the selected variant ('ptm') for end-to-end evaluation.
"""

import csv
import json
import os
import sys
import time

import torch

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_num_threads(2)

import config
from data import get_train_test_split, load_dataset_samples
from retrieve import Retriever
import compress
import compress_metrics
import ptm


def main():
    print("=" * 70)
    print("STEP 3: COMPRESSING ALL 100 TEST CONTEXTS & EVALUATING PTM SURVIVAL")
    print("=" * 70)

    # 1. Load test samples and retriever
    samples = load_dataset_samples()
    _, test_samples = get_train_test_split(samples)
    assert len(test_samples) == 100, f"Expected 100 test samples, got {len(test_samples)}"

    retriever = Retriever()

    variants = ["none", "ptm", "ptm_ent"]
    variant_records = {v: [] for v in variants}

    chosen_variant = "ptm"
    compressed_test_rows = []

    t0_all = time.time()

    for idx, item in enumerate(test_samples, 1):
        qid = item["id"]
        question = item["question"]
        gold_answer = item["answer"]

        # Retrieve top-5 chunks
        chunks = retriever.retrieve(question, k=config.TOP_K)
        orig_context = "\n\n".join(f"[{i}] Title: {title}\n{text}" for i, (text, title, _) in enumerate(chunks, 1))
        orig_tokens = compress.count_tokens(orig_context)

        # Evaluate each variant
        for var in variants:
            res = compress.compress_context(chunks, variant=var, rate=config.COMPRESSION_RATE)
            comp_context = res["compressed_context"]

            ent_stats = compress_metrics.entity_survival_rate(orig_context, comp_context)
            gold_surv = compress_metrics.gold_answer_survival(orig_context, comp_context, gold_answer)

            rec = {
                "question_id": qid,
                "variant": var,
                "bypassed": res["bypassed"],
                "original_tokens": res["original_tokens"],
                "compressed_tokens": res["compressed_tokens"],
                "ratio": res["ratio"],
                "seconds": res["seconds"],
                "entity_survival": ent_stats,
                "gold_survival": gold_surv,
            }
            variant_records[var].append(rec)

            if var == chosen_variant:
                compressed_test_rows.append({
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "bypassed": res["bypassed"],
                    "original_tokens": res["original_tokens"],
                    "compressed_tokens": res["compressed_tokens"],
                    "ratio": res["ratio"],
                    "seconds": res["seconds"],
                    "variant": chosen_variant,
                    "retrieved_chunks": [(text, title) for text, title, _ in chunks],
                    "compressed_context": comp_context,
                })

        if idx % 20 == 0 or idx == 100:
            elapsed = time.time() - t0_all
            print(f"[{idx:3d}/100] Processed ({elapsed:.1f}s elapsed)")

    # 2. Write results/compressed_test.jsonl
    out_compressed_jsonl = config.RESULTS_DIR / "compressed_test.jsonl"
    with open(out_compressed_jsonl, "w", encoding="utf-8") as f:
        for row in compressed_test_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nSaved compressed test contexts: {out_compressed_jsonl} ({len(compressed_test_rows)} lines)")

    # 3. Compute Summary Table & Write results/ptm_survival.csv
    csv_rows = []
    print("\n" + "=" * 70)
    print("ENTITY & GOLD SURVIVAL EVALUATION (ALL 100 TEST CONTEXTS)")
    print("=" * 70)

    for var in variants:
        recs = variant_records[var]
        bypassed_cnt = sum(1 for r in recs if r["bypassed"])
        mean_orig = sum(r["original_tokens"] for r in recs) / len(recs)
        mean_comp = sum(r["compressed_tokens"] for r in recs) / len(recs)
        mean_ratio = sum(r["ratio"] for r in recs) / len(recs)
        mean_sec = sum(r["seconds"] for r in recs) / len(recs)

        tot_ent = sum(r["entity_survival"]["total_count"] for r in recs)
        kept_ent = sum(r["entity_survival"]["kept_count"] for r in recs)
        ent_rate = kept_ent / tot_ent if tot_ent > 0 else 1.0

        # By kind
        kind_rates = {}
        for k in ptm.KINDS:
            k_tot = sum(r["entity_survival"]["by_kind"].get(k, {}).get("total", 0) for r in recs)
            k_kept = sum(r["entity_survival"]["by_kind"].get(k, {}).get("kept", 0) for r in recs)
            kind_rates[k] = (k_kept / k_tot) if k_tot > 0 else 1.0

        # Gold survival
        at_risk = sum(1 for r in recs if r["gold_survival"] is not None)
        gold_kept = sum(1 for r in recs if r["gold_survival"] is True)
        gold_rate = gold_kept / at_risk if at_risk > 0 else 1.0

        print(f"\nVARIANT: {var.upper()}")
        print(f"  Bypassed (< 400 tok)     : {bypassed_cnt}/100 ({bypassed_cnt}%)")
        print(f"  Mean Tokens (Orig->Comp) : {mean_orig:.1f} -> {mean_comp:.1f} (Ratio: {mean_ratio*100:.1f}%)")
        print(f"  Mean Latency             : {mean_sec:.3f}s")
        print(f"  Overall Entity Survival  : {ent_rate*100:.1f}% ({kept_ent}/{tot_ent})")
        print(f"  - Number Survival        : {kind_rates.get('number', 1.0)*100:.1f}%")
        print(f"  - Date Survival          : {kind_rates.get('date', 1.0)*100:.1f}%")
        print(f"  - Currency Survival      : {kind_rates.get('currency', 1.0)*100:.1f}%")
        print(f"  - Negation Survival      : {kind_rates.get('negation', 1.0)*100:.1f}%")
        print(f"  - Percent Survival       : {kind_rates.get('percent', 1.0)*100:.1f}%")
        print(f"  Gold Answer Survival     : {gold_rate*100:.1f}% ({gold_kept}/{at_risk} at-risk preserved)")

        csv_rows.append({
            "variant": var,
            "bypassed_count": bypassed_cnt,
            "mean_original_tokens": round(mean_orig, 1),
            "mean_compressed_tokens": round(mean_comp, 1),
            "mean_ratio": round(mean_ratio, 4),
            "mean_latency_s": round(mean_sec, 3),
            "entity_survival_rate": round(ent_rate, 4),
            "entity_kept": kept_ent,
            "entity_total": tot_ent,
            "number_survival_rate": round(kind_rates.get("number", 1.0), 4),
            "date_survival_rate": round(kind_rates.get("date", 1.0), 4),
            "currency_survival_rate": round(kind_rates.get("currency", 1.0), 4),
            "negation_survival_rate": round(kind_rates.get("negation", 1.0), 4),
            "percent_survival_rate": round(kind_rates.get("percent", 1.0), 4),
            "gold_at_risk": at_risk,
            "gold_kept": gold_kept,
            "gold_survival_rate": round(gold_rate, 4),
        })

    out_csv = config.RESULTS_DIR / "ptm_survival.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved survival CSV: {out_csv}")


if __name__ == "__main__":
    main()
