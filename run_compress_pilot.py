"""
Run Context Compression Pilot on First 20 Train Questions.
Compares 3 variants: 'none', 'ptm', 'ptm_ent' at rate=0.5.
Saves results to results/compress_pilot.json.
Uses local embeddings and local CPU BERT only (ZERO API calls).
"""

import os
import sys
import json
import time
from typing import Any, Dict, List

import torch

# Ensure UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_num_threads(2)

import config
from data import load_dataset_samples, get_train_test_split
from retrieve import Retriever
import compress
import compress_metrics
import ptm


def main():
    print("=" * 80)
    print("OPTIRAG CONTEXT COMPRESSION PILOT: 20 TRAIN QUESTIONS (RATE=0.5)")
    print("Variants: 'none' (Baseline) | 'ptm' (Protected Entities) | 'ptm_ent' (PTM + Named Entities)")
    print("=" * 80)

    samples = load_dataset_samples()
    train_samples, _ = get_train_test_split(samples)
    subset = train_samples[:20]

    retriever = Retriever()

    variants = ["none", "ptm", "ptm_ent"]
    results_by_variant = {v: [] for v in variants}

    # Store 2 rich examples for side-by-side comparison
    side_by_side_examples = []

    for q_idx, item in enumerate(subset, 1):
        qid = item["id"]
        question = item["question"]
        gold_answer = item["answer"]

        # Retrieve top-5 chunks: list of (text, title, score)
        chunks = retriever.retrieve(question, k=config.TOP_K)
        
        # Build original uncompressed context
        orig_context = "\n\n".join(f"[{i}] Title: {title}\n{text}" for i, (text, title, _) in enumerate(chunks, 1))

        print(f"Processing [{q_idx:2d}/20] ID: {qid[:10]} | Q: \"{question[:50]}...\"")

        for var in variants:
            res = compress.compress_context(chunks, variant=var, rate=config.COMPRESSION_RATE)
            comp_context = res["compressed_context"]

            # Entity survival
            ent_stats = compress_metrics.entity_survival_rate(orig_context, comp_context)

            # Gold answer survival
            gold_surv = compress_metrics.gold_answer_survival(orig_context, comp_context, gold_answer)

            record = {
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
            results_by_variant[var].append(record)

            if var == "ptm" and len(side_by_side_examples) < 2 and not res["bypassed"]:
                # Save first chunk original vs compressed
                first_orig_chunk = chunks[0][0]
                # Extract first compressed chunk text
                first_comp_chunk = comp_context.split("\n\n")[0].split("\n", 1)[1] if "\n" in comp_context.split("\n\n")[0] else comp_context.split("\n\n")[0]
                side_by_side_examples.append({
                    "question_id": qid,
                    "title": chunks[0][1],
                    "original": first_orig_chunk,
                    "compressed": first_comp_chunk,
                    "protected_tokens": ptm.protected_tokens(first_orig_chunk),
                })

    # Summary per variant
    summary_table = {}
    print("\n" + "=" * 80)
    print("COMPRESSION PILOT EVALUATION SUMMARY (N=20)")
    print("=" * 80)

    for var in variants:
        records = results_by_variant[var]
        bypassed_count = sum(1 for r in records if r["bypassed"])
        mean_orig = sum(r["original_tokens"] for r in records) / len(records)
        mean_comp = sum(r["compressed_tokens"] for r in records) / len(records)
        mean_ratio = sum(r["ratio"] for r in records) / len(records)
        mean_sec = sum(r["seconds"] for r in records) / len(records)

        # Entity survival totals
        total_ent_spans = sum(r["entity_survival"]["total_count"] for r in records)
        kept_ent_spans = sum(r["entity_survival"]["kept_count"] for r in records)
        overall_ent_rate = (kept_ent_spans / total_ent_spans) if total_ent_spans > 0 else 1.0

        # Per-kind survival
        kind_stats = {}
        for k in ptm.KINDS:
            k_total = sum(r["entity_survival"]["by_kind"].get(k, {}).get("total", 0) for r in records)
            k_kept = sum(r["entity_survival"]["by_kind"].get(k, {}).get("kept", 0) for r in records)
            k_rate = (k_kept / k_total) if k_total > 0 else 1.0
            kind_stats[k] = {"kept": k_kept, "total": k_total, "rate": round(k_rate, 4)}

        # Gold survival
        at_risk_count = sum(1 for r in records if r["gold_survival"] is not None)
        kept_gold_count = sum(1 for r in records if r["gold_survival"] is True)
        gold_rate = (kept_gold_count / at_risk_count) if at_risk_count > 0 else 1.0

        summary_table[var] = {
            "bypassed": bypassed_count,
            "mean_original_tokens": round(mean_orig, 1),
            "mean_compressed_tokens": round(mean_comp, 1),
            "mean_ratio": round(mean_ratio, 4),
            "mean_seconds": round(mean_sec, 3),
            "entity_survival_rate": round(overall_ent_rate, 4),
            "entity_kept": kept_ent_spans,
            "entity_total": total_ent_spans,
            "by_kind": kind_stats,
            "gold_at_risk": at_risk_count,
            "gold_kept": kept_gold_count,
            "gold_survival_rate": round(gold_rate, 4),
        }

        print(f"\nVARIANT: {var.upper()}")
        print(f"  Bypassed Below {config.BYPASS_TOKENS} Tokens : {bypassed_count} / {len(records)} ({bypassed_count/len(records)*100:.1f}%)")
        print(f"  Mean Tokens (Original -> Comp) : {mean_orig:.1f} -> {mean_comp:.1f} (Ratio: {mean_ratio*100:.1f}%)")
        print(f"  Mean Compression Latency       : {mean_sec:.3f}s per context")
        print(f"  Overall Entity Survival Rate   : {overall_ent_rate*100:.1f}% ({kept_ent_spans}/{total_ent_spans} spans)")
        print("  Survival By Kind:")
        for k, ks in kind_stats.items():
            print(f"    - {k.capitalize():<10}: {ks['rate']*100:>5.1f}% ({ks['kept']}/{ks['total']})")
        print(f"  Gold Answer Survival           : {gold_rate*100:.1f}% ({kept_gold_count}/{at_risk_count} preserved when present)")

    # Print Side-by-Side Examples
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE EXAMPLES FOR 'PTM' VARIANT (ORIGINAL VS COMPRESSED)")
    print("=" * 80)

    for i, ex in enumerate(side_by_side_examples, 1):
        print(f"\n--- EXAMPLE {i} (Title: {ex['title']}) ---")
        print(f"Protected Tokens Force-Kept: {ex['protected_tokens']}")
        print(f"\n[ORIGINAL CHUNK TEXT]:\n{ex['original']}")
        print(f"\n[PTM COMPRESSED CHUNK TEXT]:\n{ex['compressed']}")
        print("-" * 80)

    # Save to results/compress_pilot.json
    out_data = {
        "summary": summary_table,
        "side_by_side_examples": side_by_side_examples,
        "records": results_by_variant,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = config.RESULTS_DIR / "compress_pilot.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nResults successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
