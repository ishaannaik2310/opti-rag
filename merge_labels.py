"""
OptiRAG-lite Label Merger.
Joins SMALL and LARGE tier labeling results into results/labels_{split}.jsonl,
computes Content-Token F1, exact match, and correctness flags, adds retrieval coverage flags
(gold_all_in_top5, gold_any_in_top5), and outputs the 2x2 error matrix.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from data import load_dataset_samples
import config
import metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Merge SMALL and LARGE labels into a single unified split file.")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split to merge.")
    return parser.parse_args()


def main():
    args = parse_args()
    split = args.split.lower()

    small_path = config.RESULTS_DIR / f"labels_small_{split}.jsonl"
    large_path = config.RESULTS_DIR / f"labels_large_{split}.jsonl"
    out_path = config.RESULTS_DIR / f"labels_{split}.jsonl"

    if not small_path.is_file() or not large_path.is_file():
        print(f"Error: Required tier label files do not exist:\n  {small_path}\n  {large_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"MERGING OPTIRAG LABELS: SPLIT={split.upper()}")
    print("=" * 70)

    # 1. Load dataset samples to obtain gold supporting fact titles
    all_samples = load_dataset_samples()
    samples_by_id = {s["id"]: s for s in all_samples}

    # 2. Read Small tier records
    small_records = {}
    with open(small_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                small_records[rec["question_id"]] = rec

    # 3. Read Large tier records
    large_records = {}
    with open(large_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                large_records[rec["question_id"]] = rec

    common_ids = [qid for qid in small_records if qid in large_records]
    print(f"Loaded Small: {len(small_records)} | Loaded Large: {len(large_records)} | Overlap: {len(common_ids)}")

    # 4. Merge and compute metrics
    merged_rows = []
    matrix = {
        "both_correct": 0,
        "small_only": 0,
        "large_only": 0,  # Hard = 1
        "both_wrong": 0,
    }

    with open(out_path, "w", encoding="utf-8") as out_f:
        for qid in common_ids:
            s_rec = small_records[qid]
            l_rec = large_records[qid]
            gold = s_rec["gold_answer"]

            small_ans = s_rec["answer"]
            large_ans = l_rec["answer"]

            # Compute Content-Token F1 and correctness exclusively at merge time
            small_f1 = metrics.compute_f1(small_ans, gold)
            large_f1 = metrics.compute_f1(large_ans, gold)

            small_corr = metrics.is_correct(small_ans, gold)
            large_corr = metrics.is_correct(large_ans, gold)

            # Hard = 1 iff Small is wrong and Large is correct
            hard = 1 if (not small_corr and large_corr) else 0

            if small_corr and large_corr:
                matrix["both_correct"] += 1
            elif small_corr and not large_corr:
                matrix["small_only"] += 1
            elif not small_corr and large_corr:
                matrix["large_only"] += 1
            else:
                matrix["both_wrong"] += 1

            # Compute retrieval coverage of gold supporting facts
            sample_info = samples_by_id.get(qid, {})
            gold_supp_titles = set(sample_info.get("supporting_facts", {}).get("title", []))
            retrieved_titles = {title for title, _ in s_rec.get("retrieved_chunks", [])}

            gold_any_in_top5 = bool(gold_supp_titles & retrieved_titles)
            gold_all_in_top5 = gold_supp_titles.issubset(retrieved_titles) if gold_supp_titles else False

            merged_record = {
                "question_id": qid,
                "question": s_rec["question"],
                "gold_answer": gold,
                "retrieved_chunks": s_rec["retrieved_chunks"],
                "prompt_tokens": s_rec["prompt_tokens"],
                # Retrieval coverage flags
                "gold_any_in_top5": gold_any_in_top5,
                "gold_all_in_top5": gold_all_in_top5,
                # Small tier
                "small_answer": small_ans,
                "small_in_tokens": s_rec["in_tokens"],
                "small_out_tokens": s_rec["out_tokens"],
                "small_latency_s": s_rec["latency_s"],
                "small_f1": small_f1,
                "small_correct": small_corr,
                # Large tier
                "large_answer": large_ans,
                "large_in_tokens": l_rec["in_tokens"],
                "large_out_tokens": l_rec["out_tokens"],
                "large_latency_s": l_rec["latency_s"],
                "large_f1": large_f1,
                "large_correct": large_corr,
                # Router label
                "hard": hard,
            }

            out_f.write(json.dumps(merged_record) + "\n")
            merged_rows.append(merged_record)

    total = len(merged_rows)
    print("\n" + "=" * 70)
    print(f"MERGED LABEL MATRIX & SUMMARY (N={total}):")
    print("=" * 70)
    if total > 0:
        small_acc = (matrix["both_correct"] + matrix["small_only"]) / total * 100
        large_acc = (matrix["both_correct"] + matrix["large_only"]) / total * 100
        hard_rate = matrix["large_only"] / total * 100

        print(f"Both Correct (Easy)     : {matrix['both_correct']} ({matrix['both_correct']/total*100:.1f}%)")
        print(f"Small-Only Correct      : {matrix['small_only']} ({matrix['small_only']/total*100:.1f}%)")
        print(f"Large-Only Correct (Hard): {matrix['large_only']} ({hard_rate:.1f}%) -> Target Class 1")
        print(f"Both Wrong (Unsolvable) : {matrix['both_wrong']} ({matrix['both_wrong']/total*100:.1f}%)")
        print("-" * 70)
        print(f"Small Model Accuracy    : {small_acc:.1f}%")
        print(f"Large Model Accuracy    : {large_acc:.1f}%")
        print(f"Hard Label Rate (Class 1): {hard_rate:.1f}%")

        any_cov = sum(1 for r in merged_rows if r["gold_any_in_top5"]) / total * 100
        all_cov = sum(1 for r in merged_rows if r["gold_all_in_top5"]) / total * 100
        print(f"Retrieval Recall (Any Gold In Top-5): {any_cov:.1f}%")
        print(f"Retrieval Recall (All Gold In Top-5): {all_cov:.1f}%")
    print("=" * 70)
    print(f"Output saved to: {out_path}\n")


if __name__ == "__main__":
    main()
