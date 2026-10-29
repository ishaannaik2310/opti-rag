"""
Measure PTM protection statistics across the top-5 retrieved contexts
for the first 20 HotpotQA train questions.
Uses local embeddings only, ZERO API calls.
"""

import sys
import os
from data import load_dataset_samples, get_train_test_split
from retrieve import Retriever
import ptm

# Ensure UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    print("=" * 80)
    print("PTM HEADROOM ANALYSIS ON TOP-5 RETRIEVED CONTEXT (FIRST 20 TRAIN QUESTIONS)")
    print("=" * 80)

    samples = load_dataset_samples()
    train_samples, _ = get_train_test_split(samples)
    subset = train_samples[:20]

    retriever = Retriever()

    results = []
    kind_totals = {"number": 0, "date": 0, "currency": 0, "percent": 0, "negation": 0}

    print(f"{'#':<3} | {'Question ID':<10} | {'Total Words':<11} | {'Protected W':<11} | {'% Protected':<11} | Top Spans Found")
    print("-" * 80)

    for idx, item in enumerate(subset, 1):
        qid = item["id"]
        question = item["question"]
        chunks = retriever.retrieve(question, k=5)

        # Build context
        context = "\n\n".join(f"[{title}]\n{text}" for text, title, _ in chunks)

        stats = ptm.protection_stats(context)
        tokens = ptm.protected_tokens(context)
        spans_text = ptm.protected_spans_text(context)

        for k in kind_totals:
            kind_totals[k] += stats["by_kind"].get(k, 0)

        results.append({
            "idx": idx,
            "qid": qid,
            "total_words": stats["total_words"],
            "protected_words": stats["protected_words"],
            "pct": stats["protected_word_pct"],
            "sample_spans": spans_text[:4],
        })

        sample_str = ", ".join(spans_text[:3])
        if len(spans_text) > 3:
            sample_str += f" (+{len(spans_text)-3} more)"
        print(f"{idx:<3} | {qid[:10]:<10} | {stats['total_words']:<11} | {stats['protected_words']:<11} | {stats['protected_word_pct']:>6.2f}%    | {sample_str[:28]}")

    avg_words = sum(r["total_words"] for r in results) / len(results)
    avg_prot_words = sum(r["protected_words"] for r in results) / len(results)
    avg_pct = sum(r["pct"] for r in results) / len(results)

    print("=" * 80)
    print("PROTECTION & COMPRESSION HEADROOM SUMMARY:")
    print(f"  Questions Evaluated            : {len(results)}")
    print(f"  Average Context Word Count     : {avg_words:.1f} words (~{avg_words * 1.3:.0f} tokens)")
    print(f"  Average Protected Words        : {avg_prot_words:.1f} words")
    print(f"  Average Protected Word Ratio   : {avg_pct:.2f}%")
    print("-" * 80)
    print("  Protected Entity Breakdown (Total Spans across 20 contexts):")
    for k, v in kind_totals.items():
        print(f"    - {k.capitalize():<12}: {v} spans")
    print("-" * 80)
    print("  Compression Feasibility:")
    print(f"    - Target Compression Rate    : 50.0% (keep 50% of tokens)")
    print(f"    - Guaranteed Protected Floor : {avg_pct:.2f}% of words force-kept")
    print(f"    - Flexible Compressible Room : {100.0 - avg_pct:.2f}% eligible for pruning")
    print(f"    - Headroom Margin to Target  : {50.0 - avg_pct:.2f}% margin safely above floor")
    print("=" * 80)


if __name__ == "__main__":
    main()
