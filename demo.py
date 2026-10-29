"""
OptiRAG-lite Offline Interactive Demonstration Script.

Runs entirely from disk cache and pre-computed test splits in under 1 second.
Makes ZERO external LLM API calls (verified against usage counter files).

Usage:
    python demo.py [--idx INDEX] [--qid QUESTION_ID]
Example:
    python demo.py --idx 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ptm

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_demo_data() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Load test records, compressed contexts, and precomputed router probabilities."""
    with open("results/labels_test.jsonl", "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    with open("results/compressed_test.jsonl", "r", encoding="utf-8") as f:
        compressed_dict = {json.loads(line)["question_id"]: json.loads(line) for line in f if line.strip()}

    router_probs = {}
    with open("results/router_test_probs.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            router_probs[row["question_id"]] = {
                "prob_telemetry": float(row["prob_telemetry_only_logistic"]),
                "prob_query": float(row["prob_query_only_logistic"]),
                "prob_both": float(row["prob_both_logistic"]),
                "hard_gold": int(float(row["hard_gold"])),
            }

    return test_records, compressed_dict, router_probs


def highlight_ptm_spans(text: str) -> str:
    """Highlight protected spans in text by enclosing them in square brackets."""
    spans = ptm.find_protected_spans(text)
    if not spans:
        return text

    # Sort spans in reverse order so character offsets remain valid during insertion
    spans = sorted(spans, key=lambda s: s[0], reverse=True)
    out = text
    for start, end, kind in spans:
        span_str = out[start:end]
        # Avoid double brackets if already enclosed
        if not (start > 0 and out[start - 1] == "[" and end < len(out) and out[end] == "]"):
            out = out[:start] + f"[{span_str}]" + out[end:]
    return out


def run_demo(idx: int = 1, qid: Optional[str] = None):
    test_records, compressed_dict, router_probs = load_demo_data()

    if qid is not None:
        matches = [r for r in test_records if r["question_id"] == qid]
        if not matches:
            print(f"Error: Question ID '{qid}' not found in test split.")
            sys.exit(1)
        record = matches[0]
    else:
        if idx < 0 or idx >= len(test_records):
            print(f"Error: Index {idx} out of range (0 to {len(test_records) - 1}).")
            sys.exit(1)
        record = test_records[idx]

    qid = record["question_id"]
    question = record["question"]
    gold_answer = record["gold_answer"]
    chunks = record["retrieved_chunks"]

    comp_info = compressed_dict.get(qid, {})
    comp_context = comp_info.get("compressed_context", "")
    comp_chunks = comp_info.get("retrieved_chunks", [])

    # Router calibrated threshold (Method A: Target 30% Escalation on Train CV)
    ROUTER_THRESHOLD = 0.520
    probs = router_probs.get(qid, {"prob_telemetry": 0.50, "hard_gold": record.get("hard", 0)})
    p_hard = probs["prob_telemetry"]
    decision = "ESCALATE -> LARGE TIER" if p_hard >= ROUTER_THRESHOLD else "ROUTE -> SMALL TIER"

    # Reconstruct uncompressed context layout from full retrieved texts
    uncompressed_blocks = [
        f"[{i}] Title: {c[1]}\n{c[0]}"
        for i, c in enumerate(comp_chunks, 1)
    ]
    raw_context = "\n\n".join(uncompressed_blocks)

    # Protected tokens
    protected_toks = ptm.protected_tokens(raw_context)

    print("=" * 85)
    print("OPTIRAG-LITE: OFFLINE SYSTEM INFERENCE DEMONSTRATION")
    print("=" * 85)
    print(f"Question ID : {qid}")
    print(f"Question    : {question}")
    print(f"Gold Answer : {gold_answer}")
    print("-" * 85)

    # 1. Retrieval
    chunk_text_map = {c[1]: c[0] for c in comp_chunks if len(c) >= 2}

    print("\n[STEP 1: RETRIEVAL TELEMETRY (Dense MiniLM FAISS Top-5)]")
    for i, c in enumerate(chunks, 1):
        if isinstance(c, (list, tuple)):
            if len(c) >= 2 and isinstance(c[1], (int, float)):
                title = str(c[0])
                score = float(c[1])
            elif len(c) >= 3:
                title = str(c[1])
                score = float(c[2])
            else:
                title = str(c[0])
                score = 0.0
        else:
            title = c.get("title", "Unknown")
            score = float(c.get("score", 0.0))

        text_snippet = chunk_text_map.get(title, title)
        snippet = text_snippet[:130].replace("\n", " ")
        print(f"  Rank #{i}: Score = {score:.4f} | Title: '{title}'")
        print(f"           Snippet: \"{snippet}...\"")

    all_in_top5 = record.get("gold_all_in_top5", False)
    print(f"\n  Retrieval Completeness (`gold_all_in_top5`): {all_in_top5} "
          f"({'All gold documents retrieved' if all_in_top5 else 'Missing supporting facts'})")

    # 2. PTM Mask and Context Compression
    print("\n" + "-" * 85)
    print("[STEP 2: PROTECTED TOKEN MASK (PTM) & CONTEXT COMPRESSION]")
    print(f"  Protected Entity Tokens Extracted by PTM ({len(protected_toks)} unique tokens):")
    print(f"    {protected_toks[:18]}{' ...' if len(protected_toks) > 18 else ''}")

    orig_tokens = comp_info.get("original_tokens", len(raw_context.split()))
    comp_tokens = comp_info.get("compressed_tokens", len(comp_context.split()))
    ratio = comp_info.get("ratio", 1.0)
    savings = (1.0 - ratio) * 100.0

    print(f"\n  Compression Ratio: {orig_tokens} tokens -> {comp_tokens} tokens ({savings:.1f}% token reduction)")
    print("\n  Original Context (with PTM Protected Tokens in [brackets]):")
    highlighted_raw = highlight_ptm_spans(raw_context)
    for line in highlighted_raw.split("\n")[:12]:
        print(f"    {line}")
    if len(highlighted_raw.split("\n")) > 12:
        print("    [... remaining context truncated for display ...]")

    print("\n  PTM-Compressed Context (Entities Preserved by Construction):")
    for line in comp_context.split("\n")[:10]:
        print(f"    {line}")
    if len(comp_context.split("\n")) > 10:
        print("    [... remaining compressed context truncated for display ...]")

    # 3. Router Hardness Assessment
    print("\n" + "-" * 85)
    print("[STEP 3: TELEMETRY-GUIDED CASCADE ROUTER DECISION]")
    print(f"  Router Architecture    : Telemetry-Only Logistic Regression (8 retrieval features)")
    print(f"  Predicted P(hard = 1)  : {p_hard:.4f}")
    print(f"  Train-Calibrated Tau   : {ROUTER_THRESHOLD:.3f} (Calibrated on train CV to budget 30% Large)")
    print(f"  Routing Decision       : {decision}")
    print(f"  Ground Truth Phenotype : {'HARD (Class 1: Small fails, Large succeeds)' if record.get('hard') == 1 else 'EASY or BOTH WRONG'}")

    # 4. Answers from Both Tiers
    print("\n" + "-" * 85)
    print("[STEP 4: MODEL ANSWERS & ECONOMICS]")
    small_ans = record.get("small_answer", "N/A")
    small_corr = record.get("small_correct", False)
    small_lat = record.get("small_latency_s", 0.0)
    small_cost = (record.get("small_in_tokens", 0) / 1e6 * 0.05) + (record.get("small_out_tokens", 0) / 1e6 * 0.08)

    large_ans = record.get("large_answer", "N/A")
    large_corr = record.get("large_correct", False)
    large_lat = record.get("large_latency_s", 0.0)
    large_cost = (record.get("large_in_tokens", 0) / 1e6 * 0.30) + (record.get("large_out_tokens", 0) / 1e6 * 2.50)

    print(f"  SMALL TIER (allam-2-7b on Groq, 8B-class proxy):")
    print(f"    Answer     : \"{small_ans}\"")
    print(f"    Correct?   : {small_corr} | Latency: {small_lat:.3f}s | Cost: ${small_cost * 1000:.4f} / 1k")

    print(f"\n  LARGE TIER (gemini-3.5-flash-lite on Google AI Studio):")
    print(f"    Answer     : \"{large_ans}\"")
    print(f"    Correct?   : {large_corr} | Latency: {large_lat:.3f}s | Cost: ${large_cost * 1000:.4f} / 1k")

    # Final summary
    routed_ans = large_ans if "LARGE" in decision else small_ans
    routed_corr = large_corr if "LARGE" in decision else small_corr
    routed_cost = large_cost if "LARGE" in decision else small_cost
    routed_lat = large_lat if "LARGE" in decision else small_lat

    print("\n" + "=" * 85)
    print(f"FINAL DISPATCH: Served by {decision.split('-> ')[1]}")
    print(f"Resulting Output  : \"{routed_ans}\"")
    print(f"Outcome Status    : {'CORRECT (+)' if routed_corr else 'INCORRECT (-)'}")
    print(f"Inference Latency : {routed_lat:.3f}s | Effective Cost: ${routed_cost * 1000:.4f} / 1k queries")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptiRAG-lite Offline Demo")
    parser.add_argument("--idx", type=int, default=1, help="Index of test query (0-99)")
    parser.add_argument("--qid", type=str, default=None, help="Optional HotpotQA Question ID")
    args = parser.parse_args()

    run_demo(idx=args.idx, qid=args.qid)
