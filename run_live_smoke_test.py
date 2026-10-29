"""
Live smoke test execution for SMALL (Groq) and LARGE (Gemini) tiers.
Executes exactly 4 real API calls + 4 cache hits (8 calls total).
Never logs or prints secrets.
"""

import os
import sys
import re
import json
import string
from collections import Counter
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from data import load_dataset_samples, get_train_test_split
from retrieve import Retriever
import config
import llm


def normalize_answer(s: str) -> str:
    """Normalize text for official HotpotQA F1 evaluation."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Standard HotpotQA token F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def format_qa_prompt(question: str, chunks: list) -> str:
    """Build exact prompt as run_labels.py will."""
    context_str = "\n".join([f"[{i+1}] Title: {title}\n{text}" for i, (text, title, _) in enumerate(chunks)])
    return (
        "Answer the question based on the context. If the context does not contain the answer, "
        "say \"I don't know\". Keep your answer concise.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def main():
    print("=" * 80)
    print("OPTIRAG-LITE: LIVE SMOKE TEST (SMALL TIER & LARGE TIER)")
    print("=" * 80)

    # 1. Load train questions #0 and #1
    all_samples = load_dataset_samples()
    train_samples, _ = get_train_test_split(all_samples)
    q0 = train_samples[0]
    q1 = train_samples[1]

    retriever = Retriever()
    chunks0 = retriever.retrieve(q0["question"], k=config.TOP_K)
    chunks1 = retriever.retrieve(q1["question"], k=config.TOP_K)

    prompt0 = format_qa_prompt(q0["question"], chunks0)
    prompt1 = format_qa_prompt(q1["question"], chunks1)

    tiktoken_0 = llm.count_tokens(prompt0)
    tiktoken_1 = llm.count_tokens(prompt1)

    print(f"\n[Question #0]: {q0['question']}")
    print(f"  Gold Answer: {q0['answer']}")
    print(f"  tiktoken Tokens: {tiktoken_0}")

    print(f"\n[Question #1]: {q1['question']}")
    print(f"  Gold Answer: {q1['answer']}")
    print(f"  tiktoken Tokens: {tiktoken_1}")

    # =========================================================================
    # STEP 3: LIVE SMOKE TEST - SMALL TIER (Groq llama-3.1-8b-instant)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: SMALL TIER (Groq llama-3.1-8b-instant) LIVE CALLS")
    print("=" * 80)
    small_tier = config.SMALL_TIER

    # Question #0 (Call 1: live; Call 2: cache hit)
    res_s0_live = llm.generate(small_tier, prompt0, live=True)
    f1_s0 = compute_f1(res_s0_live["text"], q0["answer"])
    print("\n--- Question #0: Pass 1 (Live API Call) ---")
    print(f"  Answer Text : {res_s0_live['text']}")
    print(f"  Gold Answer : {q0['answer']}")
    print(f"  Token F1    : {f1_s0:.4f}")
    print(f"  In / Out Tok: {res_s0_live['in_tokens']} / {res_s0_live['out_tokens']}")
    print(f"  Latency     : {res_s0_live['latency_s']:.2f}s | Cached: {res_s0_live['cached']}")

    res_s0_cached = llm.generate(small_tier, prompt0, live=True)
    print("--- Question #0: Pass 2 (Cache Hit) ---")
    print(f"  Answer Text : {res_s0_cached['text']}")
    print(f"  In / Out Tok: {res_s0_cached['in_tokens']} / {res_s0_cached['out_tokens']}")
    print(f"  Latency     : {res_s0_cached['latency_s']:.4f}s | Cached: {res_s0_cached['cached']}")

    # Question #1 (Call 1: live; Call 2: cache hit)
    res_s1_live = llm.generate(small_tier, prompt1, live=True)
    f1_s1 = compute_f1(res_s1_live["text"], q1["answer"])
    print("\n--- Question #1: Pass 1 (Live API Call) ---")
    print(f"  Answer Text : {res_s1_live['text']}")
    print(f"  Gold Answer : {q1['answer']}")
    print(f"  Token F1    : {f1_s1:.4f}")
    print(f"  In / Out Tok: {res_s1_live['in_tokens']} / {res_s1_live['out_tokens']}")
    print(f"  Latency     : {res_s1_live['latency_s']:.2f}s | Cached: {res_s1_live['cached']}")

    res_s1_cached = llm.generate(small_tier, prompt1, live=True)
    print("--- Question #1: Pass 2 (Cache Hit) ---")
    print(f"  Answer Text : {res_s1_cached['text']}")
    print(f"  In / Out Tok: {res_s1_cached['in_tokens']} / {res_s1_cached['out_tokens']}")
    print(f"  Latency     : {res_s1_cached['latency_s']:.4f}s | Cached: {res_s1_cached['cached']}")

    # =========================================================================
    # STEP 4: LIVE SMOKE TEST - LARGE TIER (Gemini gemini-3.5-flash-lite)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: LARGE TIER (Gemini gemini-3.5-flash-lite) LIVE CALLS")
    print("=" * 80)
    large_tier = config.LARGE_TIER

    # Question #0 (Call 1: live; Call 2: cache hit)
    res_l0_live = llm.generate(large_tier, prompt0, live=True)
    f1_l0 = compute_f1(res_l0_live["text"], q0["answer"])
    print("\n--- Question #0: Pass 1 (Live API Call) ---")
    print(f"  Answer Text : {res_l0_live['text']}")
    print(f"  Gold Answer : {q0['answer']}")
    print(f"  Token F1    : {f1_l0:.4f}")
    print(f"  In / Out Tok: {res_l0_live['in_tokens']} / {res_l0_live['out_tokens']}")
    print(f"  Latency     : {res_l0_live['latency_s']:.2f}s | Cached: {res_l0_live['cached']}")

    res_l0_cached = llm.generate(large_tier, prompt0, live=True)
    print("--- Question #0: Pass 2 (Cache Hit) ---")
    print(f"  Answer Text : {res_l0_cached['text']}")
    print(f"  In / Out Tok: {res_l0_cached['in_tokens']} / {res_l0_cached['out_tokens']}")
    print(f"  Latency     : {res_l0_cached['latency_s']:.4f}s | Cached: {res_l0_cached['cached']}")

    # Question #1 (Call 1: live; Call 2: cache hit)
    res_l1_live = llm.generate(large_tier, prompt1, live=True)
    f1_l1 = compute_f1(res_l1_live["text"], q1["answer"])
    print("\n--- Question #1: Pass 1 (Live API Call) ---")
    print(f"  Answer Text : {res_l1_live['text']}")
    print(f"  Gold Answer : {q1['answer']}")
    print(f"  Token F1    : {f1_l1:.4f}")
    print(f"  In / Out Tok: {res_l1_live['in_tokens']} / {res_l1_live['out_tokens']}")
    print(f"  Latency     : {res_l1_live['latency_s']:.2f}s | Cached: {res_l1_live['cached']}")

    res_l1_cached = llm.generate(large_tier, prompt1, live=True)
    print("--- Question #1: Pass 2 (Cache Hit) ---")
    print(f"  Answer Text : {res_l1_cached['text']}")
    print(f"  In / Out Tok: {res_l1_cached['in_tokens']} / {res_l1_cached['out_tokens']}")
    print(f"  Latency     : {res_l1_cached['latency_s']:.4f}s | Cached: {res_l1_cached['cached']}")

    # =========================================================================
    # STEP 5: CALIBRATION & RE-PROJECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: TOKEN CALIBRATION (API REAL VS TIKTOKEN)")
    print("=" * 80)
    # Groq Calibration
    groq_ratio_0 = res_s0_live["in_tokens"] / tiktoken_0
    groq_ratio_1 = res_s1_live["in_tokens"] / tiktoken_1
    groq_mean_ratio = (groq_ratio_0 + groq_ratio_1) / 2.0

    print(f"Groq (Llama BPE):")
    print(f"  Q#0: Real={res_s0_live['in_tokens']} vs tiktoken={tiktoken_0} -> Ratio = {groq_ratio_0:.4f}")
    print(f"  Q#1: Real={res_s1_live['in_tokens']} vs tiktoken={tiktoken_1} -> Ratio = {groq_ratio_1:.4f}")
    print(f"  Mean Ratio (Groq / tiktoken) = {groq_mean_ratio:.4f}")

    # Gemini Calibration
    gem_ratio_0 = res_l0_live["in_tokens"] / tiktoken_0
    gem_ratio_1 = res_l1_live["in_tokens"] / tiktoken_1
    gem_mean_ratio = (gem_ratio_0 + gem_ratio_1) / 2.0

    print(f"\nGemini (Google BPE):")
    print(f"  Q#0: Real={res_l0_live['in_tokens']} vs tiktoken={tiktoken_0} -> Ratio = {gem_ratio_0:.4f}")
    print(f"  Q#1: Real={res_l1_live['in_tokens']} vs tiktoken={tiktoken_1} -> Ratio = {gem_ratio_1:.4f}")
    print(f"  Mean Ratio (Gemini / tiktoken) = {gem_mean_ratio:.4f}")

    # Re-project for 400 calls
    base_tiktoken_total = 231542
    calibrated_groq_tokens = int(base_tiktoken_total * groq_mean_ratio)
    calibrated_gem_tokens = int(base_tiktoken_total * gem_mean_ratio)

    print("\n--- Re-Projected 400-Call Budget with Calibrated Ratios ---")
    print(f"Original tiktoken Total (400 calls) : {base_tiktoken_total:,} tokens")
    print(f"Calibrated Groq SMALL Total         : {calibrated_groq_tokens:,} tokens ({calibrated_groq_tokens/500000*100:.1f}% of 500k TPD)")
    print(f"Calibrated Gemini LARGE Total       : {calibrated_gem_tokens:,} tokens (~{calibrated_gem_tokens/1000:.1f}k tokens)")
    print(f"Total Requests Needed               : 400 requests per tier")

    # =========================================================================
    # STEP 6: GROQ RATE LIMIT HEADERS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: GROQ RATE-LIMIT HEADERS (.cache/ratelimit_headers.jsonl)")
    print("=" * 80)
    if config.RATELIMIT_LOG_PATH.is_file():
        with open(config.RATELIMIT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Total header log records: {len(lines)}")
        if lines:
            last_record = json.loads(lines[-1])
            print(f"Most Recent Logged Headers (Model: {last_record['model']}):")
            for k, v in sorted(last_record["headers"].items()):
                print(f"  {k}: {v}")
    else:
        print("No rate limit header log found.")

    # =========================================================================
    # STEP 7: FINAL USAGE SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: FINAL USAGE SUMMARY")
    print("=" * 80)
    llm.print_usage_summary()


if __name__ == "__main__":
    main()
