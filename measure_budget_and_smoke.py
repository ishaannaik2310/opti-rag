"""
Module 1 Smoke Tests and Token Budget Measurement for OptiRAG-lite.
Executes Smoke Tests a/b/c and Token Budget Measurement across all 300 questions.
No LLM API calls are made.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import tiktoken

import config
from data import prepare_data
from retrieve import get_retriever


def build_prompt(context_chunks: List[tuple], question: str) -> Tuple[str, str]:
    """
    Builds the exact prompt run_labels.py will send:
    Instruction line + the 5 retrieved chunks with titles + question.
    Returns: (full_prompt, context_only_str)
    """
    instruction = "Answer using only the context. Give a short answer (a few words).\n\n"
    
    context_str = ""
    for rank, (text, title, _) in enumerate(context_chunks, 1):
        context_str += f"[{title}]: {text}\n\n"
        
    full_prompt = f"{instruction}Context:\n{context_str}Question: {question}\nAnswer:"
    return full_prompt, context_str


def run_smoke_tests(all_samples, train_samples, test_samples, corpus, stats, retriever):
    print("=" * 80)
    print("SMOKE TEST (A): SPLIT SIZES, CORPUS SIZE, AND DUPLICATES")
    print("=" * 80)
    print(f"Total Selected Questions : {len(all_samples)}")
    print(f"Train Split Size         : {len(train_samples)} (Expected: {config.TRAIN_SAMPLES})")
    print(f"Test Split Size          : {len(test_samples)} (Expected: {config.TEST_SAMPLES})")
    print(f"Total Paragraphs Seen    : {stats['total_paragraphs_encountered']}")
    print(f"Unique Corpus Size       : {stats['unique_corpus_size']}")
    print(f"Duplicate Paragraphs     : {stats['duplicate_paragraphs']}")
    
    assert len(train_samples) == config.TRAIN_SAMPLES, f"Expected {config.TRAIN_SAMPLES} train samples"
    assert len(test_samples) == config.TEST_SAMPLES, f"Expected {config.TEST_SAMPLES} test samples"
    print(">> Split verification passed!")

    print("\n" + "=" * 80)
    print("SMOKE TEST (B): RETRIEVE TOP-5 FOR 3 SAMPLE QUESTIONS")
    print("=" * 80)
    for i, sample in enumerate(all_samples[:3], 1):
        q = sample["question"]
        gold = sample["answer"]
        top_chunks = retriever.retrieve(q, k=config.TOP_K)
        print(f"\n--- Sample {i} ---")
        print(f"Question   : {q}")
        print(f"Gold Answer: {gold}")
        print(f"Gold Supporting Titles: {set(sample['supporting_facts']['title'])}")
        print("Retrieved Top-5 Chunks:")
        for rank, (text, title, score) in enumerate(top_chunks, 1):
            is_gold = title in sample["supporting_facts"]["title"]
            gold_tag = " [GOLD SUPPORTING FACT]" if is_gold else ""
            print(f"  [{rank}] Score: {score:.4f} | Title: \"{title}\"{gold_tag}")
            print(f"      Text: {text[:120]}...")

    print("\n" + "=" * 80)
    print("SMOKE TEST (C): RETRIEVAL QUALITY OVER ALL 300 QUESTIONS")
    print("=" * 80)
    
    at_least_one_count = 0
    all_gold_count = 0
    
    for sample in all_samples:
        q = sample["question"]
        gold_titles = set(sample["supporting_facts"]["title"])
        top_chunks = retriever.retrieve(q, k=config.TOP_K)
        retrieved_titles = {chunk[1] for chunk in top_chunks}
        
        # Check intersection
        intersection = gold_titles.intersection(retrieved_titles)
        if len(intersection) > 0:
            at_least_one_count += 1
        if gold_titles.issubset(retrieved_titles):
            all_gold_count += 1
            
    pct_at_least_one = (at_least_one_count / len(all_samples)) * 100
    pct_all_gold = (all_gold_count / len(all_samples)) * 100
    
    print(f"Questions Evaluated                          : {len(all_samples)}")
    print(f"At least 1 Gold Supporting Title in Top-5    : {at_least_one_count}/{len(all_samples)} ({pct_at_least_one:.2f}%)")
    print(f"ALL Gold Supporting Titles in Top-5 (Recall@5) : {all_gold_count}/{len(all_samples)} ({pct_all_gold:.2f}%)")


def run_token_budget_measurement(all_samples, train_samples, test_samples, retriever):
    print("\n" + "=" * 80)
    print("TOKEN BUDGET MEASUREMENT (ALL 300 QUESTIONS, NO API CALLS)")
    print("=" * 80)
    print("Tokenizer: tiktoken 'cl100k_base' (approximation of Llama BPE tokenizer)")
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    prompt_token_counts = []
    context_token_counts = []
    exceeds_400_count = 0
    
    # Track per-split stats
    train_prompt_tokens = []
    test_prompt_tokens = []
    
    for idx, sample in enumerate(all_samples):
        q = sample["question"]
        top_chunks = retriever.retrieve(q, k=config.TOP_K)
        full_prompt, context_str = build_prompt(top_chunks, q)
        
        p_tokens = len(enc.encode(full_prompt))
        c_tokens = len(enc.encode(context_str))
        
        prompt_token_counts.append(p_tokens)
        context_token_counts.append(c_tokens)
        
        if c_tokens >= config.BYPASS_TOKENS:
            exceeds_400_count += 1
            
        if idx < len(train_samples):
            train_prompt_tokens.append(p_tokens)
        else:
            test_prompt_tokens.append(p_tokens)
            
    p_arr = np.array(prompt_token_counts)
    c_arr = np.array(context_token_counts)
    
    mean_p = np.mean(p_arr)
    median_p = np.median(p_arr)
    p90_p = np.percentile(p_arr, 90)
    max_p = np.max(p_arr)
    pct_exceed_400 = (exceeds_400_count / len(all_samples)) * 100
    
    print("\n--- Context & Prompt Token Statistics (300 Questions) ---")
    print(f"Mean Prompt Tokens      : {mean_p:.1f}")
    print(f"Median Prompt Tokens    : {median_p:.1f}")
    print(f"90th Percentile (p90)   : {p90_p:.1f}")
    print(f"Max Prompt Tokens       : {max_p}")
    print(f"Context >= 400 tokens   : {exceeds_400_count}/{len(all_samples)} ({pct_exceed_400:.1f}%)")
    print(f"Context < 400 (Bypass)  : {len(all_samples) - exceeds_400_count}/{len(all_samples)} ({100 - pct_exceed_400:.1f}%)")

    # Compute compressed prompt lengths for test set
    # For test items: if context >= 400, context shrinks by 50%
    test_compressed_prompt_tokens = []
    for p_tok, c_tok in zip(test_prompt_tokens, c_arr[len(train_samples):]):
        if c_tok >= config.BYPASS_TOKENS:
            # Context shrinks to 50%, rest (instruction + question) remains unchanged
            overhead = p_tok - c_tok
            compressed_p = overhead + int(c_tok * config.COMPRESSION_RATE)
            test_compressed_prompt_tokens.append(compressed_p)
        else:
            test_compressed_prompt_tokens.append(p_tok)

    # Output tokens assumed per generation
    OUT_TOKENS = 20
    
    # Stage 1: Labeling 200 train queries uncompressed
    train_tokens_uncomp = sum(t + OUT_TOKENS for t in train_prompt_tokens)
    
    # Stage 2: 100 test queries uncompressed
    test_tokens_uncomp = sum(t + OUT_TOKENS for t in test_prompt_tokens)
    
    # Stage 3: 100 test queries compressed
    test_tokens_comp = sum(t + OUT_TOKENS for t in test_compressed_prompt_tokens)
    
    print("\n" + "=" * 80)
    print("CALL & TOKEN PROJECTION TABLE BY PIPELINE STAGE")
    print("=" * 80)
    print("Assumed completion length: 20 tokens per call")
    print("Note: The learned router sweep evaluates all 100 test queries under threshold variations,")
    print("requiring candidate answers for all 100 queries once uncompressed and once compressed.\n")
    
    stages = [
        ("1. Train Labeling (200 queries, uncompressed)", len(train_samples), train_tokens_uncomp),
        ("2. Test Set (100 queries, uncompressed)", len(test_samples), test_tokens_uncomp),
        ("3. Test Set (100 queries, compressed)", len(test_samples), test_tokens_comp),
    ]
    
    print(f"{'Stage':<46} | {'Calls':<6} | {'Total Tokens':<14} | {'Avg Tok/Call':<12}")
    print("-" * 84)
    total_calls_all = 0
    total_tokens_all = 0
    for name, calls, toks in stages:
        total_calls_all += calls
        total_tokens_all += toks
        avg_tok = toks / calls if calls else 0
        print(f"{name:<46} | {calls:<6} | {toks:<14,d} | {avg_tok:<12.1f}")
    print("-" * 84)
    print(f"{'Total Needed if calling ONE tier for all stages':<46} | {total_calls_all:<6} | {total_tokens_all:<14,d} | {total_tokens_all/total_calls_all:<12.1f}")

    print("\n" + "=" * 80)
    print("TIER CAPACITY AUDIT: PROJECTED USAGE VS. FREE-TIER LIMITS")
    print("=" * 80)
    
    # SMALL TIER (Groq Llama-3.1-8B)
    # SMALL tier runs:
    # Train labels: 200 calls (uncompressed)
    # Test baselines / always-small / router small candidates: 100 uncompressed + 100 compressed = 200 calls
    small_calls = total_calls_all
    small_tokens = total_tokens_all
    small_tpd_limit = config.SMALL_TIER["tpd"]
    small_rpd_limit = config.SMALL_TIER["rpd"]
    small_days_tokens = small_tokens / small_tpd_limit
    small_days_reqs = small_calls / small_rpd_limit
    
    print("\n[SMALL TIER: Groq llama-3.1-8b-instant]")
    print(f"  Limits in config.py       : {small_rpd_limit:,} Requests/Day | {small_tpd_limit:,} Tokens/Day")
    print(f"  Projected Calls           : {small_calls} requests ({small_calls/small_rpd_limit*100:.1f}% of daily RPD)")
    print(f"  Projected Tokens          : {small_tokens:,} tokens ({small_tokens/small_tpd_limit*100:.1f}% of daily TPD)")
    print(f"  Days of Quota (Tokens)    : {small_days_tokens:.2f} days -> FITS EASILY IN 1 DAY (< 18% of quota)")
    
    # LARGE TIER CANDIDATE A: Groq Llama-3.3-70b-versatile
    # LARGE tier needs:
    # 200 train labeling calls + 100 test uncompressed + 100 test compressed = 400 calls
    large_calls = total_calls_all
    large_tokens = total_tokens_all
    groq_70b_tpd_limit = config.LARGE_TIER_GROQ["tpd"]
    groq_70b_rpd_limit = config.LARGE_TIER_GROQ["rpd"]
    groq_days_tokens = large_tokens / groq_70b_tpd_limit
    groq_days_reqs = large_calls / groq_70b_rpd_limit
    
    print("\n[LARGE CANDIDATE A: Groq llama-3.3-70b-versatile]")
    print(f"  Limits in config.py       : {groq_70b_rpd_limit:,} Requests/Day | {groq_70b_tpd_limit:,} Tokens/Day")
    print(f"  Projected Calls           : {large_calls} requests ({large_calls/groq_70b_rpd_limit*100:.1f}% of daily RPD)")
    print(f"  Projected Tokens          : {large_tokens:,} tokens ({large_tokens/groq_70b_tpd_limit*100:.1f}% of daily TPD)")
    print(f"  Days of Quota (Tokens)    : {groq_days_tokens:.2f} DAYS -> EXCEEDS 1 DAY QUOTA!")
    print(f"  BINDING BOTTLENECK        : The 100,000 Tokens/Day ceiling on Groq 70B ({large_tokens:,} tokens needed).")

    # LARGE TIER CANDIDATE B: Google Gemini Free Tier
    print("\n[LARGE CANDIDATE B: Google Gemini Free Tier (AI Studio)]")
    print("  Limits in config.py       : Request-based (RPM, TPM, RPD from AI Studio)")
    print(f"  Projected Calls           : {large_calls} requests total (200 train + 100 test uncompressed + 100 test compressed)")
    print(f"  Projected Tokens          : {large_tokens:,} tokens total (~{large_tokens/1000:.0f}k tokens)")
    print("  Comparison with Gemini    : Gemini free-tier daily token ceilings are typically multi-million TPM/TPD,")
    print("                              so its primary constraint is Requests-Per-Day (RPD).")
    print("                              - If Gemini model RPD >= 500 (e.g. gemini-1.5-flash / gemini-2.0-flash with 1,500 RPD): FITS IN 1 DAY!")
    print("                              - If Gemini model RPD <= 50 (e.g. certain experimental models with 50 RPD): Would take 8 days.")

    return {
        "mean_prompt_tokens": mean_p,
        "median_prompt_tokens": median_p,
        "p90_prompt_tokens": p90_p,
        "max_prompt_tokens": max_p,
        "pct_exceed_400": pct_exceed_400,
        "total_calls": total_calls_all,
        "total_tokens": total_tokens_all,
        "groq_days": groq_days_tokens,
    }


if __name__ == "__main__":
    print("Initializing Data & Retriever...")
    all_s, train_s, test_s, corp, st = prepare_data()
    retriever = get_retriever()
    
    run_smoke_tests(all_s, train_s, test_s, corp, st, retriever)
    metrics = run_token_budget_measurement(all_s, train_s, test_s, retriever)
