"""
OptiRAG-lite Compressed Context Labeling Pipeline.
Generates candidate answers for compressed test queries using Small (Groq) and Large (Gemini) tiers.
Uses existing prompt formatting, caching, and concurrency-safe budget guards.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import config
import llm
from prompts import SHARED_INSTRUCTION


def build_rag_prompt_from_context(question: str, context_str: str) -> str:
    """Builds standardized RAG prompt using pre-compressed context."""
    return (
        f"{SHARED_INSTRUCTION}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run labeling job for compressed test contexts.")
    parser.add_argument("--tier", choices=["small", "large"], required=True, help="Model tier to execute.")
    parser.add_argument("--live", action="store_true", help="Execute real live API calls (default: DRY_RUN).")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of questions.")
    return parser.parse_args()


def main():
    args = parse_args()
    tier_name = args.tier.lower()
    tier_config = config.SMALL_TIER if tier_name == "small" else config.LARGE_TIER

    compressed_input = config.RESULTS_DIR / "compressed_test.jsonl"
    out_path = config.RESULTS_DIR / f"labels_{tier_name}_test_compressed.jsonl"

    if not compressed_input.is_file():
        print(f"Error: Required input file does not exist: {compressed_input}")
        sys.exit(1)

    print("=" * 70)
    print(f"OPTIRAG COMPRESSED LABELING JOB: TIER={tier_name.upper()} | LIVE={args.live}")
    print(f"Provider: {tier_config['provider']} | Model: {tier_config['model']}")
    print(f"Input: {compressed_input}")
    print(f"Output: {out_path}")
    print("=" * 70)

    # 1. Load existing completed IDs for resumability
    completed_ids = set()
    if out_path.is_file():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["question_id"])
                    except Exception:
                        pass
        print(f"Resuming: Found {len(completed_ids)} already completed questions.")

    # 2. Read compressed test records
    compressed_items = []
    with open(compressed_input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                compressed_items.append(json.loads(line))

    if args.limit is not None:
        compressed_items = compressed_items[:args.limit]

    remaining = [item for item in compressed_items if item["question_id"] not in completed_ids]
    print(f"Total targeted questions: {len(compressed_items)} | Remaining to process: {len(remaining)}\n")

    # 3. Budget Pre-flight Check
    provider = tier_config["provider"]
    model = tier_config["model"]
    safe_rpd = int(tier_config["rpd"] * config.RPD_SAFETY_FACTOR)
    
    # Check cache status in dry-run mode
    total_prompt_tokens = 0
    cached_count = 0
    real_calls_projected = 0

    eval_items = []
    for item in remaining:
        qid = item["question_id"]
        question = item["question"]
        gold_answer = item["gold_answer"]
        comp_context = item["compressed_context"]

        prompt = build_rag_prompt_from_context(question, comp_context)
        prompt_tokens = llm.count_tokens(prompt)
        total_prompt_tokens += prompt_tokens

        # Check if already cached
        is_cached = (llm.read_cache(provider, model, prompt) is not None)

        if is_cached:
            cached_count += 1
        else:
            real_calls_projected += 1

        eval_items.append((item, prompt, prompt_tokens, is_cached))

    # Read current provider usage
    usage_path = config.CACHE_DIR / f"usage_{provider}_{model.replace('/', '_')}.json"
    used_requests = 0
    if usage_path.is_file():
        try:
            with open(usage_path, "r", encoding="utf-8") as uf:
                u_data = json.load(uf)
                tier_key = f"{provider}:{model}"
                used_requests = sum(d.get("requests", 0) for d in u_data.get(tier_key, {}).values())
        except Exception:
            pass

    remaining_safe_budget = max(0, safe_rpd - used_requests)
    print("BUDGET GUARD CHECK:")
    print(f"  Current Requests Used Today   : {used_requests}")
    print(f"  Safe Daily Limit (RPD Cap)    : {safe_rpd}")
    print(f"  Remaining Safe Quota          : {remaining_safe_budget} calls")
    print(f"  Disk Cache Hits (0 API calls) : {cached_count}")
    print(f"  Real Live Calls Required      : {real_calls_projected}")

    if real_calls_projected > remaining_safe_budget:
        print(f"\n[SAFETY STOP] Projected live calls ({real_calls_projected}) exceed remaining safe budget ({remaining_safe_budget})!")
        if args.live:
            print("Aborting live execution to protect daily quota.")
            sys.exit(1)

    if not args.live:
        print("\n[DRY-RUN MODE ACTIVE] Real API calls and file writes are disabled.")
        print(f"To execute live generation, run with `--live`.")
        return

    # 4. Live Execution
    out_f = open(out_path, "a", encoding="utf-8")
    try:
        for idx, (item, prompt, prompt_tokens, is_cached) in enumerate(eval_items, 1):
            qid = item["question_id"]
            question = item["question"]
            gold_answer = item["gold_answer"]

            res = llm.generate(tier_config, prompt, live=True)
            status = "CACHED (0 API calls)" if res["cached"] else "LIVE CALL"

            short_q = (question[:50] + "...") if len(question) > 50 else question
            print(f"[{idx}/{len(eval_items)}] ID: {qid[:8]} | {status} | InTokens={prompt_tokens} | Q: \"{short_q}\"", flush=True)

            record = {
                "question_id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "bypassed": item.get("bypassed", False),
                "original_tokens": item.get("original_tokens", 0),
                "compressed_tokens": item.get("compressed_tokens", 0),
                "ratio": item.get("ratio", 1.0),
                "prompt_tokens": prompt_tokens,
                "answer": res["text"],
                "in_tokens": res["in_tokens"],
                "out_tokens": res["out_tokens"],
                "latency_s": res["latency_s"],
                "cached": res["cached"],
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            print(f"    --> Saved ({out_path.name}): In={res['in_tokens']} Out={res['out_tokens']} Latency={res['latency_s']:.2f}s", flush=True)
    finally:
        out_f.close()

    print("\n" + "=" * 70)
    print("LIVE COMPRESSED LABELING PASS COMPLETE.")
    print(f"Records successfully written to: {out_path}")
    print("=" * 70)
    llm.print_usage_summary()


if __name__ == "__main__":
    main()
