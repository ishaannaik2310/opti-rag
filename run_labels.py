"""
OptiRAG-lite Labeling Pipeline.
Generates candidate answers and token telemetry line-by-line in a resumable job.
Supports simultaneous independent execution for SMALL (Groq) and LARGE (Gemini) tiers.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from data import load_dataset_samples, get_train_test_split
from retrieve import Retriever
from prompts import build_rag_prompt
import config
import llm


def parse_args():
    parser = argparse.ArgumentParser(description="Run labeling job for OptiRAG-lite.")
    parser.add_argument("--tier", choices=["small", "large"], required=True, help="Model tier to execute.")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split to label.")
    parser.add_argument("--live", action="store_true", help="Execute real live API calls (default: DRY_RUN).")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of questions.")
    return parser.parse_args()


def main():
    args = parse_args()
    tier_name = args.tier.lower()
    tier_config = config.SMALL_TIER if tier_name == "small" else config.LARGE_TIER

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"labels_{tier_name}_{args.split}.jsonl"

    print("=" * 70)
    print(f"OPTIRAG LABELING JOB: TIER={tier_name.upper()} | SPLIT={args.split.upper()} | LIVE={args.live}")
    print(f"Provider: {tier_config['provider']} | Model: {tier_config['model']}")
    print(f"Output File: {out_path}")
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

    # 2. Load dataset split
    all_samples = load_dataset_samples()
    train_samples, test_samples = get_train_test_split(all_samples)
    dataset = train_samples if args.split == "train" else test_samples

    if args.limit is not None:
        dataset = dataset[:args.limit]

    retriever = Retriever()

    remaining = [item for item in dataset if item["id"] not in completed_ids]
    print(f"Total targeted questions: {len(dataset)} | Remaining to process: {len(remaining)}\n")

    if not args.live:
        print("[DRY-RUN MODE ACTIVE] Real API calls and file writes are disabled.")
        print("Evaluating prompts, disk cache status, token projections, and rate limit pacing...\n")

    total_prompt_tokens = 0
    cached_count = 0
    real_calls_projected = 0

    out_f = open(out_path, "a", encoding="utf-8") if args.live else None

    try:
        for idx, item in enumerate(remaining, 1):
            qid = item["id"]
            question = item["question"]
            gold_answer = item["answer"]

            # Retrieve top-5 supporting chunks
            chunks = retriever.retrieve(question, k=config.TOP_K)
            prompt = build_rag_prompt(question, chunks)
            prompt_tokens = llm.count_tokens(prompt)
            total_prompt_tokens += prompt_tokens

            # Check cache or simulate
            res = llm.generate(tier_config, prompt, live=args.live)

            if res["cached"]:
                cached_count += 1
                status = "CACHED (0 API calls)"
            elif args.live:
                status = "LIVE CALL"
            else:
                real_calls_projected += 1
                status = "DRY_RUN (Simulated)"

            short_q = (question[:55] + "...") if len(question) > 55 else question
            print(f"[{idx}/{len(remaining)}] ID: {qid[:8]} | {status} | InTokens={prompt_tokens} | Q: \"{short_q}\"", flush=True)

            if args.live and out_f:
                record = {
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "retrieved_chunks": [(title, float(score)) for _, title, score in chunks],
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
        if out_f:
            out_f.close()

    print("\n" + "=" * 70)
    if not args.live:
        # Calculate pacing projections
        safe_rpm = tier_config["rpm"] * config.SAFETY_FACTOR
        safe_tpm = tier_config["tpm"] * config.SAFETY_FACTOR
        safe_rpd = int(tier_config["rpd"] * config.RPD_SAFETY_FACTOR)

        proj_out_tokens = real_calls_projected * 50
        proj_total_tokens = total_prompt_tokens + proj_out_tokens

        # Time based on RPM pacing
        time_by_rpm_min = (real_calls_projected / safe_rpm) if safe_rpm > 0 else 0
        # Time based on TPM pacing
        time_by_tpm_min = (proj_total_tokens / safe_tpm) if safe_tpm > 0 else 0
        expected_time_min = max(time_by_rpm_min, time_by_tpm_min)

        print("DRY-RUN PROJECTION SUMMARY:")
        print(f"  Target Questions Evaluated:    {len(remaining)}")
        print(f"  Disk Cache Hits (0 API calls): {cached_count}")
        print(f"  Real API Calls Required:       {real_calls_projected}")
        print(f"  Projected Input Tokens:        {total_prompt_tokens:,}")
        print(f"  Projected Output Tokens:       ~{proj_out_tokens:,} (assumed 50 max tokens/call)")
        print(f"  Projected Total Tokens:        ~{proj_total_tokens:,}")
        print(f"  Provider Rate Limits:          Safe RPM: {safe_rpm:.1f} | Safe TPM: {safe_tpm:,.0f} | Safe RPD: {safe_rpd}")
        print(f"  Estimated Wall-Clock Time:     ~{expected_time_min:.1f} minutes ({expected_time_min * 60:.0f} seconds)")
        print(f"  Estimated Cost:                $0.00 (100% Free Tier)")
        print(f"  Output File Written:           None (Dry-run makes 0 file changes)")
        print("=" * 70)
        print("To execute live labeling, run with the `--live` flag.")
    else:
        print("LIVE LABELING PASS COMPLETE.")
        print(f"Records successfully written to: {out_path}")
        print("=" * 70)
        llm.print_usage_summary()


if __name__ == "__main__":
    main()
