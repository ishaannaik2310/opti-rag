"""
Pilot experiment across 3 candidate SMALL models on Groq and LARGE Gemini 3.5 Flash Lite.
Evaluates 12 train questions with the new strict concise prompt.
Computes accuracy, token stats, latency, and provides a comparative summary.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from data import load_dataset_samples, get_train_test_split
from retrieve import Retriever
from prompts import build_rag_prompt
import metrics
import config
import llm

PILOT_MODELS = [
    {
        "key": "allam-2-7b",
        "display_name": "Groq Allam 2 7B",
        "tier": {
            "provider": "groq",
            "model": "allam-2-7b",
            "rpm": 30,
            "tpm": 6000,
            "rpd": 7000,
            "tpd": 500000,
        }
    },
    {
        "key": "gpt-oss-20b",
        "display_name": "Groq GPT-OSS 20B",
        "tier": {
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "rpm": 30,
            "tpm": 8000,
            "rpd": 1000,
            "tpd": 500000,
        }
    },
    {
        "key": "qwen-27b",
        "display_name": "Groq Qwen 3.8 27B",
        "tier": {
            "provider": "groq",
            "model": "qwen/qwen3.8-27b",
            "rpm": 30,
            "tpm": 8000,
            "rpd": 1000,
            "tpd": 500000,
        }
    },
    {
        "key": "gemini-flash-lite",
        "display_name": "Gemini 3.5 Flash Lite",
        "tier": config.LARGE_TIER,
    }
]


def main():
    print("=" * 90)
    print("OPTIRAG-LITE: PILOT EVALUATION (12 TRAIN QUESTIONS x 4 MODELS)")
    print("=" * 90)

    all_samples = load_dataset_samples()
    train_samples, _ = get_train_test_split(all_samples)
    pilot_samples = train_samples[:12]

    retriever = Retriever()

    results = []
    stats = {m["key"]: {"correct": 0, "in_tokens": [], "out_tokens": [], "latencies": [], "errors": 0} for m in PILOT_MODELS}

    for idx, item in enumerate(pilot_samples, 1):
        q = item["question"]
        gold = item["answer"]
        chunks = retriever.retrieve(q, k=config.TOP_K)
        prompt = build_rag_prompt(q, chunks)

        print(f"\n[{idx}/12] Question: {q}")
        print(f"       Gold Answer: {gold}")

        row = {
            "idx": idx,
            "id": item["id"],
            "question": q,
            "gold_answer": gold,
            "models": {}
        }

        for m_info in PILOT_MODELS:
            m_key = m_info["key"]
            tier = m_info["tier"]
            try:
                res = llm.generate(tier, prompt, live=True)
                ans = res["text"].strip()
                corr = metrics.is_correct(ans, gold)
                f1 = metrics.compute_f1(ans, gold)

                if corr:
                    stats[m_key]["correct"] += 1
                stats[m_key]["in_tokens"].append(res["in_tokens"])
                stats[m_key]["out_tokens"].append(res["out_tokens"])
                stats[m_key]["latencies"].append(res["latency_s"])

                row["models"][m_key] = {
                    "answer": ans,
                    "correct": corr,
                    "f1": f1,
                    "in_tokens": res["in_tokens"],
                    "out_tokens": res["out_tokens"],
                    "latency_s": res["latency_s"],
                    "error": None,
                }
                status_str = "CORRECT" if corr else "WRONG"
                print(f"  - {m_info['display_name']:<22}: [{status_str}] \"{ans}\" (F1={f1:.2f}, Lat={res['latency_s']:.2f}s)")
            except Exception as e:
                stats[m_key]["errors"] += 1
                row["models"][m_key] = {
                    "answer": "",
                    "correct": False,
                    "f1": 0.0,
                    "in_tokens": 0,
                    "out_tokens": 0,
                    "latency_s": 0.0,
                    "error": str(e),
                }
                print(f"  - {m_info['display_name']:<22}: [ERROR] {e}")

        results.append(row)

    # Save detailed pilot results
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pilot_out_path = config.RESULTS_DIR / "pilot_results.json"
    with open(pilot_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'PILOT SUMMARY (12 QUESTIONS)':^90}")
    print("=" * 90)
    print(f"{'Model':<25} | {'Acc (N=12)':<12} | {'Mean Latency':<14} | {'Mean In Tok':<13} | {'Mean Out Tok':<13} | {'Errors':<6}")
    print("-" * 90)

    for m_info in PILOT_MODELS:
        m_key = m_info["key"]
        st = stats[m_key]
        n = len(st["in_tokens"]) or 1
        acc = (st["correct"] / 12.0) * 100
        mean_lat = sum(st["latencies"]) / n
        mean_in = sum(st["in_tokens"]) / n
        mean_out = sum(st["out_tokens"]) / n
        err = st["errors"]
        print(f"{m_info['display_name']:<25} | {acc:>5.1f}% ({st['correct']}/12) | {mean_lat:>10.2f}s   | {mean_in:>10.1f}   | {mean_out:>10.1f}   | {err:>6}")

    print("=" * 90)
    llm.print_usage_summary()


if __name__ == "__main__":
    main()
