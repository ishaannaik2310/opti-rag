"""
Live smoke test for SMALL_TIER (Groq llama-3.1-8b-instant).
Executes exactly 2 prompts live, then re-runs them to prove caching (4 calls total: 2 real, 2 cached).
"""

import os
import sys
from dotenv import load_dotenv

import config
import llm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(config.BASE_DIR / ".env")


def run_live_smoke():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set in .env. Skipping live smoke test.")
        return

    tier = config.SMALL_TIER
    prompts = [
        "What is the capital of Maine? Answer concisely.",
        "What direction does the Nile river flow? Answer concisely.",
    ]

    print("=" * 80)
    print("LIVE SMOKE TEST: SMALL TIER (Groq llama-3.1-8b-instant)")
    print("=" * 80)

    # First Pass: Expect Real API calls
    print("\n--- FIRST PASS (Expecting Real API Calls) ---")
    for i, p in enumerate(prompts, 1):
        print(f"\n[Prompt {i}] {p}")
        res = llm.generate(tier, p, live=True)
        print(f"  Response : {res['text']}")
        print(f"  Tokens   : In={res['in_tokens']}, Out={res['out_tokens']}")
        print(f"  Latency  : {res['latency_s']:.2f}s | Cached: {res['cached']}")

    # Second Pass: Expect Cache Hits
    print("\n--- SECOND PASS (Expecting Cache Hits - 0 API Calls) ---")
    for i, p in enumerate(prompts, 1):
        print(f"\n[Prompt {i}] {p}")
        res = llm.generate(tier, p, live=True)
        print(f"  Response : {res['text']}")
        print(f"  Tokens   : In={res['in_tokens']}, Out={res['out_tokens']}")
        print(f"  Latency  : {res['latency_s']:.4f}s | Cached: {res['cached']}")

    llm.print_usage_summary()


if __name__ == "__main__":
    run_live_smoke()
