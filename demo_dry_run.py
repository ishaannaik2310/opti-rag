"""
Dry-run demonstration script over 5 sample questions from the HotpotQA train split.
Simulates execution for both SMALL_TIER and LARGE_TIER with zero API calls.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from data import load_dataset_samples, get_train_test_split
from retrieve import Retriever
import config
import llm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def format_qa_prompt(question: str, chunks: list) -> str:
    """Standard HotpotQA RAG prompt template."""
    context_str = "\n".join([f"[{i+1}] Title: {title}\n{text}" for i, (text, title, _) in enumerate(chunks)])
    return (
        "Answer the question based on the context. If the context does not contain the answer, "
        "say \"I don't know\". Keep your answer concise.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def run_demo():
    print("=" * 80)
    print("OPTIRAG-LITE: DRY-RUN DEMO OVER 5 SAMPLE TRAIN QUESTIONS")
    print("=" * 80)

    samples = load_dataset_samples()
    train_data, _ = get_train_test_split(samples)
    retriever = Retriever()

    sample_questions = train_data[:5]

    for idx, item in enumerate(sample_questions, 1):
        q = item["question"]
        gold_ans = item["answer"]
        chunks = retriever.retrieve(q, k=config.TOP_K)
        prompt = format_qa_prompt(q, chunks)
        prompt_tokens = llm.count_tokens(prompt)

        print(f"\n[{idx}/5] Question: {q}")
        print(f"      Gold Answer: {gold_ans}")
        print(f"      Prompt Tokens: {prompt_tokens}")

        # Dry-run for SMALL_TIER
        res_small = llm.generate(config.SMALL_TIER, prompt, live=False)
        print(f"      SMALL ({res_small['provider']}:{res_small['model']}):")
        print(f"        Output   : {res_small['text']}")
        print(f"        Est. In  : {res_small['in_tokens']} tokens | Est. Out: {res_small['out_tokens']} tokens | Cached: {res_small['cached']}")

        # Dry-run for LARGE_TIER
        res_large = llm.generate(config.LARGE_TIER, prompt, live=False)
        print(f"      LARGE ({res_large['provider']}:{res_large['model']}):")
        print(f"        Output   : {res_large['text']}")
        print(f"        Est. In  : {res_large['in_tokens']} tokens | Est. Out: {res_large['out_tokens']} tokens | Cached: {res_large['cached']}")

    # Print summary of what would have been called
    llm.print_usage_summary()


if __name__ == "__main__":
    run_demo()
