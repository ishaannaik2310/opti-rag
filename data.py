"""
Data loader and corpus builder for OptiRAG-lite.
Loads HotpotQA (distractor setting), performs seeded shuffle and 200/100 train/test split,
pools all paragraphs into a corpus, and saves split IDs to results/split.json.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from datasets import load_dataset
import config


def load_dataset_samples(seed: int = config.SEED, total: int = config.TOTAL_SAMPLES) -> List[Dict[str, Any]]:
    """Loads HotpotQA distractor split, shuffles with fixed seed, and extracts `total` questions."""
    raw_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    shuffled = raw_dataset.shuffle(seed=seed)
    selected = shuffled.select(range(total))
    
    samples = []
    for item in selected:
        samples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item.get("type", ""),
            "level": item.get("level", ""),
            "supporting_facts": {
                "title": item["supporting_facts"]["title"],
                "sent_id": item["supporting_facts"]["sent_id"],
            },
            "context": {
                "title": item["context"]["title"],
                "sentences": item["context"]["sentences"],
            }
        })
    return samples


def get_train_test_split(samples: List[Dict[str, Any]], train_size: int = config.TRAIN_SAMPLES) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits selected samples into train and test sets, and writes IDs to results/split.json."""
    train_samples = samples[:train_size]
    test_samples = samples[train_size:]
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_info = {
        "train_ids": [s["id"] for s in train_samples],
        "test_ids": [s["id"] for s in test_samples],
        "total_train": len(train_samples),
        "total_test": len(test_samples),
        "seed": config.SEED,
    }
    
    with open(config.RESULTS_DIR / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
        
    return train_samples, test_samples


def build_corpus(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Pools all paragraphs across the samples into a unified retrieval corpus.
    Keeps title and text (sentences joined with spaces).
    Returns corpus list and duplicate statistics.
    """
    seen_chunks = {}  # key: (title, text) -> chunk_id
    corpus = []
    duplicate_count = 0
    total_paragraphs_encountered = 0
    
    for sample in samples:
        titles = sample["context"]["title"]
        sentences_list = sample["context"]["sentences"]
        
        for title, sents in zip(titles, sentences_list):
            total_paragraphs_encountered += 1
            text = " ".join(s.strip() for s in sents if s.strip())
            key = (title, text)
            
            if key in seen_chunks:
                duplicate_count += 1
            else:
                chunk_id = len(corpus)
                seen_chunks[key] = chunk_id
                corpus.append({
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text,
                })
                
    stats = {
        "total_paragraphs_encountered": total_paragraphs_encountered,
        "unique_corpus_size": len(corpus),
        "duplicate_paragraphs": duplicate_count,
    }
    return corpus, stats


def prepare_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience helper returning all_samples, train_samples, test_samples, corpus, and stats."""
    all_samples = load_dataset_samples(seed=config.SEED, total=config.TOTAL_SAMPLES)
    train_samples, test_samples = get_train_test_split(all_samples, train_size=config.TRAIN_SAMPLES)
    corpus, stats = build_corpus(all_samples)
    return all_samples, train_samples, test_samples, corpus, stats


if __name__ == "__main__":
    print("Executing data.py standalone test...")
    all_s, train_s, test_s, corp, st = prepare_data()
    print(f"Total samples: {len(all_s)}")
    print(f"Train split: {len(train_s)}")
    print(f"Test split: {len(test_s)}")
    print(f"Corpus size: {st['unique_corpus_size']}")
    print(f"Encountered: {st['total_paragraphs_encountered']}")
    print(f"Duplicates: {st['duplicate_paragraphs']}")
