"""
OptiRAG-lite Feature Extraction Module.

Extracts three feature representations from queries and retrieval telemetry:
1. 'query-only' (385 features):
   - 384-dimensional MiniLM query embedding (all-MiniLM-L6-v2)
   - 1 scalar query token count (cl100k_base tiktoken)
2. 'telemetry-only' (8 features):
   - top-1 retrieval cosine score
   - top-2 retrieval cosine score
   - top1-top2 margin (gap)
   - top-5 mean score
   - top-5 std score
   - softmax entropy of top-5 scores
   - mean pairwise cosine similarity among top-5 retrieved chunks
   - total context token count (cl100k_base tiktoken)
3. 'both' (393 features):
   - Concatenation of query-only (385) and telemetry-only (8).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tiktoken

import config
from retrieve import Retriever

# Force offline mode for local HF cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
_RETRIEVER_INSTANCE: Optional[Retriever] = None
_CORPUS_TITLE_MAP: Optional[Dict[str, str]] = None


def count_tokens(text: str) -> int:
    """Return BPE token count using cl100k_base."""
    if not text:
        return 0
    return len(_TIKTOKEN_ENC.encode(text))


def get_shared_retriever() -> Retriever:
    """Lazy-loads a singleton Retriever instance for embedding extraction."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = Retriever()
        _RETRIEVER_INSTANCE.build_or_load_index()
    return _RETRIEVER_INSTANCE


def get_corpus_title_map(retriever: Optional[Retriever] = None) -> Dict[str, str]:
    """Lazy-loads a mapping of chunk title -> text from the HotpotQA corpus."""
    global _CORPUS_TITLE_MAP
    if _CORPUS_TITLE_MAP is None:
        r = retriever or get_shared_retriever()
        if r.corpus is None:
            r.build_or_load_index()
        _CORPUS_TITLE_MAP = {c["title"]: c["text"] for c in r.corpus}
    return _CORPUS_TITLE_MAP


def extract_query_features(query: str, retriever: Optional[Retriever] = None) -> np.ndarray:
    """
    Extracts 385-d query features:
      - 384-d normalized query embedding
      - 1-d query token count
    """
    r = retriever or get_shared_retriever()
    model = r._get_model()
    
    q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
    q_tok_count = float(count_tokens(query))
    
    features = np.concatenate([q_vec, np.array([q_tok_count], dtype=np.float32)])
    assert features.shape == (385,), f"Expected shape (385,), got {features.shape}"
    return features


def extract_telemetry_features(
    retrieved_chunks: List[Any],
    retriever: Optional[Retriever] = None,
) -> np.ndarray:
    """
    Extracts 8-d retrieval-derived telemetry features only (no query features):
      1. top-1 score
      2. top-2 score
      3. top1-top2 gap
      4. top-5 mean score
      5. top-5 std score
      6. softmax entropy of the 5 scores
      7. mean pairwise cosine similarity among chunk embeddings
      8. context token count
    """
    r = retriever or get_shared_retriever()
    model = r._get_model()
    title_map = get_corpus_title_map(r)

    parsed_texts: List[str] = []
    parsed_titles: List[str] = []
    parsed_scores: List[float] = []

    for c in retrieved_chunks:
        text = ""
        title = ""
        score = 0.0

        if isinstance(c, (list, tuple)):
            if len(c) >= 3:
                # Format: (text, title, score)
                text = str(c[0])
                title = str(c[1])
                score = float(c[2])
            elif len(c) == 2:
                # Could be [title, score] or [text, title]
                if isinstance(c[1], (int, float)):
                    title = str(c[0])
                    score = float(c[1])
                    text = title_map.get(title, "")
                else:
                    text = str(c[0])
                    title = str(c[1])
            elif len(c) == 1:
                text = str(c[0])
        elif isinstance(c, dict):
            text = str(c.get("text", ""))
            title = str(c.get("title", ""))
            score = float(c.get("score", 0.0))
            if not text and title:
                text = title_map.get(title, "")
        else:
            text = str(c)

        # Fallback to title lookup if text is still empty
        if not text and title:
            text = title_map.get(title, "")

        parsed_texts.append(text)
        parsed_titles.append(title)
        parsed_scores.append(score)

    # Pad or truncate to top-5
    k = config.TOP_K
    if len(parsed_scores) < k:
        pad_len = k - len(parsed_scores)
        parsed_scores.extend([0.0] * pad_len)
        parsed_texts.extend([""] * pad_len)
        parsed_titles.extend([""] * pad_len)
    else:
        parsed_scores = parsed_scores[:k]
        parsed_texts = parsed_texts[:k]
        parsed_titles = parsed_titles[:k]

    scores_arr = np.array(parsed_scores, dtype=np.float32)

    # 1. Top-1 score
    s1 = float(scores_arr[0])
    # 2. Top-2 score
    s2 = float(scores_arr[1])
    # 3. Gap
    gap = float(s1 - s2)
    # 4. Mean
    mean_score = float(np.mean(scores_arr))
    # 5. Std
    std_score = float(np.std(scores_arr))

    # 6. Softmax entropy
    exp_s = np.exp(scores_arr - np.max(scores_arr))
    probs = exp_s / (np.sum(exp_s) + 1e-12)
    softmax_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # 7. Mean pairwise cosine similarity among chunks
    valid_texts = [t for t in parsed_texts if t.strip()]
    if len(valid_texts) >= 2:
        chunk_vecs = model.encode(valid_texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        sim_matrix = np.dot(chunk_vecs, chunk_vecs.T)
        n = len(valid_texts)
        triu_indices = np.triu_indices(n, k=1)
        mean_pairwise_cos = float(np.mean(sim_matrix[triu_indices]))
    elif len(valid_texts) == 1:
        mean_pairwise_cos = 1.0
    else:
        mean_pairwise_cos = 0.0

    # 8. Context token count
    context_str = "\n\n".join(
        f"[{i}] Title: {title}\n{text}"
        for i, (text, title) in enumerate(zip(parsed_texts, parsed_titles), 1)
        if text.strip()
    )
    context_tokens = float(count_tokens(context_str))

    telemetry = np.array([
        s1,
        s2,
        gap,
        mean_score,
        std_score,
        softmax_entropy,
        mean_pairwise_cos,
        context_tokens,
    ], dtype=np.float32)

    assert telemetry.shape == (8,), f"Expected shape (8,), got {telemetry.shape}"
    return telemetry


def extract_features(
    query: str,
    retrieved_chunks: List[Any],
    variant: str = "both",
    retriever: Optional[Retriever] = None,
) -> np.ndarray:
    """
    Extracts features for a given query and retrieved context chunks.

    Variants:
      - 'query-only'     : 385 features
      - 'telemetry-only' : 8 features
      - 'both'           : 393 features
    """
    if variant == "query-only":
        return extract_query_features(query, retriever=retriever)
    elif variant == "telemetry-only":
        return extract_telemetry_features(retrieved_chunks, retriever=retriever)
    elif variant == "both":
        q_feat = extract_query_features(query, retriever=retriever)
        t_feat = extract_telemetry_features(retrieved_chunks, retriever=retriever)
        both = np.concatenate([q_feat, t_feat])
        assert both.shape == (393,), f"Expected shape (393,), got {both.shape}"
        return both
    else:
        raise ValueError(f"Unknown feature variant '{variant}'. Choose from 'query-only', 'telemetry-only', 'both'.")
