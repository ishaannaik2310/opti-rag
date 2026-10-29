"""
Dense retrieval module using SentenceTransformers (all-MiniLM-L6-v2) and FAISS IndexFlatIP.
Embeds the pooled corpus, normalizes vectors (for exact cosine similarity),
caches embeddings to disk, and provides retrieve(query, k=5) returning (text, title, score).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config
from data import prepare_data


class Retriever:
    """Manages dense vector indexing and cosine retrieval over HotpotQA corpus."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        cache_dir: Path = config.CACHE_DIR,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_path = self.cache_dir / "corpus_embeddings.npy"
        self.corpus_path = self.cache_dir / "corpus.json"
        
        self.model: Optional[SentenceTransformer] = None
        self.corpus: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build_or_load_index(self, corpus: Optional[List[Dict[str, Any]]] = None) -> None:
        """Loads cached corpus and embeddings if present; otherwise computes and caches them."""
        if self.embeddings_path.exists() and self.corpus_path.exists() and corpus is None:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
            self.embeddings = np.load(self.embeddings_path)
        else:
            if corpus is None:
                _, _, _, corpus, _ = prepare_data()
            self.corpus = corpus
            
            # Format chunks with title for stronger entity representation
            passages = [f"{c['title']}: {c['text']}" for c in self.corpus]
            
            model = self._get_model()
            embeddings = model.encode(
                passages,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype(np.float32)
            
            self.embeddings = embeddings
            np.save(self.embeddings_path, self.embeddings)
            with open(self.corpus_path, "w", encoding="utf-8") as f:
                json.dump(self.corpus, f, indent=2)

        # Build FAISS IndexFlatIP (inner product of normalized vectors == cosine similarity)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k chunks for a query.
        Returns: list of (chunk_text, title, score) sorted descending by score.
        """
        if self.index is None:
            self.build_or_load_index()

        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        
        scores, indices = self.index.search(q_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.corpus[idx]
            results.append((chunk["text"], chunk["title"], float(score)))
            
        return results

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Returns normalized 384-d embedding vector for a query (used in features.py)."""
        model = self._get_model()
        return model.encode(query, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def get_chunk_embeddings(self, chunk_indices: List[int]) -> np.ndarray:
        """Returns matrix of embeddings for specific corpus chunk indices."""
        if self.embeddings is None:
            self.build_or_load_index()
        return self.embeddings[chunk_indices]


# Module-level singleton instance for convenient access
_retriever_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
        _retriever_instance.build_or_load_index()
    return _retriever_instance


def retrieve(query: str, k: int = config.TOP_K) -> List[Tuple[str, str, float]]:
    """Functional interface required by PROJECT_CONTEXT.md section 6."""
    return get_retriever().retrieve(query, k=k)


if __name__ == "__main__":
    print("Testing retrieve.py...")
    r = get_retriever()
    test_q = "Were Scott Derrickson and Ed Wood of the same nationality?"
    res = r.retrieve(test_q, k=5)
    print(f"Query: {test_q}")
    for rank, (text, title, sc) in enumerate(res, 1):
        print(f"[{rank}] Score: {sc:.4f} | Title: {title}")
        print(f"     Preview: {text[:100]}...")
