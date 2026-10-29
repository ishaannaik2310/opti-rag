"""
OptiRAG-lite Safe Context Compression Module.
Uses LLMLingua-2 with Protected Token Mask (PTM) to safely compress retrieved context
without hallucinating or dropping critical numerical, temporal, currency, or negation entities.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
import torch

import config
import ptm

# Force offline mode for Hugging Face Hub if cached
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Pinned BPE Tokenizer for accurate local token estimation (cl100k_base)
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

# Global singleton compressor instance (lazy loaded)
_COMPRESSOR = None


def count_tokens(text: str) -> int:
    """Return BPE token count using cl100k_base (consistent with budget analysis)."""
    if not text:
        return 0
    return len(_TIKTOKEN_ENC.encode(text))


def get_prompt_compressor():
    """Lazy load LLMLingua-2 PromptCompressor on CPU with limited threads."""
    global _COMPRESSOR
    if _COMPRESSOR is None:
        torch.set_num_threads(2)
        from llmlingua import PromptCompressor

        _COMPRESSOR = PromptCompressor(
            model_name=config.LINGUA_MODEL_NAME,
            device_map="cpu",
            use_llmlingua2=True,
        )
    return _COMPRESSOR


def extract_entity_tokens(text: str) -> List[str]:
    """
    Extract capitalized words and multi-word names/entities for the 'ptm_ent' variant.
    Excludes common sentence-initial starter words.
    """
    sentence_starters = {
        "The", "A", "An", "In", "On", "At", "By", "For", "With", "About",
        "It", "This", "That", "These", "Those", "There", "Here", "He", "She",
        "They", "We", "You", "I", "His", "Her", "Their", "Our", "My",
        "When", "Where", "Why", "How", "What", "Which", "Who", "Whom", "Whose",
        "After", "Before", "During", "Since", "Until", "While", "Although",
        "Because", "If", "Unless", "However", "Moreover", "Furthermore",
        "According", "As", "Later", "Then", "So", "Thus", "Therefore",
    }
    tokens = set()
    pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

    for match in pattern.finditer(text):
        phrase = match.group()
        words = phrase.split()
        if len(words) == 1 and words[0] in sentence_starters:
            start_idx = match.start()
            prefix = text[:start_idx].strip()
            if not prefix or prefix.endswith((".", "?", "!", "\n")):
                continue
        for w in words:
            tokens.add(w)

    return sorted(tokens)


def _extract_chunk_info(
    chunk: Union[Tuple[Any, ...], Dict[str, Any]]
) -> Tuple[str, str]:
    """Helper to extract (text, title) from diverse chunk representations."""
    if isinstance(chunk, tuple):
        if len(chunk) == 2:
            # Could be (title, text) or (text, title)
            # By retrieve.py convention: chunks are (text, title, score)
            # If 2 elements, test if first looks like title or text
            return str(chunk[0]), str(chunk[1])
        elif len(chunk) >= 3:
            # retrieve.py standard format: (text, title, score)
            return str(chunk[0]), str(chunk[1])
    elif isinstance(chunk, dict):
        return str(chunk.get("text", "")), str(chunk.get("title", ""))
    return str(chunk), "Unknown"


def compress_context(
    chunks: Union[List[Tuple[str, str, float]], List[Tuple[str, str]], List[Dict[str, Any]]],
    variant: str = "ptm",
    rate: float = config.COMPRESSION_RATE,
) -> Dict[str, Any]:
    """
    Compress a list of retrieved chunks using LLMLingua-2 and optional PTM force-keep rules.

    Variants:
        - 'none'    : LLMLingua-2 with no forced tokens.
        - 'ptm'     : Force-keep numerical, date, currency, percent, and negation tokens.
        - 'ptm_ent' : 'ptm' plus capitalized named-entity tokens.

    Rules:
        - If total chunk text tokens < config.BYPASS_TOKENS (400), skips compression (bypassed=True).
        - Chunk headers ('[i] Title: ...') remain uncompressed.
        - Formats context identically to prompts.py uncompressed layout.

    Returns:
        dict:
            compressed_context (str): Full reconstructed context string
            original_tokens (int): Token count before compression
            compressed_tokens (int): Token count after compression
            ratio (float): compressed_tokens / original_tokens
            seconds (float): Execution wall-clock time
            bypassed (bool): True if compression was bypassed by token threshold
            variant (str): Selected variant
    """
    if variant not in {"none", "ptm", "ptm_ent"}:
        raise ValueError(f"Unknown compression variant '{variant}'. Choose from 'none', 'ptm', 'ptm_ent'.")

    # 1. Normalize chunks to [(text, title)]
    parsed_chunks: List[Tuple[str, str]] = []
    for c in chunks:
        t_text, t_title = _extract_chunk_info(c)
        parsed_chunks.append((t_text, t_title))

    # 2. Build uncompressed context string for reference
    uncompressed_blocks = [
        f"[{i}] Title: {title}\n{text}"
        for i, (text, title) in enumerate(parsed_chunks, 1)
    ]
    uncompressed_context_str = "\n\n".join(uncompressed_blocks)

    # 3. Check total chunk text tokens against BYPASS_TOKENS (400)
    raw_texts_tokens = sum(count_tokens(text) for text, _ in parsed_chunks)
    total_original_tokens = count_tokens(uncompressed_context_str)

    if raw_texts_tokens < config.BYPASS_TOKENS:
        return {
            "compressed_context": uncompressed_context_str,
            "original_tokens": total_original_tokens,
            "compressed_tokens": total_original_tokens,
            "ratio": 1.0,
            "seconds": 0.0,
            "bypassed": True,
            "variant": variant,
        }

    # 4. Perform Compression
    torch.set_num_threads(2)
    compressor = get_prompt_compressor()
    t0 = time.time()

    compressed_blocks = []
    for i, (text, title) in enumerate(parsed_chunks, 1):
        if not text.strip():
            compressed_blocks.append(f"[{i}] Title: {title}\n{text}")
            continue

        # Determine force tokens and digit reservation per variant
        if variant == "none":
            force_tokens = []
            force_digits = False
        elif variant == "ptm":
            force_tokens = ptm.protected_tokens(text)
            force_digits = True
        elif variant == "ptm_ent":
            base_tokens = set(ptm.protected_tokens(text))
            ent_tokens = set(extract_entity_tokens(text))
            force_tokens = sorted(base_tokens | ent_tokens)
            force_digits = True

        # Call LLMLingua-2
        res = compressor.compress_prompt_llmlingua2(
            [text],
            rate=rate,
            force_tokens=force_tokens,
            force_reserve_digit=force_digits,
            drop_consecutive=False,
        )
        compressed_text = res["compressed_prompt"].strip()
        compressed_blocks.append(f"[{i}] Title: {title}\n{compressed_text}")

    elapsed_s = time.time() - t0
    compressed_context_str = "\n\n".join(compressed_blocks)
    total_compressed_tokens = count_tokens(compressed_context_str)

    actual_ratio = (
        round(total_compressed_tokens / total_original_tokens, 4)
        if total_original_tokens > 0
        else 1.0
    )

    return {
        "compressed_context": compressed_context_str,
        "original_tokens": total_original_tokens,
        "compressed_tokens": total_compressed_tokens,
        "ratio": actual_ratio,
        "seconds": round(elapsed_s, 3),
        "bypassed": False,
        "variant": variant,
    }
