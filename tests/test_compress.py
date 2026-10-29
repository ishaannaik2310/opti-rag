"""
Unit tests for OptiRAG-lite Context Compression and Compression Metrics.
"""

import pytest
import compress
import compress_metrics
import config


def test_entity_survival_rate_hand_made():
    """Verify entity survival rate logic on synthetic strings."""
    orig = "The revenue in 2023 was $1.5M, down 15%."
    # Case A: all entities kept
    comp_all = "In 2023 was $1.5M down 15%."
    res_all = compress_metrics.entity_survival_rate(orig, comp_all)
    assert res_all["overall_rate"] == 1.0
    assert res_all["kept_count"] == 3
    assert res_all["total_count"] == 3

    # Case B: 1 entity dropped ($1.5M missing)
    comp_drop = "In 2023 down 15%."
    res_drop = compress_metrics.entity_survival_rate(orig, comp_drop)
    assert res_drop["kept_count"] == 2
    assert res_drop["total_count"] == 3
    assert res_drop["overall_rate"] == round(2 / 3, 4)
    assert res_drop["by_kind"]["currency"]["kept"] == 0
    assert res_drop["by_kind"]["currency"]["total"] == 1


def test_gold_answer_survival_cases():
    """Verify gold answer survival behavior for extractive and edge cases."""
    orig = "The capital of Maine is Augusta. It was established in 1754."
    
    # 1. Gold answer kept
    comp_kept = "Capital Maine Augusta. Established 1754."
    assert compress_metrics.gold_answer_survival(orig, comp_kept, "Augusta") is True

    # 2. Gold answer dropped
    comp_lost = "Capital Maine. Established 1754."
    assert compress_metrics.gold_answer_survival(orig, comp_lost, "Augusta") is False

    # 3. Gold answer not in original context -> returns None
    assert compress_metrics.gold_answer_survival(orig, comp_kept, "Portland") is None

    # 4. Yes/No gold answers -> returns None
    assert compress_metrics.gold_answer_survival(orig, comp_kept, "yes") is None
    assert compress_metrics.gold_answer_survival(orig, comp_kept, "no") is None


def test_bypass_rule_below_400_tokens():
    """Verify that chunks with < 400 raw text tokens bypass compression completely."""
    short_chunks = [
        ("Short passage one.", "Title 1"),
        ("Short passage two.", "Title 2"),
    ]
    res = compress.compress_context(short_chunks, variant="ptm")
    assert res["bypassed"] is True
    assert res["seconds"] == 0.0
    assert res["ratio"] == 1.0
    assert "[1] Title: Title 1\nShort passage one." in res["compressed_context"]
    assert "[2] Title: Title 2\nShort passage two." in res["compressed_context"]


def test_compress_context_headers_preserved():
    """Verify that chunk headers '[i] Title: ...' are never mangled by compression."""
    # Chunk with enough repetition to exceed 400 tokens
    long_text = ("The quick brown fox jumps over the lazy dog. " * 50).strip()
    chunks = [
        (long_text, "Alpha Document"),
        (long_text, "Beta Document"),
    ]
    res = compress.compress_context(chunks, variant="none")
    assert res["bypassed"] is False
    assert "[1] Title: Alpha Document\n" in res["compressed_context"]
    assert "[2] Title: Beta Document\n" in res["compressed_context"]
    assert res["compressed_tokens"] < res["original_tokens"]


def test_ptm_variant_entity_survival():
    """Verify that ptm variant keeps 'not', '2023', '15%' on real LLMLingua-2 model."""
    text = (
        "Revenue is not growing in 2023, down 15%. "
        + ("Additional financial background details are discussed extensively in the quarterly review. " * 30)
    )
    chunks = [(text, "Quarterly Financials")]
    res = compress.compress_context(chunks, variant="ptm", rate=0.5)

    comp_text = res["compressed_context"].lower()
    assert "not" in comp_text, "Expected negation 'not' to be preserved"
    assert "2023" in comp_text, "Expected year '2023' to be preserved"
    assert "15%" in comp_text, "Expected percentage '15%' to be preserved"
