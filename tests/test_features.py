"""
Unit tests for OptiRAG-lite Feature Extraction Module (features.py).
Zero LLM calls, deterministic local tests.
"""

import numpy as np
import pytest

import config
import features
from retrieve import Retriever


@pytest.fixture(scope="module")
def shared_retriever():
    return Retriever()


def test_query_features_shape_and_norm(shared_retriever):
    query = "Where was the 1996 Summer Olympics held?"
    feat = features.extract_query_features(query, retriever=shared_retriever)
    
    assert feat.shape == (385,)
    # First 384 dimensions are normalized MiniLM embedding
    emb = feat[:384]
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4
    
    # 385th dimension is query token count
    tok_count = feat[384]
    assert tok_count > 0
    assert tok_count == float(features.count_tokens(query))


def test_telemetry_features_shape_and_values(shared_retriever):
    sample_chunks = [
        ("Atlanta hosted the 1996 Olympic games in Centennial Olympic Park.", "1996 Olympics", 0.88),
        ("The Games of the XXVI Olympiad took place in Georgia.", "Centennial Games", 0.76),
        ("Summer Olympic games are held every four years by IOC.", "Olympic Games", 0.65),
        ("Athens was the host city for the 2004 Summer Olympics.", "Athens 2004", 0.52),
        ("Sydney hosted the subsequent games in 2000.", "Sydney 2000", 0.49),
    ]

    telemetry = features.extract_telemetry_features(sample_chunks, retriever=shared_retriever)
    assert telemetry.shape == (8,)

    s1, s2, gap, mean_s, std_s, entropy, mean_cos, ctx_tokens = telemetry

    # Verify mathematics
    assert abs(s1 - 0.88) < 1e-4
    assert abs(s2 - 0.76) < 1e-4
    assert abs(gap - (0.88 - 0.76)) < 1e-4
    assert s1 >= s2
    assert mean_s > 0
    assert std_s >= 0
    assert entropy > 0  # Softmax entropy must be strictly positive
    assert -1.0 <= mean_cos <= 1.0
    assert ctx_tokens > 0


def test_both_features_concatenation(shared_retriever):
    query = "What is the capital of France?"
    sample_chunks = [
        ("Paris is the capital and most populous city of France.", "Paris", 0.92),
        ("France is a country located in Western Europe.", "France", 0.80),
        ("The Seine river flows through the heart of Paris.", "Seine", 0.70),
        ("Lyon is the third-largest city in France.", "Lyon", 0.60),
        ("Marseille is the second largest city in France.", "Marseille", 0.55),
    ]

    q_feat = features.extract_features(query, sample_chunks, variant="query-only", retriever=shared_retriever)
    t_feat = features.extract_features(query, sample_chunks, variant="telemetry-only", retriever=shared_retriever)
    b_feat = features.extract_features(query, sample_chunks, variant="both", retriever=shared_retriever)

    assert q_feat.shape == (385,)
    assert t_feat.shape == (8,)
    assert b_feat.shape == (393,)

    # Verify exact concatenation match
    np.testing.assert_allclose(b_feat[:385], q_feat, rtol=1e-5)
    np.testing.assert_allclose(b_feat[385:], t_feat, rtol=1e-5)


def test_telemetry_edge_cases(shared_retriever):
    # Test with fewer than 5 chunks
    short_chunks = [
        ("A single chunk text.", "Title A", 0.85),
    ]
    telemetry = features.extract_telemetry_features(short_chunks, retriever=shared_retriever)
    assert telemetry.shape == (8,)
    assert not np.isnan(telemetry).any()
    assert not np.isinf(telemetry).any()

    # Test with empty chunks
    empty_chunks = []
    telemetry_empty = features.extract_telemetry_features(empty_chunks, retriever=shared_retriever)
    assert telemetry_empty.shape == (8,)
    assert not np.isnan(telemetry_empty).any()
    assert not np.isinf(telemetry_empty).any()


def test_invalid_variant_raises_value_error(shared_retriever):
    with pytest.raises(ValueError):
        features.extract_features("query", [], variant="invalid_variant", retriever=shared_retriever)
