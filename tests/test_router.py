"""
Unit tests for OptiRAG-lite Router Module (router.py).
Tests feature dimensions (385 / 8 / 393), probabilities in [0, 1], and all models.
Zero network calls.
"""

import numpy as np
import pytest

from router import HeuristicRouter, LogisticRouter, MLPRouter, tune_logistic_cv


def test_heuristic_router():
    router = HeuristicRouter(score_threshold=0.70)
    
    # Low score -> hard=1, prob=1.0
    is_hard, prob = router.predict_single("Where was the match held?", 0.65)
    assert is_hard == 1
    assert prob == 1.0

    # High score, no trigger word -> hard=0, prob=0.0
    is_hard, prob = router.predict_single("Where was the match held?", 0.85)
    assert is_hard == 0
    assert prob == 0.0

    # High score, but has trigger word 'compare' -> hard=1, prob=1.0
    is_hard, prob = router.predict_single("Compare the two authors.", 0.85)
    assert is_hard == 1
    assert prob == 1.0

    # Batch test
    queries = ["What is X?", "Compare A and B", "Who won?"]
    scores = [0.80, 0.90, 0.50]
    probs = router.predict_proba(queries, scores)
    assert probs.shape == (3,)
    assert set(probs).issubset({0.0, 1.0})


@pytest.mark.parametrize("dim", [385, 8, 393])
def test_logistic_router_dimensions_and_probabilities(dim):
    np.random.seed(42)
    N = 40
    X = np.random.randn(N, dim).astype(np.float32)
    # Balanced classes
    y = np.array([1 if i % 4 == 0 else 0 for i in range(N)], dtype=int)

    clf = LogisticRouter(C=0.1)
    clf.fit(X, y)

    X_test = np.random.randn(10, dim).astype(np.float32)
    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test, threshold=0.5)

    assert probs.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("dim", [8, 385, 393])
def test_mlp_router_dimensions_and_probabilities(dim):
    np.random.seed(42)
    N = 40
    X = np.random.randn(N, dim).astype(np.float32)
    y = np.array([1 if i % 4 == 0 else 0 for i in range(N)], dtype=int)

    mlp = MLPRouter(in_features=dim, hidden_dim=16, epochs=5)
    mlp.fit(X, y)

    X_test = np.random.randn(10, dim).astype(np.float32)
    probs = mlp.predict_proba(X_test)
    preds = mlp.predict(X_test, threshold=0.5)

    assert probs.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert set(preds).issubset({0, 1})


def test_tune_logistic_cv():
    np.random.seed(42)
    N = 50
    dim = 8
    X = np.random.randn(N, dim).astype(np.float32)
    y = np.array([1 if i % 3 == 0 else 0 for i in range(N)], dtype=int)

    best_c, results = tune_logistic_cv(X, y, c_candidates=[0.01, 1.0], n_splits=3)
    assert best_c in [0.01, 1.0]
    assert len(results) == 2
    assert "mean_auc" in results[best_c]
    assert 0.0 <= results[best_c]["mean_auc"] <= 1.0
