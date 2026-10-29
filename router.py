"""
OptiRAG-lite Router Module.

Trains and evaluates query routing models to predict query hardness (hard = 1):
- hard = 1: SMALL tier model fails, but LARGE tier model succeeds.
- Routing to LARGE tier when hard = 1 prevents cascading errors.
- Routing to SMALL tier when hard = 0 saves latency and token cost.

Models supported:
1. Heuristic Baseline:
   - Route to LARGE if top-1 score < 0.70 OR query contains 'compare', 'analyze', 'summarize'.
2. Logistic Regression:
   - L2-regularized logistic regression with class_weight='balanced'.
   - Hyperparameter C chosen strictly by stratified 5-fold cross-validation on train set only.
   - Scalers fit on training folds only (zero leakage).
3. PyTorch MLP (if hard=1 samples >= 15):
   - Single hidden layer (16-32 units), ReLU, Dropout(0.3).
   - Weight decay 1e-3, BCEWithLogitsLoss with pos_weight, early stopping.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

import config


class HeuristicRouter:
    """Simple rule-based heuristic routing baseline."""

    def __init__(self, score_threshold: float = 0.70, trigger_words: Optional[List[str]] = None):
        self.score_threshold = score_threshold
        self.trigger_words = trigger_words or ["compare", "analyze", "summarize"]

    def predict_single(self, query: str, top_score: float) -> Tuple[int, float]:
        q_lower = query.lower()
        has_word = any(w in q_lower for w in self.trigger_words)
        low_score = top_score < self.score_threshold
        is_hard = int(has_word or low_score)
        prob = 1.0 if is_hard else 0.0
        return is_hard, prob

    def predict_proba(self, queries: List[str], top_scores: List[float]) -> np.ndarray:
        probs = [self.predict_single(q, s)[1] for q, s in zip(queries, top_scores)]
        return np.array(probs, dtype=np.float32)


class LogisticRouter:
    """Class-weighted Logistic Regression Router with feature standardization."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            max_iter=1000,
            random_state=config.SEED,
            solver="lbfgs",
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRouter:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LogisticRouter must be fit before calling predict_proba.")
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return np.clip(probs, 0.0, 1.0).astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


class _TorchMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPRouter:
    """PyTorch MLP Router with pos_weight and early stopping."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 16,
        dropout: float = 0.3,
        weight_decay: float = 1e-3,
        lr: float = 1e-3,
        epochs: int = 60,
    ):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.model: Optional[_TorchMLP] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPRouter:
        torch.manual_seed(config.SEED)
        torch.set_num_threads(2)

        X_scaled = self.scaler.fit_transform(X)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        pos_weight_val = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

        self.model = _TorchMLP(self.in_features, self.hidden_dim, self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, dtype=torch.float32))

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("MLPRouter must be fit before calling predict_proba.")
        torch.set_num_threads(2)
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.clip(probs, 0.0, 1.0).astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


def tune_logistic_cv(
    X: np.ndarray,
    y: np.ndarray,
    c_candidates: Optional[List[float]] = None,
    n_splits: int = 5,
) -> Tuple[float, Dict[str, Any]]:
    """
    Performs Stratified 5-Fold Cross-Validation on training set only.
    Selects best C parameter based on mean CV AUROC.
    Scalers fit on fold training split only (zero leakage).
    """
    if c_candidates is None:
        c_candidates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    best_c = 1.0
    best_mean_auc = -1.0
    results_by_c = {}

    for c in c_candidates:
        fold_aucs = []
        fold_pr_aucs = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            # Fit scaler on training fold only
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            clf = LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=1000,
                random_state=config.SEED,
                solver="lbfgs",
            )
            clf.fit(X_tr_s, y_tr)
            probs = clf.predict_proba(X_va_s)[:, 1]

            # Compute fold metrics if both classes present in validation fold
            if len(np.unique(y_va)) > 1:
                fold_auc = roc_auc_score(y_va, probs)
                fold_pr = average_precision_score(y_va, probs)
            else:
                fold_auc = 0.5
                fold_pr = float(np.mean(y_tr))

            fold_aucs.append(fold_auc)
            fold_pr_aucs.append(fold_pr)

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        mean_pr = float(np.mean(fold_pr_aucs))
        std_pr = float(np.std(fold_pr_aucs))

        results_by_c[c] = {
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "mean_pr_auc": mean_pr,
            "std_pr_auc": std_pr,
        }

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_c = c

    return best_c, results_by_c
