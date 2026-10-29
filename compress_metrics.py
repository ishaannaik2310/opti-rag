"""
Metrics for OptiRAG-lite Context Compression.
Evaluates entity survival rate (PTM entities) and gold answer survival.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set

import ptm
from metrics import normalize_answer


def entity_survival_rate(
    original: str, compressed: str, kinds: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes fraction of protected span strings from original that survive in compressed text.

    Args:
        original: Full uncompressed text
        compressed: Compressed text
        kinds: Optional list of kinds to evaluate (e.g. ['number', 'date']). Defaults to all.

    Returns:
        dict containing:
            overall_rate (float): 0.0 to 1.0
            kept_count (int): number of preserved spans
            total_count (int): total protected spans in original
            by_kind (dict): per-kind breakdown with kept, total, and rate
    """
    spans = ptm.find_protected_spans(original)
    if kinds is not None:
        kinds_set = set(kinds)
        spans = [s for s in spans if s[2] in kinds_set]

    if not spans:
        return {
            "overall_rate": 1.0,
            "kept_count": 0,
            "total_count": 0,
            "by_kind": {},
        }

    comp_lower = compressed.lower()
    by_kind: Dict[str, Dict[str, Any]] = {}
    overall_kept = 0

    for start, end, kind in spans:
        span_str = original[start:end].strip().lower()
        if kind not in by_kind:
            by_kind[kind] = {"kept": 0, "total": 0, "rate": 0.0}
        by_kind[kind]["total"] += 1

        # Check if preserved in compressed text
        if span_str and span_str in comp_lower:
            by_kind[kind]["kept"] += 1
            overall_kept += 1

    for k, data in by_kind.items():
        data["rate"] = (
            round(data["kept"] / data["total"], 4) if data["total"] > 0 else 1.0
        )

    return {
        "overall_rate": round(overall_kept / len(spans), 4),
        "kept_count": overall_kept,
        "total_count": len(spans),
        "by_kind": by_kind,
    }


def gold_answer_survival(original: str, compressed: str, gold: str) -> Optional[bool]:
    """
    Evaluates whether the gold answer survives compression.

    Rules:
    - If gold is 'yes' or 'no', returns None (not an extractive text span).
    - If normalized gold answer does NOT occur in original context, returns None
      (nothing was present to lose).
    - If normalized gold answer is present in original context:
      returns True if present in compressed, False if dropped.
    """
    if not gold:
        return None

    clean_gold = gold.strip().lower()
    if clean_gold in {"yes", "no"}:
        return None

    norm_gold = normalize_answer(gold)
    if not norm_gold:
        return None

    norm_orig = normalize_answer(original)
    if norm_gold not in norm_orig:
        return None

    norm_comp = normalize_answer(compressed)
    return norm_gold in norm_comp
