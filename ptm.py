"""
OptiRAG-lite Protected Token Mask (PTM).
Identifies high-entropy critical entities (numbers, dates, currency, percentages, negation)
to force-keep during safe context compression, preventing semantic hallucinations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

# Categories of protected entities
KINDS = ("currency", "date", "percent", "negation", "number")

# Kind precedence when resolving overlapping spans (higher wins)
PRIORITY = {
    "currency": 5,
    "date": 4,
    "percent": 3,
    "negation": 2,
    "number": 1,
}

# Negation words and contractions
_NEGATION_WORDS = (
    r"\b(?:not|no|never|none|nor|neither|nobody|nothing|nowhere|except|without|cannot)\b"
)
_NEGATION_CONTRACTIONS = r"\b\w+(?:n['’]t)\b"

# Months regex
_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

# Currency symbols and codes
_CURRENCY_SYMBOLS = r"[$€£¥₹]|(?:Rs\.?)"
_CURRENCY_CODES = r"(?:USD|EUR|GBP|JPY|INR|CAD|AUD|dollars?|cents?|euros?|pounds?|rupees?|yen)"

# Compiled Regex Patterns
RE_CURRENCY = re.compile(
    rf"(?:({_CURRENCY_SYMBOLS})\s*\d+(?:,\d{{3}})*(?:\.\d+)?(?:\s*(?:billion|million|trillion|k|m|b|thousand|crore|lakh))?\b)|"
    rf"(?:({_CURRENCY_SYMBOLS})\s*\d+(?:\.\d+)?[MBKmbk]\b)|"
    rf"(?:\b\d+(?:,\d{{3}})*(?:\.\d+)?\s*{_CURRENCY_CODES}\b)",
    re.IGNORECASE,
)

RE_PERCENT = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage)\b|\b\d+(?:\.\d+)?%",
    re.IGNORECASE,
)

RE_DATE = re.compile(
    # Quarters: Q1 2023, Q3 2024, Q2
    r"\bQ[1-4](?:\s*(?:19|20)\d{2})?\b|"
    # ISO dates: 2024-03-04
    r"\b\d{4}-\d{2}-\d{2}\b|"
    # Numeric dates: 03/04/2024, 03-04-2024
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    # Month Day, Year: March 4, 2024 or March 4 2024
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b|"
    # Day Month Year: 4 March 2024 or 4th March, 2024
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS},?\s+(?:19|20)\d{{2}}\b|"
    # Month Day: March 4 or March 4th
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b|"
    # Month Year: March 2024
    rf"\b{_MONTHS}\s+(?:19|20)\d{{2}}\b|"
    # Decades: 1990s, 2020s
    r"\b(?:19|20)\d{2}s\b",
    re.IGNORECASE,
)

RE_NEGATION = re.compile(
    rf"{_NEGATION_WORDS}|{_NEGATION_CONTRACTIONS}",
    re.IGNORECASE,
)

RE_NUMBER = re.compile(
    # Ordinals: 1st, 2nd, 3rd, 24th
    r"\b\d+(?:st|nd|rd|th)\b|"
    # Numbers with commas: 1,250,000
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|"
    # Decimals: 3.1415
    r"\b\d+\.\d+\b|"
    # Integers: 1999, 42
    r"\b\d+\b",
    re.IGNORECASE,
)


def find_protected_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all character-level protected spans (start, end, kind).
    Merges overlapping or touching spans, giving precedence to more specific kinds.

    Returns:
        List of (start_idx, end_idx, kind) sorted by start_idx.
    """
    if not text:
        return []

    raw_spans: List[Tuple[int, int, str]] = []

    # 1. Collect all candidates
    for m in RE_CURRENCY.finditer(text):
        raw_spans.append((m.start(), m.end(), "currency"))
    for m in RE_PERCENT.finditer(text):
        raw_spans.append((m.start(), m.end(), "percent"))
    for m in RE_DATE.finditer(text):
        raw_spans.append((m.start(), m.end(), "date"))
    for m in RE_NEGATION.finditer(text):
        raw_spans.append((m.start(), m.end(), "negation"))
    for m in RE_NUMBER.finditer(text):
        raw_spans.append((m.start(), m.end(), "number"))

    if not raw_spans:
        return []

    # Sort spans by start asc, length desc
    raw_spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    # 2. Merge overlapping or touching spans
    merged: List[Tuple[int, int, str]] = []
    for cur in raw_spans:
        if not merged:
            merged.append(cur)
            continue

        prev_start, prev_end, prev_kind = merged[-1]
        cur_start, cur_end, cur_kind = cur

        # Overlapping or contiguous spans
        if cur_start <= prev_end:
            new_start = min(prev_start, cur_start)
            new_end = max(prev_end, cur_end)
            # Pick kind with higher priority
            resolved_kind = (
                prev_kind
                if PRIORITY.get(prev_kind, 0) >= PRIORITY.get(cur_kind, 0)
                else cur_kind
            )
            merged[-1] = (new_start, new_end, resolved_kind)
        else:
            merged.append(cur)

    return merged


def protected_spans_text(text: str) -> List[str]:
    """Return raw substring for each protected span."""
    spans = find_protected_spans(text)
    return [text[start:end] for start, end, _ in spans]


def protected_tokens(text: str) -> List[str]:
    """
    Return sorted unique list of clean protected tokens.
    Splits spans into words and strips surrounding punctuation.
    Usable as force-keep token list in LLMLingua-2.
    """
    spans = find_protected_spans(text)
    tokens: Set[str] = set()

    # Punctuation to strip from word edges (preserving symbols like $, %, etc.)
    punct_to_strip = ",;:()[]{}\"'“”‘’!?"

    for start, end, _ in spans:
        span_str = text[start:end]
        for word in span_str.split():
            clean = word.strip(punct_to_strip)
            # Strip trailing period only if not part of abbreviation or number
            if clean.endswith(".") and not clean.replace(".", "").isdigit():
                clean = clean[:-1]
            if clean:
                tokens.add(clean)

    return sorted(tokens)


def protection_stats(text: str) -> Dict[str, Any]:
    """
    Compute protection statistics for a given text.
    Reports counts per kind, word counts, and protected percentages.
    """
    if not text:
        return {
            "total_words": 0,
            "protected_words": 0,
            "protected_word_pct": 0.0,
            "total_chars": 0,
            "protected_chars": 0,
            "protected_char_pct": 0.0,
            "by_kind": {k: 0 for k in KINDS},
            "spans_count": 0,
        }

    spans = find_protected_spans(text)
    by_kind = {k: 0 for k in KINDS}
    for _, _, kind in spans:
        by_kind[kind] = by_kind.get(kind, 0) + 1

    # Word-level coverage
    all_words = text.split()
    total_words = len(all_words)

    # A word is protected if any protected span overlaps with its character boundaries
    protected_word_count = 0
    cur_pos = 0
    for w in all_words:
        w_start = text.find(w, cur_pos)
        w_end = w_start + len(w)
        cur_pos = w_end

        is_protected = any(
            max(w_start, span_start) < min(w_end, span_end)
            for span_start, span_end, _ in spans
        )
        if is_protected:
            protected_word_count += 1

    protected_chars = sum(end - start for start, end, _ in spans)

    return {
        "total_words": total_words,
        "protected_words": protected_word_count,
        "protected_word_pct": round(protected_word_count / total_words * 100, 2)
        if total_words > 0
        else 0.0,
        "total_chars": len(text),
        "protected_chars": protected_chars,
        "protected_char_pct": round(protected_chars / len(text) * 100, 2)
        if len(text) > 0
        else 0.0,
        "by_kind": by_kind,
        "spans_count": len(spans),
    }
