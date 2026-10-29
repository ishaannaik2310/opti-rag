"""
OptiRAG-lite Evaluation Metrics.

Implements SQuAD/HotpotQA-style text normalization, Content-Token F1, Exact Match (EM),
and a robust correctness classifier tailored for RAG multi-hop evaluations.
"""

import re
import string
from collections import Counter
from typing import Dict, Any, List

# Stopwords excluded during content-token F1 computation to prevent false positives
STOPWORDS = {
    "of", "in", "on", "at", "to", "for", "by", "with", "and", "or",
    "is", "was", "were", "are", "be", "been", "that", "which", "who", "what", "from", "as"
}


def normalize_answer(s: str) -> str:
    """
    Standard SQuAD-style string normalization:
    - Lowercase
    - Strip punctuation
    - Remove English articles ('a', 'an', 'the')
    - Collapse extra whitespace
    """
    if not s:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Returns True if normalized prediction exactly equals normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def get_content_tokens(text: str) -> List[str]:
    """Extract content tokens after SQuAD normalization by filtering common stopwords."""
    tokens = normalize_answer(text).split()
    content = [t for t in tokens if t not in STOPWORDS]
    return content if content else tokens


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Computes token-level F1 score over CONTENT tokens (filtering stopwords).
    Prevents false positives on low-content overlapping phrases (e.g. 'battle of').
    """
    pred_tokens = get_content_tokens(prediction)
    gold_tokens = get_content_tokens(ground_truth)

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def is_correct(prediction: str, ground_truth: str) -> bool:
    """
    Evaluates whether a model prediction is correct according to OptiRAG criteria:

    Rules:
    1. Empty responses or [EMPTY_RESPONSE...] / [NO_CONTENT...] markers are ALWAYS incorrect.
    2. 'I don't know' predictions are ALWAYS incorrect.
    3. Special case for Yes/No questions:
       Correct ONLY if the first word of the normalized prediction equals the gold ('yes' or 'no').
    4. General questions:
       Correct if:
       - Exact Match (EM) == True
       OR
       - Content Token F1 >= 0.5
       OR
       - The normalized gold answer appears as a whole phrase inside the normalized prediction,
         AND (the prediction has at most 12 words OR the specific line/sentence containing
         the gold phrase has at most 12 words).
    """
    if not prediction or not prediction.strip():
        return False

    raw_lower = prediction.strip().lower()
    if "[empty_response" in raw_lower or "[no_content" in raw_lower:
        return False

    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)

    if not norm_pred or not norm_gold:
        return False

    # Predictions of "i don't know" are always incorrect
    if norm_pred in ["i dont know", "i do not know"]:
        return False
    if norm_pred.startswith("i dont know") and len(norm_pred.split()) <= 4:
        return False

    # Special case: Yes / No questions
    if norm_gold in ["yes", "no"]:
        pred_words = norm_pred.split()
        if not pred_words:
            return False
        return pred_words[0] == norm_gold

    # 1. Exact Match
    if norm_pred == norm_gold:
        return True

    # 2. Content Token F1 >= 0.5
    f1 = compute_f1(prediction, ground_truth)
    if f1 >= 0.5:
        return True

    # 3. Whole phrase containment check
    phrase_pattern = r"\b" + re.escape(norm_gold) + r"\b"
    if re.search(phrase_pattern, norm_pred):
        # Case A: Entire prediction is at most 12 words
        if len(norm_pred.split()) <= 12:
            return True

        # Case B: The line or trailing answer clause containing the gold phrase is at most 12 words
        # (e.g., 'Therefore, the answer is "Knox County Regional Airport".')
        for line in prediction.splitlines():
            norm_line = normalize_answer(line)
            if re.search(phrase_pattern, norm_line) and len(norm_line.split()) <= 12:
                return True

    return False


def evaluate_qa_pair(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Helper returning dict of em, f1, and is_correct."""
    return {
        "exact_match": compute_exact_match(prediction, ground_truth),
        "f1": compute_f1(prediction, ground_truth),
        "is_correct": is_correct(prediction, ground_truth),
    }
