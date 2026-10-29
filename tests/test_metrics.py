"""
Unit tests for OptiRAG-lite metrics and correctness classifier.
"""

import pytest
import metrics


def test_bosworth_false_positive_eliminated():
    """
    Critical test: 'Battle of Bosworth' vs gold 'Second Battle of St Albans'.
    With content-F1, content overlap is only ['battle'] out of ['battle', 'bosworth'] and
    ['second', 'battle', 'st', 'albans'], giving content-F1 = 0.333 < 0.5.
    Must be classified as INCORRECT.
    """
    pred = "Battle of Bosworth"
    gold = "Second Battle of St Albans"
    assert metrics.compute_exact_match(pred, gold) is False
    assert metrics.compute_f1(pred, gold) == pytest.approx(0.3333, rel=1e-2)
    assert metrics.is_correct(pred, gold) is False


def test_transcendentalist_movement():
    """
    'transcendentalist movement' vs gold 'transcendentalist'.
    'transcendentalist' is contained as a whole phrase in a 2-word prediction (<= 12 words),
    and content-F1 = 0.667 >= 0.5. Must be classified as CORRECT.
    """
    pred = "transcendentalist movement"
    gold = "transcendentalist"
    assert metrics.compute_f1(pred, gold) == pytest.approx(0.6666, rel=1e-2)
    assert metrics.is_correct(pred, gold) is True


def test_verbose_knox_county_answer():
    """Verbose multi-sentence answer ending in Knox County Regional Airport should be correct."""
    pred = (
        'The context mentions that Knox County Regional Airport is located in Maine, '
        'while Sacramento International Airport is located in California. \n'
        'Therefore, the answer is "Knox County Regional Airport".'
    )
    gold = "Knox County Regional Airport"
    assert metrics.is_correct(pred, gold) is True


def test_failsworth_inside_short_sentence():
    """Gold answer 'Failsworth' appearing inside an 11-word sentence should be correct."""
    pred = "Based on the context, Russell Hobbs is based in Failsworth, Greater Manchester."
    gold = "Failsworth"
    assert metrics.is_correct(pred, gold) is True


def test_yes_no_cases():
    """Yes/No evaluation requires the first word to match gold."""
    # Positive Yes case
    assert metrics.is_correct("Yes, both Gabriela Mistral and G. K. Chesterton were authors.", "yes") is True
    assert metrics.is_correct("yes", "yes") is True

    # Negative Yes case (answered no instead)
    assert metrics.is_correct("No, they were not both authors.", "yes") is False
    assert metrics.is_correct("Neither was an author.", "yes") is False

    # Positive No case
    assert metrics.is_correct("No, it was founded elsewhere.", "no") is True

    # False positive prevention: 'yes' buried later in sentence when gold is 'no'
    assert metrics.is_correct("It is not true, although yes was suggested.", "no") is False


def test_i_dont_know_is_incorrect():
    """'I don't know' predictions must always be classified as incorrect."""
    gold = "Failsworth"
    assert metrics.is_correct("I don't know.", gold) is False
    assert metrics.is_correct("i don't know", gold) is False
    assert metrics.is_correct("I do not know", gold) is False
    assert metrics.is_correct("Therefore, the answer is I don't know.", gold) is False


def test_empty_or_marker_is_incorrect():
    """Empty strings and [EMPTY_RESPONSE...] markers must be classified as incorrect."""
    gold = "Failsworth"
    assert metrics.is_correct("", gold) is False
    assert metrics.is_correct("   ", gold) is False
    assert metrics.is_correct("[EMPTY_RESPONSE: finish_reason=FinishReason.SAFETY]", gold) is False
    assert metrics.is_correct("[NO_CONTENT: finish_reason=MAX_TOKENS]", gold) is False


def test_exact_match_and_f1():
    """Basic EM and Content F1 score accuracy."""
    pred = "The Knox County Regional Airport"
    gold = "Knox County Regional Airport"
    assert metrics.compute_exact_match(pred, gold) is True
    assert metrics.compute_f1(pred, gold) == 1.0

    pred_partial = "Knox County"
    gold_full = "Knox County Regional Airport"
    assert metrics.compute_exact_match(pred_partial, gold_full) is False
    assert metrics.compute_f1(pred_partial, gold_full) == pytest.approx(0.6666, rel=1e-2)
    assert metrics.is_correct(pred_partial, gold_full) is True
