"""
Unit tests for OptiRAG-lite Protected Token Mask (PTM).
"""

import pytest
import ptm


def test_revenue_is_not_growing():
    """Verify 'not' is identified as negation."""
    text = "Revenue is not growing"
    spans = ptm.find_protected_spans(text)
    assert len(spans) == 1
    start, end, kind = spans[0]
    assert text[start:end] == "not"
    assert kind == "negation"
    assert ptm.protected_tokens(text) == ["not"]


def test_financial_entities_and_percentages():
    """Verify $1.5M, Q3 2023, and 15% are protected."""
    text = "$1.5M in Q3 2023, up 15%"
    spans = ptm.find_protected_spans(text)
    spans_text = ptm.protected_spans_text(text)
    tokens = ptm.protected_tokens(text)

    assert "$1.5M" in spans_text
    assert "15%" in spans_text
    # Q3 2023 can be captured as a date span
    assert any("Q3" in s for s in spans_text)
    assert any("2023" in s for s in spans_text)

    # Tokens check
    assert "$1.5M" in tokens
    assert "15%" in tokens
    assert "2023" in tokens
    assert "Q3" in tokens


def test_date_formats_single_spans():
    """Verify various standard date formats are extracted as single complete spans."""
    # Format A: numeric
    text1 = "Meeting on 03/04/2024."
    spans1 = ptm.find_protected_spans(text1)
    assert len(spans1) == 1
    assert spans1[0][2] == "date"
    assert text1[spans1[0][0]:spans1[0][1]] == "03/04/2024"

    # Format B: Month Day, Year
    text2 = "Signed on March 4, 2024 at noon."
    spans2 = ptm.find_protected_spans(text2)
    assert len(spans2) == 1
    assert spans2[0][2] == "date"
    assert text2[spans2[0][0]:spans2[0][1]] == "March 4, 2024"

    # Format C: Day Month Year
    text3 = "Event on 4 March 2024 concluded."
    spans3 = ptm.find_protected_spans(text3)
    assert len(spans3) == 1
    assert spans3[0][2] == "date"
    assert text3[spans3[0][0]:spans3[0][1]] == "4 March 2024"


def test_negation_words_and_contractions():
    """Verify contractions and specific negation keywords are protected."""
    text = "We don't know, isn't that clear? You cannot leave except without permission."
    tokens = ptm.protected_tokens(text)

    expected = ["cannot", "don't", "except", "isn't", "without"]
    for word in expected:
        assert word in tokens, f"Expected {word} to be protected"


def test_formatted_numbers_with_commas():
    """Verify numbers with thousand-separator commas are captured as a single span."""
    text = "The population is 1,250,000 residents."
    spans = ptm.find_protected_spans(text)
    assert len(spans) == 1
    start, end, kind = spans[0]
    assert kind == "number"
    assert text[start:end] == "1,250,000"
    assert ptm.protected_tokens(text) == ["1,250,000"]


def test_no_protected_spans_on_plain_entities():
    """Verify named entity without numbers/dates/negations yields 0 protected spans."""
    text = "Knox County Regional Airport"
    spans = ptm.find_protected_spans(text)
    assert spans == []
    assert ptm.protected_tokens(text) == []
    stats = ptm.protection_stats(text)
    assert stats["protected_words"] == 0
    assert stats["spans_count"] == 0


def test_negation_boundary_false_positives():
    """Verify word boundaries prevent matching inside 'another' or 'notable'."""
    text = "Another notable feature is that nothing and nobody was seen."
    tokens = ptm.protected_tokens(text)

    assert "nothing" in tokens
    assert "nobody" in tokens
    assert "another" not in tokens
    assert "notable" not in tokens
    assert "Another" not in tokens


def test_empty_string():
    """Verify empty string is handled gracefully without errors."""
    assert ptm.find_protected_spans("") == []
    assert ptm.protected_spans_text("") == []
    assert ptm.protected_tokens("") == []
    stats = ptm.protection_stats("")
    assert stats["total_words"] == 0
    assert stats["protected_words"] == 0
    assert stats["protected_word_pct"] == 0.0
