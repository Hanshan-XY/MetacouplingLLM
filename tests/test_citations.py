"""Tests for knowledge/citations.py — citation sanitization."""

from __future__ import annotations

import logging

import pytest

from metacouplingllm.knowledge.citations import (
    CITATION_PATTERN,
    extract_cited_ids,
    sanitize_citations,
)


# ---------------------------------------------------------------------------
# Core sanitizer behaviour
# ---------------------------------------------------------------------------


class TestSanitizeCitations:
    def test_all_valid_preserved(self):
        text = "Soybean trade [1] grew rapidly. Mato Grosso [2] is key."
        result, invalid = sanitize_citations(text, n_valid=2)
        assert result == "Soybean trade [1] grew rapidly. Mato Grosso [2] is key."
        assert invalid == set()

    def test_invalid_stripped(self):
        text = "Claim A [1] and claim B [99]."
        result, invalid = sanitize_citations(text, n_valid=2)
        assert "[99]" not in result
        assert "[1]" in result
        assert invalid == {99}

    def test_zero_valid_strips_all(self):
        text = "Citing [1] and [2] and [42]."
        result, invalid = sanitize_citations(text, n_valid=0)
        assert "[1]" not in result
        assert "[2]" not in result
        assert "[42]" not in result
        assert invalid == {1, 2, 42}

    def test_no_citations_unchanged(self):
        text = "Plain text with no citations whatsoever."
        result, invalid = sanitize_citations(text, n_valid=5)
        assert result == "Plain text with no citations whatsoever."
        assert invalid == set()

    def test_web_citations_not_touched(self):
        text = "Web source [W1] is preserved alongside paper [1]."
        result, invalid = sanitize_citations(text, n_valid=1)
        assert "[W1]" in result
        assert "[1]" in result
        assert invalid == set()

    def test_boundary_conditions(self):
        # [1] and [n_valid] are valid; [0] and [n_valid+1] are not.
        text = "Tokens [0] [1] [3] [4]."
        result, invalid = sanitize_citations(text, n_valid=3)
        assert "[1]" in result
        assert "[3]" in result
        assert "[0]" not in result
        assert "[4]" not in result
        assert invalid == {0, 4}

    def test_multiple_occurrences_of_same_invalid(self):
        # Set semantics: [99] appearing 5 times yields invalid_ids == {99}.
        text = "Bad [99] x [99] y [99] z [99] w [99]."
        result, invalid = sanitize_citations(text, n_valid=2)
        assert "[99]" not in result
        assert invalid == {99}

    def test_returns_tuple(self):
        result = sanitize_citations("plain", n_valid=0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], set)


# ---------------------------------------------------------------------------
# Logging behaviour
# ---------------------------------------------------------------------------


class TestSanitizeCitationsLogging:
    def test_logs_warning_on_invalid(self, caplog):
        with caplog.at_level(logging.WARNING, logger="metacouplingllm.knowledge.citations"):
            sanitize_citations("Bad [99] and [100].", n_valid=5)
        assert any("99" in record.message for record in caplog.records)
        assert any("100" in record.message for record in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_no_log_when_all_valid(self, caplog):
        with caplog.at_level(logging.WARNING, logger="metacouplingllm.knowledge.citations"):
            sanitize_citations("Clean [1] [2] [3].", n_valid=3)
        # No warning records from the citations logger
        citations_records = [
            r for r in caplog.records
            if r.name == "metacouplingllm.knowledge.citations"
        ]
        assert citations_records == []


# ---------------------------------------------------------------------------
# Whitespace cleanup pass (Decision #11)
# ---------------------------------------------------------------------------


class TestCleanupAfterStrip:
    def test_collapses_double_spaces_after_strip(self):
        # Stripping [99] from "claim [99] and more" naively leaves
        # "claim  and more" — cleanup must collapse the double space.
        text = "claim [99] and more"
        result, _ = sanitize_citations(text, n_valid=0)
        assert "  " not in result
        assert result == "claim and more"

    def test_removes_space_before_period(self):
        # Stripping [99] from "claim [99]." leaves "claim ." —
        # cleanup must remove the orphan space before the period.
        text = "claim [99]."
        result, _ = sanitize_citations(text, n_valid=0)
        assert result == "claim."

    def test_removes_space_before_other_punctuation(self):
        text = "first [99], second [99]; third [99]: fourth [99]!"
        result, _ = sanitize_citations(text, n_valid=0)
        assert result == "first, second; third: fourth!"

    def test_preserves_newlines(self):
        text = "Line one [99].\nLine two [99].\nLine three."
        result, _ = sanitize_citations(text, n_valid=0)
        assert "\n" in result
        assert result.count("\n") == 2
        assert "Line one." in result
        assert "Line two." in result
        assert "Line three." in result

    def test_idempotent_on_clean_text(self):
        # Already-clean text with valid citations passes through unchanged
        # (modulo whitespace which is already normal).
        text = "Claim A [1] and claim B [2]."
        result, _ = sanitize_citations(text, n_valid=2)
        assert result == text


# ---------------------------------------------------------------------------
# extract_cited_ids
# ---------------------------------------------------------------------------


class TestExtractCitedIds:
    def test_basic(self):
        assert extract_cited_ids("[1] foo [2] bar [1]") == {1, 2}

    def test_empty(self):
        assert extract_cited_ids("plain text") == set()

    def test_ignores_web_citations(self):
        assert extract_cited_ids("paper [1] web [W1] paper [2]") == {1, 2}

    def test_pattern_constant_exported(self):
        # Sanity check that the regex is reusable for downstream tooling
        assert CITATION_PATTERN.findall("[1] [42]") == ["1", "42"]
