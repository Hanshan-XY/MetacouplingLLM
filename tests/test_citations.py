"""Tests for knowledge/citations.py — turn-scoped citation sanitization."""

from __future__ import annotations

import logging

import pytest

from metacouplingllm.knowledge.citations import (
    TURN_CITATION_PATTERN,
    extract_turn_cited_ids,
    sanitize_turn_citations,
)


# ---------------------------------------------------------------------------
# Turn-scoped sanitizer — the only public sanitizer in v0.1.3+
# ---------------------------------------------------------------------------


class TestSanitizeTurnCitations:
    """``sanitize_turn_citations`` validates ``[Tk:N]`` and ``[Tk:Wn]``
    tokens against per-turn passage and web counts."""

    def test_current_turn_valid_preserved(self):
        text = "Claim A [T1:1] and claim B [T1:2]."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5},
            turn_web_counts={},
            current_turn=1,
        )
        assert "[T1:1]" in result
        assert "[T1:2]" in result
        assert invalid == set()

    def test_prior_turn_back_reference_preserved(self):
        # A turn-2 answer can back-reference [T1:3] as long as turn 1
        # had at least 3 passages.
        text = "As established in [T1:3], the new data [T2:1] extends it."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5, 2: 4},
            turn_web_counts={},
            current_turn=2,
        )
        assert "[T1:3]" in result
        assert "[T2:1]" in result
        assert invalid == set()

    def test_forward_reference_stripped(self):
        # turn 3 doesn't exist yet at current_turn=2.
        text = "See [T3:1] for details."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5, 2: 4},
            turn_web_counts={},
            current_turn=2,
        )
        assert "[T3:1]" not in result
        assert (3, "", 1) in invalid

    def test_turn_zero_invalid(self):
        # Turns are 1-indexed; T0 is never valid.
        text = "Bogus [T0:1] reference."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5},
            turn_web_counts={},
            current_turn=1,
        )
        assert "[T0:1]" not in result
        assert (0, "", 1) in invalid

    def test_out_of_range_passage_stripped(self):
        # Turn 1 had 5 passages; T1:99 is out of range.
        text = "Claim [T1:1] and bad [T1:99]."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5},
            turn_web_counts={},
            current_turn=1,
        )
        assert "[T1:1]" in result
        assert "[T1:99]" not in result
        assert (1, "", 99) in invalid

    def test_web_citations_preserved(self):
        text = "Trade data [T1:W1] shows growth."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5},
            turn_web_counts={1: 3},
            current_turn=1,
        )
        assert "[T1:W1]" in result
        assert invalid == set()

    def test_web_out_of_range_stripped(self):
        text = "Citing [T1:W1] and [T1:W99]."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5},
            turn_web_counts={1: 3},
            current_turn=1,
        )
        assert "[T1:W1]" in result
        assert "[T1:W99]" not in result
        assert (1, "W", 99) in invalid

    def test_web_from_turn_with_no_web_results(self):
        # turn 2 ran without web search; any [T2:Wn] is out of range.
        text = "Bogus [T2:W1]."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5, 2: 4},
            turn_web_counts={1: 3},  # turn 2 absent → counts as 0
            current_turn=2,
        )
        assert "[T2:W1]" not in result
        assert (2, "W", 1) in invalid

    def test_bare_legacy_tokens_stripped_silently(self):
        # Bare [N] and [W1] (pre-v0.1.3 grammar) are not a supported
        # input form. They are stripped silently as a defensive measure
        # against LLM slips — the invalid set does not report them.
        text = "Legacy [1] and [W1] tokens removed."
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5},
            turn_web_counts={1: 3},
            current_turn=1,
        )
        assert "[1]" not in result
        assert "[W1]" not in result
        assert invalid == set()

    def test_whitespace_cleanup_after_stripping(self):
        text = "claim [T1:99] and more."
        result, _ = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 5},
            turn_web_counts={},
            current_turn=1,
        )
        assert "  " not in result
        assert result == "claim and more."

    def test_mixed_current_and_prior_turn_with_invalids(self):
        text = (
            "Good [T1:2] and good [T2:1] and bad [T2:99] and "
            "very-bad [T9:1] and back-ref [T1:5]."
        )
        result, invalid = sanitize_turn_citations(
            text,
            turn_passage_counts={1: 8, 2: 4},
            turn_web_counts={},
            current_turn=2,
        )
        assert "[T1:2]" in result
        assert "[T2:1]" in result
        assert "[T1:5]" in result  # valid back-reference
        assert "[T2:99]" not in result
        assert "[T9:1]" not in result
        assert (2, "", 99) in invalid
        assert (9, "", 1) in invalid

    def test_returns_tuple(self):
        result = sanitize_turn_citations(
            "plain", {1: 0}, {}, current_turn=1
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], set)


# ---------------------------------------------------------------------------
# Logging behaviour
# ---------------------------------------------------------------------------


class TestSanitizeCitationsLogging:
    def test_logs_warning_on_invalid_turn_scoped(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="metacouplingllm.knowledge.citations"
        ):
            sanitize_turn_citations(
                "Bad [T1:99] and [T1:100].",
                turn_passage_counts={1: 5},
                turn_web_counts={},
                current_turn=1,
            )
        assert any("99" in record.message for record in caplog.records)
        assert any("100" in record.message for record in caplog.records)
        assert any(
            record.levelno == logging.WARNING for record in caplog.records
        )

    def test_no_log_when_all_valid(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="metacouplingllm.knowledge.citations"
        ):
            sanitize_turn_citations(
                "Clean [T1:1] [T1:2] [T1:3].",
                turn_passage_counts={1: 3},
                turn_web_counts={},
                current_turn=1,
            )
        # No warning records from the citations logger
        citations_records = [
            r for r in caplog.records
            if r.name == "metacouplingllm.knowledge.citations"
        ]
        assert citations_records == []

    def test_no_log_for_silent_bare_strip(self, caplog):
        # Bare-legacy stripping is silent — no warning.
        with caplog.at_level(
            logging.WARNING, logger="metacouplingllm.knowledge.citations"
        ):
            sanitize_turn_citations(
                "Some [1] bare [W1] tokens.",
                turn_passage_counts={1: 5},
                turn_web_counts={1: 3},
                current_turn=1,
            )
        citations_records = [
            r for r in caplog.records
            if r.name == "metacouplingllm.knowledge.citations"
        ]
        assert citations_records == []


# ---------------------------------------------------------------------------
# Whitespace cleanup pass
# ---------------------------------------------------------------------------


class TestCleanupAfterStrip:
    def test_collapses_double_spaces_after_strip(self):
        text = "claim [T1:99] and more"
        result, _ = sanitize_turn_citations(
            text, {1: 5}, {}, current_turn=1
        )
        assert "  " not in result
        assert result == "claim and more"

    def test_removes_space_before_period(self):
        text = "claim [T1:99]."
        result, _ = sanitize_turn_citations(
            text, {1: 5}, {}, current_turn=1
        )
        assert result == "claim."

    def test_removes_space_before_other_punctuation(self):
        text = "first [T1:99], second [T1:99]; third [T1:99]: fourth [T1:99]!"
        result, _ = sanitize_turn_citations(
            text, {1: 5}, {}, current_turn=1
        )
        assert result == "first, second; third: fourth!"

    def test_preserves_newlines(self):
        text = "Line one [T1:99].\nLine two [T1:99].\nLine three."
        result, _ = sanitize_turn_citations(
            text, {1: 5}, {}, current_turn=1
        )
        assert "\n" in result
        assert result.count("\n") == 2
        assert "Line one." in result
        assert "Line two." in result
        assert "Line three." in result

    def test_idempotent_on_clean_text(self):
        text = "Claim A [T1:1] and claim B [T1:2]."
        result, _ = sanitize_turn_citations(
            text, {1: 2}, {}, current_turn=1
        )
        assert result == text


# ---------------------------------------------------------------------------
# extract_turn_cited_ids
# ---------------------------------------------------------------------------


class TestExtractTurnCitedIds:
    def test_basic_literature(self):
        result = extract_turn_cited_ids("[T1:1] foo [T1:2] bar [T1:1]")
        assert result == {(1, "", 1), (1, "", 2)}

    def test_basic_web(self):
        result = extract_turn_cited_ids("citing [T1:W1] and [T2:W3]")
        assert result == {(1, "W", 1), (2, "W", 3)}

    def test_mixed(self):
        result = extract_turn_cited_ids(
            "paper [T1:1] web [T1:W2] paper [T2:3]"
        )
        assert result == {(1, "", 1), (1, "W", 2), (2, "", 3)}

    def test_ignores_bare_tokens(self):
        # Bare [N] and [W1] (pre-v0.1.3 grammar) are not recognised by
        # the turn-scoped extractor.
        result = extract_turn_cited_ids(
            "bare [3] and [W5] alongside turn-scoped [T1:1]"
        )
        assert result == {(1, "", 1)}

    def test_empty(self):
        assert extract_turn_cited_ids("plain text") == set()

    def test_pattern_constant_exported(self):
        # TURN_CITATION_PATTERN should be reusable for downstream tooling.
        matches = list(
            TURN_CITATION_PATTERN.finditer("[T1:1] foo [T2:W3] bar")
        )
        assert len(matches) == 2
        assert matches[0].group(1) == "1"
        assert matches[0].group(2) == ""
        assert matches[0].group(3) == "1"
        assert matches[1].group(1) == "2"
        assert matches[1].group(2) == "W"
        assert matches[1].group(3) == "3"


# ---------------------------------------------------------------------------
# Pre-v0.1.3 API is gone
# ---------------------------------------------------------------------------


class TestLegacyAPIRemoved:
    """Confirm the pre-v0.1.3 public API is no longer importable.

    Callers should migrate to ``sanitize_turn_citations`` and
    ``extract_turn_cited_ids``.
    """

    def test_sanitize_citations_not_exported(self):
        with pytest.raises(ImportError):
            from metacouplingllm.knowledge.citations import (
                sanitize_citations,  # noqa: F401
            )

    def test_extract_cited_ids_not_exported(self):
        with pytest.raises(ImportError):
            from metacouplingllm.knowledge.citations import (
                extract_cited_ids,  # noqa: F401
            )

    def test_citation_pattern_not_exported(self):
        with pytest.raises(ImportError):
            from metacouplingllm.knowledge.citations import (
                CITATION_PATTERN,  # noqa: F401
            )

    def test_legacy_constants_not_exported(self):
        with pytest.raises(ImportError):
            from metacouplingllm.knowledge.citations import (
                LEGACY_CITATION_PATTERN,  # noqa: F401
            )
        with pytest.raises(ImportError):
            from metacouplingllm.knowledge.citations import (
                LEGACY_WEB_PATTERN,  # noqa: F401
            )
