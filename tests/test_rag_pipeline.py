"""End-to-end tests for the pre-retrieval RAG pipeline.

Verifies that:
- pre_retrieval is the new default mode
- retrieved passages are injected into the user message as XML
- citation rules appear in the system prompt only when pre_retrieval
- the post_hoc code path is unchanged when explicitly selected
- refine() re-retrieves with a labeled merged query
- the citation sanitizer strips out-of-range tokens with a warning
- empty retrievals still emit a self-closing literature block
- a failing retrieval doesn't crash analyze()

Mocks live in ``tests/conftest.py`` (see the ``mock_llm_client``,
``mock_rag_engine`` and ``advisor_pre_retrieval`` fixtures).
"""

from __future__ import annotations

import logging

import pytest

from metacouplingllm.core import AnalysisResult, MetacouplingAssistant
from metacouplingllm.knowledge.rag import RetrievalResult, TextChunk


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------


class TestPreRetrievalDefaults:
    def test_default_rag_mode_is_pre_retrieval(self, mock_llm_client):
        advisor = MetacouplingAssistant(llm_client=mock_llm_client, max_examples=0)
        assert advisor._rag_mode == "pre_retrieval"

    def test_default_rag_top_k_is_8(self, mock_llm_client):
        advisor = MetacouplingAssistant(llm_client=mock_llm_client, max_examples=0)
        assert advisor._rag_top_k == 8

    def test_invalid_rag_mode_raises(self, mock_llm_client):
        with pytest.raises(ValueError, match="rag_mode"):
            MetacouplingAssistant(
                llm_client=mock_llm_client,
                max_examples=0,
                rag_mode="bogus",
            )

    def test_named_builtin_rag_corpus_resolves(self):
        source = MetacouplingAssistant._resolve_rag_source(
            rag_papers_dir=None,
            rag_corpus="journal_articles_2025",
        )
        assert source == "__metacoupling_builtin_journal_articles_2025__"

    def test_custom_rag_dir_and_named_corpus_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="either rag_corpus or rag_papers_dir"):
            MetacouplingAssistant._resolve_rag_source(
                rag_papers_dir="Papers",
                rag_corpus="journal_articles_2025",
            )

    def test_unknown_named_rag_corpus_raises(self):
        with pytest.raises(ValueError, match="Unknown rag_corpus"):
            MetacouplingAssistant._resolve_rag_source(
                rag_papers_dir=None,
                rag_corpus="unknown",
            )


# ---------------------------------------------------------------------------
# Prompt-injection behavior
# ---------------------------------------------------------------------------


class TestPreRetrievalPromptInjection:
    def test_passages_injected_into_user_message(self, advisor_pre_retrieval):
        advisor_pre_retrieval.analyze("Soybean trade Brazil to China")
        # The user message is the second message in history (index 1)
        user_msg = advisor_pre_retrieval._history[1]
        assert user_msg.role == "user"
        assert "<retrieved_literature>" in user_msg.content
        assert "</retrieved_literature>" in user_msg.content

    def test_citation_ids_sequential_and_stable(self, advisor_pre_retrieval):
        advisor_pre_retrieval.analyze("Soybean trade Brazil to China")
        user_msg = advisor_pre_retrieval._history[1].content
        # Five passages from the fixture → ids 1..5 must all appear
        for i in range(1, 6):
            assert f'id="{i}"' in user_msg
        # And no id="6" (we only have 5 fake hits)
        assert 'id="6"' not in user_msg

    def test_literature_block_before_research_description(
        self, advisor_pre_retrieval
    ):
        advisor_pre_retrieval.analyze("Soybean trade Brazil to China")
        user_msg = advisor_pre_retrieval._history[1].content
        # Literature block must come BEFORE the research description
        # so the user's actual ask is the last thing the LLM reads
        lit_pos = user_msg.index("<retrieved_literature>")
        ask_pos = user_msg.index("Soybean trade Brazil to China")
        assert lit_pos < ask_pos

    def test_passage_text_is_present(
        self, advisor_pre_retrieval, fake_retrieval_results
    ):
        advisor_pre_retrieval.analyze("Soybean trade Brazil to China")
        user_msg = advisor_pre_retrieval._history[1].content
        # Each fake passage's text should appear in the prompt
        for hit in fake_retrieval_results:
            # Use a snippet to avoid newline-formatting differences
            snippet = hit.chunk.text.split(".")[0]
            assert snippet in user_msg


# ---------------------------------------------------------------------------
# System-prompt rules
# ---------------------------------------------------------------------------


class TestCitationRulesInSystemPrompt:
    def test_citation_rules_in_system_prompt_when_pre_retrieval(
        self, advisor_pre_retrieval
    ):
        advisor_pre_retrieval.analyze("Soybean trade")
        system_msg = advisor_pre_retrieval._history[0]
        assert system_msg.role == "system"
        assert "CITATION RULES" in system_msg.content

    def test_post_hoc_mode_omits_citation_rules(self, advisor_post_hoc):
        advisor_post_hoc.analyze("Soybean trade")
        system_msg = advisor_post_hoc._history[0]
        assert system_msg.role == "system"
        assert "CITATION RULES" not in system_msg.content

    def test_post_hoc_mode_no_literature_in_user_msg(self, advisor_post_hoc):
        advisor_post_hoc.analyze("Soybean trade")
        user_msg = advisor_post_hoc._history[1].content
        # post_hoc mode does NOT inject the XML block — RAG runs after
        assert "<retrieved_literature>" not in user_msg
        assert "<retrieved_literature/>" not in user_msg


# ---------------------------------------------------------------------------
# Output / evidence block
# ---------------------------------------------------------------------------


class TestEvidenceBlockOutput:
    def test_format_evidence_appended(self, advisor_pre_retrieval):
        result = advisor_pre_retrieval.analyze("Soybean trade")
        assert "SUPPORTING EVIDENCE FROM LITERATURE" in result.formatted

    def test_evidence_uses_pre_retrieved_hits(
        self, advisor_pre_retrieval, mock_rag_engine
    ):
        # The pre_retrieval branch of _build_result must NOT call
        # retrieve() a second time — the evidence comes from the hits
        # that were already fetched at analyze() time.
        advisor_pre_retrieval.analyze("Soybean trade")
        assert len(mock_rag_engine.calls) == 1


# ---------------------------------------------------------------------------
# Citation sanitization
# ---------------------------------------------------------------------------


class TestCitationSanitization:
    def test_invalid_citations_stripped_with_warning(
        self, mock_rag_engine, fake_retrieval_results, caplog
    ):
        from tests.conftest import _RecordingMockLLMClient

        # LLM cites both valid (1, 2) and invalid (99) IDs.
        # The sanitizer must keep [1] and [2], strip [99], and log.
        client = _RecordingMockLLMClient(
            responses=[
                "### 1. Coupling Classification\n"
                "Soybean trade is telecoupling [1] [2] [99]."
            ]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            verbose=False,
            rag_mode="pre_retrieval",
        )
        advisor._rag_engine = mock_rag_engine

        with caplog.at_level(logging.WARNING):
            result = advisor.analyze("Soybean trade")

        assert "[1]" in result.formatted
        assert "[2]" in result.formatted
        assert "[99]" not in result.formatted
        # Either citations.py or core.py logger should have warned
        assert any("99" in record.message for record in caplog.records)

    def test_empty_retrieval_still_includes_block(self, mock_llm_client):
        """When retrieval returns [] the user message must still
        contain the self-closing tag so the LLM knows retrieval ran."""
        from tests.conftest import _RecordingMockRagEngine

        empty_engine = _RecordingMockRagEngine(results=[])
        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client, max_examples=0, rag_mode="pre_retrieval"
        )
        advisor._rag_engine = empty_engine

        advisor.analyze("Obscure niche topic with no matches")
        user_msg = advisor._history[1].content
        # The self-closing form signals "retrieval ran but found nothing"
        assert "<retrieved_literature/>" in user_msg

    def test_rag_engine_failure_does_not_crash_analyze(self, mock_llm_client):
        from tests.conftest import _RecordingMockRagEngine

        failing_engine = _RecordingMockRagEngine(results=[])
        failing_engine.raise_on_retrieve = RuntimeError("BGE model unavailable")

        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client, max_examples=0, rag_mode="pre_retrieval"
        )
        advisor._rag_engine = failing_engine

        # Should not raise — should fall back to empty hits
        result = advisor.analyze("Anything")
        assert isinstance(result, AnalysisResult)
        # Hits get reset to [] on failure
        assert advisor._last_rag_hits == []


# ---------------------------------------------------------------------------
# Refine() merged-query behavior
# ---------------------------------------------------------------------------


class TestRefineMergedQuery:
    def test_refine_uses_labeled_merged_query(
        self, advisor_pre_retrieval, mock_rag_engine
    ):
        advisor_pre_retrieval.analyze("Soybean trade from Mato Grosso to China")
        advisor_pre_retrieval.refine("Focus more on labor dynamics")

        # The second retrieve() call is the refinement.
        assert len(mock_rag_engine.calls) == 2
        refine_query = mock_rag_engine.calls[1]["query"]
        assert "Original research question:" in refine_query
        assert "Refinement request:" in refine_query
        assert "Soybean trade from Mato Grosso to China" in refine_query
        assert "Focus more on labor dynamics" in refine_query
        # The original-question label must come before the refinement label
        assert refine_query.index("Original research question:") < refine_query.index(
            "Refinement request:"
        )

    def test_refine_overwrites_last_rag_hits(
        self, advisor_pre_retrieval, mock_rag_engine, fake_retrieval_results
    ):
        advisor_pre_retrieval.analyze("Soybean trade")
        first_hits = advisor_pre_retrieval._last_rag_hits

        # Swap the engine's results so refine() returns a different set
        new_chunk = TextChunk(
            paper_key="newpaper_2024",
            paper_title="A different paper",
            authors="New Author",
            year=2024,
            section="Body",
            text="Different content for the refinement.",
        )
        mock_rag_engine._results = [RetrievalResult(chunk=new_chunk, score=0.95)]

        advisor_pre_retrieval.refine("Focus on something different")
        second_hits = advisor_pre_retrieval._last_rag_hits

        assert second_hits is not first_hits
        assert len(second_hits) == 1
        assert second_hits[0].chunk.paper_key == "newpaper_2024"

    def test_original_query_anchored_across_refines(
        self, advisor_pre_retrieval, mock_rag_engine
    ):
        original = "Soybean trade from Mato Grosso to China"
        advisor_pre_retrieval.analyze(original)
        advisor_pre_retrieval.refine("First refinement")
        advisor_pre_retrieval.refine("Second refinement")

        # _original_query must NOT have been overwritten by either refine
        assert advisor_pre_retrieval._original_query == original

        # And the second refinement's merged query still references the original
        second_refine_query = mock_rag_engine.calls[2]["query"]
        assert original in second_refine_query
        assert "Second refinement" in second_refine_query

    def test_post_hoc_refine_does_not_use_merged_query(
        self, advisor_post_hoc, mock_rag_engine
    ):
        # In post_hoc mode, refine() does not pre-retrieve, so the
        # mock engine should only see _build_result()-time queries
        # (which use _build_query_from_analysis on parsed analysis,
        # not the original research description).
        advisor_post_hoc.analyze("Soybean trade")
        advisor_post_hoc.refine("More on labor")
        # No labeled merged query should appear in any retrieve() call
        for call in mock_rag_engine.calls:
            assert "Original research question:" not in call["query"]
            assert "Refinement request:" not in call["query"]


# ---------------------------------------------------------------------------
# Backward compat: post_hoc mode unchanged
# ---------------------------------------------------------------------------


class TestPostHocBackwardCompat:
    def test_post_hoc_mode_unchanged_when_explicitly_selected(
        self, advisor_post_hoc
    ):
        # post_hoc mode should still produce SUPPORTING EVIDENCE block
        # via the legacy code path
        result = advisor_post_hoc.analyze("Soybean trade")
        assert "SUPPORTING EVIDENCE FROM LITERATURE" in result.formatted

    def test_post_hoc_mode_history_has_no_literature_xml(
        self, advisor_post_hoc
    ):
        advisor_post_hoc.analyze("Soybean trade")
        for msg in advisor_post_hoc._history:
            assert "<retrieved_literature>" not in msg.content
            assert "<retrieved_literature/>" not in msg.content
