"""End-to-end tests for the pre-retrieval RAG pipeline.

Verifies that:
- retrieved passages are injected into the user message as XML
- citation rules appear in the system prompt
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
    def test_default_rag_top_k_is_8(self, mock_llm_client):
        advisor = MetacouplingAssistant(llm_client=mock_llm_client, max_examples=0)
        assert advisor._rag_top_k == 8

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
        # The block carries a turn="1" attribute under the new grammar
        assert '<retrieved_literature turn="1">' in user_msg.content
        assert "</retrieved_literature>" in user_msg.content

    def test_citation_ids_sequential_and_stable(self, advisor_pre_retrieval):
        advisor_pre_retrieval.analyze("Soybean trade Brazil to China")
        user_msg = advisor_pre_retrieval._history[1].content
        # Five passages from the fixture → ids 1..5 must all appear,
        # each tagged with turn="1"
        for i in range(1, 6):
            assert f'<passage turn="1" id="{i}"' in user_msg
        # And no id="6" (we only have 5 fake hits)
        assert 'id="6"' not in user_msg

    def test_literature_block_before_research_description(
        self, advisor_pre_retrieval
    ):
        advisor_pre_retrieval.analyze("Soybean trade Brazil to China")
        user_msg = advisor_pre_retrieval._history[1].content
        # Literature block must come BEFORE the research description
        # so the user's actual ask is the last thing the LLM reads
        lit_pos = user_msg.index('<retrieved_literature turn="1">')
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

        # LLM cites both valid ([T1:1], [T1:2]) and invalid ([T1:99]) IDs.
        # The sanitizer must keep the valid tokens, strip [T1:99], and log.
        client = _RecordingMockLLMClient(
            responses=[
                "### 1. Coupling Classification\n"
                "Soybean trade is telecoupling [T1:1] [T1:2] [T1:99]."
            ]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            verbose=False,
        )
        advisor._rag_engine = mock_rag_engine

        with caplog.at_level(logging.WARNING):
            result = advisor.analyze("Soybean trade")

        assert "[T1:1]" in result.formatted
        assert "[T1:2]" in result.formatted
        assert "[T1:99]" not in result.formatted
        # Either citations.py or core.py logger should have warned
        assert any("99" in record.message for record in caplog.records)

    def test_empty_retrieval_still_includes_block(self, mock_llm_client):
        """When retrieval returns [] the user message must still
        contain the self-closing tag so the LLM knows retrieval ran."""
        from tests.conftest import _RecordingMockRagEngine

        empty_engine = _RecordingMockRagEngine(results=[])
        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client, max_examples=0
        )
        advisor._rag_engine = empty_engine

        advisor.analyze("Obscure niche topic with no matches")
        user_msg = advisor._history[1].content
        # The self-closing form signals "retrieval ran but found nothing".
        # The turn attribute is still emitted so the LLM knows which
        # Tk: prefix is in play.
        assert '<retrieved_literature turn="1"/>' in user_msg

    def test_rag_engine_failure_does_not_crash_analyze(self, mock_llm_client):
        from tests.conftest import _RecordingMockRagEngine

        failing_engine = _RecordingMockRagEngine(results=[])
        failing_engine.raise_on_retrieve = RuntimeError("BGE model unavailable")

        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client, max_examples=0
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

