"""Tests for the RAG-only Q&A mode (``coupling_analysis=False``).

Covers:
- The default ``coupling_analysis=True`` keeps producing AnalysisResult.
- ``coupling_analysis=False`` switches ``analyze()`` to return RAGResult.
- Multi-turn conversation: prior turns are remembered, each turn runs
  fresh RAG retrieval, ``clear_history()`` resets state, the
  framework-mode ``_history`` is left untouched.
- Reference extraction: only papers actually cited in the answer appear
  in ``RAGResult.references``, dedup'd by paper key in cite order.
- Citation sanitizer: hallucinated ``[N]`` markers are stripped.
- Web search is honoured when ``web_search=True``.
- Empty queries raise ``ValueError``.
"""

from __future__ import annotations

import pytest

from metacouplingllm.core import (
    AnalysisResult,
    MetacouplingAssistant,
    RAGResult,
)


# ---------------------------------------------------------------------------
# Fixtures: a RAG-only advisor wired to the existing mock LLM + RAG engine
# ---------------------------------------------------------------------------


@pytest.fixture
def advisor_rag_only(mock_llm_client, mock_rag_engine):
    """A MetacouplingAssistant in RAG-only mode with the mock engine injected."""
    advisor = MetacouplingAssistant(
        llm_client=mock_llm_client,
        max_examples=0,
        verbose=False,
        coupling_analysis=False,
    )
    advisor._rag_engine = mock_rag_engine
    return advisor


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


class TestRagOnlyMode:
    def test_default_constructor_runs_framework_analysis(
        self, mock_llm_client, mock_rag_engine
    ):
        # Default: coupling_analysis=True → analyze() returns AnalysisResult
        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client,
            max_examples=0,
        )
        advisor._rag_engine = mock_rag_engine
        assert advisor._coupling_analysis is True
        result = advisor.analyze("Soybean trade Brazil to China")
        assert isinstance(result, AnalysisResult)

    def test_coupling_analysis_false_returns_ragresult(
        self, advisor_rag_only, mock_llm_client
    ):
        mock_llm_client._responses = [
            "Brazil exports soybeans to China at scale [1]. Land-use "
            "change in Mato Grosso has been documented [3]."
        ]
        result = advisor_rag_only.analyze(
            "What's the research status of China-Brazil soybean trade?"
        )
        assert isinstance(result, RAGResult)
        assert result.turn_number == 1
        assert "[1]" in result.answer
        assert result.usage is not None

    def test_ragresult_only_includes_cited_papers_in_references(
        self, advisor_rag_only, mock_llm_client
    ):
        # Answer cites passages [1] and [3] only — references should
        # contain exactly those two papers, in that order, dedup'd.
        mock_llm_client._responses = [
            "Brazil exports soybeans to China [1]. Land-use change "
            "in Mato Grosso has been documented [3]. The trade has "
            "grown rapidly [3]."  # repeated [3] should not duplicate
        ]
        result = advisor_rag_only.analyze("Soybean trade")
        keys = [p.key for p in result.references]
        assert keys == ["liu_framing_2013", "sun_telecoupled_2017"]

    def test_ragresult_sanitizes_invalid_citation_brackets(
        self, advisor_rag_only, mock_llm_client
    ):
        # [99] is out of range (only 5 passages); should be stripped
        # from the answer and excluded from references.
        mock_llm_client._responses = [
            "Telecoupling is well-established [1]. There is also work [99]."
        ]
        result = advisor_rag_only.analyze("Telecoupling overview")
        assert "[99]" not in result.answer
        assert "[1]" in result.answer
        assert all(p.key != "" for p in result.references)
        # Only one valid citation → exactly one reference
        assert len(result.references) == 1

    def test_ragresult_raises_on_empty_query(self, advisor_rag_only):
        with pytest.raises(ValueError, match="non-empty"):
            advisor_rag_only.analyze("")
        with pytest.raises(ValueError, match="non-empty"):
            advisor_rag_only.analyze("   ")

    # ----- Multi-turn behaviour -----

    def test_multi_turn_remembers_prior_query(
        self, advisor_rag_only, mock_llm_client
    ):
        mock_llm_client._responses = [
            "Soybean trade between Brazil and China is well-studied [1].",
            "The environmental impacts include cropland expansion [3].",
        ]
        r1 = advisor_rag_only.analyze("Tell me about China-Brazil soybean trade")
        r2 = advisor_rag_only.analyze("What about its environmental impacts?")
        assert r1.turn_number == 1
        assert r2.turn_number == 2
        # Both turns end up in the RAG history (system + 2*user + 2*asst)
        assert len(advisor_rag_only._rag_history) == 5
        assert advisor_rag_only._rag_history[0].role == "system"
        assert advisor_rag_only._rag_history[1].role == "user"
        assert advisor_rag_only._rag_history[2].role == "assistant"
        assert advisor_rag_only._rag_history[3].role == "user"
        assert advisor_rag_only._rag_history[4].role == "assistant"
        # Turn 2's LLM call must have received the full prior history
        msgs_for_turn2 = mock_llm_client.calls[1]
        assert len(msgs_for_turn2) == 4  # system + user1 + asst1 + user2

    def test_each_turn_runs_fresh_rag_retrieval(
        self, advisor_rag_only, mock_rag_engine
    ):
        advisor_rag_only.analyze("first question")
        advisor_rag_only.analyze("follow-up question")
        # Mock engine recorded two retrieve() calls, each with the
        # respective query — not just turn 1's.
        assert len(mock_rag_engine.calls) == 2
        assert mock_rag_engine.calls[0]["query"] == "first question"
        assert mock_rag_engine.calls[1]["query"] == "follow-up question"

    def test_clear_history_resets_conversation(
        self, advisor_rag_only, mock_llm_client
    ):
        advisor_rag_only.analyze("first question")
        advisor_rag_only.analyze("second question")
        assert advisor_rag_only.conversation_turns == 2
        advisor_rag_only.clear_history()
        assert advisor_rag_only.conversation_turns == 0
        assert advisor_rag_only._rag_history == []
        # A new analyze() call after clear starts at turn 1 again
        result = advisor_rag_only.analyze("fresh question")
        assert result.turn_number == 1

    def test_conversation_turns_property_increments(self, advisor_rag_only):
        assert advisor_rag_only.conversation_turns == 0
        advisor_rag_only.analyze("q1")
        assert advisor_rag_only.conversation_turns == 1
        advisor_rag_only.analyze("q2")
        assert advisor_rag_only.conversation_turns == 2

    def test_conversation_turns_zero_in_framework_mode(
        self, mock_llm_client, mock_rag_engine
    ):
        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client,
            max_examples=0,
            coupling_analysis=True,  # framework mode
        )
        advisor._rag_engine = mock_rag_engine
        advisor.analyze("anything")
        # Property should report 0 in framework mode regardless of what
        # the framework history contains.
        assert advisor.conversation_turns == 0

    def test_framework_history_untouched_by_rag_mode_calls(
        self, advisor_rag_only
    ):
        # advisor_rag_only is coupling_analysis=False; calling
        # analyze() must not write to the framework `_history`.
        assert advisor_rag_only._history == []
        advisor_rag_only.analyze("q1")
        advisor_rag_only.analyze("q2")
        assert advisor_rag_only._history == []
        # And the RAG history is the one that grew.
        assert len(advisor_rag_only._rag_history) == 5

    # ----- formatted property -----

    def test_formatted_includes_answer_and_references_block(
        self, advisor_rag_only, mock_llm_client
    ):
        mock_llm_client._responses = [
            "Telecoupling is a useful framing [1]. Land-use change "
            "in Mato Grosso has been documented [3]."
        ]
        result = advisor_rag_only.analyze("Soybean trade")
        formatted = result.formatted
        # Bibliography block is present
        assert "REFERENCES (cited in this answer)" in formatted
        # Both cited papers appear in the bibliography
        assert "Framing Sustainability in a Telecoupled World" in formatted
        assert "Telecoupled land-use changes in distant countries" in formatted
        # Original answer text appears (with possibly-renumbered markers)
        assert "Telecoupling is a useful framing" in formatted
        assert "Mato Grosso" in formatted

    def test_formatted_renumbers_sparse_citations_to_sequential(
        self, advisor_rag_only, mock_llm_client
    ):
        # LLM cited [1] and [3] (sparse). formatted should renumber to
        # [1] and [2] sequentially in the answer body so they line up
        # with the bibliography's [1] and [2] entries.
        mock_llm_client._responses = [
            "First claim [1]. Second claim [3]. Repeat second claim [3]."
        ]
        result = advisor_rag_only.analyze("query")
        # The original answer keeps the LLM's numbering
        assert "[1]" in result.answer
        assert "[3]" in result.answer
        # The formatted view renumbers
        formatted = result.formatted
        assert "First claim [1]" in formatted
        assert "Second claim [2]" in formatted
        assert "Repeat second claim [2]" in formatted
        assert "[3]" not in formatted.split("REFERENCES")[0]

    def test_formatted_with_no_references_returns_renumbered_answer_only(
        self, advisor_rag_only, mock_llm_client
    ):
        # If the LLM produces no citations, references is empty and
        # formatted should not include the bibliography header.
        mock_llm_client._responses = [
            "I cannot answer this from the literature provided."
        ]
        result = advisor_rag_only.analyze("very off-topic question")
        assert result.references == []
        assert "REFERENCES" not in result.formatted


# ---------------------------------------------------------------------------
# Web search opt-in
# ---------------------------------------------------------------------------


class TestRagOnlyWebSearch:
    def test_ragresult_uses_web_search_when_enabled(
        self, mock_llm_client, mock_rag_engine, monkeypatch
    ):
        # Stub out search_web inside the websearch module so we don't
        # touch the network. The advisor calls it lazily inside
        # _analyze_rag_only via `from metacouplingllm.knowledge.websearch
        # import ... search_web`.
        from metacouplingllm.knowledge import websearch as ws

        fake_results = [
            {
                "title": "Brazil-China soy trade jumps 12%",
                "snippet": "USDA report …",
                "url": "https://example.org/usda",
            }
        ]

        def _fake_search_web(query, max_results=5, backend=None, metadata=None):
            return fake_results

        monkeypatch.setattr(ws, "search_web", _fake_search_web)

        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client,
            max_examples=0,
            coupling_analysis=False,
            web_search=True,
            web_search_max_results=3,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("China-Brazil soybean trade in 2024")
        assert isinstance(result, RAGResult)
        assert result.web_sources == fake_results
        # The user message sent to the LLM should include the web block.
        user_msg = mock_llm_client.calls[0][1].content
        assert "<web_search_results>" in user_msg


# ---------------------------------------------------------------------------
# Visibility — web search status / failures / formatted block
# ---------------------------------------------------------------------------


class TestRagOnlyVisibility:
    def test_web_search_status_prints_unconditionally(
        self, mock_llm_client, mock_rag_engine, monkeypatch, capsys
    ):
        """Status line must print even when verbose=False."""
        from metacouplingllm.knowledge import websearch as ws

        def _fake_search_web(query, max_results=5, backend=None, metadata=None):
            if metadata is not None:
                metadata["backend_used"] = "ddgs_fallback"
            return [{"title": "t", "snippet": "s", "url": "https://x"}]

        monkeypatch.setattr(ws, "search_web", _fake_search_web)

        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client,
            max_examples=0,
            verbose=False,            # <-- intentionally OFF
            coupling_analysis=False,
            web_search=True,
        )
        advisor._rag_engine = mock_rag_engine
        advisor.analyze("anything")

        out = capsys.readouterr().out
        assert "(RAG mode) Searching the web..." in out
        assert "Web search via ddgs_fallback" in out
        assert "Web search returned 1 results" in out

    def test_web_search_failure_prints_unconditionally(
        self, mock_llm_client, mock_rag_engine, monkeypatch, capsys
    ):
        """A failure inside search_web should print + still return RAGResult."""
        from metacouplingllm.knowledge import websearch as ws

        def _broken_search_web(query, max_results=5, backend=None, metadata=None):
            raise ConnectionError("DuckDuckGo unreachable")

        monkeypatch.setattr(ws, "search_web", _broken_search_web)

        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client,
            max_examples=0,
            verbose=False,            # <-- intentionally OFF
            coupling_analysis=False,
            web_search=True,
        )
        advisor._rag_engine = mock_rag_engine
        result = advisor.analyze("anything")

        out = capsys.readouterr().out
        assert "(RAG mode) Web search failed" in out
        assert "ConnectionError" in out
        # Failure didn't crash analyze()
        assert isinstance(result, RAGResult)
        assert result.web_sources is None

    def test_formatted_includes_web_sources_when_present(
        self, mock_llm_client, mock_rag_engine, monkeypatch
    ):
        from metacouplingllm.knowledge import websearch as ws

        fake = [
            {
                "title": "Brazil-China soy 2024 update",
                "snippet": "USDA reports record exports...",
                "url": "https://example.org/usda-2024",
            },
            {
                "title": "Cerrado deforestation analysis",
                "snippet": "Recent satellite analysis shows ...",
                "url": "https://example.org/cerrado",
            },
        ]
        monkeypatch.setattr(
            ws, "search_web",
            lambda q, max_results=5, backend=None, metadata=None: fake,
        )

        mock_llm_client._responses = [
            "Brazil exports soybeans at scale [1]."
        ]
        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client,
            max_examples=0,
            coupling_analysis=False,
            web_search=True,
        )
        advisor._rag_engine = mock_rag_engine
        result = advisor.analyze("Soybean trade")

        formatted = result.formatted
        assert "WEB SOURCES (background context, not used for citations)" in formatted
        assert "[w1] Brazil-China soy 2024 update" in formatted
        assert "https://example.org/usda-2024" in formatted
        assert "[w2] Cerrado deforestation analysis" in formatted
        # And the literature bibliography is still there too
        assert "REFERENCES (cited in this answer)" in formatted

    def test_formatted_omits_web_sources_when_empty(
        self, advisor_rag_only, mock_llm_client
    ):
        # No web search → no web block in formatted (regression guard)
        mock_llm_client._responses = ["Some claim [1]."]
        result = advisor_rag_only.analyze("query")
        assert result.web_sources is None
        assert "WEB SOURCES" not in result.formatted

    def test_formatted_includes_retrieval_scores_for_each_reference(
        self, advisor_rag_only, mock_llm_client
    ):
        """Each cited reference should show a Confidence line + the
        raw score from its highest-scoring retrieved passage."""
        mock_llm_client._responses = [
            "Claim A [1]. Claim B [3]."
        ]
        result = advisor_rag_only.analyze("Soybean trade")
        formatted = result.formatted
        # The first cited passage in the fixture has score 0.92
        # (liu_framing_2013); the third has score 0.81 (sun_telecoupled_2017)
        assert "Confidence: High (score: 0.920)" in formatted
        assert "Confidence: High (score: 0.810)" in formatted
        # And both papers' confidence lines appear AFTER their title lines
        # (sanity check on relative ordering)
        liu_idx = formatted.index("Framing Sustainability")
        liu_score_idx = formatted.index("0.920", liu_idx)
        assert liu_idx < liu_score_idx
