"""Tests for the RAG-only Q&A mode (``coupling_analysis=False``).

Covers:
- The default ``coupling_analysis=True`` keeps producing AnalysisResult.
- ``coupling_analysis=False`` switches ``analyze()`` to return RAGResult.
- Multi-turn conversation: prior turns are remembered, each turn runs
  fresh RAG retrieval, ``clear_history()`` resets state, the
  framework-mode ``_history`` is left untouched.
- Reference extraction: only **current-turn** ``[Tk:N]`` citations are
  resolved into ``RAGResult.references`` (prior-turn back-references
  are kept in the answer but excluded from the current bibliography).
- Citation sanitizer: invalid turn-scoped tokens and bare legacy
  ``[N]`` / ``[W1]`` are stripped.
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
            "Brazil exports soybeans to China at scale [T1:1]. Land-use "
            "change in Mato Grosso has been documented [T1:3]."
        ]
        result = advisor_rag_only.analyze(
            "What's the research status of China-Brazil soybean trade?"
        )
        assert isinstance(result, RAGResult)
        assert result.turn_number == 1
        assert "[T1:1]" in result.answer
        assert result.usage is not None

    def test_ragresult_only_includes_cited_papers_in_references(
        self, advisor_rag_only, mock_llm_client
    ):
        # Answer cites passages [T1:1] and [T1:3] only — references
        # should contain exactly those two papers, in that order,
        # dedup'd.
        mock_llm_client._responses = [
            "Brazil exports soybeans to China [T1:1]. Land-use change "
            "in Mato Grosso has been documented [T1:3]. The trade has "
            "grown rapidly [T1:3]."  # repeated [T1:3] should not duplicate
        ]
        result = advisor_rag_only.analyze("Soybean trade")
        keys = [p.key for p in result.references]
        assert keys == ["liu_framing_2013", "sun_telecoupled_2017"]

    def test_ragresult_sanitizes_invalid_citation_brackets(
        self, advisor_rag_only, mock_llm_client
    ):
        # [T1:99] is out of range (only 5 passages); should be stripped
        # from the answer and excluded from references.
        mock_llm_client._responses = [
            "Telecoupling is well-established [T1:1]. "
            "There is also work [T1:99]."
        ]
        result = advisor_rag_only.analyze("Telecoupling overview")
        assert "[T1:99]" not in result.answer
        assert "[T1:1]" in result.answer
        assert all(p.key != "" for p in result.references)
        # Only one valid citation → exactly one reference
        assert len(result.references) == 1

    def test_ragresult_strips_bare_legacy_tokens(
        self, advisor_rag_only, mock_llm_client
    ):
        # If the LLM slips and emits a bare [1] (legacy form), it's
        # silently stripped and contributes no reference.
        mock_llm_client._responses = [
            "Good cite [T1:1]. Legacy slip [1]. Bad legacy [W1]."
        ]
        result = advisor_rag_only.analyze("Legacy slip test")
        assert "[T1:1]" in result.answer
        assert "[1]" not in result.answer
        assert "[W1]" not in result.answer
        # Only the valid turn-scoped citation contributes a reference
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
            "Soybean trade between Brazil and China is well-studied [T1:1].",
            "The environmental impacts include cropland expansion [T2:3].",
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
        # Turn 1's [T1:1] is preserved verbatim in the stored
        # assistant message even after turn 2 happens — that's the
        # whole point of turn-scoped citations.
        assert "[T1:1]" in advisor_rag_only._rag_history[2].content
        # Turn 2's LLM call must have received the full prior history
        msgs_for_turn2 = mock_llm_client.calls[1]
        assert len(msgs_for_turn2) == 4  # system + user1 + asst1 + user2

    def test_multi_turn_back_reference_to_prior_turn(
        self, advisor_rag_only, mock_llm_client
    ):
        """Turn 2 can back-reference turn 1's evidence by copying the
        original [T1:N] token verbatim."""
        mock_llm_client._responses = [
            "Soybean trade is well-studied [T1:1] and [T1:3].",
            "Extending [T1:3] with the new data: cropland in Mato "
            "Grosso continues to expand [T2:1].",
        ]
        advisor_rag_only.analyze("China-Brazil soybean trade")
        r2 = advisor_rag_only.analyze("What about new data?")

        # Both turn-1 back-reference and turn-2 fresh citation survive
        assert "[T1:3]" in r2.answer
        assert "[T2:1]" in r2.answer
        # Only the current-turn citation contributes to references
        keys = [p.key for p in r2.references]
        assert keys == ["liu_framing_2013"]

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
            "Telecoupling is a useful framing [T1:1]. Land-use change "
            "in Mato Grosso has been documented [T1:3]."
        ]
        result = advisor_rag_only.analyze("Soybean trade")
        formatted = result.formatted
        # Bibliography block is present (turn-scoped header)
        assert "REFERENCES (cited in turn 1)" in formatted
        # Both cited papers appear in the bibliography with their
        # actual cited passage IDs (1 and 3 — not renumbered to 1, 2).
        assert "[T1:1] Framing Sustainability in a Telecoupled World" in formatted
        assert "[T1:3] Telecoupled land-use changes in distant countries" in formatted
        # Original answer text appears with its stable turn-scoped markers
        assert "Telecoupling is a useful framing [T1:1]" in formatted
        assert "Mato Grosso has been documented [T1:3]" in formatted

    def test_formatted_bibliography_labels_match_inline_citations(
        self, advisor_rag_only, mock_llm_client
    ):
        # The bibliography must use the LLM's original passage IDs
        # (e.g. [T1:3]) so the reader can match each citation in the
        # body to its entry in the references list. No renumbering.
        mock_llm_client._responses = [
            "First claim [T1:1]. Second claim [T1:3]. Repeat second [T1:3]."
        ]
        result = advisor_rag_only.analyze("query")
        # The answer keeps the LLM's exact tokens — no remapping
        assert "[T1:1]" in result.answer
        assert "[T1:3]" in result.answer
        formatted = result.formatted
        assert "First claim [T1:1]" in formatted
        assert "Second claim [T1:3]" in formatted
        assert "Repeat second [T1:3]" in formatted
        # The bibliography uses the actual passage IDs [T1:1] and
        # [T1:3], not a renumbered [T1:2].
        bib = formatted.split("REFERENCES")[1]
        assert "[T1:1]" in bib
        assert "[T1:3]" in bib
        assert "[T1:2]" not in bib

        # And the parallel passage-id list reflects what was cited.
        assert result.reference_passage_ids == [1, 3]

    def test_formatted_with_no_references_returns_answer_only(
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
                "model_summary": "USDA report …",
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
        # The user message sent to the LLM should include the web block,
        # tagged with the current turn so the LLM knows which Tk: prefix
        # to use for inline citations.
        user_msg = mock_llm_client.calls[0][1].content
        assert '<web_search_results turn="1">' in user_msg


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
            return [{"title": "t", "model_summary": "s", "url": "https://x"}]

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
                "model_summary": "USDA reports record exports...",
                "url": "https://example.org/usda-2024",
            },
            {
                "title": "Cerrado deforestation analysis",
                "model_summary": "Recent satellite analysis shows ...",
                "url": "https://example.org/cerrado",
            },
        ]
        monkeypatch.setattr(
            ws, "search_web",
            lambda q, max_results=5, backend=None, metadata=None: fake,
        )

        mock_llm_client._responses = [
            "Brazil exports soybeans at scale [T1:1]."
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
        assert "WEB SOURCES (turn 1, background context)" in formatted
        assert "[T1:W1] Brazil-China soy 2024 update" in formatted
        assert "https://example.org/usda-2024" in formatted
        assert "[T1:W2] Cerrado deforestation analysis" in formatted
        # And the literature bibliography is still there too
        assert "REFERENCES (cited in turn 1)" in formatted

    def test_formatted_omits_web_sources_when_empty(
        self, advisor_rag_only, mock_llm_client
    ):
        # No web search → no web block in formatted (regression guard)
        mock_llm_client._responses = ["Some claim [T1:1]."]
        result = advisor_rag_only.analyze("query")
        assert result.web_sources is None
        assert "WEB SOURCES" not in result.formatted

    def test_formatted_includes_retrieval_scores_for_each_reference(
        self, advisor_rag_only, mock_llm_client
    ):
        """Each cited reference should show a Confidence line + the
        raw score from its highest-scoring retrieved passage."""
        mock_llm_client._responses = [
            "Claim A [T1:1]. Claim B [T1:3]."
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


# ---------------------------------------------------------------------------
# RAG-only LLM passage budget — full chunks reach the LLM
# ---------------------------------------------------------------------------


class TestRagOnlyLLMPassageBudget:
    """The RAG-only path sends a literature block to the LLM as part
    of the user message.  Previously this used the dual-purpose
    ``format_evidence`` helper's default 300-char excerpt — severely
    under-using retrieved evidence in RAG-only mode (~16x smaller per
    passage than the framework path).  The new
    ``max_chars=_LLM_PASSAGE_MAX_CHARS`` (5000) parameter brings the
    LLM-bound rendering up to parity with the framework path."""

    def test_rag_only_sends_full_chunk_text_to_llm(self):
        """End-to-end: a chunk with a marker at char ~1500 reaches
        the LLM in RAG-only mode (would have been dropped under the
        old 300-char excerpt cap)."""
        from tests.conftest import (
            _RecordingMockLLMClient, _RecordingMockRagEngine,
        )
        from metacouplingllm.knowledge.rag import (
            RetrievalResult, TextChunk,
        )

        # ~3000-char chunk with a sentinel deliberately past char 1500
        body = "lorem ipsum dolor sit amet " * 60   # ~1620 chars
        marker = " SENTINEL_PAST_300_CHAR_EXCERPT_LIMIT "
        chunk_text = body + marker + body
        assert len(chunk_text) > 1500
        # Marker sits at char ~1620 — far past the old 300-char excerpt.
        assert chunk_text.index("SENTINEL") > 1500

        hits = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="long_paper",
                    paper_title="Long bilateral paper",
                    authors="Test",
                    year=2024,
                    section="Results",
                    text=chunk_text,
                    chunk_index=0,
                ),
                score=0.9,
            )
        ]

        client = _RecordingMockLLMClient(
            responses=["A short answer with no citations."],
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            coupling_analysis=False,    # RAG-only mode
        )
        advisor._rag_engine = _RecordingMockRagEngine(
            results=hits, backend="embeddings",
        )

        advisor.analyze("a query that triggers retrieval")

        # The captured user message should contain the marker —
        # confirming the LLM saw the full chunk text, not a 300-char
        # excerpt.
        assert client.call_count == 1
        user_msg = next(
            m.content for m in client.calls[0] if m.role == "user"
        )
        assert "SENTINEL_PAST_300_CHAR_EXCERPT_LIMIT" in user_msg, (
            "RAG-only LLM should see full chunk text per passage; the "
            "300-char excerpt cap is for human display only."
        )
