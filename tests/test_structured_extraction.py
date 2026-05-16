"""End-to-end tests for the ``rag_structured_extraction`` supplement.

Verifies that:
- the feature defaults to ``False`` and adds no LLM call when disabled
- enabling it triggers a second LLM call on top of the main analysis
- the returned supplement is surfaced on
  :attr:`AnalysisResult.structured_supplement` and rendered as a visible
  ``SUPPLEMENTARY STRUCTURED EXTRACTION`` block in ``formatted``
- invalid JSON from the extraction LLM is caught and does not break
  the pipeline
- empty ``_last_rag_hits`` short-circuits the feature cleanly
- the main analysis body (from the first LLM call) is NOT modified by
  the supplement — the reader can always tell the two apart

Mocks live in ``tests/conftest.py``.
"""

from __future__ import annotations

import json

import pytest

from metacouplingllm.core import AnalysisResult, MetacouplingAssistant


# A minimal first-turn response just rich enough to exercise the parser.
_DRAFT_ANALYSIS = """\
### 1. Coupling Classification

This research involves **telecoupling** between UK feed barley
production and export destinations.

### 2. Systems Identification

**Sending System**: United Kingdom barley-producing regions [T1:1]
- **Human subsystem**:
- **Natural subsystem**:
- **Geographic scope**:

**Receiving System**: International feed importers
- **Human subsystem**:
- **Natural subsystem**:
- **Geographic scope**:

### 3. Flows Analysis

- [Matter] United Kingdom → Ireland: Feed barley exports [T1:1]
"""


# A plausible JSON payload the extraction LLM would return.
_VALID_SUPPLEMENT_JSON = json.dumps(
    {
        "additional_sending_mentions": [
            {
                "name": "UK cereal merchants",
                "evidence_passage_ids": [1, 2],
            }
        ],
        "additional_receiving_mentions": [
            {
                "name": "UK feed mill sector",
                "evidence_passage_ids": [1],
            }
        ],
        "additional_spillover_mentions": [],
        "sending_subsystem_fills": {
            "human_subsystem": "Arable farmers and cooperatives",
            "natural_subsystem": None,
            "geographic_scope": None,
            "evidence_passage_ids": [2],
        },
        "receiving_subsystem_fills": {
            "human_subsystem": None,
            "natural_subsystem": None,
            "geographic_scope": None,
            "evidence_passage_ids": [],
        },
        "spillover_subsystem_fills": {
            "human_subsystem": None,
            "natural_subsystem": None,
            "geographic_scope": None,
            "evidence_passage_ids": [],
        },
        "supplementary_flows": [
            {
                "category": "information",
                "direction": "UK farmers \u2192 UK policy agencies",
                "description": "Market intelligence and price feedback",
                "evidence_passage_ids": [3],
            }
        ],
    }
)


# ---------------------------------------------------------------------------
# Default OFF — no second LLM call
# ---------------------------------------------------------------------------


class TestStructuredExtractionDefault:
    def test_default_is_false(self, mock_llm_client):
        advisor = MetacouplingAssistant(
            llm_client=mock_llm_client, max_examples=0,
        )
        assert advisor._rag_structured_extraction is False

    def test_disabled_by_default_no_extra_llm_call(
        self, mock_rag_engine,
    ):
        from tests.conftest import _RecordingMockLLMClient

        # Only ONE response queued — if extraction ran, the second call
        # would fall back to the default_response and call_count would
        # reach 2. We assert it stays at 1.
        client = _RecordingMockLLMClient(responses=[_DRAFT_ANALYSIS])
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            # rag_structured_extraction defaults to False
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Impact of feed barley supply in the UK")
        assert isinstance(result, AnalysisResult)
        assert client.call_count == 1
        assert result.structured_supplement is None
        assert "SUPPLEMENTARY STRUCTURED EXTRACTION" not in result.formatted


# ---------------------------------------------------------------------------
# Enabled + pre_retrieval + hits → second LLM call, supplement surfaced
# ---------------------------------------------------------------------------


class TestStructuredExtractionEnabled:
    def test_second_llm_call_happens(self, mock_rag_engine):
        from tests.conftest import _RecordingMockLLMClient

        client = _RecordingMockLLMClient(
            responses=[_DRAFT_ANALYSIS, _VALID_SUPPLEMENT_JSON]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        advisor.analyze("Impact of feed barley supply in the UK")
        assert client.call_count == 2

    def test_supplement_attached_to_result(self, mock_rag_engine):
        from tests.conftest import _RecordingMockLLMClient

        client = _RecordingMockLLMClient(
            responses=[_DRAFT_ANALYSIS, _VALID_SUPPLEMENT_JSON]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Impact of feed barley supply in the UK")
        sup = result.structured_supplement
        assert isinstance(sup, dict)
        assert {
            "additional_sending_mentions",
            "additional_receiving_mentions",
            "additional_spillover_mentions",
            "sending_subsystem_fills",
            "receiving_subsystem_fills",
            "spillover_subsystem_fills",
            "supplementary_flows",
        } <= set(sup.keys())

        # Items survived validation
        assert sup["additional_sending_mentions"]
        assert sup["additional_sending_mentions"][0]["name"] == \
            "UK cereal merchants"
        assert sup["additional_sending_mentions"][0][
            "evidence_passage_ids"
        ] == [1, 2]
        assert sup["supplementary_flows"]
        assert sup["supplementary_flows"][0]["category"] == "information"

    def test_supplement_block_rendered_in_formatted(self, mock_rag_engine):
        from tests.conftest import _RecordingMockLLMClient

        client = _RecordingMockLLMClient(
            responses=[_DRAFT_ANALYSIS, _VALID_SUPPLEMENT_JSON]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Impact of feed barley supply in the UK")
        assert "SUPPLEMENTARY STRUCTURED EXTRACTION" in result.formatted
        assert "UK cereal merchants" in result.formatted
        assert "UK feed mill sector" in result.formatted
        assert "Market intelligence and price feedback" in result.formatted
        # Citation markers carry through
        assert "[T1:1]" in result.formatted

    def test_supplement_block_ordered_before_evidence_block(
        self, mock_rag_engine,
    ):
        """The supplement should appear between the main analysis and the
        SUPPORTING EVIDENCE FROM LITERATURE block so readers see it in
        context before the evidence list."""
        from tests.conftest import _RecordingMockLLMClient

        client = _RecordingMockLLMClient(
            responses=[_DRAFT_ANALYSIS, _VALID_SUPPLEMENT_JSON]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Impact of feed barley supply in the UK")
        sup_idx = result.formatted.find("SUPPLEMENTARY STRUCTURED EXTRACTION")
        evid_idx = result.formatted.find("SUPPORTING EVIDENCE FROM LITERATURE")
        assert sup_idx != -1
        assert evid_idx != -1
        assert sup_idx < evid_idx

    def test_main_analysis_body_not_modified_by_supplement(
        self, mock_rag_engine,
    ):
        """The supplement must never silently rewrite content that came
        from the first (main) LLM response — users need to tell the
        two apart."""
        from tests.conftest import _RecordingMockLLMClient

        client = _RecordingMockLLMClient(
            responses=[_DRAFT_ANALYSIS, _VALID_SUPPLEMENT_JSON]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Impact of feed barley supply in the UK")
        # The original LLM-written flow should still be in the output
        # and should appear BEFORE the supplement block
        main_flow_idx = result.formatted.find("Feed barley exports")
        sup_idx = result.formatted.find("SUPPLEMENTARY STRUCTURED EXTRACTION")
        assert main_flow_idx != -1
        assert sup_idx != -1
        assert main_flow_idx < sup_idx
        # parsed flows are not mutated by the supplement — still has
        # exactly what the parser extracted from the first response
        assert all(
            "Market intelligence" not in flow.get("description", "")
            for flow in result.parsed.iter_flow_entries()
        )


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestStructuredExtractionFallbacks:
    def test_invalid_json_fallback(self, mock_rag_engine, caplog):
        from tests.conftest import _RecordingMockLLMClient

        # Second call returns non-JSON garbage
        client = _RecordingMockLLMClient(
            responses=[_DRAFT_ANALYSIS, "not JSON at all — just prose"]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Impact of feed barley supply in the UK")
        # Both LLM calls happened
        assert client.call_count == 2
        # But supplement is None and the block is absent
        assert result.structured_supplement is None
        assert "SUPPLEMENTARY STRUCTURED EXTRACTION" not in result.formatted
        # And the main analysis still comes through intact
        assert "Feed barley exports" in result.formatted

    def test_llm_exception_fallback(self, mock_rag_engine):
        """If the second LLM call raises, the pipeline should log a
        warning but still return a valid AnalysisResult."""

        class _FlakyClient:
            def __init__(self):
                self._call = 0
                self.last_messages = None

            def chat(self, messages, temperature=0.7, max_tokens=None):
                from metacouplingllm.llm.client import LLMResponse
                self._call += 1
                self.last_messages = messages
                if self._call == 1:
                    return LLMResponse(
                        content=_DRAFT_ANALYSIS,
                        usage={"prompt_tokens": 100, "completion_tokens": 50},
                    )
                raise RuntimeError("simulated extraction failure")

        client = _FlakyClient()
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Impact of feed barley supply in the UK")
        assert isinstance(result, AnalysisResult)
        assert result.structured_supplement is None
        assert "SUPPLEMENTARY STRUCTURED EXTRACTION" not in result.formatted

    def test_empty_hits_no_extraction(self, mock_llm_client):
        """If pre-retrieval returned no passages, the supplement step
        must short-circuit without an LLM call."""
        from tests.conftest import _RecordingMockLLMClient, _RecordingMockRagEngine

        empty_engine = _RecordingMockRagEngine(results=[])
        client = _RecordingMockLLMClient(responses=[_DRAFT_ANALYSIS])
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = empty_engine

        result = advisor.analyze("Obscure topic the corpus won't match")
        # Only one LLM call happened (the main analysis)
        assert client.call_count == 1
        assert result.structured_supplement is None

    def test_post_hoc_mode_does_not_run_supplement(self, mock_rag_engine):
        """Structured extraction is gated on pre_retrieval mode because
        the supplement uses ``self._last_rag_hits`` which is populated
        only in that mode."""
        from tests.conftest import _RecordingMockLLMClient

        client = _RecordingMockLLMClient(responses=[_DRAFT_ANALYSIS])
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="post_hoc",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Impact of feed barley supply in the UK")
        assert client.call_count == 1
        assert result.structured_supplement is None


# ---------------------------------------------------------------------------
# Validation / normalisation in _structured_extract_supplement
# ---------------------------------------------------------------------------


class TestStructuredExtractionValidation:
    def test_out_of_range_passage_ids_stripped(self, mock_rag_engine):
        """Passage IDs outside the 1..N range must be filtered out."""
        from tests.conftest import _RecordingMockLLMClient

        # mock_rag_engine has 5 fixture passages → valid ids are 1..5.
        # Include 99 (invalid) to confirm it's stripped.
        bad_json = json.dumps({
            "additional_sending_mentions": [
                {"name": "valid mention", "evidence_passage_ids": [1, 99]},
                {"name": "only-invalid", "evidence_passage_ids": [99]},
            ],
            "additional_receiving_mentions": [],
            "additional_spillover_mentions": [],
            "sending_subsystem_fills": {
                "human_subsystem": None,
                "natural_subsystem": None,
                "geographic_scope": None,
                "evidence_passage_ids": [],
            },
            "receiving_subsystem_fills": {
                "human_subsystem": None,
                "natural_subsystem": None,
                "geographic_scope": None,
                "evidence_passage_ids": [],
            },
            "spillover_subsystem_fills": {
                "human_subsystem": None,
                "natural_subsystem": None,
                "geographic_scope": None,
                "evidence_passage_ids": [],
            },
            "supplementary_flows": [],
        })

        client = _RecordingMockLLMClient(
            responses=[_DRAFT_ANALYSIS, bad_json]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Test")
        sup = result.structured_supplement
        assert sup is not None
        mentions = sup["additional_sending_mentions"]
        # "valid mention" kept with only the in-range id
        assert len(mentions) == 1
        assert mentions[0]["name"] == "valid mention"
        assert mentions[0]["evidence_passage_ids"] == [1]

    def test_invalid_flow_category_rejected(self, mock_rag_engine):
        """Flows with a category outside the whitelist are dropped."""
        from tests.conftest import _RecordingMockLLMClient

        bad_json = json.dumps({
            "additional_sending_mentions": [],
            "additional_receiving_mentions": [],
            "additional_spillover_mentions": [],
            "sending_subsystem_fills": {
                "human_subsystem": None,
                "natural_subsystem": None,
                "geographic_scope": None,
                "evidence_passage_ids": [],
            },
            "receiving_subsystem_fills": {
                "human_subsystem": None,
                "natural_subsystem": None,
                "geographic_scope": None,
                "evidence_passage_ids": [],
            },
            "spillover_subsystem_fills": {
                "human_subsystem": None,
                "natural_subsystem": None,
                "geographic_scope": None,
                "evidence_passage_ids": [],
            },
            "supplementary_flows": [
                {
                    "category": "bogus",   # not in whitelist → dropped
                    "direction": "A \u2192 B",
                    "description": "invalid",
                    "evidence_passage_ids": [1],
                },
                {
                    "category": "matter",  # valid → kept
                    "direction": "A \u2192 B",
                    "description": "valid",
                    "evidence_passage_ids": [1],
                },
            ],
        })

        client = _RecordingMockLLMClient(
            responses=[_DRAFT_ANALYSIS, bad_json]
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = mock_rag_engine

        result = advisor.analyze("Test")
        sup = result.structured_supplement
        assert sup is not None
        flows = sup["supplementary_flows"]
        assert len(flows) == 1
        assert flows[0]["category"] == "matter"
        assert flows[0]["description"] == "valid"


# ---------------------------------------------------------------------------
# Truncation removal
# ---------------------------------------------------------------------------


class TestStructuredExtractionNoTruncation:
    """Regression guard: passages are no longer truncated at 600 chars
    before being handed to the extraction LLM. This was previously
    cutting off bilateral country data at position ~689 in long
    Results chunks (e.g., Duan 2022 § 4)."""

    def test_long_passage_passed_in_full(self, mock_llm_client, fake_retrieval_results):
        """Make the extraction LLM record what it actually sees, then
        check that text past char 600 (e.g., a sentinel at pos 700)
        is present in the prompt."""
        from tests.conftest import _RecordingMockLLMClient, _RecordingMockRagEngine
        from metacouplingllm.knowledge.rag import RetrievalResult, TextChunk

        # Build a single chunk with a sentinel deliberately past char 700
        long_body = ("lorem ipsum dolor sit amet " * 30)
        assert len(long_body) > 800
        # Inject an unmistakable marker string near char 750
        injection_site = 750
        chunk_text = (
            long_body[:injection_site]
            + "SENTINEL_PAST_600_Korea_Japan_Russia "
            + long_body[injection_site:]
        )
        hits = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="test_long",
                    paper_title="Long paper",
                    authors="Test",
                    year=2024,
                    section="4. Results",
                    text=chunk_text,
                    chunk_index=0,
                ),
                score=0.9,
            )
        ]
        engine = _RecordingMockRagEngine(results=hits)

        # Second call (extraction) returns empty-but-valid JSON so the
        # test is about WHAT the LLM saw, not what it extracted.
        draft = "### 1. Coupling Classification\nTest."
        empty_supp = json.dumps({
            "additional_sending_mentions": [],
            "additional_receiving_mentions": [],
            "additional_spillover_mentions": [],
            "sending_subsystem_fills": {
                "human_subsystem": None, "natural_subsystem": None,
                "geographic_scope": None, "evidence_passage_ids": [],
            },
            "receiving_subsystem_fills": {
                "human_subsystem": None, "natural_subsystem": None,
                "geographic_scope": None, "evidence_passage_ids": [],
            },
            "spillover_subsystem_fills": {
                "human_subsystem": None, "natural_subsystem": None,
                "geographic_scope": None, "evidence_passage_ids": [],
            },
            "supplementary_flows": [],
        })
        client = _RecordingMockLLMClient(responses=[draft, empty_supp])
        advisor = MetacouplingAssistant(
            llm_client=client, max_examples=0,
            rag_mode="pre_retrieval", rag_structured_extraction=True,
        )
        advisor._rag_engine = engine

        advisor.analyze("test query")

        # The SECOND LLM call (index 1) is the extraction. Its user
        # message should contain the full chunk text, including our
        # sentinel string that sits past the old 600-char cutoff.
        assert client.call_count == 2
        extraction_messages = client.calls[1]
        extraction_user = next(
            m.content for m in extraction_messages if m.role == "user"
        )
        assert "SENTINEL_PAST_600_Korea_Japan_Russia" in extraction_user


# ---------------------------------------------------------------------------
# Supplement → map_data bridge
# ---------------------------------------------------------------------------


class TestSupplementIntoMapBridge:
    """Verifies that `_merge_supplement_into_map_data` enriches
    `parsed.map_data` with countries extracted into the supplement."""

    def _base_map_data(self, focal="CHN"):
        return {
            "focal_country": focal,
            "receiving_countries": [],
            "spillover_countries": [],
            "flows": [],
        }

    def _supplement_with_countries(self):
        return {
            "additional_sending_mentions": [
                {"name": "Thailand (2.10 Mt outbound)",
                 "evidence_passage_ids": [6]},
            ],
            "additional_receiving_mentions": [
                {"name": "Korea (2.65 MtCO2 inbound tourism)",
                 "evidence_passage_ids": [6]},
                {"name": "Japan (1.92 MtCO2 inbound tourism)",
                 "evidence_passage_ids": [6]},
                {"name": "Russian Federation (1.46 MtCO2)",
                 "evidence_passage_ids": [6]},
            ],
            "additional_spillover_mentions": [],
            "sending_subsystem_fills": {},
            "receiving_subsystem_fills": {},
            "spillover_subsystem_fills": {},
            "supplementary_flows": [
                {
                    "category": "people",
                    "direction": "Korea \u2192 China",
                    "description": "Inbound Korean tourists",
                    "evidence_passage_ids": [6],
                },
                {
                    "category": "people",
                    "direction": "China \u2192 Thailand",
                    "description": "Outbound Chinese tourists",
                    "evidence_passage_ids": [6],
                },
            ],
        }

    def _advisor(self, mock_llm_client):
        return MetacouplingAssistant(
            llm_client=mock_llm_client, max_examples=0,
            rag_mode="pre_retrieval", rag_structured_extraction=True,
        )

    def test_receiving_countries_merged(self, mock_llm_client):
        """Korea, Japan, Russia should appear in receiving_countries
        as ISO alpha-3 codes after the bridge runs."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        advisor = self._advisor(mock_llm_client)
        parsed = ParsedAnalysis()
        parsed.map_data = self._base_map_data()
        advisor._merge_supplement_into_map_data(
            parsed, self._supplement_with_countries(),
        )
        recv = parsed.map_data["receiving_countries"]
        assert "KOR" in recv
        assert "JPN" in recv
        assert "RUS" in recv
        # Sending mention (Thailand) also surfaces on the map as a
        # receiving partner (symmetric tourism case).
        assert "THA" in recv

    def test_supplementary_flows_added_with_iso_codes(self, mock_llm_client):
        from metacouplingllm.llm.parser import ParsedAnalysis
        advisor = self._advisor(mock_llm_client)
        parsed = ParsedAnalysis()
        parsed.map_data = self._base_map_data()
        advisor._merge_supplement_into_map_data(
            parsed, self._supplement_with_countries(),
        )
        flows = parsed.map_data["flows"]
        # Both flows resolved to ISO codes
        pairs = {(f["source_country"], f["target_country"]) for f in flows}
        assert ("KOR", "CHN") in pairs
        assert ("CHN", "THA") in pairs
        # Every merged flow carries a category and description
        for f in flows:
            assert f["category"] in {
                "people", "matter", "capital", "information",
                "energy", "organisms",
            }
            assert f["description"]

    def test_does_not_overwrite_primary_extraction(self, mock_llm_client):
        """If `_extract_map_data_from_analysis` already populated a
        country/flow, the bridge must not duplicate or replace it."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        advisor = self._advisor(mock_llm_client)
        parsed = ParsedAnalysis()
        parsed.map_data = {
            "focal_country": "CHN",
            "receiving_countries": ["KOR"],      # already there
            "spillover_countries": [],
            "flows": [
                # already there — same shape as primary extractor
                {
                    "category": "people",
                    "source_country": "KOR",
                    "target_country": "CHN",
                    "direction": "KOR \u2192 CHN",
                    "description": "existing flow",
                    "kind": "direct",
                    "confidence": 0.9,
                    "evidence": [],
                },
            ],
        }
        before_recv = list(parsed.map_data["receiving_countries"])
        before_flows_len = len(parsed.map_data["flows"])
        advisor._merge_supplement_into_map_data(
            parsed, self._supplement_with_countries(),
        )
        # KOR appears exactly once (not duplicated)
        assert parsed.map_data["receiving_countries"].count("KOR") == 1
        # New countries DID get added
        assert "JPN" in parsed.map_data["receiving_countries"]
        # Existing flow preserved
        kor_chn_flows = [
            f for f in parsed.map_data["flows"]
            if f["source_country"] == "KOR" and f["target_country"] == "CHN"
        ]
        assert len(kor_chn_flows) == 1
        assert kor_chn_flows[0]["description"] == "existing flow"

    def test_unresolvable_country_name_skipped(self, mock_llm_client):
        """A supplement mention whose name doesn't resolve to any ISO
        code should be logged and skipped — never crash."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        advisor = self._advisor(mock_llm_client)
        parsed = ParsedAnalysis()
        parsed.map_data = self._base_map_data()
        bogus_supplement = {
            "additional_sending_mentions": [],
            "additional_receiving_mentions": [
                {"name": "Xkjzbogusland", "evidence_passage_ids": [1]},
                {"name": "Japan", "evidence_passage_ids": [1]},
            ],
            "additional_spillover_mentions": [],
            "sending_subsystem_fills": {},
            "receiving_subsystem_fills": {},
            "spillover_subsystem_fills": {},
            "supplementary_flows": [],
        }
        advisor._merge_supplement_into_map_data(parsed, bogus_supplement)
        assert parsed.map_data["receiving_countries"] == ["JPN"]

    def test_focal_country_never_added_as_receiving(self, mock_llm_client):
        """The supplement occasionally names the focal country itself
        as a 'receiving' mention (bidirectional tourism case). The
        bridge must not add the focal country to its own
        receiving_countries."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        advisor = self._advisor(mock_llm_client)
        parsed = ParsedAnalysis()
        parsed.map_data = self._base_map_data(focal="CHN")
        supp = {
            "additional_sending_mentions": [],
            "additional_receiving_mentions": [
                {"name": "China", "evidence_passage_ids": [1]},
                {"name": "Japan", "evidence_passage_ids": [1]},
            ],
            "additional_spillover_mentions": [],
            "sending_subsystem_fills": {},
            "receiving_subsystem_fills": {},
            "spillover_subsystem_fills": {},
            "supplementary_flows": [],
        }
        advisor._merge_supplement_into_map_data(parsed, supp)
        assert "CHN" not in parsed.map_data["receiving_countries"]
        assert "JPN" in parsed.map_data["receiving_countries"]


# ---------------------------------------------------------------------------
# Draft-summary caps in _structured_extract_supplement
# ---------------------------------------------------------------------------


class TestStructuredExtractionDraftCaps:
    """The supplement LLM receives a brief recap of the parsed analysis
    (the "draft summary") so it can identify what's already been
    covered and avoid re-emitting duplicates.  The per-flow
    ``description`` cap and the three per-system subfield caps were
    historically tighter than the same fields on the map-extraction
    path.  PR #9 raised them to 500 / 400 chars respectively to match
    the PR #8 changes; these tests prove the new caps actually flow
    through to what the supplement LLM sees."""

    @staticmethod
    def _empty_supplement_json() -> str:
        """JSON the supplement LLM returns when it finds nothing —
        used as a no-op response so the tests focus on what the LLM
        was *shown*, not what it extracted."""
        return json.dumps({
            "additional_sending_mentions": [],
            "additional_receiving_mentions": [],
            "additional_spillover_mentions": [],
            "sending_subsystem_fills": {
                "human_subsystem": None, "natural_subsystem": None,
                "geographic_scope": None, "evidence_passage_ids": [],
            },
            "receiving_subsystem_fills": {
                "human_subsystem": None, "natural_subsystem": None,
                "geographic_scope": None, "evidence_passage_ids": [],
            },
            "spillover_subsystem_fills": {
                "human_subsystem": None, "natural_subsystem": None,
                "geographic_scope": None, "evidence_passage_ids": [],
            },
            "supplementary_flows": [],
        })

    @staticmethod
    def _build_advisor_with_parsed(parsed) -> tuple[
        "MetacouplingAssistant", "object"
    ]:
        """Wire up an assistant with a recording LLM client and a mock
        RAG engine that returns a single retrieval hit (so the
        supplement function actually runs — it early-returns when
        ``_last_rag_hits`` is empty).

        Returns ``(advisor, captured_user_text_callable)``.  Calling
        ``advisor._structured_extract_supplement(parsed)`` populates
        the captured prompt; the returned callable yields it.
        """
        from tests.conftest import (
            _RecordingMockLLMClient, _RecordingMockRagEngine,
        )
        from metacouplingllm.knowledge.rag import RetrievalResult, TextChunk

        # One trivial RAG hit so the supplement actually runs.
        hits = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="dummy",
                    paper_title="Dummy paper",
                    authors="Test",
                    year=2024,
                    section="Discussion",
                    text="Dummy chunk content for the supplement to read.",
                    chunk_index=0,
                ),
                score=0.9,
            )
        ]
        engine = _RecordingMockRagEngine(results=hits)
        client = _RecordingMockLLMClient(
            responses=[
                TestStructuredExtractionDraftCaps._empty_supplement_json(),
            ],
        )
        advisor = MetacouplingAssistant(
            llm_client=client,
            max_examples=0,
            rag_mode="pre_retrieval",
            rag_structured_extraction=True,
        )
        advisor._rag_engine = engine
        # The supplement reads ``self._last_rag_hits``; populate it
        # directly rather than running the full ``analyze()``.
        advisor._last_rag_hits = hits

        def get_user_text() -> str:
            assert client.call_count == 1, (
                f"expected exactly one supplement call, got "
                f"{client.call_count}"
            )
            user = next(
                m.content for m in client.calls[0] if m.role == "user"
            )
            return user

        return advisor, get_user_text

    # --- Flow description cap (100 -> 500) -------------------------------

    def test_draft_flow_description_under_500_chars_preserved(self):
        from ._helpers import make_parsed_analysis

        # ~300-char description with bilateral specifics at the end
        long_desc = (
            "Soybeans exported from Mato Grosso, Brazil to Henan, "
            "Shanghai, Liaoning, Guangdong, and other Chinese coastal "
            "provinces, totaling 35.4 million tonnes per year across "
            "the 2020-2024 period, with particular concentration in "
            "the bilateral relationship with eastern China"
        )
        assert 100 < len(long_desc) <= 500
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={"sending": {"name": "Brazilian soybean industry"}},
            flows=[{
                "category": "matter",
                "direction": "Brazil → China",
                "description": long_desc,
            }],
        )
        advisor, get_user_text = self._build_advisor_with_parsed(parsed)
        advisor._structured_extract_supplement(parsed)
        user_text = get_user_text()
        # Each bilateral country name should reach the supplement LLM
        for needle in ("Henan", "Shanghai", "Liaoning", "Guangdong"):
            assert needle in user_text, (
                f"{needle!r} missing from supplement prompt — "
                f"flow description cap truncating below 500 chars?"
            )

    def test_draft_flow_description_truncated_at_500(self):
        from ._helpers import make_parsed_analysis

        # 200 chars of filler, then bilateral marker, then more
        # filler, then a marker placed past char 500.
        early_marker = " Henan, Shanghai, "
        long_desc = "X" * 200 + early_marker + "Y" * 400 + "MARKER_PAST_CAP"
        assert len(long_desc) > 500
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={"sending": {"name": "Brazilian soybean industry"}},
            flows=[{
                "category": "matter",
                "direction": "Brazil → China",
                "description": long_desc,
            }],
        )
        advisor, get_user_text = self._build_advisor_with_parsed(parsed)
        advisor._structured_extract_supplement(parsed)
        user_text = get_user_text()
        # Early content (~char 200) survives.
        assert "Henan" in user_text
        assert "Shanghai" in user_text
        # Content past char 500 is dropped.
        assert "MARKER_PAST_CAP" not in user_text

    # --- Subsystem caps (120 -> 400) -------------------------------------

    def test_draft_subsystem_caps_at_400(self):
        from ._helpers import make_parsed_analysis

        # Each subsystem field has a unique marker placed at char ~350
        # (within the new 400 cap).  All three should appear in the
        # supplement prompt.
        human_text = (
            "X" * 350 + " HUMAN_MARKER_AT_350 " + "Y" * 200
            + "PAST_CAP_HUMAN"
        )
        natural_text = (
            "X" * 350 + " NATURAL_MARKER_AT_350 " + "Y" * 200
            + "PAST_CAP_NATURAL"
        )
        geo_text = (
            "X" * 350 + " GEO_MARKER_AT_350 " + "Y" * 200
            + "PAST_CAP_GEO"
        )
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={
                "sending": {
                    "name": "Test sending system",
                    "human_subsystem": human_text,
                    "natural_subsystem": natural_text,
                    "geographic_scope": geo_text,
                },
            },
        )
        advisor, get_user_text = self._build_advisor_with_parsed(parsed)
        advisor._structured_extract_supplement(parsed)
        user_text = get_user_text()
        # Markers within the 400-char cap survive
        assert "HUMAN_MARKER_AT_350" in user_text
        assert "NATURAL_MARKER_AT_350" in user_text
        assert "GEO_MARKER_AT_350" in user_text
        # Markers past the 400-char cap are dropped
        assert "PAST_CAP_HUMAN" not in user_text
        assert "PAST_CAP_NATURAL" not in user_text
        assert "PAST_CAP_GEO" not in user_text
