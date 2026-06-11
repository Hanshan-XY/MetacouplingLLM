"""Tests for the LLM-assisted helpers in
metacouplingllm.indicators.llm (PR #36).

Uses the existing ``_RecordingMockLLMClient`` from
``tests/conftest.py`` to stub LLM responses.  The mock isn't an
instance of any real adapter, so the strict-output dispatch chain
inside ``_call_with_strict_json`` falls through to the plain-chat
path -- which still works because we make the mock return valid
JSON strings.

Module-level ``pytest.importorskip("pandas")`` skips the file
cleanly when pandas isn't installed (mirrors PR #35's pattern).
"""

from __future__ import annotations

import json
import re
import warnings

import pytest

pd = pytest.importorskip("pandas")

from metacouplingllm.indicators import (  # noqa: E402
    LLMTrace,
    check_inputs,
    classify_ambiguous_edges,
    classify_coupling,
    define_study,
    interpret_results,
    write_methods,
)
from tests.conftest import _RecordingMockLLMClient  # noqa: E402


# ---------------------------------------------------------------------------
# TestDefineStudy (3)
# ---------------------------------------------------------------------------


class TestDefineStudy:
    def test_parses_structured_config_from_json_response(self):
        config_json = json.dumps({
            "focal_system": "Brazil",
            "flow_type": "soybean trade",
            "flow_unit": "tons",
            "intracoupling_rule": "flows within Brazil",
            "pericoupling_rule": "flows to neighbours",
            "telecoupling_rule": "flows to non-adjacent",
            "recommended_input_level": "directed weighted edge list",
            "required_columns": ["origin_id", "destination_id"],
            "warnings": [],
        })
        client = _RecordingMockLLMClient(responses=[config_json])
        result, trace = define_study(
            "I study soybean flows from Brazil...", llm_client=client,
        )
        assert result["focal_system"] == "Brazil"
        assert result["flow_type"] == "soybean trade"
        assert isinstance(trace, LLMTrace)
        assert trace.prompt_version == "define_study_v1"

    def test_rejects_empty_description(self):
        client = _RecordingMockLLMClient(responses=["{}"])
        with pytest.raises(ValueError, match="description must not be empty"):
            define_study("", llm_client=client)
        with pytest.raises(ValueError, match="description must not be empty"):
            define_study("   ", llm_client=client)

    def test_unparseable_response_raises_runtime_error(self):
        client = _RecordingMockLLMClient(responses=["not json at all"])
        with pytest.raises(RuntimeError, match="parseable JSON object"):
            define_study("real description", llm_client=client)


# ---------------------------------------------------------------------------
# TestCheckInputs (3)
# ---------------------------------------------------------------------------


class TestCheckInputs:
    def test_can_compute_all_three_indicator_families(self):
        result_json = json.dumps({
            "can_compute_flow_shares": True,
            "can_compute_mfe": True,
            "can_compute_mfci": True,
            "missing_information": [],
            "warnings": [],
        })
        client = _RecordingMockLLMClient(responses=[result_json])
        result, _ = check_inputs(
            {
                "columns": ["origin_id", "destination_id", "coupling_type",
                            "flow_value"],
                "row_count": 100,
                "has_coupling_col": True,
            },
            llm_client=client,
        )
        assert result["can_compute_flow_shares"] is True
        assert result["can_compute_mfci"] is True

    def test_missing_partner_level_data_blocks_mfci(self):
        result_json = json.dumps({
            "can_compute_flow_shares": True,
            "can_compute_mfe": True,
            "can_compute_mfci": False,
            "missing_information": [
                "partner-level flows within each coupling type"
            ],
            "warnings": [],
        })
        client = _RecordingMockLLMClient(responses=[result_json])
        result, _ = check_inputs(
            {"columns": ["focal", "F_I", "F_P", "F_T"], "row_count": 1},
            llm_client=client,
        )
        assert result["can_compute_mfci"] is False
        assert "partner-level" in result["missing_information"][0]

    def test_intracoupling_self_loop_warning_surfaced(self):
        result_json = json.dumps({
            "can_compute_flow_shares": True,
            "can_compute_mfe": True,
            "can_compute_mfci": True,
            "missing_information": [],
            "warnings": [
                "Intracoupling is represented as a single self-loop; IFCI "
                "will not be substantively meaningful."
            ],
        })
        client = _RecordingMockLLMClient(responses=[result_json])
        result, _ = check_inputs(
            {"columns": ["focal", "partner", "coupling_type", "flow_value"],
             "row_count": 5},
            sample_rows=[
                {"focal": "Brazil", "partner": "Brazil", "coupling_type": "I",
                 "flow_value": 10},
            ],
            llm_client=client,
        )
        assert any("self-loop" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# TestClassifyAmbiguousEdges (4)
# ---------------------------------------------------------------------------


def _classifications_response(entries: list[dict]) -> str:
    """Helper: wrap a list of edge classifications in the schema shape."""
    return json.dumps({"classifications": entries})


class TestClassifyAmbiguousEdges:
    def test_high_confidence_classification(self):
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Chile"},
        ])
        response = _classifications_response([
            {
                "origin": "Brazil", "destination": "Chile",
                "suggested_coupling_type": "P",
                "confidence": "high",
                "reason": "Chile shares a border with Brazil.",
                "needs_user_confirmation": False,
            },
        ])
        client = _RecordingMockLLMClient(responses=[response])
        result, _ = classify_ambiguous_edges(edges, {}, llm_client=client)
        assert result.iloc[0]["suggested_coupling_type"] == "P"
        assert result.iloc[0]["confidence"] == "high"
        # pandas stores bools as numpy.bool_; use ==/bool() not `is`.
        assert bool(result.iloc[0]["needs_user_confirmation"]) is False

    def test_unknown_when_insufficient_information(self):
        edges = pd.DataFrame([
            {"origin_id": "X", "destination_id": "Y"},
        ])
        response = _classifications_response([
            {
                "origin": "X", "destination": "Y",
                "suggested_coupling_type": "unknown",
                "confidence": "low",
                "reason": "Insufficient information to classify.",
                "needs_user_confirmation": True,
            },
        ])
        client = _RecordingMockLLMClient(responses=[response])
        result, _ = classify_ambiguous_edges(edges, {}, llm_client=client)
        assert result.iloc[0]["suggested_coupling_type"] == "unknown"
        assert bool(result.iloc[0]["needs_user_confirmation"]) is True

    def test_multi_row_batch_classification(self):
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Argentina"},
            {"origin_id": "Brazil", "destination_id": "China"},
            {"origin_id": "Brazil", "destination_id": "Chile"},
        ])
        response = _classifications_response([
            {"origin": "Brazil", "destination": "Argentina",
             "suggested_coupling_type": "P", "confidence": "high",
             "reason": "border", "needs_user_confirmation": False},
            {"origin": "Brazil", "destination": "China",
             "suggested_coupling_type": "T", "confidence": "high",
             "reason": "distant", "needs_user_confirmation": False},
            {"origin": "Brazil", "destination": "Chile",
             "suggested_coupling_type": "P", "confidence": "medium",
             "reason": "same continent, no shared border",
             "needs_user_confirmation": True},
        ])
        client = _RecordingMockLLMClient(responses=[response])
        result, _ = classify_ambiguous_edges(edges, {}, llm_client=client)
        assert len(result) == 3
        assert list(result["suggested_coupling_type"]) == ["P", "T", "P"]

    def test_preserves_input_index_for_downstream_merge(self):
        """When the input DataFrame has a non-default index (e.g., the
        subset of NaN-rows from a parent DataFrame), the output must
        carry the same index so the caller can merge back by index."""
        edges = pd.DataFrame(
            [
                {"origin_id": "A", "destination_id": "X"},
                {"origin_id": "B", "destination_id": "Y"},
            ],
            index=[7, 12],  # non-sequential, non-zero-based
        )
        response = _classifications_response([
            {"origin": "A", "destination": "X",
             "suggested_coupling_type": "T", "confidence": "high",
             "reason": "", "needs_user_confirmation": False},
            {"origin": "B", "destination": "Y",
             "suggested_coupling_type": "P", "confidence": "high",
             "reason": "", "needs_user_confirmation": False},
        ])
        client = _RecordingMockLLMClient(responses=[response])
        result, _ = classify_ambiguous_edges(edges, {}, llm_client=client)
        assert list(result.index) == [7, 12]
        assert result.loc[7, "suggested_coupling_type"] == "T"
        assert result.loc[12, "suggested_coupling_type"] == "P"


# ---------------------------------------------------------------------------
# TestInterpretResults (2)
# ---------------------------------------------------------------------------


class TestInterpretResults:
    def test_academic_audience_produces_prose(self):
        results = pd.DataFrame([
            {"focal_system_id": "Brazil", "IFS": 0.10, "PFS": 0.20,
             "TFS": 0.70, "MFE": 0.73, "TFCI": 0.62},
        ])
        client = _RecordingMockLLMClient(
            responses=[
                "Brazil's flows are dominated by telecoupling (TFS = 0.70)..."
            ],
        )
        text, trace = interpret_results(
            results, llm_client=client, audience="academic",
        )
        assert "Brazil" in text
        assert "telecoupling" in text.lower()
        # System prompt is the academic preset (no bullets / briefing
        # language).
        sys_msg = next(
            m for m in client.last_messages if m.role == "system"
        )
        assert "academic prose" in sys_msg.content

    def test_invalid_audience_raises(self):
        results = pd.DataFrame([{"IFS": 0.1}])
        client = _RecordingMockLLMClient(responses=["dummy"])
        with pytest.raises(ValueError, match="audience must be one of"):
            interpret_results(results, llm_client=client, audience="bogus")


# ---------------------------------------------------------------------------
# TestWriteMethods (1)
# ---------------------------------------------------------------------------


class TestWriteMethods:
    def test_returns_prose_with_trace(self):
        spec = {
            "focal_system": "Brazil",
            "indicators": ["IFS", "PFS", "TFS", "MFE", "TFCI"],
            "flow_unit": "tons",
        }
        client = _RecordingMockLLMClient(
            responses=[
                "We calculated Metacoupled Flow Shares (IFS, PFS, TFS), "
                "Metacoupled Flow Evenness (MFE), and Metacoupled Flow "
                "Concentration Index (MFCI) per Liu (2017) and standard "
                "Shannon and HHI conventions."
            ],
        )
        text, trace = write_methods(spec, llm_client=client)
        assert "Metacoupled" in text
        assert isinstance(trace, LLMTrace)
        assert trace.prompt_version == "write_methods_v1"


# ---------------------------------------------------------------------------
# TestClassifyCouplingIntegration (3)  -- Option A
# ---------------------------------------------------------------------------


class TestClassifyCouplingIntegration:
    def test_no_llm_client_preserves_pr35_behaviour(self):
        """Backwards-compat: when llm_client is None, behaviour is
        identical to PR #35.  NaN edges stay NaN + UserWarning."""
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Argentina"},
            {"origin_id": "Brazil", "destination_id": "Chile"},
        ])
        adjacency = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Argentina",
             "adjacent": 1},
            # Chile not listed -> not adjacent -> T
        ])
        # No llm_client passed.
        out = classify_coupling(edges, focal_id="Brazil", adjacency=adjacency)
        assert list(out["coupling_type"]) == ["P", "T"]
        # No LLM trace attached.
        assert "llm_classify_trace" not in out.attrs

    def test_llm_client_resolves_nan_edges(self):
        """When adjacency is None and an llm_client is provided, the
        function should ask the LLM to classify cross-system edges
        and merge the results back."""
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Brazil"},
            {"origin_id": "Brazil", "destination_id": "Chile"},
        ])
        # First edge is a self-loop (deterministic -> I); second has
        # no adjacency info and would be NaN without the LLM.
        response = _classifications_response([
            {"origin": "Brazil", "destination": "Chile",
             "suggested_coupling_type": "P", "confidence": "medium",
             "reason": "same continent", "needs_user_confirmation": True},
        ])
        client = _RecordingMockLLMClient(responses=[response])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = classify_coupling(
                edges, focal_id="Brazil",
                adjacency=None, llm_client=client,
            )
        assert list(out["coupling_type"]) == ["I", "P"]
        assert "llm_classify_trace" in out.attrs

    def test_unknown_llm_suggestion_leaves_nan(self):
        """When the LLM returns "unknown", the row stays NaN -- the
        package never lets the LLM invent adjacency facts silently
        (spec §16 item 3)."""
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Mars"},
        ])
        response = _classifications_response([
            {"origin": "Brazil", "destination": "Mars",
             "suggested_coupling_type": "unknown", "confidence": "low",
             "reason": "off-planet edge; not classifiable.",
             "needs_user_confirmation": True},
        ])
        client = _RecordingMockLLMClient(responses=[response])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = classify_coupling(
                edges, focal_id="Brazil",
                adjacency=None, llm_client=client,
            )
        # Row stays NaN.
        assert pd.isna(out["coupling_type"].iloc[0])


# ---------------------------------------------------------------------------
# TestLLMTrace (2)
# ---------------------------------------------------------------------------


class TestLLMTrace:
    def test_all_fields_populated_after_call(self):
        config_json = json.dumps({
            "focal_system": "X", "flow_type": "y", "flow_unit": "z",
            "intracoupling_rule": "", "pericoupling_rule": "",
            "telecoupling_rule": "", "recommended_input_level": "",
            "required_columns": [], "warnings": [],
        })
        client = _RecordingMockLLMClient(responses=[config_json])
        _, trace = define_study("desc", llm_client=client)
        assert trace.timestamp_utc
        assert trace.prompt_version == "define_study_v1"
        assert "researcher" in trace.system_prompt
        assert "desc" in trace.user_prompt
        assert trace.raw_response == config_json
        assert trace.usage == {
            "prompt_tokens": 100, "completion_tokens": 50,
        }

    def test_timestamp_is_utc_iso8601(self):
        client = _RecordingMockLLMClient(responses=["{}"])
        # define_study will raise RuntimeError on "{}" (missing
        # required fields), but the trace timestamp is built BEFORE
        # the parse check.  Use check_inputs instead which would
        # also parse... actually use a complete config.
        config_json = json.dumps({
            "focal_system": "X", "flow_type": "y", "flow_unit": "z",
            "intracoupling_rule": "", "pericoupling_rule": "",
            "telecoupling_rule": "", "recommended_input_level": "",
            "required_columns": [], "warnings": [],
        })
        client = _RecordingMockLLMClient(responses=[config_json])
        _, trace = define_study("desc", llm_client=client)
        # ISO 8601 with Z suffix, e.g., "2026-05-22T16:30:00Z"
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", trace.timestamp_utc,
        ), f"unexpected timestamp format: {trace.timestamp_utc!r}"
