"""Tests for core.py — MetacouplingAssistant main interface."""

import pytest

from metacouplingllm.core import AnalysisResult, MetacouplingAssistant
from metacouplingllm.llm.client import LLMResponse, Message


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE = """\
### 1. Coupling Classification

This research involves **telecoupling** between coffee production and \
consumption systems.

### 2. Systems Identification

- **Sending**: Ethiopian coffee regions with smallholder farms.
- **Receiving**: European markets importing specialty coffee.
- **Spillover**: Competing coffee origins (Colombia, Vietnam).

### 3. Flows Analysis

- [Material] Ethiopia → Europe: Coffee beans exported
- [Financial] Europe → Ethiopia: Payment and fair-trade premiums

### 4. Agents

- Ethiopian coffee farmers
- European importers and roasters
- Fair-trade certification bodies

### 5. Causes

**Proximate causes**
- European demand for single-origin coffee

**Underlying causes**
- Global coffee market dynamics

### 6. Effects

**Sending system**
- Income for farming communities

**Receiving system**
- Access to specialty coffee

### 7. Research Gaps and Suggestions

- Assess environmental footprint of coffee trade
- Investigate spillover on competing origins
"""

MOCK_REFINE_RESPONSE = """\
### 2. Systems Identification

Expanding on the spillover systems:

- **Spillover**: Colombia faces competition from Ethiopian specialty coffees, \
potentially affecting its market share and farmer incomes. Vietnam, as a \
major robusta producer, experiences indirect price effects.

### 7. Research Gaps and Suggestions

- Conduct comparative analysis of Ethiopian vs Colombian coffee supply chains
- Map the full supply chain to identify additional spillover systems
"""


class MockLLMClient:
    """A mock LLM client for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_messages = None

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                content=MOCK_LLM_RESPONSE,
                usage={"prompt_tokens": 500, "completion_tokens": 200},
            )
        return LLMResponse(
            content=MOCK_REFINE_RESPONSE,
            usage={"prompt_tokens": 800, "completion_tokens": 150},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMetacouplingAdvisor:
    def setup_method(self):
        self.mock_client = MockLLMClient()
        self.advisor = MetacouplingAssistant(
            llm_client=self.mock_client,
            temperature=0.5,
            max_tokens=2000,
            max_examples=2,
            verbose=False,
        )

    def test_analyze_returns_result(self):
        result = self.advisor.analyze("Coffee trade between Ethiopia and Europe")
        assert isinstance(result, AnalysisResult)

    def test_analyze_result_fields(self):
        result = self.advisor.analyze("Coffee trade")
        assert result.turn_number == 1
        assert result.raw == MOCK_LLM_RESPONSE
        assert isinstance(result.formatted, str)
        assert result.parsed.is_parsed
        assert result.usage is not None

    def test_analyze_sends_system_and_user_messages(self):
        self.advisor.analyze("My study on coffee trade")
        messages = self.mock_client.last_messages
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "coffee trade" in messages[1].content.lower()

    def test_analyze_system_prompt_has_framework_knowledge(self):
        self.advisor.analyze("Any study")
        system_msg = self.mock_client.last_messages[0].content
        assert "metacoupling" in system_msg.lower()
        assert "telecoupling" in system_msg.lower()

    def test_refine_after_analyze(self):
        self.advisor.analyze("Coffee trade")
        result = self.advisor.refine("Tell me more about spillover systems")
        assert isinstance(result, AnalysisResult)
        assert result.turn_number == 2

    def test_refine_with_focus(self):
        self.advisor.analyze("Coffee trade")
        result = self.advisor.refine(
            "More detail please",
            focus_component="systems",
        )
        assert result.turn_number == 2
        # Check that the user message mentions the focus component
        user_msgs = [m for m in self.mock_client.last_messages if m.role == "user"]
        assert any("systems" in m.content.lower() for m in user_msgs)

    def test_refine_before_analyze_raises(self):
        with pytest.raises(RuntimeError, match="Cannot refine"):
            self.advisor.refine("Some refinement")

    def test_conversation_history_grows(self):
        self.advisor.analyze("Coffee trade")
        assert len(self.advisor.history) == 3  # system + user + assistant

        self.advisor.refine("More info")
        assert len(self.advisor.history) == 5  # +user + assistant

    def test_reset_clears_history(self):
        self.advisor.analyze("Coffee trade")
        assert self.advisor.turn_count == 1

        self.advisor.reset()
        assert self.advisor.turn_count == 0
        assert len(self.advisor.history) == 0

    def test_analyze_resets_previous_conversation(self):
        self.advisor.analyze("Study 1")
        self.advisor.refine("Refine study 1")
        assert self.advisor.turn_count == 2

        # Second analyze should reset
        self.mock_client.call_count = 0  # Reset counter
        self.advisor.analyze("Study 2")
        assert self.advisor.turn_count == 1
        assert len(self.advisor.history) == 3  # system + user + assistant

    def test_turn_count(self):
        assert self.advisor.turn_count == 0
        self.advisor.analyze("Study")
        assert self.advisor.turn_count == 1
        self.advisor.refine("More")
        assert self.advisor.turn_count == 2


class TestAnalysisResult:
    def test_fields(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(raw_text="test"),
            formatted="Formatted text",
            raw="test",
            turn_number=1,
            usage={"tokens": 100},
        )
        assert result.parsed.raw_text == "test"
        assert result.formatted == "Formatted text"
        assert result.raw == "test"
        assert result.turn_number == 1
        assert result.usage == {"tokens": 100}

    def test_optional_usage(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.usage is None

    def test_map_field_default_none(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.map is None

    def test_web_map_signals_default_none(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.web_map_signals is None

    def test_map_notice_default_none(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.map_notice is None

    def test_flow_parse_warnings_default_empty(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.flow_parse_warnings == []

    def test_flow_parse_warnings_independent_per_instance(self):
        # Mutating one instance's list must not affect another's
        # (regression guard against accidentally sharing a default
        # mutable across instances).
        from metacouplingllm.llm.parser import ParsedAnalysis
        a = AnalysisResult(parsed=ParsedAnalysis(), formatted="",
                           raw="", turn_number=1)
        b = AnalysisResult(parsed=ParsedAnalysis(), formatted="",
                           raw="", turn_number=1)
        a.flow_parse_warnings.append({"direction": "x"})
        assert b.flow_parse_warnings == []


# ---------------------------------------------------------------------------
# Auto-map integration tests
# ---------------------------------------------------------------------------

# Mock LLM response with subnational (ADM1) geographic scope
MOCK_ADM1_RESPONSE = """\
### 1. Coupling Classification

This research involves **telecoupling** between Michigan's pork production \
and international consumption systems.

### 2. Systems Identification

**Sending System**: Michigan Pork Industry
- **Human subsystem**: Pork farmers, processors, and exporters in Michigan
- **Natural subsystem**: Agricultural land, water resources
- **Geographic scope**: Michigan, United States

**Receiving System**: International Markets
- **Human subsystem**: Importers and consumers in Japan and South Korea
- **Natural subsystem**: Agroecosystems
- **Geographic scope**: Japan, South Korea

**Spillover System**: Neighboring States
- **Human subsystem**: Competing pork producers in Ohio, Indiana
- **Natural subsystem**: Shared watersheds
- **Geographic scope**: Ohio, United States

### 3. Flows Analysis

- [Material] Michigan → Japan: Pork products exported
- [Financial] Japan → Michigan: Payment for pork

### 4. Agents

- Michigan pork farmers
- Japanese importers

### 5. Causes

**Proximate causes**
- Japanese demand for high-quality pork

### 6. Effects

**Sending system**
- Economic benefits for Michigan farming communities

### 7. Research Gaps and Suggestions

- Study environmental impacts
"""


class MockAdm1LLMClient:
    """Mock LLM client that returns subnational research response."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages, temperature=0.7, max_tokens=None):
        self.call_count += 1
        return LLMResponse(
            content=MOCK_ADM1_RESPONSE,
            usage={"prompt_tokens": 500, "completion_tokens": 300},
        )


MOCK_WATERSHED_RESPONSE = """\
### 1. Coupling Classification

This research involves intracoupling and telecoupling in a watershed context.

### 2. Systems Identification

**Sending System**: Grand River watershed soybean production
- **Human subsystem**: Farmers and regional traders
- **Natural subsystem**: River channels, wetlands, and soils
- **Geographic scope**: Grand River watershed

**Receiving System**: Distant soybean markets
- **Human subsystem**: Importers and processors
- **Natural subsystem**: Food system demand
- **Geographic scope**: Global markets

### 3. Flows Analysis

- [Material] Watershed -> importing markets: Soybeans exported
"""


class MockWatershedLLMClient:
    """Mock LLM client that returns unsupported watershed-scale geography."""

    def chat(self, messages, temperature=0.7, max_tokens=None):
        return LLMResponse(
            content=MOCK_WATERSHED_RESPONSE,
            usage={"prompt_tokens": 300, "completion_tokens": 180},
        )


class TestResolveAdm1FromAnalysis:
    """Test the _resolve_adm1_from_analysis static method."""

    def test_michigan_us_resolves(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Michigan Pork Industry",
                    "geographic_scope": "Michigan, United States",
                },
            },
        )
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code == "USA023"

    def test_no_systems_returns_none(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(systems={})
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code is None

    def test_country_level_returns_none(self):
        """When systems contain only country names, should return None."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Ethiopia",
                    "geographic_scope": "Ethiopia",
                },
                "receiving": {
                    "name": "European Markets",
                    "geographic_scope": "Europe",
                },
            },
        )
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code is None

    def test_flat_systems_returns_none(self):
        """Flat (string) systems have no sub-fields to resolve."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": "Ethiopian coffee regions",
                "receiving": "European markets",
            },
        )
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code is None

    def test_anhui_china_resolves(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Anhui Agriculture",
                    "geographic_scope": "Anhui, China",
                },
            },
        )
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code == "CHN001"

    def test_country_scale_brazil_analysis_does_not_pick_example_adm1(self):
        """National Brazil studies should not flip to ADM1 from examples."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Soybean-producing Brazil [W5]",
                    "geographic_scope": (
                        "Brazil, especially major soybean-producing regions "
                        "such as Mato Grosso and other frontier or "
                        "consolidated production areas"
                    ),
                },
                "receiving": {
                    "name": "Major distant soybean-importing markets",
                    "geographic_scope": "China and other global markets",
                },
            },
        )
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code is None

    def test_trade_word_does_not_trigger_false_adm1_map(self):
        """Trade-heavy country descriptions should not resolve to Trad, THA."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Brazil soybean trade system",
                    "geographic_scope": "Brazil",
                },
            },
        )
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code is None


class TestAutoMapDisabled:
    """Test that auto_map=False (default) produces no map."""

    def test_no_map_by_default(self):
        client = MockLLMClient()
        advisor = MetacouplingAssistant(llm_client=client)
        result = advisor.analyze("Coffee trade")
        assert result.map is None

    def test_no_map_notice_by_default(self):
        client = MockLLMClient()
        advisor = MetacouplingAssistant(llm_client=client)
        result = advisor.analyze("Coffee trade")
        assert "map" not in result.formatted.lower() or "metacoupling map" not in result.formatted.lower()


class TestAutoMapUnavailableNotice:
    """Tests for user-facing notices when auto-map cannot render."""

    def test_watershed_input_adds_unavailable_notice(self):
        advisor = MetacouplingAssistant(
            llm_client=MockWatershedLLMClient(),
            auto_map=True,
        )

        result = advisor.analyze("Impact of the Grand River watershed on sustainability")

        assert result.map is None
        assert result.map_notice is not None
        assert "did not generate a figure" in result.map_notice
        # The notice now lists the unsupported geometry kinds in a
        # comma-separated phrase: "city, watershed, protected-area,
        # reserve, or park geometries". Just check that 'watershed'
        # is mentioned, since this test is specifically about a
        # watershed input.
        assert "watershed" in result.map_notice
        assert result.map_notice in result.formatted


class TestAutoMapTrustsLLMAndUserFraming:
    """Tests for the auto-map dispatcher's two-layer decision logic:

    1. ``_has_unsupported_automap_scope`` must TRUST the LLM's
       structured map extraction.  If ``parsed.map_data`` has a
       resolvable ``focal_country`` or ``adm1_region``, the regex
       scan over prose must not fire (which would skip the map even
       when the LLM clearly identified a country/ADM1 focal).
    2. ``_generate_map`` must RESPECT the user's framing.  When the
       user's original query named no ADM1 region, prefer
       country-level rendering even when the LLM stamped an ADM1
       focal in the extraction.
    """

    def _make_parsed_with_map_data(self, **md_overrides):
        from ._helpers import make_parsed_analysis
        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Mexico",
                    "geographic_scope": (
                        "Mexico, with focal production in Michoacan "
                        "municipalities and watersheds around Tancitaro"
                    ),
                },
                "receiving": {
                    "name": "United States",
                    "geographic_scope": "United States",
                },
            },
        )
        parsed.map_data = {
            "focal_country": "MEX",
            "adm1_region": None,
            "receiving_countries": ["USA"],
            "spillover_countries": [],
            "flows": [],
            **md_overrides,
        }
        return parsed

    # --- Layer 1: regex-gate trusts structured extraction ---------------

    def test_regex_gate_bypassed_when_focal_country_set(self):
        """A ``focal_country`` from the LLM must short-circuit the
        prose-keyword scan even when the prose mentions
        municipalities/watersheds."""
        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = (
            "Impact of avocado production in Mexico on watersheds"
        )
        parsed = self._make_parsed_with_map_data(focal_country="MEX")
        assert advisor._has_unsupported_automap_scope(parsed) is False

    def test_regex_gate_bypassed_when_adm1_region_set(self):
        """Same trust applies to ``adm1_region`` even if
        ``focal_country`` is missing."""
        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = (
            "Avocado production in Michoacan watersheds"
        )
        parsed = self._make_parsed_with_map_data(
            focal_country=None, adm1_region="MEX016",
        )
        assert advisor._has_unsupported_automap_scope(parsed) is False

    def test_regex_gate_fires_when_no_structured_focal(self):
        """When both ``focal_country`` and ``adm1_region`` are absent,
        fall back to the regex check so users still get the
        sub-ADM1-geography notice."""
        from ._helpers import make_parsed_analysis
        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = (
            "Impact of the Grand River watershed on sustainability"
        )
        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Grand River watershed",
                    "geographic_scope": "Grand River watershed, Michigan",
                },
            },
        )
        parsed.map_data = {
            "focal_country": None,
            "adm1_region": None,
            "receiving_countries": [],
            "spillover_countries": [],
            "flows": [],
        }
        assert advisor._has_unsupported_automap_scope(parsed) is True

    # --- Layer 2: user-framing override ---------------------------------

    def test_user_query_with_no_adm1_returns_false(self):
        """``Mexican avocado trade`` names no ADM1 region."""
        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = (
            "Impact of avocado production and trade in Mexico "
            "on sustainability"
        )
        assert advisor._user_query_mentions_adm1() is False

    def test_user_query_with_explicit_adm1_returns_true(self):
        """User-named state ("Michoacán") must be recognised.

        The ADM1 database stores Mexican state names with Spanish
        diacritics — users typing the accented form match directly.
        ASCII-only typing ("Michoacan") falls through to country-level,
        which is the safe default; the docstring on
        ``_user_query_mentions_adm1`` explains how to force ADM1
        explicitly via ISO code.
        """
        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = (
            "Avocado production in Michoacán, Mexico and sustainability"
        )
        assert advisor._user_query_mentions_adm1() is True

    def test_user_query_country_name_does_not_count_as_adm1(self):
        """``Mexico`` is also a state name (Estado de Mexico) but
        when used at top level it's framing the country -- must not
        return True."""
        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = "Sustainability in Mexico"
        # If this returned True we'd never get country-level rendering
        # for any query mentioning the country "Mexico".
        assert advisor._user_query_mentions_adm1() is False

    def test_user_query_with_iso_adm1_code_returns_true(self):
        """ISO-style ADM1 codes ("MEX016") are recognised."""
        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = "Analyze focal region MEX016 trade"
        assert advisor._user_query_mentions_adm1() is True

    def test_empty_query_returns_false(self):
        """No query string -> no ADM1 mention -> country-level
        preferred (safe default)."""
        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = None
        assert advisor._user_query_mentions_adm1() is False
        advisor._original_query = ""
        assert advisor._user_query_mentions_adm1() is False

    # --- Integration: dispatcher uses Layer 2 to drop adm1_region -------

    def test_generate_map_drops_adm1_when_query_is_country_scoped(
        self, monkeypatch,
    ):
        """End-to-end: structured extraction returned adm1=MEX016 but
        the user's query said only "Mexico" -> dispatcher renders
        country-level, NOT ADM1."""
        captured: dict[str, object] = {}

        def fake_country_map(parsed, **kwargs):
            captured["country_called"] = True
            captured["kwargs"] = kwargs
            return "country-figure"

        def fake_adm1_map(*args, **kwargs):
            captured["adm1_called"] = True
            return "adm1-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.plot_analysis_map",
            fake_country_map,
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = (
            "Impact of avocado production and trade in Mexico "
            "on sustainability"
        )
        parsed = self._make_parsed_with_map_data(
            focal_country="MEX",
            adm1_region="MEX016",   # LLM picked Michoacan
            receiving_countries=["USA"],
        )
        result = advisor._generate_map(parsed)

        assert result == "country-figure"
        assert captured.get("country_called") is True
        assert "adm1_called" not in captured
        assert advisor._last_map_type == "country"

    def test_generate_map_keeps_adm1_when_original_query_is_unset(
        self, monkeypatch,
    ):
        """Regression guard: when ``_generate_map`` is called
        without an ``analyze()`` call (e.g. unit tests, programmatic
        callers), ``_original_query`` is None and we MUST preserve
        the LLM's ADM1 choice as-is.  The user-framing override only
        fires when we have a real user query to consult."""
        captured: dict[str, object] = {}

        def fake_adm1_map(adm1_code, **kwargs):
            captured["adm1_called"] = True
            captured["adm1_code"] = adm1_code
            return "adm1-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        # NOTE: not calling analyze(); _original_query stays None.
        parsed = self._make_parsed_with_map_data(
            focal_country="MEX",
            adm1_region="MEX016",
            receiving_countries=["USA"],
        )
        result = advisor._generate_map(parsed)

        assert result == "adm1-figure"
        assert captured.get("adm1_called") is True
        assert advisor._last_map_type == "adm1"

    def test_generate_map_keeps_adm1_when_query_names_a_state(
        self, monkeypatch,
    ):
        """End-to-end: structured extraction returned adm1=MEX016 AND
        the user explicitly said "Michoacan" -> dispatcher renders
        ADM1 (preserves existing behaviour for explicit subnational
        queries)."""
        captured: dict[str, object] = {}

        def fake_country_map(parsed, **kwargs):
            captured["country_called"] = True
            return "country-figure"

        def fake_adm1_map(adm1_code, **kwargs):
            captured["adm1_called"] = True
            captured["adm1_code"] = adm1_code
            return "adm1-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.plot_analysis_map",
            fake_country_map,
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(), auto_map=True,
        )
        advisor._original_query = (
            "Avocado production in Michoacán, Mexico"
        )
        parsed = self._make_parsed_with_map_data(
            focal_country="MEX",
            adm1_region="MEX016",
            receiving_countries=["USA"],
        )
        result = advisor._generate_map(parsed)

        assert result == "adm1-figure"
        assert captured.get("adm1_called") is True
        assert captured.get("adm1_code") == "MEX016"
        assert "country_called" not in captured
        assert advisor._last_map_type == "adm1"


class TestEvidenceCoverageNote:
    """Tests for the §7 Evidence Coverage flow:

    1. ``parse_analysis`` extracts the section into
       ``ParsedAnalysis.evidence_coverage_note``.
    2. ``_build_result`` lifts it onto
       ``AnalysisResult.evidence_coverage_note`` and lifts
       ``suggested_followup_queries`` from ``_last_web_map_signals``
       onto ``AnalysisResult.suggested_followup_queries``.
    3. ``AnalysisFormatter.format_full`` renders the §7 block.
    4. ``_build_result`` appends a "Suggested follow-up web searches"
       footer when queries are present.
    5. Backward compatibility: when both the §7 section and
       follow-up queries are absent, ``formatted`` omits both
       blocks entirely.
    """

    def _make_assistant_with_canned_response(self, response_text: str):
        from metacouplingllm.llm.client import LLMResponse, Message

        class CannedClient:
            def __init__(self, text):
                self._text = text

            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content=self._text)

        return MetacouplingAssistant(llm_client=CannedClient(response_text))

    def _minimal_framework_response(self, extra_sections: str = "") -> str:
        """Build a minimal but parsed framework response with §7 appended."""
        body = (
            "### 1. Coupling Classification\n\n"
            "This study involves telecoupling between Mexico and the US.\n\n"
            "### 2. Intracoupling Analysis\n\n"
            "#### 2.1 Systems Identification\n"
            "**Focal System**: Mexican avocado producers\n"
            "- **Human subsystem**: Smallholders and exporters.\n"
            "- **Natural subsystem**: Pine-oak forests, avocado orchards.\n"
            "- **Geographic scope**: Michoacan and Jalisco.\n\n"
            "### 6. Research Gaps and Suggestions\n\n"
            "- Need data on cartel involvement.\n"
            "- Need recent Jalisco expansion figures.\n"
            "- Need labor condition data.\n\n"
        )
        return body + extra_sections

    # --- Layer 1: parser extraction (covered separately in
    # tests/test_parser.py — this layer is the integration through
    # _build_result onto AnalysisResult). --------------------------------

    def test_coverage_note_lifted_onto_result(self):
        """parsed.evidence_coverage_note is mirrored onto
        AnalysisResult.evidence_coverage_note for ergonomic access."""
        response = self._minimal_framework_response(
            extra_sections=(
                "### 7. Evidence Coverage\n\n"
                "Strong evidence base: trade volumes from [T1:2].\n"
                "Limited evidence: cartel involvement not in any source.\n"
            )
        )
        advisor = self._make_assistant_with_canned_response(response)
        result = advisor.analyze("Mexican avocado trade")
        assert result.evidence_coverage_note
        assert "Strong evidence base" in result.evidence_coverage_note
        assert (
            result.parsed.evidence_coverage_note
            == result.evidence_coverage_note
        )

    def test_followup_queries_lifted_from_web_map_signals(self):
        """AnalysisResult.suggested_followup_queries mirrors
        web_map_signals['suggested_followup_queries'].

        Bypasses analyze() (which resets _last_web_map_signals at the
        start) and calls _build_result directly with the pre-populated
        instance state, mirroring the existing
        test_generate_map_merges_structured_web_map_signals pattern.
        """
        from metacouplingllm.llm.client import LLMResponse

        advisor = self._make_assistant_with_canned_response(
            self._minimal_framework_response()
        )
        advisor._turn = 1
        advisor._last_web_map_signals = {
            "focal_country": "MEX",
            "receiving_systems": [],
            "spillover_systems": [],
            "flows": [],
            "evidence_cards": [],
            "suggested_followup_queries": [
                "cartel control Mexican avocado supply chain 2024",
                "Jalisco avocado expansion post-USMCA",
            ],
        }
        response = LLMResponse(content=self._minimal_framework_response())
        result = advisor._build_result(response)
        assert result.suggested_followup_queries == [
            "cartel control Mexican avocado supply chain 2024",
            "Jalisco avocado expansion post-USMCA",
        ]
        # Same data should also be reachable via web_map_signals dict.
        assert (
            result.web_map_signals["suggested_followup_queries"]
            == result.suggested_followup_queries
        )

    def test_formatted_includes_evidence_coverage_block(self):
        """When §7 prose is present, ``formatted`` contains the
        rendered '7. Evidence Coverage' block."""
        response = self._minimal_framework_response(
            extra_sections=(
                "### 7. Evidence Coverage\n\n"
                "Strong evidence base for trade volumes; cartel data thin.\n"
            )
        )
        advisor = self._make_assistant_with_canned_response(response)
        result = advisor.analyze("Mexican avocado trade")
        assert "7. Evidence Coverage" in result.formatted
        assert "Strong evidence base for trade volumes" in result.formatted

    def test_formatted_includes_followup_queries_footer(self):
        """When suggested_followup_queries is non-empty, ``formatted``
        appends the bullet footer regardless of §7's presence.

        See note on test_followup_queries_lifted_from_web_map_signals
        about bypassing analyze()'s state reset.
        """
        from metacouplingllm.llm.client import LLMResponse

        advisor = self._make_assistant_with_canned_response(
            self._minimal_framework_response()
        )
        advisor._turn = 1
        advisor._last_web_map_signals = {
            "focal_country": "MEX",
            "receiving_systems": [],
            "spillover_systems": [],
            "flows": [],
            "evidence_cards": [],
            "suggested_followup_queries": [
                "cartel control Mexican avocado supply chain 2024",
            ],
        }
        response = LLMResponse(content=self._minimal_framework_response())
        result = advisor._build_result(response)
        assert "Suggested follow-up web searches" in result.formatted
        assert "cartel control" in result.formatted

    def test_formatted_omits_blocks_when_both_empty(self):
        """Backward compat: when neither §7 nor follow-ups exist,
        ``formatted`` contains neither block (preserves legacy shape
        for tests/fixtures generated before this PR)."""
        advisor = self._make_assistant_with_canned_response(
            self._minimal_framework_response()
        )
        # No web extraction has populated signals.
        assert advisor._last_web_map_signals is None
        result = advisor.analyze("Mexican avocado trade")
        assert "7. Evidence Coverage" not in result.formatted
        assert "Suggested follow-up web searches" not in result.formatted

    def test_new_fields_have_sensible_defaults(self):
        """AnalysisResult exposes the new fields with empty defaults
        when nothing populated them."""
        advisor = self._make_assistant_with_canned_response(
            self._minimal_framework_response()
        )
        result = advisor.analyze("Mexican avocado trade")
        assert result.evidence_coverage_note == ""
        assert result.suggested_followup_queries == []


class TestCountryMapConfiguration:
    """Tests for country-level auto-map configuration passthrough."""

    def test_generate_map_passes_adm0_shapefile(self, monkeypatch):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_analysis_map(parsed, **kwargs):
            captured["parsed"] = parsed
            captured["kwargs"] = kwargs
            return "fake-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.plot_analysis_map",
            fake_plot_analysis_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
            adm0_shapefile="country.gpkg",
        )
        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Brazil",
                    "geographic_scope": "Brazil",
                },
                "receiving": {
                    "name": "China",
                    "geographic_scope": "China",
                },
            },
        )

        result = advisor._generate_map(parsed)

        assert result == "fake-figure"
        assert captured["parsed"] is parsed
        assert captured["kwargs"]["adm0_shapefile"] == "country.gpkg"
        assert captured["kwargs"]["flows"] is None

    def test_generate_map_passes_resolved_country_flows(self, monkeypatch):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_analysis_map(parsed, **kwargs):
            captured["parsed"] = parsed
            captured["kwargs"] = kwargs
            return "fake-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.plot_analysis_map",
            fake_plot_analysis_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Brazil",
                    "geographic_scope": "Brazil",
                },
                "receiving": {
                    "name": "China",
                    "geographic_scope": "China",
                },
            },
            flows=[
                {
                    "category": "Matter",
                    "direction": "Brazil -> importing countries",
                    "description": "Soybean exports",
                },
            ],
        )

        result = advisor._generate_map(parsed)

        assert result == "fake-figure"
        assert captured["parsed"] is parsed
        assert captured["kwargs"]["flows"]
        directions = " ".join(
            flow["direction"] for flow in captured["kwargs"]["flows"]
        )
        assert "Brazil" in directions
        assert "China" in directions

    def test_generate_map_merges_structured_web_map_signals(self, monkeypatch):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_analysis_map(parsed, **kwargs):
            captured["parsed"] = parsed
            captured["kwargs"] = kwargs
            return "fake-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.plot_analysis_map",
            fake_plot_analysis_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        advisor._last_web_map_signals = {
            "focal_country": "BRA",
            "receiving_systems": [
                {
                    "country": "CHN",
                    "kind": "direct",
                    "confidence": 0.91,
                    "evidence": ["W1"],
                }
            ],
            "spillover_systems": [
                {
                    "country": "USA",
                    "kind": "proxy",
                    "confidence": 0.75,
                    "evidence": ["W2"],
                }
            ],
            "flows": [
                {
                    "category": "matter",
                    "direction": "Brazil → China",
                    "description": "Soybean exports",
                }
            ],
        }
        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Brazil",
                    "geographic_scope": "Brazil",
                },
                "receiving": {
                    "name": "Distant markets",
                    "geographic_scope": "Global importers",
                },
            },
        )

        result = advisor._generate_map(parsed)

        assert result == "fake-figure"
        # Spillover (USA) must NOT be in mentioned countries — only
        # focal + receiving. Spillover renders as grey (NA) so users
        # don't confuse competitors with actual trade partners.
        assert captured["kwargs"]["extra_mentioned_countries"] == {
            "BRA", "CHN",
        }
        assert "USA" not in captured["kwargs"]["extra_mentioned_countries"]
        assert captured["kwargs"]["flows"][0]["direction"] == "Brazil → China"

    def test_generate_map_resolves_long_country_scope_flows(self, monkeypatch):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_analysis_map(parsed, **kwargs):
            captured["parsed"] = parsed
            captured["kwargs"] = kwargs
            return "fake-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.plot_analysis_map",
            fake_plot_analysis_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Soybean-producing regions in Brazil",
                    "geographic_scope": (
                        "Brazil, especially major soybean-producing regions "
                        "such as Mato Grosso and other frontier or "
                        "consolidated production areas"
                    ),
                },
                "receiving": {
                    "name": "Major distant soybean-importing regions",
                    "geographic_scope": (
                        "Most plausibly China, given strong demand for "
                        "Brazilian soybean exports"
                    ),
                },
            },
            flows=[
                {
                    "category": "Matter",
                    "direction": "Brazil -> importing countries",
                    "description": "Soybean exports",
                },
            ],
        )

        result = advisor._generate_map(parsed)

        assert result == "fake-figure"
        assert captured["kwargs"]["flows"]
        directions = " ".join(
            flow["direction"] for flow in captured["kwargs"]["flows"]
        )
        assert "Brazil" in directions
        assert "China" in directions

    def test_generate_map_attempts_adm1_without_explicit_shapefile(
        self, monkeypatch
    ):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_focal_adm1_map(adm1_code, **kwargs):
            captured["adm1_code"] = adm1_code
            captured["kwargs"] = kwargs
            return "fake-adm1-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_plot_focal_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Michigan pork production system",
                    "geographic_scope": "Michigan, United States",
                },
                "receiving": {
                    "name": "China",
                    "geographic_scope": "China",
                },
            },
        )

        result = advisor._generate_map(parsed)

        assert result == "fake-adm1-figure"
        assert captured["adm1_code"] == "USA023"
        assert captured["kwargs"]["shapefile"] is None


class TestFormatMapNotice:
    """Test the _format_map_notice static method."""

    def test_adm1_notice(self):
        notice = MetacouplingAssistant._format_map_notice("adm1")
        assert "ADM1" in notice
        assert "subnational" in notice
        assert "result.map" in notice

    def test_country_notice(self):
        notice = MetacouplingAssistant._format_map_notice("country")
        assert "country-level" in notice
        assert "result.map" in notice


class TestMapTypeNoticeConsistency:
    """The user-facing map-type notice (``"ADM1"`` vs ``"country-level"``)
    must match what ``_generate_map`` actually rendered — not what the
    inputs *would* have selected.

    Before this fix, ``_build_result`` recomputed the type from
    ``parsed.map_data["adm1_region"]`` + ``_resolve_adm1_from_analysis``.
    When the renderer's ADM1 attempt silently fell through to a
    country-level map (e.g. ``plot_focal_adm1_map`` raised), the
    notice would still claim "ADM1" while the actual figure was
    country-level.

    The fix records the actually-rendered type in
    ``self._last_map_type`` from inside ``_generate_map`` (only after
    each successful render) and has ``_build_result`` read it instead
    of recomputing.
    """

    @staticmethod
    def _parsed_with_adm1():
        """A parsed analysis whose ``map_data`` carries an ADM1 region.

        Used to trigger the ADM1 branch in ``_generate_map``.
        """
        from metacouplingllm.llm.parser import CouplingSection, ParsedAnalysis
        parsed = ParsedAnalysis(
            coupling_classification="telecoupling",
            telecoupling=CouplingSection(systems=[
                {"role": "sending", "name": "Michigan, USA"},
                {"role": "receiving", "name": "China"},
            ]),
        )
        parsed.map_data = {
            "focal_country": "USA",
            "adm1_region": "USA023",
            "receiving_countries": ["CHN"],
            "spillover_countries": [],
            "flows": [],
        }
        return parsed

    def test_initial_last_map_type_is_none(self):
        """Fresh advisor: no map has been rendered yet."""
        from unittest.mock import MagicMock
        a = MetacouplingAssistant(llm_client=MagicMock())
        assert a._last_map_type is None

    def test_last_map_type_stamped_adm1_on_successful_adm1_render(self):
        """When ``plot_focal_adm1_map`` returns a figure,
        ``_last_map_type`` is ``"adm1"``."""
        from unittest.mock import MagicMock, patch
        a = MetacouplingAssistant(llm_client=MagicMock(), auto_map=True)
        parsed = self._parsed_with_adm1()
        with patch(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map"
        ) as adm1_mock:
            adm1_mock.return_value = "FAKE_FIG"
            a._generate_map(parsed)
        assert a._last_map_type == "adm1"

    def test_last_map_type_stamped_country_on_adm1_fallback(self):
        """REGRESSION: when ``plot_focal_adm1_map`` raises, the
        renderer falls through to country-level — the stamp must
        reflect what was actually drawn, not what was tried."""
        from unittest.mock import MagicMock, patch
        a = MetacouplingAssistant(llm_client=MagicMock(), auto_map=True)
        parsed = self._parsed_with_adm1()
        with patch(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map"
        ) as adm1_mock, patch(
            "metacouplingllm.visualization.worldmap.plot_analysis_map"
        ) as country_mock:
            adm1_mock.side_effect = RuntimeError(
                "shapefile missing the region"
            )
            country_mock.return_value = "FAKE_FIG"
            a._generate_map(parsed)
        assert a._last_map_type == "country", (
            "Notice must reflect ACTUAL render. ADM1 attempt failed "
            "and renderer fell through to country-level; the stamp "
            "must say 'country' even though map_data['adm1_region'] "
            "is set."
        )

    def test_last_map_type_stamped_country_when_no_adm1(self):
        """No ADM1 region in map_data — straightforward country
        path."""
        from unittest.mock import MagicMock, patch
        a = MetacouplingAssistant(llm_client=MagicMock(), auto_map=True)
        parsed = self._parsed_with_adm1()
        parsed.map_data["adm1_region"] = None
        with patch(
            "metacouplingllm.visualization.worldmap.plot_analysis_map"
        ) as country_mock:
            country_mock.return_value = "FAKE_FIG"
            a._generate_map(parsed)
        assert a._last_map_type == "country"

    def test_last_map_type_reset_to_none_at_start_of_generate_map(self):
        """``_generate_map`` resets the stamp on each call so a prior
        run's state can't leak into the current one."""
        from unittest.mock import MagicMock, patch
        a = MetacouplingAssistant(llm_client=MagicMock(), auto_map=True)
        # Seed a stale value.
        a._last_map_type = "adm1"

        from metacouplingllm.llm.parser import ParsedAnalysis
        # ParsedAnalysis with no map_data and no resolvable focal —
        # _generate_map returns None without rendering anything.
        empty_parsed = ParsedAnalysis()
        with patch(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map"
        ), patch(
            "metacouplingllm.visualization.worldmap.plot_analysis_map"
        ):
            a._generate_map(empty_parsed)
        # Either renderer wasn't called (returned None) or fell through
        # to a no-focal-found path; in both cases the stamp must be
        # cleared, not the stale "adm1".
        assert a._last_map_type != "adm1", (
            "Stale stamp from a prior call must not survive a new "
            "_generate_map call that didn't successfully render."
        )

    def test_country_notice_used_when_adm1_falls_back(self):
        """End-to-end (through ``_build_result``-equivalent code path):
        when the ADM1 attempt falls through, the appended notice
        reads "country-level", not "ADM1"."""
        from unittest.mock import MagicMock, patch
        a = MetacouplingAssistant(llm_client=MagicMock(), auto_map=True)
        parsed = self._parsed_with_adm1()
        with patch(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map"
        ) as adm1_mock, patch(
            "metacouplingllm.visualization.worldmap.plot_analysis_map"
        ) as country_mock:
            adm1_mock.side_effect = RuntimeError("ADM1 path failed")
            country_mock.return_value = "FAKE_FIG"
            a._generate_map(parsed)
        # Build the notice the way _build_result would now:
        map_type = a._last_map_type or "country"
        notice = MetacouplingAssistant._format_map_notice(map_type)
        assert "country-level" in notice
        assert "ADM1" not in notice


class TestResolveFlowsForMap:
    """Test the _resolve_flows_for_map static method."""

    def test_resolves_specific_country_names(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Brazil → China",
                    "description": "Soybeans",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        assert len(result) >= 1
        assert "→" in result[0]["direction"]
        assert "China" in result[0]["direction"] or "CHN" in result[0]["direction"]

    def test_resolves_adm1_region_to_country(self):
        """Michigan should resolve to USA via ADM1 database."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Michigan Pork System", "geographic_scope": "Michigan"},
                "receiving": {"name": "China"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Michigan (sending) → China",
                    "description": "Pork exported",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        assert len(result) >= 1
        # Source should be USA (from Michigan ADM1), target should be China
        direction = result[0]["direction"]
        assert "United States" in direction or "USA" in direction
        assert "China" in direction

    def test_resolves_generic_receiving_reference(self):
        """'Receiving regions' should resolve to receiving system countries."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Michigan Pork System"},
                "receiving": {
                    "name": "International Import Markets",
                    "geographic_scope": "China, Japan, Mexico",
                },
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Michigan (sending) → Receiving regions",
                    "description": "Pork exported",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        # Should create arrows to CHN, JPN, MEX
        assert len(result) >= 2
        directions = " ".join(f["direction"] for f in result)
        assert "China" in directions
        assert "Japan" in directions or "Mexico" in directions

    def test_skips_internal_flows(self):
        """Flows 'within Michigan' should be skipped."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Michigan"}},
            flows=[
                {
                    "category": "energy",
                    "direction": "Mostly within Michigan and embedded in exports",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        assert len(result) == 0

    def test_bidirectional_between_pattern(self):
        """'Bidirectional between Michigan and other regions' should resolve."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Michigan"},
                "receiving": {"name": "China", "geographic_scope": "China"},
            },
            flows=[
                {
                    "category": "information",
                    "direction": "Bidirectional between Michigan and receiving systems",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        assert len(result) >= 1
        assert "Bidirectional" in result[0]["direction"]
        assert "↔" in result[0]["direction"]

    def test_skips_speculative_example_countries_for_generic_roles(self):
        """Generic role references should not become arrows from examples."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Michigan, United States",
                    "geographic_scope": "Michigan",
                },
                "receiving": {
                    "name": "Distant importing markets",
                    "geographic_scope": (
                        "Likely international destination markets rather "
                        "than one confirmed place"
                    ),
                },
                "spillover": {
                    "name": "Adjacent and competing regions",
                    "geographic_scope": (
                        "A stronger analysis would specify whether the main "
                        "receiving systems are, for example, Mexico, China, "
                        "Japan, or another market."
                    ),
                },
            },
            flows=[
                {
                    "category": "information",
                    "direction": (
                        "Bidirectional between Michigan and "
                        "receiving/spillover systems"
                    ),
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        assert result == []

    def test_resolves_softened_receiving_market_list_for_outgoing_flow(self):
        """Likely receiving-market lists should restore proxy export arrows."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Michigan, United States"},
                "receiving": {
                    "name": "Distant export markets for Michigan/U.S. pork",
                    "geographic_scope": (
                        "Likely distant foreign markets connected to U.S. "
                        "pork exports, such as Mexico, China, Japan, "
                        "South Korea, and Canada."
                    ),
                },
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Michigan -> distant receiving markets",
                    "description": "Pork exports",
                },
            ],
        )

        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        directions = " ".join(flow["direction"] for flow in result)
        assert "China" in directions
        assert "Mexico" in directions
        assert "Japan" in directions

    def test_resolves_generic_receiving_source_back_to_focal_country(self):
        """Incoming capital flows from receiving markets should render."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Michigan, United States"},
                "receiving": {
                    "name": "Distant export markets for Michigan/U.S. pork",
                    "geographic_scope": (
                        "Likely distant foreign markets connected to U.S. "
                        "pork exports, such as Mexico, China, Japan, "
                        "South Korea, and Canada."
                    ),
                },
            },
            flows=[
                {
                    "category": "capital",
                    "direction": "Receiving markets -> Michigan",
                    "description": "Export revenue",
                },
            ],
        )

        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        directions = " ".join(flow["direction"] for flow in result)
        assert "China" in directions or "Mexico" in directions
        assert "United States" in directions

    def test_resolves_importing_country_synonyms_for_generic_flows(self):
        """Importing-country wording should resolve via the receiving system."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil", "geographic_scope": "Brazil"},
                "receiving": {
                    "name": (
                        "Major distant soybean-importing countries, "
                        "especially China"
                    ),
                    "geographic_scope": (
                        "China is the most likely primary receiving system"
                    ),
                },
            },
            flows=[
                {
                    "category": "capital",
                    "direction": "Importing countries -> Brazil",
                    "description": "Payments for soybean imports",
                },
                {
                    "category": "information",
                    "direction": (
                        "Bidirectional between Brazil and importing markets"
                    ),
                    "description": "Market information and standards",
                },
                {
                    "category": "energy",
                    "direction": (
                        "Embedded within soybean trade; effectively "
                        "Brazil -> importing countries"
                    ),
                    "description": "Embodied energy in soybeans",
                },
            ],
        )

        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        directions = " | ".join(flow["direction"] for flow in result)

        assert "China" in directions
        assert "Brazil" in directions
        assert any(flow["category"] == "capital" for flow in result)
        assert any(flow["category"] == "information" for flow in result)
        assert any(flow["category"] == "energy" for flow in result)

    def test_resolves_explicit_multi_country_direction_list(self):
        """Comma-separated country lists in the direction should still resolve."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Michigan"}},
            flows=[
                {
                    "category": "matter",
                    "direction": "Michigan → China, Japan, Mexico",
                    "description": "Pork exported",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        directions = " ".join(flow["direction"] for flow in result)
        assert "China" in directions
        assert "Japan" in directions
        assert "Mexico" in directions

    def test_deduplicates_same_pair(self):
        """Multiple flows to same (category, src, tgt) should be deduplicated."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
            },
            flows=[
                {"category": "matter", "direction": "Brazil → China"},
                {"category": "matter", "direction": "Brazil → China"},
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        matter_to_china = [
            f for f in result
            if f["category"] == "matter" and "China" in f["direction"]
        ]
        assert len(matter_to_china) == 1

    def test_resolves_domestic_adm1_neighbor_flows(self):
        """Explicit nearby-state flows should resolve to ADM1 arrow endpoints."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Michigan, United States"}},
            flows=[
                {
                    "category": "matter",
                    "direction": "Michigan -> Indiana / Ohio / Wisconsin",
                    "description": "Regional pork and input flows",
                },
            ],
        )

        result, _ = MetacouplingAssistant._resolve_flows_for_adm1_map(
            parsed,
            "USA023",
            "USA",
        )
        domestic = [flow for flow in result if flow.get("target_adm1")]

        assert any(flow["target_adm1"] == "USA015" for flow in domestic)
        assert any(flow["target_adm1"] == "USA036" for flow in domestic)
        assert any(flow["target_adm1"] == "USA050" for flow in domestic)
        assert all(flow["source_adm1"] == "USA023" for flow in domestic)

    def test_resolves_bidirectional_adjacent_state_flows(self):
        """Generic adjacent-state language should fan out to domestic neighbors."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Michigan, United States"}},
            flows=[
                {
                    "category": "people",
                    "direction": "Michigan <-> adjacent states",
                    "description": "Regional labor and service exchange",
                },
            ],
        )

        result, _ = MetacouplingAssistant._resolve_flows_for_adm1_map(
            parsed,
            "USA023",
            "USA",
        )
        domestic = [flow for flow in result if flow.get("target_adm1")]

        assert domestic
        assert all(flow.get("is_bidirectional") for flow in domestic)
        assert {flow["target_adm1"] for flow in domestic} == {
            "USA015",
            "USA036",
            "USA050",
        }


class TestResolveFlowsSystemsFallback:
    """Test that generic flow directions fall back to Systems countries."""

    def test_generic_target_falls_back_to_receiving_system(self):
        """'Brazil → importing countries' should resolve via Systems."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {
                    "name": "Major soybean-importing countries, especially China",
                },
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Brazil \u2192 importing countries",
                    "description": "Soybeans exported",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        # Should resolve "importing countries" to China via receiving system
        assert len(result) >= 1
        directions = " ".join(f["direction"] for f in result)
        assert "China" in directions

    def test_reverse_flow_uses_receiving_as_source(self):
        """'Importing countries → Brazil' → China → Brazil (capital flow)."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {
                    "name": "China and Vietnam",
                },
            },
            flows=[
                {
                    "category": "capital",
                    "direction": "Importing countries \u2192 Brazil",
                    "description": "Payments for soybean purchases",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        # Should produce arrows FROM receiving countries TO Brazil
        assert len(result) >= 1
        for f in result:
            parts = f["direction"].split("\u2192")
            if len(parts) == 2:
                src = parts[0].strip()
                tgt = parts[1].strip()
                assert src != tgt  # no self-loops
                assert tgt == "Brazil"  # target is focal country
                assert src in ("China", "Vietnam", "Viet Nam")  # source is receiver

    def test_generic_receiving_does_not_use_spillover(self):
        """When receiving is generic, spillover countries must NOT be used."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "United States corn-producing regions"},
                "receiving": {"name": "Major foreign importing countries"},
                "spillover": {
                    "name": "Competing exporters such as Brazil and Argentina",
                },
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "USA \u2192 foreign importing countries",
                },
                {
                    "category": "capital",
                    "direction": "foreign importing countries \u2192 USA",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        # No arrows should be produced — receiving is generic, and
        # spillover countries must NOT be used as trade partners.
        for f in result:
            assert "Brazil" not in f["direction"]
            assert "Argentina" not in f["direction"]

    def test_no_fallback_when_target_resolves(self):
        """When direction has specific countries, no fallback needed."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China, Japan, and South Korea"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Brazil \u2192 China",
                    "description": "Soybeans",
                },
            ],
        )
        result, _ = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        assert len(result) == 1
        assert "China" in result[0]["direction"]
        # Should NOT include Japan/South Korea since direction was specific
        assert "Japan" not in result[0]["direction"]


class TestStructuredMapData:
    """Test the two-call structured map data pipeline."""

    def test_generate_map_uses_structured_data(self):
        """When map_data is present, _generate_map uses it directly."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Brazil"}},
            flows=[{"category": "matter", "direction": "Brazil \u2192 China"}],
            map_data={
                "focal_country": "BRA",
                "adm1_region": None,
                "receiving_countries": ["CHN"],
                "spillover_countries": ["USA", "ARG"],
                "flows": [
                    {
                        "category": "matter",
                        "source": "BRA",
                        "target": "CHN",
                        "direction": "Brazil \u2192 China",
                        "bidirectional": False,
                    },
                    {
                        "category": "capital",
                        "source": "CHN",
                        "target": "BRA",
                        "direction": "China \u2192 Brazil",
                        "bidirectional": False,
                    },
                ],
            },
        )
        # Verify the structured data is well-formed
        md = parsed.map_data
        assert md["focal_country"] == "BRA"
        assert "CHN" in md["receiving_countries"]
        assert len(md["flows"]) == 2

    def test_map_data_defaults_to_none(self):
        """ParsedAnalysis.map_data defaults to None for backward compat."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = ParsedAnalysis()
        assert parsed.map_data is None

    def test_web_structured_extraction_auto_enabled(self):
        """web_structured_extraction auto-enables when web_search + auto_map."""
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            web_search=True,
            auto_map=True,
            # web_structured_extraction not set explicitly
        )
        assert advisor._web_structured_extraction is True

    def test_web_structured_extraction_stays_false_without_map(self):
        """Without auto_map, web_structured_extraction stays False."""
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            web_search=True,
            auto_map=False,
        )
        assert advisor._web_structured_extraction is False


    def test_flow_source_validation_drops_spillover_flows(self):
        """Flows from spillover countries should be dropped from the map.

        E.g., if USA exports corn and Brazil is a spillover competitor,
        a flow "Brazil → Mexico" should NOT appear — only flows from
        the focal country (USA) or receiving countries should be shown.
        """
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={
                "sending": {"name": "United States"},
                "receiving": "Mexico, Japan, South Korea",
                "spillover": "Brazil, Argentina",
            },
            flows=[
                {"category": "matter", "direction": "United States → Mexico"},
            ],
            map_data={
                "focal_country": "USA",
                "adm1_region": None,
                "receiving_countries": ["MEX", "JPN", "KOR"],
                "spillover_countries": ["BRA", "ARG"],
                "flows": [
                    {
                        "category": "matter",
                        "direction": "United States → Mexico",
                    },
                    {
                        "category": "matter",
                        "direction": "United States → Japan",
                    },
                    # These should be DROPPED — source is spillover
                    {
                        "category": "matter",
                        "direction": "Brazil → Mexico",
                    },
                    {
                        "category": "matter",
                        "direction": "Argentina → Mexico",
                    },
                ],
            },
        )

        # Simulate what _generate_map does for endpoint validation
        from metacouplingllm.core import _FLOW_ARROW_RE
        from metacouplingllm.knowledge.countries import resolve_country_code

        focal_code = "USA"
        receiving = ["MEX", "JPN", "KOR"]
        spillover = ["BRA", "ARG"]
        mentioned = {focal_code} | set(receiving) | set(spillover)
        valid_sources = {focal_code} | set(receiving)

        map_flows = [
            {"category": str(f.get("category", "")),
             "direction": str(f["direction"])}
            for f in parsed.map_data["flows"]
            if isinstance(f, dict) and f.get("direction")
        ]

        def _flow_endpoints_valid(f):
            d = f.get("direction", "")
            parts = _FLOW_ARROW_RE.split(d)
            if len(parts) < 2:
                return True
            tgt_code = resolve_country_code(parts[-1].strip().rstrip(")"))
            if tgt_code and tgt_code not in mentioned:
                return False
            src_code = resolve_country_code(parts[0].strip().lstrip("("))
            if src_code and src_code not in valid_sources:
                return False
            return True

        filtered = [f for f in map_flows if _flow_endpoints_valid(f)]

        # USA→MEX and USA→JPN should survive
        assert len(filtered) == 2
        directions = [f["direction"] for f in filtered]
        assert "United States → Mexico" in directions
        assert "United States → Japan" in directions
        # BRA→MEX and ARG→MEX should be dropped
        assert "Brazil → Mexico" not in directions
        assert "Argentina → Mexico" not in directions

    def test_adm1_reference_includes_mato_grosso(self):
        """The ADM1 reference block should include BRA011=Mato Grosso."""
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_codes_for_country,
            get_adm1_info,
        )
        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = make_parsed_analysis(
            coupling_classification="Brazil soybean exports to China",
            systems={
                "sending": {
                    "name": "Mato Grosso, Brazil",
                    "geographic_scope": "Brazil",
                },
                "receiving": {
                    "name": "China",
                    "geographic_scope": "China",
                },
            },
        )

        ref = advisor._build_adm1_reference_for_prompt(
            parsed, get_adm1_codes_for_country, get_adm1_info,
        )
        assert "BRA011=Mato Grosso" in ref
        assert "VALID ADM1 CODES" in ref

    def test_adm1_reference_only_mentioned_countries(self):
        """Only countries actually mentioned in the analysis should appear."""
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_codes_for_country,
            get_adm1_info,
        )
        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = make_parsed_analysis(
            coupling_classification="USA corn exports",
            systems={
                "sending": {
                    "name": "USA",
                    "geographic_scope": "United States",
                },
                "receiving": {
                    "name": "Mexico",
                    "geographic_scope": "Mexico",
                },
            },
        )

        ref = advisor._build_adm1_reference_for_prompt(
            parsed, get_adm1_codes_for_country, get_adm1_info,
        )
        # USA should appear, Brazil should NOT
        assert "United States" in ref or "USA" in ref
        assert "Brazil" not in ref

    def test_adm1_reference_empty_when_no_countries(self):
        """No mentioned countries → empty reference block."""
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_codes_for_country,
            get_adm1_info,
        )
        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = ParsedAnalysis(coupling_classification="A generic topic")

        ref = advisor._build_adm1_reference_for_prompt(
            parsed, get_adm1_codes_for_country, get_adm1_info,
        )
        assert ref == ""

    def test_invalid_adm1_from_llm_falls_back_to_regex_resolver(self):
        """When the LLM returns an invalid ADM1 code, fall back to the
        regex resolver which correctly identifies Mato Grosso as BRA011.
        """
        import json

        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        # Mock LLM returns an invalid ADM1 code (BRA014 does not exist)
        fake_response = json.dumps({
            "focal_country": "BRA",
            "adm1_region": "BRA014",  # INVALID — should trigger fallback
            "receiving_countries": ["CHN"],
            "spillover_countries": [],
            "flows": [
                {
                    "category": "matter",
                    "source": "BRA",
                    "target": "CHN",
                    "direction": "Brazil \u2192 China",
                    "bidirectional": False,
                },
            ],
        })

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content=fake_response)

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        # Parsed analysis mentions "Mato Grosso" so the regex
        # resolver should find BRA011.
        parsed = make_parsed_analysis(
            coupling_classification=(
                "Telecoupling between Mato Grosso, Brazil and China"
            ),
            systems={
                "sending": {
                    "name": "Mato Grosso, Brazil",
                    "geographic_scope": "Mato Grosso, Brazil",
                },
                "receiving": {
                    "name": "China",
                    "geographic_scope": "China",
                },
            },
            flows=[{"category": "matter", "direction": "Brazil \u2192 China"}],
        )

        result = advisor._extract_map_data_from_analysis(parsed)
        assert result is not None
        # The invalid BRA014 should have been replaced with BRA011
        assert result["adm1_region"] == "BRA011"

    def test_valid_adm1_from_llm_is_accepted(self):
        """When the LLM returns a valid ADM1 code, it should be kept."""
        import json

        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        fake_response = json.dumps({
            "focal_country": "BRA",
            "adm1_region": "BRA011",  # VALID — Mato Grosso
            "receiving_countries": ["CHN"],
            "spillover_countries": [],
            "flows": [],
        })

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content=fake_response)

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso soybean exports",
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": "Brazil",
                },
            },
        )

        result = advisor._extract_map_data_from_analysis(parsed)
        assert result is not None
        assert result["adm1_region"] == "BRA011"

    def test_null_adm1_from_llm_uses_regex_fallback(self):
        """When the LLM returns null for adm1_region but the analysis
        clearly mentions a subnational region, the regex resolver
        should be used as a fallback.
        """
        import json

        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        fake_response = json.dumps({
            "focal_country": "BRA",
            "adm1_region": None,  # LLM missed it
            "receiving_countries": ["CHN"],
            "spillover_countries": [],
            "flows": [],
        })

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content=fake_response)

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = make_parsed_analysis(
            coupling_classification=(
                "Telecoupling between Mato Grosso, Brazil and China"
            ),
            systems={
                "sending": {
                    "name": "Mato Grosso, Brazil",
                    "geographic_scope": "Mato Grosso, Brazil",
                },
            },
        )

        result = advisor._extract_map_data_from_analysis(parsed)
        assert result is not None
        # The regex resolver should have found BRA011
        assert result["adm1_region"] == "BRA011"

    def test_extraction_prompt_contains_adm1_reference(self):
        """The prompt sent to the LLM should include valid ADM1 codes."""
        import json

        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured_messages = []

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                captured_messages.extend(messages)
                return LLMResponse(content=json.dumps({
                    "focal_country": "BRA",
                    "adm1_region": None,
                    "receiving_countries": [],
                    "spillover_countries": [],
                    "flows": [],
                }))

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso, Brazil soybean exports",
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": "Brazil",
                },
            },
        )

        advisor._extract_map_data_from_analysis(parsed)

        # Find the user message
        user_msgs = [m for m in captured_messages if m.role == "user"]
        assert len(user_msgs) >= 1
        user_text = user_msgs[0].content

        # The prompt should include the ADM1 reference list
        assert "VALID ADM1 CODES" in user_text
        assert "BRA011=Mato Grosso" in user_text

    def test_structured_web_receiving_codes_excludes_spillover(self):
        """_structured_web_receiving_codes() returns only focal + receiving."""
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        advisor._last_web_map_signals = {
            "focal_country": "BRA",
            "receiving_systems": [
                {"country": "CHN", "kind": "direct"},
                {"country": "JPN", "kind": "direct"},
            ],
            "spillover_systems": [
                {"country": "USA", "kind": "proxy"},
                {"country": "ARG", "kind": "proxy"},
            ],
        }

        receiving = advisor._structured_web_receiving_codes()
        assert receiving == {"BRA", "CHN", "JPN"}
        # Spillover should NOT appear
        assert "USA" not in receiving
        assert "ARG" not in receiving

    def test_structured_web_spillover_codes_returns_only_spillover(self):
        """_structured_web_spillover_codes() returns ONLY spillover systems."""
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        advisor._last_web_map_signals = {
            "focal_country": "BRA",
            "receiving_systems": [
                {"country": "CHN", "kind": "direct"},
            ],
            "spillover_systems": [
                {"country": "USA", "kind": "proxy"},
                {"country": "ARG", "kind": "proxy"},
            ],
        }

        spillover = advisor._structured_web_spillover_codes()
        assert spillover == {"USA", "ARG"}
        # Focal and receiving should NOT appear
        assert "BRA" not in spillover
        assert "CHN" not in spillover

    def test_structured_web_country_codes_still_returns_all(self):
        """Backward-compat: _structured_web_country_codes() returns the union."""
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        advisor._last_web_map_signals = {
            "focal_country": "BRA",
            "receiving_systems": [{"country": "CHN"}],
            "spillover_systems": [{"country": "USA"}],
        }

        all_codes = advisor._structured_web_country_codes()
        assert all_codes == {"BRA", "CHN", "USA"}

    def test_generate_map_excludes_spillover_from_mentioned(self, monkeypatch):
        """_generate_map() must NOT pass spillover countries to the renderer.

        This is the user-facing fix: in the Mato Grosso → China analysis,
        the USA (classified as spillover) should render as grey, not blue.
        """
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_focal_adm1_map(focal_adm1, **kwargs):
            captured["focal_adm1"] = focal_adm1
            captured["kwargs"] = kwargs
            return "fake-adm1-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_plot_focal_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso soybean exports",
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": "Brazil",
                },
            },
            map_data={
                "focal_country": "BRA",
                "adm1_region": "BRA011",
                "receiving_countries": ["CHN"],
                "spillover_countries": ["USA", "ARG"],
                "flows": [
                    {
                        "category": "matter",
                        "direction": "Brazil \u2192 China",
                    },
                ],
            },
        )

        result = advisor._generate_map(parsed)
        assert result == "fake-adm1-figure"
        assert captured["focal_adm1"] == "BRA011"

        mentioned = captured["kwargs"]["mentioned_countries"]
        # BRA (focal) and CHN (receiving) should be included
        assert "BRA" in mentioned
        assert "CHN" in mentioned
        # USA and ARG (spillover) should NOT be included
        assert "USA" not in mentioned
        assert "ARG" not in mentioned

    def test_generate_map_passes_mentioned_adm1_codes(self, monkeypatch):
        """_generate_map() passes mentioned_adm1_codes from map_data."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_focal_adm1_map(focal_adm1, **kwargs):
            captured["focal_adm1"] = focal_adm1
            captured["kwargs"] = kwargs
            return "fake-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_plot_focal_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso soybean exports",
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": "Brazil",
                },
            },
            map_data={
                "focal_country": "BRA",
                "adm1_region": "BRA011",
                "mentioned_adm1_regions": [
                    "BRA004", "BRA009", "BRA012", "BRA018",
                    "BRA026", "BRA031", "BOL008",
                ],
                "receiving_countries": ["CHN"],
                "spillover_countries": ["USA"],
                "flows": [],
            },
        )

        result = advisor._generate_map(parsed)
        assert result == "fake-figure"

        passed_adm1 = captured["kwargs"].get("mentioned_adm1_codes")
        assert passed_adm1 is not None
        assert "BRA004" in passed_adm1
        assert "BRA009" in passed_adm1
        assert "BOL008" in passed_adm1
        # Should not contain the focal or random non-mentioned regions
        assert "BRA013" not in passed_adm1

    def test_generate_map_falls_back_to_regex_adm1_extraction(self, monkeypatch):
        """When map_data.mentioned_adm1_regions is empty, regex fallback runs."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_focal_adm1_map(focal_adm1, **kwargs):
            captured["focal_adm1"] = focal_adm1
            captured["kwargs"] = kwargs
            return "fake-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_plot_focal_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        # Regions appear in SUBSTANTIVE locations (flow descriptions,
        # effects) — not just in systems.geographic_scope, which is
        # ignored by the narrowed fallback.
        parsed = make_parsed_analysis(
            coupling_classification=(
                "Telecoupling: Mato Grosso, Brazil soybean exports to China."
            ),
            systems={
                "sending": {
                    "name": "Mato Grosso, Brazil",
                    "geographic_scope": "Mato Grosso, Brazil",
                },
                "receiving": {
                    "name": "China",
                    "geographic_scope": "China",
                },
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Mato Grosso \u2192 China",
                    "description": (
                        "Soybeans shipped via Rondonia transport corridor"
                    ),
                },
            ],
            map_data={
                "focal_country": "BRA",
                "adm1_region": "BRA011",
                "mentioned_adm1_regions": [],  # empty — triggers fallback
                "receiving_countries": ["CHN"],
                "spillover_countries": [],
                "flows": [],
            },
        )

        result = advisor._generate_map(parsed)
        assert result == "fake-figure"

        passed_adm1 = captured["kwargs"].get("mentioned_adm1_codes")
        # The narrowed fallback should pick up Rondônia from the flow
        # description (substantive evidence).
        assert passed_adm1 is not None
        found_codes = set(passed_adm1)
        assert "BRA026" in found_codes, (
            f"Expected BRA026 (Rondonia) from flow description, "
            f"got {found_codes}"
        )

    def test_extract_mentioned_adm1_from_text_finds_multiple(self):
        """Multi-match regex extractor finds all mentioned ADM1 regions."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            coupling_classification=(
                "Mato Grosso, Brazil soybean exports to China."
            ),
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": (
                        "Mato Grosso, Brazil, connected to Para, "
                        "Rondonia, and Goias through transport corridors"
                    ),
                },
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Mato Grosso \u2192 China",
                    "description": "Soybean exports via Mato Grosso do Sul corridor",
                },
            ],
        )

        codes = MetacouplingAssistant._extract_mentioned_adm1_from_text(parsed)
        # Should find at least some of the mentioned Brazilian states
        # (exact codes depend on the pericoupling database)
        assert len(codes) >= 2
        # All returned codes should be in Brazil (relevance guard)
        from metacouplingllm.knowledge.adm1_pericoupling import get_adm1_country
        for code in codes:
            assert get_adm1_country(code) == "BRA", (
                f"Non-BRA code {code} leaked through relevance guard"
            )

    def test_extraction_prompt_contains_mentioned_adm1_field(self):
        """The LLM prompt should ask for mentioned_adm1_regions."""
        import json

        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured_messages = []

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                captured_messages.extend(messages)
                return LLMResponse(content=json.dumps({
                    "focal_country": "BRA",
                    "adm1_region": "BRA011",
                    "mentioned_adm1_regions": [],
                    "receiving_countries": [],
                    "spillover_countries": [],
                    "flows": [],
                }))

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso soybean exports",
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": "Brazil",
                },
            },
        )

        advisor._extract_map_data_from_analysis(parsed)

        user_msgs = [m for m in captured_messages if m.role == "user"]
        assert len(user_msgs) >= 1
        user_text = user_msgs[0].content

        # The prompt should mention the new field
        assert "mentioned_adm1_regions" in user_text

    def test_invalid_mentioned_adm1_codes_filtered_out(self):
        """Invalid ADM1 codes from LLM response are filtered out."""
        import json

        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        fake_response = json.dumps({
            "focal_country": "BRA",
            "adm1_region": "BRA011",
            "mentioned_adm1_regions": [
                "BRA004",    # valid
                "BRA009",    # valid
                "BRA999",    # invalid — doesn't exist
                "ZZZ000",    # invalid — doesn't exist
            ],
            "receiving_countries": ["CHN"],
            "spillover_countries": [],
            "flows": [],
        })

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content=fake_response)

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso analysis",
            systems={"sending": {"name": "Mato Grosso"}},
        )

        result = advisor._extract_map_data_from_analysis(parsed)
        assert result is not None
        mentioned_adm1 = result["mentioned_adm1_regions"]
        # Invalid codes dropped, valid ones kept
        assert "BRA004" in mentioned_adm1
        assert "BRA009" in mentioned_adm1
        assert "BRA999" not in mentioned_adm1
        assert "ZZZ000" not in mentioned_adm1

    def test_extraction_prompt_requires_substantive_evidence(self):
        """Rule 11 in the extraction prompt must require substantive
        evidence of interaction, not just a name mention in a list.
        """
        import json

        from metacouplingllm.llm.client import LLMResponse
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured_messages = []

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                captured_messages.extend(messages)
                return LLMResponse(content=json.dumps({
                    "focal_country": "BRA",
                    "adm1_region": "BRA011",
                    "mentioned_adm1_regions": [],
                    "receiving_countries": [],
                    "spillover_countries": [],
                    "flows": [],
                }))

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso analysis",
            systems={"sending": {"name": "Mato Grosso"}},
        )

        advisor._extract_map_data_from_analysis(parsed)

        user_text = captured_messages[-1].content
        assert "SUBSTANTIVE EVIDENCE" in user_text
        assert "reference list" in user_text or "reference lookups" in user_text

    def test_extract_mentioned_adm1_skips_list_style_scopes(self):
        """_extract_mentioned_adm1_from_text must ignore regions that
        only appear in systems[*].geographic_scope or systems[*].name
        (which are typically list-style enumerations from the DB hint).
        """
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso, Brazil soybean exports.",
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": "Mato Grosso, Brazil",
                },
                "spillover": {
                    # Typical list-style echo of DB neighbors — should
                    # NOT be scanned by the fallback.
                    "name": (
                        "Adjacent regions: Amazonas, Goias, Para, "
                        "Rondonia, Tocantins, Mato Grosso do Sul, "
                        "Santa Cruz Bolivia"
                    ),
                    "geographic_scope": (
                        "Amazonas, Goias, Para, Rondonia, Tocantins, "
                        "Mato Grosso do Sul, Santa Cruz (Bolivia)"
                    ),
                },
            },
            # No flows / causes / effects that substantively discuss
            # any of those regions.
        )

        codes = MetacouplingAssistant._extract_mentioned_adm1_from_text(parsed)
        # None of the echoed names should be picked up.
        assert "BRA004" not in codes  # Amazonas
        assert "BRA009" not in codes  # Goias
        assert "BRA018" not in codes  # Para
        assert "BRA026" not in codes  # Rondonia
        assert "BRA031" not in codes  # Tocantins
        assert "BRA012" not in codes  # Mato Grosso do Sul
        assert "BOL008" not in codes  # Santa Cruz Bolivia

    def test_extract_mentioned_adm1_picks_up_flow_references(self):
        """When a region is mentioned in a flow direction or
        description, it IS substantive evidence and should be picked
        up by the regex fallback.
        """
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso, Brazil soybean exports.",
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": "Mato Grosso, Brazil",
                },
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Mato Grosso, Brazil \u2192 Rondonia",
                    "description": "Soybeans shipped via Rondonia transport corridor",
                },
            ],
        )

        codes = MetacouplingAssistant._extract_mentioned_adm1_from_text(parsed)
        # Rondonia appears in a flow — substantive, should be picked up.
        assert "BRA026" in codes, (
            f"Expected BRA026 (Rondonia) from flow mention, got {codes}"
        )

    def test_extract_mentioned_adm1_picks_up_cause_effect_references(self):
        """When a region is named in a specific cause or effect
        bullet, it counts as substantive evidence.
        """
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            coupling_classification="Mato Grosso soy analysis.",
            systems={
                "sending": {
                    "name": "Mato Grosso",
                    "geographic_scope": "Mato Grosso, Brazil",
                },
            },
            effects={
                "ecological": [
                    "Soybean expansion spreads deforestation pressure "
                    "into Para and Rondonia through new transport "
                    "corridors",
                ],
            },
        )

        codes = MetacouplingAssistant._extract_mentioned_adm1_from_text(parsed)
        # Pará and Rondônia are named in a specific effect → substantive
        found = {"BRA018" in codes, "BRA026" in codes}
        assert True in found, (
            f"Expected BRA018 or BRA026 from cause/effect mention, "
            f"got {codes}"
        )

    def test_generate_map_passes_empty_set_not_none(self, monkeypatch):
        """When mentioned_adm1_set is empty after focal-discard,
        _generate_map() must pass an empty set to the renderer,
        NOT None. Passing None puts the classifier into legacy mode
        which colors all DB neighbors as pericoupling — exactly the
        bug we're fixing.
        """
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_focal_adm1_map(focal_adm1, **kwargs):
            captured["focal_adm1"] = focal_adm1
            captured["kwargs"] = kwargs
            return "fake-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_plot_focal_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        # LLM returns empty mentioned_adm1_regions. The regex
        # fallback may pick up the focal (BRA011), which then gets
        # discarded, leaving an empty set.
        parsed = make_parsed_analysis(
            coupling_classification=(
                "Telecoupling: Mato Grosso soybean exports to China."
            ),
            systems={
                "sending": {
                    "name": "Mato Grosso, Brazil",
                    "geographic_scope": "Mato Grosso, Brazil",
                },
            },
            flows=[
                {"category": "matter", "direction": "Mato Grosso \u2192 China"},
            ],
            map_data={
                "focal_country": "BRA",
                "adm1_region": "BRA011",
                "mentioned_adm1_regions": [],
                "receiving_countries": ["CHN"],
                "spillover_countries": [],
                "flows": [],
            },
        )

        advisor._generate_map(parsed)

        # The renderer must receive an empty SET (strict mode), not
        # None (legacy mode). This is the regression check.
        passed = captured["kwargs"].get("mentioned_adm1_codes")
        assert passed is not None, (
            "Empty mentioned_adm1_set was incorrectly passed as "
            "None — this would put the classifier into legacy mode "
            "and colour all DB neighbors as pericoupling."
        )
        assert isinstance(passed, set)
        assert len(passed) == 0
        # The focal must NOT be in the passed set
        assert "BRA011" not in passed

    def test_generate_map_discards_focal_from_mentioned_adm1(self, monkeypatch):
        """_generate_map() should discard the focal ADM1 from
        mentioned_adm1_codes passed to the renderer. The focal is
        handled separately (it gets intracoupling); keeping it in
        mentioned_adm1 is cosmetically noisy.
        """
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        captured: dict[str, object] = {}

        def fake_plot_focal_adm1_map(focal_adm1, **kwargs):
            captured["focal_adm1"] = focal_adm1
            captured["kwargs"] = kwargs
            return "fake-figure"

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
            fake_plot_focal_adm1_map,
        )

        advisor = MetacouplingAssistant(
            llm_client=MockLLMClient(),
            auto_map=True,
        )
        # LLM returned mentioned_adm1_regions=[] — the regex fallback
        # will run and (because the classification text mentions
        # "Mato Grosso") may pick up BRA011.
        parsed = make_parsed_analysis(
            coupling_classification=(
                "Telecoupling: Mato Grosso soybean exports to China."
            ),
            systems={
                "sending": {
                    "name": "Mato Grosso, Brazil",
                    "geographic_scope": "Mato Grosso, Brazil",
                },
            },
            flows=[
                {"category": "matter", "direction": "Mato Grosso \u2192 China"},
            ],
            map_data={
                "focal_country": "BRA",
                "adm1_region": "BRA011",
                "mentioned_adm1_regions": [],  # empty → fallback runs
                "receiving_countries": ["CHN"],
                "spillover_countries": [],
                "flows": [],
            },
        )

        advisor._generate_map(parsed)

        passed_adm1 = captured["kwargs"].get("mentioned_adm1_codes")
        # The focal must NOT appear in the passed set — it's handled
        # separately by the classifier.
        if passed_adm1 is not None:
            assert "BRA011" not in passed_adm1, (
                f"Focal ADM1 should be discarded, but found in "
                f"mentioned_adm1_codes: {passed_adm1}"
            )


class TestExtractMapDataSupranational:
    """Primary path (`_extract_map_data_from_analysis`) recognises
    supranational entities (EU / ASEAN / NAFTA / USMCA) when the LLM
    slips past the prompt's "use ISO codes" rule and emits an umbrella
    name as a flow target.

    Closes the gap left in PR #5: the resolver-path supranational
    handling there only fired for the legacy fallback path.  These
    tests exercise the structured (LLM-extracted) primary path."""

    @staticmethod
    def _make_advisor(fake_json):
        """Build an assistant with a stubbed LLM client that returns
        ``fake_json`` (a dict) as the structured-extraction response."""
        import json

        from metacouplingllm.llm.client import LLMResponse

        class _StubClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content=json.dumps(fake_json))

        return MetacouplingAssistant(
            llm_client=_StubClient(),
            auto_map=False,
        )

    @staticmethod
    def _basic_parsed():
        """A minimal ParsedAnalysis good enough to feed the extractor."""
        from ._helpers import make_parsed_analysis
        return make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={
                "sending": {
                    "name": "Michigan, USA",
                    "geographic_scope": "Michigan, USA",
                },
            },
        )

    def test_eu_target_stamps_supranational_fields(self):
        """LLM returns 'European Union' as target → flow gets the
        ``target_supranational`` and ``target_supranational_members``
        marker fields stamped on it."""
        advisor = self._make_advisor({
            "focal_country": "USA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": [],
            "spillover_countries": [],
            "flows": [
                {
                    "category": "matter",
                    "source": "USA",
                    "target": "European Union",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        assert result is not None
        assert len(result["flows"]) == 1
        flow = result["flows"][0]
        assert flow["target_supranational"] == "European Union"
        assert len(flow["target_supranational_members"]) == 27
        assert "European Union" in flow["direction"]
        assert flow["target"] is None  # supranational replaces single ISO
        assert flow["source"] == "USA"

    def test_eu_skipped_when_member_already_in_receiving(self):
        """Conditional rule: if the LLM listed any EU member country
        explicitly in receiving_countries, the umbrella mention is
        treated as redundant and the flow is dropped."""
        advisor = self._make_advisor({
            "focal_country": "USA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": ["DEU", "FRA"],
            "spillover_countries": [],
            "flows": [
                {
                    "category": "matter",
                    "source": "USA",
                    "target": "European Union",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        # Flow is dropped — no supranational stamp, no country flow.
        # The LLM's specific receiving_countries (DEU, FRA) are the
        # source of truth.
        assert result["flows"] == []

    def test_eu_skipped_when_member_in_spillover(self):
        """Same conditional rule but checking spillover_countries."""
        advisor = self._make_advisor({
            "focal_country": "USA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": [],
            "spillover_countries": ["FRA"],
            "flows": [
                {
                    "category": "matter",
                    "source": "USA",
                    "target": "European Union",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        assert result["flows"] == []

    def test_asean_target_stamps_10_members(self):
        advisor = self._make_advisor({
            "focal_country": "BRA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": [],
            "spillover_countries": [],
            "flows": [
                {
                    "category": "matter",
                    "source": "BRA",
                    "target": "ASEAN",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        assert len(result["flows"]) == 1
        assert result["flows"][0]["target_supranational"] == "ASEAN"
        assert len(result["flows"][0]["target_supranational_members"]) == 10

    def test_nafta_and_usmca_are_recognised(self):
        for alias in ("NAFTA", "USMCA"):
            advisor = self._make_advisor({
                "focal_country": "CHN",
                "adm1_region": None,
                "mentioned_adm1_regions": [],
                "receiving_countries": [],
                "spillover_countries": [],
                "flows": [
                    {
                        "category": "matter",
                        "source": "CHN",
                        "target": alias,
                        "bidirectional": False,
                    },
                ],
            })
            result = advisor._extract_map_data_from_analysis(
                self._basic_parsed(),
            )
            assert len(result["flows"]) == 1
            members = result["flows"][0]["target_supranational_members"]
            assert sorted(members) == sorted(["USA", "MEX", "CAN"])

    def test_self_loop_member_dropped(self):
        """If the source is itself a member of the supranational, the
        flow is dropped (would render as a self-loop into its own
        region)."""
        advisor = self._make_advisor({
            "focal_country": "FRA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": [],
            "spillover_countries": [],
            "flows": [
                {
                    "category": "matter",
                    "source": "FRA",
                    "target": "European Union",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        assert result["flows"] == []

    def test_unknown_target_still_dropped(self):
        """Non-supranational, non-ISO targets (typos, fictional places)
        keep the existing silent-drop behaviour."""
        advisor = self._make_advisor({
            "focal_country": "USA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": [],
            "spillover_countries": [],
            "flows": [
                {
                    "category": "matter",
                    "source": "USA",
                    "target": "Atlantis",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        assert result["flows"] == []

    def test_regular_country_pair_unchanged(self):
        """Regression: regular ISO-pair flows still emit the same
        country-to-country flow dict (no supranational fields)."""
        advisor = self._make_advisor({
            "focal_country": "USA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": ["CHN"],
            "spillover_countries": [],
            "flows": [
                {
                    "category": "matter",
                    "source": "USA",
                    "target": "CHN",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        assert len(result["flows"]) == 1
        flow = result["flows"][0]
        assert flow["source"] == "USA"
        assert flow["target"] == "CHN"
        assert "target_supranational" not in flow
        assert "target_supranational_members" not in flow

    def test_eu_aliases_resolve(self):
        """The 'EU' / 'eu' alias for 'European Union' should produce
        the same supranational stamp."""
        for alias in ("EU", "eu", "e.u.", "the european union"):
            advisor = self._make_advisor({
                "focal_country": "USA",
                "adm1_region": None,
                "mentioned_adm1_regions": [],
                "receiving_countries": [],
                "spillover_countries": [],
                "flows": [
                    {
                        "category": "matter",
                        "source": "USA",
                        "target": alias,
                        "bidirectional": False,
                    },
                ],
            })
            result = advisor._extract_map_data_from_analysis(
                self._basic_parsed(),
            )
            assert len(result["flows"]) == 1, f"alias {alias!r} dropped"
            assert (
                result["flows"][0]["target_supranational"]
                == "European Union"
            )

    def test_generate_map_propagates_supranational_fields(self):
        """The downstream `_generate_map` flow-copy loop must keep the
        supranational marker fields so the renderer can use them.

        Uses a country-level focal (no ADM1 region) so we exercise the
        country-level rendering path, then stubs ``plot_analysis_map``
        to inspect the flow dict the renderer is actually handed.
        """
        advisor = self._make_advisor({
            "focal_country": "USA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": [],
            "spillover_countries": [],
            "flows": [
                {
                    "category": "matter",
                    "source": "USA",
                    "target": "European Union",
                    "bidirectional": False,
                },
            ],
        })
        captured: dict[str, object] = {}

        def _fake_plot(parsed, **kwargs):
            captured["flows"] = kwargs.get("flows")
            return None

        import metacouplingllm.visualization.worldmap as wm
        original = wm.plot_analysis_map
        wm.plot_analysis_map = _fake_plot
        try:
            from ._helpers import make_parsed_analysis
            # Country-level system name avoids an ADM1 hit that would
            # route us through plot_focal_adm1_map instead.
            parsed = make_parsed_analysis(
                coupling_classification="telecoupling",
                systems={"sending": {"name": "United States"}},
            )
            parsed.map_data = advisor._extract_map_data_from_analysis(parsed)
            advisor._generate_map(parsed)
        finally:
            wm.plot_analysis_map = original

        assert "flows" in captured, (
            "plot_analysis_map was never called — _generate_map likely "
            "took the ADM1 branch instead of the country-level branch."
        )
        assert captured["flows"] is not None
        assert len(captured["flows"]) == 1
        passed_flow = captured["flows"][0]
        assert passed_flow.get("target_supranational") == "European Union"
        assert len(
            passed_flow.get("target_supranational_members", [])
        ) == 27


class TestValidateAdm1Pericoupling:
    """Tests for ADM1-level pericoupling validation."""

    def test_michigan_produces_adm1_info(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Michigan pork production system",
                    "geographic_scope": "Michigan, United States",
                },
            },
        )
        result = MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        assert result is True
        info = parsed.pericoupling_info
        assert info is not None
        assert info["level"] == "adm1"
        assert "Michigan" in info["focal_region"]
        assert "USA023" in info["focal_region"]
        assert "United States" in info["focal_country"]
        assert info.get("domestic_neighbors")  # Michigan has domestic neighbors
        assert "Indiana" in info["domestic_neighbors"]
        assert "Ohio" in info["domestic_neighbors"]

    def test_country_level_returns_false(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil", "geographic_scope": "Brazil"},
            },
        )
        result = MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        assert result is False
        assert parsed.pericoupling_info is None

    def test_validate_pericoupling_uses_adm1_for_michigan(self):
        """The main _validate_pericoupling should delegate to ADM1."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {
                    "name": "Michigan pork production system",
                    "geographic_scope": "Michigan, United States",
                },
                "receiving": {"name": "China"},
            },
        )
        MetacouplingAssistant._validate_pericoupling(parsed)
        info = parsed.pericoupling_info
        assert info is not None
        assert info["level"] == "adm1"

    def test_validate_pericoupling_falls_through_for_countries(self):
        """Country-level validation still works when ADM1 doesn't apply."""
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={
                "sending": {"name": "Brazil", "geographic_scope": "Brazil"},
                "receiving": {"name": "China", "geographic_scope": "China"},
            },
        )
        MetacouplingAssistant._validate_pericoupling(parsed)
        info = parsed.pericoupling_info
        assert info is not None
        # Should be country-level (no "level" key)
        assert info.get("level") is None
        assert "BRA" in info.get("focal_country", "")

    # ------------------------------------------------------------------
    # Consistency-note branches (PR #13)
    # ------------------------------------------------------------------
    #
    # Each test below picks a Michigan-focal fixture so the resolver
    # locks onto USA023 (well-populated with both domestic and cross-
    # border neighbours in the bundled DB).  Mentioned partners are
    # placed in the flow ``direction`` field so the arrow-split path
    # in ``_extract_mentioned_adm1_from_text`` picks them up.  The
    # focal's ``geographic_scope`` includes "United States" so the
    # relevance guard accepts USA-side ADM1 codes; tele-partner cases
    # add the partner country name to the classification text so the
    # guard accepts foreign codes too.

    @staticmethod
    def _michigan_parsed(classification, direction):
        """Build a Michigan-focal ParsedAnalysis with the partner in
        the flow direction (arrow-split path the extractor scans)."""
        from ._helpers import make_parsed_analysis
        return make_parsed_analysis(
            coupling_classification=classification,
            systems={
                "sending": {
                    "name": "Michigan",
                    "geographic_scope": "Michigan, United States",
                },
            },
            flows=[
                {
                    "category": "matter",
                    "direction": direction,
                    "description": "trade",
                },
            ],
        )

    def test_note_consistent_when_classification_matches_pericoupled(self):
        """LLM said pericoupling + DB shows Ohio is adjacent → consistent."""
        parsed = self._michigan_parsed(
            "Pericoupling between Michigan and Ohio in the United States",
            "Michigan → Ohio",
        )
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        note = parsed.pericoupling_info["note"]
        assert "consistent" in note.lower()
        assert "Consider revising" not in note

    def test_note_warns_when_pericoupled_partners_but_no_peri_classification(self):
        """LLM said telecoupling + DB shows Ohio is adjacent →
        warns 'at least one mentioned subnational region is adjacent
        ... Consider revising.'"""
        parsed = self._michigan_parsed(
            "Telecoupling: Michigan in United States exports to "
            "international markets",
            "Michigan → Ohio",
        )
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        note = parsed.pericoupling_info["note"]
        assert "at least one mentioned subnational region is adjacent" in note
        assert "Consider revising" in note
        assert "consistent" not in note.lower()

    def test_note_consistent_when_classification_matches_telecoupled(self):
        """LLM said telecoupling + DB shows Mato Grosso is non-adjacent
        → consistent."""
        parsed = self._michigan_parsed(
            "Telecoupling between Michigan, United States and "
            "Mato Grosso, Brazil",
            "Michigan → Mato Grosso",
        )
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        note = parsed.pericoupling_info["note"]
        assert "consistent" in note.lower()
        assert "Consider revising" not in note

    def test_note_warns_when_only_telecoupled_but_no_tele_classification(self):
        """LLM said pericoupling + DB shows Mato Grosso is non-adjacent
        (no pericoupled partners mentioned) → warns 'all mentioned
        subnational regions are non-adjacent ... Consider revising.'"""
        parsed = self._michigan_parsed(
            "Pericoupling between Michigan, United States and "
            "Mato Grosso, Brazil",
            "Michigan → Mato Grosso",
        )
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        note = parsed.pericoupling_info["note"]
        assert "all mentioned subnational regions are non-adjacent" in note
        assert "Consider revising" in note
        assert "consistent" not in note.lower()

    def test_note_neutral_when_no_classification_text(self):
        """No ``coupling_classification`` to compare against → note
        describes the DB result neutrally rather than claiming a
        consistency that wasn't checked."""
        parsed = self._michigan_parsed(
            "",  # no classification text
            "Michigan → Ohio",
        )
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        note = parsed.pericoupling_info["note"]
        assert "consistent" not in note.lower(), (
            "Note must NOT claim consistency when no classification "
            "text was available to compare against."
        )
        assert "returned neighbour information" in note


class TestFormatterAdm1PericouplingInfo:
    """Tests for ADM1-level pericoupling info in formatted output."""

    def test_adm1_info_renders_subnational_header(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis
        from metacouplingllm.output.formatter import AnalysisFormatter

        parsed = ParsedAnalysis(
            coupling_classification="telecoupling",
            pericoupling_info={
                "level": "adm1",
                "focal_region": "Michigan (USA023)",
                "focal_country": "United States of America (USA)",
                "domestic_neighbors": "Indiana (USA015), Ohio (USA036)",
                "cross_border_neighbors": "Ontario (CAN008)",
                "note": "LLM classification is consistent.",
            },
        )
        output = AnalysisFormatter.format_full(parsed)
        assert "PERICOUPLING DATABASE VALIDATION (SUBNATIONAL)" in output
        assert "Michigan (USA023)" in output
        # Headers were title-cased: "Same-country neighbors:" ->
        # "Domestic Neighbors:" and "Cross-border neighbors:" ->
        # "Cross Border Neighbors:".
        assert "Domestic Neighbors:" in output
        assert "Cross Border Neighbors:" in output
        assert "Ontario" in output

    def test_country_info_renders_unchanged(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from ._helpers import make_parsed_analysis
        from metacouplingllm.output.formatter import AnalysisFormatter

        parsed = ParsedAnalysis(
            coupling_classification="telecoupling",
            pericoupling_info={
                "focal_country": "Brazil (BRA)",
                "pair_results": "Brazil (BRA) ↔ China (CHN): TELECOUPLED",
                "note": "Consistent.",
            },
        )
        output = AnalysisFormatter.format_full(parsed)
        assert "PERICOUPLING DATABASE VALIDATION" in output
        assert "SUBNATIONAL" not in output
        assert "Brazil (BRA) ↔ China (CHN): TELECOUPLED" in output


# ---------------------------------------------------------------------------
# Prompt-budget caps in _extract_map_data_from_analysis
# ---------------------------------------------------------------------------


class TestExtractMapDataPromptBudgets:
    """Verify the loosened prompt-budget caps in
    ``_extract_map_data_from_analysis`` preserve high-signal content
    (bilateral country names, geographic scope, web snippets up to
    user-configured limits) instead of silently truncating it."""

    @staticmethod
    def _make_advisor_with_capture(**kwargs):
        """Build an advisor whose LLM client records the user_text it
        receives, then returns a minimal valid JSON response so the
        extraction completes without errors.

        ``kwargs`` are forwarded to ``MetacouplingAssistant`` (e.g.
        ``web_search_max_results=20``)."""
        import json

        captured: dict[str, str] = {}

        from metacouplingllm.llm.client import LLMResponse

        class _CaptureClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                # Grab the user message — that's where the summary
                # of the analysis lives.
                for m in messages:
                    if m.role == "user":
                        captured["user_text"] = m.content
                        break
                return LLMResponse(content=json.dumps({
                    "focal_country": "BRA",
                    "adm1_region": None,
                    "mentioned_adm1_regions": [],
                    "receiving_countries": ["CHN"],
                    "spillover_countries": [],
                    "flows": [],
                }))

        advisor = MetacouplingAssistant(
            llm_client=_CaptureClient(),
            auto_map=False,
            **kwargs,
        )
        return advisor, captured

    # --- Flow description cap (100 -> 500) -------------------------------

    def test_flow_description_under_500_chars_preserved(self):
        from ._helpers import make_parsed_analysis

        advisor, captured = self._make_advisor_with_capture()
        # 300-char description with bilateral specifics at the end
        long_desc = (
            "Soybeans exported from Mato Grosso, Brazil to Henan, "
            "Shanghai, Liaoning, Guangdong, and other Chinese coastal "
            "provinces, totaling 35.4 million tonnes per year across "
            "the 2020-2024 period covered by this analysis with "
            "particular concentration in the bilateral relationship "
            "with eastern China"
        )
        assert 100 < len(long_desc) <= 500
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={"sending": {"name": "Brazil"}},
            flows=[{
                "category": "matter",
                "direction": "Brazil → China",
                "description": long_desc,
            }],
        )
        advisor._extract_map_data_from_analysis(parsed)
        # Full description text should make it into the prompt
        for needle in ("Henan", "Shanghai", "Liaoning", "Guangdong"):
            assert needle in captured["user_text"], (
                f"{needle!r} missing from prompt — flow description cap "
                f"truncating below 500 chars?"
            )

    def test_flow_description_truncated_at_500(self):
        from ._helpers import make_parsed_analysis

        advisor, captured = self._make_advisor_with_capture()
        # Construct a description with markers we can locate precisely.
        # The 500-char cap should preserve everything up to char 500
        # but drop "MARKER_PAST_CAP" which sits well past it.
        early_marker = " Henan, Shanghai, "
        long_desc = "X" * 200 + early_marker + "Y" * 400 + "MARKER_PAST_CAP"
        assert len(long_desc) > 500
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={"sending": {"name": "Brazil"}},
            flows=[{
                "category": "matter",
                "direction": "Brazil → China",
                "description": long_desc,
            }],
        )
        advisor._extract_map_data_from_analysis(parsed)
        # Early content (chars ~200) survives the cap.
        assert "Henan" in captured["user_text"]
        assert "Shanghai" in captured["user_text"]
        # Content beyond char 500 (the marker is at char ~618) is dropped.
        assert "MARKER_PAST_CAP" not in captured["user_text"]

    # --- System text cap (400 -> 800) and field priority -----------------

    def test_system_geographic_scope_survives_verbose_subsystems(self):
        from ._helpers import make_parsed_analysis

        advisor, captured = self._make_advisor_with_capture()
        # Each subsystem ~300 chars; combined with name they would have
        # blown past the old 400-char cap. With priority ordering +
        # 800-char cap, geographic_scope must still appear.
        verbose_human = (
            "Large agribusinesses (Bunge, Cargill, ADM), smallholder "
            "cooperatives, federal agricultural agencies, port "
            "logistics operators, and rural land speculators driving "
            "the soy frontier expansion into the Cerrado biome"
        ) * 2
        verbose_natural = (
            "Cerrado savanna ecosystems, Amazon transition forests, "
            "soybean monoculture landscapes, freshwater river systems "
            "supporting both irrigation and downstream urban supply"
        ) * 2
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={
                "sending": {
                    "name": "Brazilian Soybean Production Complex",
                    "human_subsystem": verbose_human,
                    "natural_subsystem": verbose_natural,
                    "geographic_scope": (
                        "Mato Grosso, Pará, Rondônia, Goiás states"
                    ),
                },
            },
        )
        advisor._extract_map_data_from_analysis(parsed)
        assert "geographic_scope: Mato Grosso" in captured["user_text"], (
            "geographic_scope must survive priority ordering even when "
            "subsystems are verbose"
        )

    def test_system_field_priority_order(self):
        """``name`` < ``geographic_scope`` < ``human_subsystem`` in
        the rendered prompt order."""
        from ._helpers import make_parsed_analysis

        advisor, captured = self._make_advisor_with_capture()
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={
                "sending": {
                    "name": "Sample sending system",
                    "human_subsystem": "Sample humans",
                    "natural_subsystem": "Sample natural",
                    "geographic_scope": "Sample geography",
                },
            },
        )
        advisor._extract_map_data_from_analysis(parsed)
        text = captured["user_text"]
        name_idx = text.find("name: Sample sending system")
        geo_idx = text.find("geographic_scope: Sample geography")
        human_idx = text.find("human_subsystem: Sample humans")
        natural_idx = text.find("natural_subsystem: Sample natural")
        assert name_idx >= 0 and geo_idx >= 0 and human_idx >= 0
        assert name_idx < geo_idx, "name must come before geographic_scope"
        assert geo_idx < human_idx, (
            "geographic_scope must come before human_subsystem"
        )
        assert human_idx < natural_idx, (
            "low-priority field order: human before natural"
        )

    # --- Web-snippet cap respects user config ----------------------------

    def test_web_snippets_respect_max_results(self):
        """Setting ``web_search_max_results=20`` and supplying 20 mock
        web results should put all 20 in the prompt — the previous
        hardcoded ``[:10]`` cap dropped 10 of them silently."""
        from ._helpers import make_parsed_analysis

        advisor, captured = self._make_advisor_with_capture(
            web_search_max_results=20,
        )
        advisor._last_web_results = [
            {"title": f"Result {i}", "model_summary": f"snip {i}"}
            for i in range(20)
        ]
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={"sending": {"name": "Brazil"}},
        )
        advisor._extract_map_data_from_analysis(parsed)
        # All 20 snippets should appear as [W1]..[W20]
        for i in range(1, 21):
            assert f"[W{i}]" in captured["user_text"], (
                f"[W{i}] missing from prompt — web snippet cap "
                f"truncating below web_search_max_results?"
            )

    def test_web_snippets_capped_at_module_constant(self):
        """If somehow more than ``_MAX_WEB_SNIPPETS_IN_MAP_PROMPT``
        results are sitting on ``_last_web_results`` (e.g. user set
        ``web_search_max_results=1000``), the defensive ceiling
        caps the prompt at the module constant."""
        from metacouplingllm.core import _MAX_WEB_SNIPPETS_IN_MAP_PROMPT

        from ._helpers import make_parsed_analysis

        # 150 > 100, simulating a pathological config
        n = _MAX_WEB_SNIPPETS_IN_MAP_PROMPT + 50
        advisor, captured = self._make_advisor_with_capture(
            web_search_max_results=n,
        )
        advisor._last_web_results = [
            {"title": f"Result {i}", "model_summary": f"snip {i}"}
            for i in range(n)
        ]
        parsed = make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={"sending": {"name": "Brazil"}},
        )
        advisor._extract_map_data_from_analysis(parsed)
        # Up to the constant — present
        assert f"[W{_MAX_WEB_SNIPPETS_IN_MAP_PROMPT}]" in (
            captured["user_text"]
        )
        # Beyond the constant — absent
        assert f"[W{_MAX_WEB_SNIPPETS_IN_MAP_PROMPT + 1}]" not in (
            captured["user_text"]
        )
        assert f"[W{n}]" not in captured["user_text"]


# ---------------------------------------------------------------------------
# Flow-category aliasing — _FLOW_CATEGORY_ALIASES / _normalize_flow_category
# ---------------------------------------------------------------------------


class TestFlowCategoryAliases:
    """The ``_FLOW_CATEGORY_ALIASES`` table maps LLM slip / subset terms
    onto the six Liu 2017 canonical categories.  These tests exercise
    the helper directly and via the two consumer code paths."""

    # --- Helper-level tests ----------------------------------------------

    @pytest.mark.parametrize("raw,expected", [
        # canonical forms pass through unchanged
        ("matter",      "matter"),
        ("capital",     "capital"),
        ("information", "information"),
        ("energy",      "energy"),
        ("people",      "people"),
        ("organisms",   "organisms"),
    ])
    def test_canonical_categories_pass_through(self, raw, expected):
        from metacouplingllm.core import _normalize_flow_category
        assert _normalize_flow_category(raw) == expected

    @pytest.mark.parametrize("alias,canonical", [
        # matter aliases
        ("material",    "matter"),
        ("materials",   "matter"),
        ("goods",       "matter"),
        ("commodity",   "matter"),
        ("commodities", "matter"),
        ("cargo",       "matter"),
        ("food",        "matter"),
        # capital aliases
        ("financial",   "capital"),
        ("money",       "capital"),
        ("monetary",    "capital"),
        ("investment",  "capital"),
        ("cash",        "capital"),
        ("currency",    "capital"),
        ("funds",       "capital"),
        ("payment",     "capital"),
        ("payments",    "capital"),
        # information aliases
        ("data",        "information"),
        ("knowledge",   "information"),
        ("info",        "information"),
        ("signals",     "information"),
        # energy aliases
        ("electricity", "energy"),
        ("electrical",  "energy"),
        ("electric",    "energy"),
        ("fuel",        "energy"),
        ("fuels",       "energy"),
        # people aliases
        ("humans",      "people"),
        ("migration",   "people"),
        ("migrants",    "people"),
        ("labor",       "people"),
        ("labour",      "people"),
        ("workers",     "people"),
        ("tourists",    "people"),
        ("tourism",     "people"),
        ("personnel",   "people"),
        ("passengers",  "people"),
        # organisms aliases
        ("organism",    "organisms"),
        ("species",     "organisms"),
        ("wildlife",    "organisms"),
        ("animals",     "organisms"),
        ("plants",      "organisms"),
        ("livestock",   "organisms"),
    ])
    def test_alias_normalises_to_canonical(self, alias, canonical):
        from metacouplingllm.core import _normalize_flow_category
        assert _normalize_flow_category(alias) == canonical

    @pytest.mark.parametrize("alias,canonical", [
        ("Material", "matter"),
        ("GOODS", "matter"),
        ("  Money  ", "capital"),
        ("Electricity", "energy"),
        ("TOURISM", "people"),
    ])
    def test_case_and_whitespace_insensitive(self, alias, canonical):
        """``_normalize_flow_category`` strips + lowercases the input."""
        from metacouplingllm.core import _normalize_flow_category
        assert _normalize_flow_category(alias) == canonical

    @pytest.mark.parametrize("rejected", [
        # Genuinely ambiguous — kept rejected on purpose.
        "power",       # political vs electrical vs computational
        "products",    # digital (information) vs physical (matter)
        "resources",   # could be people / energy / money / matter
        "services",    # capital / people / information
        "seeds",       # organisms vs harvested commodity
        "crops",       # standing crop vs harvested grain
        "economic",    # capital vs trade-in-goods
        # Clearly outside the framework
        "miscellaneous",
        "other",
        "xyz",
        # Empty / whitespace
        "",
        "   ",
    ])
    def test_rejected_terms_return_none(self, rejected):
        """Terms that are ambiguous or outside the framework must
        return ``None`` so the caller can drop the flow."""
        from metacouplingllm.core import _normalize_flow_category
        assert _normalize_flow_category(rejected) is None

    def test_alias_table_targets_are_all_canonical(self):
        """Every value in ``_FLOW_CATEGORY_ALIASES`` must be one of the
        six canonical Liu 2017 categories — a structural invariant
        that prevents accidental typos (e.g. ``"capitall"``)."""
        from metacouplingllm.core import (
            _CANONICAL_FLOW_CATEGORIES, _FLOW_CATEGORY_ALIASES,
        )
        for alias, target in _FLOW_CATEGORY_ALIASES.items():
            assert target in _CANONICAL_FLOW_CATEGORIES, (
                f"alias {alias!r} maps to non-canonical {target!r}"
            )

    def test_alias_table_no_canonical_alias_to_itself(self):
        """Canonical names should not appear as keys of the alias
        table — they are recognised by the canonical-set check
        directly, and listing them here would be confusing."""
        from metacouplingllm.core import (
            _CANONICAL_FLOW_CATEGORIES, _FLOW_CATEGORY_ALIASES,
        )
        for canonical in _CANONICAL_FLOW_CATEGORIES:
            assert canonical not in _FLOW_CATEGORY_ALIASES, (
                f"{canonical!r} should not be in the alias table"
            )

    # --- Integration via _extract_map_data_from_analysis -----------------

    @staticmethod
    def _make_advisor(fake_json):
        import json

        from metacouplingllm.llm.client import LLMResponse

        class _StubClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content=json.dumps(fake_json))

        return MetacouplingAssistant(
            llm_client=_StubClient(),
            auto_map=False,
        )

    @staticmethod
    def _basic_parsed():
        from ._helpers import make_parsed_analysis
        return make_parsed_analysis(
            coupling_classification="telecoupling",
            systems={"sending": {"name": "United States"}},
        )

    @pytest.mark.parametrize("llm_label,canonical", [
        ("goods",       "matter"),
        ("money",       "capital"),
        ("electricity", "energy"),
        ("tourism",     "people"),
        ("livestock",   "organisms"),
        ("info",        "information"),
    ])
    def test_extract_map_data_aliases_subset_terms(
        self, llm_label, canonical,
    ):
        """LLM-emitted subset terms (e.g. ``"electricity"``) are
        normalised to their canonical category by the structured
        extraction path."""
        advisor = self._make_advisor({
            "focal_country": "USA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": ["CHN"],
            "spillover_countries": [],
            "flows": [
                {
                    "category": llm_label,
                    "source": "USA",
                    "target": "CHN",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        assert len(result["flows"]) == 1
        assert result["flows"][0]["category"] == canonical

    @pytest.mark.parametrize("rejected_label", [
        "power", "products", "resources", "services", "economic", "xyz",
    ])
    def test_extract_map_data_drops_rejected_categories(
        self, rejected_label,
    ):
        """Ambiguous / unknown category labels still cause the flow
        to be silently dropped."""
        advisor = self._make_advisor({
            "focal_country": "USA",
            "adm1_region": None,
            "mentioned_adm1_regions": [],
            "receiving_countries": ["CHN"],
            "spillover_countries": [],
            "flows": [
                {
                    "category": rejected_label,
                    "source": "USA",
                    "target": "CHN",
                    "bidirectional": False,
                },
            ],
        })
        result = advisor._extract_map_data_from_analysis(self._basic_parsed())
        assert result["flows"] == []


# ---------------------------------------------------------------------------
# _FLOW_ARROW_RE consistency tests
# ---------------------------------------------------------------------------


class TestFlowArrowRegex:
    """Verify the canonical arrow regex splits all expected patterns."""

    def test_unicode_arrow(self):
        from metacouplingllm.core import _FLOW_ARROW_RE
        assert _FLOW_ARROW_RE.split("United States → China") == [
            "United States", "China",
        ]

    def test_ascii_arrow(self):
        from metacouplingllm.core import _FLOW_ARROW_RE
        assert _FLOW_ARROW_RE.split("Brazil -> Japan") == ["Brazil", "Japan"]

    def test_fat_arrow(self):
        from metacouplingllm.core import _FLOW_ARROW_RE
        assert _FLOW_ARROW_RE.split("Brazil => Japan") == ["Brazil", "Japan"]

    def test_bidirectional_unicode(self):
        from metacouplingllm.core import _FLOW_ARROW_RE
        parts = _FLOW_ARROW_RE.split("USA \u2194 China")
        assert parts == ["USA", "China"]

    def test_bidirectional_ascii(self):
        from metacouplingllm.core import _FLOW_ARROW_RE
        parts = _FLOW_ARROW_RE.split("USA <-> China")
        assert parts == ["USA", "China"]

    def test_fat_bidirectional(self):
        from metacouplingllm.core import _FLOW_ARROW_RE
        parts = _FLOW_ARROW_RE.split("A <=> B")
        assert parts == ["A", "B"]

    def test_no_arrow_returns_single(self):
        from metacouplingllm.core import _FLOW_ARROW_RE
        parts = _FLOW_ARROW_RE.split("Within United States")
        assert parts == ["Within United States"]

    def test_whitespace_handling(self):
        from metacouplingllm.core import _FLOW_ARROW_RE
        parts = _FLOW_ARROW_RE.split("USA  →  China")
        assert parts == ["USA", "China"]


# ---------------------------------------------------------------------------
# Flow parse warnings — observable failures from _resolve_flows_for_map
# ---------------------------------------------------------------------------


class TestResolveFlowsForMapWarnings:
    """``_resolve_flows_for_map`` returns ``(arrows, warnings)`` and
    populates the warnings list whenever a flow is dropped due to an
    unparseable direction or unresolvable endpoints."""

    def test_warning_fires_on_unparseable_prose(self, caplog):
        import logging

        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Michigan, USA"}},
            flows=[
                {
                    "category": "matter",
                    "direction": (
                        "Pork from Michigan is exported to Japan."
                    ),
                    "description": "",
                },
            ],
        )
        with caplog.at_level(
            logging.WARNING, logger="metacouplingllm.core"
        ):
            arrows, warnings = (
                MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
            )
        assert arrows == []
        assert len(warnings) == 1
        w = warnings[0]
        assert w["category"] == "matter"
        assert "Michigan" in w["direction"]
        assert "no recognized arrow" in w["reason"]
        # Logger captured the same payload.
        assert any(
            "could not be resolved" in rec.getMessage()
            for rec in caplog.records
        )

    def test_warning_does_not_fire_on_resolved_flow(self, caplog):
        import logging

        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Brazil → China",
                    "description": "Soybeans",
                },
            ],
        )
        with caplog.at_level(
            logging.WARNING, logger="metacouplingllm.core"
        ):
            arrows, warnings = (
                MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
            )
        assert len(arrows) >= 1
        assert warnings == []

    def test_warning_does_not_fire_on_within_skip(self, caplog):
        import logging

        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Michigan, USA"}},
            flows=[
                {
                    "category": "energy",
                    "direction": (
                        "Mostly within Michigan and embedded in exports"
                    ),
                    "description": "",
                },
            ],
        )
        with caplog.at_level(
            logging.WARNING, logger="metacouplingllm.core"
        ):
            arrows, warnings = (
                MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
            )
        assert arrows == []
        assert warnings == []

    def test_warning_reason_for_unresolvable_endpoints(self, caplog):
        """Connector present, but neither side resolves."""
        import logging

        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Some unknown planet"}},
            flows=[
                {
                    "category": "matter",
                    "direction": "Atlantis → Wakanda",
                    "description": "",
                },
            ],
        )
        with caplog.at_level(
            logging.WARNING, logger="metacouplingllm.core"
        ):
            arrows, warnings = (
                MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
            )
        assert len(warnings) == 1
        assert "endpoints could not be resolved" in warnings[0]["reason"]


# ---------------------------------------------------------------------------
# Supranational targets — single-region detection in _resolve_flows_for_map
# ---------------------------------------------------------------------------


class TestResolveFlowsSupranational:
    """Supranational targets emit one flow with a
    ``target_supranational_members`` field for the renderer to expand."""

    def test_eu_target_emits_single_supranational_flow(self):
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Michigan, United States"},
                "receiving": {"name": "European Union"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Michigan → European Union",
                    "description": "Pork exported",
                },
            ],
        )
        arrows, warnings = (
            MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        )
        assert warnings == []
        assert len(arrows) == 1
        assert arrows[0]["target_supranational"] == "European Union"
        assert len(arrows[0]["target_supranational_members"]) == 27
        assert "European Union" in arrows[0]["direction"]

    def test_eu_skipped_when_explicit_members_present(self):
        """When the analysis already lists EU members, the umbrella
        ``European Union`` mention is treated as redundant."""
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={"sending": {"name": "Michigan, United States"}},
            flows=[
                {
                    "category": "matter",
                    "direction": (
                        "Michigan → Germany, France, European Union"
                    ),
                    "description": "Pork exported",
                },
            ],
        )
        arrows, warnings = (
            MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        )
        # Two regular country arrows (USA->DEU, USA->FRA) and NO
        # supranational entry — EU was suppressed by the conditional
        # rule.
        assert warnings == []
        assert len(arrows) == 2
        for a in arrows:
            assert "target_supranational" not in a
        targets = sorted(a["direction"].split("→")[-1].strip()
                         for a in arrows)
        assert targets == ["France", "Germany"]

    def test_asean_target_emits_supranational_flow(self):
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "ASEAN"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Brazil → ASEAN",
                    "description": "",
                },
            ],
        )
        arrows, _ = (
            MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        )
        assert len(arrows) == 1
        assert arrows[0]["target_supranational"] == "ASEAN"
        assert len(arrows[0]["target_supranational_members"]) == 10

    def test_nafta_and_usmca_aliases_emit_same_members(self):
        from ._helpers import make_parsed_analysis

        for alias in ("NAFTA", "USMCA"):
            parsed = make_parsed_analysis(
                systems={
                    "sending": {"name": "China"},
                    "receiving": {"name": alias},
                },
                flows=[
                    {
                        "category": "matter",
                        "direction": f"China → {alias}",
                        "description": "",
                    },
                ],
            )
            arrows, _ = (
                MetacouplingAssistant._resolve_flows_for_map(parsed, "CHN")
            )
            assert len(arrows) == 1
            assert sorted(arrows[0]["target_supranational_members"]) == \
                sorted(["USA", "MEX", "CAN"])

    def test_supranational_self_loop_skipped(self):
        """If src is a member of the target supranational, no arrow is
        emitted (would render as a self-loop into its own region)."""
        from ._helpers import make_parsed_analysis

        parsed = make_parsed_analysis(
            systems={
                "sending": {"name": "France"},
                "receiving": {"name": "European Union"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "France → European Union",
                    "description": "",
                },
            ],
        )
        arrows, _ = (
            MetacouplingAssistant._resolve_flows_for_map(parsed, "FRA")
        )
        # France is a member of the EU, so the supranational self-loop
        # is suppressed and we get no arrows.
        assert arrows == []
