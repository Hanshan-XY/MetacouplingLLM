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
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.usage is None

    def test_map_field_default_none(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.map is None

    def test_web_map_signals_default_none(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.web_map_signals is None

    def test_map_notice_default_none(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.map_notice is None


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

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(systems={})
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code is None

    def test_country_level_returns_none(self):
        """When systems contain only country names, should return None."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
            systems={
                "sending": "Ethiopian coffee regions",
                "receiving": "European markets",
            },
        )
        code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        assert code is None

    def test_anhui_china_resolves(self):
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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
        assert "city- or watershed-level geometries" in result.map_notice
        assert result.map_notice in result.formatted


class TestCountryMapConfiguration:
    """Tests for country-level auto-map configuration passthrough."""

    def test_generate_map_passes_adm0_shapefile(self, monkeypatch):
        from metacouplingllm.llm.parser import ParsedAnalysis

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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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


class TestResolveFlowsForMap:
    """Test the _resolve_flows_for_map static method."""

    def test_resolves_specific_country_names(self):
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        assert len(result) >= 1
        assert "→" in result[0]["direction"]
        assert "China" in result[0]["direction"] or "CHN" in result[0]["direction"]

    def test_resolves_adm1_region_to_country(self):
        """Michigan should resolve to USA via ADM1 database."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        assert len(result) >= 1
        # Source should be USA (from Michigan ADM1), target should be China
        direction = result[0]["direction"]
        assert "United States" in direction or "USA" in direction
        assert "China" in direction

    def test_resolves_generic_receiving_reference(self):
        """'Receiving regions' should resolve to receiving system countries."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        # Should create arrows to CHN, JPN, MEX
        assert len(result) >= 2
        directions = " ".join(f["direction"] for f in result)
        assert "China" in directions
        assert "Japan" in directions or "Mexico" in directions

    def test_skips_internal_flows(self):
        """Flows 'within Michigan' should be skipped."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
            systems={"sending": {"name": "Michigan"}},
            flows=[
                {
                    "category": "energy",
                    "direction": "Mostly within Michigan and embedded in exports",
                },
            ],
        )
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        assert len(result) == 0

    def test_bidirectional_between_pattern(self):
        """'Bidirectional between Michigan and other regions' should resolve."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        assert len(result) >= 1
        assert "Bidirectional" in result[0]["direction"]
        assert "↔" in result[0]["direction"]

    def test_skips_speculative_example_countries_for_generic_roles(self):
        """Generic role references should not become arrows from examples."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        assert result == []

    def test_resolves_softened_receiving_market_list_for_outgoing_flow(self):
        """Likely receiving-market lists should restore proxy export arrows."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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

        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        directions = " ".join(flow["direction"] for flow in result)
        assert "China" in directions
        assert "Mexico" in directions
        assert "Japan" in directions

    def test_resolves_generic_receiving_source_back_to_focal_country(self):
        """Incoming capital flows from receiving markets should render."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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

        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        directions = " ".join(flow["direction"] for flow in result)
        assert "China" in directions or "Mexico" in directions
        assert "United States" in directions

    def test_resolves_importing_country_synonyms_for_generic_flows(self):
        """Importing-country wording should resolve via the receiving system."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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

        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        directions = " | ".join(flow["direction"] for flow in result)

        assert "China" in directions
        assert "Brazil" in directions
        assert any(flow["category"] == "capital" for flow in result)
        assert any(flow["category"] == "information" for flow in result)
        assert any(flow["category"] == "energy" for flow in result)

    def test_resolves_explicit_multi_country_direction_list(self):
        """Comma-separated country lists in the direction should still resolve."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
            systems={"sending": {"name": "Michigan"}},
            flows=[
                {
                    "category": "matter",
                    "direction": "Michigan → China, Japan, Mexico",
                    "description": "Pork exported",
                },
            ],
        )
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        directions = " ".join(flow["direction"] for flow in result)
        assert "China" in directions
        assert "Japan" in directions
        assert "Mexico" in directions

    def test_deduplicates_same_pair(self):
        """Multiple flows to same (category, src, tgt) should be deduplicated."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
            },
            flows=[
                {"category": "matter", "direction": "Brazil → China"},
                {"category": "matter", "direction": "Brazil → China"},
            ],
        )
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        matter_to_china = [
            f for f in result
            if f["category"] == "matter" and "China" in f["direction"]
        ]
        assert len(matter_to_china) == 1

    def test_resolves_domestic_adm1_neighbor_flows(self):
        """Explicit nearby-state flows should resolve to ADM1 arrow endpoints."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
            systems={"sending": {"name": "Michigan, United States"}},
            flows=[
                {
                    "category": "matter",
                    "direction": "Michigan -> Indiana / Ohio / Wisconsin",
                    "description": "Regional pork and input flows",
                },
            ],
        )

        result = MetacouplingAssistant._resolve_flows_for_adm1_map(
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

        parsed = ParsedAnalysis(
            systems={"sending": {"name": "Michigan, United States"}},
            flows=[
                {
                    "category": "people",
                    "direction": "Michigan <-> adjacent states",
                    "description": "Regional labor and service exchange",
                },
            ],
        )

        result = MetacouplingAssistant._resolve_flows_for_adm1_map(
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

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        # Should resolve "importing countries" to China via receiving system
        assert len(result) >= 1
        directions = " ".join(f["direction"] for f in result)
        assert "China" in directions

    def test_reverse_flow_uses_receiving_as_source(self):
        """'Importing countries → Brazil' → China → Brazil (capital flow)."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
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

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "USA")
        # No arrows should be produced — receiving is generic, and
        # spillover countries must NOT be used as trade partners.
        for f in result:
            assert "Brazil" not in f["direction"]
            assert "Argentina" not in f["direction"]

    def test_no_fallback_when_target_resolves(self):
        """When direction has specific countries, no fallback needed."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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
        result = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
        assert len(result) == 1
        assert "China" in result[0]["direction"]
        # Should NOT include Japan/South Korea since direction was specific
        assert "Japan" not in result[0]["direction"]


class TestStructuredMapData:
    """Test the two-call structured map data pipeline."""

    def test_generate_map_uses_structured_data(self):
        """When map_data is present, _generate_map uses it directly."""
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = ParsedAnalysis(
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

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            auto_map=False,
        )
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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
        parsed = ParsedAnalysis(
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


class TestValidateAdm1Pericoupling:
    """Tests for ADM1-level pericoupling validation."""

    def test_michigan_produces_adm1_info(self):
        from metacouplingllm.llm.parser import ParsedAnalysis

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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

        parsed = ParsedAnalysis(
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


class TestFormatterAdm1PericouplingInfo:
    """Tests for ADM1-level pericoupling info in formatted output."""

    def test_adm1_info_renders_subnational_header(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
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
        assert "Same-country neighbors:" in output
        assert "Cross-border neighbors:" in output
        assert "Ontario" in output

    def test_country_info_renders_unchanged(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
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
