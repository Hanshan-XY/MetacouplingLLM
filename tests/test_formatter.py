"""Tests for output/formatter.py — human-readable formatting."""

from metacouplingllm.llm.parser import ParsedAnalysis
from metacouplingllm.output.formatter import AnalysisFormatter


def _make_sample_analysis() -> ParsedAnalysis:
    """Create a sample ParsedAnalysis for testing (flat systems format)."""
    return ParsedAnalysis(
        coupling_classification="This study involves telecoupling.",
        systems={
            "sending": "Brazil soybean regions",
            "receiving": "China consumer markets",
            "spillover": "Argentina and USA soy producers",
        },
        flows=[
            {"category": "matter", "direction": "Brazil → China", "description": "Soybeans"},
            {"category": "capital", "direction": "China → Brazil", "description": "Payments"},
        ],
        agents=[
            {"level": "individual", "name": "Brazilian farmers", "description": "Smallholder soy farmers"},
            {"level": "company", "name": "Chinese importers", "description": "Food processing companies"},
            {"level": "company", "name": "International traders", "description": "Commodity brokers"},
        ],
        causes={
            "socioeconomic": ["Growing Chinese demand for soybeans", "Economic growth in China"],
            "political": ["Trade liberalization policies"],
        },
        effects={
            "sending": ["Deforestation in Cerrado", "Economic growth in ag sector"],
            "receiving": ["Affordable animal feed"],
        },
        suggestions=[
            "Quantify virtual water flows",
            "Assess spillover effects on Argentina",
        ],
        raw_text="original raw text here",
    )


def _make_nested_analysis() -> ParsedAnalysis:
    """Create a sample ParsedAnalysis with nested systems (dict sub-fields)."""
    return ParsedAnalysis(
        coupling_classification="This study involves telecoupling.",
        systems={
            "sending": {
                "name": "Ethiopia",
                "human_subsystem": "Smallholder farmers, cooperatives, export agencies.",
                "natural_subsystem": "Highland forest ecosystems, shade-grown coffee.",
                "geographic_scope": "Sidamo, Yirgacheffe regions.",
            },
            "receiving": {
                "name": "European Markets",
                "human_subsystem": "Coffee importers, retailers, consumers.",
                "natural_subsystem": "Agroecosystems spared from local cultivation.",
                "geographic_scope": "Germany, Italy, UK.",
            },
            "spillover": {
                "name": "Other Coffee Origins",
                "human_subsystem": "Farmers in Colombia and Vietnam.",
                "natural_subsystem": "Competing agricultural ecosystems.",
                "geographic_scope": "Major coffee-producing regions.",
            },
        },
        flows=[
            {"category": "matter", "direction": "Ethiopia → Europe", "description": "Coffee beans"},
        ],
        agents=[
            {"level": "individual", "name": "Ethiopian farmers"},
            {"level": "company", "name": "European importers"},
        ],
        causes={"cultural": ["Growing demand"]},
        effects={"sending": ["Income for communities"]},
        suggestions=["Assess environmental footprint"],
        raw_text="nested raw text here",
    )


class TestFormatFull:
    def test_returns_string(self):
        analysis = _make_sample_analysis()
        text = AnalysisFormatter.format_full(analysis)
        assert isinstance(text, str)
        assert len(text) > 200

    def test_contains_sections(self):
        text = AnalysisFormatter.format_full(_make_sample_analysis())
        assert "COUPLING CLASSIFICATION" in text
        assert "SYSTEMS IDENTIFICATION" in text
        assert "FLOWS ANALYSIS" in text
        assert "AGENTS" in text
        assert "CAUSES" in text
        assert "EFFECTS" in text
        assert "RESEARCH GAPS & SUGGESTIONS" in text

    def test_unparsed_falls_back_to_raw(self):
        analysis = ParsedAnalysis(raw_text="Just raw text.")
        text = AnalysisFormatter.format_full(analysis)
        assert text == "Just raw text."

    def test_contains_system_roles(self):
        text = AnalysisFormatter.format_full(_make_sample_analysis())
        assert "Sending" in text
        assert "Receiving" in text
        assert "Spillover" in text


class TestFormatSummary:
    def test_returns_string(self):
        text = AnalysisFormatter.format_summary(_make_sample_analysis())
        assert isinstance(text, str)

    def test_contains_summary_info(self):
        text = AnalysisFormatter.format_summary(_make_sample_analysis())
        assert "Classification" in text or "SUMMARY" in text

    def test_unparsed_truncates_raw(self):
        long_raw = "A" * 1000
        analysis = ParsedAnalysis(raw_text=long_raw)
        text = AnalysisFormatter.format_summary(analysis)
        assert len(text) < len(long_raw)
        assert text.endswith("...")


class TestFormatComponent:
    def test_classification(self):
        analysis = _make_sample_analysis()
        text = AnalysisFormatter.format_component(analysis, "classification")
        assert "telecoupling" in text.lower()

    def test_systems(self):
        text = AnalysisFormatter.format_component(
            _make_sample_analysis(), "systems"
        )
        assert "Sending" in text
        assert "Brazil" in text

    def test_flows(self):
        text = AnalysisFormatter.format_component(
            _make_sample_analysis(), "flows"
        )
        assert "Matter" in text
        assert "Soybeans" in text

    def test_agents(self):
        text = AnalysisFormatter.format_component(
            _make_sample_analysis(), "agents"
        )
        assert "Brazilian farmers" in text

    def test_causes(self):
        text = AnalysisFormatter.format_component(
            _make_sample_analysis(), "causes"
        )
        assert "Socioeconomic" in text

    def test_effects(self):
        text = AnalysisFormatter.format_component(
            _make_sample_analysis(), "effects"
        )
        assert "Sending" in text

    def test_suggestions(self):
        text = AnalysisFormatter.format_component(
            _make_sample_analysis(), "suggestions"
        )
        assert "virtual water" in text.lower()

    def test_unknown_component(self):
        text = AnalysisFormatter.format_component(
            _make_sample_analysis(), "nonexistent"
        )
        assert "Unknown component" in text

    def test_empty_component(self):
        analysis = ParsedAnalysis()
        text = AnalysisFormatter.format_component(analysis, "agents")
        assert "No agents data" in text


class TestFormatComparison:
    def test_empty_list(self):
        text = AnalysisFormatter.format_comparison([])
        assert "No analyses" in text

    def test_single_analysis(self):
        analyses = [_make_sample_analysis()]
        text = AnalysisFormatter.format_comparison(analyses)
        # Single analysis should return full format
        assert "METACOUPLING FRAMEWORK ANALYSIS" in text

    def test_multiple_analyses(self):
        a1 = _make_sample_analysis()
        a2 = ParsedAnalysis(
            coupling_classification="Pericoupling study",
            systems={"sending": "System A", "receiving": "System B"},
            flows=[{"category": "information", "direction": "A→B", "description": "Data"}],
            agents=[{"level": "individual", "name": "Agent X"}],
            raw_text="raw2",
        )
        text = AnalysisFormatter.format_comparison([a1, a2])
        assert "Analysis 1" in text
        assert "Analysis 2" in text
        assert "COMPARATIVE ANALYSIS" in text


class TestNestedSystemsFormatting:
    """Tests for formatting analyses with nested system dicts."""

    def test_format_full_nested_systems(self):
        text = AnalysisFormatter.format_full(_make_nested_analysis())
        assert "SYSTEMS IDENTIFICATION" in text
        assert "Sending" in text
        assert "Receiving" in text
        assert "Spillover" in text

    def test_format_full_shows_subsystems(self):
        text = AnalysisFormatter.format_full(_make_nested_analysis())
        assert "Human subsystem:" in text
        assert "Natural subsystem:" in text
        assert "Geographic scope:" in text

    def test_format_full_shows_system_names(self):
        text = AnalysisFormatter.format_full(_make_nested_analysis())
        assert "Ethiopia" in text
        assert "European Markets" in text
        assert "Other Coffee Origins" in text

    def test_format_full_shows_subsystem_details(self):
        text = AnalysisFormatter.format_full(_make_nested_analysis())
        assert "Smallholder farmers" in text
        assert "Highland forest" in text
        assert "importers" in text.lower()

    def test_format_component_nested_systems(self):
        text = AnalysisFormatter.format_component(
            _make_nested_analysis(), "systems"
        )
        assert "Sending" in text
        assert "Ethiopia" in text
        assert "Human subsystem:" in text
        assert "Natural subsystem:" in text

    def test_format_summary_nested_systems(self):
        text = AnalysisFormatter.format_summary(_make_nested_analysis())
        assert "Sending" in text
        assert "Receiving" in text

    def test_format_full_flat_still_works(self):
        """Ensure flat-format systems continue to render correctly."""
        text = AnalysisFormatter.format_full(_make_sample_analysis())
        assert "Brazil soybean regions" in text
        assert "China consumer markets" in text
