from metacouplingllm.core import MetacouplingAssistant
from metacouplingllm.knowledge.literature import (
    _extract_search_terms_from_analysis,
    recommend_papers,
)
from metacouplingllm.llm.parser import CouplingSection, ParsedAnalysis, parse_analysis
from metacouplingllm.output.formatter import AnalysisFormatter
from metacouplingllm.visualization.worldmap import _extract_all_analysis_countries


INTRACOUPLING_ONLY = """
### 1. Coupling Classification
- Intracoupling is present because the study focuses on within-Brazil production dynamics.

### 2. Intracoupling Analysis
#### 2.1 Systems Identification
**Focal System**: Brazil Soybean Production System
- **Human subsystem**: Farmers, traders, domestic regulators
- **Natural subsystem**: Soybean cropland, soils, water resources
- **Geographic scope**: Brazil

#### 2.2 Flows Analysis
- [Matter] Brazil -> Brazil: Soybeans move within the domestic supply chain.

#### 2.3 Agents
- [Firms / traders / corporations] Domestic agribusiness firms: Coordinate production and storage.

#### 2.4 Causes
**Economic**
- Domestic production incentives favor soybean expansion.

#### 2.5 Effects
**Ecological / Biological**
- Land-use pressure increases within the focal system.

### 5. Cross-coupling Interactions
- No cross-scale interactions are analyzed because the current scope stays domestic.

### 6. Research Gaps and Suggestions
- Add finer subnational production data.
- Measure domestic logistics bottlenecks.
- Compare regional environmental outcomes within Brazil.
"""


COUPLING_MIXED = """
### 1. Coupling Classification
- Intracoupling is present within Brazil's production system.
- Telecoupling is present through soybean exports from Brazil to China.

### 2. Intracoupling Analysis
#### 2.1 Systems Identification
**Focal System**: Brazil Soybean Production System
- **Human subsystem**: Farmers and processors
- **Natural subsystem**: Soybean cropland and water resources
- **Geographic scope**: Brazil

#### 2.2 Flows Analysis
- [Matter] Brazil -> Brazil: Domestic movement of soybeans to ports.

#### 2.3 Agents
- [Individuals / households] Farmers: Make production decisions.

#### 2.4 Causes
**Economic**
- Domestic profitability shapes expansion.

#### 2.5 Effects
**Hydrological**
- Water use increases in production regions.

### 4. Telecoupling Analysis
#### 4.1 Systems Identification
**Sending System (distant)**: Brazil Soybean Export System
- **Human subsystem**: Exporters and port operators
- **Natural subsystem**: Export-oriented cropland
- **Geographic scope**: Brazil
**Receiving System (distant)**: China Feed Import System
- **Human subsystem**: Feed manufacturers and importers
- **Natural subsystem**: Livestock production environments
- **Geographic scope**: China
**Spillover System**: Argentina Competing Export System
- **Human subsystem**: Competing traders
- **Natural subsystem**: Agricultural landscapes
- **Geographic scope**: Argentina

#### 4.2 Flows Analysis
- [Matter] Sending system -> receiving systems: Soybeans exported from Brazil to China.
- [Capital] Receiving systems -> sending system: Payments from China to Brazil.

#### 4.3 Agents
- [Firms / traders / corporations] Export traders: Coordinate overseas soybean sales.

#### 4.4 Causes
**Political / Institutional**
- Trade policy influences export competitiveness.

#### 4.5 Effects
**Economic**
- Export revenue increases in Brazil.

### 5. Cross-coupling Interactions
- Export demand amplifies domestic land-use pressure.

### 6. Research Gaps and Suggestions
- Add bilateral trade time series.
- Compare environmental costs across export destinations.
- Evaluate feedbacks between export growth and domestic governance.
"""


ALL_THREE = """
### 1. Coupling Classification
- Intracoupling is present within Brazil.
- Pericoupling is present between Brazil and adjacent South American systems.
- Telecoupling is present between Brazil and China.

### 2. Intracoupling Analysis
#### 2.1 Systems Identification
**Focal System**: Brazil Production System
- **Human subsystem**: Domestic producers
- **Natural subsystem**: Agricultural land
- **Geographic scope**: Brazil

#### 2.2 Flows Analysis
- [Matter] Brazil -> Brazil: Domestic soybean handling.

#### 2.3 Agents
- [Individuals / households] Producers: Expand planted area.

#### 2.4 Causes
**Economic**
- Domestic incentives support production.

#### 2.5 Effects
**Ecological / Biological**
- Domestic habitat conversion increases.

### 3. Pericoupling Analysis
#### 3.1 Systems Identification
**Sending System (adjacent)**: Brazil Border Export Corridor
- **Human subsystem**: Border logistics operators
- **Natural subsystem**: Border ecosystems
- **Geographic scope**: Brazil
**Receiving System (adjacent)**: Paraguay Agricultural Borderlands
- **Human subsystem**: Adjacent traders
- **Natural subsystem**: Shared landscapes
- **Geographic scope**: Paraguay

#### 3.2 Flows Analysis
- [People] Focal system -> adjacent systems: Cross-border logistics labor movement.

#### 3.3 Agents
- [Governments / policymakers] Border agencies: Manage corridor use.

#### 3.4 Causes
**Political / Institutional**
- Border regulations shape regional trade.

#### 3.5 Effects
**Economic**
- Neighboring logistics dependence increases.

### 4. Telecoupling Analysis
#### 4.1 Systems Identification
**Sending System (distant)**: Brazil Export System
- **Human subsystem**: Export traders
- **Natural subsystem**: Export cropland
- **Geographic scope**: Brazil
**Receiving System (distant)**: China Import System
- **Human subsystem**: Import buyers
- **Natural subsystem**: Feed-consuming livestock sectors
- **Geographic scope**: China
**Spillover System**: United States Competing Export System
- **Human subsystem**: Competing exporters
- **Natural subsystem**: Competing cropland
- **Geographic scope**: United States

#### 4.2 Flows Analysis
- [Matter] Brazil -> China: Soybean exports.

#### 4.3 Agents
- [Firms / traders / corporations] Trading firms: Manage exports.

#### 4.4 Causes
**Economic**
- External demand from China drives exports.

#### 4.5 Effects
**Political / Institutional**
- Trade tensions affect competing exporters.

### 5. Cross-coupling Interactions
- Regional and distant demand jointly intensify pressure on the focal system.

### 6. Research Gaps and Suggestions
- Quantify pericoupling spillovers on adjacent borders.
- Compare telecoupling and pericoupling outcomes over time.
- Add flow-specific governance indicators.
"""


def _make_direct_parsed() -> ParsedAnalysis:
    return ParsedAnalysis(
        coupling_classification="Intracoupling within Brazil and telecoupling from Brazil to China.",
        intracoupling=CouplingSection(
            systems=[
                {
                    "role": "focal",
                    "name": "Brazil Soybean Production System",
                    "geographic_scope": "Brazil",
                    "human_subsystem": "Farmers and processors",
                    "natural_subsystem": "Cropland and water resources",
                }
            ],
            flows=[
                {
                    "category": "matter",
                    "direction": "Brazil -> Brazil",
                    "description": "Domestic movement to ports",
                }
            ],
            agents=[{"level": "individuals / households", "name": "Farmers"}],
            causes={"economic": ["Domestic incentives support production."]},
            effects={"hydrological": ["Water use increases within Brazil."]},
        ),
        telecoupling=CouplingSection(
            systems=[
                {
                    "role": "sending",
                    "system_scope": "distant",
                    "name": "Brazil Export System",
                    "geographic_scope": "Brazil",
                },
                {
                    "role": "receiving",
                    "system_scope": "distant",
                    "name": "China Import System",
                    "geographic_scope": "China",
                },
                {
                    "role": "spillover",
                    "name": "Argentina Competing Export System",
                    "geographic_scope": "Argentina",
                },
            ],
            flows=[
                {
                    "category": "matter",
                    "direction": "Sending system -> receiving systems",
                    "description": "Soybeans exported from Brazil to China.",
                },
                {
                    "category": "capital",
                    "direction": "Receiving systems -> sending system",
                    "description": "Payments from China to Brazil.",
                },
            ],
            agents=[{"level": "firms / traders / corporations", "name": "Export traders"}],
            causes={"political / institutional": ["Trade policy influences export competitiveness."]},
            effects={"economic": ["Export revenue increases in Brazil."]},
        ),
        cross_coupling_interactions=[
            "Export demand amplifies domestic land-use pressure."
        ],
        research_gaps=[
            "Add bilateral trade time series.",
            "Compare environmental costs across destinations.",
            "Evaluate feedbacks between export growth and governance.",
        ],
    )


def test_parser_intracoupling_only_structure():
    parsed = parse_analysis(INTRACOUPLING_ONLY)

    assert parsed.intracoupling is not None
    assert parsed.pericoupling is None
    assert parsed.telecoupling is None
    assert parsed.cross_coupling_interactions
    assert len(parsed.research_gaps) == 3
    assert parsed.intracoupling.systems[0]["role"] == "focal"


def test_parser_intracoupling_and_telecoupling_are_isolated():
    parsed = parse_analysis(COUPLING_MIXED)

    assert parsed.intracoupling is not None
    assert parsed.telecoupling is not None
    assert parsed.pericoupling is None
    assert parsed.intracoupling.systems[0]["role"] == "focal"
    tele_roles = {entry["role"] for entry in parsed.telecoupling.systems}
    assert tele_roles == {"sending", "receiving", "spillover"}
    tele_scopes = {
        entry["role"]: entry.get("system_scope")
        for entry in parsed.telecoupling.systems
        if entry["role"] in {"sending", "receiving"}
    }
    assert tele_scopes == {"sending": "distant", "receiving": "distant"}
    tele_directions = [flow["direction"] for flow in parsed.telecoupling.flows]
    assert any("sending system" in direction.lower() for direction in tele_directions)


def test_parser_all_three_coupling_blocks_preserve_internal_schema():
    parsed = parse_analysis(ALL_THREE)

    assert parsed.intracoupling is not None
    assert parsed.pericoupling is not None
    assert parsed.telecoupling is not None
    peri_roles = {entry["role"] for entry in parsed.pericoupling.systems}
    assert peri_roles == {"sending", "receiving"}
    assert all(entry.get("system_scope") == "adjacent" for entry in parsed.pericoupling.systems)
    formatted = AnalysisFormatter.format_full(parsed)
    assert "[Sending System (adjacent)]" in formatted
    assert "[Receiving System (adjacent)]" in formatted
    assert parsed.telecoupling.flows[0]["category"] == "matter"
    assert parsed.telecoupling.effects["political / institutional"] == [
        "Trade tensions affect competing exporters."
    ]


def test_formatter_renders_cross_coupling_and_research_gaps_last():
    parsed = _make_direct_parsed()

    formatted = AnalysisFormatter.format_full(parsed)

    cross_index = formatted.index("5. Cross-coupling Interactions")
    gaps_index = formatted.index("6. Research Gaps and Suggestions")
    assert cross_index < gaps_index
    assert "[Sending System (distant)]" in formatted
    assert "[Receiving System (distant)]" in formatted
    assert "Export demand amplifies domestic land-use pressure." in formatted
    assert "Evaluate feedbacks between export growth and governance." in formatted


def test_worldmap_country_extraction_reads_all_coupling_types():
    parsed = parse_analysis(ALL_THREE)

    role_codes = _extract_all_analysis_countries(parsed)

    assert "BRA" in role_codes["focal"]
    assert "PRY" in role_codes["adjacent"]
    assert "CHN" in role_codes["receiving"]
    assert "USA" in role_codes["spillover"]


def test_flow_resolution_uses_focal_intracoupling_and_receiving_systems():
    parsed = _make_direct_parsed()

    flows = MetacouplingAssistant._resolve_flows_for_map(parsed, "BRA")
    directions = {flow["direction"] for flow in flows}

    assert "Brazil \u2192 China" in directions
    assert "China \u2192 Brazil" in directions


def test_literature_search_terms_aggregate_across_coupling_blocks_and_gaps():
    parsed = _make_direct_parsed()

    terms = _extract_search_terms_from_analysis(parsed)

    assert "brazil" in terms
    assert "china" in terms
    assert "governance" in terms
    assert "competitiveness" in terms


def test_recommend_papers_accepts_coupling_first_analysis():
    papers = recommend_papers(_make_direct_parsed(), max_results=3)

    assert papers
