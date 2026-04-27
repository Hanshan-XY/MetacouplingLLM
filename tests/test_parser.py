"""Tests for llm/parser.py — response parsing."""

from metacouplingllm.llm.parser import ParsedAnalysis, parse_analysis


# A realistic mock LLM response.
MOCK_RESPONSE = """\
### 1. Coupling Classification

This research involves **telecoupling** between distant coffee-producing and \
coffee-consuming systems. The international coffee trade connects Ethiopian \
farming communities with European consumer markets across large geographic \
distances.

### 2. Systems Identification

- **Sending**: Ethiopian coffee-growing regions (Sidamo, Yirgacheffe). Human \
components: smallholder farmers, cooperatives, export agencies. Natural \
components: highland forest ecosystems, shade-grown coffee agroforestry.

- **Receiving**: European consumer markets (Germany, Italy, UK). Human \
components: importers, roasters, consumers, retail chains. Natural \
components: minimal direct natural component at destination.

- **Spillover**: Other coffee-producing nations (Colombia, Vietnam) affected \
by market competition; neighboring Ethiopian regions affected by land-use \
change.

### 3. Flows Analysis

- [Matter] Ethiopia → Europe: Coffee beans exported from Ethiopian farms to \
European markets
- [Capital] Europe → Ethiopia: Payment for coffee; fair-trade premiums; \
development aid for coffee communities
- [Information] Bidirectional: Market prices, quality standards, \
certification requirements

### 4. Agents

- Smallholder coffee farmers in Ethiopia (sending)
- Coffee cooperatives and unions (sending)
- International coffee traders (intermediary)
- European importers and roasters (receiving)
- Consumers in Europe (receiving)
- Ethiopian and EU government trade agencies (both)

### 5. Causes

**Proximate causes**
- Growing European demand for specialty and single-origin coffee
- Ethiopian coffee's unique flavor profile commanding premium prices

**Underlying causes**
- Historical trade relationships and colonial-era commodity pathways
- Global coffee market liberalization
- Rising consumer interest in ethical sourcing and sustainability

### 6. Effects

**Sending system**
- [Socioeconomic] Income for farming communities; dependence on volatile prices
- [Environmental] Potential deforestation for coffee expansion; but shade-grown \
practices maintain forest cover

**Receiving system**
- [Socioeconomic] Access to diverse, high-quality coffee; cultural significance
- [Environmental] Embodied water and carbon footprint of imported coffee

**Spillover**
- [Socioeconomic] Competitive pressure on other coffee origins
- [Environmental] Potential displacement of production to less sustainable areas

### 7. Research Gaps and Suggestions

- Quantify virtual water and carbon flows embedded in the coffee trade
- Assess feedback effects: how European demand shapes Ethiopian land use
- Investigate spillover effects on neighboring non-coffee communities
- Consider pericoupling with adjacent farming regions within Ethiopia
- Explore how climate change may alter the telecoupling dynamics
"""


class TestParseAnalysis:
    def test_parses_classification(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert "telecoupling" in result.coupling_classification.lower()

    def test_parses_systems(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(result.systems) > 0
        # Should have sending, receiving, spillover
        keys = set(result.systems.keys())
        assert "sending" in keys
        assert "receiving" in keys
        assert "spillover" in keys

    def test_parses_flows(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(result.flows) >= 3

    def test_flow_has_category(self):
        result = parse_analysis(MOCK_RESPONSE)
        categories = {f.get("category", "") for f in result.flows}
        assert "matter" in categories

    def test_parses_agents(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(result.agents) >= 4

    def test_parses_causes(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(result.causes) > 0

    def test_parses_effects(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(result.effects) > 0

    def test_parses_suggestions(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(result.suggestions) >= 3

    def test_raw_text_preserved(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert result.raw_text == MOCK_RESPONSE

    def test_is_parsed_true(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert result.is_parsed

    def test_empty_input(self):
        result = parse_analysis("")
        assert not result.is_parsed
        assert result.raw_text == ""

    def test_unstructured_input(self):
        result = parse_analysis("Just some random text without sections.")
        assert not result.is_parsed
        assert result.raw_text == "Just some random text without sections."


MOCK_MULTILINE_FLOWS = """\
### 1. Coupling Classification

This study involves **telecoupling**.

### 2. Systems Identification

- **Sending**: Ethiopian coffee regions
- **Receiving**: European markets
- **Spillover**: Other coffee origins

### 3. Flows Analysis

**Matter Flow**

- **Direction**: Ethiopia → Europe
- **Description**: Coffee beans and coffee products exported from Ethiopia \
to European markets.

**Capital Flow**

- **Direction**: Europe → Ethiopia
- **Description**: Payments for coffee exports, investments in Ethiopian \
coffee sector, financial incentives for sustainable practices.

**Information Flow**

- **Direction**: Bidirectional
- **Description**: Market information, consumer preferences, sustainability \
standards, agricultural practices.

### 4. Agents

- Ethiopian coffee farmers
- European importers

### 5. Causes

**Proximate causes**
- Growing demand for Ethiopian coffee

### 6. Effects

**Sending system**
- Income for farming communities

### 7. Research Gaps and Suggestions

- Assess environmental footprint
"""


class TestMultilineFlows:
    def test_parses_multiline_flow_categories(self):
        result = parse_analysis(MOCK_MULTILINE_FLOWS)
        assert len(result.flows) == 3
        categories = {f.get("category", "") for f in result.flows}
        assert "matter" in categories
        assert "capital" in categories
        assert "information" in categories

    def test_parses_multiline_flow_directions(self):
        result = parse_analysis(MOCK_MULTILINE_FLOWS)
        directions = [f.get("direction", "") for f in result.flows]
        assert any("Ethiopia" in d and "Europe" in d for d in directions)
        assert any("Europe" in d and "Ethiopia" in d for d in directions)
        assert any("Bidirectional" in d or "bidirectional" in d.lower() for d in directions)

    def test_parses_multiline_flow_descriptions(self):
        result = parse_analysis(MOCK_MULTILINE_FLOWS)
        descriptions = [f.get("description", "") for f in result.flows]
        assert any("coffee" in d.lower() for d in descriptions)
        assert any("payment" in d.lower() for d in descriptions)

    def test_multiline_still_parses_other_sections(self):
        result = parse_analysis(MOCK_MULTILINE_FLOWS)
        assert result.is_parsed
        assert "telecoupling" in result.coupling_classification.lower()
        assert len(result.systems) >= 2
        assert len(result.agents) >= 2
        assert len(result.suggestions) >= 1


MOCK_NUMBERED_FLOWS = """\
### 1. Coupling Classification

This study involves **telecoupling**.

### 2. Systems Identification

- **Sending**: Ethiopia
- **Receiving**: European markets

### 3. Flows Analysis

1. **Matter Flow**
- Ethiopia → Europe
- Export of coffee beans from Ethiopia to European markets.

2. **Capital Flow**
- Europe → Ethiopia
- Payment for coffee, potentially including premiums for sustainably sourced coffee.

3. **Information Flow**
- Bidirectional (Ethiopia ↔ Europe)
- Market demand signals, quality standards, and information on sustainable practices.

### 4. Agents

- Ethiopian coffee farmers

### 5. Causes

**Proximate causes**
- Growing demand

### 6. Effects

**Sending system**
- Income for communities

### 7. Research Gaps and Suggestions

- Assess environmental footprint
"""


class TestNumberedFlows:
    """Tests for flows with numbered headings BEFORE bold markers: 1. **Material Flow**."""

    def test_parses_three_flows(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        assert len(result.flows) == 3

    def test_parses_categories(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        categories = {f.get("category", "") for f in result.flows}
        assert "matter" in categories
        assert "capital" in categories
        assert "information" in categories

    def test_no_unspecified_category(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        for flow in result.flows:
            cat = flow.get("category", "")
            assert cat != "", f"Flow should have a category: {flow}"
            assert cat.lower() != "unspecified", f"Category should not be Unspecified: {flow}"

    def test_matter_flow_direction(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        mat_flow = [f for f in result.flows if f.get("category") == "matter"][0]
        assert "Ethiopia" in mat_flow.get("direction", "")
        assert "Europe" in mat_flow.get("direction", "")

    def test_capital_flow_direction(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        fin_flow = [f for f in result.flows if f.get("category") == "capital"][0]
        assert "Europe" in fin_flow.get("direction", "")
        assert "Ethiopia" in fin_flow.get("direction", "")

    def test_information_flow_bidirectional(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        info_flow = [f for f in result.flows if f.get("category") == "information"][0]
        direction = info_flow.get("direction", "")
        assert "bidirectional" in direction.lower() or "↔" in direction

    def test_matter_flow_has_description(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        mat_flow = [f for f in result.flows if f.get("category") == "matter"][0]
        assert "coffee" in mat_flow.get("description", "").lower()

    def test_capital_flow_has_description(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        fin_flow = [f for f in result.flows if f.get("category") == "capital"][0]
        assert "payment" in fin_flow.get("description", "").lower()

    def test_information_flow_has_description(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        info_flow = [f for f in result.flows if f.get("category") == "information"][0]
        assert "market" in info_flow.get("description", "").lower()


MOCK_NESTED_SYSTEMS = """\
### 1. Coupling Classification

This research involves **telecoupling**.

### 2. Systems Identification

**Sending System**: Ethiopia

- **Human subsystem**: Smallholder coffee farmers, cooperatives, export agencies, \
government trade regulators.
- **Natural subsystem**: Highland forest ecosystems, shade-grown coffee \
agroforestry systems, biodiversity-rich montane forests.
- **Geographic scope**: Sidamo, Yirgacheffe, and Harar regions of Ethiopia.

**Receiving System**: European Markets

- **Human subsystem**: Coffee importers, retailers, consumers, trade regulators.
- **Natural subsystem**: Agroecosystems in Europe spared from local coffee \
cultivation.
- **Geographic scope**: Various European countries involved in importing Ethiopian \
coffee.

**Spillover System**: Other Coffee Origins

- **Human subsystem**: Coffee farmers in Colombia, Vietnam, and other exporting \
nations affected by market competition.
- **Natural subsystem**: Forest and agricultural ecosystems in competing regions \
experiencing land-use pressure.
- **Geographic scope**: Major global coffee-producing regions outside Ethiopia.

### 3. Flows Analysis

- [Matter] Ethiopia → Europe: Coffee beans exported

### 4. Agents

- Ethiopian coffee farmers

### 5. Causes

**Proximate causes**
- Growing demand

### 6. Effects

**Sending system**
- Income for communities

### 7. Research Gaps and Suggestions

- Assess environmental footprint
"""


class TestNestedSystems:
    def test_parses_nested_roles(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        assert "sending" in result.systems
        assert "receiving" in result.systems
        assert "spillover" in result.systems

    def test_nested_systems_are_dicts(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        for role in ("sending", "receiving", "spillover"):
            assert isinstance(result.systems[role], dict), (
                f"{role} system should be a dict"
            )

    def test_sending_name(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        sending = result.systems["sending"]
        assert isinstance(sending, dict)
        assert sending.get("name") == "Ethiopia"

    def test_sending_human_subsystem(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        sending = result.systems["sending"]
        assert isinstance(sending, dict)
        assert "farmers" in sending.get("human_subsystem", "").lower()

    def test_sending_natural_subsystem(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        sending = result.systems["sending"]
        assert isinstance(sending, dict)
        assert "forest" in sending.get("natural_subsystem", "").lower()

    def test_sending_geographic_scope(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        sending = result.systems["sending"]
        assert isinstance(sending, dict)
        assert "sidamo" in sending.get("geographic_scope", "").lower()

    def test_receiving_has_subsystems(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        receiving = result.systems["receiving"]
        assert isinstance(receiving, dict)
        assert receiving.get("name") == "European Markets"
        assert "importers" in receiving.get("human_subsystem", "").lower()
        assert "agroecosystem" in receiving.get("natural_subsystem", "").lower()

    def test_spillover_has_subsystems(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        spillover = result.systems["spillover"]
        assert isinstance(spillover, dict)
        assert "Other Coffee Origins" in spillover.get("name", "")
        assert "colombia" in spillover.get("human_subsystem", "").lower()

    def test_get_system_detail_name(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        assert result.get_system_detail("sending", "name") == "Ethiopia"

    def test_get_system_detail_subsystem(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        human = result.get_system_detail("receiving", "human_subsystem")
        assert "importers" in human.lower()

    def test_get_system_detail_summary(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        summary = result.get_system_detail("sending")
        assert "Ethiopia" in summary
        assert "Human subsystem:" in summary
        assert "Natural subsystem:" in summary

    def test_get_system_detail_missing_role(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        assert result.get_system_detail("nonexistent") == ""

    def test_get_system_detail_missing_subfield(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        assert result.get_system_detail("sending", "nonexistent_field") == ""

    def test_flat_systems_still_work(self):
        """Ensure the original flat format still parses correctly."""
        result = parse_analysis(MOCK_RESPONSE)
        assert "sending" in result.systems
        assert "receiving" in result.systems
        assert "spillover" in result.systems


class TestParsedAnalysis:
    def test_default_values(self):
        pa = ParsedAnalysis()
        assert pa.coupling_classification == ""
        assert pa.systems == {}
        assert pa.flows == []
        assert pa.agents == []
        assert pa.causes == {}
        assert pa.effects == {}
        assert pa.suggestions == []
        assert pa.raw_text == ""
        assert not pa.is_parsed

    def test_is_parsed_with_partial_data(self):
        pa = ParsedAnalysis(coupling_classification="telecoupling")
        assert pa.is_parsed

    def test_get_system_detail_flat_string(self):
        """get_system_detail on flat-format systems returns the string."""
        pa = ParsedAnalysis(systems={"sending": "Brazil soybean regions"})
        assert pa.get_system_detail("sending") == "Brazil soybean regions"
        # sub_field on flat string returns empty
        assert pa.get_system_detail("sending", "human_subsystem") == ""

    def test_get_system_detail_nested_dict(self):
        """get_system_detail on nested-format systems works correctly."""
        pa = ParsedAnalysis(systems={
            "sending": {
                "name": "Ethiopia",
                "human_subsystem": "farmers",
                "natural_subsystem": "forests",
                "geographic_scope": "Sidamo region",
            }
        })
        assert pa.get_system_detail("sending", "name") == "Ethiopia"
        assert pa.get_system_detail("sending", "human_subsystem") == "farmers"
        summary = pa.get_system_detail("sending")
        assert "Ethiopia" in summary
        assert "farmers" in summary


# GPT-5.1-style response with #### headings and colon inside **...**
MOCK_GPT51_RESPONSE = """\
### 1. Coupling Classification

This research examines the **telecoupling** involved in Michigan's pork exports.

### 2. Systems Identification

#### **Sending System: Michigan Pork Production System**

- **Human Subsystem**: Pork producers, meatpackers, export agencies
- **Natural Subsystem**: Agricultural lands, water resources, feed crops
- **Geographic Scope**: Michigan, United States

#### **Receiving System: International Import Markets**

- **Human Subsystem**: Importers, retailers, consumers in China, Japan, Mexico
- **Natural Subsystem**: Local ecosystems impacted by increased demand
- **Geographic Scope**: China, Japan, Mexico

#### **Spillover System: Adjacent Agricultural Regions**

- **Human Subsystem**: Neighboring state farmers, regional suppliers
- **Natural Subsystem**: Great Lakes ecosystem, shared watersheds
- **Geographic Scope**: Ohio, Indiana, Wisconsin

### 3. Flows Analysis

**1. Material Flows**
- **Direction**: Michigan → China, Japan, Mexico
- **Description**: Pork products exported internationally

**2. Capital Flows**
- **Direction**: China, Japan, Mexico → Michigan
- **Description**: Payment for pork exports

### 4. Agents

- Michigan pork farmers (sending)
- Meatpacking companies (sending)
- International importers (receiving)

### 5. Causes

**Socioeconomic**
- Growing demand for affordable protein in import markets

### 6. Effects

**Biogeochemical**
- Nutrient runoff from concentrated pork production
"""


class TestGPT51SystemParsing:
    """Test parsing of GPT-5.1 style #### **Sending System: Name** headings."""

    def test_parses_all_three_systems(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        assert "sending" in result.systems
        assert "receiving" in result.systems
        assert "spillover" in result.systems

    def test_sending_system_is_nested_dict(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        sending = result.systems["sending"]
        assert isinstance(sending, dict)

    def test_sending_system_name(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        sending = result.systems["sending"]
        assert isinstance(sending, dict)
        assert "Michigan Pork Production System" in sending.get("name", "")

    def test_sending_has_subsystems(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        sending = result.systems["sending"]
        assert isinstance(sending, dict)
        assert "producers" in sending.get("human_subsystem", "").lower()
        assert "agricultural" in sending.get("natural_subsystem", "").lower()

    def test_sending_geographic_scope(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        sending = result.systems["sending"]
        assert isinstance(sending, dict)
        assert "Michigan" in sending.get("geographic_scope", "")

    def test_receiving_system_name(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        receiving = result.systems["receiving"]
        assert isinstance(receiving, dict)
        assert "International Import Markets" in receiving.get("name", "")

    def test_spillover_system_name(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        spillover = result.systems["spillover"]
        assert isinstance(spillover, dict)
        assert "Adjacent Agricultural Regions" in spillover.get("name", "")

    def test_flows_parsed_correctly(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        assert len(result.flows) >= 2
        categories = {f.get("category", "") for f in result.flows}
        assert "matter" in categories
        assert "capital" in categories
