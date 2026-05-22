"""Tests for llm/parser.py — response parsing."""

from metacouplingllm.llm.parser import ParsedAnalysis, parse_analysis

from ._helpers import (
    legacy_agents,
    legacy_causes,
    legacy_effects,
    legacy_flows,
    legacy_suggestions,
    legacy_systems,
    make_parsed_analysis,
)


# A realistic mock LLM response.
MOCK_RESPONSE = """\
### 1. Coupling Classification

This research involves **telecoupling** between distant coffee-producing and \
coffee-consuming systems. The international coffee trade connects Ethiopian \
farming communities with European consumer markets across large geographic \
distances.

### 2. Telecoupling Analysis

#### 2.1 Systems Identification

- **Sending**: Ethiopian coffee-growing regions (Sidamo, Yirgacheffe). Human \
components: smallholder farmers, cooperatives, export agencies. Natural \
components: highland forest ecosystems, shade-grown coffee agroforestry.

- **Receiving**: European consumer markets (Germany, Italy, UK). Human \
components: importers, roasters, consumers, retail chains. Natural \
components: minimal direct natural component at destination.

- **Spillover**: Other coffee-producing nations (Colombia, Vietnam) affected \
by market competition; neighboring Ethiopian regions affected by land-use \
change.

#### 2.2 Flows Analysis

- [Matter] Ethiopia → Europe: Coffee beans exported from Ethiopian farms to \
European markets
- [Capital] Europe → Ethiopia: Payment for coffee; fair-trade premiums; \
development aid for coffee communities
- [Information] Bidirectional: Market prices, quality standards, \
certification requirements

#### 2.3 Agents

- Smallholder coffee farmers in Ethiopia (sending)
- Coffee cooperatives and unions (sending)
- International coffee traders (intermediary)
- European importers and roasters (receiving)
- Consumers in Europe (receiving)
- Ethiopian and EU government trade agencies (both)

#### 2.4 Causes

**Proximate causes**
- Growing European demand for specialty and single-origin coffee
- Ethiopian coffee's unique flavor profile commanding premium prices

**Underlying causes**
- Historical trade relationships and colonial-era commodity pathways
- Global coffee market liberalization
- Rising consumer interest in ethical sourcing and sustainability

#### 2.5 Effects

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

### 3. Research Gaps and Suggestions

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
        assert len(legacy_systems(result)) > 0
        # Should have sending, receiving, spillover
        keys = set(legacy_systems(result).keys())
        assert "sending" in keys
        assert "receiving" in keys
        assert "spillover" in keys

    def test_parses_flows(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(legacy_flows(result)) >= 3

    def test_flow_has_category(self):
        result = parse_analysis(MOCK_RESPONSE)
        categories = {f.get("category", "") for f in legacy_flows(result)}
        assert "matter" in categories

    def test_parses_agents(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(legacy_agents(result)) >= 4

    def test_parses_causes(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(legacy_causes(result)) > 0

    def test_parses_effects(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(legacy_effects(result)) > 0

    def test_parses_suggestions(self):
        result = parse_analysis(MOCK_RESPONSE)
        assert len(legacy_suggestions(result)) >= 3

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

    def test_evidence_coverage_section_extracted(self):
        """§7 Evidence Coverage prose is extracted into
        ``ParsedAnalysis.evidence_coverage_note`` when present."""
        response = (
            "### 1. Coupling Classification\n\n"
            "Telecoupling.\n\n"
            "### 7. Evidence Coverage\n\n"
            "Strong evidence base: trade volumes from [T1:2].\n\n"
            "Limited evidence: cartel involvement, no source.\n"
        )
        result = parse_analysis(response)
        assert result.evidence_coverage_note
        assert "Strong evidence base" in result.evidence_coverage_note
        assert "Limited evidence" in result.evidence_coverage_note
        assert "cartel involvement" in result.evidence_coverage_note

    def test_evidence_coverage_default_empty_when_section_missing(self):
        """Backward compatibility: responses without §7 give an empty
        ``evidence_coverage_note`` string."""
        result = parse_analysis(MOCK_RESPONSE)
        assert result.evidence_coverage_note == ""


MOCK_MULTILINE_FLOWS = """\
### 1. Coupling Classification

This study involves **telecoupling**.

### 2. Telecoupling Analysis

#### 2.1 Systems Identification

- **Sending**: Ethiopian coffee regions
- **Receiving**: European markets
- **Spillover**: Other coffee origins

#### 2.2 Flows Analysis

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

#### 2.3 Agents

- Ethiopian coffee farmers
- European importers

#### 2.4 Causes

**Proximate causes**
- Growing demand for Ethiopian coffee

#### 2.5 Effects

**Sending system**
- Income for farming communities

### 3. Research Gaps and Suggestions

- Assess environmental footprint
"""


class TestMultilineFlows:
    def test_parses_multiline_flow_categories(self):
        result = parse_analysis(MOCK_MULTILINE_FLOWS)
        assert len(legacy_flows(result)) == 3
        categories = {f.get("category", "") for f in legacy_flows(result)}
        assert "matter" in categories
        assert "capital" in categories
        assert "information" in categories

    def test_parses_multiline_flow_directions(self):
        result = parse_analysis(MOCK_MULTILINE_FLOWS)
        directions = [f.get("direction", "") for f in legacy_flows(result)]
        assert any("Ethiopia" in d and "Europe" in d for d in directions)
        assert any("Europe" in d and "Ethiopia" in d for d in directions)
        assert any("Bidirectional" in d or "bidirectional" in d.lower() for d in directions)

    def test_parses_multiline_flow_descriptions(self):
        result = parse_analysis(MOCK_MULTILINE_FLOWS)
        descriptions = [f.get("description", "") for f in legacy_flows(result)]
        assert any("coffee" in d.lower() for d in descriptions)
        assert any("payment" in d.lower() for d in descriptions)

    def test_multiline_still_parses_other_sections(self):
        result = parse_analysis(MOCK_MULTILINE_FLOWS)
        assert result.is_parsed
        assert "telecoupling" in result.coupling_classification.lower()
        assert len(legacy_systems(result)) >= 2
        assert len(legacy_agents(result)) >= 2
        assert len(legacy_suggestions(result)) >= 1


MOCK_NUMBERED_FLOWS = """\
### 1. Coupling Classification

This study involves **telecoupling**.

### 2. Telecoupling Analysis

#### 2.1 Systems Identification

- **Sending**: Ethiopia
- **Receiving**: European markets

#### 2.2 Flows Analysis

1. **Matter Flow**
- Ethiopia → Europe
- Export of coffee beans from Ethiopia to European markets.

2. **Capital Flow**
- Europe → Ethiopia
- Payment for coffee, potentially including premiums for sustainably sourced coffee.

3. **Information Flow**
- Bidirectional (Ethiopia ↔ Europe)
- Market demand signals, quality standards, and information on sustainable practices.

#### 2.3 Agents

- Ethiopian coffee farmers

#### 2.4 Causes

**Proximate causes**
- Growing demand

#### 2.5 Effects

**Sending system**
- Income for communities

### 3. Research Gaps and Suggestions

- Assess environmental footprint
"""


class TestNumberedFlows:
    """Tests for flows with numbered headings BEFORE bold markers: 1. **Material Flow**."""

    def test_parses_three_flows(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        assert len(legacy_flows(result)) == 3

    def test_parses_categories(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        categories = {f.get("category", "") for f in legacy_flows(result)}
        assert "matter" in categories
        assert "capital" in categories
        assert "information" in categories

    def test_no_unspecified_category(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        for flow in legacy_flows(result):
            cat = flow.get("category", "")
            assert cat != "", f"Flow should have a category: {flow}"
            assert cat.lower() != "unspecified", f"Category should not be Unspecified: {flow}"

    def test_matter_flow_direction(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        mat_flow = [f for f in legacy_flows(result) if f.get("category") == "matter"][0]
        assert "Ethiopia" in mat_flow.get("direction", "")
        assert "Europe" in mat_flow.get("direction", "")

    def test_capital_flow_direction(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        fin_flow = [f for f in legacy_flows(result) if f.get("category") == "capital"][0]
        assert "Europe" in fin_flow.get("direction", "")
        assert "Ethiopia" in fin_flow.get("direction", "")

    def test_information_flow_bidirectional(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        info_flow = [f for f in legacy_flows(result) if f.get("category") == "information"][0]
        direction = info_flow.get("direction", "")
        assert "bidirectional" in direction.lower() or "↔" in direction

    def test_matter_flow_has_description(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        mat_flow = [f for f in legacy_flows(result) if f.get("category") == "matter"][0]
        assert "coffee" in mat_flow.get("description", "").lower()

    def test_capital_flow_has_description(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        fin_flow = [f for f in legacy_flows(result) if f.get("category") == "capital"][0]
        assert "payment" in fin_flow.get("description", "").lower()

    def test_information_flow_has_description(self):
        result = parse_analysis(MOCK_NUMBERED_FLOWS)
        info_flow = [f for f in legacy_flows(result) if f.get("category") == "information"][0]
        assert "market" in info_flow.get("description", "").lower()


MOCK_NESTED_SYSTEMS = """\
### 1. Coupling Classification

This research involves **telecoupling**.

### 2. Telecoupling Analysis

#### 2.1 Systems Identification

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

#### 2.2 Flows Analysis

- [Matter] Ethiopia → Europe: Coffee beans exported

#### 2.3 Agents

- Ethiopian coffee farmers

#### 2.4 Causes

**Proximate causes**
- Growing demand

#### 2.5 Effects

**Sending system**
- Income for communities

### 3. Research Gaps and Suggestions

- Assess environmental footprint
"""


class TestNestedSystems:
    def test_parses_nested_roles(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        assert "sending" in legacy_systems(result)
        assert "receiving" in legacy_systems(result)
        assert "spillover" in legacy_systems(result)

    def test_nested_systems_are_dicts(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        for role in ("sending", "receiving", "spillover"):
            assert isinstance(legacy_systems(result)[role], dict), (
                f"{role} system should be a dict"
            )

    def test_sending_name(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        sending = legacy_systems(result)["sending"]
        assert isinstance(sending, dict)
        assert sending.get("name") == "Ethiopia"

    def test_sending_human_subsystem(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        sending = legacy_systems(result)["sending"]
        assert isinstance(sending, dict)
        assert "farmers" in sending.get("human_subsystem", "").lower()

    def test_sending_natural_subsystem(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        sending = legacy_systems(result)["sending"]
        assert isinstance(sending, dict)
        assert "forest" in sending.get("natural_subsystem", "").lower()

    def test_sending_geographic_scope(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        sending = legacy_systems(result)["sending"]
        assert isinstance(sending, dict)
        assert "sidamo" in sending.get("geographic_scope", "").lower()

    def test_receiving_has_subsystems(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        receiving = legacy_systems(result)["receiving"]
        assert isinstance(receiving, dict)
        assert receiving.get("name") == "European Markets"
        assert "importers" in receiving.get("human_subsystem", "").lower()
        assert "agroecosystem" in receiving.get("natural_subsystem", "").lower()

    def test_spillover_has_subsystems(self):
        result = parse_analysis(MOCK_NESTED_SYSTEMS)
        spillover = legacy_systems(result)["spillover"]
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
        assert "sending" in legacy_systems(result)
        assert "receiving" in legacy_systems(result)
        assert "spillover" in legacy_systems(result)


class TestParsedAnalysis:
    def test_default_values(self):
        pa = ParsedAnalysis()
        assert pa.coupling_classification == ""
        assert legacy_systems(pa) == {}
        assert legacy_flows(pa) == []
        assert legacy_agents(pa) == []
        assert legacy_causes(pa) == {}
        assert legacy_effects(pa) == {}
        assert legacy_suggestions(pa) == []
        assert pa.raw_text == ""
        assert not pa.is_parsed

    def test_is_parsed_with_partial_data(self):
        pa = ParsedAnalysis(coupling_classification="telecoupling")
        assert pa.is_parsed

    def test_get_system_detail_flat_string(self):
        """get_system_detail on flat-format systems returns the string."""
        pa = make_parsed_analysis(systems={"sending": "Brazil soybean regions"})
        assert pa.get_system_detail("sending") == "Brazil soybean regions"
        # sub_field on flat string returns empty
        assert pa.get_system_detail("sending", "human_subsystem") == ""

    def test_get_system_detail_nested_dict(self):
        """get_system_detail on nested-format systems works correctly."""
        pa = make_parsed_analysis(systems={
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

### 2. Telecoupling Analysis

#### 2.1 Systems Identification

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

#### 2.2 Flows Analysis

**1. Material Flows**
- **Direction**: Michigan → China, Japan, Mexico
- **Description**: Pork products exported internationally

**2. Capital Flows**
- **Direction**: China, Japan, Mexico → Michigan
- **Description**: Payment for pork exports

#### 2.3 Agents

- Michigan pork farmers (sending)
- Meatpacking companies (sending)
- International importers (receiving)

#### 2.4 Causes

**Socioeconomic**
- Growing demand for affordable protein in import markets

#### 2.5 Effects

**Biogeochemical**
- Nutrient runoff from concentrated pork production
"""


class TestGPT51SystemParsing:
    """Test parsing of GPT-5.1 style #### **Sending System: Name** headings."""

    def test_parses_all_three_systems(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        assert "sending" in legacy_systems(result)
        assert "receiving" in legacy_systems(result)
        assert "spillover" in legacy_systems(result)

    def test_sending_system_is_nested_dict(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        sending = legacy_systems(result)["sending"]
        assert isinstance(sending, dict)

    def test_sending_system_name(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        sending = legacy_systems(result)["sending"]
        assert isinstance(sending, dict)
        assert "Michigan Pork Production System" in sending.get("name", "")

    def test_sending_has_subsystems(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        sending = legacy_systems(result)["sending"]
        assert isinstance(sending, dict)
        assert "producers" in sending.get("human_subsystem", "").lower()
        assert "agricultural" in sending.get("natural_subsystem", "").lower()

    def test_sending_geographic_scope(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        sending = legacy_systems(result)["sending"]
        assert isinstance(sending, dict)
        assert "Michigan" in sending.get("geographic_scope", "")

    def test_receiving_system_name(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        receiving = legacy_systems(result)["receiving"]
        assert isinstance(receiving, dict)
        assert "International Import Markets" in receiving.get("name", "")

    def test_spillover_system_name(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        spillover = legacy_systems(result)["spillover"]
        assert isinstance(spillover, dict)
        assert "Adjacent Agricultural Regions" in spillover.get("name", "")

    def test_flows_parsed_correctly(self):
        result = parse_analysis(MOCK_GPT51_RESPONSE)
        assert len(legacy_flows(result)) >= 2
        categories = {f.get("category", "") for f in legacy_flows(result)}
        assert "matter" in categories
        assert "capital" in categories


# ---------------------------------------------------------------------------
# PR #34: parser fixes for the fragmented LLM output patterns that
# surfaced in the Mexico avocado live trace (2026-05-22).
# ---------------------------------------------------------------------------


class TestParseFragmentedFlows:
    """The LLM sometimes emits each logical flow as 3 lines (header,
    direction, description) wrapped in top-level bullets.  Without
    PR #34's ``_merge_fragmented_flow_entries`` pass, the parser
    would produce 3 separate flow dicts per logical flow."""

    def test_fragmented_flow_block_merges_into_one_dict(self):
        from metacouplingllm.llm.parser import _parse_flows

        # Format observed in the live Mexico avocado trace.
        text = (
            "- Matter Flow\n"
            "  - Direction: Orchards → Packinghouses\n"
            "  - Description: Avocados moved locally [T1:W3].\n"
            "- Capital Flow\n"
            "  - Direction: Exporter financing → Orchards\n"
            "  - Description: Investments in compliance [T1:W2].\n"
        )
        flows = _parse_flows(text)
        # Two logical flows after merge.
        assert len(flows) == 2
        assert flows[0]["category"] == "matter"
        assert "Orchards → Packinghouses" in flows[0]["direction"]
        assert "Avocados moved locally" in flows[0]["description"]
        assert flows[1]["category"] == "capital"
        assert "Exporter financing" in flows[1]["direction"]
        assert "Investments in compliance" in flows[1]["description"]

    def test_already_clean_flows_pass_through_unchanged(self):
        """Idempotence: well-formed flow blocks must not be
        mis-merged."""
        from metacouplingllm.llm.parser import _parse_flows

        text = (
            "**1. Matter Flow**\n"
            "- **Direction**: A → B\n"
            "- **Description**: Clean test.\n"
            "\n"
            "**2. Capital Flow**\n"
            "- **Direction**: B → A\n"
            "- **Description**: Payments.\n"
        )
        flows = _parse_flows(text)
        assert len(flows) == 2
        assert flows[0]["category"] == "matter"
        assert flows[0]["direction"] == "A → B"
        assert flows[1]["category"] == "capital"

    def test_orphan_subentry_without_header_passes_through(self):
        """A direction/description with no preceding canonical
        header still gets emitted (don't silently drop data)."""
        from metacouplingllm.llm.parser import (
            _merge_fragmented_flow_entries,
        )

        orphan = [
            {"direction": "X → Y", "description": "no header above"}
        ]
        merged = _merge_fragmented_flow_entries(orphan)
        assert merged == orphan

    def test_empty_flows_returns_empty(self):
        from metacouplingllm.llm.parser import (
            _merge_fragmented_flow_entries,
        )

        assert _merge_fragmented_flow_entries([]) == []


class TestParseUnboldCauseEffectCategories:
    """The LLM sometimes emits cause/effect categories as PLAIN-TEXT
    bullets ("- Economic\\n- Strong demand...") instead of bold
    headings.  PR #34 teaches ``_extract_categorized_bullets`` to
    recognise those as section dividers via the existing
    ``_CAUSE_EFFECT_CATEGORY_ALIASES`` table."""

    def test_unbold_categories_split_into_real_buckets(self):
        from metacouplingllm.llm.parser import (
            _extract_categorized_bullets,
        )

        text = (
            "General:\n"
            "- Economic\n"
            "- Strong U.S. demand and price incentives.\n"
            "- Anticipated revenue from market access.\n"
            "- Political / Institutional\n"
            "- Phytosanitary requirements from SENASICA.\n"
            "- Hydrological\n"
            "- Local water availability conditioning yields.\n"
        )
        result = _extract_categorized_bullets(text)
        # The bogus "general" bucket should be gone.
        assert "general" not in result
        # Real categories present with their items.
        assert "economic" in result
        assert len(result["economic"]) == 2
        assert "Strong U.S. demand" in result["economic"][0]
        assert "political / institutional" in result
        assert "Phytosanitary requirements" in result[
            "political / institutional"
        ][0]
        assert "hydrological" in result
        assert (
            "Local water availability"
            in result["hydrological"][0]
        )

    def test_bold_categories_still_work(self):
        """Existing bold-heading format is unchanged."""
        from metacouplingllm.llm.parser import (
            _extract_categorized_bullets,
        )

        text = (
            "**Proximate causes**\n"
            "- Economic incentives.\n"
            "**Distal causes**\n"
            "- Climate change.\n"
        )
        result = _extract_categorized_bullets(text)
        assert "proximate causes" in result
        assert "distal causes" in result

    def test_unbold_category_alias_normalises(self):
        """Common short forms ("Political", "Cultural") normalise
        to the full canonical name via the alias table."""
        from metacouplingllm.llm.parser import (
            _extract_categorized_bullets,
        )

        text = (
            "- Political\n"
            "- A political cause.\n"
            "- Cultural\n"
            "- A cultural cause.\n"
        )
        result = _extract_categorized_bullets(text)
        # Both short forms normalise to their canonical labels.
        assert "political / institutional" in result
        assert "cultural / social / demographic" in result

    def test_non_category_bullets_remain_as_items(self):
        """A bullet whose text doesn't match any known category
        stays as a content item under the current bucket."""
        from metacouplingllm.llm.parser import (
            _extract_categorized_bullets,
        )

        text = (
            "- Economic\n"
            "- This isn't a category name, just prose.\n"
            "- Another item under Economic.\n"
        )
        result = _extract_categorized_bullets(text)
        assert "economic" in result
        assert len(result["economic"]) == 2

    def test_general_bucket_kept_when_no_inline_categories(self):
        """Defensive: when items have NO inline category names,
        the parser still groups them under the default "general"
        key (the old behaviour stays unchanged for this case)."""
        from metacouplingllm.llm.parser import (
            _extract_categorized_bullets,
        )

        text = (
            "- Just some generic cause.\n"
            "- Another generic cause without category labels.\n"
        )
        result = _extract_categorized_bullets(text)
        assert "general" in result
        assert len(result["general"]) == 2
