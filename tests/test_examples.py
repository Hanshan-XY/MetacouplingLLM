"""Tests for knowledge/examples.py — curated examples and relevance matching."""

from metacouplingllm.knowledge.examples import (
    ALL_EXAMPLES,
    BEIJING_WATER,
    PANDA_SDGS,
    RIFA_INVASION,
    SOYBEAN_TRADE,
    WOLONG_METACOUPLING,
    format_example,
    get_relevant_examples,
)


class TestExampleData:
    def test_five_examples_exist(self):
        assert len(ALL_EXAMPLES) == 5

    def test_all_examples_have_required_fields(self):
        for ex in ALL_EXAMPLES:
            assert ex.title
            assert ex.source
            assert ex.domain
            assert ex.coupling_type in ("telecoupling", "metacoupling")
            assert ex.description
            assert len(ex.systems) > 0
            assert len(ex.flows) > 0
            assert len(ex.agents) > 0
            assert len(ex.causes) > 0
            assert len(ex.effects) > 0

    def test_soybean_is_telecoupling(self):
        assert SOYBEAN_TRADE.coupling_type == "telecoupling"
        assert "sending" in SOYBEAN_TRADE.systems
        assert "receiving" in SOYBEAN_TRADE.systems
        assert "spillover" in SOYBEAN_TRADE.systems

    def test_panda_is_metacoupling(self):
        assert PANDA_SDGS.coupling_type == "metacoupling"

    def test_all_examples_have_nested_systems(self):
        """All examples use nested dict format for systems."""
        for ex in ALL_EXAMPLES:
            for role, value in ex.systems.items():
                assert isinstance(value, dict), (
                    f"{ex.title} system '{role}' should be a dict"
                )
                assert "name" in value, (
                    f"{ex.title} system '{role}' should have a 'name' key"
                )
                assert "human_subsystem" in value, (
                    f"{ex.title} system '{role}' should have 'human_subsystem'"
                )
                assert "natural_subsystem" in value, (
                    f"{ex.title} system '{role}' should have 'natural_subsystem'"
                )
                assert "geographic_scope" in value, (
                    f"{ex.title} system '{role}' should have 'geographic_scope'"
                )

    def test_spillover_systems_have_detail(self):
        """Every example's spillover system has non-empty subsystem fields."""
        for ex in ALL_EXAMPLES:
            spillover = ex.systems.get("spillover")
            assert spillover is not None, f"{ex.title} missing spillover system"
            assert isinstance(spillover, dict)
            assert spillover.get("human_subsystem"), (
                f"{ex.title} spillover should have human_subsystem"
            )
            assert spillover.get("natural_subsystem"), (
                f"{ex.title} spillover should have natural_subsystem"
            )
            assert spillover.get("geographic_scope"), (
                f"{ex.title} spillover should have geographic_scope"
            )

    def test_flows_have_required_keys(self):
        for ex in ALL_EXAMPLES:
            for flow in ex.flows:
                assert "category" in flow
                assert "direction" in flow
                assert "description" in flow

    def test_agents_have_fixed_categories(self):
        valid_categories = {
            "individuals / households",
            "firms / traders / corporations",
            "governments / policymakers",
            "organizations / NGOs",
            "non-human agents",
        }
        for ex in ALL_EXAMPLES:
            for agent in ex.agents:
                assert "level" in agent, f"{ex.title} agent missing 'level' key"
                assert agent["level"] in valid_categories, (
                    f"{ex.title} agent has invalid level: {agent['level']}"
                )

    def test_causes_have_category(self):
        valid_categories = {
            "economic",
            "political / institutional",
            "ecological / biological",
            "technological / infrastructural",
            "cultural / social / demographic",
            "hydrological",
            "climatic / atmospheric",
            "geological / geomorphological",
        }
        for ex in ALL_EXAMPLES:
            for cause in ex.causes:
                assert "category" in cause, f"{ex.title} cause missing 'category' key"
                assert cause["category"] in valid_categories, (
                    f"{ex.title} cause has invalid category: {cause['category']}"
                )

    def test_effects_have_system_and_type(self):
        valid_categories = {
            "economic",
            "political / institutional",
            "ecological / biological",
            "technological / infrastructural",
            "cultural / social / demographic",
            "hydrological",
            "climatic / atmospheric",
            "geological / geomorphological",
        }
        for ex in ALL_EXAMPLES:
            for effect in ex.effects:
                assert "system" in effect
                assert "type" in effect
                assert "description" in effect
                assert effect["type"] in valid_categories, (
                    f"{ex.title} effect has invalid type: {effect['type']}"
                )


class TestGetRelevantExamples:
    def test_empty_context_returns_defaults(self):
        result = get_relevant_examples("", max_examples=2)
        assert len(result) == 2

    def test_max_examples_respected(self):
        result = get_relevant_examples("trade agriculture", max_examples=1)
        assert len(result) == 1

    def test_trade_context_favors_soybean(self):
        result = get_relevant_examples(
            "international commodity trade and agriculture deforestation",
            max_examples=1,
        )
        assert result[0].title == SOYBEAN_TRADE.title

    def test_water_context_favors_beijing(self):
        result = get_relevant_examples(
            "urban water supply infrastructure and water scarcity",
            max_examples=1,
        )
        assert result[0].title == BEIJING_WATER.title

    def test_conservation_context_favors_panda(self):
        result = get_relevant_examples(
            "biodiversity conservation sustainable development goals SDG",
            max_examples=1,
        )
        assert result[0].title == PANDA_SDGS.title

    def test_invasive_species_context_favors_rifa(self):
        result = get_relevant_examples(
            "invasive species biological invasion pest biosecurity",
            max_examples=1,
        )
        assert result[0].title == RIFA_INVASION.title


class TestFormatExample:
    def test_returns_string(self):
        text = format_example(SOYBEAN_TRADE)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_contains_title_and_source(self):
        text = format_example(SOYBEAN_TRADE)
        assert SOYBEAN_TRADE.title in text
        assert SOYBEAN_TRADE.source in text

    def test_contains_all_sections(self):
        text = format_example(BEIJING_WATER)
        assert "Systems" in text
        assert "Flows" in text
        assert "Agents" in text
        assert "Causes" in text
        assert "Effects" in text

    def test_format_nested_systems(self):
        """format_example renders nested system dicts with subsystem labels."""
        text = format_example(SOYBEAN_TRADE)
        assert "Sending System" in text
        assert "Receiving System" in text
        assert "Spillover System" in text
        assert "Human subsystem" in text
        assert "Natural subsystem" in text
        assert "Geographic scope" in text

    def test_format_spillover_subsystem_content(self):
        """Spillover system details appear in formatted output."""
        text = format_example(SOYBEAN_TRADE)
        assert "Argentina" in text  # spillover human subsystem content
