"""Tests for knowledge/framework.py — enums, definitions, and knowledge formatter."""

from metacouplingllm.knowledge.framework import (
    COMPONENT_DEFINITIONS,
    TELECOUPLING_CATEGORIES,
    AgentLevel,
    CauseCategory,
    CouplingType,
    EffectCategory,
    FlowCategory,
    FrameworkComponent,
    SystemRole,
    get_framework_knowledge,
)


class TestCouplingType:
    def test_values(self):
        assert CouplingType.INTRACOUPLING.value == "intracoupling"
        assert CouplingType.PERICOUPLING.value == "pericoupling"
        assert CouplingType.TELECOUPLING.value == "telecoupling"

    def test_has_descriptions(self):
        for ct in CouplingType:
            assert ct.description, f"{ct} has no description"
            assert len(ct.description) > 20

    def test_is_str_enum(self):
        # CouplingType extends str, so it can be used as a string
        assert "telecoupling" in CouplingType.TELECOUPLING


class TestSystemRole:
    def test_values(self):
        assert SystemRole.SENDING.value == "sending"
        assert SystemRole.RECEIVING.value == "receiving"
        assert SystemRole.SPILLOVER.value == "spillover"

    def test_has_descriptions(self):
        for sr in SystemRole:
            assert sr.description, f"{sr} has no description"


class TestFlowCategory:
    def test_all_categories(self):
        expected = {"capital", "energy", "information", "matter", "organisms", "people"}
        actual = {fc.value for fc in FlowCategory}
        assert actual == expected

    def test_has_descriptions(self):
        for fc in FlowCategory:
            assert fc.description


class TestAgentLevel:
    def test_all_levels(self):
        expected = {
            "individuals / households",
            "firms / traders / corporations",
            "governments / policymakers",
            "organizations / NGOs",
            "non-human agents",
        }
        actual = {al.value for al in AgentLevel}
        assert actual == expected

    def test_has_descriptions(self):
        for al in AgentLevel:
            assert al.description, f"{al} has no description"


class TestCauseCategory:
    def test_all_categories(self):
        expected = {
            "economic",
            "political / institutional",
            "ecological / biological",
            "technological / infrastructural",
            "cultural / social / demographic",
            "hydrological",
            "climatic / atmospheric",
            "geological / geomorphological",
        }
        actual = {cc.value for cc in CauseCategory}
        assert actual == expected

    def test_has_descriptions(self):
        for cc in CauseCategory:
            assert cc.description, f"{cc} has no description"


class TestEffectCategory:
    def test_matches_cause_categories(self):
        assert {ec.value for ec in EffectCategory} == {
            cc.value for cc in CauseCategory
        }

    def test_has_descriptions(self):
        for ec in EffectCategory:
            assert ec.description, f"{ec} has no description"


class TestFrameworkComponent:
    def test_five_components(self):
        assert len(FrameworkComponent) == 5
        assert FrameworkComponent.SYSTEMS.value == "systems"
        assert FrameworkComponent.FLOWS.value == "flows"
        assert FrameworkComponent.AGENTS.value == "agents"
        assert FrameworkComponent.CAUSES.value == "causes"
        assert FrameworkComponent.EFFECTS.value == "effects"


class TestComponentDefinitions:
    def test_all_components_have_definitions(self):
        for comp in FrameworkComponent:
            assert comp in COMPONENT_DEFINITIONS, f"Missing definition for {comp}"

    def test_definitions_have_content(self):
        for comp, defn in COMPONENT_DEFINITIONS.items():
            assert defn.name
            assert defn.description
            assert len(defn.sub_elements) > 0, f"{comp} has no sub-elements"
            assert len(defn.guiding_questions) > 0, f"{comp} has no questions"


class TestTelecouplingCategories:
    def test_fourteen_categories(self):
        assert len(TELECOUPLING_CATEGORIES) == 14

    def test_categories_have_content(self):
        for cat in TELECOUPLING_CATEGORIES:
            assert cat.name
            assert cat.description
            assert cat.example
            assert len(cat.keywords) > 0

    def test_known_categories_present(self):
        names = {cat.name for cat in TELECOUPLING_CATEGORIES}
        expected_subset = {"Trade", "Tourism", "Water transfer", "Species invasion"}
        assert expected_subset.issubset(names)


class TestGetFrameworkKnowledge:
    def test_returns_nonempty_string(self):
        text = get_framework_knowledge()
        assert isinstance(text, str)
        assert len(text) > 1000  # Should be substantial

    def test_contains_key_sections(self):
        text = get_framework_knowledge()
        assert "METACOUPLING FRAMEWORK OVERVIEW" in text
        assert "FIVE FRAMEWORK COMPONENTS" in text
        assert "TELECOUPLING CATEGORIES" in text
        assert "KEY ANALYTICAL INSIGHTS" in text

    def test_contains_operationalization(self):
        text = get_framework_knowledge()
        assert "Operationalization Procedure" in text
        assert "Set research goals" in text

    def test_contains_coupling_transformations(self):
        text = get_framework_knowledge()
        assert "Coupling Transformations" in text
        assert "Noncoupling" in text or "noncoupling" in text
        assert "Decoupling" in text or "decoupling" in text
        assert "Recoupling" in text or "recoupling" in text

    def test_contains_cascading_interactions(self):
        text = get_framework_knowledge()
        assert "Cascading Interactions" in text

    def test_contains_coupling_types(self):
        text = get_framework_knowledge()
        assert "Intracoupling" in text
        assert "Pericoupling" in text
        assert "Telecoupling" in text

    def test_contains_all_14_categories(self):
        text = get_framework_knowledge()
        for cat in TELECOUPLING_CATEGORIES:
            assert cat.name in text, f"Category '{cat.name}' not in knowledge text"
