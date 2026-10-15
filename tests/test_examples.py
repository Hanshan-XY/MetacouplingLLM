"""Tests for knowledge/examples.py — curated examples and relevance matching."""

import re
import zipfile
from pathlib import Path

import metacouplingllm
from metacouplingllm.knowledge.countries import resolve_country_code
from metacouplingllm.knowledge.examples import (
    AGENT_LEVELS,
    ALL_EXAMPLES,
    CATEGORIES,
    COUPLING_TYPES,
    CROSS_COUPLING_TERMS,
    FLOW_CATEGORIES,
    GLOBAL_MARINE_FISHING,
    MICHIGAN_GROUNDWATER_STREAMS,
    SOUTH_AMERICA_FOOD_TRADE,
    SYSTEM_ROLES,
    TRANSATLANTIC_WOOD_PELLETS,
    WOLONG_METACOUPLING,
    format_example,
    get_relevant_examples,
)
from metacouplingllm.knowledge.pericoupling import is_pericoupled
from metacouplingllm.prompts.builder import PromptBuilder

PAPERS_ZIP = Path(metacouplingllm.__file__).parent / "data" / "Papers.zip"


class TestExampleData:
    def test_five_examples_exist(self):
        assert len(ALL_EXAMPLES) == 5

    def test_all_examples_have_required_fields(self):
        for ex in ALL_EXAMPLES:
            for value in (ex.title, ex.source, ex.corpus_file, ex.domain, ex.description):
                assert value, f"{ex.title} has an empty field"
            assert ex.systems and ex.flows and ex.agents and ex.causes and ex.effects

    def test_intracoupling_is_always_classified(self):
        # The output template makes the intracoupling section mandatory.
        for ex in ALL_EXAMPLES:
            assert "intracoupling" in ex.classification, ex.title
            assert set(ex.classification) <= set(COUPLING_TYPES), ex.title

    def test_the_set_covers_every_coupling_type_and_an_omitted_one(self):
        covered = {c for ex in ALL_EXAMPLES for c in ex.coupling_types}
        assert covered == set(COUPLING_TYPES)
        # At least one example shows a study whose pericoupling section is absent.
        assert any("pericoupling" not in ex.classification for ex in ALL_EXAMPLES)

    def test_components_belong_to_a_classified_coupling_type(self):
        for ex in ALL_EXAMPLES:
            for group in (ex.systems, ex.flows, ex.agents, ex.causes, ex.effects):
                for item in group:
                    assert item["coupling"] in ex.classification, (
                        f"{ex.title}: component tagged {item['coupling']} is not classified"
                    )

    def test_each_classified_coupling_has_systems_and_flows(self):
        for ex in ALL_EXAMPLES:
            for coupling in ex.coupling_types:
                assert any(s["coupling"] == coupling for s in ex.systems), (ex.title, coupling)
                assert any(f["coupling"] == coupling for f in ex.flows), (ex.title, coupling)

    def test_systems_have_valid_roles_and_subsystems(self):
        for ex in ALL_EXAMPLES:
            for system in ex.systems:
                assert system["role"] in SYSTEM_ROLES, (ex.title, system["role"])
                is_intra = system["coupling"] == "intracoupling"
                assert (system["role"] == "focal") == is_intra, (
                    f"{ex.title}: only intracoupling systems are focal ({system['name']})"
                )
                for key in ("name", "human_subsystem", "natural_subsystem", "geographic_scope"):
                    assert system.get(key), f"{ex.title}: system missing {key}"

    def test_flows_use_the_six_categories_and_a_direction(self):
        for ex in ALL_EXAMPLES:
            for flow in ex.flows:
                assert flow["category"] in FLOW_CATEGORIES, (ex.title, flow["category"])
                assert "→" in flow["direction"], (ex.title, flow["direction"])
                assert flow["description"]

    def test_agents_have_fixed_levels(self):
        for ex in ALL_EXAMPLES:
            for agent in ex.agents:
                assert agent["level"] in AGENT_LEVELS, (ex.title, agent["level"])
                assert agent["name"] and agent["description"]

    def test_causes_and_effects_use_the_eight_categories(self):
        for ex in ALL_EXAMPLES:
            for cause in ex.causes:
                assert cause["category"] in CATEGORIES, (ex.title, cause["category"])
            for effect in ex.effects:
                assert effect["category"] in CATEGORIES, (ex.title, effect["category"])
                assert effect["system"] and effect["description"]

    def test_cross_coupling_labels_use_the_approved_vocabulary(self):
        terms = tuple(t.lower() for t in CROSS_COUPLING_TERMS)
        for ex in ALL_EXAMPLES:
            assert ex.cross_coupling, f"{ex.title} has no cross-coupling interaction"
            for label, text in ex.cross_coupling:
                assert label.lower().startswith(terms), (ex.title, label)
                assert text


class TestExampleSources:
    def test_every_source_paper_is_in_the_bundled_corpus(self):
        names = set(zipfile.ZipFile(PAPERS_ZIP).namelist())
        for ex in ALL_EXAMPLES:
            assert f"Papers/{ex.corpus_file}.md" in names, ex.corpus_file

    def test_source_reference_matches_the_corpus_file(self):
        for ex in ALL_EXAMPLES:
            author, year, _title = ex.corpus_file.split(" - ", 2)
            surname = author.replace(" et al.", "").split()[-1]
            assert surname in ex.source, (ex.corpus_file, ex.source)
            assert f"({year})" in ex.source, (ex.corpus_file, ex.source)

    def test_country_pericouplings_are_adjacent_in_the_database(self):
        """Flows between named countries agree with the ADM0 adjacency data."""
        checked = 0
        for ex in ALL_EXAMPLES:
            for flow in ex.flows:
                if flow["coupling"] == "intracoupling":
                    continue
                origin, destination = (side.strip() for side in flow["direction"].split("→"))
                for a in re.split(r",? and |, ", origin):
                    for b in re.split(r",? and |, ", destination):
                        if not (resolve_country_code(a) and resolve_country_code(b)):
                            continue
                        expected = flow["coupling"] == "pericoupling"
                        assert (is_pericoupled(a, b) is True) == expected, (ex.title, a, b)
                        checked += 1
        assert checked >= 8  # the trade example's country pairs are all covered


class TestGetRelevantExamples:
    def test_empty_context_returns_the_default_pair(self):
        result = get_relevant_examples("", max_examples=2)
        assert result == [WOLONG_METACOUPLING, SOUTH_AMERICA_FOOD_TRADE]

    def test_max_examples_respected(self):
        result = get_relevant_examples("trade agriculture", max_examples=1)
        assert len(result) == 1

    def test_trade_context_favors_food_trade(self):
        result = get_relevant_examples(
            "international commodity trade and agriculture deforestation", max_examples=1,
        )
        assert result[0] is SOUTH_AMERICA_FOOD_TRADE

    def test_avocado_trade_description_favors_food_trade(self):
        result = get_relevant_examples(
            "Impact of avocado production and trade in Mexico on sustainability", max_examples=1,
        )
        assert result[0] is SOUTH_AMERICA_FOOD_TRADE

    def test_water_context_favors_michigan_streams(self):
        result = get_relevant_examples(
            "urban water supply infrastructure and water scarcity", max_examples=1,
        )
        assert result[0] is MICHIGAN_GROUNDWATER_STREAMS

    def test_conservation_context_favors_wolong(self):
        result = get_relevant_examples(
            "biodiversity conservation in protected areas and wildlife tourism", max_examples=1,
        )
        assert result[0] is WOLONG_METACOUPLING

    def test_fisheries_context_favors_marine_fishing(self):
        result = get_relevant_examples(
            "marine fisheries catch by distant-water fleets", max_examples=1,
        )
        assert result[0] is GLOBAL_MARINE_FISHING

    def test_energy_context_favors_wood_pellets(self):
        result = get_relevant_examples(
            "renewable energy from biomass replacing coal in power plants", max_examples=1,
        )
        assert result[0] is TRANSATLANTIC_WOOD_PELLETS


class TestFormatExample:
    def test_returns_string_with_title_and_source(self):
        text = format_example(WOLONG_METACOUPLING)
        assert isinstance(text, str)
        assert WOLONG_METACOUPLING.title in text
        assert WOLONG_METACOUPLING.source in text

    def test_sections_follow_the_coupling_first_order(self):
        text = format_example(WOLONG_METACOUPLING)
        headings = [
            "**Coupling Classification**",
            "**Intracoupling (within the focal system)**",
            "**Pericoupling (adjacent systems)**",
            "**Telecoupling (distant systems)**",
            "**Cross-coupling Interactions**",
        ]
        positions = [text.index(h) for h in headings]
        assert positions == sorted(positions)

    def test_absent_coupling_type_has_no_section(self):
        text = format_example(TRANSATLANTIC_WOOD_PELLETS)
        assert "Pericoupling (adjacent systems)" not in text
        assert "Telecoupling (distant systems)" in text

    def test_systems_are_labelled_by_role_and_scale(self):
        text = format_example(WOLONG_METACOUPLING)
        for label in (
            "**Focal System**",
            "**Sending System (adjacent)**",
            "**Receiving System (adjacent)**",
            "**Sending System (distant)**",
            "**Receiving System (distant)**",
            "**Spillover System**",
        ):
            assert label in text, label
        for label in ("**Human subsystem**", "**Natural subsystem**", "**Geographic scope**"):
            assert label in text, label

    def test_flows_use_the_template_format(self):
        text = format_example(SOUTH_AMERICA_FOOD_TRADE)
        assert "**Matter Flow**" in text
        assert "- **Direction**: Brazil → China" in text
        assert "- **Description**: " in text

    def test_agent_and_category_labels_match_the_template(self):
        text = format_example(MICHIGAN_GROUNDWATER_STREAMS)
        assert "[Organizations / NGOs]" in text
        assert "**Political / Institutional**" in text
        assert "(applies to: " in text

    def test_adaptation_note_is_shown(self):
        text = format_example(SOUTH_AMERICA_FOOD_TRADE)
        assert "Adaptation: " in text
        assert "Paraguay-Uruguay" in text


class TestPromptIntegration:
    def test_system_prompt_carries_the_selected_examples(self):
        prompt = PromptBuilder(max_examples=2).build_system_prompt(
            research_context="Marine fisheries catch and renewable energy from biomass",
        )
        assert "## REFERENCE EXAMPLES" in prompt
        assert GLOBAL_MARINE_FISHING.title in prompt
        assert TRANSATLANTIC_WOOD_PELLETS.title in prompt
