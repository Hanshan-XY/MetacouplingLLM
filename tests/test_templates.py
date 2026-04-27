"""Tests for prompts/templates.py — prompt template content and placeholders."""

from metacouplingllm.prompts.templates import (
    CITATION_RULES_LAYER,
    EXAMPLES_LAYER,
    INITIAL_USER_TEMPLATE,
    INTERACTION_LAYER,
    KNOWLEDGE_LAYER,
    METHODOLOGY_LAYER,
    OUTPUT_FORMAT_LAYER,
    REFINEMENT_USER_TEMPLATE,
    REFINEMENT_WITH_FOCUS_TEMPLATE,
    ROLE_LAYER,
)


class TestRoleLayer:
    def test_nonempty(self):
        assert len(ROLE_LAYER) > 100

    def test_mentions_framework(self):
        assert "metacoupling" in ROLE_LAYER.lower()
        assert "telecoupling" in ROLE_LAYER.lower()

    def test_mentions_five_components(self):
        assert "systems" in ROLE_LAYER.lower()
        assert "flows" in ROLE_LAYER.lower()
        assert "agents" in ROLE_LAYER.lower()
        assert "causes" in ROLE_LAYER.lower()
        assert "effects" in ROLE_LAYER.lower()


class TestKnowledgeLayer:
    def test_has_placeholder(self):
        assert "{framework_definitions}" in KNOWLEDGE_LAYER

    def test_fillable(self):
        filled = KNOWLEDGE_LAYER.format(framework_definitions="<definitions>")
        assert "<definitions>" in filled
        assert "{framework_definitions}" not in filled


class TestMethodologyLayer:
    def test_contains_steps(self):
        assert "Classify the coupling" in METHODOLOGY_LAYER
        assert "Identify systems" in METHODOLOGY_LAYER
        assert "Map flows" in METHODOLOGY_LAYER

    def test_walks_by_coupling_type(self):
        """Methodology must instruct the LLM to repeat the per-component
        analysis once per active coupling type."""
        text = METHODOLOGY_LAYER.lower()
        assert "intracoupling" in text
        assert "pericoupling" in text
        assert "telecoupling" in text
        # Per-coupling-type sub-steps a-e exist
        assert "a. **identify systems**" in text
        assert "b. **map flows**" in text
        assert "c. **identify agents**" in text
        assert "d. **analyze causes**" in text
        assert "e. **assess effects**" in text

    def test_mentions_intracoupling_always_present(self):
        """Intracoupling is always part of any metacoupling analysis."""
        assert "always present" in METHODOLOGY_LAYER.lower()

    def test_mentions_cross_coupling_interactions(self):
        """A separate cross-coupling interactions analysis step must appear."""
        text = METHODOLOGY_LAYER.lower()
        assert "cross-coupling" in text or "interactions" in text

    def test_mentions_duplication_across_scales(self):
        """When a flow spans multiple coupling types, it must appear under
        every applicable section — duplication is intentional."""
        text = METHODOLOGY_LAYER.lower()
        assert "duplication is intentional" in text


class TestOutputFormatLayerCouplingStructure:
    def test_intracoupling_section_present(self):
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "intracoupling analysis" in text
        # §2 is always present per the spec
        assert "always present" in text

    def test_pericoupling_section_conditional(self):
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "pericoupling analysis" in text
        assert "include this section only if pericoupling is present" in text

    def test_telecoupling_section_conditional(self):
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "telecoupling analysis" in text
        assert "include this section only if telecoupling is present" in text

    def test_omit_placeholder_instruction(self):
        """When a coupling type isn't present, the section must be OMITTED
        entirely — no empty placeholder header allowed."""
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "omit" in text
        assert "placeholder" in text

    def test_telecoupling_section_keeps_spillover_rigor(self):
        """The spillover-rigor warning must still appear inside the
        telecoupling section after the restructure."""
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "spillover" in text
        assert "do not leave the spillover system vague" in text

    def test_fixed_category_grouping_inside_each_block(self):
        """Causes/Effects keep the fixed category grouping inside each
        coupling-type block."""
        text = OUTPUT_FORMAT_LAYER.lower()
        for cat in (
            "economic",
            "political / institutional",
            "ecological / biological",
            "technological / infrastructural",
            "cultural / social / demographic",
            "hydrological",
            "climatic / atmospheric",
            "geological / geomorphological",
        ):
            assert f"**{cat}**" in text
        assert "fixed eight-category grouping" in text

    def test_duplication_across_types_intentional(self):
        """A flow/agent/cause/effect that spans multiple coupling types
        should appear under every applicable section."""
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "duplication is intentional" in text

    def test_cross_coupling_interactions_section_present(self):
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "cross-coupling interactions" in text

    def test_section_6_research_gaps_marked_mandatory(self):
        """§6 was being dropped by the LLM when the analysis ran long.
        The directive must mark §6 as MANDATORY and require at least
        3 specific gaps so the LLM doesn't wrap up after §5."""
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "research gaps" in text
        assert "mandatory" in text and "§6" in OUTPUT_FORMAT_LAYER
        # Numeric requirement keeps the LLM honest if it tries to
        # phone in a single bullet
        assert "at least 3" in text

    def test_output_completeness_checklist_present(self):
        """A self-check checklist at the END of the spec is the most
        reliable enforcement for §6 — even MANDATORY language inline
        gets ignored when the LLM treats §5 as the natural conclusion."""
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "completeness checklist" in text or "checklist" in text
        # The checklist must explicitly list §6 as ALWAYS required
        assert "§6 research gaps" in text
        assert "always" in text
        # Common-failure-mode reminder ("don't wrap up after §5")
        assert "wrap up after §5" in text or "stop there" in text


class TestExamplesLayer:
    def test_has_placeholder(self):
        assert "{formatted_examples}" in EXAMPLES_LAYER


class TestOutputFormatLayer:
    def test_contains_all_sections(self):
        text = OUTPUT_FORMAT_LAYER.lower()
        assert "coupling classification" in text
        assert "systems identification" in text
        assert "flows analysis" in text
        assert "agents" in text
        assert "causes" in text
        assert "effects" in text
        assert "research gaps" in text or "suggestions" in text


class TestInteractionLayer:
    def test_mentions_multi_turn(self):
        text = INTERACTION_LAYER.lower()
        assert "first turn" in text or "follow-up" in text
        assert "refine" in text or "refinement" in text


class TestCitationRulesLayer:
    def test_nonempty(self):
        assert len(CITATION_RULES_LAYER) > 200

    def test_mentions_brackets(self):
        # The rules should reference the [N] citation syntax explicitly
        assert "[1]" in CITATION_RULES_LAYER
        assert "[2]" in CITATION_RULES_LAYER

    def test_mentions_recency(self):
        # The "cite from the most recent block" rule must be present
        # so the LLM doesn't reuse stale citation numbers across turns
        text = CITATION_RULES_LAYER.lower()
        assert "most recent" in text
        assert "previous turn" in text or "turn-local" in text

    def test_mentions_no_invent(self):
        # Forbids fabricating citations
        text = CITATION_RULES_LAYER.lower()
        assert "never invent" in text or "do not fabricate" in text

    def test_mentions_retrieved_literature_block(self):
        # Should reference the XML block name so the LLM knows where to look
        assert "retrieved_literature" in CITATION_RULES_LAYER

    def test_mentions_empty_block_handling(self):
        # The rule for empty <retrieved_literature/> blocks must exist
        text = CITATION_RULES_LAYER.lower()
        assert "empty" in text
        assert "no relevant passages" in text or "no specific evidence" in text


class TestUserTemplates:
    def test_initial_has_placeholder(self):
        assert "{research_description}" in INITIAL_USER_TEMPLATE

    def test_initial_fillable(self):
        filled = INITIAL_USER_TEMPLATE.format(
            research_description="My coffee trade study"
        )
        assert "My coffee trade study" in filled

    def test_refinement_has_placeholder(self):
        assert "{additional_info}" in REFINEMENT_USER_TEMPLATE

    def test_refinement_with_focus_has_both_placeholders(self):
        assert "{focus_component}" in REFINEMENT_WITH_FOCUS_TEMPLATE
        assert "{additional_info}" in REFINEMENT_WITH_FOCUS_TEMPLATE

    def test_refinement_with_focus_fillable(self):
        filled = REFINEMENT_WITH_FOCUS_TEMPLATE.format(
            focus_component="systems",
            additional_info="More details please",
        )
        assert "systems" in filled
        assert "More details please" in filled
