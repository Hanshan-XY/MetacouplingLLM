"""Tests for prompts/builder.py — prompt assembly."""

from metacouplingllm.prompts.builder import PromptBuilder


class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder(max_examples=2)

    def test_build_system_prompt_returns_string(self):
        prompt = self.builder.build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 2000

    def test_system_prompt_contains_all_layers(self):
        prompt = self.builder.build_system_prompt()
        prompt_lower = prompt.lower()
        # Role layer
        assert "expert advisor" in prompt_lower
        # Knowledge layer
        assert "metacoupling framework overview" in prompt_lower
        # Methodology layer
        assert "analysis methodology" in prompt_lower
        # Examples layer
        assert "reference examples" in prompt_lower
        # Output format layer
        assert "output format" in prompt_lower
        # Interaction layer
        assert "interaction guidelines" in prompt_lower
        # References
        assert "key references" in prompt_lower

    def test_system_prompt_includes_examples(self):
        prompt = self.builder.build_system_prompt(
            research_context="soybean trade agriculture"
        )
        assert "Soybean" in prompt

    def test_system_prompt_selects_relevant_examples(self):
        prompt = self.builder.build_system_prompt(
            research_context="urban water supply infrastructure"
        )
        assert "Beijing" in prompt or "Water" in prompt

    def test_max_examples_zero(self):
        builder = PromptBuilder(max_examples=0)
        prompt = builder.build_system_prompt()
        # Should still have framework knowledge, just no examples section
        assert "METACOUPLING FRAMEWORK OVERVIEW" in prompt


class TestBuildInitialMessage:
    def test_returns_formatted_message(self):
        msg = PromptBuilder.build_initial_message("My coffee trade study.")
        assert "My coffee trade study." in msg
        assert "metacoupling framework" in msg.lower()

    def test_strips_whitespace(self):
        msg = PromptBuilder.build_initial_message("  padded text  ")
        assert "padded text" in msg
        assert "  padded text  " not in msg


class TestBuildRefinementMessage:
    def test_without_focus(self):
        msg = PromptBuilder.build_refinement_message("More detail on causes.")
        assert "More detail on causes." in msg
        assert "focus" not in msg.lower() or "component" not in msg.lower()

    def test_with_focus(self):
        msg = PromptBuilder.build_refinement_message(
            "Tell me more", focus_component="systems"
        )
        assert "systems" in msg
        assert "Tell me more" in msg


class TestBuildAdm1PericouplingHint:
    """Tests for the ADM1-level pericoupling hint builder."""

    def test_michigan_returns_hint_with_neighbors(self):
        hint = PromptBuilder._build_adm1_pericoupling_hint(
            "Impact of Michigan's pork exports"
        )
        assert hint is not None
        assert "Michigan" in hint
        assert "ADJACENCY DATABASE" in hint
        # Michigan should have domestic neighbors
        assert "Same-country" in hint or "pericoupled" in hint

    def test_michigan_has_domestic_neighbors(self):
        hint = PromptBuilder._build_adm1_pericoupling_hint(
            "Michigan pork exports"
        )
        assert hint is not None
        # Michigan borders Indiana, Ohio, Wisconsin
        assert "Indiana" in hint
        assert "Ohio" in hint
        assert "Wisconsin" in hint

    def test_country_only_text_returns_none(self):
        hint = PromptBuilder._build_adm1_pericoupling_hint(
            "Brazil soy exports to China"
        )
        # "Brazil" and "China" are countries, not ADM1 regions
        assert hint is None

    def test_nonsense_text_returns_none(self):
        hint = PromptBuilder._build_adm1_pericoupling_hint(
            "random nonsense words here"
        )
        assert hint is None

    def test_system_prompt_includes_adm1_hint_for_michigan(self):
        builder = PromptBuilder(max_examples=0)
        prompt = builder.build_system_prompt(
            research_context="Impact of Michigan's pork exports"
        )
        assert "ADJACENCY DATABASE" in prompt
        assert "Michigan" in prompt

    def test_hint_wording_is_candidate_not_ground_truth(self):
        """The hint must tell the LLM that adjacency alone is NOT
        evidence of pericoupling, and must not claim neighbors are
        pericoupled by default.
        """
        hint = PromptBuilder._build_adm1_pericoupling_hint(
            "Impact of Michigan's pork exports"
        )
        assert hint is not None
        # New framing: adjacency != pericoupling
        assert "Adjacency alone is NOT evidence" in hint
        assert "reference only" in hint.lower()
        # Old authoritative framing must be gone
        assert "ground-truth" not in hint.lower()
        assert "are **pericoupled**" not in hint

    def test_system_prompt_uses_country_hint_when_no_adm1(self):
        builder = PromptBuilder(max_examples=0)
        prompt = builder.build_system_prompt(
            research_context="Brazil soybean exports to China"
        )
        # Should fall through to country-level hint
        assert "PERICOUPLING DATABASE INFORMATION" in prompt
        assert "ADM1" not in prompt.split("PERICOUPLING DATABASE")[0][-5:]


class TestFullLengthPassageSurvivesTruncation:
    """Regression guard: realistic chunks (up to ~2000 chars) should
    not be truncated by ``_format_literature_block`` — the 800-char
    cap was cutting bilateral country data off in long Results
    sections. The cap is now 5000 which is well above the 99th-pctile
    chunk length."""

    def test_1500_char_passage_not_truncated(self):
        from metacouplingllm.knowledge.rag import RetrievalResult, TextChunk

        text = ("Foreign countries sending tourists to China include "
                "Korea, Japan, and Russia. ") * 20
        assert 1000 < len(text) < 2500
        hit = RetrievalResult(
            chunk=TextChunk(
                paper_key="test",
                paper_title="Test",
                authors="Author",
                year=2024,
                section="4. Results",
                text=text,
                chunk_index=0,
            ),
            score=0.9,
        )
        block = PromptBuilder._format_literature_block([hit])
        # The "..." truncation marker should NOT appear
        assert "..." not in block
        # The closing "Russia" from the final sentence should survive
        assert "Russia" in block

    def test_pathological_long_passage_still_truncated(self):
        """Chunks exceeding 5000 chars (rare outliers) still get the
        truncation-with-ellipsis treatment so the user message doesn't
        balloon indefinitely."""
        from metacouplingllm.knowledge.rag import RetrievalResult, TextChunk

        text = "x " * 3500  # 7000 chars
        hit = RetrievalResult(
            chunk=TextChunk(
                paper_key="test",
                paper_title="Test",
                authors="Author",
                year=2024,
                section="X",
                text=text,
                chunk_index=0,
            ),
            score=0.9,
        )
        block = PromptBuilder._format_literature_block([hit])
        assert "..." in block
