"""
Prompt assembly — combines the six template layers with domain knowledge.

The :class:`PromptBuilder` injects framework definitions, curated examples,
and references into the prompt templates to produce complete system prompts
and user messages.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from metacouplingllm.knowledge.adm1_pericoupling import (
    get_adm1_info,
    get_adm1_neighbors,
    get_cross_border_neighbors,
    resolve_adm1_code,
)
from metacouplingllm.knowledge.countries import (
    ISO_ALPHA3_NAMES,
    get_country_name,
    resolve_country_code,
)
from metacouplingllm.knowledge.examples import (
    format_example,
    get_relevant_examples,
)
from metacouplingllm.knowledge.framework import get_framework_knowledge
from metacouplingllm.knowledge.pericoupling import PairCouplingType, lookup_pericoupling
from metacouplingllm.knowledge.references import format_references
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

if TYPE_CHECKING:  # pragma: no cover - types only
    from metacouplingllm.knowledge.rag import RetrievalResult


# Maximum characters of passage text to include in the prompt for any
# single retrieved chunk. Longer chunks are truncated with a "..." suffix.
#
# The chunker in ``knowledge.rag._chunk_markdown`` caps each chunk at
# ~250 words (~1600 chars median, p99 under 2000 chars, absolute max
# observed ~4700 chars). Setting this cap at 5000 is effectively a
# no-op for legitimate chunks — it only kicks in if the chunker
# produces a pathological chunk. The prior 800-char cap was cutting
# ~50% of every chunk's content, causing the main analysis LLM to
# miss bilateral country data that lives past the section-opening
# paragraph (e.g., "Korea (2.65 MtCO2)" at char 689 in a long Results
# chunk). A research tool should not silently lose half the evidence.
_PASSAGE_MAX_CHARS = 5000


class PromptBuilder:
    """Assembles multi-layer system prompts and user messages.

    Parameters
    ----------
    max_examples:
        Maximum number of curated examples to include in the prompt.
    """

    def __init__(self, max_examples: int = 2) -> None:
        self._max_examples = max_examples

    def build_system_prompt(
        self,
        research_context: str | None = None,
        web_context: str | None = None,
        include_citation_rules: bool = False,
    ) -> str:
        """Build the full system prompt by combining all template layers.

        Parameters
        ----------
        research_context:
            Optional research description used to select the most relevant
            examples.  If ``None``, default examples are used.
        web_context:
            Optional pre-formatted web search context to inject into the
            prompt.  Typically produced by
            :func:`~metacouplingllm.knowledge.websearch.format_web_context`.
        include_citation_rules:
            When ``True``, inject :data:`CITATION_RULES_LAYER` between the
            output-format layer and the interaction layer. Used by the
            pre-retrieval RAG pipeline so the LLM knows how to cite the
            ``<retrieved_literature>`` block that will appear in user
            messages.  Defaults to ``False`` for backward compatibility
            with the post-hoc RAG pipeline.

        Returns
        -------
        A complete system prompt string ready to be sent as the system
        message to an LLM.
        """
        # Layer 1 — Role
        parts: list[str] = [ROLE_LAYER]

        # Layer 2 — Knowledge (framework definitions)
        framework_text = get_framework_knowledge()
        parts.append(KNOWLEDGE_LAYER.format(framework_definitions=framework_text))

        # Layer 3 — Methodology
        parts.append(METHODOLOGY_LAYER)

        # Layer 4 — Examples
        examples = get_relevant_examples(
            research_context or "",
            max_examples=self._max_examples,
        )
        if examples:
            formatted = "\n\n".join(format_example(ex) for ex in examples)
            parts.append(EXAMPLES_LAYER.format(formatted_examples=formatted))

        # Layer 5 — Output format
        parts.append(OUTPUT_FORMAT_LAYER)

        # Layer 5b — Citation rules (only in pre-retrieval RAG mode).
        # Placed after OUTPUT_FORMAT_LAYER so the LLM reads citation
        # mechanics adjacent to the format spec, and immediately before
        # INTERACTION_LAYER so recency bias keeps the strict rules
        # fresh in the model's attention.
        if include_citation_rules:
            parts.append(CITATION_RULES_LAYER)

        # Layer 6 — Interaction guidelines
        parts.append(INTERACTION_LAYER)

        # Pericoupling database hint (pre-LLM injection)
        # Try subnational (ADM1) hint first; fall back to country-level.
        if research_context:
            hint = self._build_adm1_pericoupling_hint(research_context)
            if hint is None:
                hint = self._build_pericoupling_hint(research_context)
            if hint:
                parts.append(hint)

        # Web search context (pre-LLM injection)
        if web_context:
            parts.append(web_context)

        # Append references
        parts.append(format_references())

        return "\n\n".join(parts)

    @staticmethod
    def build_initial_message(
        research_description: str,
        literature_passages: "list[RetrievalResult] | None" = None,
    ) -> str:
        """Format the user's research description for the first turn.

        Parameters
        ----------
        research_description:
            The researcher's description of their study.
        literature_passages:
            Optional list of pre-retrieved RAG passages to inject as a
            ``<retrieved_literature>`` XML block. The block is placed
            BEFORE the research description (scope-setting) so the
            user's actual question remains the last thing the LLM
            reads. ``None`` (default) and ``[]`` are both handled —
            see :meth:`_format_literature_block`. When ``None`` no block
            is injected at all (post-hoc / no-RAG behavior).

        Returns
        -------
        A formatted user message string.
        """
        body = INITIAL_USER_TEMPLATE.format(
            research_description=research_description.strip(),
        )
        if literature_passages is None:
            return body
        # Pre-retrieval mode: scope-setting block goes BEFORE the
        # research description so the user's ask is the last thing
        # the LLM reads (recency bias).
        block = PromptBuilder._format_literature_block(literature_passages)
        return f"{block}\n\n{body}"

    @staticmethod
    def build_refinement_message(
        additional_info: str,
        focus_component: str | None = None,
        literature_passages: "list[RetrievalResult] | None" = None,
    ) -> str:
        """Format a follow-up message for refining the analysis.

        Parameters
        ----------
        additional_info:
            Additional context or questions from the researcher.
        focus_component:
            Optional framework component to focus on (e.g., ``"systems"``,
            ``"flows"``).  When provided, the prompt asks the LLM to
            elaborate on that specific component.
        literature_passages:
            Optional list of pre-retrieved RAG passages to inject as a
            ``<retrieved_literature>`` XML block. Unlike the initial
            message, the block is placed AFTER the refinement text so
            it reads as supporting evidence for the user's follow-up
            question rather than scope-setting context.

        Returns
        -------
        A formatted refinement message string.
        """
        info = additional_info.strip()

        if focus_component:
            body = REFINEMENT_WITH_FOCUS_TEMPLATE.format(
                focus_component=focus_component,
                additional_info=info,
            )
        else:
            body = REFINEMENT_USER_TEMPLATE.format(additional_info=info)

        if literature_passages is None:
            return body
        block = PromptBuilder._format_literature_block(literature_passages)
        return f"{body}\n\n{block}"

    # ------------------------------------------------------------------
    # Literature block formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_literature_block(
        results: "list[RetrievalResult] | None",
    ) -> str:
        """Render retrieved passages as an XML ``<retrieved_literature>`` block.

        Parameters
        ----------
        results:
            List of :class:`~metacouplingllm.knowledge.rag.RetrievalResult`
            objects in the order they should be cited (1-indexed). May
            be ``None`` or empty — in either case the block is emitted
            as the self-closing tag ``<retrieved_literature/>`` so the
            LLM knows that retrieval ran but found no relevant passages
            (preventing it from inventing citations).

        Returns
        -------
        A string containing one ``<retrieved_literature>`` block. Each
        passage is rendered as
        ``<passage id="N" paper_key="..." authors="..." year="..." section="..." score="0.87">text</passage>``.
        Passage text is truncated to roughly :data:`_PASSAGE_MAX_CHARS`
        characters with a ``...`` suffix.
        """
        if not results:
            return "<retrieved_literature/>"

        lines: list[str] = ["<retrieved_literature>"]
        for idx, r in enumerate(results, 1):
            chunk = r.chunk
            authors = _xml_attr_escape(chunk.authors)
            paper_key = _xml_attr_escape(chunk.paper_key)
            section = _xml_attr_escape(chunk.section)
            text = chunk.text.strip()
            if len(text) > _PASSAGE_MAX_CHARS:
                text = text[: _PASSAGE_MAX_CHARS - 3].rstrip() + "..."
            text = _xml_text_escape(text)
            lines.append(
                f'<passage id="{idx}" paper_key="{paper_key}" '
                f'authors="{authors}" year="{chunk.year}" '
                f'section="{section}" score="{r.score:.2f}">'
            )
            lines.append(text)
            lines.append("</passage>")
        lines.append("</retrieved_literature>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pericoupling hint
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pericoupling_hint(research_context: str) -> str | None:
        """Scan research text for country names and build a pericoupling hint.

        Only generates pairs that involve the **first detected country**
        (assumed to be the focal/sending country).  This avoids spurious
        pairs between receiving countries that the researcher did not
        intend to compare (e.g. USA ↔ Canada when the study is about
        Mexico exporting to both).

        Returns ``None`` if fewer than two countries are found.
        """
        # Detect country codes mentioned in the text.
        found_codes: list[str] = []
        seen: set[str] = set()

        # Try to resolve each word-sequence.  We scan overlapping n-grams
        # (up to 4 words) to catch multi-word names like "United States".
        words = re.findall(r"[A-Za-z'']+(?:\s+[A-Za-z'']+){0,3}", research_context)
        for phrase in words:
            code = resolve_country_code(phrase)
            if code and code not in seen:
                found_codes.append(code)
                seen.add(code)

        if len(found_codes) < 2:
            return None

        # Use the first detected country as the focal country and only
        # look up pairs between the focal country and the others.
        focal_code = found_codes[0]
        other_codes = found_codes[1:]

        lines: list[str] = []
        for code_b in other_codes:
            result = lookup_pericoupling(focal_code, code_b)
            name_a = get_country_name(focal_code)
            name_b = get_country_name(code_b)
            if result.pair_type == PairCouplingType.PERICOUPLED:
                lines.append(
                    f"- {name_a} ({focal_code}) and {name_b} ({code_b}) are "
                    f"**pericoupled** (geographically adjacent) according "
                    f"to the pericoupling database."
                )
            elif result.pair_type == PairCouplingType.TELECOUPLED:
                lines.append(
                    f"- {name_a} ({focal_code}) and {name_b} ({code_b}) are "
                    f"**telecoupled** (geographically distant) according "
                    f"to the pericoupling database."
                )

        if not lines:
            return None

        focal_name = get_country_name(focal_code)
        header = (
            "## PERICOUPLING DATABASE INFORMATION\n\n"
            f"Focal country detected: {focal_name} ({focal_code}).\n"
            "The following country-pair classifications come from a curated "
            "pericoupling database (ISO 3166-1 alpha-3). Use these ground-truth "
            "adjacency relationships when classifying coupling types in your "
            "analysis:\n"
        )
        return header + "\n".join(lines)

    @staticmethod
    def _build_adm1_pericoupling_hint(research_context: str) -> str | None:
        """Scan research text for ADM1 region names and build a subnational hint.

        Uses the same n-gram scanning approach as
        :meth:`_build_pericoupling_hint`, but resolves phrases against the
        ADM1 pericoupling database.  On the first successful match (the
        *focal region*), retrieves its domestic and cross-border neighbors
        and formats them into a prompt section.

        Returns ``None`` if no ADM1 region is found in the text.
        """
        # Scan for ADM1 region names via overlapping n-grams (up to 4 words).
        focal_code: str | None = None
        words = re.findall(
            r"[A-Za-z'']+(?:\s+[A-Za-z'']+){0,3}", research_context,
        )
        for phrase in words:
            code = resolve_adm1_code(phrase)
            if code:
                focal_code = code
                break

        if focal_code is None:
            return None

        focal_info = get_adm1_info(focal_code)
        if not focal_info:
            return None

        all_neighbors = get_adm1_neighbors(focal_code)
        cross_border = get_cross_border_neighbors(focal_code)
        domestic = all_neighbors - cross_border

        # Build readable neighbor lists.
        def _format_neighbors(codes: set[str]) -> list[str]:
            items: list[str] = []
            for c in sorted(codes):
                info = get_adm1_info(c)
                if info:
                    items.append(
                        f"{info['name']} ({c}, {info['country_name']})"
                    )
            return items

        lines: list[str] = []

        if domestic:
            domestic_names = _format_neighbors(domestic)
            lines.append(
                "**Same-country (pericoupled) neighbors:**\n"
                + "\n".join(f"- {n}" for n in domestic_names)
            )

        if cross_border:
            cross_names = _format_neighbors(cross_border)
            lines.append(
                "**Cross-border (pericoupled) neighbors:**\n"
                + "\n".join(f"- {n}" for n in cross_names)
            )

        if not lines:
            return None

        focal_name = focal_info["name"]
        country_name = focal_info["country_name"]
        header = (
            "## ADM1 ADJACENCY DATABASE (REFERENCE ONLY)\n\n"
            f"Focal subnational region detected: **{focal_name}** "
            f"({focal_code}, {country_name}).\n\n"
            "The following regions share a geographic border with the "
            "focal region according to a curated ADM1 adjacency database. "
            "**Adjacency alone is NOT evidence of pericoupling.** Only "
            "classify a region as pericoupled if you have independent "
            "evidence (from the research context, web snippets, or your "
            "training knowledge) of actual interactions, flows, impacts, "
            "or exchanges between that region and the focal region — "
            "e.g., labor movement, transport corridors, commodity flows, "
            "land-use displacement, or shared watersheds. If you have no "
            "specific evidence for a listed region, do NOT name it in "
            "your analysis and do NOT classify it as pericoupled.\n\n"
            "These adjacency relationships are provided for reference only.\n\n"
        )
        return header + "\n\n".join(lines)


# ----------------------------------------------------------------------
# XML escaping helpers for the literature block
# ----------------------------------------------------------------------
#
# Kept module-private and minimal — we are not building a full XML
# document, just emitting tags the LLM can read. Five characters
# (& < > " ') need to be escaped to keep the attributes/content from
# breaking the surrounding tag structure.

_XML_ATTR_ESCAPES = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&apos;",
}

_XML_TEXT_ESCAPES = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
}


def _xml_attr_escape(value: str) -> str:
    """Escape ``&``, ``<``, ``>``, ``"`` and ``'`` for use in an XML attribute."""
    out = value
    for ch, esc in _XML_ATTR_ESCAPES.items():
        out = out.replace(ch, esc)
    return out


def _xml_text_escape(value: str) -> str:
    """Escape ``&``, ``<``, ``>`` for use as XML text content."""
    out = value
    for ch, esc in _XML_TEXT_ESCAPES.items():
        out = out.replace(ch, esc)
    return out
