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
    _fold_diacritics,
    _region_names,
    get_adm1_country,
    get_adm1_info,
    get_adm1_neighbors,
    get_cross_border_neighbors,
    resolve_adm1_code,
)
from metacouplingllm.knowledge.countries import (
    ISO_ALPHA3_NAMES,
    _country_mentions,
    _normalize_country_text,
    get_country_name,
)
from metacouplingllm.knowledge.examples import (
    format_example,
    get_relevant_examples,
)
from metacouplingllm.knowledge.framework import get_framework_knowledge
from metacouplingllm.knowledge.pericoupling import (
    PairCouplingType,
    _check_standard,
    get_pericoupled_neighbors,
    is_pericoupled,
    lookup_pericoupling,
)
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


# --- Focal ADM1 region of a research description (the ADM1 hint) -----------

# Latin letters with diacritics, those of Vietnamese included ("Bình Phước").
_HINT_WORD_RE = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]+")
_APOSTROPHES = "'’"
# Lower-case words a region's name holds between capitalised ones ("Mato
# Grosso do Sul", "State of Mexico", "Côte d'Ivoire", "Dar es Salaam").
_NAME_CONNECTORS = frozenset({
    "ad", "al", "am", "an", "and", "as", "ash", "d", "da", "das", "de", "del",
    "della", "den", "der", "des", "di", "do", "dos", "du", "e", "ed", "el",
    "es", "et", "la", "le", "les", "of", "on", "sur", "the", "ul", "upon", "y",
})
# Lower-case unit words a region's name may end with ("Kyyivs'ka oblast",
# "the Tanga region").
_TRAILING_UNIT_WORDS = frozenset({
    "county", "department", "district", "division", "governorate", "oblast",
    "prefecture", "province", "region", "state", "voivodeship", "voivodship",
})
# A name followed by one of these words is a feature's ("Ohio River", "Mary
# River mine", "Paraná Basin"), and a name after "Lake", "Mount" or "Gulf of"
# is too ("Lake Tanganyika", "Gulf of California").
_FEATURE_WORDS_AFTER = frozenset({
    "aquifer", "basin", "bay", "canal", "dam", "delta", "desert", "estuary",
    "falls", "glacier", "hills", "island", "islands", "lagoon", "lake",
    "mine", "mountain", "mountains", "peninsula", "plateau", "range",
    "reservoir", "river", "sea", "sound", "strait", "valley",
})
_FEATURE_WORDS_BEFORE = frozenset({"lake", "mount", "mt"})
_FEATURE_PAIRS_BEFORE = frozenset({"bay of", "gulf of", "sea of", "strait of"})
# Words that are region names in some country but stand for a direction, a
# feature or a unit in most texts ("Bay", Somalia; "West", Rwanda; "Delta",
# Nigeria; "Forest", Guernsey).  Alone they name a region only in a text that
# names the region's country.
_GENERIC_PLACE_WORDS = frozenset({
    "bay", "capital", "central", "coast", "coastal", "delta", "desert",
    "east", "eastern", "forest", "greater", "gulf", "highlands", "inner",
    "island", "islands", "lake", "lakes", "lower", "lowlands", "middle",
    "mountains", "north", "northeast", "northern", "northwest", "outer",
    "plateau", "pool", "river", "rivers", "sea", "south", "southeast",
    "southern", "southwest", "upper", "valley", "west", "western",
})


def _hint_words(text: str) -> list[re.Match[str]]:
    """Return the words of *text*, without the "s" of a possessive "'s"."""
    return [
        m for m in _HINT_WORD_RE.finditer(text)
        if not (m.group(0) == "s" and m.start() > 0
                and text[m.start() - 1] in _APOSTROPHES)
    ]


def _written_as_name(text: str, words: list[re.Match[str]]) -> bool:
    """Return True when *words* are written as a place name: capitalised,
    with lower-case connectors inside, a lower-case unit word at the end,
    or a word continuing after an apostrophe ("Murang'a", "d'Azur")."""
    last = len(words) - 1
    for k, m in enumerate(words):
        word = m.group(0)
        if word[0].isupper():
            continue
        if k > 0 and text[m.start() - 1] in _APOSTROPHES:
            continue
        if 0 < k < last and word.lower() in _NAME_CONNECTORS:
            continue
        if 0 < k == last and word.lower() in _TRAILING_UNIT_WORDS:
            continue
        return False
    return True


def _region_at(
    text: str,
    words: list[re.Match[str]],
    i: int,
    size: int,
    named: list[str],
    hints: list[str],
    country_spans: list[tuple[int, int]],
    namesakes: dict[tuple[int, int], list[str]],
) -> str | None:
    """Return the code of the ADM1 region that words ``i .. i+size-1`` of
    *text* name, or ``None``.  *named* are the countries the text names,
    *hints* the countries a name several regions share is read in, and
    *namesakes* the countries' names to read as regions ("Georgia", the US
    state), by span, with the countries those regions lie in."""
    span = words[i:i + size]
    if not _written_as_name(text, span):
        return None
    start, end = span[0].start(), span[-1].end()
    # Never a piece of a country's name ("Zealand" in "New Zealand",
    # "Bissau" in "Guinea-Bissau"); a longer name may hold one ("State of
    # Mexico", "Western Australia").
    for a, b in country_spans:
        if a < end and start < b and not (start <= a and b <= end and (a, b) != (start, end)):
            return None
    before = [m.group(0).lower() for m in words[max(0, i - 2):i]]
    after = words[i + size].group(0).lower() if i + size < len(words) else ""
    if (before[-1:] and before[-1] in _FEATURE_WORDS_BEFORE) or (
        " ".join(before[-2:]) in _FEATURE_PAIRS_BEFORE
    ) or after in _FEATURE_WORDS_AFTER:
        return None
    phrase = " ".join(m.group(0) for m in span)
    if (start, end) in namesakes:
        return next(filter(None, (
            resolve_adm1_code(phrase, iso) for iso in namesakes[(start, end)]
        )), None)
    code = resolve_adm1_code(phrase)
    if code is None or get_adm1_country(code) not in hints:
        # A name several regions share is read in the text's countries
        # ("Central Region" in Ghana, "Punjab" in India).  A single word must
        # then be the region's whole name ("Santa" is not Santa Elena).
        for iso in hints:
            hinted = resolve_adm1_code(phrase, iso)
            info = get_adm1_info(hinted) if hinted else None
            if info and (size > 1 or _fold_diacritics(info["name"]).casefold()
                         == _fold_diacritics(phrase).casefold()):
                code = hinted
                break
    if code is None:
        return None
    if size == 1 and phrase.lower() in _GENERIC_PLACE_WORDS and (
        get_adm1_country(code) not in named
    ):
        return None
    return code


def _scan_regions(
    text: str,
    words: list[re.Match[str]],
    named: list[str],
    hints: list[str],
    country_spans: list[tuple[int, int]],
    namesakes: dict[tuple[int, int], list[str]],
) -> list[tuple[str, tuple[int, int]]]:
    """Return ``(code, span)`` for each ADM1 region *text* names, in order,
    trying runs of up to five words, longest first, and going on after each."""
    found: list[tuple[str, tuple[int, int]]] = []
    i = 0
    while i < len(words):
        for size in range(min(5, len(words) - i), 0, -1):
            code = _region_at(
                text, words, i, size, named, hints, country_spans, namesakes,
            )
            if code:
                found.append((code, (words[i].start(), words[i + size - 1].end())))
                i += size
                break
        else:
            i += 1
    return found


# ADM1 region names and aliases by normalized text, with the countries of the
# regions each names (lazily built).
_region_countries_by_name: dict[str, frozenset[str]] | None = None


def _countries_with_region_named(name: str) -> frozenset[str]:
    """Return the countries that have an ADM1 region called *name*."""
    global _region_countries_by_name
    if _region_countries_by_name is None:
        index: dict[str, frozenset[str]] = {}
        for key, isos in _region_names().items():
            key = _normalize_country_text(key)
            index[key] = index.get(key, frozenset()) | isos
        _region_countries_by_name = index
    return _region_countries_by_name.get(_normalize_country_text(name), frozenset())


def _read_places(text: str) -> tuple[list[str], list[str]]:
    """Return the countries *text* names and the ADM1 regions it names, in
    the order it names them.

    A run of words counts as a region's name only when it is written as one
    (:func:`_written_as_name`), is not part of a country's name ("Zealand" in
    "New Zealand") or of a feature's ("Lake Tanganyika", "Ohio River"), and,
    for a lone direction or feature word ("Bay", "West"), when the text names
    the region's country.  A name several regions share is read in the
    countries the text names, then in those of the other regions it names.

    A country's name that a region elsewhere also has ("Georgia", a US state;
    "Gaza", a province of Mozambique) is read as the one the rest of the text
    supports more: the region when the text names the region's country or
    more of its regions, the country when it names more of the country's
    regions or of the countries bordering it.  A tie keeps the country.  The
    region's reading lets the text's other names be read in its country: in
    "Florida and Georgia", "Florida", which Uruguay also has, is the US state.
    """
    words = _hint_words(text)
    mentions = []
    for code, a, b in _country_mentions(text):
        if text[b - 2:b] in ("'s", "’s"):  # "Australia's" in "Western Australia's"
            b -= 2
        mentions.append((code, a, b))
    namesakes: dict[tuple[int, int], list[str]] = {}
    for code, a, b in mentions:
        isos = _countries_with_region_named(text[a:b]) - {code}
        if isos:
            namesakes[(a, b)] = sorted(isos)
    if namesakes:
        named = list(dict.fromkeys(code for code, _, _ in mentions))
        hints = list(dict.fromkeys(named + [i for v in namesakes.values() for i in v]))
        others = [(a, b) for _, a, b in mentions if (a, b) not in namesakes]
        found = _scan_regions(text, words, named, hints, others, namesakes)
        regions = [get_adm1_country(c) for c, span in found if span not in namesakes]
        as_regions = set()
        for code, a, b in mentions:
            if (a, b) not in namesakes:
                continue
            hit = next((c for c, span in found if span == (a, b)), None)
            if hit is None:
                continue
            iso = get_adm1_country(hit)
            elsewhere = {c for c, x, y in mentions if (x, y) != (a, b)}
            for_region = (iso in elsewhere) + regions.count(iso)
            for_country = regions.count(code) + sum(
                bool(is_pericoupled(code, c)) for c in elsewhere - {code}
            )
            if for_region > for_country:
                as_regions.add((a, b))
        namesakes = {span: v for span, v in namesakes.items() if span in as_regions}
        mentions = [m for m in mentions if (m[1], m[2]) not in as_regions]
    named = list(dict.fromkeys(code for code, _, _ in mentions))
    spans = [(a, b) for _, a, b in mentions]
    hints = list(dict.fromkeys(named + [i for v in namesakes.values() for i in v]))
    found = _scan_regions(text, words, named, hints, spans, namesakes)
    more = list(dict.fromkeys(hints + [get_adm1_country(c) for c, _ in found]))
    if len(more) > len(hints):
        found = _scan_regions(text, words, named, more, spans, namesakes)
    return named, [code for code, _ in found]


def _find_focal_adm1(text: str) -> str | None:
    """Return the code of the ADM1 region *text* is about, or ``None``.

    The regions and countries are read as :func:`_read_places` reads them.
    The focal region is the first region named whose country the text names,
    whose country another region named shares, or whose country borders one
    of those; failing all three, the first region named.  So "Santa Rosa"
    (Guatemala) before "El Oro" in a text about Ecuador and Peru is passed
    over as a namesake.
    """
    named, found = _read_places(text)
    if not found:
        return None
    countries = {code: get_adm1_country(code) for code in found}

    def anchored(code: str) -> bool:
        iso = countries[code]
        others = {countries[c] for c in found if c != code}
        return iso in named or iso in others or any(
            is_pericoupled(iso, other) for other in set(named) | others
        )

    return next((code for code in found if anchored(code)), found[0])


class PromptBuilder:
    """Assembles multi-layer system prompts and user messages.

    Parameters
    ----------
    max_examples:
        Maximum number of curated examples to include in the prompt.
    de_facto_borders, coupling_standard:
        Adjacency settings for the pericoupling-database hint, passed to
        the country and ADM1 lookups (see
        :func:`~metacouplingllm.knowledge.pericoupling.lookup_pericoupling`).
        The defaults are the lookups' own.
    """

    def __init__(
        self,
        max_examples: int = 2,
        *,
        de_facto_borders: bool = True,
        coupling_standard: str = "moderate",
    ) -> None:
        self._max_examples = max_examples
        self._de_facto_borders = bool(de_facto_borders)
        self._coupling_standard = _check_standard(coupling_standard)

    def build_system_prompt(
        self,
        research_context: str | None = None,
        web_context: str | None = None,
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

        # Layer 5b — Citation rules.
        # Always included now that pre-retrieval RAG is the only mode
        # (see PR #38).  Placed after OUTPUT_FORMAT_LAYER so the LLM
        # reads citation mechanics adjacent to the format spec, and
        # immediately before INTERACTION_LAYER so recency bias keeps
        # the strict rules fresh in the model's attention.
        parts.append(CITATION_RULES_LAYER)

        # Layer 6 — Interaction guidelines
        parts.append(INTERACTION_LAYER)

        # Pericoupling database hint (pre-LLM injection)
        # Try subnational (ADM1) hint first; fall back to country-level.
        if research_context:
            hint = self._build_adm1_pericoupling_hint(
                research_context,
                de_facto_borders=self._de_facto_borders,
                coupling_standard=self._coupling_standard,
            )
            if hint is None:
                hint = self._build_coupling_hint(
                    research_context,
                    de_facto_borders=self._de_facto_borders,
                    coupling_standard=self._coupling_standard,
                )
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
        turn: int = 1,
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
        turn:
            1-indexed turn number used to tag the
            ``<retrieved_literature turn="...">`` block and its
            ``<passage turn="...">`` entries. Default ``1``.

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
        block = PromptBuilder._format_literature_block(
            literature_passages, turn=turn
        )
        return f"{block}\n\n{body}"

    @staticmethod
    def build_refinement_message(
        additional_info: str,
        focus_component: str | None = None,
        literature_passages: "list[RetrievalResult] | None" = None,
        turn: int = 2,
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
        turn:
            1-indexed turn number used to tag the
            ``<retrieved_literature turn="...">`` block and its
            ``<passage turn="...">`` entries. Default ``2``.

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
        block = PromptBuilder._format_literature_block(
            literature_passages, turn=turn
        )
        return f"{body}\n\n{block}"

    # ------------------------------------------------------------------
    # Literature block formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_literature_block(
        results: "list[RetrievalResult] | None",
        turn: int = 1,
    ) -> str:
        """Render retrieved passages as an XML ``<retrieved_literature>`` block.

        Parameters
        ----------
        results:
            List of :class:`~metacouplingllm.knowledge.rag.RetrievalResult`
            objects in the order they should be cited (1-indexed). May
            be ``None`` or empty — in either case the block is emitted
            as the self-closing tag
            ``<retrieved_literature turn="K"/>`` so the LLM knows that
            retrieval ran for this turn but found no relevant passages
            (preventing it from inventing citations).
        turn:
            1-indexed turn number used to tag the block. The LLM uses
            this attribute to assemble the turn-scoped citation token
            ``[Tk:N]``. Default ``1``.

        Returns
        -------
        A string containing one ``<retrieved_literature turn="k">``
        block. Each passage is rendered as
        ``<passage turn="k" id="N" paper_key="..." authors="..." year="..." section="..." score="0.87">text</passage>``.
        Passage text is truncated to roughly :data:`_PASSAGE_MAX_CHARS`
        characters with a ``...`` suffix.
        """
        if not results:
            return f'<retrieved_literature turn="{turn}"/>'

        lines: list[str] = [f'<retrieved_literature turn="{turn}">']
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
                f'<passage turn="{turn}" id="{idx}" paper_key="{paper_key}" '
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
    def _build_coupling_hint(
        research_context: str,
        *,
        de_facto_borders: bool = True,
        coupling_standard: str = "moderate",
    ) -> str | None:
        """Scan research text for country names and build a country-pair
        classification hint.

        Renamed from ``_build_pericoupling_hint`` (PR #48): the method
        emits BOTH pericoupled AND telecoupled per-pair lines, so the
        old ``pericoupling`` name understated its scope.  The
        LLM-facing section heading is intentionally unchanged.

        If ≥2 countries are detected, reports each focal-vs-other pair
        with its database classification (PERICOUPLED for adjacent
        pairs, TELECOUPLED for distant pairs).

        If exactly 1 country is detected (PR #44), enumerates that
        focal country's pericoupled neighbors from the database as
        REFERENCE context the LLM may want to consider as candidate
        pericoupled partners.  Added to surface underexplored
        pericoupling angles for single-country queries that
        previously got no hint at all.

        Both lookups use the given ``de_facto_borders`` and
        ``coupling_standard`` (the assistant's adjacency settings), so
        the hint agrees with the post-LLM validation.

        Framing is REFERENCE-ONLY, parallel to the ADM1 hint
        (PR #20, harmonized in PR #43): the LLM is told that
        database classification is supporting context but the actual
        coupling type still depends on whether real cross-system
        flows exist.

        Only generates pairs that involve the **first detected
        country** (assumed to be the focal/sending country).  This
        avoids spurious pairs between receiving countries that the
        researcher did not intend to compare (e.g. USA ↔ Canada when
        the study is about Mexico exporting to both).  Countries are
        detected in the order the text first names them (see
        :func:`~metacouplingllm.knowledge.countries.find_country_codes`),
        leaving out a country's name read as a region's ("Georgia", the US
        state; :func:`_read_places`).

        Returns ``None`` when:
        - No country is detected, OR
        - A 1-country query's focal country has zero pericoupled
          neighbors (island nations like Australia, Japan, Cuba).
        """
        # Detect country codes mentioned in the text, in the order the
        # text first names them.
        found_codes, _ = _read_places(research_context)

        if len(found_codes) < 1:
            return None  # no country detected -> no hint

        # Use the first detected country as the focal country and only
        # look up pairs between the focal country and the others.
        focal_code = found_codes[0]
        other_codes = found_codes[1:]
        focal_name = get_country_name(focal_code)

        body: str
        if other_codes:
            # >=2 countries: per-pair classification lookup (unchanged).
            lines: list[str] = []
            for code_b in other_codes:
                result = lookup_pericoupling(
                    focal_code,
                    code_b,
                    de_facto_borders=de_facto_borders,
                    coupling_standard=coupling_standard,
                )
                name_b = get_country_name(code_b)
                if result.pair_type == PairCouplingType.PERICOUPLED:
                    lines.append(
                        f"- {focal_name} ({focal_code}) and {name_b} ({code_b}) are "
                        f"**pericoupled** (geographically adjacent) according "
                        f"to the pericoupling database."
                    )
                elif result.pair_type == PairCouplingType.TELECOUPLED:
                    lines.append(
                        f"- {focal_name} ({focal_code}) and {name_b} ({code_b}) are "
                        f"**telecoupled** (geographically distant) according "
                        f"to the pericoupling database."
                    )
            if not lines:
                return None
            body = "\n".join(lines)
        else:
            # PR #44: exactly 1 country detected -> enumerate focal's
            # pericoupled neighbors as REFERENCE context.  Island
            # nations with no neighbors yield None (no empty
            # "neighbors:" block).
            neighbors = sorted(
                get_pericoupled_neighbors(
                    focal_code,
                    de_facto_borders=de_facto_borders,
                    coupling_standard=coupling_standard,
                )
            )
            if not neighbors:
                return None
            neighbor_lines = [
                f"- {get_country_name(c)} ({c})" for c in neighbors
            ]
            body = (
                f"No specific partner country was named in the research "
                f"description.  Based on the database, {focal_name}'s "
                f"pericoupled (geographically adjacent) neighbors are:\n"
                + "\n".join(neighbor_lines)
                + "\n\nAny actual pericoupling involving these neighbors "
                "must still be supported by independent evidence of "
                "cross-system flows."
            )

        header = (
            "## PERICOUPLING DATABASE LOOKUP (REFERENCE)\n\n"
            f"Focal country detected: {focal_name} ({focal_code}).\n"
            "The following classifications come from a curated country-pair "
            "database (ISO 3166-1 alpha-3).  For PERICOUPLED pairs, "
            "database adjacency is a strong indicator but NOT proof of "
            "pericoupling — confirm with independent evidence of actual "
            "flows (trade, migration, policy spillover, etc.) before "
            "classifying.  For TELECOUPLED pairs, geographic distance is "
            "corroborating context but the classification still depends on "
            "whether actual cross-system flows exist.\n\n"
        )
        return header + body

    @staticmethod
    def _build_adm1_pericoupling_hint(
        research_context: str,
        *,
        de_facto_borders: bool = True,
        coupling_standard: str = "moderate",
    ) -> str | None:
        """Scan research text for ADM1 region names and build a subnational hint.

        Finds the *focal region* (:func:`_find_focal_adm1`), retrieves its
        domestic and cross-border neighbors under the given
        ``de_facto_borders`` and ``coupling_standard`` and formats them into
        a prompt section.

        Returns ``None`` if no ADM1 region is found in the text, or if
        the focal region has no neighbor under those settings.
        """
        focal_code = _find_focal_adm1(research_context)
        if focal_code is None:
            return None

        focal_info = get_adm1_info(focal_code)
        if not focal_info:
            return None

        all_neighbors = get_adm1_neighbors(
            focal_code,
            de_facto_borders=de_facto_borders,
            coupling_standard=coupling_standard,
        )
        cross_border = get_cross_border_neighbors(
            focal_code,
            de_facto_borders=de_facto_borders,
            coupling_standard=coupling_standard,
        )
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
            "evidence (from the research context, web results, or your "
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
