"""
Main user-facing interface for the Metacoupling package.

:class:`MetacouplingAssistant` is the primary entry point.  Researchers
create an advisor with their LLM client, then call :meth:`analyze` and
:meth:`refine` to iteratively build a telecoupling/metacoupling analysis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from metacouplingllm.knowledge.citations import (
    TURN_CITATION_PATTERN,
    sanitize_turn_citations,
)
from metacouplingllm.knowledge.countries import get_country_name, resolve_country_code
from metacouplingllm.knowledge.literature import Paper
from metacouplingllm.knowledge.pericoupling import (
    PairCouplingType,
    lookup_pericoupling,
)
from metacouplingllm.llm.client import (
    AnthropicAdapter,
    GeminiAdapter,
    GrokAdapter,
    LLMClient,
    LLMResponse,
    Message,
    OpenAIAdapter,
)
from metacouplingllm.llm.parser import ParsedAnalysis, parse_analysis
from metacouplingllm.output.formatter import AnalysisFormatter
from metacouplingllm.prompts.builder import PromptBuilder

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from metacouplingllm.knowledge.rag import RetrievalResult


logger = logging.getLogger(__name__)


# Allowed values for the rag_mode parameter on MetacouplingAssistant.
_VALID_RAG_MODES = frozenset({"pre_retrieval", "post_hoc"})

# Public built-in RAG corpus names. Use these when the bundled journal
# corpus should be used; ``rag_papers_dir`` is reserved for custom folders.
JOURNAL_ARTICLES_2025 = "journal_articles_2025"
_BUILTIN_RAG_CORPUS_ALIASES = frozenset({
    JOURNAL_ARTICLES_2025,
    "journal_articles",
    "bundled",
})
_BUILTIN_RAG_SENTINEL = "__metacoupling_builtin_journal_articles_2025__"

# Canonical regex for splitting flow direction strings at arrow symbols.
# Used across flow resolution and validation to ensure consistency.
_FLOW_ARROW_RE = re.compile(r"\s*(?:\u2192|\u2194|<->|<=>|->|=>)\s*")
# Detect whether a flow direction contains a connector (arrow or "between X and Y").
_FLOW_HAS_CONNECTOR_RE = re.compile(
    r"\bbetween\b.+\band\b|\u2192|\u2194|<->|<=>|->|=>",
    re.IGNORECASE,
)
_UNSUPPORTED_AUTOMAP_SCOPE_RE = re.compile(
    r"\b("
    r"city|cities|municipalit(?:y|ies)|municipal|town|village|"
    r"watershed|watersheds|river\s+basin|basin|"
    r"protected\s+area|nature\s+reserve|national\s+nature\s+reserve|"
    r"national\s+park|park|reserve|conservation\s+area"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Prompt-budget configuration for the structured map-extraction LLM call
# ---------------------------------------------------------------------------
#
# These constants govern how much of the parsed analysis is summarised into
# the second LLM call inside ``_extract_map_data_from_analysis``.  They
# replace previously hardcoded inline literals so the trade-offs are
# documented and tunable in one place.
#
# Field-priority tuples: when summarising a system entry as
# ``"key: value | key: value | ..."``, fields are emitted in this order.
# ``name`` + ``geographic_scope`` carry the highest signal for map
# extraction (they're how the LLM identifies countries / ADM1 regions),
# so they go first and are guaranteed to survive truncation even when
# verbose subsystem prose pushes the joined text past the per-system cap.
_HIGH_PRIORITY_SYSTEM_FIELDS: tuple[str, ...] = ("name", "geographic_scope")
_LOW_PRIORITY_SYSTEM_FIELDS: tuple[str, ...] = (
    "human_subsystem", "natural_subsystem", "description",
)

# Defensive ceiling on the number of web-search results included in the
# structured map-extraction prompt.  This is independent of the user's
# ``web_search_max_results`` setting -- that setting already caps
# ``self._last_web_results`` upstream.  The constant exists only to bound
# prompt size if a user accidentally sets ``web_search_max_results`` to a
# very large number.
_MAX_WEB_SNIPPETS_IN_MAP_PROMPT: int = 100


# ---------------------------------------------------------------------------
# Flow-category aliasing
# ---------------------------------------------------------------------------
#
# The Liu 2017 telecoupling framework (Table 1) defines six canonical flow
# categories: matter, capital, information, energy, people, organisms.
#
# LLMs occasionally emit narrower or near-synonym labels (e.g. "goods",
# "money", "electricity", "tourism", "livestock").  Each entry below maps a
# slip into the right canonical bucket so the flow survives validation
# instead of being silently dropped.
#
# Inclusion criteria — a candidate is added iff:
#   1. It is a clear semantic equivalent of, or subset of, exactly one
#      canonical category, AND
#   2. It plausibly appears in real LLM output or research literature, AND
#   3. It is NOT ambiguous between two categories.
#
# Notable rejections (kept here for posterity / reviewer reference):
#   - "power"     — political vs market vs electrical power (ambiguous)
#   - "products"  — digital products (information) vs physical goods (matter)
#   - "resources" — too vague (people / energy / money / matter)
#   - "services"  — capital / people / information all plausible
#   - "seeds"     — propagules (organisms) vs harvested commodity (matter)
#   - "crops"     — standing crop (organisms) vs harvested grain (matter)
#   - "economic"  — capital vs trade-in-goods (matter)
#
# The lookup is case-insensitive — callers should ``str.strip().lower()``
# the LLM-emitted category before consulting this dict.
_FLOW_CATEGORY_ALIASES: dict[str, str] = {
    # matter (physical goods, commodities)
    "material":     "matter",
    "materials":    "matter",
    "goods":        "matter",
    "commodity":    "matter",
    "commodities":  "matter",
    "cargo":        "matter",
    "food":         "matter",
    # capital (money, finance)
    "financial":    "capital",
    "money":        "capital",
    "monetary":     "capital",
    "investment":   "capital",
    "cash":         "capital",
    "currency":     "capital",
    "funds":        "capital",
    "payment":      "capital",
    "payments":     "capital",
    # information (data, knowledge, signals)
    "data":         "information",
    "knowledge":    "information",
    "info":         "information",
    "signals":      "information",
    # energy (electrical, chemical, embodied)
    "electricity":  "energy",
    "electrical":   "energy",
    "electric":     "energy",
    "fuel":         "energy",
    "fuels":        "energy",
    # people (humans, migration, labor, tourism)
    "humans":       "people",
    "migration":    "people",
    "migrants":     "people",
    "labor":        "people",
    "labour":       "people",
    "workers":      "people",
    "tourists":     "people",
    "tourism":      "people",
    "personnel":    "people",
    "passengers":   "people",
    # organisms (species, wildlife, livestock)
    "organism":     "organisms",
    "species":      "organisms",
    "wildlife":     "organisms",
    "animals":      "organisms",
    "plants":       "organisms",
    "livestock":    "organisms",
}

# The six canonical category names from Liu 2017 — the closed set every
# valid flow must end up in after alias normalisation.
_CANONICAL_FLOW_CATEGORIES: frozenset[str] = frozenset({
    "matter",
    "capital",
    "information",
    "energy",
    "people",
    "organisms",
})


def _normalize_flow_category(raw: str) -> str | None:
    """Normalize a raw category label to a canonical Liu 2017 category.

    Returns one of ``"matter" / "capital" / "information" / "energy" /
    "people" / "organisms"``, or ``None`` if the label cannot be
    confidently mapped.

    The lookup is case-insensitive and tolerates surrounding whitespace.
    """
    cat = raw.strip().lower()
    if not cat:
        return None
    cat = _FLOW_CATEGORY_ALIASES.get(cat, cat)
    if cat in _CANONICAL_FLOW_CATEGORIES:
        return cat
    return None


@dataclass
class AnalysisResult:
    """The result of an analysis or refinement turn.

    Attributes
    ----------
    parsed:
        Structured representation (best-effort parsed).
    formatted:
        Human-readable text ready to display.
    raw:
        The unprocessed LLM response.
    turn_number:
        Which conversational turn produced this result (1-indexed).
    usage:
        Token-usage information from the LLM (if available).
    map:
        Auto-generated matplotlib Figure (country-level or ADM1), or
        ``None`` when ``auto_map=False`` or visualization deps are missing.
    structured_supplement:
        When ``rag_structured_extraction=True`` and the pre-retrieval
        RAG pipeline produced retrieval hits, a second LLM pass
        returns a dict with keys ``additional_{sending,receiving,
        spillover}_mentions``, ``{sending,receiving,spillover}_subsystem_fills``,
        and ``supplementary_flows``. Each entry carries evidence
        passage IDs pointing back at the ``[Tk:N]`` references in the
        ``SUPPORTING EVIDENCE FROM LITERATURE`` block. Also rendered
        visibly in ``formatted`` under a ``SUPPLEMENTARY STRUCTURED
        EXTRACTION`` block. ``None`` when the feature is disabled or
        extraction failed.
    map_notice:
        Human-readable map status note appended to ``formatted`` when
        ``auto_map=True``. This explains whether a map was generated or
        why no map was produced.
    flow_parse_warnings:
        Flow direction strings the legacy regex map path could not
        resolve into endpoints. Each entry is a dict with keys
        ``direction`` (the original string), ``category`` (the flow's
        category), and ``reason`` (a short string explaining why
        resolution failed). Empty when all flows parsed cleanly. Only
        populated by the legacy text-extraction map path; the
        structured (``parsed.map_data``) path uses ``logger.debug``
        for its own drops.
    """

    parsed: ParsedAnalysis
    formatted: str
    raw: str
    turn_number: int
    usage: dict[str, int] | None = None
    map: Figure | None = None
    web_map_signals: dict[str, object] | None = None
    structured_supplement: dict[str, object] | None = None
    map_notice: str | None = None
    flow_parse_warnings: list[dict[str, str]] = field(default_factory=list)
    # §7 Evidence Coverage — model self-assessment by the MAIN
    # analysis LLM call.  Considers both RAG literature AND web
    # evidence (only the main call sees both streams), so a "gap"
    # here means "no source supports this," not "web didn't cover
    # it" (RAG may have).  Empty string when the LLM omitted §7.
    # Mirror of ``self.parsed.evidence_coverage_note`` lifted onto
    # the result for easier programmatic access.
    evidence_coverage_note: str = ""
    # Suggested follow-up web-search queries emitted by the
    # web-extraction LLM call (``extract_web_map_signals``).  Each
    # entry is a short, specific search query the model thinks
    # would fill gaps in the WEB evidence (RAG-side gaps are
    # discussed in ``evidence_coverage_note`` instead).  Empty
    # list when no web search ran or no queries were suggested.
    suggested_followup_queries: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        parts = [f"turn={self.turn_number}"]
        if self.parsed.is_parsed:
            parts.append(f"parsed=True")
            if self.parsed.coupling_classification:
                cc = self.parsed.coupling_classification[:30]
                parts.append(f"classification='{cc}'")
        parts.append(f"formatted={len(self.formatted)} chars")
        if self.map is not None:
            parts.append("map=Figure")
        if self.usage:
            total = self.usage.get(
                "total_tokens",
                sum(self.usage.values()),
            )
            parts.append(f"tokens={total}")
        if self.flow_parse_warnings:
            parts.append(f"flow_parse_warnings={len(self.flow_parse_warnings)}")
        return f"AnalysisResult({', '.join(parts)})"


@dataclass
class RAGResult:
    """The result of a single RAG-only Q&A turn (``coupling_analysis=False``).

    Attributes
    ----------
    answer:
        The LLM's narrative response with turn-scoped citation tokens
        like ``[T1:1]`` (literature) and ``[T1:W1]`` (web). Markers
        may be sparse — the LLM cites only the passages that directly
        support a claim. In multi-turn sessions the answer may also
        back-reference prior turns (e.g., ``[T1:3]`` appearing in a
        turn-2 answer). Invalid markers (e.g., ``[T1:99]`` when only
        8 passages were retrieved for turn 1) have already been
        stripped by the sanitizer. Bare legacy ``[N]`` / ``[W1]``
        tokens are also stripped.
    references:
        Papers cited by the LLM in the current turn's answer,
        deduplicated by ``key`` and ordered by first appearance.
        Prior-turn back-references are NOT included here — only
        current-turn citations.
    retrieved_passages:
        All passages shown to the LLM for this turn (length ``rag_top_k``).
    web_sources:
        Web search results used to supplement the answer, or ``None`` if
        ``web_search=False``.
    turn_number:
        Which conversational turn produced this result (1-indexed within
        the current RAG-mode session).
    usage:
        Token-usage information from the LLM (if available).
    raw:
        The unsanitized LLM response text.
    """

    answer: str
    references: list[Paper]
    retrieved_passages: list[RetrievalResult]
    web_sources: list[dict[str, str]] | None = None
    turn_number: int = 1
    usage: dict[str, int] | None = None
    raw: str = ""
    retrieval_backend: str = "embeddings"
    # Parallel to ``references``: the passage ID (the `N` in ``[Tk:N]``)
    # at which each paper was first cited in ``answer``. Used by the
    # bibliography renderer so its labels match the inline citations
    # exactly. Empty when ``references`` is empty.
    reference_passage_ids: list[int] = field(default_factory=list)

    @property
    def formatted(self) -> str:
        """Display-friendly answer with the bibliography and (when
        ``web_search=True``) the web-sources block appended.

        The answer body is shown verbatim — the LLM already emitted
        stable turn-scoped tokens like ``[T1:3]`` and ``[T2:1]`` so
        no remapping is needed. A ``REFERENCES`` block lists each
        cited paper as ``[Tk:N] Title …`` where ``N`` is the actual
        passage ID the LLM used inline (so the label in the bibliography
        always matches the inline citation). When the LLM also cited
        web sources, a ``WEB SOURCES`` block follows with ``[Tk:W1]``-style
        IDs. Use this in place of ``answer`` when you want a single
        ``print(result.formatted)`` to show everything a user needs.
        """
        parts = [self.answer]
        if self.references:
            parts.append(_format_references_block(
                self.references,
                self.retrieved_passages,
                backend=self.retrieval_backend,
                turn=self.turn_number,
                passage_ids=self.reference_passage_ids,
            ))
        if self.web_sources:
            parts.append(_format_web_sources_block(
                self.web_sources, turn=self.turn_number,
            ))
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        parts = [f"turn={self.turn_number}"]
        parts.append(f"answer={len(self.answer)} chars")
        parts.append(f"refs={len(self.references)}")
        parts.append(f"passages={len(self.retrieved_passages)}")
        if self.web_sources:
            parts.append(f"web={len(self.web_sources)}")
        if self.usage:
            total = self.usage.get(
                "total_tokens", sum(self.usage.values())
            )
            parts.append(f"tokens={total}")
        return f"RAGResult({', '.join(parts)})"


def _format_references_block(
    refs: list[Paper],
    passages: list[RetrievalResult] | None = None,
    backend: str = "embeddings",
    turn: int = 1,
    passage_ids: list[int] | None = None,
) -> str:
    """Render the cited-papers list as a printable bibliography.

    Each reference is labelled with the turn-scoped token ``[Tk:N]``
    so the reader can match it back to the inline citations in the
    answer body. When ``passage_ids`` is provided (a parallel list to
    ``refs``), each entry uses the actual passage ID the LLM cited
    inline; otherwise the entries are numbered sequentially
    ``1..len(refs)`` as a fallback.

    When ``passages`` is provided, each reference also shows the
    retrieval confidence (High / Medium / Low / Very Low) and the
    raw similarity score, taken from the highest-scoring passage of
    that paper in the retrieval set. ``backend`` ("embeddings" or
    "tfidf") selects the appropriate confidence thresholds.
    """
    # Map paper_key -> best score across all retrieved passages of that paper.
    score_by_key: dict[str, float] = {}
    if passages:
        for r in passages:
            key = r.chunk.paper_key
            if key not in score_by_key or r.score > score_by_key[key]:
                score_by_key[key] = r.score

    # Lazy import to avoid a hard dependency at import time.
    from metacouplingllm.knowledge.rag import _score_to_confidence

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"  REFERENCES (cited in turn {turn})")
    lines.append("=" * 70)
    for idx, paper in enumerate(refs, start=1):
        # Use the actual cited passage ID when available; fall back to
        # sequential numbering otherwise.
        if passage_ids is not None and idx - 1 < len(passage_ids):
            label_n = passage_ids[idx - 1]
        else:
            label_n = idx
        # Trim very long author strings the same way format_evidence does.
        authors = paper.authors
        if len(authors) > 60:
            parts = authors.split(" and ")
            if len(parts) > 2:
                authors = parts[0] + " et al."
        lines.append("")
        lines.append(f"  [T{turn}:{label_n}] {paper.title}")
        lines.append(f"      {authors} ({paper.year})")
        score = score_by_key.get(paper.key)
        if score is not None:
            confidence = _score_to_confidence(score, backend=backend)
            lines.append(
                f"      Confidence: {confidence} (score: {score:.3f})"
            )
        lines.append(f"      key: {paper.key}")
    return "\n".join(lines)


def _format_web_sources_block(
    web_sources: list[dict[str, str]], turn: int = 1,
) -> str:
    """Render the web search hits as a printable block.

    Uses turn-scoped ``[Tk:W1], [Tk:W2], ...`` IDs to match the
    citation grammar used inline in the answer body. Snippets are
    truncated to ~140 chars to keep the block readable.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(
        f"  WEB SOURCES (turn {turn}, background context)"
    )
    lines.append("=" * 70)
    for i, src in enumerate(web_sources, start=1):
        title = (src.get("title") or "(no title)").strip()
        url = (src.get("url") or "").strip()
        # Accept legacy "snippet" key alongside the canonical
        # ``model_summary`` (the rename ships in the same PR; older
        # downstream callers / serialised state may still carry
        # ``snippet``).
        model_summary = (
            src.get("model_summary")
            or src.get("snippet")
            or ""
        ).strip().replace("\n", " ")
        if len(model_summary) > 140:
            model_summary = model_summary[:137].rstrip() + "..."
        lines.append("")
        lines.append(f"  [T{turn}:W{i}] {title}")
        if url:
            lines.append(f"       {url}")
        if model_summary:
            lines.append(f"       {model_summary}")
    return "\n".join(lines)


# System prompt for the RAG-only Q&A mode (coupling_analysis=False).
# Multi-turn: the advisor maintains _rag_history across analyze() calls
# so users can ask follow-up questions naturally. Citations are
# turn-scoped — `[Tk:N]` for literature and `[Tk:Wn]` for web sources,
# where `k` is the turn index. Prior-turn citations remain valid for
# back-reference; the same number never changes meaning.
_RAG_ONLY_SYSTEM_PROMPT = """\
You are a literature assistant for the metacoupling and telecoupling
research community. The user already understands the framework — do
not lecture them about it.

This is a multi-turn conversation. Use the prior turns as context to
interpret follow-up questions, and ground every factual claim in the
retrieved literature.

CITATION GRAMMAR (turn-scoped):

Each user message includes a `<retrieved_literature turn="k">` block
with passages labelled `<passage turn="k" id="N" ...>`, and optionally
a `<web_search_results turn="k">` block with web sources. The `turn`
attribute is the turn index — it never changes for that block.

- Cite literature as `[Tk:N]` and web sources as `[Tk:Wn]`, where
  `k` is the block's `turn` attribute and `N` (or `n`) is the
  passage's `id`. For example, the 2nd literature passage of turn 3
  is cited as `[T3:2]`. The 1st web source of turn 1 is `[T1:W1]`.
- Prior-turn citations stay valid. To back-reference an earlier
  turn's evidence, copy its exact token verbatim
  (e.g., *"as established in [T1:3], soybean imports..."*).
- NEVER emit a bare `[N]` or `[W1]` — those are illegal under the
  current grammar and will be silently stripped.
- NEVER invent citations. `k` cannot exceed the highest turn you
  have seen; `N` cannot exceed that turn's passage count.

Answer using ONLY the retrieved literature passages provided in this
conversation. Cite every factual claim with one or more `[Tk:N]`
markers. Do not introduce facts that are not supported by the
passages. Prefer the literature for citable claims; web sources are
background context.

If the literature does not cover the question — or covers only part
of it — say so explicitly and describe the gap. A short, well-cited
answer is better than a long, weakly-supported one.
"""

# Regex for extracting turn-scoped cited passage IDs from a sanitized
# answer. Matches [Tk:N] (literature) and [Tk:Wn] (web). Groups:
#   1 = turn k, 2 = "W" or "", 3 = ID N.
_CITATION_TOKEN_RE = TURN_CITATION_PATTERN


class MetacouplingAssistant:
    """Advisor that applies the metacoupling framework to user research.

    Parameters
    ----------
    llm_client:
        Any object satisfying the :class:`~metacouplingllm.llm.client.LLMClient`
        protocol (e.g., :class:`OpenAIAdapter`, :class:`AnthropicAdapter`,
        or a custom implementation).
    temperature:
        Sampling temperature for the LLM.
    max_tokens:
        Maximum response tokens.  ``None`` uses the model default.
    max_examples:
        Number of curated examples to inject into the system prompt.
    verbose:
        If ``True``, print diagnostic information during execution.
    recommend_papers:
        If ``True``, append literature recommendations to ``formatted``.
    max_recommendations:
        Maximum number of paper recommendations to include.
    auto_map:
        If ``True``, automatically generate a map from the analysis.

        - **Subnational research** (e.g., "Michigan's pork exports"):
          generates an ADM1 map where the focal region is intracoupling,
          adjacent ADM1 regions are pericoupling, and distant regions are
          telecoupling.  Uses *adm1_shapefile* when provided, otherwise
          tries the cached / auto-downloaded ADM1 basemap.
        - **National research** (e.g., "USA's pork exports"):
          generates a country-level world map.  Requires geopandas.

        The map is stored on ``AnalysisResult.map``.  If visualisation
        dependencies are not installed the field is silently ``None``.
    adm1_shapefile:
        Path to a World Bank Admin 1 GeoPackage (``.gpkg``) or shapefile.
        Optional override for subnational (ADM1) maps. When omitted,
        metacoupling first looks for a cached/local file and then tries
        the hosted ADM1 download mirror.
    adm0_shapefile:
        Path to a World Bank Admin 0 GeoPackage for country-level maps.
        When omitted, metacoupling first looks for a local
        ``Admin 0_all_layers.gpkg`` before trying the hosted mirror and
        official World Bank fallback download.
    rag_papers_dir:
        Path to a custom directory containing markdown papers. Use this
        only when you want to override the bundled RAG corpus with your
        own files.
    rag_corpus:
        Name of a built-in RAG corpus. Recommended:
        ``"journal_articles_2025"``. This uses the bundled rebuilt
        corpus of metacoupling/telecoupling journal-paper records with
        BGE-base embeddings. Leave both ``rag_corpus`` and
        ``rag_papers_dir`` unset to disable RAG.
    rag_top_k:
        Maximum number of evidence passages to retrieve per analysis.
        Default ``8``. In ``rag_mode="pre_retrieval"`` this also bounds
        the number of ``<retrieved_literature>`` passages injected into
        the user message and therefore the number of valid
        ``[T{k}:1]..[T{k}:N]`` citation labels for the LLM in turn ``k``.
    rag_mode:
        Selects how RAG integrates with the LLM call. Two options:

        - ``"pre_retrieval"`` (default): retrieve corpus passages
          **before** the LLM runs, embed them in the user message as
          a labeled ``<retrieved_literature turn="k">`` XML block, and
          instruct the LLM (via a citation-rules layer in the system
          prompt) to cite them inline as ``[Tk:N]``. After the LLM
          responds, invalid citation tokens are stripped (with a
          logged warning) by
          :func:`~metacouplingllm.knowledge.citations.sanitize_turn_citations`,
          and the same evidence block as the post-hoc path is appended
          to the formatted output. This is the recommended mode and
          gives the LLM literature to ground its analysis in rather
          than relying purely on training memory.
        - ``"post_hoc"`` (alternative): the LLM generates from training
          memory, then a keyword-overlap pass annotates ``[Tk:N]``
          citations onto sentences that match retrieved passages. Use
          this mode when downstream tooling expects citations to be
          assigned by post-hoc keyword matching rather than inline by
          the LLM.

        On ``refine()`` in pre_retrieval mode, the merged retrieval
        query is the **original** research description (anchored at
        ``analyze()`` time and never overwritten) plus the new
        refinement text, in a labeled structure — see
        :meth:`refine` for details.
    rag_max_chunks_per_paper:
        Maximum number of chunks from the same paper that may appear
        in a single retrieval result. Default ``3``. Set to ``1`` for
        the legacy one-chunk-per-paper behavior, or raise to ``5``
        (or higher) when a single paper is expected to be strongly
        relevant and its key evidence is spread across many sections
        — for example, a systematic framework paper whose results
        section has been split by the chunker into several distinct
        sub-topic chunks (inbound vs outbound, aggregate vs bilateral).
        Higher values improve coverage at the cost of reduced
        paper-level diversity in the retrieved set.
    rag_structured_extraction:
        If ``True`` and ``rag_mode="pre_retrieval"`` with a RAG engine
        configured, runs a second LLM pass over the already-retrieved
        passages to extract systems and flows that may have been missed
        or under-specified in the free-form analysis. Covers all three
        system roles (sending, receiving, spillover) and produces a
        schema-validated list of supplementary flows. Results are
        rendered as a clearly labelled ``SUPPLEMENTARY STRUCTURED
        EXTRACTION`` block in ``AnalysisResult.formatted`` and exposed
        programmatically as ``AnalysisResult.structured_supplement``
        (a dict keyed by ``additional_sending_mentions``,
        ``additional_receiving_mentions``,
        ``additional_spillover_mentions``,
        ``sending_subsystem_fills``, ``receiving_subsystem_fills``,
        ``spillover_subsystem_fills``, and ``supplementary_flows``).
        The main analysis body is not modified — the supplement is
        purely additive so the reader can always tell LLM-authored
        content apart from RAG-extracted content. Defaults to
        ``False`` because this adds one extra LLM call per
        ``analyze()`` / ``refine()`` turn.
    web_structured_extraction:
        If ``True`` and ``web_search=True``, runs a second LLM pass over the
        web results to extract validated map-ready countries and flows.
        These signals are used as conservative hints for map generation and
        are exposed on ``AnalysisResult.web_map_signals``. Recommended when
        ``web_search=True`` and ``auto_map=True`` are both enabled.

    Example
    -------
    >>> from openai import OpenAI
    >>> from metacouplingllm import (
    ...     JOURNAL_ARTICLES_2025,
    ...     MetacouplingAssistant,
    ...     OpenAIAdapter,
    ... )
    >>> advisor = MetacouplingAssistant(
    ...     OpenAIAdapter(OpenAI(), model="gpt-4o"),
    ...     rag_corpus=JOURNAL_ARTICLES_2025,
    ...     web_search=True,
    ...     web_structured_extraction=True,
    ...     auto_map=True,
    ... )
    >>> result = advisor.analyze("My study examines Michigan's pork exports...")
    >>> print(result.formatted)
    >>> if result.map:
    ...     result.map.savefig("map.png", dpi=150)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_examples: int = 2,
        verbose: bool = False,
        recommend_papers: bool = False,
        max_recommendations: int = 5,
        auto_map: bool = False,
        adm1_shapefile: str | Path | None = None,
        adm0_shapefile: str | Path | None = None,
        rag_papers_dir: str | Path | None = None,
        rag_corpus: str | None = None,
        rag_top_k: int = 8,
        rag_backend: str = "auto",
        rag_mode: str = "pre_retrieval",
        rag_max_chunks_per_paper: int = 3,
        rag_structured_extraction: bool = False,
        web_search: bool = False,
        web_search_max_results: int = 10,
        web_structured_extraction: bool = False,
        web_structured_min_confidence: float = 0.7,
        web_structured_max_targets: int = 6,
        rag_min_score: float | None = None,
        coupling_analysis: bool = True,
    ) -> None:
        if rag_mode not in _VALID_RAG_MODES:
            raise ValueError(
                f"rag_mode must be one of {sorted(_VALID_RAG_MODES)}, "
                f"got {rag_mode!r}"
            )
        self._client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._verbose = verbose
        self._recommend_papers = recommend_papers
        self._max_recommendations = max_recommendations
        self._auto_map = auto_map
        self._adm1_shapefile = adm1_shapefile
        self._adm0_shapefile = adm0_shapefile
        self._rag_top_k = rag_top_k
        self._rag_backend = rag_backend
        self._rag_mode = rag_mode
        self._rag_max_chunks_per_paper = max(1, int(rag_max_chunks_per_paper))
        self._rag_structured_extraction = rag_structured_extraction
        self._web_search = web_search
        self._web_search_max_results = web_search_max_results
        # Auto-enable structured web extraction when both web_search and
        # auto_map are on — the map needs structured data to work well.
        if web_structured_extraction is False and web_search and auto_map:
            web_structured_extraction = True
        self._web_structured_extraction = web_structured_extraction
        self._web_structured_min_confidence = web_structured_min_confidence
        self._web_structured_max_targets = web_structured_max_targets
        self._rag_min_score = rag_min_score
        self._last_web_results: list[dict[str, str]] = []
        self._last_web_map_signals: dict[str, object] | None = None
        self._last_map_notice: str | None = None
        self._last_flow_parse_warnings: list[dict[str, str]] = []
        # Records what map type ``_generate_map`` actually rendered on
        # the most recent call.  ``"adm1"`` / ``"country"`` / ``None``
        # (when no figure was produced).  ``_build_result`` reads this
        # so the user-facing notice always describes what was actually
        # drawn, instead of recomputing the type from the same inputs
        # and risking divergence when the renderer's ADM1 attempt
        # silently falls through to a country-level map.
        self._last_map_type: str | None = None
        self._prompt_builder = PromptBuilder(max_examples=max_examples)
        self._formatter = AnalysisFormatter()

        # Pre-retrieval RAG state. Set in analyze() / refine().
        # `_original_query` is anchored to the very first analyze() call
        # and survives across refine()s so the merged retrieval query
        # always preserves the original topic. `_last_rag_hits` is the
        # most recent list of retrieved passages — used both for prompt
        # injection and for the post-LLM evidence block.
        self._last_rag_hits: list[RetrievalResult] | None = None
        self._original_query: str | None = None

        rag_source = self._resolve_rag_source(
            rag_papers_dir=rag_papers_dir,
            rag_corpus=rag_corpus,
        )

        # Initialise RAG engine if a custom directory or built-in corpus
        # is selected.
        self._rag_engine = None
        if rag_source is not None:
            try:
                from metacouplingllm.knowledge.rag import RAGEngine

                self._rag_engine = RAGEngine(
                    papers_dir=rag_source,
                    verbose=verbose,
                    backend=rag_backend,
                )
                self._rag_engine.load()

                if self._rag_engine.total_chunks == 0:
                    print(
                        "[MetacouplingAssistant] WARNING: RAG engine loaded "
                        "but has 0 chunks. Evidence retrieval will return "
                        "no results."
                    )
            except Exception as exc:
                print(
                    f"[MetacouplingAssistant] WARNING: RAG initialisation "
                    f"failed: {exc}"
                )
                self._rag_engine = None
        self._history: list[Message] = []
        self._turn: int = 0

        # Per-turn citation-validation state. Populated at retrieval
        # time in analyze() / refine() / _analyze_rag_only(), read by
        # sanitize_turn_citations to keep `[Tk:N]` tokens whose `(k, N)`
        # is in-range for the turn that produced the block. Cleared
        # alongside the corresponding history.
        self._turn_passage_counts: dict[int, int] = {}
        self._turn_web_counts: dict[int, int] = {}
        self._rag_turn_passage_counts: dict[int, int] = {}
        self._rag_turn_web_counts: dict[int, int] = {}

        # RAG-only mode (coupling_analysis=False). Maintains a separate
        # multi-turn conversation history; framework `_history` is left
        # untouched so the two modes don't interfere.
        self._coupling_analysis = bool(coupling_analysis)
        self._rag_history: list[Message] = []
        self._rag_turn: int = 0
        if not self._coupling_analysis:
            self._warn_disabled_framework_options(
                auto_map=auto_map,
                recommend_papers=recommend_papers,
                rag_structured_extraction=rag_structured_extraction,
                web_structured_extraction=web_structured_extraction,
            )

    @staticmethod
    def _warn_disabled_framework_options(
        *,
        auto_map: bool,
        recommend_papers: bool,
        rag_structured_extraction: bool,
        web_structured_extraction: bool,
    ) -> None:
        """Print a notice for framework-only options ignored in RAG mode."""
        disabled = []
        if auto_map:
            disabled.append("auto_map")
        if recommend_papers:
            disabled.append("recommend_papers")
        if rag_structured_extraction:
            disabled.append("rag_structured_extraction")
        if web_structured_extraction:
            disabled.append("web_structured_extraction")
        if disabled:
            print(
                "[MetacouplingAssistant] coupling_analysis=False — the "
                "following framework-only options are ignored: "
                + ", ".join(disabled)
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_rag_source(
        *,
        rag_papers_dir: str | Path | None,
        rag_corpus: str | None,
    ) -> str | Path | None:
        """Resolve user-facing RAG corpus options to a RAGEngine source."""
        if rag_papers_dir is not None and rag_corpus is not None:
            raise ValueError(
                "Pass either rag_corpus or rag_papers_dir, not both. "
                "Use rag_corpus='journal_articles_2025' for the bundled "
                "corpus, or rag_papers_dir=... for a custom folder."
            )
        if rag_papers_dir is not None:
            return rag_papers_dir
        if rag_corpus is None:
            return None

        normalized = str(rag_corpus).strip().lower().replace("-", "_")
        if normalized in _BUILTIN_RAG_CORPUS_ALIASES:
            return _BUILTIN_RAG_SENTINEL
        raise ValueError(
            f"Unknown rag_corpus={rag_corpus!r}. Supported built-in "
            f"corpora: {sorted(_BUILTIN_RAG_CORPUS_ALIASES)}"
        )

    @property
    def conversation_turns(self) -> int:
        """Number of completed RAG-mode turns in the current session.

        Returns 0 in framework mode (``coupling_analysis=True``) or
        before the first :meth:`analyze` call in RAG mode. Reset by
        :meth:`clear_history`.
        """
        return self._rag_turn if not self._coupling_analysis else 0

    def clear_history(self) -> None:
        """Reset all conversation state.

        Clears both the framework-mode history (used by
        :meth:`refine`) and the RAG-mode conversation history. Safe to
        call from either mode.
        """
        self._history.clear()
        self._turn = 0
        self._turn_passage_counts.clear()
        self._turn_web_counts.clear()
        self._last_rag_hits = None
        self._original_query = None
        self._rag_history.clear()
        self._rag_turn = 0
        self._rag_turn_passage_counts.clear()
        self._rag_turn_web_counts.clear()

    def _analyze_rag_only(self, query: str) -> RAGResult:
        """RAG-only Q&A path used when ``coupling_analysis=False``.

        Multi-turn: appends to ``self._rag_history`` so follow-up
        questions can use prior turns as context. Citations are
        turn-scoped (``[Tk:N]``) so prior-turn references remain
        stable in conversation history.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        next_turn = self._rag_turn + 1

        # Optional web search (per-turn). The map-signal extraction is
        # framework-only, so we deliberately skip extract_web_map_signals.
        web_context: str | None = None
        web_sources: list[dict[str, str]] | None = None
        if self._web_search:
            try:
                from metacouplingllm.knowledge.websearch import (
                    AnthropicWebSearchBackend,
                    GeminiWebSearchBackend,
                    GrokWebSearchBackend,
                    OpenAIWebSearchBackend,
                    format_web_context,
                    search_web,
                )

                print("[MetacouplingAssistant] (RAG mode) Searching the web...")
                web_backend = None
                if isinstance(self._client, OpenAIAdapter):
                    web_backend = OpenAIWebSearchBackend(
                        client=self._client.raw_client,
                        model=self._client.model,
                        reasoning="default",
                    )
                elif isinstance(self._client, AnthropicAdapter):
                    web_backend = AnthropicWebSearchBackend(
                        client=self._client.raw_client,
                        model=self._client.model,
                    )
                elif isinstance(self._client, GeminiAdapter):
                    web_backend = GeminiWebSearchBackend(
                        client=self._client.raw_client,
                        model=self._client.model,
                    )
                elif isinstance(self._client, GrokAdapter):
                    web_backend = GrokWebSearchBackend(
                        client=self._client.raw_client,
                        model=self._client.model,
                    )
                web_search_metadata: dict[str, object] = {}
                web_sources = search_web(
                    query.strip(),
                    max_results=self._web_search_max_results,
                    backend=web_backend,
                    metadata=web_search_metadata,
                )
                backend_used = web_search_metadata.get("backend_used")
                if backend_used:
                    fallback_from = web_search_metadata.get("fallback_from")
                    if fallback_from:
                        print(
                            f"[MetacouplingAssistant] (RAG mode) Web search via "
                            f"{backend_used} — fallback after "
                            f"{fallback_from}"
                        )
                    else:
                        print(
                            f"[MetacouplingAssistant] (RAG mode) Web search via "
                            f"{backend_used}"
                        )
                if web_sources:
                    web_context = format_web_context(
                        web_sources, turn=next_turn,
                    )
                    print(
                        f"[MetacouplingAssistant] (RAG mode) Web search returned "
                        f"{len(web_sources)} results."
                    )
                else:
                    print(
                        "[MetacouplingAssistant] (RAG mode) Web search returned "
                        "0 results."
                    )
            except Exception as exc:
                print(
                    "[MetacouplingAssistant] (RAG mode) Web search failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                web_sources = None
                web_context = None

        # RAG retrieval — fresh on every turn so follow-ups get the
        # most relevant passages for *their* specific question.
        passages: list[RetrievalResult] = []
        if self._rag_engine is not None:
            try:
                passages = list(
                    self._rag_engine.retrieve(
                        query,
                        top_k=self._rag_top_k,
                        min_score=self._rag_min_score,
                        max_chunks_per_paper=self._rag_max_chunks_per_paper,
                    )
                )
                if self._verbose:
                    print(
                        f"[MetacouplingAssistant] (RAG mode) Retrieved "
                        f"{len(passages)} passages."
                    )
            except Exception as exc:
                logger.warning(
                    "RAG retrieval failed in _analyze_rag_only(): %s. "
                    "Continuing with empty literature block.",
                    exc,
                )
                passages = []

        # Record per-turn passage / web counts before the LLM call so
        # the post-LLM sanitizer can validate `[Tk:N]` tokens emitted by
        # the model (or back-referenced from prior turns).
        self._rag_turn_passage_counts[next_turn] = len(passages)
        self._rag_turn_web_counts[next_turn] = (
            len(web_sources) if web_sources else 0
        )

        # Build this turn's user message: query + literature block
        # (+ optional web block). The literature block is rendered with
        # turn-scoped headers so the LLM sees `[Tk:N]` style labels
        # next to each passage.
        # ``max_chars=_LLM_PASSAGE_MAX_CHARS`` (5000) makes the LLM see
        # the FULL chunk text per passage -- matching the framework
        # analysis path's ``_PASSAGE_MAX_CHARS`` budget.  Without this,
        # the dual-purpose ``format_evidence`` would default to ~300-char
        # excerpts (right for user display, wrong for LLM grounding).
        from metacouplingllm.knowledge.rag import (
            _LLM_PASSAGE_MAX_CHARS,
            format_evidence,
        )

        literature_block = (
            format_evidence(
                passages,
                anchor_text=query,
                turn=next_turn,
                max_chars=_LLM_PASSAGE_MAX_CHARS,
            )
            if passages
            else "No literature passages were retrieved for this query."
        )
        user_parts = [
            query.strip(),
            f'<retrieved_literature turn="{next_turn}">',
            literature_block,
            "</retrieved_literature>",
        ]
        if web_context:
            user_parts.extend([
                "",
                f'<web_search_results turn="{next_turn}">',
                web_context,
                "</web_search_results>",
            ])
        user_msg = "\n\n".join(user_parts)

        # On turn 1, seed the conversation with the system prompt.
        if not self._rag_history:
            self._rag_history.append(
                Message(role="system", content=_RAG_ONLY_SYSTEM_PROMPT)
            )
        self._rag_history.append(Message(role="user", content=user_msg))

        # Token-budget heuristic: warn (verbose only) when the projected
        # message size approaches typical context windows. ~4 chars/token
        # rule of thumb; warn at ~120k tokens (480k chars).
        if self._verbose:
            total_chars = sum(len(m.content) for m in self._rag_history)
            if total_chars > 480_000:
                print(
                    "[MetacouplingAssistant] (RAG mode) conversation is "
                    f"~{total_chars // 4:,} tokens — consider calling "
                    "advisor.clear_history() before the next turn."
                )

        # LLM call (uses full RAG history, NOT framework `_history`).
        response = self._client.chat(
            messages=list(self._rag_history),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Sanitize turn-scoped citations. Keeps `[Tk:N]` for any prior
        # or current turn k whose recorded passage / web counts agree,
        # strips forward references and out-of-range tokens, and
        # silently strips any bare legacy `[N]` / `[W1]` the LLM slipped.
        sanitized, dropped = sanitize_turn_citations(
            response.content,
            turn_passage_counts=self._rag_turn_passage_counts,
            turn_web_counts=self._rag_turn_web_counts,
            current_turn=next_turn,
        )
        if dropped and self._verbose:
            print(
                "[MetacouplingAssistant] (RAG mode) Sanitized "
                f"{len(dropped)} invalid citation token(s): "
                f"{sorted(dropped)}"
            )

        # Append the (sanitized) assistant response to history so the
        # next turn sees it as context.
        self._rag_history.append(
            Message(role="assistant", content=sanitized)
        )
        self._rag_turn = next_turn

        # Extract current-turn cited references from the sanitized
        # answer, dedup'd by paper key and ordered by first appearance.
        # Prior-turn back-references (e.g. `[T1:3]` in a turn-2 answer)
        # are intentionally excluded — they belong to turn 1's bibliography.
        references, reference_passage_ids = (
            self._build_references_from_citations(
                sanitized, passages, current_turn=next_turn,
            )
        )

        retrieval_backend = "embeddings"
        if self._rag_engine is not None:
            try:
                retrieval_backend = str(
                    getattr(self._rag_engine, "backend", "embeddings")
                )
            except Exception:
                retrieval_backend = "embeddings"

        return RAGResult(
            answer=sanitized,
            references=references,
            retrieved_passages=passages,
            web_sources=web_sources,
            turn_number=self._rag_turn,
            usage=response.usage,
            raw=response.content,
            retrieval_backend=retrieval_backend,
            reference_passage_ids=reference_passage_ids,
        )

    @staticmethod
    def _build_references_from_citations(
        answer: str,
        passages: list[RetrievalResult],
        current_turn: int,
    ) -> tuple[list[Paper], list[int]]:
        """Return ``(papers, passage_ids)`` cited in the current turn.

        Both lists are parallel and in first-appearance order, dedup'd
        by paper key:

        - ``papers[i]`` is the i-th unique cited paper as a ``Paper``.
        - ``passage_ids[i]`` is the passage ID (the ``N`` in
          ``[T{current_turn}:N]``) at which it was first cited.

        Only ``[Tk:N]`` tokens whose ``k`` matches ``current_turn`` are
        counted as current-turn references. Prior-turn back-references
        (``[T1:N]`` appearing in a turn-2 answer) are deliberately
        excluded — they have already been listed in the prior turn's
        REFERENCES block. Web tokens (``[Tk:Wn]``) are also ignored.
        """
        if not passages:
            return [], []
        seen: set[str] = set()
        out_papers: list[Paper] = []
        out_ids: list[int] = []
        for match in _CITATION_TOKEN_RE.finditer(answer):
            try:
                k = int(match.group(1))
            except ValueError:
                continue
            kind = match.group(2)
            if kind == "W":
                continue  # web citation, not a literature ref
            if k != current_turn:
                continue  # prior-turn back-reference
            try:
                idx = int(match.group(3))
            except ValueError:
                continue
            if not (1 <= idx <= len(passages)):
                continue
            chunk = passages[idx - 1].chunk
            if chunk.paper_key in seen:
                continue
            seen.add(chunk.paper_key)
            try:
                year_int = int(chunk.year)
            except (TypeError, ValueError):
                year_int = 0
            out_papers.append(
                Paper(
                    key=chunk.paper_key,
                    title=chunk.paper_title,
                    authors=chunk.authors,
                    year=year_int,
                )
            )
            out_ids.append(idx)
        return out_papers, out_ids

    def analyze(
        self, research_description: str
    ) -> AnalysisResult | RAGResult:
        """Run an analysis on a research description.

        Behaviour depends on the ``coupling_analysis`` flag set at
        construction:

        - ``coupling_analysis=True`` (default): runs the full framework
          pipeline and returns an :class:`AnalysisResult`. Resets any
          existing framework conversation history and starts fresh.
        - ``coupling_analysis=False``: runs a RAG-only literature Q&A
          and returns a :class:`RAGResult`. Multi-turn — each call
          appends to the running RAG conversation; call
          :meth:`clear_history` to reset.

        Parameters
        ----------
        research_description:
            In framework mode, the researcher's description of their
            study, research idea, or draft. In RAG mode, a free-form
            literature question (or follow-up).

        Returns
        -------
        :class:`AnalysisResult` (framework mode) or :class:`RAGResult`
        (RAG-only mode).

        Notes
        -----
        Framework mode: when ``rag_mode="pre_retrieval"`` (the default)
        and a RAG engine is configured, this method retrieves the top
        ``rag_top_k`` passages from the corpus *before* calling the LLM
        and embeds them in the user message as a
        ``<retrieved_literature>`` block, so the LLM can cite them
        inline as ``[1]..[N]``. The retrieved hits and the original
        research description are stored on the instance for later reuse
        by :meth:`refine`.
        """
        if not self._coupling_analysis:
            return self._analyze_rag_only(research_description)

        # Reset for a new framework analysis
        self._history.clear()
        self._turn = 0
        self._turn_passage_counts.clear()
        self._turn_web_counts.clear()
        # Reset pre-retrieval state — even if RAG is disabled or fails,
        # leaving stale hits from a previous session would be wrong.
        self._last_rag_hits = None
        self._original_query = research_description

        next_turn = self._turn + 1

        # Optional: pre-search web context
        web_context: str | None = None
        self._last_web_results: list[dict[str, str]] = []
        self._last_web_map_signals = None
        if self._web_search:
            try:
                from metacouplingllm.knowledge.websearch import (
                    AnthropicWebSearchBackend,
                    GeminiWebSearchBackend,
                    GrokWebSearchBackend,
                    OpenAIWebSearchBackend,
                    extract_web_map_signals,
                    format_web_context,
                    format_web_map_signals_context,
                    search_web,
                )

                print("[MetacouplingAssistant] Searching the web...")
                # Auto-wire the native web-search tool for the matching
                # provider. If the backend fails or returns no results,
                # search_web() falls back to the DuckDuckGo cascade.
                web_backend = None
                if isinstance(self._client, OpenAIAdapter):
                    web_backend = OpenAIWebSearchBackend(
                        client=self._client.raw_client,
                        model=self._client.model,
                        reasoning="default",
                    )
                elif isinstance(self._client, AnthropicAdapter):
                    web_backend = AnthropicWebSearchBackend(
                        client=self._client.raw_client,
                        model=self._client.model,
                    )
                elif isinstance(self._client, GeminiAdapter):
                    web_backend = GeminiWebSearchBackend(
                        client=self._client.raw_client,
                        model=self._client.model,
                    )
                elif isinstance(self._client, GrokAdapter):
                    web_backend = GrokWebSearchBackend(
                        client=self._client.raw_client,
                        model=self._client.model,
                    )
                web_search_metadata: dict[str, object] = {}
                self._last_web_results = search_web(
                    research_description.strip(),
                    max_results=self._web_search_max_results,
                    backend=web_backend,
                    metadata=web_search_metadata,
                )
                backend_used = web_search_metadata.get("backend_used")
                if backend_used:
                    fallback_from = web_search_metadata.get("fallback_from")
                    if fallback_from:
                        print(
                            f"[MetacouplingAssistant] Web search via "
                            f"{backend_used} \u2014 fallback after "
                            f"{fallback_from}"
                        )
                    else:
                        print(
                            f"[MetacouplingAssistant] Web search via "
                            f"{backend_used}"
                        )
                if self._last_web_results:
                    web_context = format_web_context(
                        self._last_web_results, turn=next_turn,
                    )
                    if self._web_structured_extraction:
                        try:
                            self._last_web_map_signals = (
                                extract_web_map_signals(
                                    research_description.strip(),
                                    self._last_web_results,
                                    self._client,
                                    min_confidence=(
                                        self._web_structured_min_confidence
                                    ),
                                    max_targets=self._web_structured_max_targets,
                                )
                            )
                        except Exception as exc:
                            print(
                                "[MetacouplingAssistant] Structured web "
                                f"extraction failed: {exc}"
                            )
                            self._last_web_map_signals = None
                        if self._last_web_map_signals:
                            structured_context = (
                                format_web_map_signals_context(
                                    self._last_web_map_signals,
                                )
                            )
                            if structured_context:
                                web_context += "\n\n" + structured_context
                            if self._verbose:
                                structured_flows = self._last_web_map_signals.get(
                                    "flows", []
                                )
                                structured_receiving = (
                                    self._last_web_map_signals.get(
                                        "receiving_systems", []
                                    )
                                )
                                print(
                                    "[MetacouplingAssistant] Structured web "
                                    "extraction accepted "
                                    f"{len(structured_receiving)} receiving "
                                    f"systems and {len(structured_flows)} "
                                    "flows."
                                )
                        elif self._verbose:
                            print(
                                "[MetacouplingAssistant] Structured web "
                                "extraction found no validated map signals."
                            )
                    print(
                        f"[MetacouplingAssistant] Web search returned "
                        f"{len(self._last_web_results)} results."
                    )
                else:
                    print(
                        "[MetacouplingAssistant] Web search returned 0 results."
                    )
            except Exception as exc:
                if self._verbose:
                    print(f"[MetacouplingAssistant] Web search failed: {exc}")

        # Pre-retrieval RAG: fetch corpus passages BEFORE calling the LLM
        # so we can inject them into the user message and the LLM can
        # cite them inline. Disabled in post_hoc mode (which retrieves
        # and annotates AFTER the LLM responds) and when no RAG engine
        # is configured.
        if self._rag_mode == "pre_retrieval" and self._rag_engine is not None:
            try:
                results = self._rag_engine.retrieve(
                    research_description,
                    top_k=self._rag_top_k,
                    min_score=self._rag_min_score,
                    max_chunks_per_paper=self._rag_max_chunks_per_paper,
                )
                self._last_rag_hits = list(results)
                if self._verbose:
                    print(
                        f"[MetacouplingAssistant] Pre-retrieval RAG: "
                        f"{len(self._last_rag_hits)} passages."
                    )
            except Exception as exc:
                logger.warning(
                    "Pre-retrieval RAG failed in analyze(): %s. "
                    "Continuing with empty literature block.",
                    exc,
                )
                self._last_rag_hits = []

        # Build the system prompt (examples selected based on context).
        # The citation rules layer is only injected when pre_retrieval
        # mode is active so post_hoc users see no behavioral change.
        system_prompt = self._prompt_builder.build_system_prompt(
            research_context=research_description,
            web_context=web_context,
            include_citation_rules=(self._rag_mode == "pre_retrieval"),
        )
        self._history.append(Message(role="system", content=system_prompt))

        # Build the user message. In pre_retrieval mode the literature
        # passages are prepended as a <retrieved_literature> XML block.
        # In post_hoc mode (or when RAG is disabled) literature_passages
        # is None and the original behavior is preserved exactly.
        literature_for_msg = (
            self._last_rag_hits
            if self._rag_mode == "pre_retrieval"
            and self._rag_engine is not None
            else None
        )

        # Record per-turn citation-validation counts so the post-LLM
        # sanitizer can keep / strip `[Tk:N]` tokens correctly.
        self._turn_passage_counts[next_turn] = (
            len(literature_for_msg) if literature_for_msg else 0
        )
        self._turn_web_counts[next_turn] = len(self._last_web_results or [])

        user_msg = self._prompt_builder.build_initial_message(
            research_description,
            literature_passages=literature_for_msg,
            turn=next_turn,
        )
        self._history.append(Message(role="user", content=user_msg))

        if self._verbose:
            print(f"[MetacouplingAssistant] System prompt: {len(system_prompt)} chars")
            print(f"[MetacouplingAssistant] Sending initial analysis request...")

        # Call the LLM
        response = self._call_llm()

        # Record assistant response in history
        self._history.append(Message(role="assistant", content=response.content))
        self._turn = next_turn

        return self._build_result(response)

    def refine(
        self,
        additional_info: str,
        focus_component: str | None = None,
    ) -> AnalysisResult:
        """Refine the analysis with additional information or focus.

        Must be called after :meth:`analyze`.

        Parameters
        ----------
        additional_info:
            Follow-up questions, corrections, or additional context.
        focus_component:
            Optional component to elaborate on (e.g., ``"systems"``,
            ``"flows"``, ``"agents"``, ``"causes"``, ``"effects"``).

        Returns
        -------
        An :class:`AnalysisResult` with the refined analysis.

        Raises
        ------
        RuntimeError
            If called before :meth:`analyze`.

        Notes
        -----
        When ``rag_mode="pre_retrieval"``, this method always re-runs
        retrieval using a **labeled merged query** that combines the
        original research description (anchored at :meth:`analyze` time
        and never overwritten) with the new ``additional_info``::

            Original research question:
            <original research_description>

            Refinement request:
            <additional_info>

        The labeled structure gives the embedding model clearer
        structural cues than raw concatenation and prevents the
        refinement text from "drifting" the retrieval away from the
        original topic. Fresh hits replace ``self._last_rag_hits`` and
        are injected into the new user message.

        Citations are turn-scoped (``[Tk:N]``); turn 1's citations
        remain stable across refines, and the LLM may back-reference
        prior-turn evidence by copying the original token verbatim.
        """
        if self._turn == 0:
            raise RuntimeError(
                "Cannot refine before an initial analysis. "
                "Call `analyze()` first."
            )

        next_turn = self._turn + 1

        # Pre-retrieval RAG on refinement: re-retrieve with the merged
        # query so the LLM gets fresh evidence for the refined ask.
        if (
            self._rag_mode == "pre_retrieval"
            and self._rag_engine is not None
            and self._original_query is not None
        ):
            merged_query = (
                f"Original research question:\n{self._original_query}\n\n"
                f"Refinement request:\n{additional_info}"
            )
            try:
                results = self._rag_engine.retrieve(
                    merged_query,
                    top_k=self._rag_top_k,
                    min_score=self._rag_min_score,
                    max_chunks_per_paper=self._rag_max_chunks_per_paper,
                )
                self._last_rag_hits = list(results)
                if self._verbose:
                    print(
                        f"[MetacouplingAssistant] Refine pre-retrieval: "
                        f"{len(self._last_rag_hits)} passages."
                    )
            except Exception as exc:
                logger.warning(
                    "Pre-retrieval RAG failed in refine(): %s. "
                    "Continuing with empty literature block.",
                    exc,
                )
                self._last_rag_hits = []

        literature_for_msg = (
            self._last_rag_hits
            if self._rag_mode == "pre_retrieval"
            and self._rag_engine is not None
            else None
        )

        # Record per-turn citation-validation counts. Refine() does NOT
        # re-run web search (only re-retrieves literature), so we
        # record the most recent web-result count under this turn so
        # the LLM can still back-reference prior-turn `[Tk:Wn]` tokens.
        self._turn_passage_counts[next_turn] = (
            len(literature_for_msg) if literature_for_msg else 0
        )
        self._turn_web_counts[next_turn] = 0

        user_msg = self._prompt_builder.build_refinement_message(
            additional_info,
            focus_component=focus_component,
            literature_passages=literature_for_msg,
            turn=next_turn,
        )
        self._history.append(Message(role="user", content=user_msg))

        if self._verbose:
            print(f"[MetacouplingAssistant] Refining (turn {next_turn})...")

        response = self._call_llm()
        self._history.append(Message(role="assistant", content=response.content))
        self._turn = next_turn

        return self._build_result(response)

    def reset(self) -> None:
        """Clear conversation history and pre-retrieval state."""
        self._history.clear()
        self._turn = 0
        self._turn_passage_counts.clear()
        self._turn_web_counts.clear()
        self._last_rag_hits = None
        self._original_query = None
        if self._verbose:
            print("[MetacouplingAssistant] Conversation reset.")

    @property
    def history(self) -> list[Message]:
        """Return a copy of the conversation history."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        """Return the number of completed turns."""
        return self._turn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self) -> LLMResponse:
        """Send the current history to the LLM client."""
        return self._client.chat(
            messages=list(self._history),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    # ------------------------------------------------------------------
    # Auto-map helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_adm1_from_analysis(parsed: ParsedAnalysis) -> str | None:
        """Try to resolve a subnational ADM1 code from the parsed analysis.

        Scans the sending, receiving, and spillover systems' ``geographic_scope``
        and ``name`` fields for subnational region names (e.g., "Michigan,
        United States") and attempts to resolve them to ADM1 codes.

        Falls back to scanning flow directions and coupling classification
        if systems don't contain resolvable region names.

        Returns the first successfully resolved ADM1 code, or ``None``.
        """
        from metacouplingllm.knowledge.adm1_pericoupling import resolve_adm1_code

        def _clean_candidate_text(text: str) -> str:
            text = re.sub(r"\[[^\]]+\]", " ", text)
            text = text.replace("*", " ").replace("`", " ")
            text = re.sub(r"\s+", " ", text)
            return text.strip().rstrip(".,;:")

        def _looks_like_direct_location(text: str) -> bool:
            lowered = text.lower().strip()
            if not lowered:
                return False
            if any(
                marker in lowered
                for marker in (
                    "such as",
                    "for example",
                    "for instance",
                    "e.g.",
                    "including",
                    "especially",
                    "depending on",
                    "potentially",
                    "plausibly",
                    "likely ",
                    "possible ",
                )
            ):
                return False
            words = re.findall(r"[A-Za-z][A-Za-z'’-]*", lowered)
            return 0 < len(words) <= 8

        def _try_resolve_text(text: str) -> str | None:
            """Try resolving a text string to an ADM1 code."""
            text = _clean_candidate_text(text)
            if not text:
                return None
            # Split on comma to separate region from country hint
            parts = [p.strip() for p in text.split(",")]
            region = parts[0]
            country_hint: str | None = None
            if len(parts) > 1:
                country_hint = parts[-1]
                hint_code = resolve_country_code(country_hint)
                if hint_code:
                    country_hint = hint_code
                else:
                    country_hint = None
            if _looks_like_direct_location(region):
                code = resolve_adm1_code(region, country=country_hint)
                if code:
                    return code
            # Also try each comma-separated part individually
            for part in parts:
                part = _clean_candidate_text(part)
                if part and len(part) > 2 and _looks_like_direct_location(part):
                    code = resolve_adm1_code(part)
                    if code:
                        return code
            return None

        # --- Relevance guard ------------------------------------------------
        # After resolving an ADM1 code, verify the corresponding country
        # actually appears in the analysis text.  This prevents false
        # positives like "Gulf" (US Gulf Coast) → PNG008 (Papua New
        # Guinea) when the analysis is about US corn exports.
        from metacouplingllm.knowledge.adm1_pericoupling import get_adm1_country

        def _build_analysis_country_mentions() -> set[str]:
            """Collect ISO codes of all countries mentioned in the analysis."""
            codes: set[str] = set()
            combined = " ".join(parsed.iter_text_fragments())
            # Try to resolve country names found in the text
            for word_group in re.findall(
                r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", combined
            ):
                code = resolve_country_code(word_group)
                if code:
                    codes.add(code)
            # Also pick up explicit 3-letter ISO codes like "USA", "BRA"
            for m in re.findall(r"\b([A-Z]{3})\b", combined):
                name = get_country_name(m)
                if name:
                    codes.add(m)
            return codes

        _mentioned_countries: set[str] | None = None

        def _is_relevant_adm1(code: str) -> bool:
            """Check if the ADM1 code's country appears in the analysis."""
            nonlocal _mentioned_countries
            if _mentioned_countries is None:
                try:
                    _mentioned_countries = _build_analysis_country_mentions()
                except Exception:
                    _mentioned_countries = set()  # Fallback: accept all
            adm1_country = get_adm1_country(code)
            if not adm1_country:
                return True  # Can't verify — accept optimistically
            return adm1_country in _mentioned_countries

        # Wrap _try_resolve_text with the relevance guard
        _orig_try_resolve = _try_resolve_text

        def _try_resolve_text_guarded(text: str) -> str | None:
            code = _orig_try_resolve(text)
            if code and not _is_relevant_adm1(code):
                return None
            return code

        # --- Step 1: Scan system dict entries (nested format) ---
        for role in ("focal", "adjacent", "sending", "receiving", "spillover"):
            for entry in parsed.get_system_entries(role):
                for key in ("geographic_scope", "name"):
                    text = entry.get(key, "")
                    if text:
                        code = _try_resolve_text_guarded(text)
                        if code:
                            return code

        # --- Step 2: Scan flow directions ---
        for flow in parsed.iter_flow_entries():
            direction = flow.get("direction", "")
            if direction:
                # Split arrows and try each endpoint
                for part in _FLOW_ARROW_RE.split(direction):
                    part = part.strip()
                    # Remove parenthesized annotations like "(sending)"
                    part = re.sub(r"\([^)]*\)", "", part).strip()
                    if part:
                        code = _try_resolve_text_guarded(part)
                        if code:
                            return code

        # --- Step 3: Scan coupling classification text ---
        if parsed.coupling_classification:
            # Look for quoted or parenthesized topic descriptions
            # e.g., '"impact of Michigan\'s pork exports"'
            for m in re.finditer(r'["\u201c]([^"\u201d]+)["\u201d]',
                                 parsed.coupling_classification):
                topic = m.group(1)
                # Extract potential region names by splitting on possessives and prepositions
                for chunk in re.split(r"[''']s\s+|\s+of\s+|\s+in\s+|\s+from\s+", topic):
                    chunk = chunk.strip().rstrip(".,;:")
                    if chunk and len(chunk) > 2:
                        code = _try_resolve_text_guarded(chunk)
                        if code:
                            return code

        return None

    @staticmethod
    def _extract_mentioned_adm1_from_text(
        parsed: ParsedAnalysis,
    ) -> set[str]:
        """Return ALL ADM1 codes mentioned in the substantive analysis.

        Used as a fallback when the LLM's ``mentioned_adm1_regions``
        list is empty. Scans parsed systems (sending/receiving/spillover
        names + geographic_scope), flow directions/descriptions, and
        causes/effects. Does NOT scan the coupling classification text
        (which often describes the focal region), and never scans any
        pericoupling database validation text.

        Each candidate is validated against the ADM1 database and
        guarded against false positives (e.g., "Gulf" resolving to
        Papua New Guinea) by requiring the resolved region's country
        to appear in the analysis text.
        """
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_country,
            get_adm1_info,
            resolve_adm1_code,
        )

        # Build the set of countries mentioned in the analysis for the
        # relevance guard.
        country_mentions: set[str] = set()
        combined = " ".join(parsed.iter_text_fragments())
        for word_group in re.findall(
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", combined
        ):
            code = resolve_country_code(word_group)
            if code:
                country_mentions.add(code)
        for m in re.findall(r"\b([A-Z]{3})\b", combined):
            if get_country_name(m):
                country_mentions.add(m)

        def _resolve_candidate(text: str) -> str | None:
            cleaned = re.sub(r"\[[^\]]+\]", " ", text)
            cleaned = cleaned.replace("*", " ").replace("`", " ")
            cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".,;:")
            if not cleaned or len(cleaned) < 3:
                return None
            # Bail out on fuzzy language
            lowered = cleaned.lower()
            if any(
                marker in lowered
                for marker in (
                    "such as", "for example", "e.g.", "including",
                    "especially", "depending", "likely ", "possible ",
                )
            ):
                return None
            parts = [p.strip() for p in cleaned.split(",")]
            country_hint: str | None = None
            if len(parts) > 1:
                hint_code = resolve_country_code(parts[-1])
                if hint_code:
                    country_hint = hint_code
            for part in parts:
                if not part or len(part) < 3:
                    continue
                code = resolve_adm1_code(part, country=country_hint)
                if code is None:
                    continue
                # Relevance guard: the ADM1 region's country must
                # appear in the analysis text.
                adm1_country = get_adm1_country(code)
                if adm1_country and country_mentions and (
                    adm1_country not in country_mentions
                ):
                    continue
                if get_adm1_info(code) is not None:
                    return code
            return None

        found: set[str] = set()

        # Scan substantive system fields only — NOT `name` or
        # `geographic_scope`, which typically hold list-style
        # enumerations of neighbouring regions (e.g., "Para,
        # Rondonia, Tocantins, Amazonas") that the LLM echoed from
        # the adjacency database hint. Such lists are not evidence
        # of actual interaction.
        for role in ("focal", "adjacent", "sending", "receiving", "spillover"):
            for entry in parsed.get_system_entries(role):
                for key in ("human_subsystem", "natural_subsystem", "description"):
                    text = entry.get(key, "")
                    if not text:
                        continue
                    # Try the full text first
                    code = _resolve_candidate(text)
                    if code:
                        found.add(code)
                    # Also scan comma-separated sub-chunks
                    for chunk in re.split(r"[;\u2022]|\s+and\s+", text):
                        code = _resolve_candidate(chunk)
                        if code:
                            found.add(code)
            # Flat string values are skipped for the same reason
            # — they rarely describe substantive interactions.

        # Scan flow directions and descriptions
        for flow in parsed.iter_flow_entries():
            for key in ("direction", "description"):
                text = flow.get(key, "")
                if not text:
                    continue
                # Arrow-split for directions
                for part in _FLOW_ARROW_RE.split(text):
                    part = re.sub(r"\([^)]*\)", "", part).strip()
                    if part:
                        code = _resolve_candidate(part)
                        if code:
                            found.add(code)
                # Also try chunk-splitting the description
                for chunk in re.split(r"[;\u2022]|\s+and\s+", text):
                    code = _resolve_candidate(chunk)
                    if code:
                        found.add(code)

        # Scan causes and effects lists
        for kind in ("causes", "effects"):
            for _section_name, _category, item in parsed.iter_category_items(kind):
                if not isinstance(item, str):
                    continue
                for chunk in re.split(r"[;\u2022]|\s+and\s+", item):
                    code = _resolve_candidate(chunk)
                    if code:
                        found.add(code)

        return found

    @staticmethod
    def _resolve_flows_for_map(
        parsed: ParsedAnalysis,
        focal_country: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Resolve flow directions to country-level endpoints for map arrows.

        Returns ``(resolved_flows, parse_warnings)``.  Each warning is a
        dict ``{"direction": str, "category": str, "reason": str}``
        describing a flow that could not be turned into any arrow.

        Supranational targets (EU / ASEAN / NAFTA / USMCA) are detected
        via ``expand_supranational``.  When mentioned without any of
        their member countries also appearing in the same endpoint
        context, they are emitted as **single-region** flow entries
        carrying ``target_supranational`` and
        ``target_supranational_members`` fields, which the map renderer
        uses to draw one labelled arrow at the union centroid plus a
        member-region highlight overlay.

        This version is conservative about speculative examples, but still
        allows softened receiving-market lists such as
        ``"likely distant foreign markets ... such as Mexico, China..."``
        to inform map arrows when no confirmed subnational partner list is
        available.
        """
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_country,
            resolve_adm1_code,
        )
        from metacouplingllm.knowledge.countries import (
            expand_supranational,
            get_country_name,
            resolve_country_code,
            supranational_display_name,
        )

        warnings: list[dict[str, str]] = []

        def _record_warning(
            direction: str, category: str, reason: str,
        ) -> None:
            """Append a parse warning and mirror it to the module logger."""
            warnings.append({
                "direction": direction,
                "category": category,
                "reason": reason,
            })
            logger.warning(
                "Flow direction could not be resolved: %r "
                "(category=%s) — %s. Falling back to systems-level inference.",
                direction, category, reason,
            )

        def _extract_country_list_codes(
            text: str,
            *,
            allow_soft_markers: bool = False,
        ) -> tuple[set[str], set[str]]:
            """Return ``(country_iso_codes, supranational_display_names)``.

            Supranationals are filtered post-hoc: if any of a
            supranational's members also appear in ``codes``, that
            supranational is dropped to avoid double-counting (the
            "EU should only fire when no EU countries detected" rule).
            """
            if not text or not text.strip():
                return set(), set()

            clean = re.sub(r"\[[^\]]+\]", "", text).strip()
            lowered = clean.lower()
            hard_blockers = (
                "for example", "e.g.", "depending on", "depending upon",
                "another market", "another destination",
                "does not specify", "not specify", "not yet",
                "uncertain", "unknown", "would specify whether",
                "should be confirmed", "to be confirmed",
            )
            if any(marker in lowered for marker in hard_blockers):
                return set(), set()

            soft_markers = (
                "such as", "likely", "may ", " may", "might", "could",
                "plausible", "potential", "possible", "include",
                "includes", "including", "proxy",
            )
            if (
                not allow_soft_markers
                and any(marker in lowered for marker in soft_markers)
            ):
                return set(), set()

            chunks = re.split(r"[,;/]+|\band\b|\bor\b", clean)
            codes: set[str] = set()
            supranationals: set[str] = set()
            resolved_chunks: list[str] = []

            for chunk in chunks:
                candidate = re.sub(r"\([^)]*\)", "", chunk).strip().rstrip(".:")
                if not candidate:
                    continue
                code = resolve_country_code(candidate)
                if not code:
                    adm1 = resolve_adm1_code(candidate)
                    if adm1:
                        code = get_adm1_country(adm1)
                if code:
                    codes.add(code)
                    resolved_chunks.append(candidate.lower())
                    continue
                # Supranational fallback (EU / ASEAN / NAFTA / USMCA).
                members = expand_supranational(candidate)
                if members:
                    display = supranational_display_name(members)
                    if display:
                        supranationals.add(display)
                        resolved_chunks.append(candidate.lower())

            if not codes and not supranationals:
                return set(), set()

            residue = lowered
            for chunk in resolved_chunks:
                residue = residue.replace(chunk, " ")
            residue = re.sub(r"\([^)]*\)", " ", residue)
            residue = re.sub(
                r"\b(and|or|countries?|country|markets?|market|regions?|region|"
                r"systems?|system|states?|state|destinations?|destination|"
                r"importers?|exporters?|trade|trading|importing|exporting|"
                r"distant|domestic|foreign|international|major|key|selected|"
                r"target|receiving|sending|spillover|focal|adjacent|"
                r"nonadjacent|other|main|principal|likely|possible|potential|"
                r"plausible|connected|connect|linked|link|including|include|"
                r"includes|such|proxy|confirmed|top)\b",
                " ",
                residue,
            )
            residue = re.sub(r"[^a-z]+", " ", residue)
            leftover_tokens = [tok for tok in residue.split() if len(tok) > 2]
            if len(leftover_tokens) > 1:
                return set(), set()

            # Conditional rule: drop any supranational whose members
            # already appear in ``codes`` -- the LLM listed both the
            # specific countries and the umbrella term, treat the
            # umbrella as redundant.
            filtered_supranationals: set[str] = set()
            for sup in supranationals:
                members = expand_supranational(sup)
                if members and not (codes & set(members)):
                    filtered_supranationals.add(sup)

            return codes, filtered_supranationals

        def _extract_explicit_codes(
            text: str,
        ) -> tuple[set[str], set[str]]:
            return _extract_country_list_codes(
                text,
                allow_soft_markers=False,
            )

        def _extract_proxy_receiving_codes(
            text: str,
        ) -> tuple[set[str], set[str]]:
            lowered = text.lower()
            if not any(
                term in lowered
                for term in (
                    "receiving", "market", "markets", "destination",
                    "destinations", "export", "exports", "import",
                    "imports",
                )
            ):
                return set(), set()
            return _extract_country_list_codes(
                text,
                allow_soft_markers=True,
            )

        role_codes: dict[str, set[str]] = {
            "focal": set(),
            "adjacent": set(),
            "sending": set(),
            "receiving": set(),
            "spillover": set(),
        }
        proxy_role_codes: dict[str, set[str]] = {
            "sending": set(),
            "receiving": set(),
            "spillover": set(),
            "adjacent": set(),
        }
        # Role-level role_codes/proxy_role_codes track ISO codes only.
        # Supranationals are detected per-flow (target side); discarding
        # them here keeps role-based fallback unambiguous.
        for role in role_codes:
            for entry in parsed.get_system_entries(role):
                texts: list[str] = []
                for key in ("name", "geographic_scope"):
                    candidate = entry.get(key, "")
                    if candidate:
                        texts.append(candidate)
                for text in texts:
                    codes_for_text, _ = _extract_explicit_codes(text)
                    role_codes[role].update(codes_for_text)
                    if entry.get("system_scope") == "adjacent":
                        role_codes["adjacent"].update(codes_for_text)
                    if role == "receiving":
                        proxy_codes, _ = _extract_proxy_receiving_codes(
                            text,
                        )
                        proxy_role_codes[role].update(proxy_codes)
        role_codes["sending"].update(role_codes["focal"])

        def _try_resolve(text: str) -> str | None:
            clean = re.sub(r"\([^)]*\)", "", text).strip()
            if not clean:
                return None

            candidates = [clean]
            clause_parts = [
                part.strip()
                for part in re.split(r"[;:]", clean)
                if part.strip()
            ]
            if clause_parts:
                candidates.extend(clause_parts)

            normalised: list[str] = []
            for candidate in candidates:
                simplified = re.sub(
                    r"^(?:effectively|primarily|mainly|mostly|especially|"
                    r"roughly|approximately|about|notably)\s+",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()
                if simplified:
                    normalised.append(simplified)

            for candidate in normalised:
                code = resolve_country_code(candidate)
                if code:
                    return code
                adm1 = resolve_adm1_code(candidate)
                if adm1:
                    return get_adm1_country(adm1)
            return None

        def _resolve_role_reference(text: str) -> set[str]:
            lowered = text.lower()
            codes: set[str] = set()
            if any(
                term in lowered
                for term in (
                    "receiving",
                    "importing",
                    "importer",
                    "importers",
                    "destination",
                    "destinations",
                )
            ):
                codes.update(role_codes.get("receiving", set()))
                codes.update(proxy_role_codes.get("receiving", set()))
            if any(
                term in lowered
                for term in (
                    "spillover",
                    "competing",
                    "competitor",
                    "competitors",
                    "indirectly affected",
                )
            ):
                codes.update(role_codes.get("spillover", set()))
            if any(
                term in lowered
                for term in (
                    "adjacent",
                    "neighboring",
                    "neighbouring",
                    "pericoupled",
                    "nearby",
                )
            ):
                codes.update(role_codes.get("adjacent", set()))
            if any(
                term in lowered
                for term in (
                    "sending",
                    "exporting",
                    "exporter",
                    "exporters",
                    "origin",
                    "origins",
                )
            ):
                codes.update(role_codes.get("sending", set()))
            return codes

        def _resolve_source(text: str) -> set[str]:
            """Source-side resolution -- supranationals are flattened to
            their member countries (no special source-side rendering)."""
            explicit_codes, explicit_sup = _extract_explicit_codes(text)
            if explicit_codes:
                return explicit_codes
            if explicit_sup:
                # Source-side supranational: expand to all members.
                flattened: set[str] = set()
                for sup in explicit_sup:
                    members = expand_supranational(sup) or []
                    flattened.update(members)
                if flattened:
                    return flattened
            code = _try_resolve(text)
            if code:
                return {code}
            return _resolve_role_reference(text)

        def _resolve_target(text: str) -> tuple[set[str], set[str]]:
            """Target-side resolution returning
            ``(country_codes, supranational_names)``."""
            explicit_codes, explicit_sup = _extract_explicit_codes(text)
            if explicit_codes or explicit_sup:
                return explicit_codes, explicit_sup
            code = _try_resolve(text)
            if code:
                return {code}, set()
            return _resolve_role_reference(text), set()

        resolved: list[dict[str, str]] = []
        seen_pairs: set[tuple[str, str, str]] = set()

        for flow in parsed.iter_flow_entries():
            direction = flow.get("direction", "")
            category = flow.get("category", "")
            if not direction:
                continue
            has_connector = bool(_FLOW_HAS_CONNECTOR_RE.search(direction))
            if (
                not has_connector
                and re.search(r"\bwithin\b", direction, re.IGNORECASE)
            ):
                # Intentional skip -- internal flow, no warning.
                continue

            is_bidir = (
                "bidirectional" in direction.lower()
                or "↔" in direction
                or "<->" in direction
                or "<=>" in direction
            )

            src_codes: set[str] = set()
            tgt_codes: set[str] = set()
            tgt_supranationals: set[str] = set()

            between_m = re.search(
                r"[Bb](?:etween|idirectional\s+between)\s+(.+?)\s+and\s+(.+)",
                direction,
            )
            if between_m:
                src_codes = _resolve_source(between_m.group(1))
                tgt_codes, tgt_supranationals = _resolve_target(
                    between_m.group(2),
                )
                is_bidir = True
            else:
                parts = _FLOW_ARROW_RE.split(direction)
                if len(parts) >= 2:
                    src_codes = _resolve_source(parts[0])
                    tgt_codes, tgt_supranationals = _resolve_target(parts[1])

            # Fallback: when the LLM used a generic label like
            # "importing countries" instead of specific country names,
            # harvest concrete countries from the Systems section.
            if not tgt_codes and not tgt_supranationals:
                try:
                    from metacouplingllm.visualization.worldmap import (
                        _extract_all_analysis_countries,
                    )

                    receiving = _extract_all_analysis_countries(parsed).get(
                        "receiving", set()
                    )
                    if receiving:
                        tgt_codes = receiving
                except (ImportError, AttributeError, KeyError):
                    pass  # Viz deps unavailable -- use default resolution

            if not src_codes:
                # When source is generic (e.g., "Importing countries")
                # and target IS the focal country, the unresolved source
                # is the receiving systems (e.g., China sends capital
                # TO Brazil).  Only use receiving -- never spillover,
                # which contains competing exporters, not trade partners.
                if tgt_codes and focal_country in tgt_codes:
                    try:
                        from metacouplingllm.visualization.worldmap import (
                            _extract_all_analysis_countries,
                        )

                        receiving = _extract_all_analysis_countries(
                            parsed,
                        ).get("receiving", set())
                        partner_codes = receiving - tgt_codes
                        if partner_codes:
                            src_codes = partner_codes
                    except (ImportError, AttributeError, KeyError):
                        pass  # Viz deps unavailable
                if not src_codes:
                    src_codes = {focal_country}
            if not tgt_codes and not tgt_supranationals:
                # All resolution paths exhausted and the flow has no
                # endpoints to draw. Record a warning and move on.
                if not has_connector:
                    reason = (
                        "no recognized arrow or 'between X and Y' pattern"
                    )
                else:
                    reason = "endpoints could not be resolved to countries"
                _record_warning(direction, category, reason)
                continue

            # Safety net: avoid self-loops where src == tgt (no
            # supranationals involved).  Try to swap in receiving
            # partners as targets.
            if (
                src_codes == tgt_codes
                and not tgt_supranationals
            ):
                try:
                    from metacouplingllm.visualization.worldmap import (
                        _extract_all_analysis_countries,
                    )

                    receiving = _extract_all_analysis_countries(
                        parsed,
                    ).get("receiving", set())
                    partners = receiving - src_codes
                    if partners:
                        tgt_codes = partners
                except (ImportError, AttributeError, KeyError):
                    pass  # Viz deps unavailable

            for src_code in sorted(src_codes):
                # Country-to-country arrows (existing behaviour).
                target_codes = set(tgt_codes)
                target_codes.discard(src_code)
                for tgt in sorted(target_codes):
                    key = (category, src_code, tgt)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    src_name = get_country_name(src_code) or src_code
                    tgt_name = get_country_name(tgt) or tgt
                    if is_bidir:
                        dir_str = (
                            f"Bidirectional ({src_name} ↔ {tgt_name})"
                        )
                    else:
                        dir_str = f"{src_name} → {tgt_name}"
                    resolved.append({
                        "category": category,
                        "direction": dir_str,
                        "description": flow.get("description", ""),
                    })

                # Supranational target arrows (one entry per
                # supranational, carrying the member-list field for the
                # renderer to use as a single-region endpoint).
                for sup_name in sorted(tgt_supranationals):
                    members = expand_supranational(sup_name) or []
                    if not members:
                        continue
                    # Skip if src is a member of the supranational
                    # (would render as a self-loop arrow into its own
                    # region).
                    if src_code in members:
                        continue
                    key = (category, src_code, f"__SUP__{sup_name}")
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    src_name = get_country_name(src_code) or src_code
                    if is_bidir:
                        dir_str = (
                            f"Bidirectional ({src_name} ↔ {sup_name})"
                        )
                    else:
                        dir_str = f"{src_name} → {sup_name}"
                    resolved.append({
                        "category": category,
                        "direction": dir_str,
                        "description": flow.get("description", ""),
                        "target_supranational": sup_name,
                        "target_supranational_members": list(members),
                    })

        return resolved, warnings

    @staticmethod
    def _resolve_flows_for_adm1_map(
        parsed: ParsedAnalysis,
        focal_adm1: str,
        focal_country: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Resolve mixed ADM1/country flow arrows for ADM1 maps.

        Returns ``(resolved_flows, parse_warnings)``.  Warnings are
        forwarded unchanged from :py:meth:`_resolve_flows_for_map`.

        International or cross-country flows remain country-level arrows.
        Same-country nearby flows involving adjacent ADM1 neighbors are
        emitted as ADM1-to-ADM1 arrows so they can be visualized locally.
        """
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_country,
            get_adm1_info,
            get_adm1_neighbors,
        )

        resolved, warnings = MetacouplingAssistant._resolve_flows_for_map(
            parsed,
            focal_country,
        )

        focal_info = get_adm1_info(focal_adm1)
        focal_name = focal_info["name"] if focal_info else focal_adm1
        domestic_neighbors = sorted(
            code
            for code in get_adm1_neighbors(focal_adm1)
            if get_adm1_country(code) == focal_country
        )
        neighbor_name_to_code = {
            get_adm1_info(code)["name"].lower(): code
            for code in domestic_neighbors
            if get_adm1_info(code) is not None
        }
        seen_domestic: set[tuple[str, str, str]] = set()

        def _has_local_marker(text: str) -> bool:
            lowered = text.lower()
            return any(
                marker in lowered
                for marker in (
                    "adjacent state",
                    "adjacent states",
                    "neighboring state",
                    "neighboring states",
                    "pericoupled",
                    "pericoupling",
                    "regional flow",
                    "regional flows",
                )
            )

        def _mentioned_neighbor_codes(text: str) -> set[str]:
            lowered = text.lower()
            found: set[str] = set()
            for name, code in neighbor_name_to_code.items():
                if re.search(rf"\b{re.escape(name)}\b", lowered):
                    found.add(code)
            return found

        for flow in parsed.iter_flow_entries():
            direction = flow.get("direction", "")
            description = flow.get("description", "")
            category = flow.get("category", "")
            combined = f"{direction} {description}".strip()
            if not combined:
                continue

            mentioned_neighbors = _mentioned_neighbor_codes(combined)
            if not mentioned_neighbors and _has_local_marker(combined):
                mentioned_neighbors = set(domestic_neighbors)
            if not mentioned_neighbors:
                continue

            lowered_direction = direction.lower()
            is_bidir = (
                "bidirectional" in lowered_direction
                or "\u2194" in direction
                or "<->" in direction
                or "<=>" in direction
            )

            src_is_neighbor = False
            tgt_is_neighbor = False
            parts = _FLOW_ARROW_RE.split(direction)
            if len(parts) >= 2:
                src_text = parts[0].lower()
                tgt_text = parts[1].lower()
                src_is_neighbor = bool(_mentioned_neighbor_codes(src_text)) or (
                    _has_local_marker(src_text) and "michigan" not in src_text
                )
                tgt_is_neighbor = bool(_mentioned_neighbor_codes(tgt_text)) or (
                    _has_local_marker(tgt_text) and "michigan" not in tgt_text
                )

            for neighbor_code in sorted(mentioned_neighbors):
                key = (category, focal_adm1, neighbor_code)
                if key in seen_domestic:
                    continue
                seen_domestic.add(key)

                neighbor_info = get_adm1_info(neighbor_code)
                if neighbor_info is None:
                    continue
                neighbor_name = neighbor_info["name"]

                source_code = focal_adm1
                target_code = neighbor_code
                if src_is_neighbor and not tgt_is_neighbor:
                    source_code = neighbor_code
                    target_code = focal_adm1

                if is_bidir:
                    direction_label = (
                        f"Bidirectional ({focal_name} \u2194 {neighbor_name})"
                    )
                elif source_code == focal_adm1:
                    direction_label = f"{focal_name} \u2192 {neighbor_name}"
                else:
                    direction_label = f"{neighbor_name} \u2192 {focal_name}"

                resolved.append({
                    "category": category,
                    "direction": direction_label,
                    "description": description,
                    "source_adm1": source_code,
                    "target_adm1": target_code,
                    "is_bidirectional": is_bidir,
                })

        return resolved, warnings

    def _build_adm1_reference_for_prompt(
        self,
        parsed: ParsedAnalysis,
        get_adm1_codes_for_country,
        get_adm1_info,
    ) -> str:
        """Build an ADM1 code reference block for the extraction prompt.

        Collects countries mentioned in the analysis, looks up their
        ADM1 codes in the pericoupling database, and formats them as a
        closed-set reference list. This prevents the LLM from
        hallucinating codes like ``BRA014`` (which doesn't exist) when
        the correct code is ``BRA011`` (Mato Grosso).

        Returns an empty string when no mentioned countries have ADM1
        data in the database.
        """
        # Gather country ISO codes mentioned anywhere in the analysis
        mentioned_iso: set[str] = set()
        combined = " ".join(parsed.iter_text_fragments())

        # Resolve country names found in the text
        for word_group in re.findall(
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", combined
        ):
            code = resolve_country_code(word_group)
            if code:
                mentioned_iso.add(code)
        # Pick up explicit 3-letter ISO codes
        for m in re.findall(r"\b([A-Z]{3})\b", combined):
            if get_country_name(m):
                mentioned_iso.add(m)

        if not mentioned_iso:
            return ""

        # Build the reference list — one line per country with all
        # its ADM1 codes. Cap at a reasonable size to keep the prompt
        # manageable.
        lines: list[str] = []
        for iso in sorted(mentioned_iso):
            codes = get_adm1_codes_for_country(iso)
            if not codes:
                continue
            country_name = get_country_name(iso) or iso
            pairs: list[tuple[str, str]] = []
            for code in sorted(codes):
                info = get_adm1_info(code)
                if info:
                    name = info.get("name", "")
                    if name:
                        pairs.append((code, name))
            if not pairs:
                continue
            pair_str = ", ".join(f"{c}={n}" for c, n in pairs)
            lines.append(f"- {country_name} ({iso}): {pair_str}")

        if not lines:
            return ""

        return (
            "VALID ADM1 CODES (closed set — only use codes from this "
            "list, never invent new ones):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    def _extract_map_data_from_analysis(
        self,
        parsed: ParsedAnalysis,
    ) -> dict[str, object] | None:
        """Second LLM call: extract structured map data from the analysis.

        Sends the parsed analysis text to the LLM with a focused JSON
        schema, validates the response, and returns a dict with
        ``focal_country``, ``adm1_region``, ``receiving_countries``,
        ``spillover_countries``, and ``flows``.

        Returns ``None`` if extraction fails.
        """
        import json

        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_codes_for_country,
            get_adm1_info,
        )
        from metacouplingllm.knowledge.countries import (
            expand_supranational,
            get_country_name,
            resolve_country_code,
            supranational_display_name,
        )
        from metacouplingllm.knowledge.websearch import _extract_json_object
        from metacouplingllm.llm.client import Message

        # Build a concise summary of the analysis for the extraction call
        parts: list[str] = []
        if parsed.coupling_classification:
            parts.append(
                "Coupling classification:\n"
                + parsed.coupling_classification[:600]
            )
        for role in ("focal", "adjacent", "sending", "receiving", "spillover"):
            entries = parsed.get_system_entries(role)
            for idx, entry in enumerate(entries[:2], 1):
                # Emit fields in priority order so ``name`` and
                # ``geographic_scope`` are guaranteed to survive truncation
                # at ``text[:800]`` even when subsystem prose is verbose.
                ordered_parts: list[str] = []
                seen_keys: set[str] = set()
                for key in (
                    _HIGH_PRIORITY_SYSTEM_FIELDS
                    + _LOW_PRIORITY_SYSTEM_FIELDS
                ):
                    v = entry.get(key, "")
                    if v:
                        ordered_parts.append(f"{key}: {v}")
                        seen_keys.add(key)
                # Include any unexpected fields after the prioritised
                # ones so we never silently lose data the parser
                # surfaces under a non-standard key.
                for k, v in entry.items():
                    if k == "role" or k in seen_keys or not v:
                        continue
                    ordered_parts.append(f"{k}: {v}")
                text = " | ".join(ordered_parts)
                if text:
                    parts.append(f"{role.title()} system {idx}: {text[:800]}")
        for flow in list(parsed.iter_flow_entries())[:8]:
            cat = flow.get("category", "")
            dirn = flow.get("direction", "")
            # 500-char cap keeps bilateral country names (often listed
            # late in a flow description) from being truncated mid-word.
            desc = flow.get("description", "")[:500]
            parts.append(f"Flow [{cat}] {dirn}: {desc}")

        analysis_summary = "\n\n".join(parts)

        # Include web search results so the LLM can identify specific
        # countries even when the analysis uses generic labels like
        # "foreign importing countries."
        web_results_text = ""
        if self._last_web_results:
            lines: list[str] = []
            for idx, wr in enumerate(
                self._last_web_results[:_MAX_WEB_SNIPPETS_IN_MAP_PROMPT], 1,
            ):
                title = wr.get("title", "").strip()
                # Accept legacy "snippet" key as fallback.
                summary = (
                    wr.get("model_summary")
                    or wr.get("snippet", "")
                ).strip()[:200]
                lines.append(f"[W{idx}] {title}: {summary}")
            web_results_text = (
                "\n\nWeb search results (use these to identify specific "
                "countries when the analysis is vague):\n"
                + "\n".join(lines)
            )

        # Build a reference list of valid ADM1 codes for countries
        # mentioned in the analysis. This prevents the LLM from
        # hallucinating codes like BRA014 (which doesn't exist) when
        # the correct code is BRA011 (Mato Grosso).
        adm1_reference_text = self._build_adm1_reference_for_prompt(
            parsed,
            get_adm1_codes_for_country,
            get_adm1_info,
        )

        system_text = (
            "You extract structured map data from a metacoupling analysis "
            "and optional web search context. "
            "Return ONLY a JSON object, no commentary."
        )
        user_text = (
            "Extract map-ready data from this metacoupling analysis.\n\n"
            "Return a JSON object with these fields:\n"
            "- focal_country: ISO alpha-3 code of the primary sending "
            "country (e.g., \"BRA\", \"USA\")\n"
            "- adm1_region: subnational ADM1 code if the study focuses "
            "on a specific region. MUST be chosen from the reference "
            "list below — NEVER invent codes. Use null if the study is "
            "not focused on a specific subnational region.\n"
            "- mentioned_adm1_regions: list of ADM1 codes (from the "
            "reference list below) for subnational regions EXPLICITLY "
            "discussed in the analysis as sending, receiving, "
            "pericoupled, or spillover systems. Use an empty list [] "
            "if no specific subnational regions are discussed. This "
            "is used to filter the map and must only include regions "
            "substantively discussed, NOT regions that merely appear "
            "in a reference/validation section.\n"
            "- receiving_countries: list of ISO alpha-3 codes for "
            "receiving systems (countries that RECEIVE goods/services "
            "FROM the focal country)\n"
            "- spillover_countries: list of ISO alpha-3 codes for "
            "spillover systems (countries indirectly affected)\n"
            "- flows: list of objects with:\n"
            "  - category: one of matter, capital, information, energy, "
            "people, organisms\n"
            "  - source: ISO alpha-3 code of the exporter/sender\n"
            "  - target: ISO alpha-3 code of the importer/receiver\n"
            "  - bidirectional: true or false\n\n"
            "IMPORTANT RULES:\n"
            "1. Use ISO alpha-3 codes (USA, BRA, CHN, MEX, JPN, etc)\n"
            "2. receiving_countries = trade partners that BUY from or "
            "RECEIVE from the focal country. They are NOT competitors.\n"
            "3. spillover_countries = competitors, indirectly affected "
            "countries, environmental spillover. They are NOT trade "
            "partners.\n"
            "4. NEVER put competitor/competing exporter countries in "
            "receiving_countries. Example: if USA exports corn, Brazil "
            "and Argentina are COMPETITORS (spillover), not receivers.\n"
            "5. Most flows go FROM focal_country TO receiving_countries "
            "(matter, information, energy, people, organisms).\n"
            "6. Capital/payment flows go in REVERSE: FROM "
            "receiving_countries TO focal_country (e.g., "
            "'Japan \u2192 Australia' for beef payments). Create one "
            "capital flow per receiving country.\n"
            "7. No flows should involve spillover countries as source "
            "or target.\n"
            "8. When the analysis says 'importing countries' without "
            "naming them, use the web results to identify the most "
            "likely specific countries\n"
            "9. Do not invent countries with no supporting evidence\n"
            "10. For adm1_region: ONLY use codes from the reference "
            "list below. If the study's subnational focus does not "
            "appear in the list, return null for adm1_region.\n"
            "11. For mentioned_adm1_regions: include codes ONLY for "
            "regions where the analysis provides SUBSTANTIVE EVIDENCE "
            "of actual interaction with the focal region — specific "
            "flows, impacts, shared infrastructure, commodity "
            "transport, labor movement, land-use displacement, or "
            "other concrete linkages. Do NOT include a region just "
            "because it is named in a reference list, coupling "
            "classification, or pericoupled-neighbors enumeration. If "
            "a region is only mentioned as \"adjacent\" without any "
            "specific interaction described, do NOT include it. The "
            "focal_adm1 region should NOT be listed here — it is "
            "tracked separately via adm1_region.\n"
            "12. Supranational unions as flow targets: when the "
            "analysis or web context describes flows to a union as a "
            "whole (e.g., 'avocado exports to the European Union', "
            "'virtual water flows to the EU') without naming "
            "individual member states for that flow, you MAY use the "
            "union name as the 'target' instead of an ISO-3 code.  "
            "Accepted values: 'European Union' (or 'EU'), 'ASEAN', "
            "'USMCA', 'NAFTA'.  The renderer treats these as single "
            "regions at the union centroid.  If specific member "
            "states ARE named in the same flow context, prefer those "
            "ISO codes and omit the umbrella.\n\n"
            f"{adm1_reference_text}"
            f"Analysis:\n{analysis_summary}"
            f"{web_results_text}"
        )

        try:
            response = self._client.chat(
                messages=[
                    Message(role="system", content=system_text),
                    Message(role="user", content=user_text),
                ],
                temperature=0.0,
                max_tokens=8192,
            )
        except Exception as exc:
            print(
                "[MetacouplingAssistant] Map data extraction LLM call "
                f"failed: {exc}"
            )
            return None

        try:
            raw_obj = _extract_json_object(response.content)
        except Exception as exc:
            print(
                f"[MetacouplingAssistant] JSON parsing error: {exc}. "
                f"Raw (first 300 chars): {response.content[:300]}"
            )
            return None

        if raw_obj is None:
            print(
                "[MetacouplingAssistant] Map data extraction returned "
                "non-JSON response. Raw response (first 300 chars): "
                f"{response.content[:300]}"
            )
            return None

        if not isinstance(raw_obj, dict):
            print(
                "[MetacouplingAssistant] Map data extraction returned "
                f"non-dict JSON: {type(raw_obj).__name__}"
            )
            return None

        # --- Validate and normalise ---
        focal = resolve_country_code(str(raw_obj.get("focal_country", "")))
        if not focal:
            print(
                "[MetacouplingAssistant] Map data extraction: could not "
                "resolve focal country."
            )
            return None

        adm1_region = raw_obj.get("adm1_region")
        if isinstance(adm1_region, str) and adm1_region.strip():
            adm1_region = adm1_region.strip()
            # Validate against the pericoupling database. The LLM
            # sometimes hallucinates codes (e.g., "BRA014" instead of
            # the correct "BRA011" for Mato Grosso).
            if get_adm1_info(adm1_region) is None:
                print(
                    f"[MetacouplingAssistant] LLM returned invalid ADM1 "
                    f"code '{adm1_region}' — falling back to regex "
                    f"resolver."
                )
                adm1_region = None
        else:
            adm1_region = None

        # Fallback: if the LLM didn't return a valid ADM1 code, try the
        # regex-based resolver which reads the analysis text directly
        # and resolves region names against the database. This is the
        # same resolver used by the pericoupling database validation.
        if adm1_region is None:
            fallback = self._resolve_adm1_from_analysis(parsed)
            if fallback:
                print(
                    f"[MetacouplingAssistant] ADM1 code resolved from "
                    f"analysis text: {fallback}."
                )
                adm1_region = fallback

        def _resolve_code_list(items: object) -> list[str]:
            if not isinstance(items, list):
                return []
            codes: list[str] = []
            for item in items:
                code = resolve_country_code(str(item).strip())
                if code:
                    codes.append(code)
            return codes

        def _resolve_adm1_list(items: object) -> list[str]:
            """Validate a list of ADM1 codes against the database."""
            if not isinstance(items, list):
                return []
            codes: list[str] = []
            seen: set[str] = set()
            for item in items:
                code = str(item).strip()
                if not code or code in seen:
                    continue
                if get_adm1_info(code) is not None:
                    codes.append(code)
                    seen.add(code)
            return codes

        receiving = _resolve_code_list(raw_obj.get("receiving_countries"))
        spillover = _resolve_code_list(raw_obj.get("spillover_countries"))
        mentioned_adm1 = _resolve_adm1_list(
            raw_obj.get("mentioned_adm1_regions"),
        )

        # Validate flows.  Category labels are normalised through the
        # shared ``_FLOW_CATEGORY_ALIASES`` table so common slips like
        # "goods", "money", "electricity", "tourism", "livestock" map
        # to the right Liu 2017 canonical category instead of being
        # silently dropped.
        flows: list[dict[str, object]] = []
        raw_flows = raw_obj.get("flows", [])
        # Pre-compute the set of ISO codes the LLM already named
        # explicitly, used by the conditional supranational rule below
        # (only stamp single-region treatment when no member is already
        # named \u2014 otherwise the umbrella is redundant).
        already_named: set[str] = set(receiving) | set(spillover)
        if isinstance(raw_flows, list):
            for item in raw_flows:
                if not isinstance(item, dict):
                    continue
                cat = _normalize_flow_category(
                    str(item.get("category", "")),
                )
                if cat is None:
                    continue
                raw_src = str(item.get("source", ""))
                raw_tgt = str(item.get("target", ""))
                src = resolve_country_code(raw_src)
                tgt = resolve_country_code(raw_tgt)
                bidir = bool(item.get("bidirectional", False))

                # Country-pair flow (the common case).
                if src and tgt and src != tgt:
                    src_name = get_country_name(src) or src
                    tgt_name = get_country_name(tgt) or tgt
                    if bidir:
                        direction = (
                            f"Bidirectional ({src_name} \u2194 {tgt_name})"
                        )
                    else:
                        direction = f"{src_name} \u2192 {tgt_name}"
                    flows.append({
                        "category": cat,
                        "source": src,
                        "target": tgt,
                        "direction": direction,
                        "bidirectional": bidir,
                    })
                    continue

                # Supranational target fallback: the LLM may slip past
                # rule #1 ("use ISO codes") and emit "EU" / "European
                # Union" / "ASEAN" / "NAFTA" / "USMCA" as the target.
                # Recognise the umbrella and stamp the supranational
                # fields so the renderer treats it as a single region.
                if src and not tgt:
                    members = expand_supranational(raw_tgt)
                    if not members:
                        continue
                    # Conditional rule (mirrors the resolver-path
                    # behaviour): if any member is already named in
                    # receiving_countries / spillover_countries, the
                    # umbrella is redundant \u2014 skip this flow.
                    if already_named & set(members):
                        continue
                    # Avoid self-loops where src is a member.
                    if src in members:
                        continue
                    display = supranational_display_name(members)
                    if not display:
                        continue
                    src_name = get_country_name(src) or src
                    if bidir:
                        direction = (
                            f"Bidirectional ({src_name} \u2194 {display})"
                        )
                    else:
                        direction = f"{src_name} \u2192 {display}"
                    flows.append({
                        "category": cat,
                        "source": src,
                        "target": None,
                        "direction": direction,
                        "bidirectional": bidir,
                        "target_supranational": display,
                        "target_supranational_members": list(members),
                    })

        result = {
            "focal_country": focal,
            "adm1_region": adm1_region,
            "mentioned_adm1_regions": mentioned_adm1,
            "receiving_countries": receiving,
            "spillover_countries": spillover,
            "flows": flows,
        }

        if self._verbose:
            print(
                f"[MetacouplingAssistant] Map extraction: focal={focal}, "
                f"receiving={receiving}, spillover={spillover}, "
                f"mentioned_adm1={mentioned_adm1}, "
                f"flows={len(flows)}."
            )

        return result

    # ------------------------------------------------------------------
    # Structured extraction supplement (rag_structured_extraction=True)
    # ------------------------------------------------------------------

    def _structured_extract_supplement(
        self,
        parsed: ParsedAnalysis,
    ) -> dict[str, object] | None:
        """Second LLM call: extract systems / flows the draft may have missed.

        Runs over ``self._last_rag_hits`` (the already-retrieved RAG
        passages) plus a concise summary of the draft analysis, and
        returns a dict with additional system mentions, empty-subfield
        fills, and supplementary flows — each carrying the passage IDs
        that support it.

        The result is attached to :class:`AnalysisResult` as
        ``structured_supplement`` and rendered as a visible block in
        ``formatted``. The caller (``_build_result``) does NOT merge
        the supplement into ``parsed``; the main analysis body stays
        100% LLM-authored so readers can tell the two apart.

        Returns ``None`` when there are no retrieval hits to ground
        the extraction in, or when the LLM call / JSON parse fails.
        """
        import json

        from metacouplingllm.knowledge.websearch import _extract_json_object
        from metacouplingllm.llm.client import Message

        if not self._last_rag_hits:
            return None

        # --- Build the passages block -----------------------------------
        passage_lines: list[str] = []
        for idx, r in enumerate(self._last_rag_hits, 1):
            chunk = r.chunk
            authors = chunk.authors or ""
            if len(authors) > 60:
                authors = authors.split(" and ")[0] + " et al."
            # Pass the FULL chunk text — the chunker already caps
            # chunks at ~250 words (~1600 chars median, <2000 chars at
            # p99, <5000 chars at max). The prior 600-char truncation
            # here was cutting off bilateral country data that lives
            # past the section-opening summary (e.g., Korea/Japan/
            # Russia MtCO2 values at position ~689 in long Results
            # chunks). Full passages let the extraction LLM find
            # specific partner countries with quantitative evidence.
            text = (chunk.text or "").strip().replace("\n", " ")
            passage_lines.append(
                f"[{idx}] {authors} ({chunk.year}) — "
                f"{chunk.paper_title[:80]}:\n    {text}"
            )
        passages_block = "\n\n".join(passage_lines)

        # --- Build a concise summary of the current draft ---------------
        draft_lines: list[str] = []
        if parsed.coupling_classification:
            draft_lines.append(
                "Coupling classification (truncated):\n"
                + parsed.coupling_classification[:500]
            )
        role_aliases = {
            "sending": ("sending", "focal"),
            "receiving": ("receiving",),
            "spillover": ("spillover", "adjacent"),
        }
        for label, candidate_roles in role_aliases.items():
            entry = None
            for role in candidate_roles:
                entry = parsed.get_first_system_entry(role)
                if entry:
                    break
            if entry is not None:
                # 400-char per-subfield cap mirrors the loosening done
                # in `_extract_map_data_from_analysis` (PR #8).  The
                # draft summary tells the supplement LLM what the main
                # analysis already covered, so undersized caps here
                # produced duplicate "additional mention" entries for
                # content that was actually present in the draft.
                bits = [
                    f"name={entry.get('name', '')!r}",
                    f"human={entry.get('human_subsystem', '')[:400]!r}",
                    f"natural={entry.get('natural_subsystem', '')[:400]!r}",
                    f"geographic_scope={entry.get('geographic_scope', '')[:400]!r}",
                ]
                draft_lines.append(f"{label.title()} system: " + ", ".join(bits))
            else:
                draft_lines.append(f"{label.title()} system: (empty)")
        parsed_flows = list(parsed.iter_flow_entries())
        if parsed_flows:
            draft_lines.append("Existing flows:")
            for flow in parsed_flows[:12]:
                cat = flow.get("category", "?")
                direction = flow.get("direction", "?")
                # 500-char cap (matches the map-extraction path) keeps
                # bilateral country names that often sit late in a flow
                # description from being truncated mid-word.
                desc = flow.get("description", "")[:500]
                draft_lines.append(f"  - [{cat}] {direction}: {desc}")
        else:
            draft_lines.append("Existing flows: (none)")
        draft_summary = "\n".join(draft_lines)

        # --- Schema shown to the LLM ------------------------------------
        schema = {
            "additional_sending_mentions": [
                {
                    "name": "short descriptive name",
                    "evidence_passage_ids": [1],
                }
            ],
            "additional_receiving_mentions": [
                {
                    "name": "short descriptive name",
                    "evidence_passage_ids": [1],
                }
            ],
            "additional_spillover_mentions": [
                {
                    "name": "short descriptive name",
                    "evidence_passage_ids": [1],
                }
            ],
            "sending_subsystem_fills": {
                "human_subsystem": "fill if draft field is empty, else null",
                "natural_subsystem": "fill if draft field is empty, else null",
                "geographic_scope": "fill if draft field is empty, else null",
                "evidence_passage_ids": [1],
            },
            "receiving_subsystem_fills": {
                "human_subsystem": None,
                "natural_subsystem": None,
                "geographic_scope": None,
                "evidence_passage_ids": [1],
            },
            "spillover_subsystem_fills": {
                "human_subsystem": None,
                "natural_subsystem": None,
                "geographic_scope": None,
                "evidence_passage_ids": [1],
            },
            "supplementary_flows": [
                {
                    "category": "matter|capital|information|energy|people|organisms",
                    "direction": "Source → Target",
                    "description": "short flow description",
                    "evidence_passage_ids": [1, 2],
                }
            ],
        }

        system_text = (
            "You review a draft metacoupling analysis against its "
            "source research passages and identify additional systems "
            "or flows the draft may have missed, and empty subfields "
            "that the passages can fill. You NEVER invent information "
            "that is not supported by the passages, and you NEVER "
            "repeat items already in the draft. Return JSON only."
        )

        query = (self._original_query or "").strip()
        user_text = (
            (f"## Research query\n{query}\n\n" if query else "")
            + "## Retrieved passages\n"
            + passages_block
            + "\n\n## Draft analysis (extracted fields)\n"
            + draft_summary
            + "\n\n## Task\n"
            + "Look at the retrieved passages and identify content that "
            "the draft analysis missed or under-specified. Do NOT repeat "
            "items already present in the draft.\n\n"
            + "1. For each role (sending, receiving, spillover), list any "
            "ADDITIONAL system mentions that appear in the passages but "
            "are NOT in the draft's system identification. Use short "
            "descriptive names (1-8 words) and cite the passage IDs that "
            "support each mention.\n\n"
            + "2. For each role, look at the draft's subsystem fields "
            "(human_subsystem, natural_subsystem, geographic_scope). If "
            "the draft field is EMPTY, fill it using content from the "
            "passages (1-2 short sentences). Set the value to null if "
            "the draft already has content or the passages do not "
            "address that subsystem. Include the evidence passage IDs.\n\n"
            + "3. List SUPPLEMENTARY flows — flows mentioned in the "
            "passages that are NOT already in the draft's existing "
            "flows. Each flow must have a category from "
            "(matter, capital, information, energy, people, organisms), "
            "a direction in 'Source → Target' form, a short "
            "description, and at least one evidence passage ID.\n\n"
            + "Rules:\n"
            + "- Use ONLY information supported by the retrieved "
            "passages (not your training knowledge).\n"
            + "- Every item MUST include evidence_passage_ids pointing "
            f"at passage numbers 1..{len(self._last_rag_hits)}.\n"
            + "- NEVER repeat items already in the draft.\n"
            + "- Use null or [] when there is nothing to add for a "
            "field.\n"
            + "- When a passage lists SPECIFIC COUNTRIES with numeric "
            "values (e.g., 'Korea (2.65 MtCO2), Japan (1.92 MtCO2), "
            "Russian Federation (1.46 MtCO2)', or 'Thailand (2.10 Mt), "
            "Japan (0.80 Mt) and United States (0.66 MtCO2) are the "
            "three largest contributors'), emit EACH named country as "
            "its own separate entry in the appropriate "
            "additional_{sending,receiving,spillover}_mentions list. "
            "Do NOT collapse them into a grouped abstraction like "
            "'Pacific Rim countries', 'major origin countries', or "
            "'top three countries' — that loses the country-level "
            "detail the map needs. When a numeric value is attached, "
            "include it parenthetically in the name field, e.g.:\n"
            "    {\"name\": \"Korea (2.65 MtCO2 inbound tourism)\",\n"
            "     \"evidence_passage_ids\": [6]}\n"
            "    {\"name\": \"Japan (1.92 MtCO2 inbound tourism)\",\n"
            "     \"evidence_passage_ids\": [6]}\n"
            "  This rule applies ONLY to country / region names "
            "(sovereign states, territories, ADM1 regions). For "
            "non-geographic groupings (e.g., 'airlines', "
            "'hospitality sector') the grouped form is fine.\n"
            + "- Keep at most 12 items per additional_*_mentions list "
            "(raised from 6 so per-country entries have room) and "
            "at most 12 supplementary_flows.\n\n"
            + f"Return JSON matching this schema exactly:\n"
            + json.dumps(schema, ensure_ascii=True, indent=2)
        )

        try:
            response = self._client.chat(
                messages=[
                    Message(role="system", content=system_text),
                    Message(role="user", content=user_text),
                ],
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            logger.warning(
                "Structured extraction LLM call failed: %s. "
                "Skipping supplement.", exc,
            )
            return None

        raw_obj = _extract_json_object(response.content)
        if not isinstance(raw_obj, dict):
            logger.warning(
                "Structured extraction produced non-JSON response; "
                "skipping supplement."
            )
            return None

        # --- Validate + normalise ---------------------------------------
        n_valid_ids = len(self._last_rag_hits)

        def _clean_ids(obj: object) -> list[int]:
            if not isinstance(obj, list):
                return []
            out: list[int] = []
            for item in obj:
                try:
                    n = int(item)
                except (TypeError, ValueError):
                    continue
                if 1 <= n <= n_valid_ids:
                    out.append(n)
            # dedupe, preserve order
            seen: set[int] = set()
            dedup: list[int] = []
            for n in out:
                if n not in seen:
                    seen.add(n)
                    dedup.append(n)
            return dedup

        def _clean_mentions(obj: object, cap: int) -> list[dict[str, object]]:
            if not isinstance(obj, list):
                return []
            out: list[dict[str, object]] = []
            for item in obj:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name or len(name) > 200:
                    continue
                ids = _clean_ids(item.get("evidence_passage_ids", []))
                if not ids:
                    continue
                out.append({"name": name, "evidence_passage_ids": ids})
                if len(out) >= cap:
                    break
            return out

        def _clean_fills(obj: object) -> dict[str, object]:
            if not isinstance(obj, dict):
                return {
                    "human_subsystem": None,
                    "natural_subsystem": None,
                    "geographic_scope": None,
                    "evidence_passage_ids": [],
                }
            cleaned: dict[str, object] = {}
            for field in (
                "human_subsystem", "natural_subsystem", "geographic_scope",
            ):
                v = obj.get(field)
                if isinstance(v, str):
                    v = v.strip()
                    cleaned[field] = v if v else None
                else:
                    cleaned[field] = None
            cleaned["evidence_passage_ids"] = _clean_ids(
                obj.get("evidence_passage_ids", []),
            )
            return cleaned

        def _clean_flows(obj: object, cap: int) -> list[dict[str, object]]:
            if not isinstance(obj, list):
                return []
            out: list[dict[str, object]] = []
            for item in obj:
                if not isinstance(item, dict):
                    continue
                cat = _normalize_flow_category(
                    str(item.get("category", "")),
                )
                if cat is None:
                    continue
                direction = str(item.get("direction", "")).strip()
                if not direction:
                    continue
                description = str(item.get("description", "")).strip()
                if not description:
                    continue
                ids = _clean_ids(item.get("evidence_passage_ids", []))
                if not ids:
                    continue
                out.append({
                    "category": cat,
                    "direction": direction[:200],
                    "description": description[:400],
                    "evidence_passage_ids": ids,
                })
                if len(out) >= cap:
                    break
            return out

        # Caps match the prompt's announced limits (12 per list) so
        # per-country entries from long bilateral breakdowns aren't
        # silently trimmed.
        supplement: dict[str, object] = {
            "additional_sending_mentions": _clean_mentions(
                raw_obj.get("additional_sending_mentions", []), cap=12,
            ),
            "additional_receiving_mentions": _clean_mentions(
                raw_obj.get("additional_receiving_mentions", []), cap=12,
            ),
            "additional_spillover_mentions": _clean_mentions(
                raw_obj.get("additional_spillover_mentions", []), cap=12,
            ),
            "sending_subsystem_fills": _clean_fills(
                raw_obj.get("sending_subsystem_fills", {}),
            ),
            "receiving_subsystem_fills": _clean_fills(
                raw_obj.get("receiving_subsystem_fills", {}),
            ),
            "spillover_subsystem_fills": _clean_fills(
                raw_obj.get("spillover_subsystem_fills", {}),
            ),
            "supplementary_flows": _clean_flows(
                raw_obj.get("supplementary_flows", []), cap=12,
            ),
        }

        if self._verbose:
            n_flows = len(supplement["supplementary_flows"])  # type: ignore[arg-type]
            n_mentions = sum(
                len(supplement[f"additional_{role}_mentions"])  # type: ignore[arg-type]
                for role in ("sending", "receiving", "spillover")
            )
            print(
                f"[MetacouplingAssistant] Structured supplement: "
                f"{n_flows} supplementary flows, "
                f"{n_mentions} additional mentions."
            )

        return supplement

    @staticmethod
    def _format_structured_supplement(
        supplement: dict[str, object],
    ) -> str:
        """Render a structured-supplement dict as a visible text block.

        The block is appended to the formatted analysis output between
        the main body and the ``SUPPORTING EVIDENCE FROM LITERATURE``
        section, so the reader can tell at a glance which content came
        from the primary LLM pass and which came from the second
        structured-extraction pass.
        """
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("  SUPPLEMENTARY STRUCTURED EXTRACTION")
        lines.append("  (Additional systems / flows identified from retrieved literature)")
        lines.append("=" * 72)
        lines.append("")

        def _cite(ids: list[int]) -> str:
            if not ids:
                return ""
            return " " + "".join(f"[{n}]" for n in ids)

        wrote_anything = False

        # --- Per-role additional mentions -------------------------------
        for role, heading in (
            ("sending", "Additional Sending System Mentions"),
            ("receiving", "Additional Receiving System Mentions"),
            ("spillover", "Additional Spillover System Mentions"),
        ):
            mentions = supplement.get(f"additional_{role}_mentions", [])
            if not isinstance(mentions, list) or not mentions:
                continue
            lines.append(f"{heading}:")
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                name = str(m.get("name", "")).strip()
                ids = m.get("evidence_passage_ids", []) or []
                if not isinstance(ids, list):
                    ids = []
                if name:
                    lines.append(f"  - {name}{_cite(ids)}")
                    wrote_anything = True
            lines.append("")

        # --- Per-role subsystem fills -----------------------------------
        for role, heading in (
            ("sending", "Sending"),
            ("receiving", "Receiving"),
            ("spillover", "Spillover"),
        ):
            fills = supplement.get(f"{role}_subsystem_fills", {})
            if not isinstance(fills, dict):
                continue
            non_null = {
                k: v for k, v in fills.items()
                if k in ("human_subsystem", "natural_subsystem", "geographic_scope")
                and isinstance(v, str) and v
            }
            if not non_null:
                continue
            ids = fills.get("evidence_passage_ids", []) or []
            if not isinstance(ids, list):
                ids = []
            lines.append(f"{heading} System — Subsystem Fills (from literature):")
            for field, value in non_null.items():
                pretty = field.replace("_", " ").capitalize()
                lines.append(f"  - {pretty}: {value}{_cite(ids)}")
                wrote_anything = True
            lines.append("")

        # --- Supplementary flows ----------------------------------------
        supp_flows = supplement.get("supplementary_flows", [])
        if isinstance(supp_flows, list) and supp_flows:
            lines.append("Supplementary Flows:")
            for f in supp_flows:
                if not isinstance(f, dict):
                    continue
                cat = str(f.get("category", "?"))
                direction = str(f.get("direction", ""))
                desc = str(f.get("description", ""))
                ids = f.get("evidence_passage_ids", []) or []
                if not isinstance(ids, list):
                    ids = []
                lines.append(f"  - [{cat}] {direction}: {desc}{_cite(ids)}")
                wrote_anything = True
            lines.append("")

        if not wrote_anything:
            lines.append("(No additional systems or flows identified "
                         "from the retrieved literature.)")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Structured supplement → map_data enrichment
    # ------------------------------------------------------------------

    def _merge_supplement_into_map_data(
        self,
        parsed: ParsedAnalysis,
        supplement: dict[str, object],
    ) -> None:
        """Enrich ``parsed.map_data`` with countries/flows from the
        structured-extraction supplement.

        ``_extract_map_data_from_analysis`` reads the finished analysis
        text, which the main LLM often writes with generic wording
        ("foreign destination countries"). The structured supplement,
        by contrast, reads the retrieved passages directly with a
        schema that requires per-country entries — so it can surface
        specific bilateral partners (Korea, Japan, Russia with MtCO2
        numbers) that the analysis abstracted away. This helper
        reconciles the two: the primary extraction stays authoritative,
        and the supplement fills in the gaps.

        The merge is purely additive. We never DROP countries or flows
        that the primary extraction produced. We only ADD:

        - Receiving / spillover countries the primary path missed, IF
          the supplement mention resolves to a valid ISO alpha-3 code
          (via ``resolve_country_code``).
        - Flows from supplementary_flows, parsed into source/target/
          category, IF both endpoints resolve and the pair is not
          already represented.

        Never raises — all resolution failures are logged at DEBUG and
        that specific entry is skipped.
        """
        map_data = parsed.map_data
        if not isinstance(map_data, dict):
            return
        if not isinstance(supplement, dict):
            return

        focal_code = map_data.get("focal_country")
        # Existing sets so we don't duplicate
        existing_recv: set[str] = set()
        for c in map_data.get("receiving_countries", []) or []:
            if isinstance(c, str):
                existing_recv.add(c.upper())
        existing_spill: set[str] = set()
        for c in map_data.get("spillover_countries", []) or []:
            if isinstance(c, str):
                existing_spill.add(c.upper())

        # --- Receiving mentions ----------------------------------------
        for mention in supplement.get("additional_receiving_mentions", []) or []:
            if not isinstance(mention, dict):
                continue
            name = str(mention.get("name", "")).strip()
            if not name:
                continue
            code = resolve_country_code(name)
            if not code:
                logger.debug(
                    "Supplement receiving mention %r did not resolve "
                    "to an ISO code; skipping.", name,
                )
                continue
            if code == focal_code:
                continue  # already the focal; not a separate receiving
            if code in existing_spill:
                continue  # already classified as spillover; don't double-tag
            if code in existing_recv:
                continue
            map_data.setdefault("receiving_countries", []).append(code)
            existing_recv.add(code)

        # --- Spillover mentions ----------------------------------------
        for mention in supplement.get("additional_spillover_mentions", []) or []:
            if not isinstance(mention, dict):
                continue
            name = str(mention.get("name", "")).strip()
            if not name:
                continue
            code = resolve_country_code(name)
            if not code:
                logger.debug(
                    "Supplement spillover mention %r did not resolve "
                    "to an ISO code; skipping.", name,
                )
                continue
            if code == focal_code:
                continue
            if code in existing_recv:
                continue
            if code in existing_spill:
                continue
            map_data.setdefault("spillover_countries", []).append(code)
            existing_spill.add(code)

        # --- Sending mentions ------------------------------------------
        # For bidirectional phenomena (e.g., inbound tourism), a
        # "sending" country at the paper level IS effectively a partner
        # on the map. We surface them as receiving countries on the
        # map so they render as trade partners rather than being
        # dropped. The focal stays the focal.
        for mention in supplement.get("additional_sending_mentions", []) or []:
            if not isinstance(mention, dict):
                continue
            name = str(mention.get("name", "")).strip()
            if not name:
                continue
            code = resolve_country_code(name)
            if not code:
                logger.debug(
                    "Supplement sending mention %r did not resolve "
                    "to an ISO code; skipping.", name,
                )
                continue
            if code == focal_code:
                continue
            if code in existing_spill or code in existing_recv:
                continue
            map_data.setdefault("receiving_countries", []).append(code)
            existing_recv.add(code)

        # --- Supplementary flows ---------------------------------------
        # Parse each flow's "Source → Target" direction into a pair of
        # ISO codes and append to map_data['flows']. Skip flows where
        # either endpoint can't be resolved, or where the same pair is
        # already present. We match the shape used by
        # ``_extract_map_data_from_analysis`` so the downstream
        # ``_resolve_flows_for_map`` pipeline can consume both
        # transparently.
        existing_flow_keys: set[tuple[str, str, str]] = set()
        for f in map_data.get("flows", []) or []:
            if not isinstance(f, dict):
                continue
            key = (
                str(f.get("source_country", "")).upper(),
                str(f.get("target_country", "")).upper(),
                str(f.get("category", "")).lower(),
            )
            existing_flow_keys.add(key)

        for sup_flow in supplement.get("supplementary_flows", []) or []:
            if not isinstance(sup_flow, dict):
                continue
            direction = str(sup_flow.get("direction", "")).strip()
            if not direction:
                continue
            # Split on the canonical arrow regex used elsewhere
            parts = _FLOW_ARROW_RE.split(direction)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) < 2:
                logger.debug(
                    "Supplementary flow direction %r did not contain a "
                    "resolvable arrow; skipping.", direction,
                )
                continue
            src_name, tgt_name = parts[0], parts[1]
            src_code = resolve_country_code(src_name)
            tgt_code = resolve_country_code(tgt_name)
            if not src_code or not tgt_code:
                logger.debug(
                    "Supplementary flow %r → %r: endpoint(s) did not "
                    "resolve (src=%r, tgt=%r); skipping.",
                    src_name, tgt_name, src_code, tgt_code,
                )
                continue
            category = str(sup_flow.get("category", "")).strip().lower() or "matter"
            key = (src_code.upper(), tgt_code.upper(), category.lower())
            if key in existing_flow_keys:
                continue
            existing_flow_keys.add(key)
            map_data.setdefault("flows", []).append({
                "category": category,
                "source_country": src_code,
                "target_country": tgt_code,
                "direction": f"{src_code} \u2192 {tgt_code}",
                "description": str(
                    sup_flow.get("description", "")
                )[:300],
                # Same confidence shape the primary extractor uses
                "kind": "proxy",
                "confidence": 0.6,
                "evidence": [],
            })
            # If the target isn't already a receiving country on the
            # map, add it — flows without visible endpoints produce
            # arrows pointing at grey countries, which is confusing.
            if (
                tgt_code != focal_code
                and tgt_code not in existing_spill
                and tgt_code not in existing_recv
            ):
                map_data.setdefault("receiving_countries", []).append(
                    tgt_code,
                )
                existing_recv.add(tgt_code)

    def _generate_map(self, parsed: ParsedAnalysis) -> Figure | None:
        """Generate the appropriate map based on analysis level.

        **Primary path**: uses ``parsed.map_data`` (from a second LLM
        call) which contains validated ISO codes and flow endpoints.

        **Fallback path**: regex extraction from analysis text (legacy).

        Returns ``None`` if nothing can be resolved or deps are missing.
        """
        self._last_map_notice = None
        self._last_flow_parse_warnings = []
        self._last_map_type = None
        try:
            if self._has_unsupported_automap_scope(parsed):
                self._last_map_notice = self._format_map_unavailable_notice(
                    "unsupported_local_geography"
                )
                if self._verbose:
                    print(
                        "[MetacouplingAssistant] Map generation skipped: "
                        "the focal geography is below country/ADM1 scale."
                    )
                return None

            # ==============================================================
            # PRIMARY PATH: structured map data from second LLM call
            # ==============================================================
            md = parsed.map_data
            if md and isinstance(md, dict) and md.get("focal_country"):
                focal_code = md["focal_country"]
                adm1_region = md.get("adm1_region")
                # Respect the user's framing: when the original query
                # named no ADM1 region, prefer country-level rendering
                # even if the LLM stamped an ADM1 focal in the
                # extraction.  Mexican avocado production IS centred
                # on Michoacan, but a user who asked about "Mexican
                # avocado trade" expects a world map of trade flows,
                # not a Michoacan zoom.  When they explicitly name a
                # state ("avocado production in Michoacan, Mexico"),
                # the helper returns True and the LLM's ADM1 choice
                # is preserved.
                #
                # Guard on ``self._original_query`` being non-empty:
                # when ``_generate_map`` is called directly without an
                # ``analyze()`` call (e.g. unit tests, programmatic
                # callers), we have no user framing to consult and
                # must preserve the LLM's ADM1 choice as-is.
                if (
                    adm1_region
                    and self._original_query
                    and not self._user_query_mentions_adm1()
                ):
                    if self._verbose:
                        print(
                            f"[MetacouplingAssistant] Structured "
                            f"extraction returned adm1={adm1_region}, "
                            f"but the user's query did not name an ADM1 "
                            f"region; rendering country-level map instead."
                        )
                    adm1_region = None
                receiving = md.get("receiving_countries", [])
                # Spillover countries are retained in map_data for
                # downstream use but intentionally NOT used here —
                # they render as grey (NA) to avoid conflating them
                # with actual trade partners.
                raw_flows = md.get("flows", [])

                # ADM1 regions explicitly mentioned in the analysis.
                # The LLM provides this list (see extraction prompt);
                # when missing or empty, fall back to a regex scan.
                mentioned_adm1_list = md.get("mentioned_adm1_regions", [])
                if not mentioned_adm1_list:
                    mentioned_adm1_list = sorted(
                        self._extract_mentioned_adm1_from_text(parsed)
                    )
                mentioned_adm1_set: set[str] = set(mentioned_adm1_list)
                # The focal ADM1 is handled separately by the classifier
                # (it gets "intracoupling"); drop it from the set to
                # keep the log output clean and the set minimal.
                if adm1_region:
                    mentioned_adm1_set.discard(adm1_region)

                # Build flow dicts for the renderer.  Propagate the
                # optional supranational marker fields when present so
                # the renderer can do single-region rendering for
                # primary-path flows that target an umbrella entity
                # (EU / ASEAN / NAFTA / USMCA).
                map_flows: list[dict[str, str]] = []
                for f in raw_flows:
                    if isinstance(f, dict) and f.get("direction"):
                        entry: dict[str, object] = {
                            "category": str(f.get("category", "")),
                            "direction": str(f["direction"]),
                            "description": str(f.get("description", "")),
                        }
                        sup_name = f.get("target_supranational")
                        sup_members = f.get("target_supranational_members")
                        if sup_name and sup_members:
                            entry["target_supranational"] = sup_name
                            entry["target_supranational_members"] = list(
                                sup_members
                            )
                        map_flows.append(entry)

                # Merge with web-structured flows if available
                map_flows = self._merge_map_flows(
                    map_flows,
                    self._structured_web_flow_dicts(),
                )

                # Collect countries that should be highlighted as
                # telecoupling on the map. Only focal + receiving
                # belong here. Spillover countries (competitors,
                # indirectly affected) render as grey (NA) to avoid
                # visually conflating them with actual trade partners.
                mentioned = (
                    {focal_code}
                    | set(receiving)
                    | self._structured_web_receiving_codes()
                )

                # Drop flows whose endpoints don't belong on the map.
                # 1. Flow TARGETS must be in mentioned countries.
                # 2. Flow SOURCES must be the focal country or a
                #    receiving country — never a spillover country.
                #    Spillover countries are competitors/indirectly
                #    affected; arrows FROM them (e.g. Brazil → Mexico)
                #    don't belong on a map about the focal country's
                #    trade.
                if map_flows:
                    from metacouplingllm.knowledge.countries import (
                        resolve_country_code as _rcc,
                    )

                    # Valid flow sources: focal + receiving (not spillover)
                    _valid_sources = {focal_code} | set(receiving)

                    def _flow_endpoints_valid(f: dict) -> bool:
                        d = f.get("direction", "")
                        parts = _FLOW_ARROW_RE.split(d)
                        if len(parts) < 2:
                            return True  # can't validate → keep

                        # Validate target
                        tgt_name = parts[-1].strip().rstrip(")")
                        tgt_code = _rcc(tgt_name)
                        if tgt_code and tgt_code not in mentioned:
                            return False

                        # Validate source: must be focal or receiving
                        src_name = parts[0].strip().lstrip("(")
                        src_code = _rcc(src_name)
                        if src_code and src_code not in _valid_sources:
                            return False

                        return True

                    map_flows = [
                        f for f in map_flows
                        if _flow_endpoints_valid(f)
                    ]

                # Decide: ADM1 or country-level map?
                if adm1_region:
                    try:
                        from metacouplingllm.visualization.adm1_map import (
                            plot_focal_adm1_map,
                        )

                        print(
                            f"[MetacouplingAssistant] Structured map → "
                            f"ADM1 {adm1_region}, "
                            f"{len(map_flows)} flows, "
                            f"mentioned_adm1={sorted(mentioned_adm1_set)}."
                        )
                        # Render first, stamp the type only on success —
                        # if plot_focal_adm1_map raises, we fall through
                        # to the country-level path below and the type
                        # will be stamped "country" there instead.
                        fig = plot_focal_adm1_map(
                            adm1_region,
                            shapefile=self._adm1_shapefile,
                            mentioned_countries=(
                                mentioned if mentioned else None
                            ),
                            # Always pass the set, even when empty.
                            # An empty set means "strict mode, no
                            # regions qualify" — NOT "fall back to
                            # legacy coloring of all DB neighbors".
                            mentioned_adm1_codes=mentioned_adm1_set,
                            flows=map_flows if map_flows else None,
                        )
                        self._last_map_type = "adm1"
                        return fig
                    except Exception as exc:
                        if self._verbose:
                            print(
                                "[MetacouplingAssistant] Structured ADM1 "
                                f"map failed: {exc}"
                            )
                        # Fall through to country-level

                # Country-level map from structured data
                from metacouplingllm.visualization.worldmap import (
                    plot_analysis_map,
                )

                print(
                    f"[MetacouplingAssistant] Structured map → "
                    f"focal={focal_code}, "
                    f"receiving={receiving}, "
                    f"{len(map_flows)} flows."
                )
                fig = plot_analysis_map(
                    parsed,
                    adm0_shapefile=self._adm0_shapefile,
                    focal_code_override=focal_code,
                    mentioned_countries_override=mentioned,
                    flows=map_flows if map_flows else None,
                )
                self._last_map_type = "country"
                return fig

            # ==============================================================
            # FALLBACK PATH: regex extraction (legacy)
            # ==============================================================
            if md is not None:
                # map_data existed but was invalid — already warned
                pass
            else:
                print(
                    "[MetacouplingAssistant] No structured map data — "
                    "falling back to text extraction."
                )

            # Legacy Attempt 1: Try subnational (ADM1) map
            adm1_code = self._resolve_adm1_from_analysis(parsed)
            if adm1_code:
                try:
                    from metacouplingllm.visualization.adm1_map import (
                        plot_focal_adm1_map,
                    )
                    from metacouplingllm.visualization.worldmap import (
                        _extract_all_analysis_countries,
                    )

                    all_role_codes = _extract_all_analysis_countries(parsed)
                    mentioned_legacy: set[str] = set()
                    mentioned_legacy.update(
                        all_role_codes.get("sending", set())
                    )
                    mentioned_legacy.update(
                        all_role_codes.get("receiving", set())
                    )
                    mentioned_legacy.update(
                        all_role_codes.get("other", set())
                    )
                    # Skip spillover — receiving partners only.
                    mentioned_legacy.update(
                        self._structured_web_receiving_codes()
                    )

                    from metacouplingllm.knowledge.adm1_pericoupling import (
                        get_adm1_country,
                    )

                    focal_country = get_adm1_country(adm1_code) or ""
                    parsed_flows = list(parsed.iter_flow_entries())
                    if parsed_flows:
                        map_flows_legacy, flow_warnings = (
                            self._resolve_flows_for_adm1_map(
                                parsed, adm1_code, focal_country,
                            )
                        )
                        self._last_flow_parse_warnings = list(flow_warnings)
                    else:
                        map_flows_legacy = []
                    map_flows_legacy = self._merge_map_flows(
                        map_flows_legacy,
                        self._structured_web_flow_dicts(),
                    )

                    # Regex fallback for mentioned ADM1 regions.
                    mentioned_adm1_legacy = (
                        self._extract_mentioned_adm1_from_text(parsed)
                    )
                    # The focal ADM1 is handled separately by the
                    # classifier via ``focal_code``; drop it here.
                    mentioned_adm1_legacy.discard(adm1_code)

                    if self._verbose:
                        print(
                            f"[MetacouplingAssistant] Legacy ADM1 map for "
                            f"{adm1_code}, flows={len(map_flows_legacy)}, "
                            f"mentioned_adm1="
                            f"{sorted(mentioned_adm1_legacy)}."
                        )
                    # Render first, stamp the type only on success.
                    fig = plot_focal_adm1_map(
                        adm1_code,
                        shapefile=self._adm1_shapefile,
                        mentioned_countries=(
                            mentioned_legacy if mentioned_legacy else None
                        ),
                        # Always pass — empty set means strict mode
                        # with no matches, NOT legacy fallback.
                        mentioned_adm1_codes=mentioned_adm1_legacy,
                        flows=(
                            map_flows_legacy if map_flows_legacy else None
                        ),
                    )
                    self._last_map_type = "adm1"
                    return fig
                except Exception as exc:
                    if self._verbose:
                        print(
                            f"[MetacouplingAssistant] Legacy ADM1 map "
                            f"failed: {exc}"
                        )

            # Legacy Attempt 2: Country-level map
            from metacouplingllm.visualization.worldmap import (
                _extract_all_analysis_countries,
                _extract_countries_from_analysis,
                plot_analysis_map,
            )

            if self._verbose:
                print(
                    "[MetacouplingAssistant] Legacy country-level map..."
                )
            focal_code = ""
            all_role_codes = _extract_all_analysis_countries(parsed)
            sending_codes = all_role_codes.get("sending", set())
            focal_codes = all_role_codes.get("focal", set())
            if sending_codes:
                focal_code = sorted(sending_codes)[0]
            elif focal_codes:
                focal_code = sorted(focal_codes)[0]
            if not focal_code:
                focal_code = (
                    _extract_countries_from_analysis(parsed).get(
                        "sending"
                    )
                    or _extract_countries_from_analysis(parsed).get("focal")
                    or ""
                )
            if not focal_code and self._last_web_map_signals:
                structured_focal = self._last_web_map_signals.get(
                    "focal_country"
                )
                if isinstance(structured_focal, str):
                    focal_code = structured_focal
            if not focal_code:
                self._last_map_notice = self._infer_unavailable_map_notice(
                    parsed
                )
                if self._verbose:
                    print(
                        "[MetacouplingAssistant] Map generation skipped: "
                        "no country or ADM1 focal geography could be "
                        "resolved."
                    )
                return None
            parsed_flows = list(parsed.iter_flow_entries())
            if parsed_flows and focal_code:
                map_flows_legacy, flow_warnings = self._resolve_flows_for_map(
                    parsed, focal_code,
                )
                self._last_flow_parse_warnings = list(flow_warnings)
            else:
                map_flows_legacy = []
            map_flows_legacy = self._merge_map_flows(
                map_flows_legacy,
                self._structured_web_flow_dicts(),
            )
            if self._verbose:
                print(
                    f"[MetacouplingAssistant] Legacy map focal="
                    f"{focal_code or 'unknown'}, "
                    f"flows={len(map_flows_legacy)}."
                )
            fig = plot_analysis_map(
                parsed,
                adm0_shapefile=self._adm0_shapefile,
                # Skip spillover — they render as grey (NA).
                extra_mentioned_countries=(
                    self._structured_web_receiving_codes()
                ),
                flows=(
                    map_flows_legacy if map_flows_legacy else None
                ),
            )
            self._last_map_type = "country"
            return fig

        except ImportError:
            self._last_map_notice = self._format_map_unavailable_notice(
                "missing_dependencies"
            )
            if self._verbose:
                print(
                    "[MetacouplingAssistant] Map generation skipped "
                    "(geopandas/matplotlib not installed)."
                )
            return None
        except Exception as exc:
            self._last_map_notice = self._format_map_unavailable_notice(
                "generation_error"
            )
            if self._verbose:
                print(
                    f"[MetacouplingAssistant] Map generation failed: {exc}"
                )
            return None

    def _structured_web_country_codes(self) -> set[str]:
        """Return validated country codes extracted from structured web hints.

        Includes focal, receiving, and spillover systems. Prefer
        :meth:`_structured_web_receiving_codes` and
        :meth:`_structured_web_spillover_codes` when callers need to
        distinguish the two roles (e.g., to colour them differently).
        """
        return (
            self._structured_web_receiving_codes()
            | self._structured_web_spillover_codes()
        )

    def _structured_web_receiving_codes(self) -> set[str]:
        """Return validated receiving-system codes from web hints.

        Also includes the focal country so that it can be used as a
        map-ready "mentioned countries" set.
        """
        signals = self._last_web_map_signals
        if not signals:
            return set()

        codes: set[str] = set()
        focal = signals.get("focal_country")
        if isinstance(focal, str) and focal:
            codes.add(focal)
        items = signals.get("receiving_systems", [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Supranational receivers (PR #24): expand to member
                # ISO codes so the returned set stays semantically
                # pure (ISO codes only) for downstream callers like
                # get_country_name() and ISO_ALPHA3_NAMES lookups.
                # The display name ("European Union") is preserved
                # in the source signals dict for prompt-context
                # formatting.
                members = item.get("supranational_members")
                if isinstance(members, list) and members:
                    codes.update(
                        m for m in members if isinstance(m, str) and m
                    )
                    continue
                code = item.get("country")
                if isinstance(code, str) and code:
                    codes.add(code)
        return codes

    def _structured_web_spillover_codes(self) -> set[str]:
        """Return validated spillover-system codes from web hints."""
        signals = self._last_web_map_signals
        if not signals:
            return set()

        codes: set[str] = set()
        items = signals.get("spillover_systems", [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Same supranational-expansion path as
                # _structured_web_receiving_codes; see comment there.
                members = item.get("supranational_members")
                if isinstance(members, list) and members:
                    codes.update(
                        m for m in members if isinstance(m, str) and m
                    )
                    continue
                code = item.get("country")
                if isinstance(code, str) and code:
                    codes.add(code)
        return codes

    def _structured_web_flow_dicts(self) -> list[dict[str, str]]:
        """Return structured web-extracted flows as map-ready dicts."""
        signals = self._last_web_map_signals
        if not signals:
            return []

        flows = signals.get("flows", [])
        if not isinstance(flows, list):
            return []

        result: list[dict[str, str]] = []
        for item in flows:
            if not isinstance(item, dict):
                continue
            direction = item.get("direction")
            category = item.get("category")
            if not isinstance(direction, str) or not direction:
                continue
            if not isinstance(category, str) or not category:
                continue
            result.append(
                {
                    "category": category,
                    "direction": direction,
                    "description": str(item.get("description", "")).strip(),
                }
            )
        return result

    @staticmethod
    def _merge_map_flows(
        primary: list[dict[str, str]] | None,
        secondary: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """Merge map flow lists without duplicating category-direction pairs."""
        merged: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for flow_group in (primary or [], secondary or []):
            for flow in flow_group:
                if not isinstance(flow, dict):
                    continue
                category = str(flow.get("category", "")).strip().lower()
                direction = str(flow.get("direction", "")).strip()
                if not category or not direction:
                    continue
                key = (category, direction)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(flow)
        return merged

    @staticmethod
    def _format_map_notice(map_type: str) -> str:
        """Return a brief notice to append to formatted output."""
        if map_type == "adm1":
            return (
                "\n\n---\n"
                "📍 An ADM1 (subnational) metacoupling map has been "
                "generated.  Access it via `result.map`."
            )
        return (
            "\n\n---\n"
            "🗺️ A country-level metacoupling map has been generated.  "
            "Access it via `result.map`."
        )

    @staticmethod
    def _format_map_unavailable_notice(reason: str) -> str:
        """Return a brief notice when auto-map could not produce a figure."""
        if reason == "unsupported_local_geography":
            return (
                "\n\n---\n"
                "Auto-map did not generate a figure. It currently supports "
                "only countries and ADM1 subnational regions, not city, "
                "watershed, protected-area, reserve, or park geometries. "
                "Specify the parent ADM1 region and country only if an "
                "ADM1-level proxy map is acceptable."
            )
        if reason == "missing_dependencies":
            return (
                "\n\n---\n"
                "Auto-map did not generate a figure because the map "
                "dependencies are not installed. Install geopandas and "
                "matplotlib to enable map output."
            )
        if reason == "generation_error":
            return (
                "\n\n---\n"
                "Auto-map did not generate a figure. It currently supports "
                "only countries and ADM1 subnational regions, and this "
                "analysis did not resolve cleanly to one of those map types."
            )
        return (
            "\n\n---\n"
            "Auto-map did not generate a figure. It currently supports only "
            "countries and ADM1 subnational regions. Specify a resolvable "
            "country or ADM1 region to get a map."
        )

    def _infer_unavailable_map_notice(self, parsed: ParsedAnalysis) -> str:
        """Infer the most useful notice when auto-map cannot render."""
        scope_parts = [self._original_query or ""]
        for role in ("focal", "adjacent", "sending", "receiving", "spillover"):
            scope_parts.append(parsed.get_system_detail(role, "name"))
            scope_parts.append(parsed.get_system_detail(role, "geographic_scope"))
        combined = " ".join(part for part in scope_parts if part)
        if _UNSUPPORTED_AUTOMAP_SCOPE_RE.search(combined):
            return self._format_map_unavailable_notice(
                "unsupported_local_geography"
            )
        return self._format_map_unavailable_notice(
            "unresolved_supported_geography"
        )

    def _has_unsupported_automap_scope(self, parsed: ParsedAnalysis) -> bool:
        """Return True when no country/ADM1-scale map can be rendered.

        Trusts the LLM's structured map extraction first: if it
        produced a ``focal_country`` or ``adm1_region``, those are
        validated ISO codes and the map IS renderable -- even when
        the prose mentions municipalities, watersheds, or other
        sub-ADM1 geographies in secondary roles.

        Only falls back to the prose-keyword regex when the
        structured extraction produced no resolvable focal at all.
        In that case the keyword scan helps emit a sensible
        ``unsupported_local_geography`` notice instead of a generic
        "unresolved" one.
        """
        md = parsed.map_data or {}
        if md.get("focal_country") or md.get("adm1_region"):
            # LLM structured extraction is authoritative; the map can
            # be rendered at country or ADM1 scale regardless of what
            # the prose mentions in secondary roles.
            return False

        # Structured extraction produced no focal -- fall back to the
        # legacy regex scan over prose to decide if this is a
        # "sub-ADM1 geography" case worth a specific user notice.
        scope_parts = [self._original_query or ""]
        for role in ("focal", "adjacent", "sending", "receiving", "spillover"):
            scope_parts.append(parsed.get_system_detail(role, "name"))
            scope_parts.append(parsed.get_system_detail(role, "geographic_scope"))
        combined = " ".join(part for part in scope_parts if part)
        return bool(_UNSUPPORTED_AUTOMAP_SCOPE_RE.search(combined))

    def _user_query_mentions_adm1(self) -> bool:
        """Return True when the user's original query names any ADM1 region.

        Used by the auto-map dispatcher to prefer country-level
        rendering when the user's framing was country-level (no
        subnational region named), even when the LLM's structured
        extraction identified an ADM1 focal in the analysis content.
        This avoids zooming into Michoacan when the user asked about
        "Mexican avocado trade" -- they expected a world map.

        Implementation: scan capitalised word groups in the query
        against the ADM1 database via ``resolve_adm1_code``.  Word
        groups that also resolve to a country (e.g. "Mexico" -- both
        a country and a Mexican state) are treated as country
        mentions and skipped, because users naming a country at the
        top level are framing at country scale.  ISO-style codes
        ("MEX016") are also recognised.
        """
        query = (self._original_query or "").strip()
        if not query:
            return False

        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_info,
            resolve_adm1_code,
        )

        # Capitalised word groups: "Michoacán", "São Paulo",
        # "New South Wales".  Same regex shape as
        # ``_extract_mentioned_adm1_from_text`` for consistency.
        # Note: ``resolve_adm1_code`` matches the database's exact
        # form, which preserves Spanish/Portuguese diacritics
        # ("Michoacán", not "Michoacan").  A user typing the
        # unaccented form will hit the country-level fallback;
        # that's the safe default — users can use ISO codes
        # ("MEX016") to force ADM1 explicitly.
        for word_group in re.findall(
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query
        ):
            # Country mentions take precedence: if the word group
            # resolves to a country, the user is framing at country
            # scale (e.g. "Mexico" the country, not the state).
            if resolve_country_code(word_group):
                continue
            if resolve_adm1_code(word_group):
                return True

        # ISO-style ADM1 codes: "MEX016", "USA023".
        # ``get_adm1_info`` is the direct-code lookup;
        # ``resolve_adm1_code`` only matches names.
        for code in re.findall(r"\b([A-Z]{3}\d{3})\b", query):
            if get_adm1_info(code) is not None:
                return True

        return False

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _format_web_sources(
        results: list[dict[str, str]],
        turn: int = 1,
    ) -> str:
        """Format web search results as a references section.

        Uses turn-scoped ``[Tk:Wn]`` headers to match the inline
        citation grammar in the answer body.
        """
        if not results:
            return ""
        lines = [
            "\n\n======================================================================",
            f"  WEB SOURCES (turn {turn})",
            "======================================================================\n",
        ]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            # Accept legacy "snippet" key as fallback.
            model_summary = (
                r.get("model_summary") or r.get("snippet", "")
            )
            lines.append(f"  [T{turn}:W{i}] {title}")
            if url:
                lines.append(f"      {url}")
            if model_summary:
                lines.append(f"      {model_summary[:200]}")
            lines.append("")
        return "\n".join(lines)

    def _build_result(self, response: LLMResponse) -> AnalysisResult:
        """Parse and format an LLM response into an AnalysisResult."""
        # Pre-retrieval turn-scoped citation sanitization. Run BEFORE
        # parse_analysis() so the parser never sees invalid tokens —
        # the parser preserves [Tk:N] tokens through structured field
        # extraction, so any forward refs or out-of-range tokens would
        # otherwise survive into the final output. Bare legacy [N] /
        # [W1] from a slipping LLM are also stripped here silently.
        # The sanitizer also runs an idempotent whitespace/punctuation
        # cleanup pass so the result reads naturally even when tokens
        # were stripped.
        if (
            self._rag_mode == "pre_retrieval"
            and self._last_rag_hits is not None
        ):
            sanitized, invalid_tokens = sanitize_turn_citations(
                response.content,
                turn_passage_counts=self._turn_passage_counts,
                turn_web_counts=self._turn_web_counts,
                current_turn=self._turn,
            )
            if invalid_tokens:
                logger.warning(
                    "MetacouplingAssistant sanitized %d invalid citation "
                    "token(s) from LLM response: %s",
                    len(invalid_tokens),
                    sorted(invalid_tokens),
                )
            response = LLMResponse(content=sanitized, usage=response.usage)

        parsed = parse_analysis(response.content)

        # Post-LLM pericoupling validation
        self._validate_pericoupling(parsed)

        formatted = self._formatter.format_full(parsed)

        # Optional: append literature recommendations
        if self._recommend_papers:
            from metacouplingllm.knowledge.literature import (
                format_recommendations,
                recommend_papers as _recommend,
            )

            papers = _recommend(parsed, max_results=self._max_recommendations)
            if papers:
                formatted += "\n\n" + format_recommendations(papers)

        # RAG evidence assembly. Two modes:
        #
        # - "post_hoc" (legacy): retrieve from corpus AFTER LLM response,
        #   keyword-annotate citations onto the formatted text, then
        #   append the evidence block. Code path unchanged from
        #   pre-upgrade behavior.
        #
        # - "pre_retrieval": passages were already retrieved at
        #   analyze() / refine() time and stored on
        #   self._last_rag_hits. The LLM has already cited them inline
        #   (subject to sanitization above). We just need to append
        #   the human-readable evidence block — same format as post_hoc
        #   mode so downstream consumers see no shape change.
        structured_supplement: dict[str, object] | None = None
        if self._rag_engine is not None:
            if self._rag_mode == "pre_retrieval":
                # Optional: second LLM call that scans the retrieved
                # passages for systems / flows the draft missed. The
                # supplement is rendered as its own visible block so
                # the main analysis body stays 100% LLM-authored.
                if (
                    self._rag_structured_extraction
                    and self._last_rag_hits
                ):
                    try:
                        structured_supplement = (
                            self._structured_extract_supplement(parsed)
                        )
                    except Exception as exc:
                        logger.warning(
                            "Structured extraction supplement failed: "
                            "%s. Skipping supplement.", exc,
                        )
                        structured_supplement = None

                if structured_supplement is not None:
                    try:
                        formatted += "\n\n" + self._format_structured_supplement(
                            structured_supplement
                        )
                    except Exception as exc:
                        if self._verbose:
                            print(
                                f"[MetacouplingAssistant] Structured "
                                f"supplement formatting failed: {exc}"
                            )

                try:
                    from metacouplingllm.knowledge.rag import format_evidence

                    if self._last_rag_hits:
                        rag_backend = (
                            self._rag_engine.backend
                            if self._rag_engine is not None
                            else "tfidf"
                        )
                        formatted += "\n\n" + format_evidence(
                            self._last_rag_hits,
                            anchor_text=self._original_query or "",
                            backend=rag_backend or "tfidf",
                            turn=self._turn,
                        )
                    elif self._verbose:
                        print(
                            "[MetacouplingAssistant] Pre-retrieval RAG: "
                            "no passages to append (empty hits)."
                        )
                except Exception as exc:
                    if self._verbose:
                        print(
                            f"[MetacouplingAssistant] Pre-retrieval RAG "
                            f"evidence formatting failed: {exc}"
                        )
            else:  # post_hoc — existing behavior preserved exactly
                try:
                    from metacouplingllm.knowledge.rag import (
                        _build_query_from_analysis,
                        annotate_citations,
                        format_evidence,
                    )

                    query = _build_query_from_analysis(parsed)
                    if self._verbose:
                        print(
                            f"[MetacouplingAssistant] RAG query "
                            f"({len(query)} chars): {query[:120]}..."
                        )

                    if not query.strip():
                        if self._verbose:
                            print(
                                "[MetacouplingAssistant] RAG query is empty — "
                                "skipping evidence retrieval."
                            )
                    else:
                        results = self._rag_engine.retrieve(
                            query,
                            top_k=self._rag_top_k,
                            min_score=self._rag_min_score,
                            max_chunks_per_paper=self._rag_max_chunks_per_paper,
                        )
                        if self._verbose:
                            print(
                                f"[MetacouplingAssistant] RAG returned "
                                f"{len(results)} evidence passages."
                            )
                        if results:
                            # Add inline [Tk:N] citations to analysis
                            # statements via keyword-overlap matching.
                            formatted = annotate_citations(
                                formatted, results, turn=self._turn,
                            )
                            # Append the evidence reference block, passing
                            # the active backend so confidence thresholds
                            # match the score range.
                            rag_backend = (
                                self._rag_engine.backend
                                if self._rag_engine is not None
                                else "tfidf"
                            )
                            formatted += "\n\n" + format_evidence(
                                results,
                                anchor_text=query,
                                backend=rag_backend or "tfidf",
                                turn=self._turn,
                            )
                except Exception as exc:
                    if self._verbose:
                        print(f"[MetacouplingAssistant] RAG evidence failed: {exc}")
                    # Non-fatal: continue without RAG evidence

        # Optional: annotate and append web search sources
        if self._last_web_results:
            try:
                from metacouplingllm.knowledge.websearch import (
                    annotate_web_citations,
                )

                formatted = annotate_web_citations(
                    formatted, self._last_web_results,
                    turn=self._turn,
                )
            except Exception:
                pass  # Non-fatal: skip web annotation
            formatted += self._format_web_sources(
                self._last_web_results, turn=self._turn,
            )

        # Optional: second LLM call to extract structured map data
        if self._auto_map and parsed.map_data is None:
            try:
                parsed.map_data = self._extract_map_data_from_analysis(
                    parsed,
                )
            except Exception as exc:
                print(
                    "[MetacouplingAssistant] Map data extraction "
                    f"failed: {exc}"
                )
                parsed.map_data = None

        # Enrich map data with specific countries / flows the
        # structured-extraction supplement found in the retrieved
        # passages. ``_extract_map_data_from_analysis`` reads the
        # finished analysis text, which often uses generic wording
        # ("foreign origin countries"); the supplement, by contrast,
        # reads the RAG passages directly with a schema that forces
        # per-country entries. Merging the two gives the map the
        # specific bilateral partners mentioned in the corpus.
        if (
            self._auto_map
            and parsed.map_data is not None
            and structured_supplement is not None
        ):
            try:
                self._merge_supplement_into_map_data(
                    parsed, structured_supplement,
                )
            except Exception as exc:
                logger.warning(
                    "Merging structured supplement into map_data "
                    "failed: %s. Map will use only the direct "
                    "extraction output.", exc,
                )

        # Optional: auto-generate map
        fig: Figure | None = None
        map_notice: str | None = None
        if self._auto_map:
            fig = self._generate_map(parsed)
            if fig is not None:
                # Close the figure from pyplot's state manager so it does
                # NOT auto-display in Jupyter / Colab notebooks.  The
                # Figure object itself remains fully usable — the user
                # can still call result.map.savefig(...) or
                # display(result.map).
                try:
                    import matplotlib.pyplot as _plt

                    _plt.close(fig)
                except Exception:
                    pass

                # Use the map type that ``_generate_map`` actually
                # rendered (recorded in ``self._last_map_type``).  This
                # avoids the prior bug where the notice was recomputed
                # from the same inputs the renderer used, and could
                # disagree with reality when the renderer's ADM1 attempt
                # silently fell through to a country-level map.
                map_type = self._last_map_type or "country"
                map_notice = self._format_map_notice(map_type)
                formatted += map_notice
            elif self._last_map_notice:
                map_notice = self._last_map_notice
                formatted += map_notice

        # Append a "Suggested follow-up web searches" footer when the
        # web extraction emitted any.  ``evidence_coverage_note`` is
        # already rendered into ``formatted`` by ``AnalysisFormatter
        # .format_full`` via ``parsed.evidence_coverage_note``; only
        # the follow-up queries live OUTSIDE ParsedAnalysis (they're
        # on ``self._last_web_map_signals``), so they're appended here.
        followup_queries: list[str] = []
        if isinstance(self._last_web_map_signals, dict):
            raw_q = self._last_web_map_signals.get(
                "suggested_followup_queries", []
            )
            if isinstance(raw_q, list):
                followup_queries = [str(q) for q in raw_q if str(q).strip()]
        if followup_queries:
            footer = [
                "",
                "----------------------------------------------------------------------",
                "  Suggested follow-up web searches:",
                "",
            ]
            for q in followup_queries:
                footer.append(f"  - {q}")
            footer.append(
                "======================================================================"
            )
            formatted += "\n" + "\n".join(footer)

        return AnalysisResult(
            parsed=parsed,
            formatted=formatted,
            raw=response.content,
            turn_number=self._turn,
            usage=response.usage or None,
            map=fig,
            web_map_signals=self._last_web_map_signals,
            structured_supplement=structured_supplement,
            map_notice=map_notice,
            flow_parse_warnings=list(self._last_flow_parse_warnings),
            evidence_coverage_note=parsed.evidence_coverage_note,
            suggested_followup_queries=followup_queries,
        )

    @staticmethod
    def _validate_adm1_pericoupling(parsed: ParsedAnalysis) -> bool:
        """Try to validate at the ADM1 (subnational) level.

        If the analysis contains a resolvable subnational region,
        populates ``parsed.pericoupling_info`` with ADM1-level neighbor
        information and returns ``True``.  Otherwise returns ``False``
        so the caller can fall through to country-level validation.
        """
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_info,
            get_adm1_neighbors,
            get_cross_border_neighbors,
        )

        adm1_code = MetacouplingAssistant._resolve_adm1_from_analysis(parsed)
        if adm1_code is None:
            return False

        focal_info = get_adm1_info(adm1_code)
        if not focal_info:
            return False

        all_neighbors = get_adm1_neighbors(adm1_code)
        cross_border = get_cross_border_neighbors(adm1_code)
        domestic = all_neighbors - cross_border

        def _names(codes: set[str]) -> str:
            items: list[str] = []
            for c in sorted(codes):
                info = get_adm1_info(c)
                if info:
                    items.append(f"{info['name']} ({c})")
            return ", ".join(items)

        peri_info: dict[str, str] = {
            "level": "adm1",
            "focal_region": (
                f"{focal_info['name']} ({adm1_code})"
            ),
            "focal_country": (
                f"{focal_info['country_name']} ({focal_info['iso_a3']})"
            ),
        }

        if domestic:
            peri_info["domestic_neighbors"] = _names(domestic)
        if cross_border:
            peri_info["cross_border_neighbors"] = _names(cross_border)

        # Real consistency check (mirrors the country-level sibling
        # `_validate_pericoupling`).  Classify each mentioned ADM1
        # region against the focal's database neighbour set: a
        # mentioned region is "pericoupled" with the focal iff it
        # appears in the focal's adjacency set (domestic or
        # cross-border), otherwise "telecoupled".  Then compare the
        # LLM's text-level classification against what the database
        # actually shows.
        mentioned_adm1 = (
            MetacouplingAssistant._extract_mentioned_adm1_from_text(parsed)
        )
        mentioned_adm1.discard(adm1_code)  # never compare focal with itself
        all_neighbours = domestic | cross_border
        peri_partners = mentioned_adm1 & all_neighbours
        tele_partners = mentioned_adm1 - all_neighbours
        has_peri = bool(peri_partners)
        has_tele = bool(tele_partners)

        if parsed.coupling_classification:
            llm_class = parsed.coupling_classification.lower()
            if has_peri and "pericoupl" not in llm_class:
                peri_info["note"] = (
                    "The ADM1 pericoupling database indicates at "
                    "least one mentioned subnational region is "
                    "adjacent to the focal region, but the LLM did "
                    "not classify this study as pericoupling.  "
                    "Consider revising."
                )
            elif has_tele and not has_peri and "telecoupl" not in llm_class:
                peri_info["note"] = (
                    "The ADM1 pericoupling database indicates all "
                    "mentioned subnational regions are non-adjacent "
                    "(telecoupled) to the focal region, but the LLM "
                    "did not classify this study as telecoupling.  "
                    "Consider revising."
                )
            else:
                peri_info["note"] = (
                    "LLM classification is consistent with the "
                    "ADM1 pericoupling database."
                )
        else:
            # No LLM classification text to compare against — describe
            # the database result neutrally instead of claiming a
            # consistency that wasn't checked.
            peri_info["note"] = (
                "ADM1 pericoupling database returned neighbour "
                "information for the focal region (no LLM "
                "classification to cross-check)."
            )

        parsed.pericoupling_info = peri_info
        return True

    @staticmethod
    def _validate_pericoupling(parsed: ParsedAnalysis) -> None:
        """Extract country names from systems and validate against the DB.

        Only validates pairs involving the **sending system** (focal
        country).  This avoids spurious validation of pairs between
        receiving countries that the researcher did not intend to
        compare (e.g. USA ↔ Canada when the study is Mexico → USA/CAN).

        Populates ``parsed.pericoupling_info`` with the validation result.
        """
        # Try ADM1 (subnational) validation first.
        if MetacouplingAssistant._validate_adm1_pericoupling(parsed):
            return

        # --- Identify the focal (sending) country first ---
        focal_code: str | None = None
        focal_name: str | None = None
        sending_entry = (
            parsed.get_first_system_entry("sending")
            or parsed.get_first_system_entry("focal")
        )
        if sending_entry is not None:
            for key in ("name", "geographic_scope"):
                value = sending_entry.get(key, "")
                if value:
                    code = resolve_country_code(value)
                    if code:
                        focal_code = code
                        focal_name = value
                        break

        # Fall back to first detected country from all systems
        if focal_code is None:
            countries = MetacouplingAssistant._extract_country_names(parsed)
            if countries:
                code = resolve_country_code(countries[0])
                if code:
                    focal_code = code
                    focal_name = countries[0]

        if focal_code is None:
            return  # Cannot identify any country

        # --- Collect other (non-focal) countries ---
        all_countries = MetacouplingAssistant._extract_country_names(parsed)
        other_codes: list[tuple[str, str]] = []  # (code, display_name)
        seen: set[str] = {focal_code}
        for name in all_countries:
            code = resolve_country_code(name)
            if code and code not in seen:
                seen.add(code)
                other_codes.append((code, name))

        if not other_codes:
            return  # Only one country detected

        # --- Build validation info (focal ↔ each other only) ---
        info: dict[str, str] = {}
        info[f"focal_country"] = f"{get_country_name(focal_code)} ({focal_code})"

        pair_lines: list[str] = []
        for code_b, _name_b in other_codes:
            result = lookup_pericoupling(focal_code, code_b)
            cn_a = get_country_name(focal_code)
            cn_b = get_country_name(code_b)
            if result.pair_type == PairCouplingType.PERICOUPLED:
                pair_lines.append(
                    f"{cn_a} ({focal_code}) ↔ {cn_b} ({code_b}): PERICOUPLED"
                )
            elif result.pair_type == PairCouplingType.TELECOUPLED:
                pair_lines.append(
                    f"{cn_a} ({focal_code}) ↔ {cn_b} ({code_b}): TELECOUPLED"
                )

        if pair_lines:
            info["pair_results"] = "; ".join(pair_lines)

        # Check agreement with LLM classification.
        if parsed.coupling_classification:
            llm_class = parsed.coupling_classification.lower()
            has_peri = any("PERICOUPLED" in p for p in pair_lines)
            has_tele = any("TELECOUPLED" in p for p in pair_lines)

            if has_peri and "pericoupl" not in llm_class:
                info["note"] = (
                    "The pericoupling database indicates at least one "
                    "pericoupled country pair, but the LLM classified "
                    "this study differently. Consider revising."
                )
            elif has_tele and not has_peri and "telecoupl" not in llm_class:
                info["note"] = (
                    "The pericoupling database indicates all detected "
                    "pairs are telecoupled, but the LLM classified "
                    "this study differently. Consider revising."
                )
            else:
                info["note"] = (
                    "LLM classification is consistent with the "
                    "pericoupling database."
                )

        parsed.pericoupling_info = info

    @staticmethod
    def _extract_country_names(parsed: ParsedAnalysis) -> list[str]:
        """Extract country-like names from parsed systems data.

        Looks at sending, receiving, and spillover system names/descriptions
        and tries to identify country references.
        """
        texts: list[str] = []
        for role in ("focal", "adjacent", "sending", "receiving", "spillover"):
            for entry in parsed.get_system_entries(role):
                if entry.get("name"):
                    texts.append(entry["name"])
                if entry.get("geographic_scope"):
                    texts.append(entry["geographic_scope"])

        # Also scan the coupling_classification for country mentions
        if parsed.coupling_classification:
            texts.append(parsed.coupling_classification)

        # Resolve unique countries from all collected text
        seen: set[str] = set()
        countries: list[str] = []
        for text in texts:
            # Try resolving the full text first (e.g. "Mexico")
            code = resolve_country_code(text)
            if code and code not in seen:
                seen.add(code)
                countries.append(text)
                continue
            # Otherwise scan for country names within the text
            # Split on common delimiters
            for chunk in re.split(r"[,;/()]+", text):
                chunk = chunk.strip()
                if not chunk:
                    continue
                code = resolve_country_code(chunk)
                if code and code not in seen:
                    seen.add(code)
                    countries.append(chunk)

        return countries
