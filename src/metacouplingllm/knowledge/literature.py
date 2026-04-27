"""
Literature recommendation engine for telecoupling/metacoupling research.

Parses a bundled BibTeX database of telecoupling and metacoupling papers
and recommends relevant publications based on keyword matching against
a user's research topic or parsed LLM analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metacouplingllm.llm.parser import ParsedAnalysis


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    """A single bibliographic entry from the literature database.

    Attributes
    ----------
    key:
        BibTeX citation key (e.g., ``"liu_telecoupling_2013"``).
    title:
        Paper title.
    authors:
        Author string (e.g., ``"Liu, Jianguo and Hull, Vanessa"``).
    year:
        Publication year.
    journal:
        Journal name.
    doi:
        DOI string (without ``https://doi.org/`` prefix).
    keywords:
        Set of normalised (lowercased) keywords.
    cited_by:
        Citation count from Web of Science (0 if unavailable).
    """

    key: str = ""
    title: str = ""
    authors: str = ""
    year: int = 0
    journal: str = ""
    doi: str = ""
    keywords: set[str] = field(default_factory=set)
    cited_by: int = 0


# ---------------------------------------------------------------------------
# Relevance filter — only keep coupling-related papers
# ---------------------------------------------------------------------------

_RELEVANCE_TERMS = {
    "telecoupling", "tele-coupling", "metacoupling", "meta-coupling",
    "pericoupling", "peri-coupling", "intracoupling", "intra-coupling",
    "coupled human and natural", "coupled human-natural",
    "chans", "coupled systems",
    "sending system", "receiving system", "spillover system",
    "socioeconomic-environmental", "human-nature interaction",
    "land use teleconnection", "teleconnection",
}


def _is_relevant(paper: Paper) -> bool:
    """Return True if the paper is related to telecoupling/metacouplingllm.

    All papers in the bundled database are curated, so this filter uses
    title and keywords only (no abstract dependency).
    """
    searchable = paper.title.lower() + " " + " ".join(paper.keywords)
    return any(term in searchable for term in _RELEVANCE_TERMS)


# ---------------------------------------------------------------------------
# BibTeX parser (lightweight, no external dependency)
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r"@(\w+)\{([^,]+),\s*(.*?)\n\}",
    re.DOTALL,
)

_FIELD_RE = re.compile(
    r"(\w+)\s*=\s*\{(.*?)\}(?:\s*,)?",
    re.DOTALL,
)


def _parse_bibtex(text: str) -> list[Paper]:
    """Parse a BibTeX string into a list of :class:`Paper` objects."""
    papers: list[Paper] = []

    for match in _ENTRY_RE.finditer(text):
        entry_type = match.group(1).lower()
        key = match.group(2).strip()
        body = match.group(3)

        if entry_type not in ("article", "incollection", "inproceedings", "book"):
            continue

        fields: dict[str, str] = {}
        for fm in _FIELD_RE.finditer(body):
            fname = fm.group(1).lower().strip()
            fval = fm.group(2).strip()
            # Normalise whitespace
            fval = re.sub(r"\s+", " ", fval)
            fields[fname] = fval

        # Parse citation count from annote field
        cited_by = 0
        annote = fields.get("annote", "")
        cite_m = re.search(r"Cited by:\s*(\d+)", annote)
        if cite_m:
            cited_by = int(cite_m.group(1))

        # Parse keywords into a set
        raw_kw = fields.get("keywords", "")
        keywords: set[str] = set()
        if raw_kw:
            for kw in re.split(r"\s*,\s*", raw_kw):
                kw = kw.strip().lower()
                if kw:
                    keywords.add(kw)

        year = 0
        year_str = fields.get("year", "")
        if year_str.isdigit():
            year = int(year_str)

        paper = Paper(
            key=key,
            title=fields.get("title", "").replace("{", "").replace("}", ""),
            authors=fields.get("author", "").replace("{", "").replace("}", ""),
            year=year,
            journal=fields.get("journal", "").replace("{", "").replace("}", "").replace("\\&", "&"),
            doi=fields.get("doi", ""),
            keywords=keywords,
            cited_by=cited_by,
        )
        papers.append(paper)

    return papers


# ---------------------------------------------------------------------------
# Database loading (lazy, cached)
# ---------------------------------------------------------------------------

_paper_db: list[Paper] | None = None


def _locate_bib() -> Path | None:
    """Find the bundled BibTeX database."""
    try:
        data_pkg = resources.files("metacouplingllm") / "data"
    except (TypeError, ModuleNotFoundError):
        return None

    candidate = data_pkg / "telecoupling_literature.bib"
    try:
        path = Path(str(candidate))
        if path.is_file():
            return path
    except Exception:
        pass
    return None


def _load_database() -> list[Paper]:
    """Load and filter the literature database."""
    bib_path = _locate_bib()
    if bib_path is None:
        return []

    text = bib_path.read_text(encoding="utf-8-sig")
    all_papers = _parse_bibtex(text)

    # All papers in the bundled database are curated for
    # telecoupling/metacoupling research — no filtering needed.
    return all_papers


def _get_database() -> list[Paper]:
    """Lazily load and cache the paper database."""
    global _paper_db
    if _paper_db is None:
        _paper_db = _load_database()
    return _paper_db


# ---------------------------------------------------------------------------
# Keyword extraction from analysis
# ---------------------------------------------------------------------------


def _extract_search_terms_from_analysis(parsed: ParsedAnalysis) -> set[str]:
    """Extract search keywords from a parsed LLM analysis."""
    terms: set[str] = set()

    # From coupling classification
    if parsed.coupling_classification:
        text = parsed.coupling_classification.lower()
        for word in re.findall(r"[a-z][\w-]+", text):
            if len(word) > 3:
                terms.add(word)

    # From active coupling types
    for coupling_type in parsed.active_coupling_types():
        terms.add(coupling_type.lower())

    # From all substantive text fragments across coupling blocks
    for fragment in parsed.iter_text_fragments():
        for word in re.findall(r"[a-z][\w-]+", fragment.lower()):
            if len(word) > 3:
                terms.add(word)

    # From grouped cause/effect category labels
    for kind in ("causes", "effects"):
        for _coupling_type, category, _item in parsed.iter_category_items(kind):
            if category:
                terms.add(category.lower())

    # Remove very common stopwords
    stopwords = {
        "this", "that", "with", "from", "have", "been", "were", "will",
        "also", "more", "than", "such", "each", "which", "their", "there",
        "these", "those", "about", "into", "over", "between", "through",
        "during", "before", "after", "other", "some", "both", "most",
        "very", "just", "only", "well", "including", "based", "using",
        "system", "systems", "analysis", "study", "research", "framework",
        "intracoupling", "pericoupling", "telecoupling",
    }
    terms -= stopwords

    return terms


def _extract_search_terms_from_text(text: str) -> set[str]:
    """Extract search keywords from a plain text query."""
    terms: set[str] = set()
    for word in re.findall(r"[a-z][\w-]+", text.lower()):
        if len(word) > 3:
            terms.add(word)

    stopwords = {
        "this", "that", "with", "from", "have", "been", "were", "will",
        "also", "more", "than", "such", "each", "which", "their", "there",
        "these", "those", "about", "into", "over", "between", "through",
        "during", "before", "after", "other", "some", "both", "most",
        "very", "just", "only", "well", "including", "based", "using",
        "analyze", "analysis", "study", "research", "please", "help",
    }
    terms -= stopwords

    return terms


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_paper(
    paper: Paper,
    search_terms: set[str],
    fulltext_scores: dict[str, float] | None = None,
) -> float:
    """Score a paper's relevance to the search terms.

    Scoring components:

    - Keyword match: each matching keyword adds 3.0 points
    - Title match: each matching word in title adds 2.0 points
    - Full-text match: cosine similarity × 3.0 (up to 3.0 points),
      sourced from the RAG TF-IDF index over bundled MD files
    - Citation bonus: log-scaled citation count adds up to 2.0 points

    Parameters
    ----------
    paper:
        The paper to score.
    search_terms:
        Lowercased keyword tokens from the query.
    fulltext_scores:
        Optional mapping of ``paper_key → cosine_similarity`` from the
        RAG engine.  When provided, the full-text score is included.
    """
    import math

    score = 0.0

    # Keyword matches (highest weight — these are curated by authors)
    for term in search_terms:
        for kw in paper.keywords:
            if term in kw or kw in term:
                score += 3.0
                break

    # Title matches
    title_lower = paper.title.lower()
    for term in search_terms:
        if term in title_lower:
            score += 2.0

    # Full-text relevance from RAG TF-IDF index (replaces abstract scoring)
    if fulltext_scores and paper.key in fulltext_scores:
        score += min(3.0, fulltext_scores[paper.key] * 3.0)

    # Citation bonus (log-scaled, max ~2.0 for 100+ citations)
    if paper.cited_by > 0:
        score += min(2.0, math.log10(paper.cited_by + 1))

    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_fulltext_scores(
    search_terms: set[str],
    top_k: int = 50,
    backend: str = "auto",
) -> dict[str, float]:
    """Query the RAG index and return per-paper relevance scores.

    Returns a ``{paper_key: similarity_score}`` dict.  If the RAG
    engine cannot be loaded, prints a warning and returns an empty dict.

    Parameters
    ----------
    search_terms:
        Lowercased query tokens.
    top_k:
        Maximum number of matching chunks to return.
    backend:
        Retrieval backend (``"auto"``, ``"embeddings"``, or
        ``"tfidf"``). See :class:`~metacouplingllm.knowledge.rag.RAGEngine`.
    """
    try:
        from metacouplingllm.knowledge.rag import RAGEngine, _extract_bundled_papers
    except ImportError as exc:
        print(
            f"[Literature] Full-text scoring unavailable: {exc}. "
            "Recommendations use title + keyword matching only."
        )
        return {}

    try:
        papers_dir = _extract_bundled_papers(verbose=False)
    except Exception as exc:
        print(
            f"[Literature] Full-text scoring unavailable: {exc}. "
            "Recommendations use title + keyword matching only."
        )
        return {}

    engine = RAGEngine(papers_dir, backend=backend)
    try:
        engine.load()
    except Exception as exc:
        print(
            f"[Literature] Full-text scoring unavailable: {exc}. "
            "Recommendations use title + keyword matching only."
        )
        return {}

    query_text = " ".join(sorted(search_terms))
    # Pass None so the engine picks a backend-appropriate default.
    # For TF-IDF this is 0.01; for embeddings 0.3. But since this is
    # a permissive paper-scoring use case (not evidence surfacing),
    # use a very low threshold to keep almost all matches.
    min_score = 0.001 if engine.backend == "tfidf" else 0.0
    results = engine.retrieve(query_text, top_k=top_k, min_score=min_score)
    return {r.chunk.paper_key: r.score for r in results}


def recommend_papers(
    query: str | ParsedAnalysis,
    *,
    max_results: int = 5,
) -> list[Paper]:
    """Recommend relevant papers from the telecoupling literature database.

    Uses title, keywords, citation count, and **full-text TF-IDF
    relevance** (from bundled MD files) to rank papers.

    Parameters
    ----------
    query:
        Either a plain-text research topic (e.g., ``"soybean trade Brazil
        China deforestation"``) or a :class:`ParsedAnalysis` from an LLM
        analysis result.
    max_results:
        Maximum number of papers to return.  Default is 5.

    Returns
    -------
    A list of :class:`Paper` objects, ranked by relevance (highest first).

    Examples
    --------
    >>> papers = recommend_papers("avocado trade Mexico United States")
    >>> for p in papers:
    ...     print(f"{p.year} - {p.title} (cited: {p.cited_by})")

    >>> result = advisor.analyze("Coffee trade Ethiopia Europe")
    >>> papers = recommend_papers(result.parsed, max_results=10)
    """
    # Import here to avoid circular imports
    from metacouplingllm.llm.parser import ParsedAnalysis as _PA

    db = _get_database()
    if not db:
        return []

    # Extract search terms
    if isinstance(query, _PA):
        search_terms = _extract_search_terms_from_analysis(query)
    elif isinstance(query, str):
        search_terms = _extract_search_terms_from_text(query)
    else:
        return []

    if not search_terms:
        return []

    # Build full-text relevance scores from RAG engine
    fulltext_scores = _build_fulltext_scores(search_terms)

    # Score and rank
    scored: list[tuple[float, Paper]] = []
    for paper in db:
        s = _score_paper(paper, search_terms, fulltext_scores)
        if s > 0:
            scored.append((s, paper))

    # Sort by score descending, then by citation count, then by year
    scored.sort(key=lambda t: (t[0], t[1].cited_by, t[1].year), reverse=True)

    return [paper for _, paper in scored[:max_results]]


def format_recommendations(papers: list[Paper]) -> str:
    """Format a list of recommended papers as a human-readable string.

    Parameters
    ----------
    papers:
        List of :class:`Paper` objects from :func:`recommend_papers`.

    Returns
    -------
    A formatted string ready for display.
    """
    if not papers:
        return "No relevant papers found in the literature database."

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  RECOMMENDED LITERATURE")
    lines.append("=" * 70)
    lines.append("")

    for i, p in enumerate(papers, 1):
        # Shorten author list if too long
        authors = p.authors
        if len(authors) > 80:
            # Show first two authors + "et al."
            parts = authors.split(" and ")
            if len(parts) > 2:
                authors = parts[0] + " et al."

        lines.append(f"  [{i}] {p.title}")
        lines.append(f"      {authors} ({p.year})")
        lines.append(f"      {p.journal}")
        if p.doi:
            lines.append(f"      DOI: {p.doi}")
        if p.cited_by > 0:
            lines.append(f"      Cited by: {p.cited_by}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def get_database_info() -> dict[str, int]:
    """Return summary statistics about the loaded literature database.

    Returns
    -------
    A dict with keys ``total_papers``, ``with_keywords``,
    ``year_min``, ``year_max``, ``total_citations``.
    """
    db = _get_database()
    if not db:
        return {"total_papers": 0}

    years = [p.year for p in db if p.year > 0]
    return {
        "total_papers": len(db),
        "with_keywords": sum(1 for p in db if p.keywords),
        "year_min": min(years) if years else 0,
        "year_max": max(years) if years else 0,
        "total_citations": sum(p.cited_by for p in db),
    }
