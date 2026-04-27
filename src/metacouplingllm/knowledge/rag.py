"""
Retrieval-Augmented Generation (RAG) for metacoupling paper evidence.

Loads full-text markdown papers, matches them against the BibTeX literature
database, chunks them into passages, and retrieves relevant passages given
a query (typically derived from an LLM analysis).

Two retrieval backends are available:

- :class:`EmbeddingRetriever` (default): semantic retrieval using
  ``fastembed`` + pre-computed corpus embeddings bundled with the
  package. Catches synonyms, paraphrases, and related concepts.
- :class:`TfIdfIndex` (fallback): lexical retrieval using TF-IDF +
  cosine similarity. Fast, no external dependencies, but misses
  semantic matches. Used when ``fastembed`` is not installed or the
  pre-computed embedding file is missing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metacouplingllm.knowledge.literature import Paper


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundled papers extraction
# ---------------------------------------------------------------------------

_BUNDLED_ZIP = Path(__file__).parent / ".." / "data" / "Papers.zip"


def _extract_bundled_papers(verbose: bool = False) -> Path:
    """Extract the bundled ``data/Papers.zip`` to a cache directory.

    Returns the path to the extracted ``Papers/`` directory.  If the
    files have already been extracted (same zip mtime), the cached
    version is reused.
    """
    import shutil
    import tempfile
    import zipfile

    zip_path = _BUNDLED_ZIP.resolve()
    if not zip_path.exists():
        return Path("__nonexistent__")

    # Use a stable cache directory based on the package location
    cache_root = Path(tempfile.gettempdir()) / "metacoupling_rag_cache"
    papers_dir = cache_root / "Papers"
    stamp_file = cache_root / ".extracted_stamp"

    # Check if already extracted and up-to-date
    zip_mtime = zip_path.stat().st_mtime
    if papers_dir.is_dir() and stamp_file.exists():
        try:
            cached_mtime = float(stamp_file.read_text().strip())
            if cached_mtime == zip_mtime:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    expected_files = {
                        Path(name).name
                        for name in zf.namelist()
                        if name.startswith("Papers/") and name.endswith(".md")
                    }
                extracted_files = {
                    path.name for path in papers_dir.glob("*.md")
                }
                if expected_files == extracted_files:
                    if verbose:
                        print(
                            f"[RAG] Using cached bundled papers: {papers_dir}"
                        )
                    return papers_dir
                if verbose:
                    print("[RAG] Refreshing bundled papers cache: file list changed.")
        except (ValueError, OSError):
            pass

    # Extract
    if verbose:
        print(f"[RAG] Extracting bundled Papers.zip → {cache_root}")

    cache_root.mkdir(parents=True, exist_ok=True)
    if papers_dir.exists():
        shutil.rmtree(papers_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_root)

    stamp_file.write_text(str(zip_mtime))

    if verbose:
        md_count = sum(1 for f in papers_dir.iterdir() if f.suffix == ".md")
        print(f"[RAG] Extracted {md_count} bundled paper files.")

    return papers_dir


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A passage extracted from a full-text paper.

    Attributes
    ----------
    paper_key:
        BibTeX citation key of the matched paper.
    paper_title:
        Title of the paper.
    authors:
        Author string.
    year:
        Publication year.
    section:
        Section heading the passage comes from (e.g., "INTRODUCTION").
    text:
        The actual passage text.
    chunk_index:
        Position of this chunk within the paper (0-indexed).
    """

    paper_key: str = ""
    paper_title: str = ""
    authors: str = ""
    year: int = 0
    section: str = ""
    text: str = ""
    chunk_index: int = 0


@dataclass
class RetrievalResult:
    """A retrieved passage with its relevance score.

    Attributes
    ----------
    chunk:
        The text chunk that was retrieved.
    score:
        Cosine similarity score (0–1).
    """

    chunk: TextChunk
    score: float = 0.0


# ---------------------------------------------------------------------------
# Paper matching: filename → BibTeX entry
# ---------------------------------------------------------------------------

# Pattern for Zotero-style markdown filenames:
#   "Author et al. - YYYY - Title.md"
#   "Author and Author2 - YYYY - Title.md"
_FILENAME_RE = re.compile(
    r"^(?P<authors>.+?)\s*-\s*(?P<year>\d{4})\s*-\s*(?P<title>.+)\.md$",
    re.IGNORECASE,
)


def _normalise_for_match(text: str) -> str:
    """Normalise text for fuzzy matching: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match_paper_to_db(
    filename: str, db_papers: list[Paper],
) -> Paper | None:
    """Match a markdown filename to a BibTeX Paper entry.

    Uses a two-step approach:
    1. Extract year from filename and filter candidates.
    2. Fuzzy-match the title from the filename against candidate titles.

    Returns the best matching Paper, or None if no match found.
    """
    m = _FILENAME_RE.match(filename)
    if not m:
        return None

    file_year = int(m.group("year"))
    file_title = _normalise_for_match(m.group("title"))
    file_authors = _normalise_for_match(m.group("authors"))

    # Filter by year (allow ±1 year tolerance for pre-prints)
    candidates = [p for p in db_papers if abs(p.year - file_year) <= 1]
    if not candidates:
        return None

    # Score each candidate by title word overlap
    best_score = 0.0
    best_paper: Paper | None = None

    file_title_words = set(file_title.split())
    file_author_words = set(file_authors.split())

    for paper in candidates:
        paper_title = _normalise_for_match(paper.title)
        paper_title_words = set(paper_title.split())

        if not paper_title_words:
            continue

        # Jaccard similarity on title words
        intersection = file_title_words & paper_title_words
        union = file_title_words | paper_title_words
        title_score = len(intersection) / len(union) if union else 0.0

        # Bonus for author match
        paper_author_norm = _normalise_for_match(paper.authors)
        paper_author_words = set(paper_author_norm.split())
        author_overlap = len(file_author_words & paper_author_words)
        author_bonus = min(0.2, author_overlap * 0.05)

        score = title_score + author_bonus

        if score > best_score:
            best_score = score
            best_paper = paper

    # Require at least 30% title overlap
    if best_score < 0.3:
        return None

    return best_paper


def _paper_from_filename(filename: str) -> Paper | None:
    """Create fallback paper metadata from a Zotero-style markdown filename."""
    from metacouplingllm.knowledge.literature import Paper

    m = _FILENAME_RE.match(filename)
    if m:
        title = re.sub(r"\s+", " ", m.group("title")).strip()
        authors = re.sub(r"\s+", " ", m.group("authors")).strip()
        try:
            year = int(m.group("year"))
        except ValueError:
            year = 0
    else:
        stem = Path(filename).stem
        year_match = re.search(r"\b((?:18|19|20)\d{2})\b", stem)
        year = int(year_match.group(1)) if year_match else 0
        if " - " in stem:
            authors, title = stem.split(" - ", 1)
        elif "_" in stem:
            authors, title = stem.split("_", 1)
        else:
            authors, title = "Unknown", stem
        authors = re.sub(r"\s+", " ", authors).strip() or "Unknown"
        title = re.sub(r"[_\s]+", " ", title).strip() or stem
        if not title:
            return None
    if not title:
        return None
    if not authors:
        authors = "Unknown"
    if not isinstance(year, int):
        year = 0
    key_source = f"{authors}_{year}_{title}".lower()
    key = re.sub(r"[^a-z0-9]+", "_", key_source).strip("_")[:80]
    return Paper(
        key=key or _normalise_for_match(filename).replace(" ", "_")[:80],
        title=title,
        authors=authors,
        year=year,
        keywords={
            "telecoupling",
            "metacoupling",
            "rag-fallback-metadata",
        },
    )


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

# Section heading patterns in markdown
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_REFERENCE_SECTION_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?"
    r"(references|bibliography|literature cited)\s*$"
)
_BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"^(?:(?:18|19|20)\d{2}|\d{3,})\."
)
_REFERENCE_LEAD_RE = re.compile(
    r"^(?:(?:[A-Z]\.\s*)?(?:18|19|20)\d{2}\.|"
    r"\d{3,}\.\s+[A-Z])"
)
_REFERENCE_AUTHOR_RE = re.compile(
    r"\b[A-Z][A-Za-z'`.-]+,\s*(?:[A-Z]\.\s*){1,3}"
)
_REFERENCE_YEAR_RE = re.compile(r"(?:\(|\b)(?:18|19|20)\d{2}(?:\)|\.)")
_REFERENCE_TERM_RE = re.compile(
    r"\b(?:doi|proceedings|thesis|dissertation|editors?|"
    r"university|press|journal|rome|fao|pages?)\b",
    re.IGNORECASE,
)
_REFERENCE_URL_RE = re.compile(r"https?://|doi[:/]", re.IGNORECASE)


def _truncate_at_references(text: str) -> str:
    """Drop everything from the first references-style section onward."""
    match = _REFERENCE_SECTION_RE.search(text)
    if not match:
        return text
    return text[:match.start()].rstrip()


def _is_reference_heading(heading: str) -> bool:
    """Return True when *heading* looks like bibliography content."""
    heading = heading.strip()
    heading_lower = heading.lower()
    if heading_lower in (
        "references", "bibliography", "literature cited",
        "acknowledgments", "acknowledgements", "appendix",
        "supplementary material", "supporting information",
    ):
        return True
    return bool(_BIBLIOGRAPHY_HEADING_RE.match(heading))


def _looks_like_reference_chunk(text: str) -> bool:
    """Heuristically reject chunks that read like bibliography entries."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False

    year_count = len(_REFERENCE_YEAR_RE.findall(compact))
    author_count = len(_REFERENCE_AUTHOR_RE.findall(compact))
    term_count = len(_REFERENCE_TERM_RE.findall(compact))
    has_url = bool(_REFERENCE_URL_RE.search(compact))
    has_page_ref = bool(re.search(r"\bpages?\s+\d", compact, re.IGNORECASE))
    has_reference_lead = bool(_REFERENCE_LEAD_RE.match(compact))

    if term_count >= 3:
        return True
    if has_reference_lead and (term_count >= 1 or has_page_ref or year_count >= 1):
        return True
    if has_page_ref and term_count >= 2:
        return True
    if author_count >= 2 and (year_count >= 1 or term_count >= 1 or has_page_ref):
        return True
    if year_count >= 2 and (term_count >= 1 or has_url or has_page_ref):
        return True
    return False


def _chunk_markdown(
    text: str,
    paper: Paper,
    max_chunk_words: int = 250,
    overlap_words: int = 50,
) -> list[TextChunk]:
    """Split a markdown document into overlapping text chunks.

    Strategy:
    1. Split by section headings (## or ###).
    2. Within each section, split into ~max_chunk_words chunks with overlap.
    3. Filter out very short chunks (< 30 words) and non-text chunks.
    """
    # Remove image placeholders and citation artifacts
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    text = re.sub(r"\|\s*#\s*\|.*?\n", "", text)  # table headers
    text = _truncate_at_references(text)

    # Split into sections by headings
    sections: list[tuple[str, str]] = []
    parts = _HEADING_RE.split(text)

    # parts alternates: [pre-heading text, heading1, text1, heading2, text2, ...]
    current_heading = "HEADER"
    i = 0
    if len(parts) > 0 and not _HEADING_RE.match(parts[0] if parts[0] else ""):
        sections.append((current_heading, parts[0]))
        i = 1

    while i < len(parts) - 1:
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((heading, body))
        i += 2

    chunks: list[TextChunk] = []
    chunk_idx = 0

    for heading, body in sections:
        if _is_reference_heading(heading):
            continue

        # Clean the body text
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            continue

        words = body.split()
        if len(words) < 30:
            continue

        # Split into chunks with overlap
        start = 0
        while start < len(words):
            end = min(start + max_chunk_words, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if len(chunk_words) >= 30 and not _looks_like_reference_chunk(
                chunk_text
            ):
                chunks.append(TextChunk(
                    paper_key=paper.key,
                    paper_title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                    section=heading,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1

            # Advance with overlap
            if end >= len(words):
                break
            start = end - overlap_words

    return chunks


# ---------------------------------------------------------------------------
# TF-IDF retrieval engine
# ---------------------------------------------------------------------------

# Stopwords for TF-IDF
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
    "are", "been", "being", "has", "had", "have", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "nor", "so", "if", "then", "than", "that", "this",
    "these", "those", "their", "its", "our", "his", "her", "my", "your",
    "we", "they", "he", "she", "you", "i", "me", "us", "them", "him",
    "also", "more", "most", "much", "many", "some", "any", "all", "each",
    "every", "both", "few", "several", "other", "another", "such", "only",
    "just", "very", "too", "even", "still", "already", "about", "over",
    "after", "before", "between", "through", "during", "into", "out",
    "up", "down", "which", "who", "whom", "what", "where", "when", "how",
    "while", "there", "here", "et", "al", "fig", "figure", "table",
    "however", "thus", "therefore", "although", "because", "since",
    "whether", "either", "neither", "per", "via",
})

_WORD_RE = re.compile(r"[a-z][a-z0-9-]+")


def _tokenise(text: str) -> list[str]:
    """Tokenise text into lowercase words, removing stopwords."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


_GENERIC_CITATION_TOKENS = frozenset({
    "agent", "agents", "cause", "causes", "coupled", "coupling",
    "effect", "effects", "flow", "flows", "framework", "human",
    "humannature", "impact", "impacts", "information", "interaction",
    "interactions", "intracoupling", "metacoupling", "nature",
    "pericoupling", "receiving", "sending", "spillover", "system",
    "systems", "telecoupling",
})


def _specific_tokens(tokens: set[str] | list[str]) -> set[str]:
    """Return the non-generic subset of *tokens* for citation matching."""
    return {t for t in tokens if t not in _GENERIC_CITATION_TOKENS}


def _citation_policy(section: str) -> tuple[int, float, int, int]:
    """Return citation thresholds for the current output section.

    Returns
    -------
    ``(min_overlap, min_ratio, min_specific_overlap, max_citations)``
    """
    if section in {"CAUSES", "EFFECTS", "RESEARCH GAPS"}:
        return 4, 0.35, 2, 2
    return 3, 0.20, 1, 3


def _best_excerpt(
    text: str,
    anchor_text: str = "",
    max_chars: int = 300,
) -> str:
    """Extract a short excerpt centered on the most relevant passage."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact

    anchor_tokens = _specific_tokens(set(_tokenise(anchor_text)))
    if not anchor_tokens:
        return compact[:max_chars].rstrip() + "..."

    words = compact.split()
    window_words = max(30, min(60, max_chars // 6))
    step = max(10, window_words // 3)
    best_range = (0, min(len(words), window_words))
    best_score = (-1.0, -1.0, -1.0)

    for start in range(0, len(words), step):
        end = min(len(words), start + window_words)
        window_text = " ".join(words[start:end])
        window_tokens = _specific_tokens(set(_tokenise(window_text)))
        overlap = anchor_tokens & window_tokens
        if not overlap:
            continue

        overlap_ratio = len(overlap) / len(anchor_tokens)
        density = len(overlap) / max(1, len(window_tokens))
        score = (float(len(overlap)), overlap_ratio, density)
        if score > best_score:
            best_score = score
            best_range = (start, end)

    start, end = best_range
    excerpt = " ".join(words[start:end]).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit(" ", 1)[0].rstrip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(words):
        excerpt = excerpt.rstrip() + "..."
    return excerpt


# ---------------------------------------------------------------------------
# Embedding retrieval (default backend)
# ---------------------------------------------------------------------------

# Default embedding model. BGE-base improves semantic recall over BGE-small
# while keeping the model download at runtime rather than bundling it.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_EMBEDDING_DIMS = 768

# Path to the pre-computed corpus embeddings bundled with the package.
_BUNDLED_EMBEDDINGS_PATH = (
    Path(__file__).parent / ".." / "data" / "chunk_embeddings.npy"
)

# Path to the sidecar manifest that records the chunk-order fingerprint the
# .npy was built against. Used to detect silent chunk/embedding mismatches
# that can occur when the .npy is built on one platform and loaded on
# another — see ``compute_chunk_fingerprint`` below for rationale.
_BUNDLED_MANIFEST_PATH = (
    Path(__file__).parent / ".." / "data" / "chunk_embeddings.manifest.json"
)


def compute_chunk_fingerprint(chunks: "list[TextChunk]") -> str:
    """Return a deterministic SHA-256 fingerprint of a chunk list's order.

    The fingerprint is computed over ``paper_key:chunk_index`` pairs
    (joined with ``|``) and is independent of chunk text content. Two
    chunk lists produce the same fingerprint iff they reference the
    same papers in the same order with the same per-paper chunking.

    This is used by :class:`EmbeddingRetriever` to verify that the
    runtime chunks are in the same order as the chunks that were used
    to build ``chunk_embeddings.npy``. A mismatch means the pre-
    computed embeddings at index ``k`` correspond to a different chunk
    than the runtime chunk at the same index — every retrieval result
    would silently return the wrong passage.

    Parameters
    ----------
    chunks:
        Ordered list of :class:`TextChunk`. Normally obtained from
        :meth:`RAGEngine.load`.

    Returns
    -------
    Hex-digest SHA-256 string.
    """
    joined = b"|".join(
        f"{c.paper_key}:{c.chunk_index}".encode("utf-8") for c in chunks
    )
    return hashlib.sha256(joined).hexdigest()


def _load_precomputed_embeddings() -> "np.ndarray | None":
    """Load the pre-computed corpus embeddings shipped with the package.

    Returns ``None`` when the file is missing (e.g., in development
    before the build step has been run) so callers can fall back to
    regenerating or using a different backend.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return None

    if not _BUNDLED_EMBEDDINGS_PATH.exists():
        return None
    try:
        return np.load(_BUNDLED_EMBEDDINGS_PATH)
    except Exception as exc:
        print(f"[RAG] Failed to load pre-computed embeddings: {exc}")
        return None


def _load_chunk_manifest() -> "dict | None":
    """Load the sidecar chunk-order manifest, if it exists.

    Returns ``None`` when the file is missing (legacy installs that
    shipped before the manifest was introduced — the loader warns but
    proceeds) or when the file is malformed.
    """
    if not _BUNDLED_MANIFEST_PATH.exists():
        return None
    try:
        with _BUNDLED_MANIFEST_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Failed to read chunk-embeddings manifest at %s: %s",
            _BUNDLED_MANIFEST_PATH.resolve(), exc,
        )
        return None


class EmbeddingRetriever:
    """Semantic retrieval over pre-computed corpus embeddings.

    Unlike :class:`TfIdfIndex`, this retriever uses dense vector
    embeddings from a pre-trained transformer model (via ``fastembed``
    + ONNX Runtime). Queries and chunks are compared via cosine
    similarity on 768-dimensional vectors, which captures semantic
    similarity — synonyms, paraphrases, and related concepts that
    TF-IDF misses.

    The corpus embeddings are **pre-computed and shipped with the
    package** as a ``.npy`` file (see ``scripts/build_embeddings.py``).
    At runtime the retriever only needs to embed the query, which
    takes ~40 ms on CPU after a one-time ~2 s model load.

    Parameters
    ----------
    chunks:
        List of :class:`TextChunk` objects (must be in the SAME order
        used when the pre-computed embeddings were generated).
    model_name:
        Name of the embedding model. Must match the model used to
        produce the pre-computed corpus embeddings. Defaults to
        ``"BAAI/bge-base-en-v1.5"``.
    precomputed_embeddings:
        Optional pre-computed ``(n_chunks, dims)`` numpy array. When
        ``None``, the retriever tries to load the bundled file. When
        the shape doesn't match ``len(chunks)``, the embeddings are
        discarded and a warning is printed (falling back to re-encoding
        is caller's responsibility).

    Raises
    ------
    ImportError
        If ``fastembed`` is not installed.
    RuntimeError
        If no pre-computed embeddings are available AND
        ``allow_rebuild`` is False (the default).
    """

    def __init__(
        self,
        chunks: "list[TextChunk]",
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        precomputed_embeddings: "np.ndarray | None" = None,
        allow_rebuild: bool = False,
        verbose: bool = False,
    ) -> None:
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "numpy is required for EmbeddingRetriever. "
                "Install with `pip install numpy`."
            ) from exc

        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for EmbeddingRetriever. "
                "Install with `pip install fastembed`, or use the "
                "TF-IDF backend by passing rag_backend='tfidf' to "
                "MetacouplingAssistant."
            ) from exc

        self._chunks = chunks
        self._n_docs = len(chunks)
        self._model_name = model_name
        self._verbose = verbose
        self._np = np
        self._TextEmbedding = TextEmbedding

        # Lazily initialised embedder (first query triggers download +
        # model load). Keeping this lazy avoids a ~2 s startup cost
        # when the retriever is constructed but never used.
        self._embedder: object | None = None

        # Load or validate the corpus embeddings.
        embeddings = precomputed_embeddings
        if embeddings is None:
            embeddings = _load_precomputed_embeddings()

        if embeddings is None:
            if not allow_rebuild:
                raise RuntimeError(
                    "No pre-computed embeddings found at "
                    f"{_BUNDLED_EMBEDDINGS_PATH.resolve()}. Run "
                    "`python scripts/build_embeddings.py` to generate "
                    "the file, or pass allow_rebuild=True to build it "
                    "at runtime (~60 s on CPU)."
                )
            embeddings = self._build_corpus_embeddings()

        if embeddings.shape[0] != self._n_docs:
            raise RuntimeError(
                f"Pre-computed embedding count ({embeddings.shape[0]}) "
                f"does not match chunk count ({self._n_docs}). The "
                "embeddings file is stale — re-run "
                "`python scripts/build_embeddings.py`."
            )

        # Verify chunk-order fingerprint against the bundled manifest.
        # The shape check above only guarantees counts match. We ALSO
        # need to verify the chunks are in the SAME ORDER they were in
        # when the .npy was built — otherwise embedding [k] points to
        # the wrong chunk. This matters because Python's ``Path.__lt__``
        # sorts case-INsensitively on Windows and case-sensitively on
        # Linux/macOS, so building the .npy on one OS and loading it on
        # another can silently corrupt the mapping whenever the corpus
        # has mixed-case filenames.
        #
        # Only run the manifest check when the BUNDLED embeddings were
        # loaded (precomputed_embeddings was None on entry). When a
        # caller passes their own embeddings, the bundled manifest
        # doesn't apply — it's the caller's responsibility to ensure
        # the order matches their custom embeddings.
        using_bundled = precomputed_embeddings is None
        manifest = _load_chunk_manifest() if using_bundled else None
        if manifest is not None:
            stored_hash = manifest.get("paper_key_order_sha256", "")
            runtime_hash = compute_chunk_fingerprint(chunks)
            if stored_hash and stored_hash != runtime_hash:
                raise RuntimeError(
                    "Chunk-order fingerprint mismatch between the "
                    "pre-computed embeddings and the chunks loaded at "
                    "runtime. The chunks in chunk_embeddings.npy were "
                    "built in a different order than the one produced "
                    "by RAGEngine.load() on this machine — every "
                    "retrieval result would silently return the wrong "
                    "passage. This usually happens when the package "
                    "was built on one OS and installed on another, or "
                    "when Papers.zip was updated without rebuilding "
                    "the embeddings.\n\n"
                    "Fix: run `python scripts/build_embeddings.py "
                    "--force` on this machine to regenerate the .npy "
                    "and manifest, or install a newer package version.\n\n"
                    f"  expected (manifest): {stored_hash[:16]}...\n"
                    f"  actual   (runtime):  {runtime_hash[:16]}..."
                )
        elif self._verbose:
            # Legacy install without a manifest — warn but proceed.
            logger.warning(
                "No chunk-embeddings manifest found at %s. Cannot "
                "verify chunk-order integrity; retrieval may silently "
                "return wrong passages if the .npy was built on a "
                "different platform. Consider re-running "
                "`python scripts/build_embeddings.py --force`.",
                _BUNDLED_MANIFEST_PATH.resolve(),
            )

        # L2-normalize for cosine similarity via dot product. Bundled
        # embeddings may be stored as float16 to keep the wheel small;
        # scoring is always done in float32.
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._chunk_vecs = (embeddings / norms).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_embedder(self) -> object:
        """Lazily initialise the fastembed encoder."""
        if self._embedder is None:
            if self._verbose:
                print(
                    f"[RAG] Loading embedding model "
                    f"{self._model_name} (first use may take ~10 s "
                    "while the ONNX model is downloaded/cached)..."
                )
            self._embedder = self._TextEmbedding(
                model_name=self._model_name,
            )
        return self._embedder

    def _build_corpus_embeddings(self) -> "np.ndarray":
        """Encode the chunks with the active model (runtime fallback).

        Only called when the pre-computed file is missing AND the user
        opts in via ``allow_rebuild=True``. Prefer pre-computing once
        and shipping the .npy file with the package.
        """
        embedder = self._ensure_embedder()
        texts = [chunk.text for chunk in self._chunks]
        if self._verbose:
            print(
                f"[RAG] Encoding {len(texts)} chunks with "
                f"{self._model_name} (this takes ~60 s on CPU)..."
            )
        vectors = self._np.array(
            list(embedder.embed(texts)),
            dtype=self._np.float32,
        )
        return vectors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.3,
        max_chunks_per_paper: int = 3,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for a query.

        Parameters
        ----------
        query_text:
            The search query text.
        top_k:
            Maximum number of results (chunks) to return. Note: this
            counts individual chunks, not unique papers — multiple
            chunks from the same paper may appear when that paper's
            most relevant content is spread across different sections.
        min_score:
            Minimum cosine similarity threshold. BGE-small cosine
            similarity typically ranges 0.3–0.9 for relevant results,
            so this default is stricter than the TF-IDF default.
        max_chunks_per_paper:
            Maximum number of chunks from the same paper that can
            appear in the result set. Default ``3`` lets strongly-
            matching papers contribute several high-scoring chunks
            rather than a single chunk that may not cover all of the
            paper's relevant content. Set to ``1`` for the legacy
            one-chunk-per-paper behavior, or raise to ``5+`` when a
            single paper is expected to be highly relevant and its
            evidence is spread across many sections.

        Returns
        -------
        List of :class:`RetrievalResult` sorted by score descending,
        with at most ``max_chunks_per_paper`` chunks from any single
        paper.
        """
        if not self._chunks:
            return []
        if not query_text or not query_text.strip():
            return []
        if max_chunks_per_paper < 1:
            max_chunks_per_paper = 1

        embedder = self._ensure_embedder()
        query_vec = next(iter(embedder.embed([query_text])))
        query_vec = self._np.asarray(query_vec, dtype=self._np.float32)
        q_norm = float(self._np.linalg.norm(query_vec))
        if q_norm == 0.0:
            return []
        query_vec = query_vec / q_norm

        # Cosine similarity = dot product of L2-normalized vectors.
        scores = self._chunk_vecs @ query_vec  # shape (n_chunks,)

        # Two-pass selection:
        #   Pass 1 prefers chunks from sections we have NOT yet seen
        #   for a given paper — this buys section diversity first
        #   (Introduction + Results + Discussion when all score highly).
        #   Chunks from sections we've already seen get deferred.
        #   Pass 2 fills any remaining slots (per-paper cap and
        #   top_k) from the deferred list — so long sections that
        #   the chunker split into several sub-topic chunks (e.g.,
        #   "4. Results" with inbound vs outbound sub-parts) can
        #   still contribute multiple chunks when the paper has
        #   budget to spare.
        order = self._np.argsort(-scores)
        deduped: list[RetrievalResult] = []
        paper_counts: dict[str, int] = {}
        paper_sections_seen: dict[str, set[str]] = {}
        deferred: list[tuple[int, float]] = []
        for idx in order:
            score = float(scores[idx])
            if score < min_score:
                break
            chunk = self._chunks[int(idx)]
            paper_key = chunk.paper_key
            if paper_counts.get(paper_key, 0) >= max_chunks_per_paper:
                continue
            sections = paper_sections_seen.setdefault(paper_key, set())
            if chunk.section in sections:
                deferred.append((int(idx), score))
                continue
            sections.add(chunk.section)
            paper_counts[paper_key] = paper_counts.get(paper_key, 0) + 1
            deduped.append(RetrievalResult(chunk=chunk, score=score))
            if len(deduped) >= top_k:
                return deduped
        # Pass 2 — use remaining per-paper budget on deferred chunks,
        # in score-desc order (``deferred`` is already score-desc
        # because ``order`` was).
        for idx, score in deferred:
            chunk = self._chunks[int(idx)]
            paper_key = chunk.paper_key
            if paper_counts.get(paper_key, 0) >= max_chunks_per_paper:
                continue
            paper_counts[paper_key] = paper_counts.get(paper_key, 0) + 1
            deduped.append(RetrievalResult(chunk=chunk, score=score))
            if len(deduped) >= top_k:
                break
        return deduped


class TfIdfIndex:
    """In-memory TF-IDF index for text chunks.

    Parameters
    ----------
    chunks:
        List of :class:`TextChunk` objects to index.
    """

    def __init__(self, chunks: list[TextChunk]) -> None:
        self._chunks = chunks
        self._n_docs = len(chunks)
        self._vocab: dict[str, int] = {}  # word → doc-frequency
        self._tf_vectors: list[dict[str, float]] = []

        self._build_index()

    def _build_index(self) -> None:
        """Compute TF and DF for all chunks."""
        doc_freq: Counter[str] = Counter()
        tf_raw: list[Counter[str]] = []

        for chunk in self._chunks:
            tokens = _tokenise(chunk.text)
            tf = Counter(tokens)
            tf_raw.append(tf)
            # Document frequency: count each term once per document
            doc_freq.update(tf.keys())

        self._vocab = dict(doc_freq)

        # Normalise TF to TF log-normalised: 1 + log(tf)
        for tf in tf_raw:
            total = sum(tf.values()) or 1
            normed: dict[str, float] = {}
            for word, count in tf.items():
                normed[word] = (1 + math.log(count)) / total
            self._tf_vectors.append(normed)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.01,
        max_chunks_per_paper: int = 3,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for a query.

        Parameters
        ----------
        query_text:
            The search query text.
        top_k:
            Maximum number of results (chunks) to return. Note: this
            counts individual chunks, not unique papers.
        min_score:
            Minimum cosine similarity threshold.
        max_chunks_per_paper:
            Maximum number of chunks from the same paper that can
            appear in the result set. Default ``3``. Set to ``1``
            for the legacy one-chunk-per-paper behavior.

        Returns
        -------
        List of :class:`RetrievalResult` sorted by score descending,
        with at most ``max_chunks_per_paper`` chunks per paper.
        """
        if not self._chunks:
            return []
        if max_chunks_per_paper < 1:
            max_chunks_per_paper = 1

        query_tokens = _tokenise(query_text)
        if not query_tokens:
            return []

        # Build query TF-IDF vector
        query_tf = Counter(query_tokens)
        query_vec: dict[str, float] = {}
        for word, count in query_tf.items():
            df = self._vocab.get(word, 0)
            if df == 0:
                continue
            idf = math.log((self._n_docs + 1) / (df + 1)) + 1
            query_vec[word] = (1 + math.log(count)) * idf

        if not query_vec:
            return []

        # Compute query vector magnitude
        q_mag = math.sqrt(sum(v * v for v in query_vec.values()))
        if q_mag == 0:
            return []

        # Score each document
        results: list[tuple[float, int]] = []
        for idx, doc_tf in enumerate(self._tf_vectors):
            dot = 0.0
            for word, q_val in query_vec.items():
                d_val = doc_tf.get(word, 0.0)
                if d_val > 0:
                    df = self._vocab.get(word, 0)
                    idf = math.log((self._n_docs + 1) / (df + 1)) + 1
                    dot += q_val * d_val * idf

            # Document magnitude (approximation — use cached if needed)
            d_mag = math.sqrt(
                sum(
                    (v * (math.log((self._n_docs + 1) / (self._vocab.get(w, 1) + 1)) + 1)) ** 2
                    for w, v in doc_tf.items()
                )
            )
            if d_mag == 0:
                continue

            cosine = dot / (q_mag * d_mag)
            if cosine >= min_score:
                results.append((cosine, idx))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        # Two-pass selection (see EmbeddingRetriever.query() for the
        # rationale): Pass 1 prefers chunks from distinct sections per
        # paper; Pass 2 fills remaining per-paper budget with deferred
        # chunks from sections we already took from, so long sections
        # split into several sub-topic chunks can still contribute
        # multiple chunks when budget allows.
        paper_counts: dict[str, int] = {}
        paper_sections_seen: dict[str, set[str]] = {}
        deferred: list[tuple[float, int]] = []
        deduped: list[RetrievalResult] = []
        for score, idx in results:
            chunk = self._chunks[idx]
            paper_key = chunk.paper_key
            if paper_counts.get(paper_key, 0) >= max_chunks_per_paper:
                continue
            sections = paper_sections_seen.setdefault(paper_key, set())
            if chunk.section in sections:
                deferred.append((score, idx))
                continue
            sections.add(chunk.section)
            paper_counts[paper_key] = paper_counts.get(paper_key, 0) + 1
            deduped.append(RetrievalResult(chunk=chunk, score=score))
            if len(deduped) >= top_k:
                return deduped
        for score, idx in deferred:
            chunk = self._chunks[idx]
            paper_key = chunk.paper_key
            if paper_counts.get(paper_key, 0) >= max_chunks_per_paper:
                continue
            paper_counts[paper_key] = paper_counts.get(paper_key, 0) + 1
            deduped.append(RetrievalResult(chunk=chunk, score=score))
            if len(deduped) >= top_k:
                break

        return deduped


# ---------------------------------------------------------------------------
# RAG Engine — high-level API
# ---------------------------------------------------------------------------


class RAGEngine:
    """Retrieval-Augmented Generation engine for metacoupling papers.

    Loads full-text papers from a directory, matches them against the
    BibTeX literature database, chunks them, and builds a retrieval
    index.

    Two retrieval backends are available:

    - ``"embeddings"`` (default): semantic retrieval using fastembed +
      pre-computed corpus embeddings bundled with the package. Catches
      synonyms, paraphrases, and related concepts.
    - ``"tfidf"``: lexical retrieval using TF-IDF + cosine similarity.
      Fallback when fastembed is unavailable.
    - ``"auto"``: try embeddings first, fall back to TF-IDF on failure.

    Parameters
    ----------
    papers_dir:
        Path to directory containing full-text markdown papers.
    max_chunk_words:
        Maximum words per text chunk.
    overlap_words:
        Number of overlapping words between consecutive chunks.
    verbose:
        Print progress information during loading.
    backend:
        Retrieval backend: ``"embeddings"`` (default), ``"tfidf"``,
        or ``"auto"``. ``"auto"`` tries embeddings first and
        transparently falls back to TF-IDF if fastembed or the
        pre-computed embedding file are missing.
    """

    def __init__(
        self,
        papers_dir: str | Path,
        max_chunk_words: int = 250,
        overlap_words: int = 50,
        verbose: bool = False,
        backend: str = "auto",
    ) -> None:
        self._papers_dir = Path(papers_dir)
        self._max_chunk_words = max_chunk_words
        self._overlap_words = overlap_words
        self._verbose = verbose
        if backend not in ("embeddings", "tfidf", "auto"):
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Expected 'embeddings', 'tfidf', or 'auto'."
            )
        self._backend = backend
        self._active_backend: str | None = None
        self._index: TfIdfIndex | EmbeddingRetriever | None = None
        self._matched_count: int = 0
        self._total_chunks: int = 0
        self._total_files: int = 0

    @property
    def is_loaded(self) -> bool:
        """Return True if the index has been built."""
        return self._index is not None

    @property
    def backend(self) -> str | None:
        """Return the name of the active retrieval backend.

        One of ``"embeddings"``, ``"tfidf"``, or ``None`` if
        :meth:`load` has not been called yet.
        """
        return self._active_backend

    @property
    def matched_papers(self) -> int:
        """Number of papers matched to the literature database."""
        return self._matched_count

    @property
    def total_chunks(self) -> int:
        """Total number of text chunks in the index."""
        return self._total_chunks

    @property
    def total_files(self) -> int:
        """Total number of markdown files found."""
        return self._total_files

    def load(self) -> None:
        """Load papers, match to database, chunk, and build index.

        This should be called once before querying.  It may take a few
        seconds depending on the number of papers.

        If the configured ``papers_dir`` does not exist, the engine
        automatically falls back to the bundled ``data/Papers.zip``
        shipped with the package.
        """
        from metacouplingllm.knowledge.literature import _get_database

        # Validate papers directory exists; fall back to bundled zip
        if not self._papers_dir.exists() or not self._papers_dir.is_dir():
            self._papers_dir = _extract_bundled_papers(self._verbose)

        if not self._papers_dir.exists() or not self._papers_dir.is_dir():
            msg = (
                f"[RAG] Papers directory does not exist: "
                f"{self._papers_dir.resolve()}"
            )
            print(msg)
            self._index = TfIdfIndex([])
            return

        db_papers = _get_database()
        if not db_papers:
            print("[RAG] WARNING: No literature database loaded. "
                  "RAG will not produce evidence.")
            self._index = TfIdfIndex([])
            return

        # Collect markdown files.
        #
        # IMPORTANT: sort by the filename STRING, not by the Path object.
        # Python's ``Path.__lt__`` is platform-dependent: on Windows it
        # uses case-INsensitive comparison (via ``_str_normcase``), but
        # on Linux/macOS it uses case-sensitive Unicode codepoint order.
        # Sorting ``Path`` objects directly would produce a different
        # chunk order on Windows vs Linux for any corpus that contains
        # filenames with mixed leading case (e.g., "da Silva..." vs
        # "Dandan..."). That in turn corrupts the chunks-to-embeddings
        # mapping for the pre-computed ``chunk_embeddings.npy`` file
        # whenever it's built on one OS and loaded on another. Using
        # ``key=lambda p: p.name`` forces pure-string (case-sensitive)
        # comparison on every platform, so the chunk order is
        # deterministic across OSes. The manifest integrity check in
        # ``EmbeddingRetriever.__init__`` catches mismatches if this
        # ever drifts again.
        md_files = sorted(
            self._papers_dir.glob("*.md"), key=lambda p: p.name,
        )
        self._total_files = len(md_files)

        if self._total_files == 0:
            print(
                f"[RAG] WARNING: No .md files found in "
                f"{self._papers_dir.resolve()}"
            )
            self._index = TfIdfIndex([])
            return

        if self._verbose:
            print(f"[RAG] Found {self._total_files} markdown files.")

        all_chunks: list[TextChunk] = []
        matched = 0
        fallback_metadata = 0

        for md_path in md_files:
            # Match to database
            paper = _match_paper_to_db(md_path.name, db_papers)
            if paper is None:
                paper = _paper_from_filename(md_path.name)
                if paper is None:
                    continue
                fallback_metadata += 1
            else:
                matched += 1

            # Read and chunk the file
            try:
                text = md_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            chunks = _chunk_markdown(
                text, paper,
                max_chunk_words=self._max_chunk_words,
                overlap_words=self._overlap_words,
            )
            all_chunks.extend(chunks)

        self._matched_count = matched
        self._total_chunks = len(all_chunks)

        indexed_papers = matched + fallback_metadata

        # Always print summary so user knows RAG loaded. Report indexed
        # papers first; BibTeX matching is metadata quality, not corpus
        # coverage.
        print(
            f"[RAG] Indexed {indexed_papers}/{self._total_files} papers "
            f"-> {len(all_chunks)} chunks."
        )
        if matched == 0 and fallback_metadata == 0:
            print(
                "[RAG] WARNING: No papers matched. Ensure filenames follow "
                "Zotero format: 'Author - YYYY - Title.md'"
            )

        self._index = self._build_retriever(all_chunks)

    def _build_retriever(
        self, chunks: list[TextChunk],
    ) -> "TfIdfIndex | EmbeddingRetriever":
        """Build the configured retriever for the given chunks.

        Handles the embeddings → TF-IDF fallback logic when
        ``backend="auto"``. Prints the selected backend so users
        know which retriever is active.
        """
        if self._backend == "tfidf":
            self._active_backend = "tfidf"
            if self._verbose:
                print("[RAG] Using TF-IDF retrieval backend.")
            return TfIdfIndex(chunks)

        # "embeddings" or "auto"
        try:
            retriever = EmbeddingRetriever(
                chunks,
                verbose=self._verbose,
            )
            self._active_backend = "embeddings"
            print(
                f"[RAG] Using embedding retrieval backend "
                f"({DEFAULT_EMBEDDING_MODEL})."
            )
            return retriever
        except (ImportError, RuntimeError) as exc:
            if self._backend == "embeddings":
                # User explicitly asked for embeddings — surface the
                # error loudly but still fall back to keep the
                # package usable.
                print(
                    f"[RAG] WARNING: embedding backend unavailable: "
                    f"{exc}. Falling back to TF-IDF."
                )
            elif self._verbose:
                print(
                    f"[RAG] Embedding backend unavailable ({exc}). "
                    "Using TF-IDF fallback."
                )
            self._active_backend = "tfidf"
            return TfIdfIndex(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        max_chunks_per_paper: int = 3,
    ) -> list[RetrievalResult]:
        """Retrieve relevant passages for a query string.

        Parameters
        ----------
        query:
            Free-text query (e.g., research description or analysis text).
        top_k:
            Maximum number of passages (chunks) to return. Multiple
            chunks may come from the same paper when that paper's
            most relevant content is spread across different sections.
        min_score:
            Minimum similarity threshold. When ``None`` (default), a
            backend-appropriate value is used: 0.01 for TF-IDF,
            0.3 for embedding cosine similarity.
        max_chunks_per_paper:
            Cap on how many chunks from the same paper may appear in
            the result set (default ``3``). Set to ``1`` for the
            legacy one-chunk-per-paper behavior. Useful when a single
            paper is highly relevant but its key evidence is scattered
            across Introduction / Results / Discussion sections that
            a single chunk can't cover on its own.

        Returns
        -------
        List of :class:`RetrievalResult` sorted by relevance.

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called.
        """
        if self._index is None:
            raise RuntimeError("RAG engine not loaded. Call load() first.")
        if min_score is None:
            min_score = (
                0.3 if self._active_backend == "embeddings" else 0.01
            )
        return self._index.query(
            query,
            top_k=top_k,
            min_score=min_score,
            max_chunks_per_paper=max_chunks_per_paper,
        )

    def retrieve_for_analysis(
        self,
        parsed: object,  # ParsedAnalysis, but avoid circular import
        top_k: int = 5,
        min_score: float | None = None,
        max_chunks_per_paper: int = 3,
    ) -> list[RetrievalResult]:
        """Retrieve passages relevant to a parsed LLM analysis.

        Extracts query terms from the parsed analysis and retrieves
        matching passages from the full-text paper index.

        Parameters
        ----------
        parsed:
            A :class:`~metacouplingllm.llm.parser.ParsedAnalysis` object.
        top_k:
            Maximum number of results.
        min_score:
            Minimum similarity score.
        max_chunks_per_paper:
            See :meth:`retrieve`. Default ``3``.

        Returns
        -------
        List of :class:`RetrievalResult`.
        """
        query = _build_query_from_analysis(parsed)
        if not query:
            return []
        return self.retrieve(
            query,
            top_k=top_k,
            min_score=min_score,
            max_chunks_per_paper=max_chunks_per_paper,
        )


# ---------------------------------------------------------------------------
# Query builder from ParsedAnalysis
# ---------------------------------------------------------------------------


def _build_query_from_analysis(parsed: object) -> str:
    """Build a search query from a ParsedAnalysis object.

    Extracts key terms from classification, systems, flows, and causes.
    """
    parts: list[str] = []

    # Coupling classification
    cc = getattr(parsed, "coupling_classification", "")
    if cc:
        parts.append(cc)

    # Systems
    systems = getattr(parsed, "systems", {})
    for role in ("sending", "receiving", "spillover"):
        val = systems.get(role, "")
        if isinstance(val, dict):
            for key in ("name", "geographic_scope", "description"):
                v = val.get(key, "")
                if v:
                    parts.append(v)
        elif isinstance(val, str) and val:
            parts.append(val)

    # Flows
    flows = getattr(parsed, "flows", [])
    for flow in flows:
        if isinstance(flow, dict):
            desc = flow.get("description", "")
            if desc:
                parts.append(desc)
            cat = flow.get("category", "")
            if cat:
                parts.append(cat)
        elif isinstance(flow, str) and flow:
            parts.append(flow)

    # Causes — can be a list of strings, a dict of lists, or a dict of strings
    causes = getattr(parsed, "causes", [])
    if isinstance(causes, dict):
        for key, val in causes.items():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item:
                        parts.append(item)
            elif isinstance(val, str) and val:
                parts.append(val)
    elif isinstance(causes, list):
        for cat in causes:
            if isinstance(cat, str) and cat:
                parts.append(cat)
            elif isinstance(cat, dict):
                desc = cat.get("description", "")
                if desc:
                    parts.append(desc)

    # Effects — can be a list of strings, a dict of lists, or a dict of strings
    effects = getattr(parsed, "effects", [])
    if isinstance(effects, dict):
        for key, val in effects.items():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item:
                        parts.append(item)
            elif isinstance(val, str) and val:
                parts.append(val)
    elif isinstance(effects, list):
        for eff in effects:
            if isinstance(eff, dict):
                desc = eff.get("description", "")
                if desc:
                    parts.append(desc)
            elif isinstance(eff, str) and eff:
                parts.append(eff)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_evidence(
    results: list[RetrievalResult],
    anchor_text: str = "",
    backend: str = "tfidf",
) -> str:
    """Format retrieved passages as a human-readable evidence block.

    Parameters
    ----------
    results:
        List of :class:`RetrievalResult` from retrieval.
    anchor_text:
        Optional query or analysis text used to choose a more relevant
        excerpt from each evidence chunk.
    backend:
        Retrieval backend that produced the scores (``"tfidf"`` or
        ``"embeddings"``). Used to pick appropriate High/Medium/Low
        thresholds since the two backends have very different score
        ranges.

    Returns
    -------
    A formatted string ready for display.
    """
    if not results:
        return "No supporting evidence found in the full-text paper database."

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  SUPPORTING EVIDENCE FROM LITERATURE")
    lines.append("=" * 70)
    lines.append("")

    for i, r in enumerate(results, 1):
        chunk = r.chunk
        # Shorten author list
        authors = chunk.authors
        if len(authors) > 60:
            parts = authors.split(" and ")
            if len(parts) > 2:
                authors = parts[0] + " et al."

        confidence = _score_to_confidence(r.score, backend=backend)

        lines.append(f"  [{i}] {chunk.paper_title}")
        lines.append(f"      {authors} ({chunk.year})")
        lines.append(f"      Section: {chunk.section}")
        lines.append(f"      Confidence: {confidence} (score: {r.score:.3f})")
        lines.append(f"      ---")

        # Show a short excerpt from the most relevant part of the chunk.
        excerpt = _best_excerpt(chunk.text, anchor_text=anchor_text)
        lines.append(f"      \"{excerpt}\"")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def annotate_citations(
    formatted: str,
    results: list[RetrievalResult],
    min_keyword_overlap: int = 3,
    min_overlap_ratio: float = 0.20,
) -> str:
    """Add inline citation numbers ``[N]`` to formatted analysis lines.

    For each content line in the formatted output, checks keyword overlap
    with each RAG evidence passage.  If a sufficient number of distinctive
    keywords match, appends ``[N]`` citation tags to the line.

    Parameters
    ----------
    formatted:
        The formatted analysis text (from :class:`AnalysisFormatter`).
    results:
        List of :class:`RetrievalResult` from RAG retrieval.
    min_keyword_overlap:
        Minimum number of shared keywords between a line and an evidence
        passage for a citation to be added.
    min_overlap_ratio:
        Minimum fraction of the line's keywords that must appear in the
        evidence passage.

    Returns
    -------
    The formatted text with ``[N]`` citations appended to matching lines.
    """
    if not results:
        return formatted

    # Build keyword sets for each evidence passage
    evidence_keywords: list[set[str]] = []
    evidence_specific_keywords: list[set[str]] = []
    for r in results:
        tokens = set(_tokenise(r.chunk.text))
        evidence_keywords.append(tokens)
        evidence_specific_keywords.append(_specific_tokens(tokens))

    # Section headers should not be annotated
    # Note: separator-only lines (all "=" or "-") are caught separately below
    _skip_prefixes = (
        "COUPLING CLASSIFICATION", "SYSTEMS IDENTIFICATION",
        "FLOWS ANALYSIS", "AGENTS", "CAUSES", "EFFECTS",
        "RESEARCH GAPS", "PERICOUPLING DATABASE",
        "METACOUPLING FRAMEWORK", "SUPPORTING EVIDENCE",
        "RECOMMENDED LITERATURE", "Note:",
    )
    _section_headers = (
        "COUPLING CLASSIFICATION", "SYSTEMS IDENTIFICATION",
        "FLOWS ANALYSIS", "AGENTS", "CAUSES", "EFFECTS",
        "RESEARCH GAPS",
    )

    lines = formatted.split("\n")
    annotated: list[str] = []

    # Track which section we're in to avoid annotating non-analysis content
    in_evidence_block = False
    current_section = ""

    for line in lines:
        stripped = line.strip()

        matched_header = next(
            (header for header in _section_headers if stripped.startswith(header)),
            "",
        )
        if matched_header:
            current_section = matched_header

        # Detect evidence block boundaries — never annotate evidence itself
        if "SUPPORTING EVIDENCE FROM LITERATURE" in stripped:
            in_evidence_block = True
        if in_evidence_block:
            annotated.append(line)
            continue

        # Skip empty lines, separators, and section headers
        if (
            not stripped
            or stripped.startswith(tuple(_skip_prefixes))
            or set(stripped) <= {"=", "-", " "}
        ):
            annotated.append(line)
            continue

        # Skip sub-headings like "  [Sending]", "  Ecological:", etc.
        if stripped.startswith("[") and stripped.endswith("]"):
            annotated.append(line)
            continue
        # Skip category headings like "Socioeconomic:" or "  **Ecological**"
        if stripped.endswith(":") and len(stripped.split()) <= 3:
            annotated.append(line)
            continue

        # Tokenize the line content
        line_tokens = set(_tokenise(stripped))
        if len(line_tokens) < 3:
            annotated.append(line)
            continue
        line_specific_tokens = _specific_tokens(line_tokens)
        section_min_overlap, section_min_ratio, min_specific_overlap, max_citations = (
            _citation_policy(current_section)
        )
        min_overlap = max(min_keyword_overlap, section_min_overlap)
        min_ratio = max(min_overlap_ratio, section_min_ratio)

        # Find matching evidence passages
        candidates: list[tuple[tuple[float, float, float, float], int]] = []
        for idx, ev_kw in enumerate(evidence_keywords):
            overlap = line_tokens & ev_kw
            n_overlap = len(overlap)
            specific_overlap = line_specific_tokens & evidence_specific_keywords[idx]
            effective_overlap = (
                len(specific_overlap) if line_specific_tokens else n_overlap
            )
            effective_base = (
                len(line_specific_tokens) if line_specific_tokens else len(line_tokens)
            )
            ratio = effective_overlap / effective_base if effective_base else 0

            if n_overlap < min_overlap or ratio < min_ratio:
                continue
            if line_specific_tokens and len(specific_overlap) < min_specific_overlap:
                continue

            score = (
                float(len(specific_overlap)),
                ratio,
                float(n_overlap),
                results[idx].score,
            )
            candidates.append((score, idx + 1))  # 1-indexed

        if candidates:
            candidates.sort(reverse=True)
            citations = sorted({
                citation
                for _, citation in candidates[:max_citations]
            })

            existing = {
                int(match.group(1))
                for match in re.finditer(r"\[(\d+)\]", line)
            }
            new_citations = [c for c in citations if c not in existing]
            if not new_citations:
                annotated.append(line)
                continue

            cite_str = " " + " ".join(f"[{n}]" for n in new_citations)
            annotated.append(f"{line}{cite_str}")
        else:
            annotated.append(line)

    return "\n".join(annotated)


def _score_to_confidence(score: float, backend: str = "tfidf") -> str:
    """Convert a similarity score to a human-readable confidence label.

    Parameters
    ----------
    score:
        Cosine similarity score.
    backend:
        Either ``"tfidf"`` or ``"embeddings"``. The two backends
        produce scores in very different ranges, so we use separate
        threshold sets:

        - TF-IDF: typical range 0.01–0.4 for relevant chunks.
          Thresholds: 0.15 / 0.08 / 0.03.
        - Embeddings (BGE-small cosine): typical range 0.5–0.9 for
          relevant chunks. Thresholds: 0.7 / 0.6 / 0.5.
    """
    if backend == "embeddings":
        if score >= 0.70:
            return "High"
        elif score >= 0.60:
            return "Medium"
        elif score >= 0.50:
            return "Low"
        else:
            return "Very Low"
    # TF-IDF (default)
    if score >= 0.15:
        return "High"
    elif score >= 0.08:
        return "Medium"
    elif score >= 0.03:
        return "Low"
    else:
        return "Very Low"
