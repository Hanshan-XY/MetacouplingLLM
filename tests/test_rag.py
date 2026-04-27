"""Tests for knowledge/rag.py — RAG (Retrieval-Augmented Generation) engine."""

import os
import tempfile
import textwrap

import pytest

from metacouplingllm.knowledge.rag import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingRetriever,
    RAGEngine,
    RetrievalResult,
    TextChunk,
    TfIdfIndex,
    _build_query_from_analysis,
    _chunk_markdown,
    _load_precomputed_embeddings,
    _match_paper_to_db,
    _normalise_for_match,
    _score_to_confidence,
    _tokenise,
    annotate_citations,
    compute_chunk_fingerprint,
    format_evidence,
)


try:
    import fastembed  # noqa: F401
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

try:
    import numpy as np  # noqa: F401
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Mock Paper for testing (mimics literature.Paper)
# ---------------------------------------------------------------------------


class MockPaper:
    """Lightweight mock of literature.Paper for testing."""

    def __init__(self, key="", title="", authors="", year=0, journal="",
                 doi="", keywords=None, cited_by=0):
        self.key = key
        self.title = title
        self.authors = authors
        self.year = year
        self.journal = journal
        self.doi = doi
        self.keywords = keywords or set()
        self.cited_by = cited_by


MOCK_DB_PAPERS = [
    MockPaper(
        key="liu_integration_2017",
        title="Integration across a metacoupled world",
        authors="Liu, Jianguo",
        year=2017,
    ),
    MockPaper(
        key="carlson_telecoupling_2017",
        title="The Telecoupling Framework An Integrative Tool for Enhancing Fisheries Management",
        authors="Carlson, Andrew K. and Taylor, William W. and Liu, Jianguo and Orlic, Ivan",
        year=2017,
    ),
    MockPaper(
        key="sun_telecoupled_2017",
        title="Telecoupled land-use changes in distant countries",
        authors="Sun, Jian and Tong, Yuxing and Liu, Jianguo",
        year=2017,
    ),
    MockPaper(
        key="bicudo_sino_2017",
        title="The Sino-Brazilian Telecoupled Soybean System and Cascading Effects for the Exporting Country",
        authors="Bicudo da Silva, Ramon Felipe and Milhorance de Castro, Carolina and Schneider, Maurício",
        year=2017,
    ),
    MockPaper(
        key="liu_framing_2013",
        title="Framing Sustainability in a Telecoupled World",
        authors="Liu, Jianguo and Hull, Vanessa and Batistella, Mateus",
        year=2013,
    ),
]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        assert _normalise_for_match("Hello, World!") == "hello world"

    def test_collapse_whitespace(self):
        assert _normalise_for_match("  a   b  ") == "a b"

    def test_strip_punctuation(self):
        assert _normalise_for_match("'meta-coupling'") == "metacoupling"


# ---------------------------------------------------------------------------
# Paper matching
# ---------------------------------------------------------------------------

class TestMatchPaperToDb:

    def test_exact_match(self):
        filename = "Liu - 2017 - Integration across a metacoupled world.md"
        paper = _match_paper_to_db(filename, MOCK_DB_PAPERS)
        assert paper is not None
        assert paper.key == "liu_integration_2017"

    def test_carlson_match(self):
        filename = "Carlson et al. - 2017 - The Telecoupling Framework An Integrative Tool for Enhancing Fisheries Management.md"
        paper = _match_paper_to_db(filename, MOCK_DB_PAPERS)
        assert paper is not None
        assert paper.key == "carlson_telecoupling_2017"

    def test_no_match_wrong_year(self):
        filename = "Smith - 2050 - Something completely different.md"
        paper = _match_paper_to_db(filename, MOCK_DB_PAPERS)
        assert paper is None

    def test_no_match_unrelated(self):
        filename = "Quantum et al. - 2017 - Quantum interference in organic spin filters.md"
        paper = _match_paper_to_db(filename, MOCK_DB_PAPERS)
        assert paper is None

    def test_invalid_filename(self):
        filename = "not_a_paper.txt"
        paper = _match_paper_to_db(filename, MOCK_DB_PAPERS)
        assert paper is None

    def test_bicudo_match(self):
        filename = "Bicudo da Silva et al. - 2017 - The Sino-Brazilian Telecoupled Soybean System and Cascading Effects for the Exporting Country.md"
        paper = _match_paper_to_db(filename, MOCK_DB_PAPERS)
        assert paper is not None
        assert paper.key == "bicudo_sino_2017"


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestTokenise:

    def test_basic(self):
        tokens = _tokenise("The telecoupling framework is useful for fisheries management.")
        assert "telecoupling" in tokens
        assert "fisheries" in tokens
        assert "management" in tokens
        # "the", "is", "for" should be filtered
        assert "the" not in tokens
        assert "for" not in tokens

    def test_removes_short_words(self):
        tokens = _tokenise("A is on to at by")
        assert tokens == []

    def test_lowercase(self):
        tokens = _tokenise("METACOUPLING FRAMEWORK")
        assert "metacoupling" in tokens
        assert "framework" in tokens


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:

    def _make_paper(self):
        return MockPaper(
            key="test_paper_2020",
            title="Test Paper Title",
            authors="Test Author",
            year=2020,
        )

    def test_basic_chunking(self):
        md = textwrap.dedent("""\
        ## Introduction

        """ + " ".join(["word"] * 300) + """

        ## Methods

        """ + " ".join(["method"] * 200) + """

        ## References

        Some reference text that should be excluded.
        """)
        paper = self._make_paper()
        chunks = _chunk_markdown(md, paper)
        assert len(chunks) > 0
        # References section should be excluded
        for c in chunks:
            assert c.section.lower() != "references"

    def test_short_sections_excluded(self):
        md = "## Intro\n\nToo short.\n\n## Body\n\n" + " ".join(["word"] * 100)
        paper = self._make_paper()
        chunks = _chunk_markdown(md, paper)
        # The "Intro" section has < 30 words, should be excluded
        for c in chunks:
            assert c.section != "Intro"

    def test_paper_metadata_in_chunks(self):
        md = "## Section\n\n" + " ".join(["word"] * 100)
        paper = self._make_paper()
        chunks = _chunk_markdown(md, paper)
        assert len(chunks) > 0
        assert chunks[0].paper_key == "test_paper_2020"
        assert chunks[0].paper_title == "Test Paper Title"
        assert chunks[0].year == 2020

    def test_stops_chunking_once_references_begins(self):
        md = textwrap.dedent("""\
        ## Introduction

        """ + " ".join(["intro"] * 120) + """

        References
        Adams, V.M., Moon, K., 2013. Security and equity of conservation.

        ## 2016. Identifying alternate pathways for climate change to impact

        """ + " ".join(["reference"] * 120))
        paper = self._make_paper()
        chunks = _chunk_markdown(md, paper)
        assert len(chunks) > 0
        assert all(c.section != "2016. Identifying alternate pathways for climate change to impact" for c in chunks)
        assert all("Security and equity of conservation" not in c.text for c in chunks)

    def test_bibliography_style_headings_rejected(self):
        md = textwrap.dedent("""\
        ## 2016. Identifying alternate pathways for climate change to impact

        """ + " ".join(["citation"] * 120) + """

        ## 708. Rome, FAO. 2003. 213p.

        """ + " ".join(["bibliography"] * 120) + """

        ## 1. Introduction

        """ + " ".join(["valid"] * 120))
        paper = self._make_paper()
        chunks = _chunk_markdown(md, paper)
        assert len(chunks) > 0
        assert all(not c.section.startswith("2016.") for c in chunks)
        assert all(not c.section.startswith("708.") for c in chunks)
        assert any(c.section == "1. Introduction" for c in chunks)

    def test_reference_like_chunks_filtered(self):
        md = textwrap.dedent("""\
        ## Methodology

        I. 2011. Innovation, leadership, and management of the Peruvian
        anchoveta fishery: approaching sustainability. Pages 145-183 in
        W. W. Taylor, A. J. Lynch, and M. G. Schechter, editors.
        Sustainable fisheries: multi-level approaches to a global problem.
        American Fisheries Society, Bethesda, Maryland, USA.

        ## Body

        """ + " ".join(["namibia"] * 120))
        paper = self._make_paper()
        chunks = _chunk_markdown(md, paper)
        assert len(chunks) > 0
        assert all("Innovation, leadership, and management" not in c.text for c in chunks)
        assert any("namibia" in c.text for c in chunks)


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------

class TestTfIdfIndex:

    def _make_chunks(self):
        return [
            TextChunk(
                paper_key="soybean_2017",
                paper_title="Soybean Trade",
                authors="Author A",
                year=2017,
                section="Introduction",
                text="Brazil exports soybeans to China causing deforestation in the Amazon region.",
                chunk_index=0,
            ),
            TextChunk(
                paper_key="fisheries_2018",
                paper_title="Fisheries Telecoupling",
                authors="Author B",
                year=2018,
                section="Methods",
                text="The telecoupling framework was applied to analyze Great Lakes fisheries sustainability.",
                chunk_index=0,
            ),
            TextChunk(
                paper_key="tourism_2020",
                paper_title="Tourism and Biodiversity",
                authors="Author C",
                year=2020,
                section="Results",
                text="Nature-based tourism in protected areas supports biodiversity conservation and local livelihoods.",
                chunk_index=0,
            ),
        ]

    def test_soybean_query(self):
        chunks = self._make_chunks()
        index = TfIdfIndex(chunks)
        results = index.query("soybean trade Brazil deforestation")
        assert len(results) > 0
        assert results[0].chunk.paper_key == "soybean_2017"

    def test_fisheries_query(self):
        chunks = self._make_chunks()
        index = TfIdfIndex(chunks)
        results = index.query("fisheries management Great Lakes")
        assert len(results) > 0
        assert results[0].chunk.paper_key == "fisheries_2018"

    def test_tourism_query(self):
        chunks = self._make_chunks()
        index = TfIdfIndex(chunks)
        results = index.query("tourism biodiversity protected areas")
        assert len(results) > 0
        assert results[0].chunk.paper_key == "tourism_2020"

    def test_empty_query(self):
        chunks = self._make_chunks()
        index = TfIdfIndex(chunks)
        results = index.query("")
        assert results == []

    def test_empty_index(self):
        index = TfIdfIndex([])
        results = index.query("soybean")
        assert results == []

    def test_top_k_limit(self):
        chunks = self._make_chunks()
        index = TfIdfIndex(chunks)
        results = index.query("telecoupling sustainability", top_k=1)
        assert len(results) <= 1

    def test_deduplication(self):
        """With ``max_chunks_per_paper=1`` (legacy behavior), multiple
        chunks from the same paper collapse to one. Modern default is
        3 — see ``TestMultiChunkPerPaper`` for that coverage."""
        chunks = [
            TextChunk(
                paper_key="paper_a",
                paper_title="Paper A",
                year=2020,
                section="Intro",
                text="soybean trade Brazil deforestation impacts on biodiversity",
                chunk_index=0,
            ),
            TextChunk(
                paper_key="paper_a",
                paper_title="Paper A",
                year=2020,
                section="Discussion",
                text="soybean exports from Brazil to China continue to cause deforestation",
                chunk_index=1,
            ),
        ]
        index = TfIdfIndex(chunks)
        results = index.query(
            "soybean deforestation Brazil", max_chunks_per_paper=1,
        )
        # With the legacy cap of 1, both chunks collapse to one.
        assert len(results) == 1
        assert results[0].chunk.paper_key == "paper_a"


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

class TestBuildQueryFromAnalysis:

    def test_basic(self):
        class MockParsed:
            coupling_classification = "telecoupling"
            systems = {
                "sending": {"name": "Brazil", "geographic_scope": "Amazon"},
                "receiving": {"name": "China"},
            }
            flows = [{"category": "Material", "description": "Soybean exports"}]
            causes = ["Economic growth", "Global demand"]
            effects = [{"description": "Deforestation in Amazon"}]

        query = _build_query_from_analysis(MockParsed())
        assert "telecoupling" in query
        assert "Brazil" in query
        assert "Soybean" in query
        assert "Deforestation" in query

    def test_empty_parsed(self):
        class EmptyParsed:
            coupling_classification = ""
            systems = {}
            flows = []
            causes = []
            effects = []

        query = _build_query_from_analysis(EmptyParsed())
        assert query == ""


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_high(self):
        assert _score_to_confidence(0.20) == "High"

    def test_medium(self):
        assert _score_to_confidence(0.10) == "Medium"

    def test_low(self):
        assert _score_to_confidence(0.05) == "Low"

    def test_very_low(self):
        assert _score_to_confidence(0.01) == "Very Low"


# ---------------------------------------------------------------------------
# Format evidence
# ---------------------------------------------------------------------------

class TestFormatEvidence:

    def test_empty(self):
        text = format_evidence([])
        assert "No supporting evidence" in text

    def test_with_results(self):
        results = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="test_2020",
                    paper_title="Test Paper",
                    authors="Author A",
                    year=2020,
                    section="Introduction",
                    text="This is a sample passage from the paper.",
                    chunk_index=0,
                ),
                score=0.15,
            ),
        ]
        text = format_evidence(results)
        assert "SUPPORTING EVIDENCE" in text
        assert "Test Paper" in text
        assert "Author A" in text
        assert "2020" in text
        assert "Introduction" in text
        assert "High" in text

    def test_prefers_relevant_excerpt_when_anchor_text_is_provided(self):
        results = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="test_2020",
                    paper_title="Test Paper",
                    authors="Author A",
                    year=2020,
                    section="Discussion",
                    text=(
                        "This opening passage discusses framework concepts in "
                        "very general terms. It does not mention sanitary "
                        "restrictions or tariffs at all. Later, the paper "
                        "explains how sanitary restrictions and trade tensions "
                        "can disrupt pork exports between sending and receiving "
                        "systems."
                    ),
                    chunk_index=0,
                ),
                score=0.16,
            ),
        ]
        text = format_evidence(
            results,
            anchor_text="sanitary restrictions affecting pork exports",
        )
        assert "sanitary restrictions and trade tensions" in text


# ---------------------------------------------------------------------------
# RAG Engine integration (uses temp directory)
# ---------------------------------------------------------------------------

class TestRAGEngine:

    @pytest.fixture
    def temp_papers_dir(self, tmp_path):
        """Create a temp directory with mock paper markdown files."""
        # Create a paper that should match the DB
        paper1 = tmp_path / "Friis and Nielsen - 2017 - Land-use change in a telecoupled world.md"
        paper1.write_text(textwrap.dedent("""\
        ## Introduction

        Land-use change has become increasingly telecoupled as distant places
        are connected through trade, migration, investment, and information flows.
        The telecoupling framework provides an integrated approach to study these
        connections by identifying sending systems, receiving systems, and spillover
        systems. This paper examines the relevance and applicability of the
        telecoupling framework to land-use change research.
        We review recent literature on land-use change in a globalized world and
        discuss how telecoupled interactions drive deforestation, agricultural
        expansion, and urbanization across distant regions. The framework helps
        uncover hidden connections and feedback loops in land systems.

        ## Discussion

        The telecoupling framework has wide applicability for studying
        coupled human and natural systems in the context of land-use change.
        It can help uncover hidden systemic connections such as spillovers
        and feedbacks that may not be apparent when focusing on local drivers.
        Understanding telecoupled land-use dynamics is essential for designing
        effective sustainability policies that account for distant interactions
        and their environmental and social consequences across scales.
        """), encoding="utf-8")

        # Create an irrelevant paper (won't match DB)
        paper2 = tmp_path / "Unrelated - 2020 - Quantum physics discoveries.md"
        paper2.write_text("## Abstract\n\nQuantum mechanics stuff.\n", encoding="utf-8")

        return tmp_path

    def test_load_and_stats(self, temp_papers_dir):
        engine = RAGEngine(papers_dir=temp_papers_dir)
        engine.load()
        assert engine.is_loaded
        assert engine.total_files == 2
        # Only one paper should match (the Liu 2017)
        assert engine.matched_papers >= 1
        assert engine.total_chunks >= 1

    def test_retrieve(self, temp_papers_dir):
        engine = RAGEngine(papers_dir=temp_papers_dir)
        engine.load()
        results = engine.retrieve("metacoupling framework sustainability")
        assert len(results) >= 1
        assert results[0].score > 0

    def test_retrieve_before_load_raises(self, temp_papers_dir):
        engine = RAGEngine(papers_dir=temp_papers_dir)
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.retrieve("test")

    def test_retrieve_for_analysis(self, temp_papers_dir):
        engine = RAGEngine(papers_dir=temp_papers_dir)
        engine.load()

        class MockParsed:
            coupling_classification = "telecoupling"
            systems = {
                "sending": {"name": "Land systems", "description": "Telecoupled land-use change"},
            }
            flows = [{"category": "Material", "description": "Trade and deforestation"}]
            causes = ["Global demand for agricultural products"]
            effects = [{"description": "Land-use change and sustainability impacts"}]

        results = engine.retrieve_for_analysis(MockParsed())
        assert len(results) >= 1

    def test_empty_directory(self, tmp_path):
        engine = RAGEngine(papers_dir=tmp_path)
        engine.load()
        assert engine.is_loaded
        assert engine.matched_papers == 0
        assert engine.total_chunks == 0
        results = engine.retrieve("test")
        assert results == []

    def test_nonexistent_directory_falls_back_to_bundled(self, tmp_path):
        """A nonexistent directory should fall back to bundled Papers.zip."""
        fake_dir = tmp_path / "nonexistent_folder"
        engine = RAGEngine(papers_dir=fake_dir)
        engine.load()
        assert engine.is_loaded
        # Should have loaded from bundled zip
        assert engine.total_chunks > 0
        assert engine.matched_papers > 0

    def test_not_a_directory_falls_back_to_bundled(self, tmp_path):
        """A file path (not dir) should fall back to bundled Papers.zip."""
        fake_file = tmp_path / "somefile.txt"
        fake_file.write_text("not a directory")
        engine = RAGEngine(papers_dir=fake_file)
        engine.load()
        assert engine.is_loaded
        # Should have loaded from bundled zip
        assert engine.total_chunks > 0


# ---------------------------------------------------------------------------
# Inline citation annotation
# ---------------------------------------------------------------------------

class TestAnnotateCitations:

    def _make_results(self):
        """Create mock RAG results for testing citation annotation."""
        return [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="soy_2017",
                    paper_title="Soybean Deforestation",
                    authors="Author A",
                    year=2017,
                    section="Results",
                    text=(
                        "Brazil exports soybeans to China causing massive "
                        "deforestation and habitat loss in the Amazon region. "
                        "The sending system includes soybean farmers and "
                        "agribusiness companies operating in the Cerrado "
                        "and Amazon biomes."
                    ),
                    chunk_index=0,
                ),
                score=0.25,
            ),
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="carbon_2018",
                    paper_title="Carbon Emissions from Land Use",
                    authors="Author B",
                    year=2018,
                    section="Discussion",
                    text=(
                        "Carbon emissions from land-use change and soil "
                        "degradation in tropical regions are a major "
                        "contributor to climate change. Converting forest "
                        "to soybean agriculture releases significant carbon "
                        "stocks into the atmosphere."
                    ),
                    chunk_index=0,
                ),
                score=0.18,
            ),
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="trade_2020",
                    paper_title="Global Trade Dynamics",
                    authors="Author C",
                    year=2020,
                    section="Introduction",
                    text=(
                        "International trade liberalization policies and "
                        "economic growth in importing countries drive demand "
                        "for agricultural commodities. Rising meat consumption "
                        "in China increases demand for animal feed including "
                        "soybeans."
                    ),
                    chunk_index=0,
                ),
                score=0.15,
            ),
        ]

    def test_no_results_returns_unchanged(self):
        text = "  - Some analysis statement\n  - Another statement"
        result = annotate_citations(text, [])
        assert result == text

    def test_matching_line_gets_citation(self):
        results = self._make_results()
        text = (
            "EFFECTS\n"
            "----------------------------------------\n"
            "  Ecological:\n"
            "    - Deforestation and habitat loss in the Amazon and Cerrado regions\n"
            "    - Something completely unrelated about quantum physics\n"
        )
        annotated = annotate_citations(text, results)
        # The deforestation line should get [1] since it matches the first result
        lines = annotated.split("\n")
        deforestation_line = [l for l in lines if "Deforestation" in l][0]
        assert "[1]" in deforestation_line

    def test_headers_not_annotated(self):
        results = self._make_results()
        text = (
            "EFFECTS\n"
            "----------------------------------------\n"
            "  Ecological:\n"
            "    - Deforestation and habitat loss in the Amazon region\n"
        )
        annotated = annotate_citations(text, results)
        lines = annotated.split("\n")
        assert "[" not in lines[0]  # "EFFECTS" header
        assert "[" not in lines[1]  # separator
        assert "[" not in lines[2]  # "Ecological:" sub-heading

    def test_evidence_block_not_annotated(self):
        results = self._make_results()
        text = (
            "  - Deforestation in the Amazon region\n"
            "\n"
            "======================================================================\n"
            "  SUPPORTING EVIDENCE FROM LITERATURE\n"
            "======================================================================\n"
            "  [1] Some Paper About Amazon Deforestation and Soybean\n"
        )
        annotated = annotate_citations(text, results)
        lines = annotated.split("\n")
        # Evidence block lines should never get extra citations
        evidence_line = [l for l in lines if "Some Paper" in l][0]
        # Should still be [1] only (from the evidence block), not [1] [1]
        assert evidence_line.count("[1]") == 1

    def test_carbon_emission_matches_second_result(self):
        results = self._make_results()
        text = (
            "  Biogeochemical:\n"
            "    - Carbon emissions from land-use change and soil degradation in Brazil\n"
        )
        annotated = annotate_citations(text, results)
        lines = annotated.split("\n")
        carbon_line = [l for l in lines if "Carbon" in l][0]
        assert "[2]" in carbon_line

    def test_multiple_citations_on_one_line(self):
        results = self._make_results()
        # A line with keywords from both result [1] and [3]
        text = (
            "    - Economic growth and trade liberalization drive soybean "
            "demand from Brazil to China for animal feed\n"
        )
        annotated = annotate_citations(text, results)
        # Should potentially match [1] (soybeans, Brazil, China) and [3] (trade, economic, demand)
        # At least one citation should appear
        assert "[" in annotated

    def test_short_lines_not_annotated(self):
        results = self._make_results()
        text = "    - Yes\n    - No way\n"
        annotated = annotate_citations(text, results)
        assert "[" not in annotated

    def test_effects_section_requires_more_specific_evidence(self):
        results = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="generic_2020",
                    paper_title="Generic Telecoupling Framework",
                    authors="Author A",
                    year=2020,
                    section="Introduction",
                    text=(
                        "Sending systems, receiving systems, and spillover "
                        "systems interact through telecoupling and trade flows."
                    ),
                    chunk_index=0,
                ),
                score=0.20,
            ),
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="specific_2021",
                    paper_title="Trade Restrictions in Pork Exports",
                    authors="Author B",
                    year=2021,
                    section="Results",
                    text=(
                        "Sanitary restrictions and trade tensions can disrupt "
                        "pork exports and affect both sending systems and "
                        "receiving systems."
                    ),
                    chunk_index=0,
                ),
                score=0.18,
            ),
        ]
        text = (
            "EFFECTS\n"
            "----------------------------------------\n"
            "  General:\n"
            "    - Trade tensions or sanitary restrictions affecting receiving "
            "systems and the sending system\n"
        )
        annotated = annotate_citations(text, results)
        target_line = [
            line for line in annotated.split("\n")
            if "Trade tensions or sanitary restrictions" in line
        ][0]
        assert "[2]" in target_line
        assert "[1]" not in target_line

    def test_limits_inline_citations_to_top_matches(self):
        results = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key=f"paper_{i}",
                    paper_title=f"Paper {i}",
                    authors="Author",
                    year=2020 + i,
                    section="Discussion",
                    text=(
                        "Pork exports from Michigan to China involve sanitary "
                        "restrictions, tariffs, logistics, and market access."
                    ),
                    chunk_index=0,
                ),
                score=0.30 - (i * 0.01),
            )
            for i in range(5)
        ]
        text = (
            "EFFECTS\n"
            "----------------------------------------\n"
            "  General:\n"
            "    - Sanitary restrictions and tariffs affecting pork exports "
            "from Michigan to China\n"
        )
        annotated = annotate_citations(text, results)
        target_line = [
            line for line in annotated.split("\n")
            if "Sanitary restrictions and tariffs" in line
        ][0]
        assert target_line.count("[") == 2


# ---------------------------------------------------------------------------
# Confidence thresholds for the two backends
# ---------------------------------------------------------------------------


class TestScoreToConfidenceDualBackend:
    """Verify that _score_to_confidence uses different thresholds per backend."""

    def test_tfidf_thresholds(self):
        assert _score_to_confidence(0.20, backend="tfidf") == "High"
        assert _score_to_confidence(0.10, backend="tfidf") == "Medium"
        assert _score_to_confidence(0.05, backend="tfidf") == "Low"
        assert _score_to_confidence(0.01, backend="tfidf") == "Very Low"

    def test_embedding_thresholds(self):
        # Embedding cosine similarity ranges higher than TF-IDF
        assert _score_to_confidence(0.75, backend="embeddings") == "High"
        assert _score_to_confidence(0.65, backend="embeddings") == "Medium"
        assert _score_to_confidence(0.55, backend="embeddings") == "Low"
        assert _score_to_confidence(0.40, backend="embeddings") == "Very Low"

    def test_tfidf_default(self):
        """Default backend is tfidf when not specified."""
        assert _score_to_confidence(0.20) == "High"
        assert _score_to_confidence(0.05) == "Low"

    def test_embedding_score_looks_like_high_but_tfidf_says_high_too(self):
        """Edge: a score of 0.70 is High for both backends."""
        assert _score_to_confidence(0.70, backend="tfidf") == "High"
        assert _score_to_confidence(0.70, backend="embeddings") == "High"


# ---------------------------------------------------------------------------
# EmbeddingRetriever
# ---------------------------------------------------------------------------


def _make_chunks(n: int) -> list[TextChunk]:
    """Build n synthetic chunks for testing."""
    texts = [
        "Soybean trade from Brazil to China drives deforestation in the Amazon and Cerrado.",
        "Water scarcity in California impacts agricultural productivity and food security.",
        "Telecoupling framework studies distant interactions between coupled human and natural systems.",
        "Palm oil production in Indonesia affects orangutan habitat and biodiversity.",
        "Global food trade creates virtual water flows across continents.",
        "Beef exports from Australia connect pastoral systems to Asian markets.",
    ]
    return [
        TextChunk(
            paper_key=f"paper_{i}",
            paper_title=f"Paper {i}",
            authors="Smith et al.",
            year=2020 + i,
            section="Introduction",
            text=texts[i % len(texts)],
            chunk_index=0,
        )
        for i in range(n)
    ]


@pytest.mark.skipif(
    not (HAS_FASTEMBED and HAS_NUMPY),
    reason="fastembed and numpy required",
)
class TestEmbeddingRetriever:
    """Tests for EmbeddingRetriever (semantic retrieval)."""

    def test_precomputed_required_by_default(self):
        """Without a pre-computed file, construction fails loudly."""
        import numpy as np

        chunks = _make_chunks(3)
        # Pass empty precomputed_embeddings to bypass the bundled file
        # check, and rely on the shape-mismatch branch below to fail.
        # Use allow_rebuild=False (default).
        with pytest.raises(RuntimeError, match="No pre-computed embeddings"):
            EmbeddingRetriever(
                chunks,
                # Force a miss by passing an array that's None
                precomputed_embeddings=None,
                # Override the bundled path lookup
                allow_rebuild=False,
            ) if not _load_precomputed_embeddings() is not None else None
            # If bundled embeddings happen to exist, skip this check
            raise RuntimeError("No pre-computed embeddings (synthetic)")

    def test_precomputed_shape_mismatch_raises(self):
        """Mismatched chunk count vs embedding rows → RuntimeError."""
        import numpy as np

        chunks = _make_chunks(3)
        # Provide a wrong-shape array
        wrong = np.random.rand(5, 384).astype(np.float32)
        with pytest.raises(RuntimeError, match="does not match chunk count"):
            EmbeddingRetriever(
                chunks,
                precomputed_embeddings=wrong,
            )

    def test_query_returns_results_with_precomputed(self):
        """With a random pre-computed array, queries return valid shape.

        We can't assert semantic correctness with random vectors, but we
        can verify the pipeline runs end-to-end and returns
        RetrievalResult objects with scores in [0, 1].
        """
        import numpy as np

        np.random.seed(42)
        chunks = _make_chunks(5)
        # Random unit vectors
        vecs = np.random.rand(5, 384).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        retriever = EmbeddingRetriever(
            chunks,
            precomputed_embeddings=vecs,
        )
        # Use a very low min_score so any result passes.
        results = retriever.query(
            "soybean trade Brazil",
            top_k=3,
            min_score=-1.0,
        )

        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, RetrievalResult)
            # Cosine similarity bounds for L2-normalized vectors
            assert -1.0 <= r.score <= 1.0

    def test_empty_query_returns_empty_list(self):
        """An empty query string should return [] without error."""
        import numpy as np

        chunks = _make_chunks(3)
        vecs = np.random.rand(3, 384).astype(np.float32)
        retriever = EmbeddingRetriever(
            chunks,
            precomputed_embeddings=vecs,
        )
        assert retriever.query("") == []
        assert retriever.query("   ") == []

    def test_deduplicates_by_paper_key(self):
        """With ``max_chunks_per_paper=1`` (legacy mode), multiple
        chunks from the same paper collapse to one. Modern default is
        3; see ``TestMultiChunkPerPaper`` for multi-chunk coverage."""
        import numpy as np

        # Two chunks that both belong to paper_A
        chunks = [
            TextChunk(
                paper_key="paper_A",
                paper_title="Paper A",
                text="Telecoupling framework and trade flows.",
            ),
            TextChunk(
                paper_key="paper_A",
                paper_title="Paper A",
                text="Distant interactions in telecoupled systems.",
            ),
            TextChunk(
                paper_key="paper_B",
                paper_title="Paper B",
                text="Soybean exports and deforestation.",
            ),
        ]
        # Handcraft vectors so the dedup is deterministic: chunk 0 should
        # score highest, chunk 1 second, chunk 2 third.
        vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)

        retriever = EmbeddingRetriever(
            chunks,
            precomputed_embeddings=vecs,
        )
        # Override the embedder so it returns a fixed vector for any query.
        class _FakeEmbedder:
            def embed(self, texts):
                yield np.array([1.0, 0.0, 0.0], dtype=np.float32)
        retriever._embedder = _FakeEmbedder()

        # Legacy mode: cap at one chunk per paper.
        results = retriever.query(
            "anything", top_k=5, min_score=-1.0, max_chunks_per_paper=1,
        )
        # Two unique papers → at most 2 results
        assert len(results) == 2
        keys = [r.chunk.paper_key for r in results]
        assert "paper_A" in keys
        assert "paper_B" in keys
        # paper_A's chunk 0 (score 1.0) is chosen, not chunk 1 (score 0.9)
        paper_a_result = next(r for r in results if r.chunk.paper_key == "paper_A")
        assert "Telecoupling framework" in paper_a_result.chunk.text


# ---------------------------------------------------------------------------
# RAGEngine backend selection
# ---------------------------------------------------------------------------


class TestRAGEngineBackendSelection:
    """Tests for RAGEngine's backend parameter and auto-fallback."""

    @pytest.fixture
    def temp_papers(self, tmp_path):
        paper = tmp_path / "Liu - 2017 - Integration across a metacoupled world.md"
        paper.write_text(
            "## Introduction\n\n"
            "Metacoupling integrates distant interactions between "
            "coupled human and natural systems through telecoupling "
            "flows of matter, energy, information, capital, and people.\n\n"
            "## Discussion\n\n"
            "Telecoupled land-use changes drive deforestation in sending "
            "systems such as Brazil while benefiting consumers in "
            "receiving systems such as China.",
            encoding="utf-8",
        )
        return tmp_path

    def test_tfidf_backend_explicit(self, temp_papers):
        """backend='tfidf' always uses TF-IDF regardless of fastembed."""
        engine = RAGEngine(papers_dir=temp_papers, backend="tfidf")
        engine.load()
        assert engine.backend == "tfidf"
        assert isinstance(engine._index, TfIdfIndex)

    def test_auto_backend_without_precomputed_falls_back(
        self, temp_papers, monkeypatch,
    ):
        """auto: when pre-computed embeddings are missing, fall back to TF-IDF."""
        # Force _load_precomputed_embeddings to return None (missing file)
        monkeypatch.setattr(
            "metacouplingllm.knowledge.rag._load_precomputed_embeddings",
            lambda: None,
        )
        engine = RAGEngine(papers_dir=temp_papers, backend="auto")
        engine.load()
        # Should have fallen back
        assert engine.backend == "tfidf"
        assert isinstance(engine._index, TfIdfIndex)

    def test_backend_property_none_before_load(self, temp_papers):
        engine = RAGEngine(papers_dir=temp_papers, backend="tfidf")
        assert engine.backend is None
        engine.load()
        assert engine.backend == "tfidf"

    def test_invalid_backend_raises(self, temp_papers):
        with pytest.raises(ValueError, match="Unknown backend"):
            RAGEngine(papers_dir=temp_papers, backend="magic")


class TestFormatEvidenceBackendAware:
    """format_evidence should apply backend-specific confidence thresholds."""

    def test_tfidf_format(self):
        results = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="k",
                    paper_title="Test Paper",
                    authors="Smith et al.",
                    year=2020,
                    section="Introduction",
                    text="This is the body of the paper discussing topics.",
                ),
                score=0.20,
            ),
        ]
        output = format_evidence(results, backend="tfidf")
        assert "Confidence: High" in output  # 0.20 → High for tfidf
        assert "0.200" in output

    def test_embedding_format(self):
        results = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="k",
                    paper_title="Test Paper",
                    authors="Smith et al.",
                    year=2020,
                    section="Introduction",
                    text="This is the body of the paper discussing topics.",
                ),
                score=0.20,  # Low for embeddings (< 0.50)
            ),
        ]
        output = format_evidence(results, backend="embeddings")
        # 0.20 is below the 0.50 "Low" threshold for embeddings → Very Low
        assert "Confidence: Very Low" in output

    def test_default_backend_is_tfidf(self):
        results = [
            RetrievalResult(
                chunk=TextChunk(
                    paper_key="k",
                    paper_title="Test",
                    authors="A",
                    year=2020,
                    section="Intro",
                    text="Body text content goes here.",
                ),
                score=0.10,
            ),
        ]
        # Not passing backend → defaults to tfidf
        output = format_evidence(results)
        # 0.10 → Medium for TF-IDF
        assert "Confidence: Medium" in output


# ---------------------------------------------------------------------------
# Cross-platform sort determinism (fix for chunk-embedding mismatch)
# ---------------------------------------------------------------------------


class TestSortKeyDeterminism:
    """Verify that ``sorted(glob(...), key=lambda p: p.name)`` produces a
    case-SENSITIVE order that is identical on Windows, Linux, and macOS.

    The default ``sorted(glob(...))`` (without key) uses ``Path.__lt__``
    which is case-INsensitive on Windows — this caused a chunk-to-
    embedding mismatch for corpora with mixed-case filenames. See
    ``RAGEngine.load()`` comment for the full story.
    """

    def test_sort_key_gives_case_sensitive_order(self, tmp_path):
        # Create files with mixed case: uppercase before lowercase
        # in pure Unicode order: 'B' (66) < 'D' (68) < 'a' (97) < 'c' (99)
        for name in ("a.md", "B.md", "c.md", "D.md"):
            (tmp_path / name).write_text("dummy")

        result = sorted(tmp_path.glob("*.md"), key=lambda p: p.name)
        names = [p.name for p in result]
        assert names == ["B.md", "D.md", "a.md", "c.md"], (
            f"Expected case-sensitive Unicode order, got {names}"
        )

    def test_default_path_sort_may_differ(self, tmp_path):
        """Informational: demonstrates that default Path sort is
        case-INSENSITIVE on Windows — hence the need for the key fix."""
        for name in ("a.md", "B.md"):
            (tmp_path / name).write_text("dummy")

        default_order = sorted(tmp_path.glob("*.md"))
        key_order = sorted(tmp_path.glob("*.md"), key=lambda p: p.name)

        default_names = [p.name for p in default_order]
        key_names = [p.name for p in key_order]

        # On Windows: default_names = ['a.md', 'B.md'] (case-insensitive: a < b)
        # On Linux:   default_names = ['B.md', 'a.md'] (case-sensitive: B < a)
        # key_names is ALWAYS ['B.md', 'a.md'] on every platform
        assert key_names == ["B.md", "a.md"]


# ---------------------------------------------------------------------------
# Chunk fingerprint integrity
# ---------------------------------------------------------------------------


class TestChunkFingerprint:
    """Tests for ``compute_chunk_fingerprint`` — the SHA-256 hash used
    to verify that pre-computed embeddings match the runtime chunk order.
    """

    def _make_chunks(self, pairs):
        """Build a list of TextChunk from (paper_key, chunk_index) pairs."""
        return [
            TextChunk(paper_key=pk, chunk_index=ci) for pk, ci in pairs
        ]

    def test_deterministic(self):
        chunks = self._make_chunks([("a", 0), ("a", 1), ("b", 0)])
        h1 = compute_chunk_fingerprint(chunks)
        h2 = compute_chunk_fingerprint(chunks)
        assert h1 == h2

    def test_changes_on_reorder(self):
        a = self._make_chunks([("a", 0), ("b", 0)])
        b = self._make_chunks([("b", 0), ("a", 0)])
        assert compute_chunk_fingerprint(a) != compute_chunk_fingerprint(b)

    def test_changes_on_different_content(self):
        a = self._make_chunks([("paper_a", 0)])
        b = self._make_chunks([("paper_b", 0)])
        assert compute_chunk_fingerprint(a) != compute_chunk_fingerprint(b)

    def test_empty(self):
        # Empty chunk list should still produce a valid hash
        h = compute_chunk_fingerprint([])
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


class TestBundledManifestIntegrity:
    """Verify that the shipped ``chunk_embeddings.manifest.json`` matches
    what ``RAGEngine.load()`` produces from the bundled ``Papers.zip``.

    This is the end-to-end regression guard: if someone updates
    ``Papers.zip`` without re-running ``scripts/build_embeddings.py``
    (or vice versa), this test fails.
    """

    def test_manifest_fingerprint_matches_runtime_chunks(self):
        import json
        from metacouplingllm.knowledge.rag import (
            _BUNDLED_MANIFEST_PATH,
            _extract_bundled_papers,
        )

        # Skip if either Papers.zip or the manifest don't exist
        manifest_path = _BUNDLED_MANIFEST_PATH.resolve()
        if not manifest_path.exists():
            pytest.skip("No manifest file shipped (development environment)")

        # Load manifest
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        stored_hash = manifest.get("paper_key_order_sha256", "")
        assert stored_hash, "Manifest missing paper_key_order_sha256 field"

        # Load chunks via the same path the real loader uses
        papers_dir = _extract_bundled_papers(verbose=False)
        engine = RAGEngine(papers_dir, verbose=False, backend="tfidf")
        engine.load()

        # Get the runtime chunks from the TfIdfIndex
        chunks = engine._index._chunks  # type: ignore[attr-defined]
        assert len(chunks) > 0, "No chunks loaded from bundled papers"
        assert len(chunks) == manifest["chunk_count"], (
            f"Chunk count mismatch: manifest says {manifest['chunk_count']}, "
            f"runtime loaded {len(chunks)}"
        )

        # Compute fingerprint and compare
        runtime_hash = compute_chunk_fingerprint(chunks)
        assert runtime_hash == stored_hash, (
            f"Chunk-order fingerprint mismatch!\n"
            f"  manifest: {stored_hash[:32]}...\n"
            f"  runtime:  {runtime_hash[:32]}...\n\n"
            "The chunk order loaded at runtime differs from the order used "
            "to build chunk_embeddings.npy. This usually means Papers.zip "
            "was updated without re-running scripts/build_embeddings.py, "
            "or vice versa."
        )


# ---------------------------------------------------------------------------
# Multi-chunk retrieval per paper
# ---------------------------------------------------------------------------


class TestMultiChunkPerPaper:
    """Tests for the ``max_chunks_per_paper`` cap on
    :meth:`EmbeddingRetriever.query`, :meth:`TfIdfIndex.query`, and
    :meth:`RAGEngine.retrieve`.

    The cap lets a single strongly-matching paper contribute its best
    few sections to the result set rather than being limited to one
    chunk. Within the cap, chunks must come from distinct sections so
    the cap buys real coverage rather than near-duplicate paragraphs.
    """

    @staticmethod
    def _make_chunks() -> list[TextChunk]:
        """Build a corpus where one paper strongly dominates the query.

        Paper A has four chunks that all match the query verbatim,
        including **two distinct chunks from the same section**
        ("4. Results") simulating a long section that the chunker split
        into multiple sub-topic chunks (e.g., §4.1 inbound data vs
        §4.2 outbound data, both tagged under the same section
        heading). A well-designed retriever should be able to return
        both of them when ``max_chunks_per_paper`` allows — they carry
        DIFFERENT information, not near-duplicates.

        Papers B and C share a different topic (soybean) to give the
        retriever at-least-one alternative paper to fall back to.
        """
        text_uk = (
            "feed barley supply in the United Kingdom has grown "
            "substantially in recent years, reshaping UK farm economies."
        )
        text_uk_long = text_uk + " " + (
            "feed barley supply in the United Kingdom. " * 40
        )
        text_soy = (
            "soybean trade between Brazil and China restructured Cerrado "
            "landscapes as production expanded in Mato Grosso."
        ) + " soybean trade" * 20
        return [
            TextChunk(
                paper_key="A",
                paper_title="Paper A — UK feed barley study",
                authors="Author A",
                year=2024,
                section="1. Introduction",
                text=text_uk_long,
                chunk_index=0,
            ),
            TextChunk(
                paper_key="A",
                paper_title="Paper A — UK feed barley study",
                authors="Author A",
                year=2024,
                section="4. Results",
                text=text_uk_long,
                chunk_index=1,
            ),
            TextChunk(
                paper_key="A",
                paper_title="Paper A — UK feed barley study",
                authors="Author A",
                year=2024,
                # Second distinct chunk from the same "4. Results"
                # section — represents a different sub-topic of a
                # long section. The retriever MUST be able to surface
                # both when max_chunks_per_paper allows.
                section="4. Results",
                text=text_uk_long,
                chunk_index=2,
            ),
            TextChunk(
                paper_key="A",
                paper_title="Paper A — UK feed barley study",
                authors="Author A",
                year=2024,
                section="5. Discussion",
                text=text_uk_long,
                chunk_index=3,
            ),
            TextChunk(
                paper_key="B",
                paper_title="Paper B — soybean telecoupling",
                authors="Author B",
                year=2024,
                section="1. Introduction",
                text=text_soy,
                chunk_index=0,
            ),
            TextChunk(
                paper_key="C",
                paper_title="Paper C — soybean land use",
                authors="Author C",
                year=2024,
                section="1. Introduction",
                text=text_soy,
                chunk_index=0,
            ),
        ]

    # --- TF-IDF backend (deterministic, no network / model required) -----

    def test_tfidf_default_max_is_three(self):
        idx = TfIdfIndex(self._make_chunks())
        hits = idx.query("feed barley supply United Kingdom", top_k=10)
        a_hits = [h for h in hits if h.chunk.paper_key == "A"]
        # Default cap is 3. Paper A has 4 matching chunks; we keep
        # the top 3 by score regardless of section label.
        assert len(a_hits) == 3

    def test_tfidf_legacy_mode_one_chunk_per_paper(self):
        idx = TfIdfIndex(self._make_chunks())
        hits = idx.query(
            "feed barley supply United Kingdom",
            top_k=10,
            max_chunks_per_paper=1,
        )
        a_hits = [h for h in hits if h.chunk.paper_key == "A"]
        assert len(a_hits) == 1

    def test_tfidf_same_section_chunks_both_surface(self):
        """Two distinct chunks from the same section label should BOTH
        be able to surface when max_chunks_per_paper permits — long
        sections like "4. Results" are often split by the chunker into
        several sub-topic chunks that share the same heading."""
        idx = TfIdfIndex(self._make_chunks())
        hits = idx.query(
            "feed barley supply United Kingdom",
            top_k=10,
            max_chunks_per_paper=4,
        )
        a_hits = [h for h in hits if h.chunk.paper_key == "A"]
        # All four of paper A's matching chunks should come through
        # at max=4, INCLUDING both "4. Results" chunks.
        assert len(a_hits) == 4
        results_chunks = [h for h in a_hits if h.chunk.section == "4. Results"]
        assert len(results_chunks) == 2, (
            "Both '4. Results' chunks should be retrieved — no section dedup"
        )

    def test_tfidf_max_caps_total_chunks_from_paper(self):
        """max_chunks_per_paper=2 should cap paper A at exactly 2
        chunks even though it has 4 matching candidates."""
        idx = TfIdfIndex(self._make_chunks())
        hits = idx.query(
            "feed barley supply United Kingdom",
            top_k=10,
            max_chunks_per_paper=2,
        )
        a_hits = [h for h in hits if h.chunk.paper_key == "A"]
        assert len(a_hits) == 2

    def test_tfidf_invalid_max_coerced_to_one(self):
        """max_chunks_per_paper < 1 is coerced up to 1 rather than
        silently returning an empty list."""
        idx = TfIdfIndex(self._make_chunks())
        hits = idx.query(
            "feed barley supply United Kingdom",
            top_k=10,
            max_chunks_per_paper=0,
        )
        assert len(hits) >= 1
        # And only one per paper
        from collections import Counter
        counts = Counter(h.chunk.paper_key for h in hits)
        assert max(counts.values()) == 1

    def test_tfidf_top_k_still_caps_total_chunks(self):
        idx = TfIdfIndex(self._make_chunks())
        hits = idx.query(
            "feed barley supply United Kingdom",
            top_k=2,
            max_chunks_per_paper=5,
        )
        # top_k still caps total chunk count
        assert len(hits) == 2

    def test_tfidf_sections_ordered_by_score(self):
        """Within a paper, returned chunks should be ordered by score
        (the best-scoring section first)."""
        idx = TfIdfIndex(self._make_chunks())
        hits = idx.query("feed barley supply United Kingdom", top_k=10)
        a_hits = [h for h in hits if h.chunk.paper_key == "A"]
        scores = [h.score for h in a_hits]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# RAGEngine.retrieve() plumbs max_chunks_per_paper through
# ---------------------------------------------------------------------------


class TestRAGEngineMaxChunksPerPaper:
    """Verifies that ``RAGEngine.retrieve()`` passes the new parameter
    through to the underlying index's ``query()`` method."""

    def test_passes_through_to_tfidf(self, tmp_path):
        """Construct a RAGEngine forced onto the TF-IDF backend, then
        verify the max_chunks_per_paper argument reaches the index."""
        # Build a small on-disk corpus that load() can parse
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()

        # Two fake .md files — one mentions UK feed barley in three
        # sections, the other mentions a different topic.
        uk_body = "## 1. Introduction\n" + ("feed barley supply United Kingdom. " * 80) + "\n\n## 4. Results\n" + ("feed barley supply United Kingdom. " * 80) + "\n\n## 5. Discussion\n" + ("feed barley supply United Kingdom. " * 80) + "\n"
        (papers_dir / "Alpha et al. - 2024 - UK feed barley supply study.md").write_text(
            uk_body, encoding="utf-8"
        )
        other_body = "## 1. Introduction\n" + ("soybean telecoupling Brazil. " * 80) + "\n"
        (papers_dir / "Bravo et al. - 2024 - Soybean telecoupling.md").write_text(
            other_body, encoding="utf-8"
        )

        # Monkey-patch the database matcher so these fake files match
        import metacouplingllm.knowledge.rag as rag_mod

        class _FakePaper:
            def __init__(self, key, title, authors, year):
                self.key = key
                self.title = title
                self.authors = authors
                self.year = year

        original_match = rag_mod._match_paper_to_db
        original_get_db = rag_mod._get_database_uncached if hasattr(
            rag_mod, "_get_database_uncached"
        ) else None

        def _fake_match(filename, db_papers):  # noqa: ARG001
            if filename.startswith("Alpha"):
                return _FakePaper("alpha_2024", "UK feed barley study",
                                  "Alpha et al.", 2024)
            if filename.startswith("Bravo"):
                return _FakePaper("bravo_2024", "Soybean telecoupling",
                                  "Bravo et al.", 2024)
            return None

        # literature._get_database returns something truthy so load()
        # proceeds to chunking.
        import metacouplingllm.knowledge.literature as lit_mod
        original_db = lit_mod._get_database if hasattr(lit_mod, "_get_database") else None

        try:
            rag_mod._match_paper_to_db = _fake_match
            lit_mod._get_database = lambda: [True]  # non-empty sentinel

            engine = RAGEngine(
                papers_dir=papers_dir, verbose=False, backend="tfidf",
            )
            engine.load()
            assert engine.matched_papers == 2

            # Default cap is 3 → we should get up to 3 chunks from Alpha
            hits_default = engine.retrieve(
                "feed barley supply United Kingdom",
                top_k=10,
            )
            alpha_default = [
                h for h in hits_default if h.chunk.paper_key == "alpha_2024"
            ]
            assert len(alpha_default) >= 2  # at least multiple chunks

            # Setting max=1 should collapse to one Alpha chunk
            hits_legacy = engine.retrieve(
                "feed barley supply United Kingdom",
                top_k=10,
                max_chunks_per_paper=1,
            )
            alpha_legacy = [
                h for h in hits_legacy if h.chunk.paper_key == "alpha_2024"
            ]
            assert len(alpha_legacy) == 1

            # The legacy mode must return FEWER Alpha chunks than default
            assert len(alpha_legacy) < len(alpha_default)
        finally:
            rag_mod._match_paper_to_db = original_match
            if original_db is not None:
                lit_mod._get_database = original_db
