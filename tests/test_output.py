"""Tests for PR #31 scholar-facing export: render_markdown / render_docx
and the AnalysisResult.to_markdown() / .to_docx() convenience methods.

Built against ``output/export.py``.  ``python-docx`` is optional --
docx tests skip when the dependency isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from metacouplingllm.core import AnalysisResult
from metacouplingllm.llm.parser import ParsedAnalysis
from metacouplingllm.output.export import (
    _build_sections,
    render_docx,
    render_markdown,
)

from ._helpers import make_parsed_analysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_rag_hit(
    title: str, authors: str, year: int, section: str, text: str, score: float,
):
    """Build a fake RetrievalResult-shaped object for tests without
    importing the real class (keeps test isolated from RAG internals)."""
    from types import SimpleNamespace

    chunk = SimpleNamespace(
        paper_key=title.lower().replace(" ", "_"),
        paper_title=title,
        authors=authors,
        year=year,
        section=section,
        text=text,
        chunk_index=0,
    )
    return SimpleNamespace(chunk=chunk, score=score)


@pytest.fixture
def minimal_result() -> AnalysisResult:
    """An AnalysisResult with just enough content for the renderers."""
    parsed = make_parsed_analysis(
        coupling_classification="Telecoupling (Mexico → USA avocado trade).",
        systems={
            "sending": {
                "name": "Mexico (Jalisco)",
                "geographic_scope": "Western Mexico",
                "description": "Major avocado-producing region.",
            },
            "receiving": {
                "name": "United States",
                "geographic_scope": "Continental USA",
                "description": "Primary avocado importer.",
            },
        },
        flows=[
            {
                "category": "matter",
                "direction": "Mexico → USA",
                "description": "Fresh avocado exports",
            },
            {
                "category": "capital",
                "direction": "USA → Mexico",
                "description": "Payments for produce",
            },
        ],
        coupling_type="telecoupling",
        raw_text="Avocado telecoupling analysis.",
    )
    parsed.cross_coupling_interactions = [
        "Trade tariffs feed back into Mexican producer prices.",
    ]
    parsed.research_gaps = [
        "Subnational variation in Jalisco's avocado supply chain.",
    ]
    parsed.evidence_coverage_note = (
        "Coverage is strong for trade volumes; weak on farmworker welfare."
    )
    parsed.map_data = {
        "focal_country": "MEX",
        "flows": [
            {
                "category": "matter",
                "source": "MEX",
                "target": "USA",
                "bidirectional": False,
                "description": "avocado exports",
            },
            {
                "category": "capital",
                "source": "USA",
                "target": "MEX",
                "bidirectional": False,
                "description": "payments",
            },
        ],
    }

    result = AnalysisResult(
        parsed=parsed,
        formatted="(formatted-text placeholder)",
        raw="(raw-text placeholder)",
        turn_number=1,
        abstract=(
            "This study examines the Mexico-USA avocado trade as a "
            "telecoupling. Jalisco's producers send fresh fruit to U.S. "
            "consumers in exchange for capital. Key findings include "
            "tariff feedback effects on producer prices."
        ),
    )
    # Mock web sources via the assistant-side attribute used by the
    # builder.  Real pipeline populates this from _last_web_results.
    result._web_sources_for_export = [
        {
            "title": "Mexico avocado exports 2024",
            "url": "https://example.com/mex-exports-2024",
            "model_summary": "Mexico shipped 1.1M tonnes of avocados to USA.",
        },
        {
            "title": "USDA avocado import data",
            "url": "https://usda.gov/avocado-imports",
            "model_summary": "USDA records the breakdown by destination state.",
        },
    ]
    # PR #31 follow-up: RAG hits attached the same way.  Real pipeline
    # attaches result._rag_hits_for_export = list(self._last_rag_hits).
    result._rag_hits_for_export = [
        _fake_rag_hit(
            title="Mexico's avocado industry and global trade",
            authors="Garcia, M. and Hernandez, L.",
            year=2023,
            section="DISCUSSION",
            text=(
                "Jalisco's accession to the USDA Systems Approach in 2022 "
                "represents a structural shift in the bi-national avocado "
                "supply chain, with cold-chain logistics concentrated at "
                "the Laredo, TX corridor."
            ),
            score=0.82,
        ),
        _fake_rag_hit(
            title="Telecoupled agricultural systems",
            authors="Liu, J. et al.",
            year=2013,
            section="INTRODUCTION",
            text=(
                "Telecoupling links distant systems through flows of "
                "matter, capital, information, energy, people, and "
                "organisms."
            ),
            score=0.74,
        ),
    ]
    return result


# ---------------------------------------------------------------------------
# AnalysisResult.abstract field
# ---------------------------------------------------------------------------


class TestAbstractField:
    def test_abstract_default_empty_string(self):
        """PR #31: AnalysisResult.abstract defaults to empty string so
        tests / callers that don't go through _build_result don't need
        to set it."""
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
        )
        assert result.abstract == ""

    def test_abstract_field_carries_text(self):
        text = "A focused abstract for a paper introduction."
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="",
            raw="",
            turn_number=1,
            abstract=text,
        )
        assert result.abstract == text


# ---------------------------------------------------------------------------
# _build_sections (internal intermediate)
# ---------------------------------------------------------------------------


class TestBuildSections:
    def test_pulls_abstract_from_result(self, minimal_result):
        s = _build_sections(minimal_result)
        assert "telecoupling" in s["abstract"].lower()

    def test_pulls_coupling_classification(self, minimal_result):
        s = _build_sections(minimal_result)
        assert "Telecoupling" in s["coupling_classification"]

    def test_pulls_flows_from_map_data(self, minimal_result):
        s = _build_sections(minimal_result)
        assert len(s["flows"]) == 2
        assert s["flows"][0]["category"] == "matter"
        assert s["flows"][0]["source"] == "MEX"
        assert s["flows"][0]["target"] == "USA"

    def test_pulls_web_sources_with_w_prefixed_ids(self, minimal_result):
        s = _build_sections(minimal_result)
        assert len(s["web_sources"]) == 2
        assert s["web_sources"][0]["id"] == "W1"
        assert s["web_sources"][1]["id"] == "W2"
        assert "usda.gov" in s["web_sources"][1]["url"]

    def test_focal_country_from_map_data(self, minimal_result):
        assert _build_sections(minimal_result)["focal_country"] == "MEX"

    def test_flows_falls_back_to_coupling_section_when_no_map_data(self):
        parsed = make_parsed_analysis(
            flows=[
                {
                    "category": "matter",
                    "direction": "A → B",
                    "description": "stuff",
                },
            ],
            coupling_type="telecoupling",
        )
        # No map_data attribute set.
        parsed.map_data = None
        result = AnalysisResult(
            parsed=parsed, formatted="", raw="", turn_number=1,
        )
        s = _build_sections(result)
        assert len(s["flows"]) == 1
        assert s["flows"][0]["category"] == "matter"
        # Source/target empty in fallback path.
        assert s["flows"][0]["source"] == ""

    def test_graceful_when_parsed_minimal(self):
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="", raw="", turn_number=1,
        )
        s = _build_sections(result)
        assert s["abstract"] == ""
        assert s["coupling_classification"] == ""
        assert s["intracoupling"] is None
        assert s["telecoupling"] is None
        assert s["flows"] == []
        assert s["web_sources"] == []
        assert s["rag_hits"] == []

    # PR #31 follow-up: RAG hits in the sections dict.

    def test_pulls_rag_hits_with_turn_scoped_ids(self, minimal_result):
        s = _build_sections(minimal_result)
        assert len(s["rag_hits"]) == 2
        # IDs follow the turn-scoped citation grammar [T<turn>:<idx>]
        # matching the inline citations in result.formatted.
        assert s["rag_hits"][0]["id"] == "T1:1"
        assert s["rag_hits"][1]["id"] == "T1:2"
        assert s["rag_hits"][0]["paper_title"].startswith("Mexico's avocado")
        assert s["rag_hits"][0]["year"] == "2023"
        assert s["rag_hits"][0]["score"] == pytest.approx(0.82)

    def test_rag_text_truncated_at_600_chars(self):
        long_text = "X" * 1200
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="", raw="", turn_number=1,
        )
        result._rag_hits_for_export = [
            _fake_rag_hit("T", "A", 2024, "S", long_text, 0.9),
        ]
        s = _build_sections(result)
        # Truncation cap is 600 chars + "..." suffix.
        assert len(s["rag_hits"][0]["text"]) <= 603
        assert s["rag_hits"][0]["text"].endswith("...")


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_returns_string_with_top_level_title(self, minimal_result):
        md = render_markdown(minimal_result)
        # Title line starts with single `# `.
        first_line = md.splitlines()[0]
        assert first_line.startswith("# Metacoupling Analysis")
        assert "MEX" in first_line

    def test_includes_abstract_section(self, minimal_result):
        md = render_markdown(minimal_result)
        assert "## Abstract" in md
        assert "telecoupling" in md.lower()

    def test_includes_coupling_classification_section(self, minimal_result):
        md = render_markdown(minimal_result)
        assert "## 1. Coupling Classification" in md

    def test_includes_telecoupling_section(self, minimal_result):
        md = render_markdown(minimal_result)
        assert "## 4. Telecoupling Analysis" in md

    def test_includes_flows_table(self, minimal_result):
        md = render_markdown(minimal_result)
        assert "## Flows" in md
        # Table header row + at least one matter row.
        assert "| Category | Source | Target | Bidirectional |" in md
        assert "| Matter | MEX | USA |" in md

    def test_includes_web_sources_with_w_ids(self, minimal_result):
        md = render_markdown(minimal_result)
        assert "## Web Sources" in md
        assert "[W1]" in md
        assert "https://example.com/mex-exports-2024" in md

    def test_includes_research_gaps_when_present(self, minimal_result):
        md = render_markdown(minimal_result)
        assert "## 6. Research Gaps" in md
        assert "Jalisco" in md

    def test_skips_empty_sections(self):
        # No abstract, no classification, no sections, no flows.
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="", raw="", turn_number=1,
        )
        md = render_markdown(result)
        assert "## Abstract" not in md
        assert "## 1. Coupling Classification" not in md
        assert "## Flows" not in md
        # The title is always present.
        assert md.startswith("# Metacoupling Analysis")

    def test_writes_file_when_path_given(self, minimal_result, tmp_path):
        out_path = tmp_path / "subdir" / "report.md"
        text = render_markdown(minimal_result, path=out_path)
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == text

    def test_returns_markdown_without_writing_when_path_none(
        self, minimal_result, tmp_path,
    ):
        # No path argument → no file should exist after the call.
        before = list(tmp_path.iterdir())
        text = render_markdown(minimal_result)
        after = list(tmp_path.iterdir())
        assert before == after
        assert "## Abstract" in text

    def test_to_markdown_method_delegates(self, minimal_result):
        md_direct = render_markdown(minimal_result)
        md_method = minimal_result.to_markdown()
        assert md_direct == md_method

    # PR #31 follow-up: Evidence from Literature rendering.

    def test_includes_rag_evidence_section_when_hits_present(
        self, minimal_result,
    ):
        md = render_markdown(minimal_result)
        assert "## Evidence from Literature" in md
        # Each hit gets its own ### heading with the turn-scoped ID.
        assert "### [T1:1]" in md
        assert "### [T1:2]" in md
        # Paper title visible.
        assert "Mexico's avocado industry and global trade" in md
        # Author/year line.
        assert "Garcia, M. and Hernandez, L." in md
        assert "2023" in md
        # Excerpt rendered as a blockquote (> prefix).
        assert "> Jalisco's accession" in md

    def test_skips_rag_section_when_no_hits(self):
        result = AnalysisResult(
            parsed=ParsedAnalysis(),
            formatted="", raw="", turn_number=1,
        )
        md = render_markdown(result)
        assert "## Evidence from Literature" not in md


# ---------------------------------------------------------------------------
# Word (.docx) renderer
# ---------------------------------------------------------------------------


# Skip the whole class when python-docx is unavailable.
docx = pytest.importorskip("docx")


class TestRenderDocx:
    def test_produces_file_at_given_path(self, minimal_result, tmp_path):
        out = tmp_path / "report.docx"
        path = render_docx(minimal_result, path=out)
        assert path == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_default_path_when_none_given(self, minimal_result, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = render_docx(minimal_result)
        assert path.name == "metacoupling_report.docx"
        assert path.exists()

    def test_contains_expected_headings(self, minimal_result, tmp_path):
        from docx import Document

        out = tmp_path / "report.docx"
        render_docx(minimal_result, path=out)
        doc = Document(str(out))
        # Collect every heading-style paragraph's text.
        headings = [
            p.text for p in doc.paragraphs
            if p.style.name.startswith("Heading")
        ]
        assert any("Abstract" in h for h in headings)
        assert any("1. Coupling Classification" in h for h in headings)
        assert any("4. Telecoupling Analysis" in h for h in headings)

    def test_contains_flows_table(self, minimal_result, tmp_path):
        from docx import Document

        out = tmp_path / "report.docx"
        render_docx(minimal_result, path=out)
        doc = Document(str(out))
        # At least one table should exist for flows.
        assert doc.tables, "no tables in document"
        # Find a table with the expected header row.
        flows_table = None
        for table in doc.tables:
            header_cells = [cell.text for cell in table.rows[0].cells]
            if (
                "Category" in header_cells
                and "Source" in header_cells
                and "Target" in header_cells
            ):
                flows_table = table
                break
        assert flows_table is not None
        # At least one data row beyond the header.
        assert len(flows_table.rows) >= 2

    def test_contains_web_sources_table(self, minimal_result, tmp_path):
        from docx import Document

        out = tmp_path / "report.docx"
        render_docx(minimal_result, path=out)
        doc = Document(str(out))
        # Look for a table whose header includes ID + URL.
        found = False
        for table in doc.tables:
            header_cells = [cell.text for cell in table.rows[0].cells]
            if "ID" in header_cells and "URL" in header_cells:
                found = True
                break
        assert found

    def test_to_docx_method_delegates(self, minimal_result, tmp_path):
        out = tmp_path / "method_call.docx"
        path = minimal_result.to_docx(out)
        assert path == out
        assert out.exists()

    # PR #31 follow-up: RAG evidence rendering in docx.

    def test_contains_rag_evidence_headings(self, minimal_result, tmp_path):
        from docx import Document

        out = tmp_path / "rag.docx"
        render_docx(minimal_result, path=out)
        doc = Document(str(out))
        headings = [
            p.text for p in doc.paragraphs
            if p.style.name.startswith("Heading")
        ]
        # Section heading + per-hit subheading.
        assert any("Evidence from Literature" in h for h in headings)
        assert any("[T1:1]" in h for h in headings)
        assert any("Mexico's avocado industry" in h for h in headings)

    def test_rag_excerpt_text_present_in_docx(self, minimal_result, tmp_path):
        from docx import Document

        out = tmp_path / "rag_text.docx"
        render_docx(minimal_result, path=out)
        doc = Document(str(out))
        all_text = "\n".join(p.text for p in doc.paragraphs)
        assert "Jalisco's accession" in all_text
        # Author/year line rendered.
        assert "Garcia, M. and Hernandez, L." in all_text
