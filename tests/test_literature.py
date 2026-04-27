"""Tests for knowledge/literature.py — literature recommendation engine."""

from metacouplingllm.knowledge.literature import (
    Paper,
    _extract_search_terms_from_text,
    _is_relevant,
    _parse_bibtex,
    _score_paper,
    format_recommendations,
    get_database_info,
    recommend_papers,
)


# ---------------------------------------------------------------------------
# BibTeX parsing
# ---------------------------------------------------------------------------


_SAMPLE_BIB = """
@article{liu_telecoupling_2013,
    title = {Framing Sustainability in a Telecoupled World},
    author = {Liu, Jianguo and Hull, Vanessa and Batistella, Mateus},
    year = {2013},
    journal = {ECOLOGY AND SOCIETY},
    keywords = {telecoupling, sustainability, coupled human and natural systems},
    doi = {10.5751/ES-05873-180226},
    annote = {Cited by: 500},
}

@article{smith_random_2020,
    title = {Random Paper About Terahertz Physics},
    author = {Smith, John},
    year = {2020},
    journal = {PHYSICS LETTERS},
    keywords = {terahertz, metasurface, photonics},
    annote = {Cited by: 5},
}

@article{hull_metacoupling_2015,
    title = {Metacoupling and Panda Conservation},
    author = {Hull, Vanessa and Liu, Jianguo},
    year = {2015},
    journal = {CONSERVATION BIOLOGY},
    keywords = {metacoupling, panda, conservation, China, CHANS},
    doi = {10.1111/cobi.12345},
    annote = {Cited by: 150},
}
"""


class TestParseBibtex:
    def test_parses_entries(self):
        papers = _parse_bibtex(_SAMPLE_BIB)
        assert len(papers) == 3

    def test_parses_title(self):
        papers = _parse_bibtex(_SAMPLE_BIB)
        assert "Telecoupled World" in papers[0].title

    def test_parses_authors(self):
        papers = _parse_bibtex(_SAMPLE_BIB)
        assert "Liu, Jianguo" in papers[0].authors

    def test_parses_year(self):
        papers = _parse_bibtex(_SAMPLE_BIB)
        assert papers[0].year == 2013

    def test_parses_keywords(self):
        papers = _parse_bibtex(_SAMPLE_BIB)
        assert "telecoupling" in papers[0].keywords
        assert "sustainability" in papers[0].keywords

    def test_parses_citation_count(self):
        papers = _parse_bibtex(_SAMPLE_BIB)
        assert papers[0].cited_by == 500

    def test_parses_doi(self):
        papers = _parse_bibtex(_SAMPLE_BIB)
        assert papers[0].doi == "10.5751/ES-05873-180226"


class TestRelevanceFilter:
    def test_telecoupling_paper_is_relevant(self):
        paper = Paper(
            title="Telecoupling Framework",
            keywords={"telecoupling", "sustainability"},
        )
        assert _is_relevant(paper) is True

    def test_random_paper_is_not_relevant(self):
        paper = Paper(
            title="Terahertz Physics",
            keywords={"terahertz", "photonics"},
        )
        assert _is_relevant(paper) is False

    def test_metacoupling_is_relevant(self):
        paper = Paper(
            title="Metacoupling study",
            keywords={"metacoupling"},
        )
        assert _is_relevant(paper) is True

    def test_chans_is_relevant(self):
        paper = Paper(
            title="Coupled systems",
            keywords={"chans", "land use"},
        )
        assert _is_relevant(paper) is True


class TestSearchTermExtraction:
    def test_extracts_from_text(self):
        terms = _extract_search_terms_from_text(
            "soybean trade Brazil China deforestation"
        )
        assert "soybean" in terms
        assert "trade" in terms
        assert "brazil" in terms
        assert "china" in terms
        assert "deforestation" in terms

    def test_removes_stopwords(self):
        terms = _extract_search_terms_from_text("this is a study about trade")
        assert "this" not in terms
        assert "study" not in terms
        assert "trade" in terms

    def test_short_words_excluded(self):
        terms = _extract_search_terms_from_text("US EU and or")
        # Words <= 3 chars are excluded
        assert "and" not in terms


class TestScoring:
    def test_keyword_match_scores_high(self):
        paper = Paper(keywords={"trade", "deforestation"})
        score = _score_paper(paper, {"trade"})
        assert score >= 3.0

    def test_title_match_scores(self):
        paper = Paper(title="Soybean trade and deforestation")
        score = _score_paper(paper, {"soybean", "trade"})
        assert score >= 4.0  # 2 title matches

    def test_no_match_scores_zero(self):
        paper = Paper(title="Unrelated topic", keywords={"physics"})
        score = _score_paper(paper, {"soybean", "trade"})
        assert score == 0.0

    def test_citations_add_bonus(self):
        paper1 = Paper(keywords={"trade"}, cited_by=0)
        paper2 = Paper(keywords={"trade"}, cited_by=100)
        s1 = _score_paper(paper1, {"trade"})
        s2 = _score_paper(paper2, {"trade"})
        assert s2 > s1


class TestRecommendPapers:
    def test_returns_list(self):
        result = recommend_papers("telecoupling trade")
        assert isinstance(result, list)

    def test_respects_max_results(self):
        result = recommend_papers("telecoupling framework", max_results=3)
        assert len(result) <= 3

    def test_returns_paper_objects(self):
        result = recommend_papers("telecoupling")
        for p in result:
            assert isinstance(p, Paper)

    def test_results_have_titles(self):
        result = recommend_papers("telecoupling framework sustainability")
        if result:
            assert all(p.title for p in result)

    def test_from_parsed_analysis(self):
        from metacouplingllm.llm.parser import CouplingSection, ParsedAnalysis

        analysis = ParsedAnalysis(
            coupling_classification="telecoupling",
            telecoupling=CouplingSection(
                systems=[
                    {"role": "sending", "name": "Brazil"},
                    {"role": "receiving", "name": "China"},
                ],
                flows=[{"category": "matter", "description": "Soybeans"}],
                causes={"socioeconomic": ["demand"]},
            ),
        )
        result = recommend_papers(analysis, max_results=5)
        assert isinstance(result, list)


class TestFormatRecommendations:
    def test_empty_list(self):
        text = format_recommendations([])
        assert "No relevant papers" in text

    def test_formats_papers(self):
        papers = [
            Paper(
                title="Test Paper",
                authors="Author A and Author B",
                year=2023,
                journal="TEST JOURNAL",
                doi="10.1234/test",
                cited_by=42,
            )
        ]
        text = format_recommendations(papers)
        assert "Test Paper" in text
        assert "2023" in text
        assert "TEST JOURNAL" in text
        assert "10.1234/test" in text
        assert "42" in text
        assert "RECOMMENDED LITERATURE" in text


class TestGetDatabaseInfo:
    def test_returns_dict(self):
        info = get_database_info()
        assert isinstance(info, dict)
        assert "total_papers" in info

    def test_has_papers(self):
        info = get_database_info()
        assert info["total_papers"] > 0
