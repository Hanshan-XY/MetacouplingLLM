"""Tests for knowledge/literature.py — literature recommendation engine."""

import re
import unicodedata

import pytest

from metacouplingllm.knowledge.literature import (
    Paper,
    _build_fulltext_scores,
    _extract_search_terms_from_text,
    _get_database,
    _is_relevant,
    _parse_bibtex,
    _score_paper,
    format_recommendations,
    get_database_info,
    recommend_papers,
)
from metacouplingllm.knowledge.rag import RAGEngine, RetrievalResult, TextChunk


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


class TestBuildFulltextScores:
    """A paper's full-text score is the best score among its chunks.

    ``RAGEngine.retrieve`` can return several chunks of one paper (up to
    ``max_chunks_per_paper``, default 3). It lists a paper's best chunk
    first and appends chunks of sections already taken after the first
    pass, so the last chunk listed for a paper is not its best.
    """

    @pytest.fixture
    def papers_dir(self, monkeypatch, tmp_path):
        """Point the engine at tmp_path instead of the bundled papers."""
        monkeypatch.setattr(
            "metacouplingllm.knowledge.rag._extract_bundled_papers",
            lambda verbose=False: tmp_path,
        )
        return tmp_path

    def test_keeps_best_chunk_score_per_paper(self, monkeypatch, papers_dir):
        def hit(key, section, score):
            return RetrievalResult(
                chunk=TextChunk(paper_key=key, section=section), score=score,
            )

        # The order the two-pass selection yields: paper A's Introduction
        # (0.9) and Discussion (0.3) in the first pass, then its second
        # Introduction chunk (0.7) in the second.
        hits = [
            hit("A", "Introduction", 0.9),
            hit("B", "Introduction", 0.5),
            hit("A", "Discussion", 0.3),
            hit("A", "Introduction", 0.7),
        ]
        monkeypatch.setattr(RAGEngine, "load", lambda self: None)
        monkeypatch.setattr(
            RAGEngine, "retrieve", lambda self, *args, **kwargs: hits,
        )

        assert _build_fulltext_scores({"avocado", "trade"}) == {
            "A": 0.9,
            "B": 0.5,
        }

    def test_tfidf_paper_scored_by_its_best_chunk(
        self, monkeypatch, papers_dir,
    ):
        filler = (
            "Land systems respond to distant demand through flows of "
            "goods, capital and information that cross administrative "
            "borders. "
        )
        # Three sections that match the query less and less, and a
        # second paper that matches it weakly. Dated 2099 so neither
        # file matches a bundled BibTeX entry: their keys come from
        # the filenames.
        (papers_dir / "Synthetic - 2099 - Avocado trade paper.md").write_text(
            "## Introduction\n\n"
            + "Avocado trade from Mexico grew fast. " * 6 + filler * 2
            + "\n\n## Results\n\n"
            + "Avocado exports rose. " * 2 + filler * 4
            + "\n\n## Discussion\n\n"
            + "Trade matters. " + filler * 5,
            encoding="utf-8",
        )
        (papers_dir / "Synthetic - 2099 - Land systems paper.md").write_text(
            "## Introduction\n\n" + "Mexico " * 3 + filler * 4,
            encoding="utf-8",
        )
        retrieved: list[RetrievalResult] = []
        original_retrieve = RAGEngine.retrieve

        def spy(self, *args, **kwargs):
            hits = original_retrieve(self, *args, **kwargs)
            retrieved.extend(hits)
            return hits

        monkeypatch.setattr(RAGEngine, "retrieve", spy)

        scores = _build_fulltext_scores(
            {"avocado", "trade", "mexico"}, backend="tfidf",
        )

        avocado = "synthetic_2099_avocado_trade_paper"
        land = "synthetic_2099_land_systems_paper"
        chunk_scores = [
            r.score for r in retrieved if r.chunk.paper_key == avocado
        ]
        # All three sections are retrieved and the last one listed is
        # not the best, so keeping the last chunk would under-score it.
        assert len(chunk_scores) == 3
        assert chunk_scores[-1] < max(chunk_scores)
        assert scores[avocado] == max(chunk_scores)
        # Scored by its best chunk, the avocado paper ranks above the
        # paper that barely mentions the query terms.
        assert scores[avocado] > scores[land]


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


# ---------------------------------------------------------------------------
# Bundled bibliography text
# ---------------------------------------------------------------------------

# Publisher markup that metadata exports such as Crossref's leave in
# titles (<i>, <scp>, <sup>), and HTML entities (&amp;, &#8211;, &#x2013;).
_MARKUP_RE = re.compile(r"</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>")
_ENTITY_RE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]*|#[0-9]+|#[xX][0-9A-Fa-f]+);")

# UTF-8 read as Windows-1252: a character whose byte starts a two-, three-
# or four-byte UTF-8 sequence (0xC2-0xDF, 0xE0-0xEF, 0xF0-0xF4), followed
# by as many characters whose bytes are continuation bytes (0x80-0xBF), as
# in "CÃ“RDOBA" for "CÓRDOBA" and "â€¦" for "…".  The five bytes
# Windows-1252 leaves undefined are read as the C1 controls of the same
# number.
_CP1252_CHAR = {
    b: bytes([b]).decode("cp1252", errors="ignore") or chr(b)
    for b in range(0x80, 0x100)
}
_CP1252_BYTE = {ch: b for b, ch in _CP1252_CHAR.items()}
_CONTINUATION = "".join(_CP1252_CHAR[b] for b in range(0x80, 0xC0))
_MOJIBAKE_RE = re.compile(
    f"[\u00c2-\u00df][{_CONTINUATION}]"
    f"|[\u00e0-\u00ef][{_CONTINUATION}]{{2}}"
    f"|[\u00f0-\u00f4][{_CONTINUATION}]{{3}}"
)


def _as_cp1252(text: str) -> str:
    """Return *text*'s UTF-8 bytes read as Windows-1252."""
    return "".join(
        _CP1252_CHAR[b] if b >= 0x80 else chr(b) for b in text.encode("utf-8")
    )


def _mojibake(text: str) -> list[str]:
    """Return the runs of *text* that are UTF-8 read as Windows-1252."""
    runs = []
    for match in _MOJIBAKE_RE.finditer(text):
        try:
            bytes(_CP1252_BYTE[ch] for ch in match.group()).decode("utf-8")
        except UnicodeDecodeError:
            continue
        runs.append(match.group())
    return runs


def _artifacts(text: str) -> list[str]:
    """Return the markup tags, HTML entities and mojibake runs in *text*."""
    return _MARKUP_RE.findall(text) + _ENTITY_RE.findall(text) + _mojibake(text)


class TestBundledBibliographyText:
    """The bundled bibliography holds plain text: RAG citations and
    ``recommend_papers()`` show its titles, authors and journals as they
    are.  Ten titles had markup, entities or mojibake (``<scp>C</scp>hina``,
    ``social&amp;#8211;ecological``, ``CÃ“RDOBA``), fifteen journals had
    ``&amp;``, eleven titles copied from file names had ``_`` for an
    apostrophe or a colon (``China_s``, ``Organic Foods_ Impactos Amb``),
    five entries had no journal, three titles stopped where their files'
    names were cut, nine author fields were an abbreviated list
    (``Wu et al.``) and five gave surnames only, eight DOIs were missing,
    cut short or ended in a stray character, and nineteen keys took "al"
    from "et al." or lost a letter (``lpezhoffman``).
    """

    @pytest.mark.parametrize("field", ["title", "authors", "journal"])
    def test_field_is_plain_text(self, field):
        found = {
            paper.key: _artifacts(getattr(paper, field))
            for paper in _get_database()
            if _artifacts(getattr(paper, field))
        }
        assert found == {}

    def test_titles_have_no_file_name_underscores(self):
        assert [p.key for p in _get_database() if "_" in p.title] == []

    def test_every_entry_names_its_journal(self):
        assert [p.key for p in _get_database() if not p.journal] == []

    def test_titles_cut_at_their_file_names_are_complete(self):
        # These three repeated their bundled files' names, which are cut
        # at about 100 characters ("... A case study of", "... Global So").
        published = {
            "wu_analysis_2023": (
                "Analysis of the proximity and telecoupling mechanism of "
                "cultivated land use change: A case study of Yangtze River "
                "Economic Belt"
            ),
            "bruck_drivers_2023": (
                "Drivers of ecosystem service specialization in a smallholder "
                "agricultural landscape of the Global South: a case study in "
                "Ethiopia"
            ),
            "friis_land_2017": (
                "Land-use change in a telecoupled world: the relevance and "
                "applicability of the telecoupling framework in the case of "
                "banana plantation expansion in Laos"
            ),
        }
        titles = {p.key: p.title for p in _get_database() if p.key in published}
        assert titles == published

    def test_author_fields_list_every_author(self):
        # "Wu et al." as a whole author field also gave every "X et al."
        # file a match bonus against the entry in RAG matching.
        assert [
            p.key for p in _get_database() if re.search(r"\bet al\b", p.authors)
        ] == []
        # Every name has its given name or initials ("Surname, Given").
        assert [
            p.key for p in _get_database()
            if any("," not in name for name in p.authors.split(" and "))
        ] == []

    def test_dois_are_complete(self):
        # Printable ASCII ending in a letter or digit: one DOI was cut at
        # "10.1007/s13280-" and one ended in U+FFFF.
        db = _get_database()
        assert [
            p.key for p in db
            if p.doi
            and not re.fullmatch(r"10\.\d{4,9}/[\x21-\x7e]*[0-9A-Za-z]", p.doi)
        ] == []
        # Every entry has one but Strecker and Veraart 2023 (ICON), for
        # which Crossref has none.
        assert [p.key for p in db if not p.doi] == ["strecker_kenya_2023"]

    def test_keys_start_with_the_first_authors_surname(self):
        # Keys are "surname_word_year": nine took "al" from "et al." and
        # ten lost a letter ("lpezhoffman", "sndergaard").
        letters = {"ø": "o", "æ": "ae", "ß": "ss", "ł": "l", "đ": "d", "œ": "oe"}

        def surname(paper):
            name = paper.authors.split(" and ")[0].split(",")[0].lower()
            name = "".join(letters.get(ch, ch) for ch in name)
            name = unicodedata.normalize("NFKD", name)
            return re.sub(r"[^a-z]", "", name.encode("ascii", "ignore").decode())

        db = _get_database()
        assert [
            p.key for p in db
            if not re.fullmatch(r"[a-z]+_[a-z0-9]+_\d{4}(_[a-z])?", p.key)
        ] == []
        assert [p.key for p in db if not p.key.startswith(surname(p) + "_")] == []

    def test_flags_the_forms_the_fixed_entries_had(self):
        assert _artifacts("dolphinfish (<i>Coryphaena hippurus</i>)") == [
            "<i>", "</i>",
        ]
        assert _artifacts("of the <scp>C</scp>entral") == ["<scp>", "</scp>"]
        assert _artifacts("Globalization<sup>☆</sup>") == ["<sup>", "</sup>"]
        assert _artifacts("social&amp;#8211;ecological") == ["&amp;"]
        assert _artifacts("Ocean &amp; Coastal Management") == ["&amp;"]
        assert _artifacts("(CÃ“RDOBA, ARGENTINA)") == ["Ã“"]
        assert _artifacts("loveâ€¦ agents, 1920â€“2020") == ["â€¦", "â€“"]
        assert _artifacts(_as_cp1252("全程耦合")) == [
            _as_cp1252(ch) for ch in "全程耦合"
        ]
        # Every non-ASCII character in the bibliography, read that way.
        chars = {
            ch
            for paper in _get_database()
            for ch in paper.title + paper.authors + paper.journal
            if ord(ch) > 127
        }
        assert len(chars) > 10
        assert [
            ch for ch in sorted(chars)
            if _mojibake(_as_cp1252(ch)) != [_as_cp1252(ch)]
        ] == []

    def test_spares_accented_and_typographic_characters(self):
        for text in (
            "SÃO PAULO",
            "Ny-Ålesund at 79° North",
            "rice in Bagré",
            "human–nature, 1920–2020",
            "From Kenya with Love… Agents",
            "“Resource-carbon” redistribution",
            "China’s and Austria's",
            "Agriculture Ecosystems & Environment",
            "pH < 7 and > 5",
        ):
            assert _artifacts(text) == []
