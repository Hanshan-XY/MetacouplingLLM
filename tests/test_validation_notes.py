"""The coupling-database validation notes claim only what was checked.

After each analysis the country validator
(``parsed.country_pericoupling_info``) and, when the research description
names a subnational region, the ADM1 validator
(``parsed.pericoupling_info``) compare the LLM's classification with the
pairs they classified against the bundled adjacency databases.  A
validator that classified no pair notes that nothing was checked instead
of calling the classification consistent, and the report's Note shows
such a note only when no validator with a note classified a pair.  Every
note of a validator that classified a pair is unchanged.
"""

from __future__ import annotations

import pytest

from metacouplingllm.core import MetacouplingAssistant
from metacouplingllm.llm.client import LLMResponse
from metacouplingllm.llm.parser import ParsedAnalysis
from metacouplingllm.output.formatter import AnalysisFormatter

from ._helpers import make_parsed_analysis

COUNTRY_UNCHECKED = (
    "No partner country was checked against the coupling database "
    "(partners in the spillover role are not checked)."
)
ADM1_UNCHECKED = (
    "No subnational region other than the focal region was checked "
    "against the ADM1 pericoupling database."
)

# The notes of a validator that classified a pair, as before.
COUNTRY_CONSISTENT = (
    "LLM classification is consistent with the coupling database."
)
COUNTRY_PERI_REVISE = (
    "The coupling database indicates at least one pericoupled country "
    "pair, but the LLM classified this study differently. Consider "
    "revising."
)
COUNTRY_TELE_REVISE = (
    "The coupling database indicates all detected pairs are telecoupled, "
    "but the LLM classified this study differently. Consider revising."
)
ADM1_CONSISTENT = (
    "LLM classification is consistent with the ADM1 pericoupling database."
)
ADM1_PERI_REVISE = (
    "The ADM1 pericoupling database indicates at least one mentioned "
    "subnational region is adjacent to the focal region, but the LLM did "
    "not classify this study as pericoupling.  Consider revising."
)
ADM1_TELE_REVISE = (
    "The ADM1 pericoupling database indicates all mentioned subnational "
    "regions are non-adjacent (telecoupled) to the focal region, but the "
    "LLM did not classify this study as telecoupling.  Consider revising."
)
ADM1_NO_CLASSIFICATION = (
    "ADM1 pericoupling database returned neighbour information for the "
    "focal region (no LLM classification to cross-check)."
)
BOTH_CONSISTENT = (
    "Note: LLM classification is consistent with the coupling databases "
    "(country adjacency + ADM1 adjacency)."
)


def _validation_block(formatted: str) -> str:
    _, found, block = formatted.partition("COUPLING DATABASE VALIDATION\n")
    assert found, "no validation block in the report"
    return block


def _note_lines(parsed: ParsedAnalysis) -> list[str]:
    """Return the Note lines of the report's validation block."""
    block = _validation_block(AnalysisFormatter.format_full(parsed))
    return [
        line.strip()
        for line in block.splitlines()
        if line.strip().startswith("Note")
    ]


# ---------------------------------------------------------------------------
# Country validator
# ---------------------------------------------------------------------------


def _thailand_parsed(classification: str) -> ParsedAnalysis:
    """Every partner country is in the spillover role."""
    return make_parsed_analysis(
        coupling_classification=classification,
        systems={
            "sending": {"name": "Thailand"},
            "spillover": {"name": "Nigeria and China"},
        },
    )


class TestCountryValidator:
    def test_premise_every_partner_is_spillover(self):
        parsed = _thailand_parsed("Telecoupling with spillover effects.")
        assert MetacouplingAssistant._extract_countries_with_roles(
            parsed
        ) == {"THA": "sending", "NGA": "spillover", "CHN": "spillover"}

    @pytest.mark.parametrize(
        "national_mode", [False, True], ids=["subnational", "national"],
    )
    def test_no_classified_pair_is_not_called_consistent(self, national_mode):
        parsed = _thailand_parsed("Telecoupling with spillover effects.")
        MetacouplingAssistant._validate_country_pericoupling(
            parsed, national_mode=national_mode,
        )
        info = parsed.country_pericoupling_info
        assert info is not None
        assert info["focal_country"] == "Thailand (THA)"
        assert "pair_results" not in info
        assert info["note"] == COUNTRY_UNCHECKED
        assert _note_lines(parsed) == [f"Note: {COUNTRY_UNCHECKED}"]
        assert "consistent" not in _validation_block(
            AnalysisFormatter.format_full(parsed)
        )

    def test_no_classification_still_gives_no_note(self):
        parsed = _thailand_parsed("")
        MetacouplingAssistant._validate_country_pericoupling(parsed)
        info = parsed.country_pericoupling_info
        assert info is not None
        assert "note" not in info
        assert _note_lines(parsed) == []

    @pytest.mark.parametrize(
        ("classification", "systems", "pairs", "note"),
        [
            (
                "Pericoupling between Mexico and the United States.",
                {"sending": {"name": "Mexico"},
                 "receiving": {"name": "United States"}},
                "Mexico (MEX) ↔ United States (USA): PERICOUPLED",
                COUNTRY_CONSISTENT,
            ),
            (
                "Telecoupling between Mexico and distant markets.",
                {"sending": {"name": "Mexico"},
                 "receiving": {"name": "United States"}},
                "Mexico (MEX) ↔ United States (USA): PERICOUPLED",
                COUNTRY_PERI_REVISE,
            ),
            (
                "Telecoupling between Brazil and China.",
                {"sending": {"name": "Brazil"},
                 "receiving": {"name": "China"}},
                "Brazil (BRA) ↔ China (CHN): TELECOUPLED",
                COUNTRY_CONSISTENT,
            ),
            (
                "Pericoupling between Brazil and China.",
                {"sending": {"name": "Brazil"},
                 "receiving": {"name": "China"}},
                "Brazil (BRA) ↔ China (CHN): TELECOUPLED",
                COUNTRY_TELE_REVISE,
            ),
            (
                # A spillover partner beside a classified one is skipped
                # as before; the note covers the classified pair.
                "Pericoupling with spillover effects on Chile and Peru.",
                {"sending": {"name": "Mexico"},
                 "receiving": {"name": "United States"},
                 "spillover": {"name": "Chile and Peru"}},
                "Mexico (MEX) ↔ United States (USA): PERICOUPLED",
                COUNTRY_CONSISTENT,
            ),
        ],
        ids=[
            "pericoupled-consistent",
            "pericoupled-revise",
            "telecoupled-consistent",
            "telecoupled-revise",
            "spillover-beside-classified",
        ],
    )
    def test_classified_pairs_keep_their_note(
        self, classification, systems, pairs, note,
    ):
        parsed = make_parsed_analysis(
            coupling_classification=classification, systems=systems,
        )
        MetacouplingAssistant._validate_country_pericoupling(
            parsed, national_mode=True,
        )
        info = parsed.country_pericoupling_info
        assert info["pair_results"] == pairs
        assert info["note"] == note


# ---------------------------------------------------------------------------
# ADM1 validator
# ---------------------------------------------------------------------------


def _michigan_parsed(classification: str, direction: str | None = None):
    """Michigan focal; a flow direction names the partner, if any."""
    flows = None
    if direction:
        flows = [{
            "category": "matter",
            "direction": direction,
            "description": "trade",
        }]
    return make_parsed_analysis(
        coupling_classification=classification,
        systems={
            "sending": {
                "name": "Michigan",
                "geographic_scope": "Michigan, United States",
            },
        },
        flows=flows,
    )


def _jalisco_parsed(michoacan_role: str) -> ParsedAnalysis:
    """Jalisco focal; Michoacán named in a flow, in the given role."""
    return make_parsed_analysis(
        coupling_classification="Pericoupling between Jalisco and Michoacán.",
        systems={
            "sending": {
                "name": "Jalisco avocado",
                "geographic_scope": "Jalisco, Mexico",
            },
            michoacan_role: {"name": "Michoacan de Ocampo"},
        },
        flows=[{
            "category": "matter",
            "direction": "Michoacán de Ocampo → Jalisco",
            "description": "inputs flow",
        }],
    )


class TestAdm1Validator:
    def test_no_other_region_is_not_called_consistent(self):
        """Ohio is named only in the classification, which the region
        scan does not read."""
        parsed = _michigan_parsed("Pericoupling between Michigan and Ohio.")
        assert MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        info = parsed.pericoupling_info
        assert info["focal_region"] == "Michigan (USA023)"
        assert "pair_results" not in info
        assert info["note"] == ADM1_UNCHECKED
        assert _note_lines(parsed) == [f"Note: {ADM1_UNCHECKED}"]

    def test_spillover_only_region_is_not_called_consistent(self):
        parsed = _jalisco_parsed("spillover")
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        info = parsed.pericoupling_info
        assert info["focal_region"] == "Jalisco (MEX014)"
        assert "pair_results" not in info
        assert info["note"] == ADM1_UNCHECKED

    def test_the_same_region_outside_spillover_keeps_its_note(self):
        parsed = _jalisco_parsed("receiving")
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        info = parsed.pericoupling_info
        assert info["pair_results"] == (
            "Jalisco (MEX014) ↔ Michoacán de Ocampo (MEX016): PERICOUPLED"
        )
        assert info["note"] == ADM1_CONSISTENT

    @pytest.mark.parametrize(
        "direction", [None, "Michigan → Ohio"], ids=["no-region", "ohio"],
    )
    def test_no_classification_note_is_unchanged(self, direction):
        parsed = _michigan_parsed("", direction)
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        assert parsed.pericoupling_info["note"] == ADM1_NO_CLASSIFICATION

    @pytest.mark.parametrize(
        ("classification", "direction", "pair", "note"),
        [
            (
                "Pericoupling between Michigan and Ohio in the United States",
                "Michigan → Ohio",
                "Michigan (USA023) ↔ Ohio (USA036): PERICOUPLED",
                ADM1_CONSISTENT,
            ),
            (
                "Telecoupling: Michigan in United States exports to "
                "international markets",
                "Michigan → Ohio",
                "Michigan (USA023) ↔ Ohio (USA036): PERICOUPLED",
                ADM1_PERI_REVISE,
            ),
            (
                "Telecoupling between Michigan, United States and "
                "Mato Grosso, Brazil",
                "Michigan → Mato Grosso",
                "Michigan (USA023) ↔ Mato Grosso (BRA011): TELECOUPLED",
                ADM1_CONSISTENT,
            ),
            (
                "Pericoupling between Michigan, United States and "
                "Mato Grosso, Brazil",
                "Michigan → Mato Grosso",
                "Michigan (USA023) ↔ Mato Grosso (BRA011): TELECOUPLED",
                ADM1_TELE_REVISE,
            ),
        ],
        ids=[
            "pericoupled-consistent",
            "pericoupled-revise",
            "telecoupled-consistent",
            "telecoupled-revise",
        ],
    )
    def test_classified_regions_keep_their_note(
        self, classification, direction, pair, note,
    ):
        parsed = _michigan_parsed(classification, direction)
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        info = parsed.pericoupling_info
        assert info["pair_results"] == pair
        assert info["note"] == note


# ---------------------------------------------------------------------------
# A whole analysis: acid rain from Ohio on Ontario
# ---------------------------------------------------------------------------

# Ohio and Ontario meet only across Lake Erie, with no fixed crossing, so
# they are adjacent under "lenient" only (tests/test_adjacency_settings.py
# uses the same case).
OHIO_ACID_RAIN = (
    "Acid rain from coal-fired power plants in Ohio, United States, "
    "falling on lakes in Ontario, Canada."
)

# The analysis names Ontario only in a system's name and scope, which the
# region scan skips, so the ADM1 validator classifies no region.
OHIO_ANALYSIS = """\
### 1. Coupling Classification
Pericoupling: sulfur emissions from Ohio fall as acid rain on Ontario.

### 3. Pericoupling Analysis
#### 3.1 Systems Identification
**Sending System (adjacent)**: Ohio Coal Power System
- **Geographic scope**: Ohio, United States
**Receiving System (adjacent)**: Ontario Lake Ecosystems
- **Geographic scope**: Ontario, Canada
"""

# A flow that names Ontario, which the region scan reads.
OHIO_FLOW = """
#### 3.2 Flows Analysis
- [Matter] Ohio -> Ontario: Sulfur dioxide carried by wind.
"""


class _Client:
    def __init__(self, content: str) -> None:
        self._content = content

    def chat(self, messages, temperature=0.7, max_tokens=None):
        return LLMResponse(content=self._content)


def _analyze(analysis: str, standard: str) -> ParsedAnalysis:
    advisor = MetacouplingAssistant(
        _Client(analysis),
        generate_abstract=False,
        coupling_standard=standard,
    )
    result = advisor.analyze(OHIO_ACID_RAIN)
    assert result.formatted == AnalysisFormatter.format_full(result.parsed)
    return result.parsed


class TestOhioAcidRain:
    @pytest.mark.parametrize(
        ("standard", "verdict", "country_note", "notes"),
        [
            ("stringent", "TELECOUPLED", COUNTRY_TELE_REVISE,
             [f"Note (country): {COUNTRY_TELE_REVISE}"]),
            ("moderate", "TELECOUPLED", COUNTRY_TELE_REVISE,
             [f"Note (country): {COUNTRY_TELE_REVISE}"]),
            ("lenient", "PERICOUPLED", COUNTRY_CONSISTENT,
             [f"Note: {COUNTRY_CONSISTENT}"]),
        ],
        ids=["stringent", "moderate", "lenient"],
    )
    def test_adm1_validator_classifies_no_region(
        self, standard, verdict, country_note, notes,
    ):
        """Under "moderate" the country note says "Consider revising";
        the ADM1 note used to call the classification consistent."""
        parsed = _analyze(OHIO_ANALYSIS, standard)
        adm1 = parsed.pericoupling_info
        assert adm1["focal_region"] == "Ohio (USA036)"
        assert "pair_results" not in adm1
        assert adm1["note"] == ADM1_UNCHECKED
        country = parsed.country_pericoupling_info
        assert country["pair_results"] == (
            f"Ohio (USA036) ↔ Canada (CAN): {verdict}"
        )
        assert country["note"] == country_note
        assert _note_lines(parsed) == notes

    @pytest.mark.parametrize(
        ("standard", "verdict", "country_note", "adm1_note", "notes"),
        [
            ("moderate", "TELECOUPLED", COUNTRY_TELE_REVISE,
             ADM1_TELE_REVISE,
             [f"Note (country): {COUNTRY_TELE_REVISE}",
              f"Note (subnational): {ADM1_TELE_REVISE}"]),
            ("lenient", "PERICOUPLED", COUNTRY_CONSISTENT,
             ADM1_CONSISTENT,
             [BOTH_CONSISTENT]),
        ],
        ids=["moderate", "lenient"],
    )
    def test_a_flow_naming_ontario_keeps_both_notes(
        self, standard, verdict, country_note, adm1_note, notes,
    ):
        parsed = _analyze(OHIO_ANALYSIS + OHIO_FLOW, standard)
        adm1 = parsed.pericoupling_info
        assert adm1["pair_results"] == (
            f"Ohio (USA036) ↔ Ontario (CAN009): {verdict}"
        )
        assert adm1["note"] == adm1_note
        country = parsed.country_pericoupling_info
        assert country["pair_results"] == (
            f"Ohio (USA036) ↔ Canada (CAN): {verdict}"
        )
        assert country["note"] == country_note
        assert _note_lines(parsed) == notes


# ---------------------------------------------------------------------------
# The report's Note
# ---------------------------------------------------------------------------

_COUNTRY_PAIR = "Ohio (USA036) ↔ Canada (CAN): TELECOUPLED"
_ADM1_PAIR = "Ohio (USA036) ↔ Michigan (USA023): PERICOUPLED"


def _report(country: dict | None, adm1: dict | None) -> ParsedAnalysis:
    if adm1 is not None:
        adm1 = {"level": "adm1", "focal_region": "Ohio (USA036)", **adm1}
    if country is not None:
        country = {"focal_country": "United States (USA)", **country}
    return ParsedAnalysis(
        coupling_classification="Pericoupling and telecoupling.",
        country_pericoupling_info=country,
        pericoupling_info=adm1,
    )


class TestReportNote:
    @pytest.mark.parametrize(
        ("country", "adm1", "notes"),
        [
            (
                {"pair_results": _COUNTRY_PAIR, "note": COUNTRY_CONSISTENT},
                {"note": ADM1_UNCHECKED},
                [f"Note: {COUNTRY_CONSISTENT}"],
            ),
            (
                {"pair_results": _COUNTRY_PAIR, "note": COUNTRY_TELE_REVISE},
                {"note": ADM1_UNCHECKED},
                [f"Note (country): {COUNTRY_TELE_REVISE}"],
            ),
            (
                {"note": COUNTRY_UNCHECKED},
                {"pair_results": _ADM1_PAIR, "note": ADM1_CONSISTENT},
                [f"Note: {ADM1_CONSISTENT}"],
            ),
            (
                {"note": COUNTRY_UNCHECKED},
                {"pair_results": _ADM1_PAIR, "note": ADM1_PERI_REVISE},
                [f"Note (subnational): {ADM1_PERI_REVISE}"],
            ),
            (
                {"note": COUNTRY_UNCHECKED},
                {"note": ADM1_UNCHECKED},
                [f"Note (country): {COUNTRY_UNCHECKED}",
                 f"Note (subnational): {ADM1_UNCHECKED}"],
            ),
            (
                {"note": COUNTRY_UNCHECKED},
                None,
                [f"Note: {COUNTRY_UNCHECKED}"],
            ),
        ],
        ids=[
            "country-checked",
            "country-checked-revise",
            "region-checked",
            "region-checked-revise",
            "nothing-checked",
            "country-only-nothing-checked",
        ],
    )
    def test_unchecked_notes_show_only_when_nothing_was_checked(
        self, country, adm1, notes,
    ):
        assert _note_lines(_report(country, adm1)) == notes

    @pytest.mark.parametrize(
        ("country", "adm1", "notes"),
        [
            (
                {"pair_results": _COUNTRY_PAIR, "note": COUNTRY_CONSISTENT},
                {"pair_results": _ADM1_PAIR, "note": ADM1_CONSISTENT},
                [BOTH_CONSISTENT],
            ),
            (
                {"pair_results": _COUNTRY_PAIR, "note": COUNTRY_CONSISTENT},
                {"pair_results": _ADM1_PAIR, "note": ADM1_PERI_REVISE},
                [f"Note (country): {COUNTRY_CONSISTENT}",
                 f"Note (subnational): {ADM1_PERI_REVISE}"],
            ),
            (
                {"pair_results": _COUNTRY_PAIR, "note": COUNTRY_CONSISTENT},
                None,
                [f"Note: {COUNTRY_CONSISTENT}"],
            ),
            (
                {"pair_results": _COUNTRY_PAIR, "note": COUNTRY_TELE_REVISE},
                None,
                [f"Note (country): {COUNTRY_TELE_REVISE}"],
            ),
            (
                None,
                {"pair_results": _ADM1_PAIR, "note": ADM1_CONSISTENT},
                [f"Note: {ADM1_CONSISTENT}"],
            ),
            (
                # No classification: the country validator gives no note.
                {"pair_results": _COUNTRY_PAIR},
                {"note": ADM1_NO_CLASSIFICATION},
                [f"Note: {ADM1_NO_CLASSIFICATION}"],
            ),
        ],
        ids=[
            "both-consistent",
            "both-one-revise",
            "country-consistent",
            "country-revise",
            "region-consistent",
            "no-classification",
        ],
    )
    def test_notes_of_classified_pairs_are_unchanged(
        self, country, adm1, notes,
    ):
        assert _note_lines(_report(country, adm1)) == notes
