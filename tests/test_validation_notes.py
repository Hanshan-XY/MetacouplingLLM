"""The coupling-database validation notes claim only what was checked.

After each analysis the country validator
(``parsed.country_pericoupling_info``) and, when the research description
names a subnational region, the ADM1 validator
(``parsed.pericoupling_info``) compare the LLM's classification with the
pairs they classified against the bundled adjacency databases.  Both read
the classification from the coupling sections the analysis holds, since
its classification text can name a coupling type only to deny it.  A
validator that classified no pair notes
that nothing was checked instead of calling the classification
consistent, and the report's Note shows such a note only when no
validator with a note classified a pair.
"""

from __future__ import annotations

import pytest

from metacouplingllm.core import MetacouplingAssistant
from metacouplingllm.llm.client import LLMResponse
from metacouplingllm.llm.parser import (
    CouplingSection,
    ParsedAnalysis,
    parse_analysis,
)
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
        ("classification", "section", "systems", "pairs", "note"),
        [
            (
                "Pericoupling between Mexico and the United States.",
                "pericoupling",
                {"sending": {"name": "Mexico"},
                 "receiving": {"name": "United States"}},
                "Mexico (MEX) ↔ United States (USA): PERICOUPLED",
                COUNTRY_CONSISTENT,
            ),
            (
                "Telecoupling between Mexico and distant markets.",
                "telecoupling",
                {"sending": {"name": "Mexico"},
                 "receiving": {"name": "United States"}},
                "Mexico (MEX) ↔ United States (USA): PERICOUPLED",
                COUNTRY_PERI_REVISE,
            ),
            (
                "Telecoupling between Brazil and China.",
                "telecoupling",
                {"sending": {"name": "Brazil"},
                 "receiving": {"name": "China"}},
                "Brazil (BRA) ↔ China (CHN): TELECOUPLED",
                COUNTRY_CONSISTENT,
            ),
            (
                "Pericoupling between Brazil and China.",
                "pericoupling",
                {"sending": {"name": "Brazil"},
                 "receiving": {"name": "China"}},
                "Brazil (BRA) ↔ China (CHN): TELECOUPLED",
                COUNTRY_TELE_REVISE,
            ),
            (
                # A spillover partner beside a classified one is skipped;
                # the note covers the classified pair.
                "Pericoupling with spillover effects on Chile and Peru.",
                "pericoupling",
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
        self, classification, section, systems, pairs, note,
    ):
        """Each analysis holds the coupling section its classification
        names, with the systems in it."""
        parsed = make_parsed_analysis(
            coupling_classification=classification, systems=systems,
            coupling_type=section,
        )
        MetacouplingAssistant._validate_country_pericoupling(
            parsed, national_mode=True,
        )
        info = parsed.country_pericoupling_info
        assert info["pair_results"] == pairs
        assert info["note"] == note


ONLY_FOCAL = (
    "No country other than Kenya (KEN) was found in the systems, so no "
    "country pair was checked against the coupling database."
)


def _kenya_parsed(partner: dict[str, str] | None) -> ParsedAnalysis:
    """Kenya's tea estates as the focal system, and *partner*, if any, as
    an adjacent system."""
    return ParsedAnalysis(
        coupling_classification="Pericoupling between the estates and their "
        "partners.",
        intracoupling=CouplingSection(systems=[{
            "role": "focal",
            "name": "Kenya's tea estates",
            "geographic_scope": "Kenya",
        }]),
        pericoupling=CouplingSection(systems=[partner]) if partner else None,
    )


class TestOnlyTheFocalCountry:
    """An analysis with adjacent or distant systems whose systems name no
    country besides the focal one gets a block that says so.  Until
    2026-11-13 the block was left out, which hid partners named in words
    the resolver does not read: a gpt-6.1-sol analysis of the Palk Bay
    named Sri Lanka only as "Sri Lankan", and its block disappeared."""

    @pytest.mark.parametrize(
        "partner",
        [
            {"role": "receiving", "name": "Distant tea buyers"},
            {"role": "sending", "name": "Kenyan smallholder suppliers"},
        ],
        ids=["no-country", "same-country"],
    )
    @pytest.mark.parametrize(
        "national_mode", [False, True], ids=["subnational", "national"],
    )
    def test_partner_systems_that_name_no_other_country(
        self, partner, national_mode,
    ):
        parsed = _kenya_parsed(partner)
        MetacouplingAssistant._validate_country_pericoupling(
            parsed, national_mode=national_mode,
        )
        info = parsed.country_pericoupling_info
        assert info is not None
        assert info["focal_country"] == "Kenya (KEN)"
        assert "pair_results" not in info
        assert info["note"] == ONLY_FOCAL
        block = _validation_block(AnalysisFormatter.format_full(parsed))
        assert "Focal System: Kenya (KEN)" in block
        assert _note_lines(parsed) == [f"Note: {ONLY_FOCAL}"]

    def test_without_partner_systems_there_is_no_block(self):
        parsed = _kenya_parsed(None)
        assert not MetacouplingAssistant._has_partner_systems(parsed)
        MetacouplingAssistant._validate_pericoupling(parsed, national_mode=True)
        assert parsed.country_pericoupling_info is None
        assert "COUPLING DATABASE VALIDATION" not in (
            AnalysisFormatter.format_full(parsed)
        )

    def test_a_focal_entry_is_not_a_partner_system(self):
        """``make_parsed_analysis`` puts the focal system in the
        telecoupling section."""
        parsed = make_parsed_analysis(
            systems={"focal": {"name": "Kenya's tea estates"}},
        )
        assert not MetacouplingAssistant._has_partner_systems(parsed)
        MetacouplingAssistant._validate_country_pericoupling(
            parsed, national_mode=True,
        )
        assert parsed.country_pericoupling_info is None

    def test_regions_checked_by_the_adm1_validator_keep_their_note(self):
        """Jalisco and Michoacán name only Mexico, so the country check
        notes that it checked no pair; the block lists the region pair and
        shows its note alone."""
        parsed = _jalisco_parsed("receiving")
        MetacouplingAssistant._validate_pericoupling(parsed)
        assert parsed.country_pericoupling_info["note"] == ONLY_FOCAL.replace(
            "Kenya (KEN)", "Mexico (MEX)"
        )
        assert parsed.pericoupling_info["pair_results"] == (
            "Jalisco (MEX014) ↔ Michoacán de Ocampo (MEX016): PERICOUPLED"
        )
        assert _note_lines(parsed) == [f"Note: {ADM1_CONSISTENT}"]


# ---------------------------------------------------------------------------
# The country note reads the coupling sections the analysis holds
# ---------------------------------------------------------------------------

# From a gpt-6.1-sol analysis of Indian bottom trawling in the Palk Bay,
# citations left out.  Its classification names telecoupling only to deny
# it, and the analysis has no telecoupling section.
PALK_BAY_ANALYSIS = """\
## 1. Coupling Classification

**The central interaction is pericoupling—not telecoupling—because India \
and Sri Lanka are geographically adjacent across a shared maritime \
boundary.** International interactions are not automatically telecoupled; \
classification depends on system boundaries and adjacency (Liu, 2017).

- **Intracoupling:** Interactions within northern Sri Lanka connect fish \
stocks and habitats with fishing practices, household livelihoods, \
markets, cooperatives, and fisheries governance.
- **Pericoupling:** Indian trawl crews enter adjacent Sri Lankan waters, \
extract resources, and interact with local fishing activities and \
enforcement institutions. These actual cross-border interactions—not \
adjacency alone—establish pericoupling.
- **Telecoupling is a relevant extension but is not established \
sufficiently for a separate analysis here.** Historical research \
identifies shrimp export demand as a driver, but the supplied evidence \
does not trace Palk Bay catches to specific distant markets.

## 2. Intracoupling Analysis (within the focal system)

### 2.1 Systems Identification

**Focal System: Northern Sri Lankan coastal fisheries and fishing \
communities**
- **Geographic scope:** Selected northern coastal communities and their \
associated Sri Lankan fishing grounds, particularly around Jaffna and \
Mannar.

## 3. Pericoupling Analysis (adjacent systems)

### 3.1 Systems Identification

**Sending System (adjacent): Tamil Nadu’s Palk Bay trawl-fishing \
centres—for crew movement**
- **Geographic scope:** Indian Palk Bay ports, particularly Rameswaram \
and Mandapam.

**Receiving System (adjacent): Northern Sri Lankan fishing grounds—for \
entering Indian crews**
- **Geographic scope:** Sri Lankan waters used by the northern \
communities included in the study.
"""

# The mirror case: a classification that names pericoupling only to deny
# it, worded after a gpt-5.5 analysis of Indonesia's coal exports, and no
# pericoupling section, while the database finds Brazil and Argentina
# adjacent.
SOY_ANALYSIS = """\
## 1. Coupling Classification

- **Intracoupling**: Present. Land clearing, farm investment and labour \
interact within Brazil's soybean frontier.
- **Telecoupling**: Present. Brazil exports soybeans to China and \
Argentina.
- **Pericoupling**: Not included as a full section because the research \
description does not provide specific evidence of flows between adjacent \
systems.

## 2. Intracoupling Analysis (within the focal system)

### 2.1 Systems Identification

**Focal System**: Brazil's soybean frontier
- **Geographic scope**: Brazil

## 4. Telecoupling Analysis (distant systems)

### 4.1 Systems Identification

**Sending System**: Brazilian soybean exporters
- **Geographic scope**: Brazil

**Receiving System**: Chinese soybean importers
- **Geographic scope**: China

**Receiving System**: Argentine soybean crushers
- **Geographic scope**: Argentina
"""


class TestTheCountryNoteReadsTheSections:
    """The analysis claims the coupling types whose sections it holds,
    since the prompt has it leave out the section of a type it finds
    absent.  Its classification text can name a type only to deny it
    ("pericoupling—not telecoupling"), so a note that read the text passed
    the denial as agreement."""

    @pytest.mark.parametrize(
        ("sea", "pair", "notes"),
        [
            (False, "Sri Lanka (LKA) ↔ India (IND): TELECOUPLED",
             [f"Note (country): {COUNTRY_TELE_REVISE}"]),
            (True, "Sri Lanka (LKA) ↔ India (IND) (at sea): PERICOUPLED",
             [f"Note: {COUNTRY_CONSISTENT}"]),
        ],
        ids=["maritime-off", "maritime-on"],
    )
    def test_telecoupling_named_only_to_be_denied(self, sea, pair, notes):
        parsed = parse_analysis(PALK_BAY_ANALYSIS)
        assert "not telecoupling" in parsed.coupling_classification
        assert parsed.active_coupling_types() == [
            "intracoupling", "pericoupling",
        ]
        MetacouplingAssistant._validate_pericoupling(
            parsed, national_mode=True, maritime_adjacency=sea,
        )
        info = parsed.country_pericoupling_info
        assert info["focal_country"] == "Sri Lanka (LKA)"
        assert info["pair_results"] == pair
        assert _note_lines(parsed) == notes

    def test_pericoupling_named_only_to_be_denied(self):
        parsed = parse_analysis(SOY_ANALYSIS)
        assert parsed.active_coupling_types() == [
            "intracoupling", "telecoupling",
        ]
        MetacouplingAssistant._validate_pericoupling(
            parsed, national_mode=True,
        )
        info = parsed.country_pericoupling_info
        assert info["focal_country"] == "Brazil (BRA)"
        assert info["pair_results"] == (
            "Brazil (BRA) ↔ China (CHN): TELECOUPLED; "
            "Brazil (BRA) ↔ Argentina (ARG): PERICOUPLED"
        )
        assert info["note"] == COUNTRY_PERI_REVISE

    def test_a_section_claims_its_type_without_the_text_naming_it(self):
        parsed = ParsedAnalysis(
            coupling_classification="Avocado trade between Mexico and the "
            "United States.",
            intracoupling=CouplingSection(systems=[{
                "role": "focal",
                "name": "Mexico's avocado sector",
                "geographic_scope": "Mexico",
            }]),
            pericoupling=CouplingSection(systems=[{
                "role": "receiving",
                "name": "United States avocado importers",
            }]),
        )
        MetacouplingAssistant._validate_country_pericoupling(
            parsed, national_mode=True,
        )
        info = parsed.country_pericoupling_info
        assert info["pair_results"] == (
            "Mexico (MEX) ↔ United States (USA): PERICOUPLED"
        )
        assert info["note"] == COUNTRY_CONSISTENT


# ---------------------------------------------------------------------------
# ADM1 validator
# ---------------------------------------------------------------------------


def _michigan_parsed(
    classification: str,
    direction: str | None = None,
    section: str = "telecoupling",
):
    """Michigan focal; a flow direction names the partner, if any; the
    systems and the flow sit in the coupling section *section*."""
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
        coupling_type=section,
    )


def _jalisco_parsed(michoacan_role: str) -> ParsedAnalysis:
    """Jalisco focal; Michoacán named in a flow, in the given role, in a
    pericoupling section."""
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
        coupling_type="pericoupling",
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
        ("classification", "section", "direction", "pair", "note"),
        [
            (
                "Pericoupling between Michigan and Ohio in the United States",
                "pericoupling",
                "Michigan → Ohio",
                "Michigan (USA023) ↔ Ohio (USA036): PERICOUPLED",
                ADM1_CONSISTENT,
            ),
            (
                "Telecoupling: Michigan in United States exports to "
                "international markets",
                "telecoupling",
                "Michigan → Ohio",
                "Michigan (USA023) ↔ Ohio (USA036): PERICOUPLED",
                ADM1_PERI_REVISE,
            ),
            (
                "Telecoupling between Michigan, United States and "
                "Mato Grosso, Brazil",
                "telecoupling",
                "Michigan → Mato Grosso",
                "Michigan (USA023) ↔ Mato Grosso (BRA011): TELECOUPLED",
                ADM1_CONSISTENT,
            ),
            (
                "Pericoupling between Michigan, United States and "
                "Mato Grosso, Brazil",
                "pericoupling",
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
        self, classification, section, direction, pair, note,
    ):
        """Each analysis holds the coupling section its classification
        names, with the systems and the flow in it."""
        parsed = _michigan_parsed(classification, direction, section)
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
# The ADM1 note reads the coupling sections the analysis holds
# ---------------------------------------------------------------------------

# The Ohio analysis and the flow that names Ontario, with a classification
# that names telecoupling only to deny it.
OHIO_DENIAL = OHIO_ANALYSIS.replace(
    "Pericoupling: sulfur", "Pericoupling, not telecoupling: sulfur",
) + OHIO_FLOW

# An avocado belt that keeps Jalisco inside its focal system: Jalisco is
# named only in the intracoupling section, and the classification sets
# pericoupling aside, while the ADM1 database finds Michoacán and Jalisco
# adjacent.
BELT_ANALYSIS = """\
## 1. Coupling Classification
- **Intracoupling**: Present. The focal system is the avocado belt of \
Michoacán and Jalisco; fruit moving between the two states stays within it.
- **Telecoupling**: Present. The belt exports avocados to the United States.
- **Pericoupling**: Not analysed separately: both states lie within the \
focal system.

## 2. Intracoupling Analysis (within the focal system)
### 2.1 Systems Identification
**Focal System**: Avocado belt of Michoacán, Mexico
- **Human subsystem**: Growers and packers in Michoacán and in Jalisco.
- **Geographic scope**: Michoacán, Mexico

### 2.2 Flows Analysis
**Matter Flow**
- **Direction**: Michoacán orchards → Jalisco packing houses
- **Description**: Fruit moves between the two states of the belt.

## 4. Telecoupling Analysis (distant systems)
### 4.1 Systems Identification
**Sending System**: Avocado belt of Michoacán, Mexico
- **Geographic scope**: Michoacán, Mexico
**Receiving System**: United States avocado importers
- **Geographic scope**: United States
"""


class TestTheAdm1NoteReadsTheSections:
    """As in the country check, the analysis claims the coupling types
    whose sections it holds, whatever its classification text names."""

    @pytest.mark.parametrize(
        ("standard", "verdict", "notes"),
        [
            ("moderate", "TELECOUPLED",
             [f"Note (country): {COUNTRY_TELE_REVISE}",
              f"Note (subnational): {ADM1_TELE_REVISE}"]),
            ("lenient", "PERICOUPLED", [BOTH_CONSISTENT]),
        ],
        ids=["moderate", "lenient"],
    )
    def test_telecoupling_named_only_to_be_denied(
        self, standard, verdict, notes,
    ):
        """Under "moderate" Ohio and Ontario are telecoupled and the
        analysis has no telecoupling section, so both notes ask for a
        revision; the ADM1 note had read "not telecoupling" as agreement,
        against the country note."""
        parsed = parse_analysis(OHIO_DENIAL)
        assert "not telecoupling" in parsed.coupling_classification
        assert parsed.active_coupling_types() == ["pericoupling"]
        MetacouplingAssistant._validate_pericoupling(
            parsed, coupling_standard=standard,
        )
        assert parsed.pericoupling_info["pair_results"] == (
            f"Ohio (USA036) ↔ Ontario (CAN009): {verdict}"
        )
        assert _note_lines(parsed) == notes

    def test_pericoupling_named_only_to_be_set_aside(self):
        """The check compares every region the analysis names besides the
        focal region, Jalisco included, though the analysis keeps it inside
        its own focal system.  Without a pericoupling section the note asks
        for a revision, whatever the classification says."""
        parsed = parse_analysis(BELT_ANALYSIS)
        assert parsed.active_coupling_types() == [
            "intracoupling", "telecoupling",
        ]
        MetacouplingAssistant._validate_pericoupling(parsed)
        adm1 = parsed.pericoupling_info
        assert adm1["focal_region"] == "Michoacán de Ocampo (MEX016)"
        assert adm1["pair_results"] == (
            "Michoacán de Ocampo (MEX016) ↔ Jalisco (MEX014): PERICOUPLED"
        )
        assert _note_lines(parsed) == [
            f"Note (country): {COUNTRY_CONSISTENT}",
            f"Note (subnational): {ADM1_PERI_REVISE}",
        ]

    def test_a_section_claims_its_type_without_the_text_naming_it(self):
        parsed = _michigan_parsed(
            "Coal and freight between Michigan and Ohio in the United "
            "States", "Michigan → Ohio", "pericoupling",
        )
        MetacouplingAssistant._validate_adm1_pericoupling(parsed)
        info = parsed.pericoupling_info
        assert info["pair_results"] == (
            "Michigan (USA023) ↔ Ohio (USA036): PERICOUPLED"
        )
        assert info["note"] == ADM1_CONSISTENT


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
