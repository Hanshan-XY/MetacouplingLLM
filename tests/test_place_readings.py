"""A country's name that a region elsewhere also has ("Georgia", the US state)
is read as the text reads it, everywhere a run resolves names."""

import pytest

from metacouplingllm import MetacouplingAssistant
from metacouplingllm.knowledge.adm1_pericoupling import resolve_adm1_code
from metacouplingllm.knowledge.countries import (
    find_country_codes,
    read_names_as_regions,
    reading_names_as_regions,
    region_read_for,
    resolve_country_code,
)
from metacouplingllm.knowledge.places import place_readings, read_places
from metacouplingllm.llm.client import LLMResponse
from metacouplingllm.visualization.worldmap import (
    _extract_countries_from_analysis,
    _reading_the_analysis_names,
)

GEORGIA_STATE = {"georgia": ("USA011", "USA")}


class TestPlaceReadings:
    @pytest.mark.parametrize("text, readings", [
        ("Peanut farms in Georgia, United States", GEORGIA_STATE),
        ("Cotton from Georgia and Alabama", GEORGIA_STATE),
        ("Gaza and Maputo provinces adjoin South Africa",
         {"gaza": ("MOZ003", "MOZ")}),
        ("Wine exports from Georgia to Russia", {}),
        ("Humanitarian aid to Gaza through Egypt", {}),
        ("Soybean exports from Brazil to China", {}),
    ])
    def test_readings_follow_the_text(self, text, readings):
        assert place_readings(text) == readings

    def test_places_are_read_as_written_inside_a_block(self):
        with read_names_as_regions(GEORGIA_STATE):
            assert read_places("Wine exports from Georgia to Russia") == (
                ["GEO", "RUS"], [],
            )


class TestReadNamesAsRegions:
    def test_outside_a_block_georgia_is_the_country(self):
        assert not reading_names_as_regions()
        assert resolve_country_code("Georgia") == "GEO"
        assert find_country_codes("Georgia and Alabama") == ["GEO"]
        assert resolve_adm1_code("Georgia") is None
        assert region_read_for("Georgia") is None

    def test_inside_a_block_georgia_is_the_us_state(self):
        with read_names_as_regions(GEORGIA_STATE):
            assert reading_names_as_regions()
            assert resolve_country_code("Georgia") == "USA"
            assert resolve_country_code("Georgia", free_text=True) == "USA"
            assert resolve_country_code(
                "Georgia's broiler belt", free_text=True,
            ) == "USA"
            assert find_country_codes("Georgia and Alabama") == ["USA"]
            assert resolve_adm1_code("Georgia") == "USA011"
            assert resolve_adm1_code("Georgia", "USA") == "USA011"
            assert resolve_adm1_code("Georgia", "GEO") is None
            assert region_read_for("Georgia") == "USA011"
        assert resolve_country_code("Georgia") == "GEO"

    def test_an_empty_block_reads_nothing_but_is_in_force(self):
        with read_names_as_regions({}):
            assert reading_names_as_regions()
            assert resolve_country_code("Georgia") == "GEO"
        assert not reading_names_as_regions()

    def test_blocks_nest(self):
        with read_names_as_regions(GEORGIA_STATE):
            with read_names_as_regions({}):
                assert resolve_country_code("Georgia") == "GEO"
            assert resolve_country_code("Georgia") == "USA"


GEORGIA_ANALYSIS = """### 1. Coupling Classification

- **Intracoupling**: Present within Georgia's broiler industry.
- **Pericoupling**: Present: manure runoff into neighbouring Alabama.
- **Telecoupling**: Present: frozen chicken exports to China and feed corn from Iowa.

### 2. Intracoupling Analysis (within the focal system)

#### 2.1 Systems Identification

**Focal System**: Georgia
- **Geographic scope**: Georgia's broiler belt in the north of the state

### 3. Pericoupling Analysis (adjacent systems)

#### 3.1 Systems Identification

**Sending System (adjacent)**: Georgia
**Receiving System (adjacent)**: Alabama

#### 3.2 Flows Analysis

**Matter Flow**
- **Direction**: Georgia → Alabama
- **Description**: Nutrient runoff in shared rivers.

### 4. Telecoupling Analysis (distant systems)

#### 4.1 Systems Identification

**Sending System (distant)**: Georgia
**Receiving System (distant)**: China
**Sending System (distant)**: Iowa

#### 4.2 Flows Analysis

**Matter Flow**
- **Direction**: Georgia → China
- **Description**: Frozen chicken exports.

**Matter Flow**
- **Direction**: Iowa → Georgia
- **Description**: Feed corn shipments.
"""


class _Stub:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, messages, temperature=0.7, max_tokens=None, **kwargs):
        return LLMResponse(content=self.content)


def _analyze(description: str, analysis: str):
    assistant = MetacouplingAssistant(_Stub(analysis), generate_abstract=False)
    return assistant.analyze(description)


class TestAssistantReadsTheRunsNames:
    def test_validation_checks_georgia_as_the_us_state(self):
        # Until 2026-10-17 the country check took Georgia (GEO) as the focal
        # country and the ADM1 check checked nothing.
        result = _analyze(
            "Broiler chicken production in Georgia, United States, ships frozen "
            "chicken to China and buys feed corn from Iowa; manure runoff "
            "affects rivers flowing into neighbouring Alabama.",
            GEORGIA_ANALYSIS,
        )
        country = result.parsed.country_pericoupling_info
        adm1 = result.parsed.pericoupling_info
        assert country["focal_country"] == "United States (USA)"
        assert "GEO" not in str(country)
        assert adm1["focal_region"] == "Georgia (USA011)"
        assert "Georgia (USA011) ↔ Alabama (USA001): PERICOUPLED" in adm1["pair_results"]
        assert "Georgia (USA011) ↔ Iowa (USA016): TELECOUPLED" in adm1["pair_results"]
        assert "(GEO)" not in str(result.formatted)

    def test_georgian_wine_stays_with_the_country(self):
        result = _analyze(
            "Wine exports from Georgia to Russia",
            GEORGIA_ANALYSIS.replace("Alabama", "Armenia").replace("Iowa", "Russia"),
        )
        assert result.parsed.country_pericoupling_info["focal_country"] == "Georgia (GEO)"

    def test_a_map_drawn_later_reads_the_analysis_names(self):
        result = _analyze("Broiler chicken production in Georgia", GEORGIA_ANALYSIS)
        parsed = result.parsed
        # Outside any block the systems' "Georgia" is the country ...
        assert _extract_countries_from_analysis(parsed)["focal"] == "GEO"
        # ... while a map function reads it as the analysis does, where
        # Alabama and Iowa make it the US state ...
        focal = _reading_the_analysis_names(
            lambda p: _extract_countries_from_analysis(p)["focal"]
        )
        assert focal(parsed) == "USA"
        # ... unless a run's reading is in force.
        with read_names_as_regions({}):
            assert focal(parsed) == "GEO"
