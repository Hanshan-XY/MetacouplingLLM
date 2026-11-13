"""The ``maritime_adjacency`` toggle: country pairs whose waters meet at sea.

``data/maritime_adjacency_pairs.csv`` holds the country pairs that share a
boundary in the Marine Regions Maritime Boundaries Geodatabase, built by
``scripts/build_maritime_adjacency.py``.  The toggle is off by default and
applies to the country (ADM0) lookups only: the lookups themselves,
``build_adjacency(level="adm0")``, and the assistant's country hint, country
validation and country map.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from metacouplingllm.core import MetacouplingAssistant
from metacouplingllm.knowledge.countries import get_country_name
from metacouplingllm.knowledge.pericoupling import (
    _get_pairs,
    get_pericoupled_neighbors,
    is_pericoupled,
    lookup_pericoupling,
)
from metacouplingllm.prompts.builder import PromptBuilder

from ._helpers import make_parsed_analysis

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "metacouplingllm" / "data"
PAIRS_CSV = DATA / "maritime_adjacency_pairs.csv"


def _rows() -> list[dict]:
    with open(PAIRS_CSV, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# The shipped table
# ---------------------------------------------------------------------------


class TestMaritimeTable:
    def test_size_and_disputed_only_pairs(self):
        rows = _rows()
        assert len(rows) == 387
        disputed = {(r["code_a"], r["code_b"]) for r in rows if r["disputed_only"] == "True"}
        # Through the waters of the Falkland Islands, the Chagos Archipelago
        # and Western Sahara, assigned as the land database assigns them.
        assert disputed == {("ARG", "GBR"), ("GBR", "MDV"), ("MAR", "MRT")}

    def test_pairs_are_country_database_codes(self):
        codes = set()
        with open(DATA / "PeriTelecoupling_clean.csv", newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                codes |= {row["Sending"].strip(), row["Receiving"].strip()}
        seen = set()
        for r in _rows():
            a, b = r["code_a"], r["code_b"]
            assert a < b, (a, b)
            assert {a, b} <= codes, (a, b)
            assert (a, b) not in seen
            seen.add((a, b))
            assert r["name_a"] == get_country_name(a)
            assert r["name_b"] == get_country_name(b)
            assert float(r["boundary_km"]) >= 0

    def test_pins_are_documented(self):
        # PROVENANCE.md carries each pin in full, docs/REPRODUCING.md its first 8 characters.
        spec = importlib.util.spec_from_file_location(
            "build_maritime_adjacency", REPO / "scripts" / "build_maritime_adjacency.py",
        )
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        provenance = (DATA / "PROVENANCE.md").read_text(encoding="utf-8")
        reproducing = (REPO / "docs" / "REPRODUCING.md").read_text(encoding="utf-8")
        assert set(build.PINNED) == {"boundaries", "eez"}
        for key, pin in build.PINNED.items():
            assert pin in provenance, f"data/PROVENANCE.md does not carry the {key} pin"
            assert f"`{pin[:8]}…" in reproducing, f"docs/REPRODUCING.md does not carry the {key} pin"

    def test_build_reproduces_the_table(self):
        # From the pinned inputs committed in build_data/marine_regions/ (the script's defaults).
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "build_maritime_adjacency.py"), "--check"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


class TestLookups:
    def test_off_by_default(self):
        # France and the United Kingdom meet only at sea.
        assert is_pericoupled("France", "United Kingdom") is False
        assert is_pericoupled("France", "United Kingdom", maritime_adjacency=True) is True

    def test_land_pairs_unchanged(self):
        assert is_pericoupled("Mexico", "United States", maritime_adjacency=True) is True
        assert is_pericoupled("Brazil", "China", maritime_adjacency=True) is False

    def test_view_sizes(self):
        assert len(_get_pairs()) == 326
        assert len(_get_pairs(maritime_adjacency=True)) == 583
        assert len(_get_pairs(de_facto_borders=False)) == 323
        assert len(_get_pairs(de_facto_borders=False, maritime_adjacency=True)) == 578

    def test_strict_view_drops_pairs_through_disputed_waters(self):
        assert is_pericoupled("ARG", "GBR", maritime_adjacency=True) is True
        assert is_pericoupled(
            "ARG", "GBR", de_facto_borders=False, maritime_adjacency=True,
        ) is False

    def test_coupling_standard_governs_rivers_not_seas(self):
        # Guyana and Suriname: a river border with no fixed crossing, and
        # waters that meet at sea.
        assert is_pericoupled("GUY", "SUR", coupling_standard="moderate") is False
        assert is_pericoupled(
            "GUY", "SUR", coupling_standard="moderate", maritime_adjacency=True,
        ) is True

    def test_neighbors(self):
        assert get_pericoupled_neighbors("Japan") == set()
        assert {"KOR", "RUS"} <= get_pericoupled_neighbors("Japan", maritime_adjacency=True)

    def test_lookup_result(self):
        result = lookup_pericoupling("Cuba", "United States", maritime_adjacency=True)
        assert result.pair_type.value == "pericoupled"
        assert (result.sending_code, result.receiving_code) == ("CUB", "USA")


class TestBuildAdjacency:
    def test_country_level(self):
        pd = pytest.importorskip("pandas")
        from metacouplingllm.indicators import build_adjacency

        pairs = pd.DataFrame({"origin_id": ["France"], "destination_id": ["United Kingdom"]})
        assert build_adjacency(pairs, level="adm0")["adjacent"].tolist() == [0]
        assert build_adjacency(pairs, level="adm0", maritime_adjacency=True)["adjacent"].tolist() == [1]

    def test_adm1_rejects_the_toggle(self):
        pd = pytest.importorskip("pandas")
        from metacouplingllm.indicators import build_adjacency

        pairs = pd.DataFrame({"origin_id": ["MEX008"], "destination_id": ["USA044"]})
        with pytest.raises(ValueError, match="maritime_adjacency"):
            build_adjacency(pairs, level="adm1", maritime_adjacency=True)


# ---------------------------------------------------------------------------
# The assistant's country hint, country validation and country map
# ---------------------------------------------------------------------------

FRANCE_UK = "Fish exports from France to the United Kingdom"


class TestPromptAndValidation:
    def test_country_hint_follows_the_toggle(self):
        default = PromptBuilder(max_examples=0).build_system_prompt(research_context=FRANCE_UK)
        sea = PromptBuilder(max_examples=0, maritime_adjacency=True).build_system_prompt(
            research_context=FRANCE_UK,
        )
        assert "France (FRA) and United Kingdom (GBR) are **telecoupled**" in default
        assert "France (FRA) and United Kingdom (GBR) are **pericoupled**" in sea

    def test_single_country_hint_lists_sea_neighbors(self):
        assert PromptBuilder._build_coupling_hint("Fishing in Japan") is None
        hint = PromptBuilder._build_coupling_hint("Fishing in Japan", maritime_adjacency=True)
        assert hint is not None and "South Korea (KOR)" in hint

    @pytest.mark.parametrize("sea, verdict", [(False, "TELECOUPLED"), (True, "PERICOUPLED")])
    def test_country_validation_follows_the_toggle(self, sea, verdict):
        parsed = make_parsed_analysis(
            coupling_classification="pericoupling",
            systems={
                "sending": {"name": "France", "geographic_scope": "France"},
                "receiving": {"name": "United Kingdom", "geographic_scope": "United Kingdom"},
            },
        )
        MetacouplingAssistant._validate_country_pericoupling(parsed, maritime_adjacency=sea)
        assert verdict in parsed.country_pericoupling_info.get("pair_results", "")

    def test_assistant_passes_the_toggle_to_its_builder(self, mock_llm_client):
        advisor = MetacouplingAssistant(
            mock_llm_client, generate_abstract=False, maritime_adjacency=True,
        )
        advisor.analyze(FRANCE_UK)
        assert "France (FRA) and United Kingdom (GBR) are **pericoupled**" in advisor.history[0].content


class TestAtSeaLabels:
    """A pair adjacent only at sea is said to be, in the hint and in the validation block."""

    def test_pair_hint_says_when_adjacency_is_at_sea(self):
        sea = PromptBuilder._build_coupling_hint(FRANCE_UK, maritime_adjacency=True)
        assert ("France (FRA) and United Kingdom (GBR) are **pericoupled** "
                "(adjacent at sea: their waters share a maritime boundary)") in sea
        land = PromptBuilder._build_coupling_hint(
            "Avocado exports from Mexico to the United States", maritime_adjacency=True,
        )
        assert "are **pericoupled** (geographically adjacent)" in land
        assert "at sea" not in land

    def test_neighbor_list_marks_neighbors_met_at_sea(self):
        hint = PromptBuilder._build_coupling_hint("Avocado production in Mexico", maritime_adjacency=True)
        assert "- Cuba (CUB) (at sea)\n" in hint
        assert "- Honduras (HND) (at sea)\n" in hint
        assert "- Guatemala (GTM)\n" in hint
        assert "- United States (USA)\n" in hint
        assert "Neighbors marked (at sea) are adjacent only at sea" in hint

    def test_a_river_border_without_a_crossing_counts_at_sea(self):
        # Guyana and Suriname meet on land only across a river with no fixed
        # crossing, so under "moderate" they are adjacent at sea alone.
        hint = PromptBuilder._build_coupling_hint(
            "Timber trade from Guyana to Suriname", coupling_standard="moderate",
            maritime_adjacency=True,
        )
        assert "(adjacent at sea: their waters share a maritime boundary)" in hint
        lenient = PromptBuilder._build_coupling_hint(
            "Timber trade from Guyana to Suriname", maritime_adjacency=True,
        )
        assert "(geographically adjacent)" in lenient

    def test_off_by_default_the_hint_has_no_sea_label(self):
        assert "at sea" not in PromptBuilder._build_coupling_hint("Avocado production in Mexico")

    def test_validation_block_marks_pairs_met_at_sea(self):
        from metacouplingllm.output.formatter import AnalysisFormatter

        parsed = make_parsed_analysis(
            coupling_classification="pericoupling",
            systems={
                "sending": {"name": "France", "geographic_scope": "France"},
                "receiving": {"name": "United Kingdom", "geographic_scope": "United Kingdom"},
            },
        )
        MetacouplingAssistant._validate_country_pericoupling(parsed, maritime_adjacency=True)
        assert "United Kingdom (GBR) (at sea): PERICOUPLED" in parsed.country_pericoupling_info["pair_results"]
        assert "United Kingdom (GBR) (at sea)" in AnalysisFormatter.format_full(parsed)


class _FranceClient:
    def chat(self, messages, temperature=0.7, max_tokens=None):
        from metacouplingllm.llm.client import LLMResponse

        return LLMResponse(content="### 1. Coupling Classification\nPericoupling.")


def test_assistant_country_map_gets_the_toggle(monkeypatch):
    calls = []

    def fake_country_map(parsed, **kwargs):
        calls.append(kwargs)
        return "country-figure"

    monkeypatch.setattr(
        "metacouplingllm.visualization.worldmap.plot_analysis_map", fake_country_map,
    )
    advisor = MetacouplingAssistant(_FranceClient(), auto_map=True, maritime_adjacency=True)
    parsed = make_parsed_analysis(
        systems={
            "sending": {"name": "France", "geographic_scope": "France"},
            "receiving": {"name": "United Kingdom", "geographic_scope": "United Kingdom"},
        },
    )
    parsed.map_data = {
        "focal_country": "FRA", "adm1_region": None, "receiving_countries": ["GBR"],
        "spillover_countries": [], "flows": [],
    }
    assert advisor._generate_map(parsed) == "country-figure"
    (kwargs,) = calls
    assert kwargs["maritime_adjacency"] is True
