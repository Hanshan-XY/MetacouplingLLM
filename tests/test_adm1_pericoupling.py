"""Tests for knowledge/adm1_pericoupling.py — ADM1 pericoupling database lookup."""

import csv
import json
from pathlib import Path

import pytest

from metacouplingllm.knowledge.adm1_pericoupling import (
    Adm1PairType,
    Adm1PericouplingResult,
    get_adm1_codes_for_country,
    get_adm1_country,
    get_adm1_info,
    get_adm1_neighbors,
    get_cross_border_neighbors,
    is_adm1_pericoupled,
    lookup_adm1_pericoupling,
    resolve_adm1_code,
    _ensure_loaded,
    _adm1_pairs,
    _adm1_country,
    _adm1_metadata,
    _resolve_adm1_name,
)


class TestAdm1DataLoading:
    """Verify that the ADM1 CSV database is loaded correctly."""

    def test_data_loads(self):
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_pairs
        assert _adm1_pairs is not None
        assert len(_adm1_pairs) > 0, "No ADM1 pairs loaded from CSV"

    def test_expected_pair_count(self):
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_pairs
        assert _adm1_pairs is not None
        assert len(_adm1_pairs) == 8462, (
            f"Expected 8462 pairs, got {len(_adm1_pairs)}"
        )

    def test_expected_code_count(self):
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country
        assert _adm1_country is not None
        assert len(_adm1_country) == 3374, (
            f"Expected 3374 unique ADM1 codes, got {len(_adm1_country)}"
        )

    def test_expected_country_count(self):
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country
        assert _adm1_country is not None
        countries = set(_adm1_country.values())
        assert len(countries) == 196, (
            f"Expected 196 unique countries, got {len(countries)}"
        )


class TestLookupAdm1Pericoupling:
    """Test the lookup_adm1_pericoupling function."""

    def test_within_country_adjacent(self):
        """AFG001 and AFG024 share a border within Afghanistan."""
        result = lookup_adm1_pericoupling("AFG001", "AFG024")
        assert result.pair_type == Adm1PairType.PERICOUPLED
        assert result.code_a == "AFG001"
        assert result.code_b == "AFG024"
        assert result.cross_country is False
        assert result.confidence == "database"

    def test_cross_country_adjacent(self):
        """AFG001 and PAK005 share a cross-country border."""
        result = lookup_adm1_pericoupling("AFG001", "PAK005")
        assert result.pair_type == Adm1PairType.PERICOUPLED
        assert result.cross_country is True
        assert result.confidence == "database"

    def test_telecoupled_different_countries(self):
        """AFG001 and BRA001 should be telecoupled (distant, different countries)."""
        # First verify both exist
        info_a = get_adm1_info("AFG001")
        assert info_a is not None
        # Find a Brazilian code
        bra_codes = get_adm1_codes_for_country("BRA")
        assert len(bra_codes) > 0
        bra_code = sorted(bra_codes)[0]

        result = lookup_adm1_pericoupling("AFG001", bra_code)
        assert result.pair_type == Adm1PairType.TELECOUPLED
        assert result.cross_country is True

    def test_same_region(self):
        """Looking up a region against itself should return SAME_REGION."""
        result = lookup_adm1_pericoupling("AFG001", "AFG001")
        assert result.pair_type == Adm1PairType.SAME_REGION
        assert result.confidence == "same_region"

    def test_unresolved_code(self):
        """Unknown ADM1 codes should return UNKNOWN."""
        result = lookup_adm1_pericoupling("ZZZZZ", "AFG001")
        assert result.pair_type == Adm1PairType.UNKNOWN
        assert result.confidence == "unresolved"
        assert result.code_a is None
        assert result.code_b == "AFG001"

    def test_both_unresolved(self):
        result = lookup_adm1_pericoupling("ZZZZZ", "YYYYY")
        assert result.pair_type == Adm1PairType.UNKNOWN
        assert result.code_a is None
        assert result.code_b is None

    def test_symmetric_lookup(self):
        """Order should not matter for adjacency."""
        r1 = lookup_adm1_pericoupling("AFG001", "PAK005")
        r2 = lookup_adm1_pericoupling("PAK005", "AFG001")
        assert r1.pair_type == r2.pair_type == Adm1PairType.PERICOUPLED

    def test_result_is_frozen_dataclass(self):
        result = lookup_adm1_pericoupling("AFG001", "AFG024")
        assert isinstance(result, Adm1PericouplingResult)

    def test_whitespace_handling(self):
        """Codes with whitespace should be stripped."""
        result = lookup_adm1_pericoupling("  AFG001  ", "AFG024")
        assert result.pair_type == Adm1PairType.PERICOUPLED


class TestGetAdm1Neighbors:
    """Test the get_adm1_neighbors function."""

    def test_afg001_neighbors(self):
        """AFG001 (Badakhshan) should have known neighbors."""
        neighbors = get_adm1_neighbors("AFG001")
        assert len(neighbors) > 0
        # Within-country neighbors
        assert "AFG024" in neighbors  # Nuristan
        assert "AFG027" in neighbors  # Panjsher
        assert "AFG031" in neighbors  # Takhar
        # Cross-border neighbors
        assert "PAK005" in neighbors  # Khyber Pakhtunkhwa
        assert "TJK001" in neighbors  # Badakhshan (Tajikistan)

    def test_unknown_code_returns_empty(self):
        neighbors = get_adm1_neighbors("ZZZZZ")
        assert neighbors == set()

    def test_returns_copy(self):
        """Should return a new set, not a reference to internal data."""
        n1 = get_adm1_neighbors("AFG001")
        n2 = get_adm1_neighbors("AFG001")
        assert n1 == n2
        assert n1 is not n2  # different objects


class TestGetCrossBorderNeighbors:
    """Test the get_cross_border_neighbors function."""

    def test_afg001_cross_border(self):
        """AFG001 should have cross-border neighbors in PAK and TJK."""
        cross = get_cross_border_neighbors("AFG001")
        assert len(cross) > 0
        assert "PAK005" in cross
        assert "TJK001" in cross
        # Within-country neighbors should NOT be here
        assert "AFG024" not in cross
        assert "AFG027" not in cross

    def test_unknown_code_returns_empty(self):
        cross = get_cross_border_neighbors("ZZZZZ")
        assert cross == set()

    def test_cross_border_subset_of_all_neighbors(self):
        """Cross-border neighbors must be a subset of all neighbors."""
        all_n = get_adm1_neighbors("AFG001")
        cross = get_cross_border_neighbors("AFG001")
        assert cross.issubset(all_n)


class TestGetAdm1CodesForCountry:
    """Test the get_adm1_codes_for_country function."""

    def test_by_iso_code(self):
        codes = get_adm1_codes_for_country("AFG")
        assert len(codes) > 0
        assert "AFG001" in codes

    def test_by_country_name(self):
        codes = get_adm1_codes_for_country("Afghanistan")
        assert len(codes) > 0
        assert "AFG001" in codes

    def test_all_codes_start_with_iso(self):
        """All Mexican ADM1 codes should start with MEX."""
        codes = get_adm1_codes_for_country("MEX")
        assert len(codes) > 0
        for code in codes:
            assert code.startswith("MEX"), f"Expected MEX prefix, got {code}"

    def test_unknown_country_returns_empty(self):
        codes = get_adm1_codes_for_country("Atlantis")
        assert codes == set()

    def test_consistent_with_name_and_code(self):
        """Country name and ISO code should return the same set."""
        by_name = get_adm1_codes_for_country("Mexico")
        by_code = get_adm1_codes_for_country("MEX")
        assert by_name == by_code


class TestGetAdm1Info:
    """Test the get_adm1_info function."""

    def test_afg001_info(self):
        info = get_adm1_info("AFG001")
        assert info is not None
        assert info["name"] == "Badakhshan"
        assert info["country_name"] == "Afghanistan"
        assert info["iso_a3"] == "AFG"
        assert info["wb_region"] == "SAR"

    def test_unknown_code_returns_none(self):
        info = get_adm1_info("ZZZZZ")
        assert info is None

    def test_whitespace_handling(self):
        info = get_adm1_info("  AFG001  ")
        assert info is not None
        assert info["name"] == "Badakhshan"


class TestGetAdm1Country:
    """Test the get_adm1_country function."""

    def test_known_code(self):
        assert get_adm1_country("AFG001") == "AFG"

    def test_unknown_code(self):
        assert get_adm1_country("ZZZZZ") is None


class TestIsAdm1Pericoupled:
    """Test the is_adm1_pericoupled convenience function."""

    def test_true_for_adjacent(self):
        assert is_adm1_pericoupled("AFG001", "AFG024") is True

    def test_true_for_cross_border(self):
        assert is_adm1_pericoupled("AFG001", "PAK005") is True

    def test_false_for_distant(self):
        """Non-adjacent regions should return False."""
        bra_codes = get_adm1_codes_for_country("BRA")
        bra_code = sorted(bra_codes)[0]
        assert is_adm1_pericoupled("AFG001", bra_code) is False

    def test_false_for_same_region(self):
        assert is_adm1_pericoupled("AFG001", "AFG001") is False

    def test_none_for_unknown(self):
        assert is_adm1_pericoupled("ZZZZZ", "AFG001") is None


class TestCrossBorderPairs:
    """Test cross-country border pair detection from the database."""

    def test_afg_pak_border(self):
        """Afghanistan-Pakistan border should have cross-country pairs."""
        result = lookup_adm1_pericoupling("AFG001", "PAK005")
        assert result.pair_type == Adm1PairType.PERICOUPLED
        assert result.cross_country is True

    def test_afg_tjk_border(self):
        """Afghanistan-Tajikistan border should have cross-country pairs."""
        result = lookup_adm1_pericoupling("AFG001", "TJK001")
        assert result.pair_type == Adm1PairType.PERICOUPLED
        assert result.cross_country is True

    def test_within_country_not_cross(self):
        """Within-country pairs should not be cross-country."""
        result = lookup_adm1_pericoupling("AFG001", "AFG024")
        assert result.pair_type == Adm1PairType.PERICOUPLED
        assert result.cross_country is False


class TestResolveAdm1Code:
    """Test the resolve_adm1_code function."""

    def test_michigan(self):
        """Michigan should resolve to USA023."""
        code = resolve_adm1_code("Michigan")
        assert code == "USA023"

    def test_anhui(self):
        """Anhui should resolve to CHN001 (Anhui Sheng in DB)."""
        code = resolve_adm1_code("Anhui")
        assert code == "CHN001"

    def test_georgia_disambiguated_by_country(self):
        """Georgia + country=USA should resolve to USA011, not the country."""
        code = resolve_adm1_code("Georgia", country="USA")
        assert code == "USA011"

    def test_khyber_pakhtunkhwa(self):
        """Khyber Pakhtunkhwa should resolve to PAK005."""
        code = resolve_adm1_code("Khyber Pakhtunkhwa")
        assert code == "PAK005"

    def test_badakhshan(self):
        """Badakhshan should resolve (AFG001 or TJK001)."""
        code = resolve_adm1_code("Badakhshan")
        assert code is not None
        assert code in ("AFG001", "TJK001")

    def test_badakhshan_disambiguated(self):
        """Badakhshan + country=AFG should resolve to AFG001."""
        code = resolve_adm1_code("Badakhshan", country="AFG")
        assert code == "AFG001"

    def test_badakhshan_tajikistan(self):
        """Badakhshan + country=TJK should resolve to TJK001."""
        code = resolve_adm1_code("Badakhshan", country="TJK")
        assert code == "TJK001"

    def test_unknown_region_returns_none(self):
        """Unknown region names should return None."""
        code = resolve_adm1_code("Atlantis")
        assert code is None

    def test_empty_string_returns_none(self):
        code = resolve_adm1_code("")
        assert code is None

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped."""
        code = resolve_adm1_code("  Michigan  ")
        assert code == "USA023"

    def test_case_insensitive(self):
        """Name matching should be case-insensitive."""
        code = resolve_adm1_code("michigan")
        assert code == "USA023"

    def test_country_as_name(self):
        """Country names should resolve using suffix-stripped or full names."""
        # Bavaria is a well-known region in Germany
        code = resolve_adm1_code("Bayern", country="DEU")
        # If found, it should be in Germany
        if code is not None:
            country = get_adm1_country(code)
            assert country == "DEU"

    def test_resolve_returns_string(self):
        """Return type should be str when found."""
        code = resolve_adm1_code("Michigan")
        assert isinstance(code, str)

    def test_country_filter_excludes_wrong_country(self):
        """Country filter should exclude results from wrong countries."""
        code = resolve_adm1_code("Michigan", country="CHN")
        assert code is None

    def test_trade_does_not_false_match_trad(self):
        """Generic words should not substring-match unrelated ADM1 names."""
        code = resolve_adm1_code("trade")
        assert code is None

    # PR #45: accent-folded fallback (Strategy 3).  The DB stores
    # region names with their canonical accents (e.g.,
    # "Michoacán de Ocampo").  English LLM output and many user
    # queries drop the accents, which previously broke resolution.
    # The fallback folds both sides via NFKD before retrying.

    def test_unaccented_michoacan_resolves(self):
        """Michoacan (no accent) should fold to match the
        canonical "Michoacán de Ocampo" entry → MEX016."""
        assert resolve_adm1_code("Michoacan", country="MEX") == "MEX016"

    def test_unaccented_michoacan_de_ocampo_resolves(self):
        """The unaccented full name should also resolve."""
        code = resolve_adm1_code("Michoacan de Ocampo", country="MEX")
        assert code == "MEX016"

    def test_unaccented_yucatan_resolves(self):
        """Yucatan (no accent) → MEX031 (Yucatán)."""
        assert resolve_adm1_code("Yucatan", country="MEX") == "MEX031"

    def test_unaccented_nuevo_leon_resolves(self):
        """Nuevo Leon (no accent) → MEX019 (Nuevo León)."""
        assert resolve_adm1_code("Nuevo Leon", country="MEX") == "MEX019"

    def test_unaccented_sao_paulo_resolves(self):
        """Sao Paulo (no accent) → BRA029 (São Paulo)."""
        assert resolve_adm1_code("Sao Paulo", country="BRA") == "BRA029"

    def test_accented_form_still_resolves(self):
        """Regression: existing accented-form resolution still works
        through Strategy 1/2 (unchanged behavior)."""
        assert resolve_adm1_code("Michoacán") == "MEX016"
        assert resolve_adm1_code("São Paulo") == "BRA029"

    def test_non_decomposable_letters_fold(self):
        """Native-script queries with letters NFKD cannot decompose
        (Ł, Đ, ø, ...) transliterate to the ASCII names the DB stores.
        Previously these returned None while their ASCII forms worked."""
        assert resolve_adm1_code("Łódź") == "POL003"
        assert resolve_adm1_code("Đắk Lắk") == "VNM016"
        assert resolve_adm1_code("Đồng Nai") == "VNM019"
        # the ASCII forms keep resolving identically
        assert resolve_adm1_code("Lodz") == "POL003"
        assert resolve_adm1_code("Dak Lak") == "VNM016"

    def test_folded_ambiguity_respects_country_filter(self):
        """Folded match honors the country filter — if a folded
        candidate exists outside the requested country, it's
        excluded.  (Mexico has no foreign-country folded
        collisions for the avocado states, so we test the
        weaker no-cross-country-leak property here.)"""
        # Michoacan only exists in Mexico; with country=CHN it
        # must return None (no leak through the folded fallback).
        assert resolve_adm1_code("Michoacan", country="CHN") is None

    # Cross-country folded-name ambiguity (Équateur / Equateur).
    #
    # Two ADM1 provinces fold to the same name across different
    # countries: the DRC stores its province accented as "Équateur"
    # (COD013), the CAR stores its unaccented as "Equateur" (CAF002).
    # A bare "Equateur" used to resolve arbitrarily to CAF002 because
    # the CAR spelling is an exact Strategy-1 match.  An exact accented
    # query is unambiguous; an unaccented query needs a country hint.

    def test_accented_equateur_resolves_to_drc(self):
        """The exact accented form uniquely identifies the DRC province."""
        assert resolve_adm1_code("Équateur") == "COD013"

    def test_unaccented_equateur_ambiguous_without_hint(self):
        """Bare "Equateur" is ambiguous across DRC/CAR — it must not
        silently pick one, so it returns None without a country hint."""
        assert resolve_adm1_code("Equateur") is None
        assert resolve_adm1_code("equateur") is None

    def test_equateur_hint_drc(self):
        """A DRC hint constrains the unaccented form to COD013."""
        assert resolve_adm1_code("Equateur", country="COD") == "COD013"

    def test_equateur_hint_car(self):
        """A CAR hint constrains the unaccented form to CAF002."""
        assert resolve_adm1_code("Equateur", country="CAF") == "CAF002"

    def test_equateur_hint_unrelated_country_returns_none(self):
        """A hint for a country with no Équateur province returns None
        (no leak through the folded fallback)."""
        assert resolve_adm1_code("Equateur", country="USA") is None

    def test_cross_country_folded_collisions_need_hint_generally(self):
        """General case: every unaccented name shared (after accent-
        folding) by ADM1 regions in more than one country must not
        silently resolve without a country hint, and must resolve
        within the hinted country when a hint is supplied.

        Data-driven, so cross-country collisions introduced by future
        data refreshes are covered automatically."""
        from metacouplingllm.knowledge.adm1_pericoupling import (
            _get_adm1_folded_name_index,
            get_adm1_country,
        )

        folded = _get_adm1_folded_name_index()
        collisions = {
            name: entries
            for name, entries in folded.items()
            if len({iso for _, iso in entries}) > 1
        }
        assert collisions, "expected >=1 cross-country folded collision"
        # The motivating Équateur pair must be among them.
        assert "equateur" in collisions, "equateur collision missing"

        for name, entries in collisions.items():
            # No hint: must not arbitrarily pick a single region.
            no_hint = resolve_adm1_code(name)
            assert no_hint is None, (
                "%r resolved without a hint to %r" % (name, no_hint)
            )
            # With each member country hint, it resolves inside that
            # country.
            for _code, iso in entries:
                resolved = resolve_adm1_code(name, country=iso)
                assert resolved is not None, (
                    "%r + country=%s unexpectedly returned None" % (name, iso)
                )
                assert get_adm1_country(resolved) == iso, (
                    "%r + country=%s -> %r which is not in %s"
                    % (name, iso, resolved, iso)
                )

    # --- Direction-aware substring guard (loose-match correctness) ---

    def test_direction_a_extra_meaningful_word_returns_none(self):
        """A query that merely CONTAINS a region name plus a meaningful extra
        word must not resolve to that region -- it denotes a different place.
        Previously 'Mexico City' wrongly returned MEX015 (the *State* of
        México) instead of CDMX, and 'New Mexico' (a US state) likewise.
        'Mexico City' now resolves to CDMX itself through its alias."""
        assert resolve_adm1_code("Mexico City", country="Mexico") == "MEX009"
        assert resolve_adm1_code("New Mexico", country="Mexico") is None
        assert resolve_adm1_code("Greater Mexico", country="Mexico") is None
        assert resolve_adm1_code("Chihuahua Desert", country="Mexico") is None

    def test_direction_a_ignorable_word_still_resolves(self):
        """Padding a region name with only administrative / connector words
        ('State', 'of') does not change the place and still resolves."""
        assert resolve_adm1_code("Mexico State", country="Mexico") == "MEX015"
        assert resolve_adm1_code("State of Yucatan", country="MEX") == "MEX031"

    def test_direction_a_romance_qualifier_still_resolves(self):
        """Spanish/Romance administrative qualifiers ('estado', 'provincia',
        'comunidad', 'departamento') are ignorable just like their English
        counterparts ('state', 'province').  A country hint is required when
        the bare core name is ambiguous or resolves as a country."""
        assert resolve_adm1_code("Estado de Yucatan", country="MEX") == "MEX031"
        assert resolve_adm1_code("Estado de Jalisco", country="MEX") == "MEX014"
        assert resolve_adm1_code("Provincia de Buenos Aires", country="ARG") == "ARG001"
        assert resolve_adm1_code("Departamento de Antioquia", country="COL") == "COL002"

    def test_direction_b_partial_name_preserved(self):
        """Direction B (a short query inside a longer official name) stays
        intact through the tightened guard."""
        assert resolve_adm1_code("Michoacan", country="MEX") == "MEX016"
        assert resolve_adm1_code("Coahuila", country="MEX") == "MEX005"

    def test_substring_ambiguity_returns_none(self):
        """When a query substring-matches more than one distinct region, the
        resolver refuses to guess (no first-match-wins).  'Carolina' matches
        both North (USA034) and South (USA041) Carolina."""
        assert resolve_adm1_code("Carolina") is None
        assert resolve_adm1_code("Carolina", country="USA") is None

    def test_direction_b_qualifier_prefix_returns_none(self):
        """A bare token that is a substring of a 'qualifier + name' canonical
        must NOT resolve to it -- a leading 'New'/'South'/'Greater' names a
        different region (PR #61 Direction-B hole).  'york' ⊄ New York, etc."""
        assert resolve_adm1_code("york") is None
        assert resolve_adm1_code("hampshire") is None
        assert resolve_adm1_code("carolina") is None
        # Hyphen-aware: the French "nouveau-brunswick" (= New Brunswick) too.
        assert resolve_adm1_code("brunswick") is None

    def test_direction_b_full_qualified_name_still_resolves(self):
        """The full 'qualifier + name' form still resolves to its own region."""
        assert resolve_adm1_code("New York") == "USA033"
        assert resolve_adm1_code("New Hampshire") == "USA030"
        assert resolve_adm1_code("North Carolina") == "USA034"
        assert resolve_adm1_code("South Carolina") == "USA041"
        # And legitimate Direction-B head matches are unaffected.
        assert resolve_adm1_code("Michoacan de Ocampo") == "MEX016"
        assert resolve_adm1_code("Veracruz") == "MEX030"


# ---------------------------------------------------------------------------
# Country names
# ---------------------------------------------------------------------------


class TestCountryGuard:
    """Without a country hint, resolve_adm1_code refuses a name that IS a
    country's ISO code, name or alias, alone or among words its substring
    strategies ignore, and resolves one with any other word besides a
    country's name.  Until 2026-10-16 it refused every name in which
    resolve_country_code found a country: 52 of the 3,374 World Bank unit
    names, some read as another country ("New South Wales" as the UK)."""

    @staticmethod
    def _units(predicate):
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_metadata
        return {
            code: meta for code, meta in _adm1_metadata.items()
            if predicate(meta["name"])
        }

    @pytest.mark.parametrize("name, code", [
        ("New South Wales", "AUS004"),
        ("Western Australia", "AUS011"),
        ("British Columbia", "CAN002"),
        ("Northern Ireland", "GBR003"),
        ("East New Britain", "PNG004"),
        ("West New Britain", "PNG020"),
        ("Port Of Spain", "TTO007"),
        ("New Jersey", "USA031"),
        ("New Mexico", "USA032"),
        ("Thai Binh", "VNM054"),
        ("Thai Nguyen", "VNM055"),
        ("Ile-de-France", "FRA013"),
        ("Mexico City", "MEX009"),
        ("Ciudad de Mexico", "MEX009"),
    ])
    def test_names_containing_a_country_name(self, name, code):
        assert resolve_adm1_code(name) == code

    def test_every_unit_name_containing_a_country_name(self):
        """The 35 unit names in which resolve_country_code finds a country
        without being one resolve to their own unit without a hint."""
        from metacouplingllm.knowledge.countries import resolve_country_code
        units = self._units(
            lambda name: resolve_country_code(name) is not None
            and resolve_country_code(name, substring=False) is None
        )
        assert len(units) == 35
        failed = [
            (meta["name"], code, resolve_adm1_code(meta["name"]))
            for code, meta in units.items()
            if resolve_adm1_code(meta["name"]) != code
        ]
        assert not failed, failed

    def test_every_unit_named_like_a_country_needs_the_hint(self):
        """The 21 unit names that are a country's code, name or alias
        ("Georgia", "Niger", "Wales", "Est" for Estonia's code, "México")
        resolve only with a country hint."""
        from metacouplingllm.knowledge.countries import resolve_country_code
        units = self._units(
            lambda name: resolve_country_code(name, substring=False)
            is not None
        )
        assert len(units) == 21
        failed = [
            (meta["name"], code)
            for code, meta in units.items()
            if resolve_adm1_code(meta["name"]) is not None
            or resolve_adm1_code(meta["name"], country=meta["iso_a3"]) != code
        ]
        assert not failed, failed

    def test_accented_country_names_need_the_hint(self):
        """"México" and "Panamá", the World Bank names of the State of
        México and Panamá province, are the countries' names in Spanish.
        Like "Mexico" and "Panama" they now need a hint; until 2026-10-16
        they resolved without one only because resolve_country_code did not
        fold accents."""
        assert resolve_adm1_code("Mexico") is None
        assert resolve_adm1_code("México") is None
        assert resolve_adm1_code("México", country="MEX") == "MEX015"
        assert resolve_adm1_code("Panama") is None
        assert resolve_adm1_code("Panamá") is None
        assert resolve_adm1_code("Panamá", country="PAN") == "PAN011"

    @pytest.mark.parametrize("name", [
        "the Mexico", "of Mexico", "region of Mexico", "State of Mexico",
        "Mexico State", "Estado de México", "and Georgia", "Georgia State",
        "the Niger", "Niger State", "de Panama", "the Gambia",
    ])
    def test_country_name_among_ignored_words_needs_the_hint(self, name):
        """The substring strategies ignore connector and administrative
        words, so n-grams of text such as "region of Mexico" (from "the
        Michoacán region of Mexico") would resolve to the State of México.
        A country's name among such words is read as the country."""
        assert resolve_adm1_code(name) is None

    def test_country_name_among_ignored_words_with_the_hint(self):
        assert resolve_adm1_code("State of Mexico", country="MEX") == "MEX015"
        assert resolve_adm1_code("Estado de México", country="MEX") == "MEX015"
        assert resolve_adm1_code("Georgia State", country="USA") == "USA011"
        assert resolve_adm1_code("Niger State", country="NGA") == "NGA027"

    def test_any_other_word_makes_a_region_name(self):
        assert resolve_adm1_code("of New Mexico") == "USA032"
        assert resolve_adm1_code("state of New Jersey") == "USA031"
        assert resolve_adm1_code("Santiago de Cuba") == "CUB014"
        assert resolve_adm1_code("Bosnia and Herzegovina") is None

    def test_other_words_around_a_country_name(self):
        assert resolve_adm1_code("Republic of Georgia") is None
        assert resolve_adm1_code("Gulf of Mexico") is None
        assert resolve_adm1_code("Northern Mexico") is None
        assert resolve_adm1_code("Mexican avocado farms") is None


# ---------------------------------------------------------------------------
# Trailing administrative designators
# ---------------------------------------------------------------------------

# Names researchers wrote that returned None, with the country hint passed,
# until the designator retry and the curated aliases (2026-10-16).
_REPORTED_NAMES = [
    ("Narok County", "KEN", "KEN033"),
    ("Kwale County", "KEN", "KEN019"),
    ("Mombasa County", "KEN", "KEN028"),
    ("Kagoshima Prefecture", "JPN", "JPN017"),
    ("Rangpur Division", "BGD", "BGD007"),
    ("Khulna Division", "BGD", "BGD004"),
    ("Beni Department", "BOL", "BOL001"),
    ("Rivas Department", "NIC", "NIC017"),
    ("Magallanes Region", "CHL", "CHL011"),
    ("Chobe District", "BWA", "BWA002"),
    ("North-West District", "BWA", "BWA007"),
    ("Troms county", "NOR", "NOR017"),
    ("Silesian Voivodeship", "POL", "POL012"),
    ("Mexico City", "MEX", "MEX009"),
]


class TestTrailingDesignator:
    """When a name does not resolve, one trailing administrative designator
    is removed and the rest resolved again ("Narok County" -> "Narok")."""

    @pytest.mark.parametrize("name, country, code", _REPORTED_NAMES)
    def test_reported_names_resolve(self, name, country, code):
        assert resolve_adm1_code(name, country=country) == code

    @pytest.mark.parametrize("name, country, code", _REPORTED_NAMES)
    def test_reported_names_resolve_without_hint(self, name, country, code):
        assert resolve_adm1_code(name) == code

    def test_bare_names_keep_their_codes(self):
        assert resolve_adm1_code("Narok", country="KEN") == "KEN033"
        assert resolve_adm1_code("Kagoshima", country="JPN") == "JPN017"
        assert resolve_adm1_code("Rajshahi Division", country="BGD") == "BGD006"
        assert resolve_adm1_code("Beni", country="BOL") == "BOL001"
        assert resolve_adm1_code("Ciudad de México", country="MEX") == "MEX009"
        assert resolve_adm1_code("Śląskie", country="POL") == "POL012"

    @pytest.mark.parametrize("iso, designator", [
        ("KEN", "County"), ("NOR", "County"), ("JPN", "Prefecture"),
        ("BGD", "Division"), ("BOL", "Department"), ("NIC", "Department"),
        ("CHL", "Region"), ("BWA", "District"), ("POL", "Voivodeship"),
        ("MEX", "State"),
    ])
    def test_every_unit_with_its_designator(self, iso, designator):
        """'Rajshahi Division' used to resolve through an alias row while
        'Rangpur Division' did not; now every unit of these countries
        resolves under its World Bank name plus the designator."""
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_metadata
        units = {
            code: meta["name"] for code, meta in _adm1_metadata.items()
            if meta["iso_a3"] == iso
        }
        assert units
        failed = [
            (name, code) for code, name in units.items()
            if resolve_adm1_code(f"{name} {designator}", country=iso) != code
        ]
        assert not failed, failed

    def test_case_and_spacing(self):
        assert resolve_adm1_code("NAROK COUNTY") == "KEN033"
        assert resolve_adm1_code("narok county") == "KEN033"
        assert resolve_adm1_code("  Narok   County  ", country="KEN") == "KEN033"

    def test_only_one_designator_is_removed(self):
        assert resolve_adm1_code("Narok County County", country="KEN") is None
        assert resolve_adm1_code("County", country="KEN") is None
        assert resolve_adm1_code("the county") is None

    def test_designator_must_name_first_level_units(self):
        """US counties, Bangladeshi and Tanzanian districts and Chinese
        prefectures lie below the first level: the region the rest names is
        not what the designated name means."""
        assert resolve_adm1_code("Washington", country="USA") == "USA048"
        assert resolve_adm1_code("Washington County") is None
        assert resolve_adm1_code("Washington County", country="USA") is None
        assert resolve_adm1_code("Hidalgo") == "MEX013"
        assert resolve_adm1_code("Hidalgo County") is None
        assert resolve_adm1_code("Khulna", country="BGD") == "BGD004"
        assert resolve_adm1_code("Khulna District", country="BGD") is None
        assert resolve_adm1_code("Arusha", country="TZA") == "TZA001"
        assert resolve_adm1_code("Arusha District", country="TZA") is None
        assert resolve_adm1_code("Jilin", country="CHN") == "CHN017"
        assert resolve_adm1_code("Jilin Prefecture", country="CHN") is None

    def test_rest_matching_several_regions_stays_unresolved(self):
        """Busia is a Kenyan county and a Ugandan district; La Paz a
        department of Bolivia, Honduras and El Salvador; 'Central River'
        matches both Central River North and South in The Gambia."""
        assert resolve_adm1_code("Busia County") is None
        assert resolve_adm1_code("Busia County", country="KEN") == "KEN004"
        assert resolve_adm1_code("Busia District", country="UGA") == "UGA023"
        assert resolve_adm1_code("La Paz Department") is None
        assert resolve_adm1_code("La Paz Department", country="BOL") == "BOL004"
        assert resolve_adm1_code("La Paz Department", country="HND") == "HND011"
        assert resolve_adm1_code("La Paz Department", country="SLV") == "SLV006"
        assert resolve_adm1_code("Central River Division", country="GMB") is None
        assert resolve_adm1_code("Lower River Division", country="GMB") == "GMB005"

    def test_city_and_surrounding_region_refused(self):
        """'Almaty' and 'Moscow' resolve to the cities, whose surrounding
        regions are other units ('almaty region', 'moscow oblast'): a
        designated form means the region, so it stays unresolved."""
        assert resolve_adm1_code("Almaty") == "KAZ004"
        assert resolve_adm1_code("Almaty Region") == "KAZ003"
        assert resolve_adm1_code("Almaty Province") is None
        assert resolve_adm1_code("Moscow") == "RUS048"
        assert resolve_adm1_code("Moscow Region") == "RUS047"
        assert resolve_adm1_code("Moscow Province") is None

    def test_qualified_names_of_other_units_do_not_block(self):
        """'West Pomeranian', 'Norte de Santander' and 'Nord-Est' are other
        regions; they do not make 'Pomeranian', 'Santander' or 'Nord'
        ambiguous."""
        assert resolve_adm1_code("Pomeranian Voivodeship", country="POL") == "POL011"
        assert resolve_adm1_code("Santander Department", country="COL") == "COL028"
        assert resolve_adm1_code("Nord Department", country="HTI") == "HTI005"

    def test_never_changes_a_resolving_name(self):
        """Wherever strategies 0-3 resolve a query, resolve_adm1_code returns
        the same code: the retry runs only after a miss.  Checked on every
        query of the recorded name-matching outputs (14,421 queries)."""
        rows = json.loads(
            (Path(__file__).parent / "data" / "adm1_resolve_golden.json")
            .read_text(encoding="utf-8")
        )["resolve_adm1_code"]
        changed = [
            (query, country, code)
            for query, country, _ in rows
            if (code := _resolve_adm1_name(query, country)) is not None
            and resolve_adm1_code(query, country=country) != code
        ]
        assert not changed, changed[:10]


# Outputs of resolve_adm1_code recorded on main 6831ec7, before the designator
# retry and the curated aliases: names resolving through each strategy, names
# with a designator that already resolved, and names that did not resolve.
_RECORDED_BEFORE_RETRY = [
    ("Michigan", None, "USA023"),
    ("Anhui", None, "CHN001"),
    ("Georgia", "USA", "USA011"),
    ("Khyber Pakhtunkhwa", None, "PAK005"),
    ("Bavaria", None, "DEU002"),
    ("Tuscany", None, "ITA016"),
    ("Saxony", "DEU", "DEU013"),
    ("Chittagong", None, "BGD002"),
    ("Kyoto", None, "JPN021"),
    ("Saigon", None, "VNM029"),
    ("State of Yucatan", "MEX", "MEX031"),
    ("Mexico State", "Mexico", "MEX015"),
    ("Provincia de Buenos Aires", "ARG", "ARG001"),
    ("Estado de Jalisco", "MEX", "MEX014"),
    ("Michoacan", "MEX", "MEX016"),
    ("Coahuila", "MEX", "MEX005"),
    ("Magallanes", None, "CHL011"),
    ("Sao Paulo", "BRA", "BRA029"),
    ("Lodz", None, "POL003"),
    ("Łódź", None, "POL003"),
    ("Équateur", None, "COD013"),
    ("Equateur", "CAF", "CAF002"),
    ("New York", None, "USA033"),
    ("North Carolina", None, "USA034"),
    ("Rajshahi Division", None, "BGD006"),
    ("Moscow Oblast", None, "RUS047"),
    ("Moscow Region", None, "RUS047"),
    ("Kyiv Oblast", None, "UKR012"),
    ("Zagreb County", None, "HRV021"),
    ("akmola region", None, "KAZ001"),
    ("almaty region", None, "KAZ003"),
    ("altai krai", None, "RUS004"),
    ("turkana county", None, "KEN043"),
    ("blekinge county", None, "SWE001"),
    ("cauca department", None, "COL012"),
    ("ariana governorate", None, "TUN024"),
    ("bor district", None, "SRB001"),
    ("bamako district", None, "MLI001"),
    ("chechen republic", None, "RUS012"),
    ("federal district", None, "BRA007"),
    ("capital district", "VEN", "VEN011"),
    ("brussels capital region", None, "BEL001"),
    ("central region", "MWI", "MWI001"),
    ("Jalisco State", None, "MEX014"),
    ("Minsk Region", None, "BLR004"),
    ("Sofia Province", None, "BGR021"),
    ("Tashkent Region", None, "UZB013"),
    ("Lima Region", None, "PER015"),
    ("Maputo Province", None, "MOZ006"),
    ("Vientiane Province", None, "LAO013"),
    ("Equateur", None, None),
    ("Carolina", None, None),
    ("york", None, None),
    ("Mexico State", None, None),
    ("Mexico City", None, None),
    ("New Mexico", "Mexico", None),
    ("central region", None, None),
    ("federal capital territory", None, None),
    ("Atlantis", None, None),
    ("trade", None, None),
]


# (name, country hint) -> result, for recorded results changed on purpose:
# since the country guard refuses only names that are a country's code, name
# or alias (2026-10-16), Mexico City resolves without a hint (see
# TestCountryGuard).
_AMENDED_SINCE_RECORDING = {
    ("Mexico City", None): "MEX009",
}


@pytest.mark.parametrize("name, country, code", _RECORDED_BEFORE_RETRY)
def test_resolution_recorded_before_retry_unchanged(name, country, code):
    expected = _AMENDED_SINCE_RECORDING.get((name, country), code)
    assert resolve_adm1_code(name, country=country) == expected


class TestIsoCodeCoverage:
    """PR #50 guard: every ISO-3 code used in the bundled pericoupling
    data must resolve via ``resolve_country_code``.  This prevents the
    legacy/modern code drift that motivated the ISO-3 migration (e.g.
    data using ``COD`` while the resolver only knew legacy ``ZAR``).
    If a future data refresh introduces a code the resolver doesn't
    know, this fails loudly instead of silently dropping lookups.
    """

    def _data_dir(self):
        import metacouplingllm.knowledge.adm1_pericoupling as mod
        from pathlib import Path
        return Path(mod.__file__).resolve().parent.parent / "data"

    def test_all_adm1_iso_codes_resolve(self):
        import csv
        from metacouplingllm.knowledge.countries import resolve_country_code
        path = self._data_dir() / "pericoupled_adm1_edge_list.csv"
        isos = set()
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                isos.add(row["ISO_A3_A"])
                isos.add(row["ISO_A3_B"])
        unresolved = sorted(c for c in isos if resolve_country_code(c) is None)
        assert not unresolved, (
            f"ADM1 edge-list ISO codes not resolvable by "
            f"resolve_country_code: {unresolved}"
        )

    def test_all_adm0_iso_codes_resolve(self):
        import csv
        from metacouplingllm.knowledge.countries import resolve_country_code
        path = self._data_dir() / "PeriTelecoupling_clean.csv"
        isos = set()
        with open(path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                isos.add(row["Sending"])
                isos.add(row["Receiving"])
        unresolved = sorted(c for c in isos if resolve_country_code(c) is None)
        assert not unresolved, (
            f"ADM0 matrix ISO codes not resolvable by "
            f"resolve_country_code: {unresolved}"
        )

    def test_modern_iso_aliases_resolve(self):
        """The PR #50 modernized aliases map to current ISO-3."""
        from metacouplingllm.knowledge.countries import resolve_country_code
        assert resolve_country_code("DR Congo") == "COD"
        assert resolve_country_code("Serbia") == "SRB"
        assert resolve_country_code("Romania") == "ROU"
        assert resolve_country_code("Timor-Leste") == "TLS"
        assert resolve_country_code("Kosovo") == "XKX"

    def test_adm1_country_names_resolve(self):
        """Every country name of the ADM1 data resolves to its units'
        ISO code (get_adm1_info gives it as ``country_name``), except the
        names listed: "Democratic Republic of Congo" gave COG, "D. P. R.
        of Korea" KOR and "Puerto Rico (U.S.)" USA until 2026-10-16."""
        from metacouplingllm.knowledge.countries import resolve_country_code
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_metadata
        pairs = {
            (meta["country_name"], meta["iso_a3"])
            for meta in _adm1_metadata.values()
        }
        assert len(pairs) == 200
        expected = {
            # French Guiana's units are labelled France and coded GUF.
            ("France", "GUF"): "FRA",
            # Spanish territories, coded ESP, not countries of their own.
            ("Ceuta (Sp.)", "ESP"): None,
            ("Melilla (Sp.)", "ESP"): None,
            ("Peñón de Vélez de la Gomera (Sp.)", "ESP"): None,
            # Also the name of the territory's one unit (MAF001), which
            # resolve_adm1_code would refuse as a country's.
            ("Saint-Martin (Fr.)", "MAF"): None,
        }
        assert expected.keys() <= pairs
        wrong = [
            (name, iso, got) for name, iso in sorted(pairs)
            if (got := resolve_country_code(name))
            != expected.get((name, iso), iso)
        ]
        assert not wrong, wrong


class TestDeFactoBordersAdm1:
    """The de_facto_borders toggle for disputed-territory pairs (ADM1)."""

    def test_overlay_pairs_present_by_default(self):
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        # Province-level de-facto pairs across disputed tracts: Arunachal Pradesh
        # <-> Tibet; Northern <-> Quneitra (Golan); Guelmim-Oued Noun <->
        # Tiris-Zemmour (Western Sahara); Haa <-> Tibet (Doklam).
        assert is_adm1_pericoupled("IND003", "CHN029") is True
        assert is_adm1_pericoupled("ISR004", "SYR012") is True
        assert is_adm1_pericoupled("MAR005", "MRT012") is True
        assert is_adm1_pericoupled("BTN005", "CHN029") is True

    def test_overlay_pairs_are_cross_country(self):
        from metacouplingllm.knowledge.adm1_pericoupling import (
            lookup_adm1_pericoupling,
        )
        assert lookup_adm1_pericoupling("IND003", "CHN029").cross_country is True
        assert lookup_adm1_pericoupling("ISR004", "SYR012").cross_country is True
        assert lookup_adm1_pericoupling("MAR005", "MRT012").cross_country is True

    def test_overlay_pairs_absent_in_strict_view(self):
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        for a, b in [("IND003", "CHN029"), ("ISR004", "SYR012"),
                     ("MAR005", "MRT012"), ("BTN005", "CHN029")]:
            assert is_adm1_pericoupled(a, b) is True
            assert is_adm1_pericoupled(a, b, de_facto_borders=False) is False

    def test_overlay_pairs_across_abyei_and_latrun(self):
        # Campaign da1: Abyei (Natural Earth "Admin. by Sudan") and the Latrun
        # no-man's-land ("Admin. By Israel") are folded into their administrators,
        # which restores three province pairs in the de-facto view only.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        for a, b in [("ISR003", "PSE013"), ("SDN012", "SSD010"),
                     ("SDN013", "SSD010")]:
            assert is_adm1_pericoupled(a, b) is True
            assert is_adm1_pericoupled(a, b, de_facto_borders=False) is False

    def test_disputed_territory_without_wb_province_is_adm0_only(self):
        # Gilgit-Baltistan and Ladakh/J&K are disputed territories EXCLUDED from
        # WB's ADM1 layer (not provinces), so the CHN/PAK relationship has NO
        # subnational overlay pair — it is carried at ADM0 only.  PAK005 Khyber
        # Pakhtunkhwa does not administer Gilgit-Baltistan, so PAK005<->CHN028 is
        # not pericoupled at ADM1 in EITHER view.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        assert is_adm1_pericoupled("PAK005", "CHN028") is False
        assert (
            is_adm1_pericoupled("PAK005", "CHN028", de_facto_borders=False)
            is False
        )

    def test_already_adjacent_disputed_pair_stays_base(self):
        # Kenya (Turkana) and South Sudan (Eastern Equatoria) already share an
        # ~80 km border SW of the Ilemi Triangle, so they are a BASE pair —
        # present in BOTH views, NOT a de-facto overlay pair.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        assert is_adm1_pericoupled("KEN043", "SSD002") is True
        assert (
            is_adm1_pericoupled("KEN043", "SSD002", de_facto_borders=False)
            is True
        )

    def test_undisputed_pair_unaffected_by_toggle(self):
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        assert is_adm1_pericoupled("AFG001", "PAK005") is True
        assert (
            is_adm1_pericoupled("AFG001", "PAK005", de_facto_borders=False)
            is True
        )

    def test_strict_view_subtracts_only_overlay_neighbors(self):
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_cross_border_neighbors,
        )
        # ISR004 (Northern) gains exactly its Golan-tract neighbours under de-facto.
        defacto = get_cross_border_neighbors("ISR004")
        strict = get_cross_border_neighbors("ISR004", de_facto_borders=False)
        assert defacto - strict == {"SYR012", "SYR006", "LBN004"}
        # PAK005 has NO ADM1 overlay neighbour (CHN/PAK is ADM0-only), so the
        # de-facto and strict views are identical for it.
        assert get_cross_border_neighbors("PAK005") == get_cross_border_neighbors(
            "PAK005", de_facto_borders=False
        )


class TestCouplingStandardAdm1:
    """The coupling_standard toggle for water-separated pairs (ADM1)."""

    def test_no_bridge_pair_dropped_under_moderate_and_stringent(self):
        # COD007 Kinshasa <-> COG002 Brazzaville: ~9 km across the Congo, ferry
        # only (no fixed crossing) -> a water-only / no-bridge pair.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        assert is_adm1_pericoupled(
            "COD007", "COG002", coupling_standard="lenient"
        ) is True
        assert is_adm1_pericoupled(
            "COD007", "COG002", coupling_standard="moderate"
        ) is False
        assert is_adm1_pericoupled(
            "COD007", "COG002", coupling_standard="stringent"
        ) is False

    def test_default_standard_is_moderate(self):
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        # No coupling_standard argument == moderate.
        assert is_adm1_pericoupled("COD007", "COG002") is False

    def test_bridge_pair_kept_under_moderate_dropped_under_stringent(self):
        # CHN011 <-> RUS005 across the Amur: a fixed crossing exists.  (The
        # earlier example, CHN019 <-> RUS014 on the Argun, was ruled not
        # water-only on 2026-10-04 and is an ordinary edge.)
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        assert is_adm1_pericoupled(
            "CHN011", "RUS005", coupling_standard="lenient"
        ) is True
        assert is_adm1_pericoupled(
            "CHN011", "RUS005", coupling_standard="moderate"
        ) is True
        assert is_adm1_pericoupled(
            "CHN011", "RUS005", coupling_standard="stringent"
        ) is False
        for s in ("lenient", "moderate", "stringent"):
            assert is_adm1_pericoupled("CHN019", "RUS014", coupling_standard=s) is True

    def test_wide_river_overlay_pairs(self):
        # The 13-pair wide-river overlay (5 km re-screen + ground-truth): existing
        # edges reclassified as water-only; 11 still are (Niassa<->Ruvuma and
        # Shkoder<->Ulcinj were ruled not water-only on 2026-10-04 and are
        # ordinary edges).  Ruse<->Giurgiu has the Friendship Bridge (kept under
        # moderate); Silistra<->Calarasi is ferry-only.
        from metacouplingllm.knowledge import adm1_pericoupling as A
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        is_adm1_pericoupled("USA044", "MEX028")   # warm the loaders
        wide = [("BGR018", "ROU012"), ("BGR016", "ROU020"), ("BGR013", "ROU037"),
                ("BGR010", "ROU018"), ("BGR028", "ROU018"), ("BGR013", "ROU031"),
                ("BGR026", "ROU037"), ("BGR016", "ROU037"), ("DEU011", "LUX002"),
                ("PRK002", "RUS060"), ("BWA002", "ZMB109")]
        for a, b in wide:                          # all are water-only + edges
            assert frozenset({a, b}) in A._adm1_water_all
            assert frozenset({a, b}) in A._adm1_pairs
        for a, b in (("MOZ008", "TZA025"), ("ALB031", "MNE020")):   # ordinary edges now
            assert frozenset({a, b}) not in A._adm1_water_all
            assert frozenset({a, b}) in A._adm1_pairs
        # bridged Danube span (Ruse<->Giurgiu): kept moderate, dropped stringent
        assert is_adm1_pericoupled("BGR016", "ROU020", coupling_standard="moderate") is True
        assert is_adm1_pericoupled("BGR016", "ROU020", coupling_standard="stringent") is False
        # ferry-only Danube span (Silistra<->Calarasi): dropped under moderate
        assert is_adm1_pericoupled("BGR018", "ROU012", coupling_standard="lenient") is True
        assert is_adm1_pericoupled("BGR018", "ROU012", coupling_standard="moderate") is False

    def test_lake_overlay_pairs(self):
        # 63-pair lake overlay: audited lake-only pairs restored as edges so
        # coupling_standard governs lakes exactly like rivers.  Bridged lake
        # pairs are kept under moderate (Houtribdijk); unbridged ones are
        # lenient-only (mid-lake meetings).
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        # Flevoland <-> Noord-Holland: Houtribdijk/Stichtse Brug -> moderate
        assert is_adm1_pericoupled("NLD002", "NLD008", coupling_standard="lenient") is True
        assert is_adm1_pericoupled("NLD002", "NLD008", coupling_standard="moderate") is True
        assert is_adm1_pericoupled("NLD002", "NLD008", coupling_standard="stringent") is False
        # Sodermanland <-> Uppsala across Malaren: ruled not water-only on
        # 2026-10-04 (the county line crosses Marson island) -> an ordinary
        # edge, pericoupled under every standard
        for s in ("lenient", "moderate", "stringent"):
            assert is_adm1_pericoupled("SWE014", "SWE016", coupling_standard=s) is True
        # Flevoland <-> Utrecht: no direct fixed link -> lenient only
        assert is_adm1_pericoupled("NLD002", "NLD010", coupling_standard="lenient") is True
        assert is_adm1_pericoupled("NLD002", "NLD010", coupling_standard="moderate") is False
        # Illinois <-> Michigan across Lake Michigan: no crossing -> lenient only
        assert is_adm1_pericoupled("USA014", "USA023", coupling_standard="lenient") is True
        assert is_adm1_pericoupled("USA014", "USA023", coupling_standard="moderate") is False
        # Tanganyika <-> Kigoma across Lake Tanganyika: lenient only
        assert is_adm1_pericoupled("COD030", "TZA010", coupling_standard="lenient") is True
        assert is_adm1_pericoupled("COD030", "TZA010", coupling_standard="moderate") is False
        # Jogeva <-> Pskov across Lake Peipus: REMOVED 2026-07-22 by
        # maintainer ruling (WB signature: Tartu-Jogeva-Pskov corner contact,
        # no shared segment) -- not pericoupled under any standard
        assert is_adm1_pericoupled("EST006", "RUS061", coupling_standard="lenient") is False
        assert is_adm1_pericoupled("EST006", "RUS061", coupling_standard="moderate") is False

    def test_audit_water_overlay_pairs(self):
        # The former 3-pair audit-water overlay after the ru1 river-screen
        # unification (2026-07-21): Shirak <-> Kars was demoted to a mixed
        # land border by maintainer map ruling (measured 0.67 water at 500 m),
        # so it is pericoupled under every standard; the other two rows now
        # ship in the water classification file (ru1 fold), still lenient-only.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        for std in ("lenient", "moderate", "stringent"):
            assert is_adm1_pericoupled("ARM007", "TUR045", coupling_standard=std) is True
        # Vratca <-> Olt (Danube, ferry only; measured 1.00 water at 500 m)
        # and Kagera <-> Ntungamo (Kagera-thalweg arc, no crossing).
        for a, b in [("BGR028", "ROU031"), ("TZA006", "UGA094")]:
            assert is_adm1_pericoupled(a, b, coupling_standard="lenient") is True
            assert is_adm1_pericoupled(a, b, coupling_standard="moderate") is False
            assert is_adm1_pericoupled(a, b, coupling_standard="stringent") is False

    def test_hydro_water_overlay_pairs(self):
        # Hydro-water rows: water-only borders the NE screens missed, found by
        # the HydroRIVERS full-database cross-check and human-verified
        # (2026-07-02); carried in the water classification file since the
        # 2026-07-28 consolidation. Spot-check one bridged and one unbridged
        # pair, plus the two pairs the same audit REMOVED from the water set.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        # Brandenburg <-> Lubuskie: Oder/Neisse, fixed crossings -> moderate keeps it.
        assert is_adm1_pericoupled("DEU004", "POL005", coupling_standard="lenient") is True
        assert is_adm1_pericoupled("DEU004", "POL005", coupling_standard="moderate") is True
        assert is_adm1_pericoupled("DEU004", "POL005", coupling_standard="stringent") is False
        # French Guiana <-> Sipaliwini: Maroni system, ferry only -> lenient-only.
        assert is_adm1_pericoupled("GUF001", "SUR009", coupling_standard="moderate") is False
        # Human-verified NOT water-only -> ordinary land edges under every standard.
        for a, b in [("HRV010", "HUN001"), ("SSD002", "UGA081")]:
            assert is_adm1_pericoupled(a, b, coupling_standard="stringent") is True

    def test_land_gap_overlay_pairs(self):
        # 2-pair land-gap overlay: genuine survey-line borders hidden by the
        # cross-source offset corridor (Kenya-Tanzania), map-verified and
        # restored as ordinary LAND edges -> pericoupled under every standard.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        for s in ("lenient", "moderate", "stringent"):
            assert is_adm1_pericoupled("KEN010", "TZA011", coupling_standard=s) is True
            assert is_adm1_pericoupled("KEN033", "TZA016", coupling_standard=s) is True

    def test_land_pair_unaffected_by_standard(self):
        # AFG001 <-> PAK005 is a land border -> pericoupled under every standard.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        for s in ("lenient", "moderate", "stringent"):
            assert is_adm1_pericoupled(
                "AFG001", "PAK005", coupling_standard=s
            ) is True

    def test_invalid_standard_raises(self):
        import pytest
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        with pytest.raises(ValueError):
            is_adm1_pericoupled("COD007", "COG002", coupling_standard="loose")

    def test_neighbors_filtered_by_standard(self):
        # The no-bridge Congo crossing is removed from COD007's moderate
        # neighbour set but kept under lenient.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_neighbors,
        )
        assert "COG002" in get_adm1_neighbors(
            "COD007", coupling_standard="lenient"
        )
        assert "COG002" not in get_adm1_neighbors(
            "COD007", coupling_standard="moderate"
        )


# ---------------------------------------------------------------------------
# Alias table tests (PR #60)
# ---------------------------------------------------------------------------

_ALIAS_CSV = (
    Path(__file__).parent.parent / "src" / "metacouplingllm" / "data" / "adm1_aliases.csv"
)
_alias_csv_present = pytest.mark.skipif(
    not _ALIAS_CSV.is_file(),
    reason="adm1_aliases.csv not yet generated — run scripts/build_adm1_aliases.py first",
)


class TestAdm1AliasLoader:
    """Tests for the alias loader and Strategy 0 wiring; run without the CSV."""

    def test_get_adm1_aliases_returns_list(self):
        from metacouplingllm.knowledge.adm1_pericoupling import get_adm1_aliases
        result = get_adm1_aliases("DEU002")
        assert isinstance(result, list)

    def test_get_adm1_aliases_unknown_code(self):
        from metacouplingllm.knowledge.adm1_pericoupling import get_adm1_aliases
        assert get_adm1_aliases("BOGUS999") == []

    def test_strategy0_bypassed_when_aliases_empty(self, monkeypatch):
        """Empty alias dict → Strategy 0 is a no-op; existing resolution unaffected."""
        import metacouplingllm.knowledge.adm1_pericoupling as mod
        monkeypatch.setattr(mod, "_adm1_aliases", {})
        assert mod.resolve_adm1_code("Michigan") == "USA023"

    def test_strategy0_country_filter_respected(self, monkeypatch):
        """Alias entry in DEU: correct hint → code; wrong hint → None (falls through, not found)."""
        import metacouplingllm.knowledge.adm1_pericoupling as mod
        fake_key = "zztestaliaszz"
        monkeypatch.setattr(mod, "_adm1_aliases", {fake_key: [("DEU002", "DEU")]})
        assert mod.resolve_adm1_code(fake_key, country="DEU") == "DEU002"
        assert mod.resolve_adm1_code(fake_key, country="ITA") is None

    def test_strategy0_no_hint_single_candidate(self, monkeypatch):
        """A globally unique alias (one entry, no hint) resolves directly."""
        import metacouplingllm.knowledge.adm1_pericoupling as mod
        fake_key = "zztestaliaszz"
        monkeypatch.setattr(mod, "_adm1_aliases", {fake_key: [("DEU002", "DEU")]})
        assert mod.resolve_adm1_code(fake_key) == "DEU002"

    def test_strategy0_folded_key_branch(self, monkeypatch):
        """An accented caller resolves against an ASCII-folded alias key
        (PR #61 — the documented 'Múnich' path; all shipped keys are folded)."""
        import metacouplingllm.knowledge.adm1_pericoupling as mod
        monkeypatch.setattr(mod, "_adm1_aliases", {"munich": [("DEU002", "DEU")]})
        assert mod.resolve_adm1_code("Múnich") == "DEU002"
        assert mod.resolve_adm1_code("munich") == "DEU002"

    def test_loader_graceful_on_malformed_csv(self, tmp_path, monkeypatch):
        """A malformed alias CSV degrades to {} (no Strategy 0), never raises --
        otherwise a corrupt file would break ALL ADM1 lookups via _ensure_loaded
        (PR #61)."""
        import metacouplingllm.knowledge.adm1_pericoupling as mod
        bad = tmp_path / "adm1_aliases.csv"
        # Missing the required 'iso_a3' column + a short row.
        bad.write_text("alias_key,code\nbavaria,DEU002\nx\n", encoding="utf-8")

        class _FakePath:
            def __truediv__(self, other):
                return bad
        monkeypatch.setattr(mod.resources, "files", lambda _pkg: _FakePath())
        # Must not raise; missing iso_a3 column → every row skipped → {}.
        assert mod._load_adm1_aliases() == {}


@_alias_csv_present
class TestAdm1AliasTable:
    """Data-integrity tests over the shipped adm1_aliases.csv."""

    @pytest.fixture(scope="class")
    def rows(self):
        with open(_ALIAS_CSV, newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))

    @pytest.fixture(scope="class")
    def adm1_meta(self):
        from metacouplingllm.knowledge.adm1_pericoupling import _load_adm1_data
        _, _, _, meta = _load_adm1_data()
        return meta

    def test_keys_unique(self, rows):
        keys = [r["alias_key"] for r in rows]
        assert len(keys) == len(set(keys)), "Duplicate alias_key values in shipped CSV"

    def test_codes_valid(self, rows, adm1_meta):
        bad = [r["alias_key"] for r in rows if r["code"] not in adm1_meta]
        assert not bad, f"Unknown ADM1 codes: {bad[:5]}"

    def test_no_country_name_keys(self, rows):
        """No key is a country's ISO code, name or alias (rule G)."""
        from metacouplingllm.knowledge.countries import resolve_country_code
        bad = [
            r["alias_key"] for r in rows
            if resolve_country_code(r["alias_key"], substring=False)
        ]
        assert not bad, f"Alias keys that are country names: {bad[:5]}"

    def test_country_named_keys_resolve_without_a_hint(self, rows):
        """42 keys contain a country name or demonym, which
        resolve_country_code finds inside them: four curated ("mexico
        city", "lesser poland") and the 38 generated ones rule G admitted
        on 2026-10-16 ("upper austria", "inner mongolia").  None is a
        country's name, so each resolves with or without a country hint."""
        from metacouplingllm.knowledge.countries import resolve_country_code
        named = [r for r in rows if resolve_country_code(r["alias_key"])]
        assert len(named) == 42
        failed = [
            (r["alias_key"], r["code"]) for r in named
            if resolve_adm1_code(r["alias_key"], country=r["iso_a3"])
            != r["code"]
            or resolve_adm1_code(r["alias_key"]) != r["code"]
        ]
        assert not failed, failed

    def test_rule_g_admissions_reviewed(self, rows):
        """When rule G came to drop only keys that ARE a country's name
        (2026-10-16), it admitted 50 generated aliases; the 38 that passed
        review ship and the 12 rejected (_REVIEW_DENYLIST) do not."""
        shipped = {
            "lower austria": "AUT003", "upper austria": "AUT004",
            "federation of bosnia and herzegovina": "BIH001",
            "brunei-muara": "BRN002",
            "magallanes and chilean antarctica": "CHL011",
            "inner mongolia": "CHN019",
            "inner mongolia autonomous region": "CHN019",
            "capital region of denmark": "DNK001",
            "central denmark": "DNK002", "central denmark region": "DNK002",
            "north denmark region": "DNK003",
            "region of southern denmark": "DNK004",
            "southern denmark": "DNK004",
            "south west ethiopia peoples region": "ETH012",
            "eastern macedonia and thrace": "GRC001",
            "western greece": "GRC003", "western macedonia": "GRC004",
            "central macedonia": "GRC007", "central greece": "GRC011",
            "north greenland": "GRL001", "east greenland": "GRL002",
            "west greenland": "GRL003",
            "east azerbaijan": "IRN004", "west azerbaijan": "IRN026",
            "north kazakhstan": "KAZ012", "north kazakhstan region": "KAZ012",
            "east kazakhstan": "KAZ013", "east kazakhstan region": "KAZ013",
            "south kazakhstan": "KAZ014", "south kazakhstan region": "KAZ014",
            "west kazakhstan": "KAZ015", "west kazakhstan region": "KAZ015",
            "rabat malta": "MLT044",
            "north holland": "NLD008", "south holland": "NLD012",
            "west panama": "PAN012",
            "southeast slovenia": "SVN003", "central slovenia": "SVN007",
        }
        rejected = {
            "kuwait city", "luxembourg city", "panama city", "french riviera",
            "northern kazakhstan", "eastern kazakhstan", "western kazakhstan",
            "southwest ethiopia", "north lebanon", "south lebanon",
            "lower congo", "isle of france",
        }
        by_key = {r["alias_key"]: r["code"] for r in rows}
        assert len(shipped) == 38
        assert {k: by_key.get(k) for k in shipped} == shipped
        assert not rejected & by_key.keys()

    def test_no_u3_collision(self, rows):
        from metacouplingllm.knowledge.adm1_pericoupling import _get_adm1_name_index
        name_index = _get_adm1_name_index()
        bad = []
        for r in rows:
            existing = name_index.get(r["alias_key"])
            if existing:
                existing_codes = {c for c, _ in existing}
                if existing_codes != {r["code"]}:
                    bad.append(r["alias_key"])
        assert not bad, f"Alias keys shadow canonical names for different codes: {bad[:5]}"

    def test_at_most_3_per_code(self, rows):
        from collections import Counter
        counts = Counter(r["code"] for r in rows)
        violations = [(code, n) for code, n in counts.items() if n > 3]
        assert not violations, f"More than 3 aliases for code(s): {violations[:5]}"

    def test_round_trip_all_rows(self, rows):
        """Every shipped alias must resolve back to its code under the country hint."""
        failed = []
        for r in rows:
            result = resolve_adm1_code(r["alias_key"], country=r["iso_a3"])
            if result != r["code"]:
                failed.append((r["alias_key"], r["iso_a3"], r["code"], result))
        assert not failed, (
            f"{len(failed)} alias round-trips failed; first 3: {failed[:3]}"
        )

    def test_no_hint_resolution_for_unique_rows(self, rows):
        """PR #60's headline feature: a globally-unique alias resolves with NO
        country hint.  U2 guarantees every shipped key is globally unique, so
        each must resolve to its own code hint-free (Strategy 0 + no-hint
        _pick_best_candidate)."""
        failed = [
            (r["alias_key"], r["code"], resolve_adm1_code(r["alias_key"]))
            for r in rows
            if resolve_adm1_code(r["alias_key"]) != r["code"]
        ]
        assert not failed, f"{len(failed)} no-hint resolutions failed; first 3: {failed[:3]}"

    def test_no_folded_shadow(self, rows):
        """No shipped alias key may shadow a DIFFERENT region in the accent-
        FOLDED name index.  This is the fold-aware invariant PR #61 added (the
        original U3 only checked the accented index, which let 'ash sharqiyah'
        → OMN002 shadow Saudi SAU008 and resolve to Oman with no hint)."""
        from metacouplingllm.knowledge.adm1_pericoupling import (
            _get_adm1_folded_name_index, _fold_diacritics,
        )
        folded = _get_adm1_folded_name_index()
        bad = []
        for r in rows:
            entry = folded.get(_fold_diacritics(r["alias_key"]))
            if entry:
                codes = {c for c, _ in entry}
                if codes != {r["code"]}:
                    bad.append((r["alias_key"], r["code"], sorted(codes)))
        assert not bad, f"Alias keys shadow a folded canonical name for a different code: {bad[:5]}"

    def test_ash_sharqiyah_no_longer_resolves_to_oman(self):
        """Regression for the PR #60 folded-collision HIGH bug: the unaccented
        English spelling of Saudi Arabia's Ash Sharqīyah (SAU008) must not
        confidently resolve to Oman (OMN002) with no hint."""
        result = resolve_adm1_code("ash sharqiyah")
        assert result != "OMN002"
        assert result == "SAU008"

    def test_reviewed_bad_aliases_absent(self, rows):
        """The Opus-reviewed wrong-region / out-of-scope aliases were removed."""
        keys = {r["alias_key"] for r in rows}
        for k in ("loire valley", "jubaland", "mysore", "smyrna", "new delhi",
                  "mosul", "golan", "cappadocia", "gbao", "snnpr", "silent city",
                  "ash sharqiyah", "sharqiyah"):
            assert k not in keys, f"reviewed-bad alias still present: {k}"

    def test_kept_former_names_present(self, rows):
        """Four still-ubiquitous former names were deliberately kept (PR #61)."""
        by_key = {r["alias_key"]: r["code"] for r in rows}
        assert by_key.get("saigon") == "VNM029"
        assert by_key.get("rangoon") == "MMR015"
        assert by_key.get("orissa") == "IND026"
        assert by_key.get("pondicherry") == "IND027"

    def test_counts_match_docs(self, rows):
        """Headline provenance counts must match the shipped CSV so docs cannot
        drift (PR #61 — the original '67 countries / 886 regions' were wrong)."""
        assert len(rows) == 1201
        assert len({r["code"] for r in rows}) == 897
        assert len({r["iso_a3"] for r in rows}) == 138

    def test_curated_aliases_shipped(self, rows):
        """The 18 curated aliases of 2026-10-16 are shipped and marked
        'curated' in the audit table, the only rows so marked."""
        curated = {
            "mexico city": "MEX009", "silesian voivodeship": "POL012",
            "lower silesian": "POL001", "lublin": "POL004",
            "lubelskie": "POL004", "lubusz": "POL005",
            "lesser poland": "POL006", "malopolskie": "POL006",
            "greater poland": "POL015", "zachodniopomorskie": "POL016",
            "north-west district": "BWA007", "barisal": "BGD001",
            "arica and parinacota": "CHL004", "arica y parinacota": "CHL004",
            "libertador general bernardo o'higgins": "CHL008",
            "aysen del general carlos ibanez del campo": "CHL001",
            "magallanes and the chilean antarctica": "CHL011",
            "magallanes y de la antartica chilena": "CHL011",
        }
        by_key = {r["alias_key"]: r["code"] for r in rows}
        assert {k: by_key.get(k) for k in curated} == curated
        audit = Path(__file__).parent.parent / "docs" / "adm1_aliases_audit.csv"
        with open(audit, newline="", encoding="utf-8-sig") as fh:
            marked = {
                r["alias_key"]: r["code"] for r in csv.DictReader(fh)
                if r["source"] == "curated"
            }
        assert marked == curated


def test_u3_shadow_predicate_positive_control():
    """Positive control for the shadow-detection logic: a synthetic alias key
    that equals a canonical name for a DIFFERENT code must be flagged (guards
    against the data-driven U3/folded-shadow tests passing vacuously)."""
    from metacouplingllm.knowledge.adm1_pericoupling import _get_adm1_name_index
    name_index = _get_adm1_name_index()
    # 'bavaria' is an exonym, never a WB canonical key; 'michigan' IS one.
    canonical_key = "michigan"
    assert canonical_key in name_index
    existing_codes = {c for c, _ in name_index[canonical_key]}
    # Pretend an alias claimed this key for a different code → must be a shadow.
    assert existing_codes != {"DEU002"}, "predicate would not flag a real shadow"


@_alias_csv_present
class TestAdm1AliasStrategy0:
    """Spot-check Strategy 0 for canonical English exonyms."""

    def test_bavaria_with_country_hint(self):
        # Bayern (DEU002) → Bavaria
        assert resolve_adm1_code("Bavaria", country="DEU") == "DEU002"

    def test_bavaria_no_hint(self):
        # Bavaria is globally unique → resolves without a country hint
        assert resolve_adm1_code("Bavaria") == "DEU002"

    def test_bavaria_wrong_country_returns_none(self):
        # Bavaria with an ITA hint: Strategy 0 country filter rejects it,
        # and no other strategy finds "bavaria" in ITA data.
        assert resolve_adm1_code("Bavaria", country="ITA") is None

    def test_saxony_resolves(self):
        # Sachsen (DEU013) → Saxony
        assert resolve_adm1_code("Saxony", country="DEU") == "DEU013"

    def test_get_adm1_aliases_bavaria(self):
        from metacouplingllm.knowledge.adm1_pericoupling import get_adm1_aliases
        aliases = get_adm1_aliases("DEU002")
        assert "bavaria" in aliases
