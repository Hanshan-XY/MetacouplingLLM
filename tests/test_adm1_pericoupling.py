"""Tests for knowledge/adm1_pericoupling.py — ADM1 pericoupling database lookup."""

import csv
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
        assert len(_adm1_pairs) == 8381, (
            f"Expected 8381 pairs, got {len(_adm1_pairs)}"
        )

    def test_expected_code_count(self):
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country
        assert _adm1_country is not None
        assert len(_adm1_country) == 3373, (
            f"Expected 3373 unique ADM1 codes, got {len(_adm1_country)}"
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
        México) instead of CDMX, and 'New Mexico' (a US state) likewise."""
        assert resolve_adm1_code("Mexico City", country="Mexico") is None
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
        # CHN019 <-> RUS014 across the Argun: a fixed crossing exists.
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )
        assert is_adm1_pericoupled(
            "CHN019", "RUS014", coupling_standard="lenient"
        ) is True
        assert is_adm1_pericoupled(
            "CHN019", "RUS014", coupling_standard="moderate"
        ) is True
        assert is_adm1_pericoupled(
            "CHN019", "RUS014", coupling_standard="stringent"
        ) is False

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
        from metacouplingllm.knowledge.countries import resolve_country_code
        bad = [r["alias_key"] for r in rows if resolve_country_code(r["alias_key"]) is not None]
        assert not bad, f"Alias keys that are country names: {bad[:5]}"

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
