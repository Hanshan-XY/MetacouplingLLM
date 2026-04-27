"""Tests for knowledge/adm1_pericoupling.py — ADM1 pericoupling database lookup."""

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
        assert len(_adm1_pairs) == 8290, (
            f"Expected 8290 pairs, got {len(_adm1_pairs)}"
        )

    def test_expected_code_count(self):
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country
        assert _adm1_country is not None
        assert len(_adm1_country) == 3366, (
            f"Expected 3366 unique ADM1 codes, got {len(_adm1_country)}"
        )

    def test_expected_country_count(self):
        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country
        assert _adm1_country is not None
        countries = set(_adm1_country.values())
        assert len(countries) == 195, (
            f"Expected 195 unique countries, got {len(countries)}"
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
