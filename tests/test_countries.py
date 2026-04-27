"""Tests for knowledge/countries.py — ISO country code resolution."""

from metacouplingllm.knowledge.countries import (
    ISO_ALPHA3_NAMES,
    get_country_name,
    resolve_country_code,
)


class TestResolveCountryCode:
    """Test resolve_country_code with various input types."""

    def test_exact_iso_code_upper(self):
        assert resolve_country_code("USA") == "USA"
        assert resolve_country_code("MEX") == "MEX"
        assert resolve_country_code("CAN") == "CAN"

    def test_exact_iso_code_lower(self):
        assert resolve_country_code("usa") == "USA"
        assert resolve_country_code("mex") == "MEX"

    def test_exact_iso_code_mixed(self):
        assert resolve_country_code("Usa") == "USA"
        assert resolve_country_code("bra") == "BRA"

    def test_common_aliases(self):
        assert resolve_country_code("US") == "USA"
        assert resolve_country_code("UK") == "GBR"
        assert resolve_country_code("America") == "USA"

    def test_full_country_names(self):
        assert resolve_country_code("United States") == "USA"
        assert resolve_country_code("Mexico") == "MEX"
        assert resolve_country_code("Canada") == "CAN"
        assert resolve_country_code("Brazil") == "BRA"
        assert resolve_country_code("China") == "CHN"
        assert resolve_country_code("Germany") == "DEU"
        assert resolve_country_code("Guatemala") == "GTM"

    def test_case_insensitive_names(self):
        assert resolve_country_code("united states") == "USA"
        assert resolve_country_code("MEXICO") == "MEX"
        assert resolve_country_code("canada") == "CAN"

    def test_alternative_names(self):
        assert resolve_country_code("Russia") == "RUS"
        assert resolve_country_code("Iran") == "IRN"
        assert resolve_country_code("South Korea") == "KOR"
        assert resolve_country_code("Ivory Coast") == "CIV"

    def test_substring_match(self):
        """LLM outputs like 'Ethiopian coffee regions' should resolve."""
        assert resolve_country_code("Ethiopian coffee regions") == "ETH"
        assert resolve_country_code("Mexican avocado farms") == "MEX"
        assert resolve_country_code("Brazilian soybean regions") == "BRA"

    def test_does_not_match_inside_other_words(self):
        assert resolve_country_code("Indiana") is None
        assert resolve_country_code(
            "Michigan and Indiana are pericoupled"
        ) is None
        assert resolve_country_code(
            "Great Lakes ports, interstate corridors, seaports used for export."
        ) is None

    def test_matches_standalone_country_terms_inside_sentences(self):
        assert resolve_country_code("Exports to United States markets") == "USA"
        assert resolve_country_code("Trade with China increased") == "CHN"

    def test_returns_none_for_unknown(self):
        assert resolve_country_code("Planet Mars") is None
        assert resolve_country_code("Atlantis") is None
        assert resolve_country_code("") is None

    def test_whitespace_handling(self):
        assert resolve_country_code("  USA  ") == "USA"
        assert resolve_country_code("  Mexico  ") == "MEX"

    def test_none_like_inputs(self):
        assert resolve_country_code("") is None


class TestGetCountryName:
    """Test get_country_name lookup."""

    def test_known_codes(self):
        assert get_country_name("USA") == "United States"
        assert get_country_name("MEX") == "Mexico"
        assert get_country_name("CAN") == "Canada"
        assert get_country_name("BRA") == "Brazil"
        assert get_country_name("CHN") == "China"
        assert get_country_name("GTM") == "Guatemala"

    def test_case_insensitive(self):
        assert get_country_name("usa") == "United States"
        assert get_country_name("mex") == "Mexico"

    def test_unknown_returns_code(self):
        assert get_country_name("XYZ") == "XYZ"


class TestISOAlpha3Names:
    """Validate the code dictionary structure."""

    def test_is_dict(self):
        assert isinstance(ISO_ALPHA3_NAMES, dict)

    def test_has_key_countries(self):
        for code in ("USA", "MEX", "CAN", "BRA", "CHN", "GTM", "DEU"):
            assert code in ISO_ALPHA3_NAMES

    def test_all_codes_are_three_letters(self):
        for code in ISO_ALPHA3_NAMES:
            assert len(code) == 3
            assert code.isalpha()
            assert code.isupper()
