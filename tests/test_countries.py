"""Tests for knowledge/countries.py — ISO country code resolution."""

import pytest

from metacouplingllm.knowledge.countries import (
    ISO_ALPHA3_NAMES,
    _ALIASES,
    _normalize_country_text,
    expand_supranational,
    find_country_codes,
    get_country_name,
    resolve_country_code,
    supranational_display_name,
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


class TestNormalization:
    """Accents, curly apostrophes and repeated spaces do not matter; until
    2026-10-16 "México" and the other accented names returned None."""

    @pytest.mark.parametrize("name, code", [
        ("México", "MEX"),
        ("Perú", "PER"),
        ("Panamá", "PAN"),
        ("Curaçao", "CUW"),
        ("Réunion", "REU"),
        ("Côte d'Ivoire", "CIV"),
        ("Türkiye", "TUR"),
        ("São Tomé and Príncipe", "STP"),
        ("Åland Islands", "ALA"),
    ])
    def test_accents_are_folded(self, name, code):
        assert resolve_country_code(name) == code
        assert resolve_country_code(name, substring=False) == code

    def test_accents_are_folded_in_free_text(self):
        assert resolve_country_code("Aguacates de México") == "MEX"

    def test_curly_apostrophes(self):
        assert resolve_country_code("Côte d’Ivoire") == "CIV"
        assert resolve_country_code(
            "Lao People’s Democratic Republic"
        ) == "LAO"

    def test_repeated_spaces(self):
        assert resolve_country_code("United   States") == "USA"
        assert resolve_country_code("Exports to  United  States") == "USA"

    def test_alias_keys_are_written_normalized(self):
        """Lookups compare normalized text with the keys as written."""
        assert all(_normalize_country_text(k) == k for k in _ALIASES)


class TestNewAliases:
    """Names that returned None, or another country, until 2026-10-16."""

    def test_names_in_the_countrys_own_language(self):
        assert resolve_country_code("España") == "ESP"
        assert resolve_country_code("Espana") == "ESP"
        assert resolve_country_code("Brasil") == "BRA"

    def test_bangladeshi(self):
        assert resolve_country_code("Bangladeshi") == "BGD"
        assert resolve_country_code("Bangladeshi rice farmers") == "BGD"

    @pytest.mark.parametrize("name, code", [
        # api.worldbank.org/v2/country
        ("Congo, Dem. Rep.", "COD"),
        ("Korea, Dem. People's Rep.", "PRK"),
        ("Kyrgyz Republic", "KGZ"),
        ("Slovak Republic", "SVK"),
        ("St. Kitts and Nevis", "KNA"),
        ("St. Lucia", "LCA"),
        ("St. Vincent and the Grenadines", "VCT"),
        ("St. Martin (French part)", "MAF"),
        ("Virgin Islands (U.S.)", "VIR"),
        ("Naoero", "NRU"),
        # The country field of the World Bank boundary data
        ("Democratic Republic of Congo", "COD"),
        ("D. P. R. of Korea", "PRK"),
        ("Lao People's Democratic Republic", "LAO"),
    ])
    def test_world_bank_names(self, name, code):
        assert resolve_country_code(name) == code
        assert resolve_country_code(name, substring=False) == code

    def test_world_bank_names_keep_the_two_congos_and_koreas_apart(self):
        assert resolve_country_code("Congo, Rep.") == "COG"
        assert resolve_country_code("Korea, Rep.") == "KOR"
        assert resolve_country_code(
            "Cobalt from the Democratic Republic of Congo"
        ) == "COD"
        assert resolve_country_code("Exports to Congo, Dem. Rep.") == "COD"

    @pytest.mark.parametrize("name, code", [
        # Debian's iso-codes data (the names pycountry gives)
        ("Korea, Democratic People's Republic of", "PRK"),
        ("Congo, The Democratic Republic of the", "COD"),
        ("Virgin Islands, British", "VGB"),
        ("Virgin Islands, U.S.", "VIR"),
        ("Virgin Islands of the United States", "VIR"),
        # Wikipedia's table of ISO 3166-1
        ("Congo, Democratic Republic of the", "COD"),
        ("Virgin Islands (British)", "VGB"),
    ])
    def test_iso_3166_names(self, name, code):
        """They gave KOR, COG, GBR and USA."""
        assert resolve_country_code(name) == code
        assert resolve_country_code(name, substring=False) == code

    def test_iso_3166_names_keep_the_two_congos_and_koreas_apart(self):
        assert resolve_country_code("Congo") == "COG"
        assert resolve_country_code("Korea, Republic of") == "KOR"
        assert resolve_country_code(
            "Imports from Korea, Democratic People’s Republic of"
        ) == "PRK"

    def test_puerto_rico(self):
        """Puerto Rico has its own ISO code, which the pericoupling
        databases and the boundary data of both maps use."""
        assert resolve_country_code("Puerto Rico") == "PRI"
        # The World Bank's boundary data and its country list
        assert resolve_country_code("Puerto Rico (U.S.)") == "PRI"
        assert resolve_country_code("Puerto Rico (US)") == "PRI"
        assert resolve_country_code("Coffee farms in Puerto Rico") == "PRI"

    def test_alias_keys_are_not_repeated(self):
        """A dict literal keeps the last value of a repeated key: a second
        "puerto rico" alias, USA, replaced the first, PRI."""
        import ast
        from pathlib import Path

        import metacouplingllm.knowledge.countries as countries

        tree = ast.parse(Path(countries.__file__).read_text(encoding="utf-8"))
        (table,) = [
            node.value for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "_ALIASES"
        ]
        keys = [key.value for key in table.keys]
        assert sorted(k for k in set(keys) if keys.count(k) > 1) == []

    @pytest.mark.parametrize("name, code, before", [
        ("North Korean", "PRK", "KOR"),
        ("DPR Korea", "PRK", "KOR"),
        ("South Sudanese", "SSD", "SDN"),
        ("Guinea Bissau", "GNB", "GIN"),
    ])
    def test_names_a_shorter_alias_read_as_another_country(
        self, name, code, before,
    ):
        """The shorter alias inside gave *before*."""
        assert resolve_country_code(name) == code
        assert resolve_country_code(name, substring=False) == code

    def test_lao(self):
        """The adjective of Laos, which free text no longer reads through
        the code LAO (codes count there only in capitals)."""
        assert resolve_country_code("Lao") == "LAO"
        assert resolve_country_code("Lao", free_text=True) == "LAO"
        assert resolve_country_code(
            "the Lao government", free_text=True
        ) == "LAO"


class TestWithoutSubstring:
    """``substring=False`` accepts only a whole code, name or alias."""

    def test_whole_names_resolve(self):
        assert resolve_country_code("Mexico", substring=False) == "MEX"
        assert resolve_country_code("mex", substring=False) == "MEX"
        assert resolve_country_code("Mexican", substring=False) == "MEX"
        assert resolve_country_code("Georgia", substring=False) == "GEO"
        assert resolve_country_code("Est", substring=False) == "EST"

    @pytest.mark.parametrize("text, found", [
        ("New South Wales", "GBR"),
        ("New Mexico", "MEX"),
        ("Mexico City", "MEX"),
        ("Northern Ireland", "IRL"),
        ("Port of Spain", "ESP"),
        ("Thai Binh", "THA"),
        ("Ethiopian coffee regions", "ETH"),
    ])
    def test_text_containing_a_country_does_not(self, text, found):
        assert resolve_country_code(text) == found
        assert resolve_country_code(text, substring=False) is None

    def test_leading_the(self):
        """English writes some country names with "the"."""
        assert resolve_country_code("the Gambia", substring=False) == "GMB"
        assert resolve_country_code("The Bahamas", substring=False) == "BHS"
        assert resolve_country_code(
            "the United States", substring=False
        ) == "USA"
        assert resolve_country_code("the Mexico", substring=False) == "MEX"
        assert resolve_country_code("The Hague", substring=False) is None


class TestFreeText:
    """``free_text=True``: a passage of text, not a name."""

    @pytest.mark.parametrize("text, found", [
        # Region names of the ADM1 database
        ("New South Wales", "GBR"),
        ("Northern Ireland", "IRL"),
        ("Port of Spain", "ESP"),
        ("Port-of-Spain", "ESP"),
        ("Thai Binh", "THA"),
        ("Thái Bình province", "THA"),
        ("New Mexico", "MEX"),
        ("New Jersey", "JEY"),
        ("British Columbia", "GBR"),
        ("East New Britain", "GBR"),
        ("Inner Mongolia", "MNG"),
        ("East Azerbaijan", "AZE"),
        ("Central Macedonia", "MKD"),
        # Continents and world regions
        ("South America", "USA"),
        ("soy from South America", "USA"),
        ("Latin American markets", "USA"),
        ("Central American receiving markets", "USA"),
        ("Indo-China", "CHN"),
        ("the British Isles", "GBR"),
        # Seas, lakes, basins, deltas, peninsulas
        ("the Gulf of Mexico", "MEX"),
        ("Indian Ocean islands", "IND"),
        ("South China Sea", "CHN"),
        ("the Sea of Japan", "JPN"),
        ("Lake Chad basin", "TCD"),
        ("pasture expansion in the Congo basin", "COG"),
        ("Niger Delta", "NER"),
        ("Inner Niger Delta", "NER"),
        ("the Korean Peninsula", "KOR"),
        ("New England states", "GBR"),
        # Not a place: the Brazilwood tree
        ("the pau-brasil tree", "BRA"),
    ])
    def test_a_country_inside_another_places_name_does_not_count(
        self, text, found,
    ):
        """*found* is what the text reads as without ``free_text``."""
        assert resolve_country_code(text) == found
        assert resolve_country_code(text, free_text=True) is None

    @pytest.mark.parametrize("text, code", [
        ("coal from New South Wales to Japan", "JPN"),
        ("Mexico and New Mexico", "MEX"),
        ("Mexico, Gulf of Mexico", "MEX"),
        ("Inner Mongolia, China", "CHN"),
        ("shrimp from the Gulf of Mexico to Japan", "JPN"),
    ])
    def test_other_countries_in_the_text_still_count(self, text, code):
        assert resolve_country_code(text, free_text=True) == code

    @pytest.mark.parametrize("text, code", [
        ("Western Australia", "AUS"),
        ("Mexico City", "MEX"),
        ("Hong Kong SAR, China", "HKG"),
        ("the Brazilian Amazon", "BRA"),
        ("Niger Delta, Nigeria", "NGA"),
    ])
    def test_a_places_own_country_counts(self, text, code):
        assert resolve_country_code(text, free_text=True) == code

    @pytest.mark.parametrize("text, code", [
        ("USA", "USA"),
        ("the USA", "USA"),
        ("MEX", "MEX"),
        ("US exports", "USA"),
        ("the US", "USA"),
        ("U.S. corn", "USA"),
    ])
    def test_codes_and_us_in_capitals_count(self, text, code):
        assert resolve_country_code(text, free_text=True) == code

    @pytest.mark.parametrize("text, found", [
        ("and", "AND"),
        ("Can", "CAN"),
        ("cod", "COD"),
        ("mex", "MEX"),
        ("us", "USA"),
        ("allowed us to map", "USA"),
        ("the Sea Around Us", "USA"),
    ])
    def test_codes_and_us_written_otherwise_do_not(self, text, found):
        """A word that is a code ("and" is Andorra's), or the pronoun."""
        assert resolve_country_code(text) == found
        assert resolve_country_code(text, free_text=True) is None

    def test_whole_names_still_resolve(self):
        assert resolve_country_code("Mexico", free_text=True) == "MEX"
        assert resolve_country_code("the Gambia", free_text=True) == "GMB"
        assert resolve_country_code(
            "Ethiopian coffee regions", free_text=True
        ) == "ETH"
        assert resolve_country_code(
            "New South Wales", substring=False, free_text=True
        ) is None

    def test_other_place_names_are_written_as_lookups_compare_them(self):
        """Each key holds a country's name or alias, normalized, with
        spaces for hyphens; each reads as no country in free text."""
        from metacouplingllm.knowledge.countries import (
            _OTHER_PLACE_NAMES,
            _place_key,
        )

        for key in _OTHER_PLACE_NAMES:
            assert _place_key(_normalize_country_text(key)) == key
            assert resolve_country_code(key) is not None, key
            assert resolve_country_code(key, free_text=True) is None, key

    def test_region_names_come_from_the_adm1_data(self):
        """Region names and aliases that hold another country's name, with
        the region's country; not those that are a country's name
        ("Georgia") or hold only their own ("Western Australia")."""
        from metacouplingllm.knowledge.countries import _get_other_places

        _, owners = _get_other_places()
        assert owners["new south wales"] == frozenset({"AUS"})
        assert owners["port of spain"] == frozenset({"TTO"})
        assert owners["thai binh"] == frozenset({"VNM"})
        assert owners["inner mongolia"] == frozenset({"CHN"})
        assert "georgia" not in owners
        assert "western australia" not in owners


class TestFindCountryCodes:
    """``find_country_codes`` reads each country name whole, wherever it
    falls in the text, and keeps the order in which the text names them."""

    @pytest.mark.parametrize(
        "text, codes",
        [
            ("Mexico and Canada avocado trade", ["MEX", "CAN"]),
            ("Ghana's imports of Nigerian natural gas", ["GHA", "NGA"]),
            ("Visitors from China, Japan and the United States",
             ["CHN", "JPN", "USA"]),
            ("Soybean exports from Brazil to China", ["BRA", "CHN"]),
        ],
    )
    def test_countries_in_order_of_mention(self, text, codes):
        assert find_country_codes(text) == codes

    @pytest.mark.parametrize(
        "text, codes",
        [
            ("Logging in Papua New Guinea", ["PNG"]),
            ("Refugees from South Sudan", ["SSD"]),
            ("Oil from Equatorial Guinea", ["GNQ"]),
            ("Cobalt from the Democratic Republic of the Congo", ["COD"]),
            ("Cashew exports from Guinea-Bissau", ["GNB"]),
            ("Sanctions on the Democratic People's Republic of Korea",
             ["PRK"]),
        ],
    )
    def test_name_holding_another_name_is_read_whole(self, text, codes):
        assert find_country_codes(text) == codes

    @pytest.mark.parametrize(
        "text, codes",
        [
            ("Cocoa exports from Côte d'Ivoire", ["CIV"]),
            ("Cocoa exports from Côte d’Ivoire", ["CIV"]),
            ("Hazelnut exports from Türkiye", ["TUR"]),
            ("Coffee from Timor-Leste", ["TLS"]),
            ("Natural gas from Trinidad and Tobago", ["TTO"]),
            ("Costa Rica's coffee", ["CRI"]),
            ("Coffee from Costa\nRica", ["CRI"]),
            ("U.S. corn exports to Mexico", ["USA", "MEX"]),
        ],
    )
    def test_name_with_punctuation_diacritics_or_line_break(self, text, codes):
        assert find_country_codes(text) == codes

    @pytest.mark.parametrize(
        "text, codes",
        [
            ("US corn exports to Mexico", ["USA", "MEX"]),
            ("Soybean flows (BRA, CHN)", ["BRA", "CHN"]),
        ],
    )
    def test_codes_written_as_codes_count(self, text, codes):
        assert find_country_codes(text) == codes

    @pytest.mark.parametrize(
        "text, codes",
        [
            # The old chunks read a lone "and" as Andorra's code.
            ("Forest cover in Kenya between 2000 and 2024", ["KEN"]),
            # The pronoun, which the old chunks read as the US.
            ("This study helps us understand Brazil's soybean exports",
             ["BRA"]),
            # An acronym within a run of words (Saint Lucia's code).
            ("An LCA of soybean production in Brazil", ["BRA"]),
            # A code word in capitals within a run of words.
            ("TRADE AND ENVIRONMENT IN KENYA", ["KEN"]),
            # Code words inside place names (Suriname, South Korea).
            ("Smoke over Lebel-sur-Quévillon", []),
            ("Ebola in Nzérékoré", []),
        ],
    )
    def test_words_that_only_look_like_codes_do_not_count(self, text, codes):
        assert find_country_codes(text) == codes

    @pytest.mark.parametrize(
        "text, codes",
        [
            # Continents, seas and basins named after a country
            ("Soybean exports from South America to China", ["CHN"]),
            ("Latin American and North-American markets", []),
            ("Shrimp from the Gulf of Mexico to Japan", ["JPN"]),
            ("pasture expansion in the Congo basin and Rwanda", ["RWA"]),
            ("Fishing in the Indian Ocean", []),
            ("Venezuela and Congo Basin", ["VEN"]),
            # Regions of another country, from the ADM1 database
            ("Coal from New South Wales to Japan", ["JPN"]),
            ("Salmon bound for British Columbia ports", []),
            ("Port-of-Spain and Port of Spain", []),
            ("Grasslands of Inner Mongolia", []),
        ],
    )
    def test_a_country_inside_another_places_name_does_not_count(
        self, text, codes,
    ):
        assert find_country_codes(text) == codes

    @pytest.mark.parametrize(
        "text, codes",
        [
            ("Inner Mongolia, China", ["CHN"]),
            ("Hong Kong SAR, China", ["HKG"]),
            ("Mexico City and Western Australia", ["MEX", "AUS"]),
            ("Floods in the Inner Niger Delta, Mali", ["MLI"]),
            ("Port of Spain shipping to Spain", ["ESP"]),
            ("Tea from Lao Cai to Laos", ["LAO"]),
            ("Gibraltar (U.K.)", ["GIB"]),
        ],
    )
    def test_a_places_own_country_counts(self, text, codes):
        assert find_country_codes(text) == codes

    @pytest.mark.parametrize(
        "text, codes",
        [
            ("Exports to the Virgin Islands (U.S.)", ["VIR"]),
            ("St. Martin (French part)", ["MAF"]),
            ("Congo, Dem. Rep. and Brazil", ["COD", "BRA"]),
            ("U.S.A. and Canada", ["USA", "CAN"]),
        ],
    )
    def test_punctuation_after_the_last_word_completes_a_name(
        self, text, codes,
    ):
        assert find_country_codes(text) == codes

    def test_brazilwood_is_not_brazil(self):
        """"pau-brasil", the tree, holds "brasil"."""
        assert find_country_codes("Logging of the pau-brasil tree") == []

    def test_each_name_reads_as_resolve_country_code_reads_it(self):
        """Every country name, alias and name of another place, scanned
        alone, gives what ``resolve_country_code(..., free_text=True)``
        gives."""
        from metacouplingllm.knowledge.countries import _get_text_scan_tables

        names, places, _ = _get_text_scan_tables()
        differ = []
        for name in [*names, *places]:
            code = resolve_country_code(name, free_text=True)
            if find_country_codes(name) != ([code] if code else []):
                differ.append(name)
        assert differ == []

    def test_accents_are_folded(self):
        assert find_country_codes("México y Guatemala") == ["MEX", "GTM"]
        assert find_country_codes("Exportaciones de Perú") == ["PER"]

    def test_lao(self):
        assert find_country_codes(
            "Lao PDR and the Lao government"
        ) == ["LAO"]

    def test_capitals_reads_only_capitalised_names(self):
        """For text whose writer capitalises names (an analysis)."""
        text = "turkey and chile peppers from Chile"
        assert find_country_codes(text) == ["TUR", "CHL"]
        assert find_country_codes(text, capitals=True) == ["CHL"]
        # The names of other places count in any case.
        assert find_country_codes(
            "the Congo basin", capitals=True
        ) == []


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


class TestExpandSupranational:
    """Test expand_supranational and supranational_display_name."""

    def test_european_union_returns_27_members(self):
        members = expand_supranational("European Union")
        assert members is not None
        assert len(members) == 27
        # Spot-check known current members.
        for code in ("FRA", "DEU", "POL", "HRV", "ESP"):
            assert code in members
        # Spot-check known non-members.
        assert "GBR" not in members  # post-Brexit
        assert "CHE" not in members  # never a member
        assert "NOR" not in members  # never a member
        # All entries must be valid ISO codes the rest of the
        # codebase recognises.
        for code in members:
            assert code in ISO_ALPHA3_NAMES

    def test_eu_abbreviations_resolve_identically(self):
        full = expand_supranational("European Union")
        assert full is not None
        for alias in ("EU", "eu", "e.u.", "The European Union"):
            assert expand_supranational(alias) == full

    def test_asean_returns_10_members(self):
        members = expand_supranational("ASEAN")
        assert members is not None
        assert len(members) == 10
        for code in (
            "BRN", "KHM", "IDN", "LAO", "MYS",
            "MMR", "PHL", "SGP", "THA", "VNM",
        ):
            assert code in members

    def test_asean_full_name_resolves(self):
        members = expand_supranational(
            "Association of Southeast Asian Nations",
        )
        assert members == expand_supranational("ASEAN")

    def test_nafta_and_usmca_are_aliases(self):
        nafta = expand_supranational("NAFTA")
        usmca = expand_supranational("USMCA")
        assert nafta == usmca
        assert sorted(nafta) == sorted(["USA", "MEX", "CAN"])

    def test_nafta_usmca_full_names_resolve(self):
        assert expand_supranational(
            "North American Free Trade Agreement",
        ) == expand_supranational("NAFTA")
        assert expand_supranational(
            "United States-Mexico-Canada Agreement",
        ) == expand_supranational("USMCA")

    def test_case_insensitive(self):
        assert expand_supranational("european union") == \
            expand_supranational("EUROPEAN UNION")

    def test_unknown_returns_none(self):
        assert expand_supranational("OPEC") is None
        assert expand_supranational("G7") is None
        assert expand_supranational("United Nations") is None
        assert expand_supranational("") is None
        assert expand_supranational("   ") is None

    def test_does_not_match_substring(self):
        # Strict whole-name match: "EU member states" should NOT match.
        assert expand_supranational("EU member states") is None
        assert expand_supranational("the asean region") is None

    def test_supranational_display_name_round_trip(self):
        for alias in ("European Union", "ASEAN", "USMCA", "NAFTA"):
            members = expand_supranational(alias)
            assert members is not None
            display = supranational_display_name(members)
            assert display is not None
            # Round-trip: the display name re-resolves to the same
            # member set.
            assert expand_supranational(display) == members

    def test_supranational_display_name_unknown_returns_none(self):
        assert supranational_display_name(["XYZ", "ABC"]) is None
        assert supranational_display_name([]) is None
