"""
ISO 3166-1 alpha-3 country code mapping for pericoupling lookups.

Provides country name → ISO code resolution so that natural-language country
names produced by LLMs can be matched against the pericoupling database.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# ISO alpha-3 code → canonical English name (covers all 224 codes in the
# pericoupling database plus common extras).
# ---------------------------------------------------------------------------

ISO_ALPHA3_NAMES: dict[str, str] = {
    "ABW": "Aruba",
    "AFG": "Afghanistan",
    "AGO": "Angola",
    "AIA": "Anguilla",
    "ALB": "Albania",
    "AND": "Andorra",
    "ANT": "Netherlands Antilles",
    "ARE": "United Arab Emirates",
    "ARG": "Argentina",
    "ARM": "Armenia",
    "ATF": "French Southern Territories",
    "ATG": "Antigua and Barbuda",
    "AUS": "Australia",
    "AUT": "Austria",
    "AZE": "Azerbaijan",
    "BDI": "Burundi",
    "BEL": "Belgium",
    "BEN": "Benin",
    "BFA": "Burkina Faso",
    "BGD": "Bangladesh",
    "BGR": "Bulgaria",
    "BHR": "Bahrain",
    "BHS": "Bahamas",
    "BIH": "Bosnia and Herzegovina",
    "BLR": "Belarus",
    "BLZ": "Belize",
    "BMU": "Bermuda",
    "BOL": "Bolivia",
    "BRA": "Brazil",
    "BRB": "Barbados",
    "BRN": "Brunei",
    "BTN": "Bhutan",
    "BWA": "Botswana",
    "CAF": "Central African Republic",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "CHL": "Chile",
    "CHN": "China",
    "CIV": "Ivory Coast",
    "CMR": "Cameroon",
    "COG": "Republic of the Congo",
    "COL": "Colombia",
    "COM": "Comoros",
    "CPV": "Cape Verde",
    "CRI": "Costa Rica",
    "CUB": "Cuba",
    "CYM": "Cayman Islands",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DJI": "Djibouti",
    "DMA": "Dominica",
    "DNK": "Denmark",
    "DOM": "Dominican Republic",
    "DZA": "Algeria",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ERI": "Eritrea",
    "ESH": "Western Sahara",
    "ESP": "Spain",
    "EST": "Estonia",
    "ETH": "Ethiopia",
    "FIN": "Finland",
    "FJI": "Fiji",
    "FLK": "Falkland Islands",
    "FRA": "France",
    "FSM": "Micronesia",
    "GAB": "Gabon",
    "GBR": "United Kingdom",
    "GEO": "Georgia",
    "GHA": "Ghana",
    "GIB": "Gibraltar",
    "GIN": "Guinea",
    "GMB": "Gambia",
    "GNB": "Guinea-Bissau",
    "GNQ": "Equatorial Guinea",
    "GRC": "Greece",
    "GRD": "Grenada",
    "GRL": "Greenland",
    "GTM": "Guatemala",
    "GUF": "French Guiana",
    "GUY": "Guyana",
    "HKG": "Hong Kong",
    "HND": "Honduras",
    "HRV": "Croatia",
    "HTI": "Haiti",
    "HUN": "Hungary",
    "IDN": "Indonesia",
    "IND": "India",
    "IRL": "Ireland",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "ISL": "Iceland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JAM": "Jamaica",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "KGZ": "Kyrgyzstan",
    "KHM": "Cambodia",
    "KIR": "Kiribati",
    "KNA": "Saint Kitts and Nevis",
    "KOR": "South Korea",
    "KWT": "Kuwait",
    "LAO": "Laos",
    "LBN": "Lebanon",
    "LBR": "Liberia",
    "LBY": "Libya",
    "LCA": "Saint Lucia",
    "LKA": "Sri Lanka",
    "LSO": "Lesotho",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "LVA": "Latvia",
    "MAC": "Macau",
    "MAR": "Morocco",
    "MDA": "Moldova",
    "MDG": "Madagascar",
    "MDV": "Maldives",
    "MEX": "Mexico",
    "MHL": "Marshall Islands",
    "MKD": "North Macedonia",
    "MLI": "Mali",
    "MLT": "Malta",
    "MMR": "Myanmar",
    "MNG": "Mongolia",
    "MNE": "Montenegro",
    "MOZ": "Mozambique",
    "MRT": "Mauritania",
    "MUS": "Mauritius",
    "MWI": "Malawi",
    "MYS": "Malaysia",
    "NAM": "Namibia",
    "NCL": "New Caledonia",
    "NER": "Niger",
    "NGA": "Nigeria",
    "NIC": "Nicaragua",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "NPL": "Nepal",
    "NRU": "Nauru",
    "NZL": "New Zealand",
    "OMN": "Oman",
    "PAK": "Pakistan",
    "PAL": "Palestine",
    "PAN": "Panama",
    "PER": "Peru",
    "PHL": "Philippines",
    "PLW": "Palau",
    "PNG": "Papua New Guinea",
    "POL": "Poland",
    "PRI": "Puerto Rico",
    "PRK": "North Korea",
    "PRT": "Portugal",
    "PRY": "Paraguay",
    "PSE": "Palestine",
    "QAT": "Qatar",
    "ROM": "Romania",
    "RUS": "Russia",
    "RWA": "Rwanda",
    "SAU": "Saudi Arabia",
    "SDN": "Sudan",
    "SEN": "Senegal",
    "SGP": "Singapore",
    "SLB": "Solomon Islands",
    "SLE": "Sierra Leone",
    "SLV": "El Salvador",
    "SMR": "San Marino",
    "SOM": "Somalia",
    "SSD": "South Sudan",
    "STP": "Sao Tome and Principe",
    "SUR": "Suriname",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SWE": "Sweden",
    "SWZ": "Eswatini",
    "SYC": "Seychelles",
    "SYR": "Syria",
    "TCD": "Chad",
    "TGO": "Togo",
    "THA": "Thailand",
    "TJK": "Tajikistan",
    "TKM": "Turkmenistan",
    "TMP": "East Timor",
    "TON": "Tonga",
    "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "TUV": "Tuvalu",
    "TWN": "Taiwan",
    "TZA": "Tanzania",
    "UGA": "Uganda",
    "UKR": "Ukraine",
    "URY": "Uruguay",
    "USA": "United States",
    "UZB": "Uzbekistan",
    "VCT": "Saint Vincent and the Grenadines",
    "VEN": "Venezuela",
    "VNM": "Vietnam",
    "VUT": "Vanuatu",
    "WSM": "Samoa",
    "YEM": "Yemen",
    "YUG": "Yugoslavia",
    "ZAF": "South Africa",
    "ZAR": "Democratic Republic of the Congo",
    "ZMB": "Zambia",
    "ZWE": "Zimbabwe",
}

# ---------------------------------------------------------------------------
# Aliases — common alternative names, demonyms, and abbreviations that map
# to the canonical ISO alpha-3 code.
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # Short forms / abbreviations
    "us": "USA",
    "usa": "USA",
    "u.s.": "USA",
    "u.s.a.": "USA",
    "uk": "GBR",
    "u.k.": "GBR",
    "uae": "ARE",
    "u.a.e.": "ARE",
    "drc": "ZAR",
    "dr congo": "ZAR",
    "rok": "KOR",
    "dprk": "PRK",
    "prc": "CHN",
    # Common alternative names
    "america": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "britain": "GBR",
    "great britain": "GBR",
    "united kingdom": "GBR",
    "england": "GBR",
    "scotland": "GBR",
    "wales": "GBR",
    "holland": "NLD",
    "the netherlands": "NLD",
    "ivory coast": "CIV",
    "cote d'ivoire": "CIV",
    "côte d'ivoire": "CIV",
    "czech republic": "CZE",
    "czechia": "CZE",
    "south korea": "KOR",
    "south sudan": "SSD",
    "greenland": "GRL",
    "montenegro": "MNE",
    "puerto rico": "PRI",
    "new caledonia": "NCL",
    "falkland islands": "FLK",
    "falklands": "FLK",
    "republic of korea": "KOR",
    "north korea": "PRK",
    "democratic people's republic of korea": "PRK",
    "burma": "MMR",
    "myanmar": "MMR",
    "persia": "IRN",
    "swaziland": "SWZ",
    "eswatini": "SWZ",
    "east timor": "TMP",
    "timor-leste": "TMP",
    "zaire": "ZAR",
    "congo-kinshasa": "ZAR",
    "democratic republic of the congo": "ZAR",
    "congo-brazzaville": "COG",
    "republic of the congo": "COG",
    "congo": "COG",
    "palestine": "PSE",
    "palestinian territories": "PSE",
    "west bank": "PSE",
    "gaza": "PSE",
    "hong kong": "HKG",
    "macau": "MAC",
    "macao": "MAC",
    "taiwan": "TWN",
    "brunei darussalam": "BRN",
    "cabo verde": "CPV",
    "cape verde": "CPV",
    "laos": "LAO",
    "lao pdr": "LAO",
    "russia": "RUS",
    "russian federation": "RUS",
    "soviet union": "RUS",
    "ussr": "RUS",
    "iran": "IRN",
    "syria": "SYR",
    "libya": "LBY",
    "venezuela": "VEN",
    "bolivia": "BOL",
    "tanzania": "TZA",
    "vietnam": "VNM",
    "viet nam": "VNM",
    "north macedonia": "MKD",
    "macedonia": "MKD",
    "bosnia": "BIH",
    "bosnia and herzegovina": "BIH",
    "serbia": "YUG",
    "yugoslavia": "YUG",
    "serbia and montenegro": "YUG",
    "western sahara": "ESH",
    "french guiana": "GUF",
    "papua new guinea": "PNG",
    "new zealand": "NZL",
    "south africa": "ZAF",
    "saudi arabia": "SAU",
    "sri lanka": "LKA",
    "trinidad and tobago": "TTO",
    "el salvador": "SLV",
    "costa rica": "CRI",
    "puerto rico": "USA",
    "central african republic": "CAF",
    "equatorial guinea": "GNQ",
    "guinea-bissau": "GNB",
    "sierra leone": "SLE",
    "burkina faso": "BFA",
    "san marino": "SMR",
    "turkiye": "TUR",
    "türkiye": "TUR",
    "turkey": "TUR",
    # Common demonyms (adjective forms used by LLMs)
    "american": "USA",
    "mexican": "MEX",
    "canadian": "CAN",
    "brazilian": "BRA",
    "chinese": "CHN",
    "indian": "IND",
    "german": "DEU",
    "french": "FRA",
    "british": "GBR",
    "japanese": "JPN",
    "korean": "KOR",
    # "Korea" alone is ambiguous (KOR vs PRK) but in academic
    # literature ~99% of bare "Korea" mentions refer to South Korea
    # (KOR) — North Korea is almost always written out in full as
    # "North Korea" or "DPRK". Default to KOR.
    "korea": "KOR",
    "australian": "AUS",
    "russian": "RUS",
    "italian": "ITA",
    "spanish": "ESP",
    "colombian": "COL",
    "chilean": "CHL",
    "peruvian": "PER",
    "argentine": "ARG",
    "argentinian": "ARG",
    "venezuelan": "VEN",
    "ecuadorian": "ECU",
    "bolivian": "BOL",
    "paraguayan": "PRY",
    "uruguayan": "URY",
    "guatemalan": "GTM",
    "honduran": "HND",
    "salvadoran": "SLV",
    "nicaraguan": "NIC",
    "costa rican": "CRI",
    "panamanian": "PAN",
    "cuban": "CUB",
    "dominican": "DOM",
    "haitian": "HTI",
    "jamaican": "JAM",
    "ethiopian": "ETH",
    "kenyan": "KEN",
    "nigerian": "NGA",
    "ghanaian": "GHA",
    "south african": "ZAF",
    "egyptian": "EGY",
    "moroccan": "MAR",
    "tunisian": "TUN",
    "algerian": "DZA",
    "sudanese": "SDN",
    "tanzanian": "TZA",
    "ugandan": "UGA",
    "thai": "THA",
    "vietnamese": "VNM",
    "indonesian": "IDN",
    "malaysian": "MYS",
    "philippine": "PHL",
    "filipino": "PHL",
    "cambodian": "KHM",
    "myanmar": "MMR",
    "burmese": "MMR",
    "pakistani": "PAK",
    "afghan": "AFG",
    "iranian": "IRN",
    "iraqi": "IRQ",
    "saudi": "SAU",
    "turkish": "TUR",
    "israeli": "ISR",
    "palestinian": "PSE",
    "jordanian": "JOR",
    "lebanese": "LBN",
    "syrian": "SYR",
    "polish": "POL",
    "ukrainian": "UKR",
    "romanian": "ROM",
    "hungarian": "HUN",
    "czech": "CZE",
    "swedish": "SWE",
    "norwegian": "NOR",
    "finnish": "FIN",
    "danish": "DNK",
    "dutch": "NLD",
    "belgian": "BEL",
    "swiss": "CHE",
    "austrian": "AUT",
    "portuguese": "PRT",
    "greek": "GRC",
    "irish": "IRL",
    "scottish": "GBR",
    "welsh": "GBR",
    "new zealand": "NZL",
}


# ---------------------------------------------------------------------------
# Internal reverse index (lazily built)
# ---------------------------------------------------------------------------

_name_to_code: dict[str, str] | None = None


def _build_reverse_index() -> dict[str, str]:
    """Build a mapping from lowercase canonical name → code."""
    index: dict[str, str] = {}
    for code, name in ISO_ALPHA3_NAMES.items():
        index[name.lower()] = code
    return index


def _get_reverse_index() -> dict[str, str]:
    global _name_to_code
    if _name_to_code is None:
        _name_to_code = _build_reverse_index()
    return _name_to_code


# Cached sorted list of (lowercase_name, code) for substring matching,
# sorted by length descending so longest matches win.
_sorted_all_names: list[tuple[str, str]] | None = None


def _get_sorted_all_names() -> list[tuple[str, str]]:
    """Return cached sorted (name, code) pairs for substring matching."""
    global _sorted_all_names
    if _sorted_all_names is None:
        rev = _get_reverse_index()
        pairs: list[tuple[str, str]] = []
        for alias, code in _ALIASES.items():
            pairs.append((alias, code))
        for name_lower, code in rev.items():
            pairs.append((name_lower, code))
        pairs.sort(key=lambda t: len(t[0]), reverse=True)
        _sorted_all_names = pairs
    return _sorted_all_names


def _contains_standalone_country_term(text: str, term: str) -> bool:
    """Return True when *term* appears as a standalone token/phrase.

    This avoids false positives from naive substring matching such as:
    - ``"Indiana"`` matching ``"ind"`` (India)
    - ``"used"`` matching ``"us"`` (United States)
    """
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_country_code(name_or_code: str) -> str | None:
    """Resolve a country name or code to an ISO 3166-1 alpha-3 code.

    Tries the following in order:

    1. Exact ISO code match (case-insensitive, e.g. ``"usa"`` → ``"USA"``)
    2. Alias match (e.g. ``"America"`` → ``"USA"``)
    3. Canonical name match (e.g. ``"United States"`` → ``"USA"``)
    4. Substring match — if *name_or_code* contains a known country name
       (e.g. ``"Ethiopian coffee regions"`` → ``"ETH"``)

    Parameters
    ----------
    name_or_code:
        Country name, ISO code, or text containing a country name.

    Returns
    -------
    The ISO alpha-3 code, or ``None`` if no match is found.
    """
    if not name_or_code or not name_or_code.strip():
        return None

    text = name_or_code.strip()
    text_lower = text.lower()

    # 1. Exact code match
    text_upper = text.upper()
    if text_upper in ISO_ALPHA3_NAMES:
        return text_upper

    # 2. Alias match
    if text_lower in _ALIASES:
        return _ALIASES[text_lower]

    # 3. Canonical name match
    rev = _get_reverse_index()
    if text_lower in rev:
        return rev[text_lower]

    # 4. Substring match — check if any alias or canonical name appears
    #    in the input text. Try longest matches first to avoid partial hits.
    #    (e.g. "South Korea" before "Korea")
    for name_lower, code in _get_sorted_all_names():
        if _contains_standalone_country_term(text_lower, name_lower):
            return code

    return None


def get_country_name(code: str) -> str:
    """Return the canonical English name for an ISO alpha-3 code.

    Parameters
    ----------
    code:
        ISO 3166-1 alpha-3 code (case-insensitive).

    Returns
    -------
    The country name, or the code itself if not found.
    """
    return ISO_ALPHA3_NAMES.get(code.upper(), code)
