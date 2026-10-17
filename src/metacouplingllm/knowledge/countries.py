"""
ISO 3166-1 alpha-3 country code mapping for pericoupling lookups.

Provides country name → ISO code resolution so that natural-language country
names produced by LLMs can be matched against the pericoupling database.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

# ---------------------------------------------------------------------------
# ISO alpha-3 code → canonical English name (covers all 244 ISO codes in the
# pericoupling databases — current ISO 3166-1 alpha-3, modernized in PR #50
# from legacy ZAR/ROM/YUG/TMP to COD/ROU/SRB/TLS — plus common extras).
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
    "ROU": "Romania",
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
    "TLS": "Timor-Leste",
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
    "ZAF": "South Africa",
    "COD": "Democratic Republic of the Congo",
    "SRB": "Serbia",
    "ZMB": "Zambia",
    "ZWE": "Zimbabwe",
    # --- Additional ISO 3166-1 alpha-3 codes present in the WB 2026
    # boundaries (PR #50). Adjacency-bearing first, then dependent
    # territories / islands for completeness so every code in the
    # bundled CSVs resolves. ---
    "XKX": "Kosovo",
    "MCO": "Monaco",
    "LIE": "Liechtenstein",
    "VAT": "Vatican City",
    "MAF": "Saint Martin",
    "SXM": "Sint Maarten",
    "ALA": "Aland Islands",
    "ASM": "American Samoa",
    "BES": "Bonaire, Sint Eustatius and Saba",
    "BLM": "Saint Barthelemy",
    "BVT": "Bouvet Island",
    "CCK": "Cocos (Keeling) Islands",
    "COK": "Cook Islands",
    "CUW": "Curacao",
    "CXR": "Christmas Island",
    "FRO": "Faroe Islands",
    "GGY": "Guernsey",
    "GLP": "Guadeloupe",
    "GUM": "Guam",
    "HMD": "Heard Island and McDonald Islands",
    "IMN": "Isle of Man",
    "JEY": "Jersey",
    "MNP": "Northern Mariana Islands",
    "MSR": "Montserrat",
    "MTQ": "Martinique",
    "MYT": "Mayotte",
    "NFK": "Norfolk Island",
    "NIU": "Niue",
    "PCN": "Pitcairn Islands",
    "PYF": "French Polynesia",
    "REU": "Reunion",
    "SHN": "Saint Helena",
    "SJM": "Svalbard and Jan Mayen",
    "SPM": "Saint Pierre and Miquelon",
    "TCA": "Turks and Caicos Islands",
    "TKL": "Tokelau",
    "UMI": "United States Minor Outlying Islands",
    "VGB": "British Virgin Islands",
    "VIR": "United States Virgin Islands",
    "WLF": "Wallis and Futuna",
}

# ---------------------------------------------------------------------------
# Aliases — common alternative names, demonyms, and abbreviations that map
# to the canonical ISO alpha-3 code.  Keys are written as lookups compare
# them (``_normalize_country_text``): lowercase, without accents, with
# straight apostrophes and single spaces, so "Côte d'Ivoire" and "España"
# match the keys "cote d'ivoire" and "espana".
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
    "drc": "COD",
    "dr congo": "COD",
    "rok": "KOR",
    "dprk": "PRK",
    "dpr korea": "PRK",  # "korea" alone gives South Korea
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
    "east timor": "TLS",
    "timor-leste": "TLS",
    "zaire": "COD",
    "congo-kinshasa": "COD",
    "democratic republic of the congo": "COD",
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
    "lao": "LAO",  # the adjective, as in "the Lao government"
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
    "serbia": "SRB",
    "yugoslavia": "SRB",
    "serbia and montenegro": "SRB",
    "kosovo": "XKX",
    "monaco": "MCO",
    "liechtenstein": "LIE",
    "vatican": "VAT",
    "vatican city": "VAT",
    "holy see": "VAT",
    "saint martin": "MAF",
    "sint maarten": "SXM",
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
    "central african republic": "CAF",
    "equatorial guinea": "GNQ",
    "guinea-bissau": "GNB",
    "guinea bissau": "GNB",
    "sierra leone": "SLE",
    "burkina faso": "BFA",
    "san marino": "SMR",
    "turkiye": "TUR",
    "turkey": "TUR",
    # Names in the country's own language
    "espana": "ESP",  # España
    "brasil": "BRA",
    # Names as the World Bank writes them, in its country list
    # (api.worldbank.org/v2/country) or in the country field of the
    # boundary data behind the ADM1 database, where the names above miss
    # them or a shorter name inside them gives another country ("Congo,
    # Dem. Rep." reads "congo", the Republic of the Congo)
    "congo, dem. rep.": "COD",
    "democratic republic of congo": "COD",
    "korea, dem. people's rep.": "PRK",
    "d. p. r. of korea": "PRK",
    "lao people's democratic republic": "LAO",
    "kyrgyz republic": "KGZ",
    "slovak republic": "SVK",
    "st. kitts and nevis": "KNA",
    "st. lucia": "LCA",
    "st. vincent and the grenadines": "VCT",
    "st. martin (french part)": "MAF",
    "virgin islands (u.s.)": "VIR",
    "naoero": "NRU",  # the World Bank's name for Nauru
    # ISO 3166-1 names as Debian's iso-codes data (the names pycountry
    # gives) and Wikipedia's table of ISO 3166-1 write them, where a
    # shorter name inside them gives another country: the short names
    # written inverted ("Virgin Islands, British" reads "british", the
    # United Kingdom) and the official name of the US Virgin Islands
    "korea, democratic people's republic of": "PRK",
    "congo, democratic republic of the": "COD",
    "congo, the democratic republic of the": "COD",
    "virgin islands, british": "VGB",
    "virgin islands (british)": "VGB",
    "virgin islands, u.s.": "VIR",
    "virgin islands of the united states": "VIR",
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
    "north korean": "PRK",
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
    "south sudanese": "SSD",
    "tanzanian": "TZA",
    "ugandan": "UGA",
    "thai": "THA",
    "vietnamese": "VNM",
    "indonesian": "IDN",
    "malaysian": "MYS",
    "philippine": "PHL",
    "filipino": "PHL",
    "cambodian": "KHM",
    "burmese": "MMR",
    "pakistani": "PAK",
    "bangladeshi": "BGD",
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
    "romanian": "ROU",
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
}


# ---------------------------------------------------------------------------
# Other places named after a country
# ---------------------------------------------------------------------------
#
# Names of places that hold a country's name or alias but are not that
# country: "South America" holds "america" (the United States), "the Gulf of
# Mexico" the name of Mexico.  In free text
# (``resolve_country_code(..., free_text=True)``) a country's name found only
# inside one of these names does not count.  Each name maps to the countries
# the place lies in, whose own names still count inside it.  Keys are written
# as lookups compare them (``_normalize_country_text``), with a space where
# the text may have a space or a hyphen ("Indo-China").  The region names of
# the ADM1 database that hold another country's name ("New South Wales",
# "Port of Spain", "Inner Mongolia") are added from that database
# (:func:`_get_other_places`).
# ---------------------------------------------------------------------------

_OTHER_PLACE_NAMES: dict[str, tuple[str, ...]] = {
    # Continents and world regions
    "north america": (),
    "south america": (),
    "central america": (),
    "latin america": (),
    "middle america": (),
    "northern america": (),
    "north american": (),
    "south american": (),
    "central american": (),
    "latin american": (),
    "america latina": (),
    "america del sur": (),
    "america central": (),
    "pan american": (),
    "inter american": (),
    "indo china": (),
    "british isles": (),
    # Seas, gulfs and straits
    "indian ocean": (),
    "indian oceans": (),
    "gulf of mexico": (),
    "gulf of guinea": (),
    "gulf of thailand": (),
    "gulf of oman": (),
    "south china sea": (),
    "east china sea": (),
    "sea of japan": (),
    "japan sea": (),
    "philippine sea": (),
    "irish sea": (),
    "mozambique channel": (),
    "korea strait": (),
    "taiwan strait": (),
    "singapore strait": (),
    "strait of gibraltar": (),
    # Lakes, rivers, basins and deltas
    "lake chad": (),
    "lake malawi": (),
    "congo basin": (),
    "congo river": (),
    "niger river": (),
    "middle niger": (),
    "niger delta": ("NGA",),
    "inner niger delta": ("MLI",),
    "paraguay river": (),
    # Peninsulas, plateaus, and regions that are not first-level units
    "korean peninsula": (),
    "iranian plateau": (),
    "mongolia plateau": (),
    "new england": ("USA",),
    "south georgia": (),
    # Not a place: the Brazilwood tree, after which Brazil is named
    "pau brasil": (),
}


# ---------------------------------------------------------------------------
# Supranational groupings (not in ISO 3166-1)
# ---------------------------------------------------------------------------
#
# These names are NOT sovereign-state ISO codes.  Treating them as such
# would silently drop flows that mention them.  Instead, the resolver
# expands a supranational mention to its member ISO codes when no
# individual member is already detected in the same endpoint context;
# the map renderer then draws ONE arrow ending at the geometric union
# centroid of the member polygons and highlights the member region as
# a single labelled entity.
#
# Membership lists are intentionally exhaustive within each grouping
# but limited to three groupings (EU, ASEAN, NAFTA/USMCA) — the most
# common in metacoupling research.  Add others (Mercosur, BRICS, AU,
# G7/G20) only when actual analyses request them.

# EU 27 member states as of 2026-05 (post-Brexit; Croatia joined 2013).
_EU_27 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
    "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK",
    "SVN", "ESP", "SWE",
]

# ASEAN 10 member states.
_ASEAN_10 = [
    "BRN", "KHM", "IDN", "LAO", "MYS",
    "MMR", "PHL", "SGP", "THA", "VNM",
]

# NAFTA / USMCA 3 (treaty order).
_USMCA_3 = ["USA", "MEX", "CAN"]

# Map of normalised supranational name -> list of member ISO codes.
# String values point at canonical keys (alias-of-alias indirection).
_SUPRANATIONAL_ALIASES: dict[str, list[str] | str] = {
    "european union":                         _EU_27,
    "eu":                                     "european union",
    "e.u.":                                   "european union",
    "the european union":                     "european union",

    "asean":                                  _ASEAN_10,
    "association of southeast asian nations": "asean",

    "nafta":                                  _USMCA_3,
    "north american free trade agreement":    "nafta",
    "usmca":                                  _USMCA_3,
    "united states-mexico-canada agreement":  "usmca",
}

# Canonical display names looked up by sorted-tuple of member codes.
# Used by the resolver to attach a stable label to each supranational
# flow target that the map renderer can show.
_SUPRANATIONAL_DISPLAY_NAMES: dict[tuple[str, ...], str] = {
    tuple(sorted(_EU_27)):    "European Union",
    tuple(sorted(_ASEAN_10)): "ASEAN",
    tuple(sorted(_USMCA_3)):  "USMCA",
}


def expand_supranational(name: str) -> list[str] | None:
    """Return member ISO 3166-1 alpha-3 codes for a supranational alias.

    Returns ``None`` when ``name`` does not match any known grouping.

    Examples
    --------
    >>> "FRA" in expand_supranational("European Union")
    True
    >>> expand_supranational("EU") == expand_supranational("European Union")
    True
    >>> expand_supranational("OPEC") is None
    True
    """
    if not name or not name.strip():
        return None
    key = name.strip().lower()
    value = _SUPRANATIONAL_ALIASES.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        # Alias-of-alias: resolve once.
        value = _SUPRANATIONAL_ALIASES.get(value)
    if isinstance(value, list):
        return list(value)
    return None


def supranational_display_name(member_codes: list[str]) -> str | None:
    """Return canonical display name for a set of supranational member codes.

    Looks up by sorted-tuple so order in ``member_codes`` doesn't matter.
    Returns ``None`` if the set isn't a recognised grouping.
    """
    return _SUPRANATIONAL_DISPLAY_NAMES.get(tuple(sorted(member_codes)))


# ---------------------------------------------------------------------------
# Internal reverse index (lazily built)
# ---------------------------------------------------------------------------


def _normalize_country_text(text: str) -> str:
    """Return *text* as country lookups compare it: lowercase, with accents
    folded (``"México"`` → ``"mexico"``), curly apostrophes made straight
    and each run of whitespace made one space."""
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    folded = folded.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(folded.lower().split())


# Country names read as regions elsewhere ("georgia" → ("USA011", "USA")),
# by normalized name; ``None`` outside :func:`read_names_as_regions`.
_names_read_as_regions: ContextVar[Mapping[str, tuple[str, str]] | None] = (
    ContextVar("names_read_as_regions", default=None)
)


@contextmanager
def read_names_as_regions(
    readings: Mapping[str, tuple[str, str]],
) -> Iterator[None]:
    """Within the block, read each country name in *readings* as the region
    it maps to.

    *readings* maps a country's name, normalized (lowercase, accents folded),
    to the ADM1 code and country of a region elsewhere that has the name, as
    :func:`~metacouplingllm.knowledge.places.place_readings` gives them for a
    text that reads the name so: ``{"georgia": ("USA011", "USA")}``.  Within
    the block :func:`resolve_country_code` and :func:`find_country_codes`
    give the region's country for the name ("Georgia" → ``"USA"``), and
    :func:`~metacouplingllm.knowledge.adm1_pericoupling.resolve_adm1_code`
    gives the region (``"USA011"``).  ``MetacouplingAssistant`` reads each
    analysis inside such a block, with the readings of its research
    description and answer.
    """
    token = _names_read_as_regions.set(dict(readings))
    try:
        yield
    finally:
        _names_read_as_regions.reset(token)


def region_read_for(name: str) -> str | None:
    """Return the ADM1 code of the region the current
    :func:`read_names_as_regions` block reads *name* as, or ``None``."""
    reading = (_names_read_as_regions.get() or {}).get(
        _normalize_country_text(name)
    )
    return reading[0] if reading else None


def reading_names_as_regions() -> bool:
    """Return True inside a :func:`read_names_as_regions` block, even one
    that reads no name as a region."""
    return _names_read_as_regions.get() is not None


_name_to_code: dict[str, str] | None = None


def _build_reverse_index() -> dict[str, str]:
    """Build a mapping from normalized canonical name → code."""
    index: dict[str, str] = {}
    for code, name in ISO_ALPHA3_NAMES.items():
        index[_normalize_country_text(name)] = code
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


# Characters a standalone term may not touch on either side.
_TERM_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def _contains_standalone_country_term(text: str, term: str) -> bool:
    """Return True when *term* appears as a standalone token/phrase.

    This avoids false positives from naive substring matching such as:
    - ``"Indiana"`` matching ``"ind"`` (India)
    - ``"used"`` matching ``"us"`` (United States)

    Standalone means no ASCII lowercase letter or digit directly before or
    after the occurrence: the meaning of
    ``re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)``,
    checked with ``str.find`` so that no pattern is built per term (a
    lookup that reaches the substring step tests every country name and
    alias in turn).
    """
    end = len(term)
    i = text.find(term)
    while i != -1:
        j = i + end
        if (i == 0 or text[i - 1] not in _TERM_CHARS) and (
            j == len(text) or text[j] not in _TERM_CHARS
        ):
            return True
        i = text.find(term, i + 1)
    return False


def _standalone_term_spans(text: str, term: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` of each standalone occurrence of *term*
    in *text*, standalone as in :func:`_contains_standalone_country_term`.
    """
    spans: list[tuple[int, int]] = []
    end = len(term)
    i = text.find(term)
    while i != -1:
        j = i + end
        if (i == 0 or text[i - 1] not in _TERM_CHARS) and (
            j == len(text) or text[j] not in _TERM_CHARS
        ):
            spans.append((i, j))
        i = text.find(term, i + 1)
    return spans


def _place_key(name: str) -> str:
    """Return *name* with each run of spaces and hyphens made one space."""
    return " ".join(name.replace("-", " ").split())


# Pattern matching the names of other places named after a country, and the
# countries each name's place lies in, by ``_place_key`` (lazily built).
_other_places: tuple[re.Pattern[str], dict[str, frozenset[str]]] | None = None


def _get_other_places() -> tuple[re.Pattern[str], dict[str, frozenset[str]]]:
    """Return a pattern matching the names of other places named after a
    country in normalized text, and the countries each one lies in.

    The names are ``_OTHER_PLACE_NAMES`` and the region names and aliases of
    the ADM1 database that hold the name or alias of a country other than the
    region's own: "New South Wales" (Australia) holds "wales", "Port of Spain"
    (Trinidad and Tobago) "spain".  A region name that is itself a country's
    name, such as "Georgia", is left out: whether it means the country is the
    question of the ADM1 country guard.
    """
    global _other_places
    if _other_places is not None:
        return _other_places
    owners: dict[str, frozenset[str]] = {
        _place_key(name): frozenset(codes)
        for name, codes in _OTHER_PLACE_NAMES.items()
    }
    try:
        from metacouplingllm.knowledge.adm1_pericoupling import _region_names

        region_names = _region_names()
    except Exception:  # the ADM1 data only adds names
        region_names = {}
    # Terms by their first word: a standalone term's first word is a whole
    # word of the text it stands in.
    by_first_word: dict[str, list[tuple[str, str]]] = {}
    for term, code in _get_sorted_all_names():
        first = re.match(r"[a-z0-9]*", term).group(0)
        by_first_word.setdefault(first, []).append((term, code))
    for name, isos in region_names.items():
        text = _normalize_country_text(name)
        if resolve_country_code(text, substring=False):
            continue
        if any(
            code not in isos and _contains_standalone_country_term(text, term)
            for word in set(re.findall(r"[a-z0-9]+", text))
            for term, code in by_first_word.get(word, ())
        ):
            key = _place_key(text)
            owners[key] = owners.get(key, frozenset()) | isos
    patterns = [
        r"(?<![a-z0-9])"
        + r"[ -]+".join(re.escape(word) for word in key.split(" "))
        + r"(?![a-z0-9])"
        for key in sorted(owners, key=len, reverse=True)
    ]
    _other_places = (re.compile("|".join(patterns)), owners)
    return _other_places


def _other_place_spans(text: str) -> list[tuple[int, int, frozenset[str]]]:
    """Return ``(start, end, countries)`` for each name of another place
    named after a country in the normalized *text*."""
    pattern, owners = _get_other_places()
    return [
        (m.start(), m.end(), owners[_place_key(m.group(0))])
        for m in pattern.finditer(text)
    ]


def _inside_other_place(
    places: list[tuple[int, int, frozenset[str]]],
    span: tuple[int, int],
    code: str,
) -> bool:
    """Return True when *span* lies inside one of *places* that is not in
    the country *code*."""
    i, j = span
    return any(a <= i and j <= b and code not in own for a, b, own in places)


def _written_in_capitals(text: str) -> bool:
    """Return True when *text*, less a leading "the", is all capitals, as
    free text writes a country code ("MEX", "the US")."""
    stripped = text.strip()
    if stripped[:4].lower() == "the ":
        stripped = stripped[4:].strip()
    return stripped.isupper()


# "US" in capitals: free text reads the alias "us" only so written.
_US_IN_CAPITALS_RE = re.compile(r"(?<![A-Za-z0-9])US(?![A-Za-z0-9])")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_country_code(
    name_or_code: str, *, substring: bool = True, free_text: bool = False,
) -> str | None:
    """Resolve a country name or code to an ISO 3166-1 alpha-3 code.

    Case, accents, curly apostrophes and repeated spaces are ignored
    (``"México"`` and ``"MEXICO"`` match ``"Mexico"``), and steps 1-3 also
    try the text without a leading "the" (``"the Gambia"``).  Tries the
    following in order:

    1. Exact ISO code match (case-insensitive, e.g. ``"usa"`` → ``"USA"``)
    2. Alias match (e.g. ``"America"`` → ``"USA"``), covering demonyms
       (``"Bangladeshi"``), names in the country's own language
       (``"Brasil"``), the World Bank's spellings (``"Congo, Dem.
       Rep."`` → ``"COD"``) and ISO 3166's inverted short names
       (``"Virgin Islands, British"`` → ``"VGB"``)
    3. Canonical name match (e.g. ``"United States"`` → ``"USA"``)
    4. Substring match — if *name_or_code* contains a known country name
       or alias as a standalone term (e.g. ``"Ethiopian coffee regions"``
       → ``"ETH"``).  The longest term found wins, which also reads a
       place named after a country as that country (``"New South
       Wales"`` → ``"GBR"``) unless *free_text* is set.

    Parameters
    ----------
    name_or_code:
        Country name, ISO code, or text containing a country name.
    substring:
        ``False`` skips step 4, so only a string that *is* a country's
        code, name or alias resolves: ``"Mexico"`` → ``"MEX"``, but
        ``"New Mexico"`` → ``None``.  ``resolve_adm1_code`` asks this way
        whether a region name is a country's.
    free_text:
        ``True`` when *name_or_code* is a passage of text (part of an
        analysis, a research query, a list of places) rather than a name.
        Step 4 then skips a country's name or alias found only inside the
        name of another place: a region of another country (``"New South
        Wales"``, ``"Port of Spain"``), a continent (``"South America"``),
        or a sea, lake or basin named after a country (``"the Gulf of
        Mexico"``, ``"Lake Chad"``); ``"coal from New South Wales to
        Japan"`` → ``"JPN"``.  A code, and the alias ``"us"``, count only
        when written in capitals: ``"and"`` is not Andorra, ``"Can"`` not
        Canada, ``"allowed us to"`` not the United States.

    Returns
    -------
    The ISO alpha-3 code, or ``None`` if no match is found.
    """
    if not name_or_code or not name_or_code.strip():
        return None

    text = _normalize_country_text(name_or_code)
    rev = _get_reverse_index()
    # Free text names a country by its code, or by "us", only in capitals.
    capitals = not free_text or _written_in_capitals(name_or_code)
    # A name read as a region elsewhere names the region's country.
    readings = _names_read_as_regions.get() or {}

    for key in dict.fromkeys((text, text.removeprefix("the "))):
        if key in readings:
            return readings[key][1]

        # 1. Exact code match
        key_upper = key.upper()
        if key_upper in ISO_ALPHA3_NAMES and capitals:
            return key_upper

        # 2. Alias match
        if key in _ALIASES and (capitals or key != "us"):
            return _ALIASES[key]

        # 3. Canonical name match
        if key in rev:
            return rev[key]

    if not substring:
        return None

    # 4. Substring match — check if any alias or canonical name appears
    #    in the input text. Try longest matches first to avoid partial hits.
    #    (e.g. "South Korea" before "Korea")
    places: list[tuple[int, int, frozenset[str]]] | None = None
    for name_lower, code in _get_sorted_all_names():
        if name_lower in readings:
            code = readings[name_lower][1]
        if not free_text:
            if _contains_standalone_country_term(text, name_lower):
                return code
            continue
        spans = _standalone_term_spans(text, name_lower)
        if not spans or (
            name_lower == "us" and not _US_IN_CAPITALS_RE.search(name_or_code)
        ):
            continue
        if places is None:
            places = _other_place_spans(text)
        if not all(_inside_other_place(places, span, code) for span in spans):
            return code

    return None


def find_country_codes(text: str, *, capitals: bool = False) -> list[str]:
    """Return the codes of the countries *text* names, in order of first mention.

    At each word the scan tries the longest run of words that is exactly a
    country's name or alias, or the name of another place named after a
    country, and after a match goes on from the word that follows it.  A
    name is read whole wherever it falls: "Coffee production in Costa Rica"
    gives CRI, "Mexico and Canada" gives MEX and then CAN, and "Papua New
    Guinea" gives PNG without a GIN for "Guinea".  The name of another place
    (the places of ``resolve_country_code(..., free_text=True)``) names no
    country but its own: "soy from South America to China" gives CHN alone,
    "Inner Mongolia, China" CHN, "Hong Kong SAR, China" HKG.  Words are
    compared as ``resolve_country_code`` compares them, so names written with
    accents, a hyphen, comma or apostrophe match ("México", "Guinea-Bissau",
    "Côte d’Ivoire"); a possessive "'s" is dropped ("Costa Rica's"), and a
    period after the last word is kept when that makes a name ("U.S.").

    Two kinds of word count only in capitals:

    - "US"; "us" is the pronoun ("helps us understand").
    - A three-letter ISO code ("BRA"), and only when it stands apart from
      other words, with punctuation or the text's edge on both sides, as in
      "(BRA)" or "BRA–CHN".  Within a run of words a code is more often an
      acronym (LCA, life cycle assessment, is Saint Lucia's code), and in
      small letters a word ("and", "can", "per").

    Parameters
    ----------
    text:
        A passage of text: a research description, an analysis.
    capitals:
        ``True`` reads a country's name only where it starts with a
        capital letter, for text whose writer capitalises names (an
        analysis): "turkey" and "chile peppers" then name no country.  The
        names of other places are recognised in any case ("the Congo
        basin").

    Returns
    -------
    The ISO alpha-3 codes, each once, in the order the text first names
    them.
    """
    found: list[str] = []
    for code, _, _ in _country_mentions(text, capitals=capitals):
        if code not in found:
            found.append(code)
    return found


# A word: a run of letters, Latin letters with diacritics included ("Côte",
# "Türkiye"), with any apostrophe that stands between two letters
# ("d'Ivoire", "People's", "Brazil’s").
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏ]+(?:['’][A-Za-zÀ-ÖØ-öø-ɏ]+)*")

# The country names and aliases by normalized text, the names of other places
# named after a country with the countries each lies in, and the most words
# any of them has (lazily built).
_text_scan_tables: (
    tuple[dict[str, str], dict[str, frozenset[str]], int] | None
) = None


def _get_text_scan_tables() -> tuple[
    dict[str, str], dict[str, frozenset[str]], int
]:
    """Return the tables :func:`find_country_codes` reads."""
    global _text_scan_tables
    if _text_scan_tables is None:
        names = dict(_get_reverse_index())
        names.update(_ALIASES)
        _, places = _get_other_places()
        max_words = max(
            len(_WORD_RE.findall(key)) for key in [*names, *places]
        )
        _text_scan_tables = (names, places, max_words)
    return _text_scan_tables


def _country_mentions(
    text: str, *, capitals: bool = False, places: bool = True,
) -> list[tuple[str, int, int]]:
    """Return ``(code, start, end)`` for each country *text* names, from left
    to right (:func:`find_country_codes`).  With ``places=False`` the names
    of other places are not looked for."""
    names, other_places, max_words = _get_text_scan_tables()
    spans = [m.span() for m in _WORD_RE.finditer(text)]
    mentions: list[tuple[str, int, int]] = []
    i = 0
    while i < len(spans):
        match = _name_at(
            text, spans, i, names, other_places if places else {},
            max_words, capitals,
        )
        if match is None:
            i += 1
            continue
        size, code, own, start, end = match
        if code is not None:
            mentions.append((code, start, end))
        else:
            # Another place's name: only its own countries' names count.
            mentions.extend(
                (inner, start + a, start + b)
                for inner, a, b in _country_mentions(
                    text[start:end], capitals=capitals, places=False,
                )
                if inner in own
            )
        i += size
    return mentions


def _name_at(
    text: str,
    spans: list[tuple[int, int]],
    i: int,
    names: dict[str, str],
    places: dict[str, frozenset[str]],
    max_words: int,
    capitals: bool,
) -> tuple[int, str | None, frozenset[str], int, int] | None:
    """Return ``(words, code, countries, start, end)`` for the longest name at
    word *i* of *text*: a country's (*code*), or another place's (*code*
    ``None``, *countries* the ones it lies in)."""
    for size in range(min(max_words, len(spans) - i), 0, -1):
        start, end = spans[i][0], spans[i + size - 1][1]
        phrase = " ".join(text[start:end].split())
        key = _normalize_country_text(phrase)
        if key.endswith("'s") and key not in names and (
            _place_key(key) not in places
        ):
            phrase, key = phrase[:-2], key[:-2]
        else:
            # Punctuation after the last word may complete a name: "U.S.",
            # "Virgin Islands (U.S.)".
            for n in range(3, 0, -1):
                tail = text[end:end + n]
                if len(tail) < n or any(c.isalnum() or c.isspace() for c in tail):
                    continue
                longer = _normalize_country_text(phrase + tail)
                if longer in names or _place_key(longer) in places:
                    phrase, key, end = phrase + tail, longer, end + n
                    break
        if _place_key(key) in places:
            return size, None, places[_place_key(key)], start, end
        if key in names:
            if key == "us" and phrase != "US":
                continue  # the pronoun
            if capitals and not phrase[:1].isupper():
                continue
            # A name read as a region elsewhere names the region's country.
            code = (_names_read_as_regions.get() or {}).get(key, (None, names[key]))[1]
            return size, code, frozenset(), start, end
        if size == 1 and _is_standalone_code(text, spans, i, phrase):
            return 1, phrase, frozenset(), start, end
    return None


def _is_standalone_code(
    text: str, spans: list[tuple[int, int]], i: int, word: str,
) -> bool:
    """Return True when *word*, word *i* of *text*, is an ISO code standing
    apart: in capitals, with punctuation or the text's edge on both sides."""
    if not (len(word) == 3 and word.isupper() and word in ISO_ALPHA3_NAMES):
        return False
    before = text[spans[i - 1][1]:spans[i][0]] if i > 0 else ""
    after = text[spans[i][1]:spans[i + 1][0]] if i + 1 < len(spans) else ""
    return not before.isspace() and not after.isspace()


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
