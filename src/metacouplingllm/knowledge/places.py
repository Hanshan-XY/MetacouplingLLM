"""
Reading the places a text names: its countries and its ADM1 regions.

:func:`read_places` finds the countries and the first-level regions a research
description or an analysis names, as the system prompt's adjacency hint reads
them.  :func:`place_readings` gives the country names such a text reads as
regions elsewhere ("Georgia" as the US state), for
:func:`~metacouplingllm.knowledge.countries.read_names_as_regions`, so that the
rest of a run resolves those names as the text reads them.
"""

from __future__ import annotations

import re

from metacouplingllm.knowledge.adm1_pericoupling import (
    _fold_diacritics,
    _region_names,
    get_adm1_country,
    get_adm1_info,
    resolve_adm1_code,
)
from metacouplingllm.knowledge.countries import (
    _country_mentions,
    _normalize_country_text,
    read_names_as_regions,
)
from metacouplingllm.knowledge.pericoupling import is_pericoupled

# Latin letters with diacritics, those of Vietnamese included ("Bình Phước").
_HINT_WORD_RE = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]+")
_APOSTROPHES = "'’"
# Lower-case words a region's name holds between capitalised ones ("Mato
# Grosso do Sul", "State of Mexico", "Côte d'Ivoire", "Dar es Salaam").
_NAME_CONNECTORS = frozenset({
    "ad", "al", "am", "an", "and", "as", "ash", "d", "da", "das", "de", "del",
    "della", "den", "der", "des", "di", "do", "dos", "du", "e", "ed", "el",
    "es", "et", "la", "le", "les", "of", "on", "sur", "the", "ul", "upon", "y",
})
# Lower-case unit words a region's name may end with ("Kyyivs'ka oblast",
# "the Tanga region").
_TRAILING_UNIT_WORDS = frozenset({
    "county", "department", "district", "division", "governorate", "oblast",
    "prefecture", "province", "region", "state", "voivodeship", "voivodship",
})
# A name followed by one of these words is a feature's ("Ohio River", "Mary
# River mine", "Paraná Basin"), and a name after "Lake", "Mount" or "Gulf of"
# is too ("Lake Tanganyika", "Gulf of California").
_FEATURE_WORDS_AFTER = frozenset({
    "aquifer", "basin", "bay", "canal", "dam", "delta", "desert", "estuary",
    "falls", "glacier", "hills", "island", "islands", "lagoon", "lake",
    "mine", "mountain", "mountains", "peninsula", "plateau", "range",
    "reservoir", "river", "sea", "sound", "strait", "valley",
})
_FEATURE_WORDS_BEFORE = frozenset({"lake", "mount", "mt"})
_FEATURE_PAIRS_BEFORE = frozenset({"bay of", "gulf of", "sea of", "strait of"})
# Words that are region names in some country but stand for a direction, a
# feature or a unit in most texts ("Bay", Somalia; "West", Rwanda; "Delta",
# Nigeria; "Forest", Guernsey).  Alone they name a region only in a text that
# names the region's country.
_GENERIC_PLACE_WORDS = frozenset({
    "bay", "capital", "central", "coast", "coastal", "delta", "desert",
    "east", "eastern", "forest", "greater", "gulf", "highlands", "inner",
    "island", "islands", "lake", "lakes", "lower", "lowlands", "middle",
    "mountains", "north", "northeast", "northern", "northwest", "outer",
    "plateau", "pool", "river", "rivers", "sea", "south", "southeast",
    "southern", "southwest", "upper", "valley", "west", "western",
})


def _hint_words(text: str) -> list[re.Match[str]]:
    """Return the words of *text*, without the "s" of a possessive "'s"."""
    return [
        m for m in _HINT_WORD_RE.finditer(text)
        if not (m.group(0) == "s" and m.start() > 0
                and text[m.start() - 1] in _APOSTROPHES)
    ]


def _written_as_name(text: str, words: list[re.Match[str]]) -> bool:
    """Return True when *words* are written as a place name: capitalised,
    with lower-case connectors inside, a lower-case unit word at the end,
    or a word continuing after an apostrophe ("Murang'a", "d'Azur")."""
    last = len(words) - 1
    for k, m in enumerate(words):
        word = m.group(0)
        if word[0].isupper():
            continue
        if k > 0 and text[m.start() - 1] in _APOSTROPHES:
            continue
        if 0 < k < last and word.lower() in _NAME_CONNECTORS:
            continue
        if 0 < k == last and word.lower() in _TRAILING_UNIT_WORDS:
            continue
        return False
    return True


def _region_at(
    text: str,
    words: list[re.Match[str]],
    i: int,
    size: int,
    named: list[str],
    hints: list[str],
    country_spans: list[tuple[int, int]],
    namesakes: dict[tuple[int, int], list[str]],
) -> str | None:
    """Return the code of the ADM1 region that words ``i .. i+size-1`` of
    *text* name, or ``None``.  *named* are the countries the text names,
    *hints* the countries a name several regions share is read in, and
    *namesakes* the countries' names to read as regions ("Georgia", the US
    state), by span, with the countries those regions lie in."""
    span = words[i:i + size]
    if not _written_as_name(text, span):
        return None
    start, end = span[0].start(), span[-1].end()
    # Never a piece of a country's name ("Zealand" in "New Zealand",
    # "Bissau" in "Guinea-Bissau"); a longer name may hold one ("State of
    # Mexico", "Western Australia").
    for a, b in country_spans:
        if a < end and start < b and not (start <= a and b <= end and (a, b) != (start, end)):
            return None
    before = [m.group(0).lower() for m in words[max(0, i - 2):i]]
    after = words[i + size].group(0).lower() if i + size < len(words) else ""
    if (before[-1:] and before[-1] in _FEATURE_WORDS_BEFORE) or (
        " ".join(before[-2:]) in _FEATURE_PAIRS_BEFORE
    ) or after in _FEATURE_WORDS_AFTER:
        return None
    phrase = " ".join(m.group(0) for m in span)
    if (start, end) in namesakes:
        return next(filter(None, (
            resolve_adm1_code(phrase, iso) for iso in namesakes[(start, end)]
        )), None)
    code = resolve_adm1_code(phrase)
    if code is None or get_adm1_country(code) not in hints:
        # A name several regions share is read in the text's countries
        # ("Central Region" in Ghana, "Punjab" in India).  A single word must
        # then be the region's whole name ("Santa" is not Santa Elena).
        for iso in hints:
            hinted = resolve_adm1_code(phrase, iso)
            info = get_adm1_info(hinted) if hinted else None
            if info and (size > 1 or _fold_diacritics(info["name"]).casefold()
                         == _fold_diacritics(phrase).casefold()):
                code = hinted
                break
    if code is None:
        return None
    if size == 1 and phrase.lower() in _GENERIC_PLACE_WORDS and (
        get_adm1_country(code) not in named
    ):
        return None
    return code


def _scan_regions(
    text: str,
    words: list[re.Match[str]],
    named: list[str],
    hints: list[str],
    country_spans: list[tuple[int, int]],
    namesakes: dict[tuple[int, int], list[str]],
) -> list[tuple[str, tuple[int, int]]]:
    """Return ``(code, span)`` for each ADM1 region *text* names, in order,
    trying runs of up to five words, longest first, and going on after each."""
    found: list[tuple[str, tuple[int, int]]] = []
    i = 0
    while i < len(words):
        for size in range(min(5, len(words) - i), 0, -1):
            code = _region_at(
                text, words, i, size, named, hints, country_spans, namesakes,
            )
            if code:
                found.append((code, (words[i].start(), words[i + size - 1].end())))
                i += size
                break
        else:
            i += 1
    return found


# ADM1 region names and aliases by normalized text, with the countries of the
# regions each names (lazily built).
_region_countries_by_name: dict[str, frozenset[str]] | None = None


def _countries_with_region_named(name: str) -> frozenset[str]:
    """Return the countries that have an ADM1 region called *name*."""
    global _region_countries_by_name
    if _region_countries_by_name is None:
        index: dict[str, frozenset[str]] = {}
        for key, isos in _region_names().items():
            key = _normalize_country_text(key)
            index[key] = index.get(key, frozenset()) | isos
        _region_countries_by_name = index
    return _region_countries_by_name.get(_normalize_country_text(name), frozenset())


def _read(text: str) -> tuple[list[str], list[str], dict[str, tuple[str, str]]]:
    """Return what :func:`read_places` and :func:`place_readings` return,
    reading the names as the text writes them, whatever
    :func:`~metacouplingllm.knowledge.countries.read_names_as_regions`
    block the caller is in."""
    with read_names_as_regions({}):
        return _read_as_written(text)


def _read_as_written(
    text: str,
) -> tuple[list[str], list[str], dict[str, tuple[str, str]]]:
    """:func:`_read` outside any reading block."""
    words = _hint_words(text)
    mentions = []
    for code, a, b in _country_mentions(text):
        if text[b - 2:b] in ("'s", "’s"):  # "Australia's" in "Western Australia's"
            b -= 2
        mentions.append((code, a, b))
    namesakes: dict[tuple[int, int], list[str]] = {}
    for code, a, b in mentions:
        isos = _countries_with_region_named(text[a:b]) - {code}
        if isos:
            namesakes[(a, b)] = sorted(isos)
    as_regions: dict[tuple[int, int], str] = {}
    if namesakes:
        named = list(dict.fromkeys(code for code, _, _ in mentions))
        hints = list(dict.fromkeys(named + [i for v in namesakes.values() for i in v]))
        others = [(a, b) for _, a, b in mentions if (a, b) not in namesakes]
        found = _scan_regions(text, words, named, hints, others, namesakes)
        regions = [get_adm1_country(c) for c, span in found if span not in namesakes]
        for code, a, b in mentions:
            if (a, b) not in namesakes:
                continue
            hit = next((c for c, span in found if span == (a, b)), None)
            if hit is None:
                continue
            iso = get_adm1_country(hit)
            elsewhere = {c for c, x, y in mentions if (x, y) != (a, b)}
            for_region = (iso in elsewhere) + regions.count(iso)
            for_country = regions.count(code) + sum(
                bool(is_pericoupled(code, c)) for c in elsewhere - {code}
            )
            if for_region > for_country:
                as_regions[(a, b)] = hit
        namesakes = {span: v for span, v in namesakes.items() if span in as_regions}
        mentions = [m for m in mentions if (m[1], m[2]) not in as_regions]
    named = list(dict.fromkeys(code for code, _, _ in mentions))
    spans = [(a, b) for _, a, b in mentions]
    hints = list(dict.fromkeys(named + [i for v in namesakes.values() for i in v]))
    found = _scan_regions(text, words, named, hints, spans, namesakes)
    more = list(dict.fromkeys(hints + [get_adm1_country(c) for c, _ in found]))
    if len(more) > len(hints):
        found = _scan_regions(text, words, named, more, spans, namesakes)
    readings = {
        _normalize_country_text(text[a:b]): (code, get_adm1_country(code))
        for (a, b), code in as_regions.items()
    }
    return named, [code for code, _ in found], readings


def read_places(text: str) -> tuple[list[str], list[str]]:
    """Return the countries *text* names and the ADM1 regions it names, in
    the order it names them.

    A run of words counts as a region's name only when it is written as one
    (:func:`_written_as_name`), is not part of a country's name ("Zealand" in
    "New Zealand") or of a feature's ("Lake Tanganyika", "Ohio River"), and,
    for a lone direction or feature word ("Bay", "West"), when the text names
    the region's country.  A name several regions share is read in the
    countries the text names, then in those of the other regions it names.

    A country's name that a region elsewhere also has ("Georgia", a US state;
    "Gaza", a province of Mozambique) is read as the one the rest of the text
    supports more: the region when the text names the region's country or
    more of its regions, the country when it names more of the country's
    regions or of the countries bordering it.  A tie keeps the country.  The
    region's reading lets the text's other names be read in its country: in
    "Florida and Georgia", "Florida", which Uruguay also has, is the US state.
    """
    named, regions, _ = _read(text)
    return named, regions


def place_readings(text: str) -> dict[str, tuple[str, str]]:
    """Return the country names *text* reads as regions elsewhere, as
    :func:`read_places` reads them: by name as
    :func:`~metacouplingllm.knowledge.countries.read_names_as_regions` takes
    them, each with the region's ADM1 code and country.

    ``{"georgia": ("USA011", "USA")}`` for "Peanut farms in Georgia, United
    States"; ``{}`` for "Wine exports from Georgia to Russia".
    """
    return _read(text)[2]
