"""
ADM1 (subnational) pericoupling database.

Loads a curated CSV edge list of first-level administrative divisions (ADM1)
to determine which subnational regions share a border (pericoupled).  The
database uses World Bank ADM1 codes (e.g., ``"MEX001"``, ``"USA035"``).

The edge list contains **8,453 border pairs** covering **3,375 unique ADM1
regions** across **196 countries**, including both within-country and
cross-country borders.  This total includes five reviewed overlays (see
``data/PROVENANCE.md``): a **6-pair river-gap overlay**
(``river_gap_overlay_pairs.csv``): cross-border river-separated pairs the
strict topology omits because the source digitizes the two banks as
non-touching geometry, recovered by the near-miss audit; a **13-pair
wide-river overlay** (``wide_river_overlay_pairs.csv``) that reclassifies
*existing* edges as water-only (it adds no edges); a **64-pair lake
overlay** (``lake_overlay_pairs.csv``) restoring the audited lake-only pairs
the build's lake filter removes (63 touching pairs plus one audited lake-gap
near-miss, Jogeva<->Pskov across Lake Peipus), so ``coupling_standard`` governs river and
lake borders uniformly (``lenient`` keeps all water borders, ``moderate``
keeps only those with a fixed crossing, ``stringent`` keeps none); and a
**2-pair land-gap overlay** (``land_gap_overlay_pairs.csv``) restoring two
map-verified dry-land survey-line borders on the Kenya-Tanzania frontier
(Kajiado<->Kilimanjaro, Narok<->Mara) that the snap tolerance misses because
the two national boundary renderings sit ~100 m apart; these are ordinary
land edges, pericoupled under every standard; and a **3-pair audit-water
overlay** (``audit_water_overlay_pairs.csv``) reclassifying as water-only three
existing edges the Natural Earth screens missed (Shirak<->Kars on the Akhurian,
Vratca<->Olt on a Danube span, Kagera<->Ntungamo on a 402 m Kagera-thalweg
arc), flagged by the 10 km completeness audit and confirmed by human map review
(flags only, no new edges).

Source
------
``pericoupled_adm1_edge_list.csv``, regenerated in PR #50 from the World Bank
Official Boundaries dataset (GeoPackage, 2026-05-14 release) using World Bank
ADM1 codes and current ISO 3166-1 alpha-3 country codes.  Adjacency is the
shared land border between region polygons (geodesic ``border_length_km``);
see ``data/PROVENANCE.md`` for the full method.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
import re

from metacouplingllm.knowledge.countries import resolve_country_code


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Adm1PairType(str, Enum):
    """Result of an ADM1 pericoupling database lookup."""

    PERICOUPLED = "pericoupled"
    TELECOUPLED = "telecoupled"
    SAME_REGION = "same_region"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Adm1PericouplingResult:
    """Result of looking up an ADM1 region pair in the pericoupling database.

    Attributes
    ----------
    pair_type:
        Whether the pair is pericoupled, telecoupled, same_region, or unknown.
    code_a:
        ADM1 code for the first region, or ``None`` if unresolved.
    code_b:
        ADM1 code for the second region, or ``None`` if unresolved.
    cross_country:
        ``True`` if the two regions belong to different countries.
    confidence:
        ``"database"`` if looked up from the database, ``"unresolved"`` if
        one or both ADM1 codes are not in the database, ``"same_region"``
        if both codes are identical.
    """

    pair_type: Adm1PairType
    code_a: str | None
    code_b: str | None
    cross_country: bool = False
    confidence: str = "database"


# ---------------------------------------------------------------------------
# Internal cached data structures
# ---------------------------------------------------------------------------

_adm1_pairs: frozenset[frozenset[str]] | None = None
_adm1_neighbors: dict[str, set[str]] | None = None
_adm1_country: dict[str, str] | None = None
_adm1_metadata: dict[str, dict[str, str]] | None = None
_adm1_disputed_overlay: frozenset[frozenset[str]] | None = None
_adm1_water_all: frozenset[frozenset[str]] | None = None
_adm1_water_nobridge: frozenset[frozenset[str]] | None = None
_adm1_aliases: dict[str, list[tuple[str, str]]] | None = None

_COUPLING_STANDARDS = ("stringent", "moderate", "lenient")


def _check_standard(coupling_standard: str) -> str:
    """Validate/normalise a ``coupling_standard`` value."""
    s = coupling_standard.strip().lower()
    if s not in _COUPLING_STANDARDS:
        raise ValueError(
            "coupling_standard must be one of "
            f"{_COUPLING_STANDARDS}, got {coupling_standard!r}"
        )
    return s


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _locate_csv() -> Path | None:
    """Find the ADM1 pericoupling CSV data file."""
    try:
        data_pkg = resources.files("metacouplingllm") / "data"
    except (TypeError, ModuleNotFoundError):
        return None

    candidate = data_pkg / "pericoupled_adm1_edge_list.csv"
    try:
        path = Path(str(candidate))
        if path.is_file():
            return path
    except Exception:
        pass
    return None


def _load_adm1_data() -> tuple[
    frozenset[frozenset[str]],
    dict[str, set[str]],
    dict[str, str],
    dict[str, dict[str, str]],
]:
    """Load all ADM1 data from the CSV file.

    Returns
    -------
    Tuple of:
    - pairs: frozenset of frozenset pairs (all adjacent ADM1 code pairs)
    - neighbors: adjacency index {code: {neighbor_codes}}
    - country_map: {adm1_code: iso_a3_code}
    - metadata: {adm1_code: {name, country_name, iso_a3, wb_region}}
    """
    csv_path = _locate_csv()
    if csv_path is None:
        return frozenset(), {}, {}, {}

    pairs: set[frozenset[str]] = set()
    neighbors: dict[str, set[str]] = {}
    country_map: dict[str, str] = {}
    metadata: dict[str, dict[str, str]] = {}

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code_a = row["ADM1_code_A"].strip()
            code_b = row["ADM1_code_B"].strip()

            # Record the pair
            pairs.add(frozenset({code_a, code_b}))

            # Build adjacency index
            neighbors.setdefault(code_a, set()).add(code_b)
            neighbors.setdefault(code_b, set()).add(code_a)

            # Record country mapping and metadata for both sides
            for suffix, code in [("_A", code_a), ("_B", code_b)]:
                if code not in country_map:
                    iso = row[f"ISO_A3{suffix}"].strip()
                    country_map[code] = iso
                    metadata[code] = {
                        "name": row[f"ADM1_name{suffix}"].strip(),
                        "country_name": row[f"country{suffix}"].strip(),
                        "iso_a3": iso,
                        "wb_region": row[f"WB_region{suffix}"].strip(),
                    }

    return frozenset(pairs), neighbors, country_map, metadata


def _load_disputed_overlay() -> frozenset[frozenset[str]]:
    """Load the de-facto ADM1 disputed-territory overlay pairs.

    Reads ``disputed_overlay_pairs.csv`` (shipped beside the data) and returns
    the set of ADM1 code pairs whose adjacency exists only under the de-facto
    view; these are subtracted when ``de_facto_borders=False``.  Empty set if
    the manifest is absent (older data) — both views then identical.
    """
    try:
        data_pkg = resources.files("metacouplingllm") / "data"
        path = Path(str(data_pkg / "disputed_overlay_pairs.csv"))
    except (TypeError, ModuleNotFoundError):
        return frozenset()
    if not path.is_file():
        return frozenset()
    overlay: set[frozenset[str]] = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("level", "").strip() == "adm1":
                overlay.add(
                    frozenset({row["code_a"].strip(), row["code_b"].strip()})
                )
    return frozenset(overlay)


def _load_water_separated() -> tuple[
    frozenset[frozenset[str]], frozenset[frozenset[str]]
]:
    """Load water-separated ADM1 pairs for the ``coupling_standard`` filter.

    Reads ``water_separated_pairs.csv`` (rows with ``level == "adm1"``): pairs
    whose shared border is water only.  Returns ``(all_water, no_bridge)`` —
    the second is the subset lacking an open fixed crossing.  Under
    ``moderate`` the ``no_bridge`` set is subtracted from the (lenient) base;
    under ``stringent`` the full ``all_water`` set is.  Empty sets if the
    manifest is absent (older data) — all standards then identical.
    """
    try:
        data_pkg = resources.files("metacouplingllm") / "data"
        path = Path(str(data_pkg / "water_separated_pairs.csv"))
    except (TypeError, ModuleNotFoundError):
        return frozenset(), frozenset()
    if not path.is_file():
        return frozenset(), frozenset()
    all_water: set[frozenset[str]] = set()
    no_bridge: set[frozenset[str]] = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("level", "").strip() != "adm1":
                continue
            pair = frozenset({row["code_a"].strip(), row["code_b"].strip()})
            all_water.add(pair)
            if row.get("has_bridge", "").strip() != "True":
                no_bridge.add(pair)
    return frozenset(all_water), frozenset(no_bridge)


def _load_adm1_aliases() -> dict[str, list[tuple[str, str]]]:
    """Load the English-exonym alias table from ``adm1_aliases.csv``.

    Returns a dict mapping lowercase alias key → [(adm1_code, iso_a3), ...]
    in the same format as the name index, so ``_pick_best_candidate`` can
    be reused for country filtering.  Returns ``{}`` if the file is absent
    OR malformed (graceful — a corrupt alias file degrades to "no Strategy 0",
    it must never break the unrelated ADM1 lookups that share ``_ensure_loaded``).
    Rows missing any of the three required columns are skipped individually.
    """
    try:
        data_pkg = resources.files("metacouplingllm") / "data"
        path = Path(str(data_pkg / "adm1_aliases.csv"))
    except (TypeError, ModuleNotFoundError):
        return {}
    if not path.is_file():
        return {}
    aliases: dict[str, list[tuple[str, str]]] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                # row.get(...) tolerates missing columns / short rows
                # (DictReader fills absent trailing fields with None).
                key = (row.get("alias_key") or "").strip()
                code = (row.get("code") or "").strip()
                iso = (row.get("iso_a3") or "").strip()
                if key and code and iso:
                    aliases.setdefault(key, []).append((code, iso))
    except (OSError, csv.Error):
        return {}
    return aliases


def _ensure_loaded() -> None:
    """Lazily load and cache all ADM1 data."""
    global _adm1_pairs, _adm1_neighbors, _adm1_country, _adm1_metadata
    global _adm1_disputed_overlay, _adm1_water_all, _adm1_water_nobridge
    global _adm1_aliases
    if _adm1_pairs is None:
        _adm1_pairs, _adm1_neighbors, _adm1_country, _adm1_metadata = (
            _load_adm1_data()
        )
    if _adm1_disputed_overlay is None:
        _adm1_disputed_overlay = _load_disputed_overlay()
    if _adm1_water_all is None:
        _adm1_water_all, _adm1_water_nobridge = _load_water_separated()
    if _adm1_aliases is None:
        _adm1_aliases = _load_adm1_aliases()


def _water_dropped(coupling_standard: str) -> frozenset[frozenset[str]]:
    """Water-only pairs to subtract from the base under a coupling standard.

    ``lenient`` keeps all water borders; ``moderate`` drops water-only pairs
    with no open fixed crossing; ``stringent`` drops every water-only pair.
    """
    if coupling_standard == "stringent":
        return _adm1_water_all or frozenset()
    if coupling_standard == "moderate":
        return _adm1_water_nobridge or frozenset()
    return frozenset()


def _pair_in_db(
    pair: frozenset[str], de_facto_borders: bool, coupling_standard: str
) -> bool:
    """Whether a pair is adjacent under the requested view.

    The shipped data is the de-facto / lenient view; the strict view subtracts
    the disputed-territory overlay, and ``coupling_standard`` subtracts
    water-separated pairs (see :func:`_water_dropped`).
    """
    assert _adm1_pairs is not None
    if pair not in _adm1_pairs:
        return False
    if not de_facto_borders and _adm1_disputed_overlay:
        if pair in _adm1_disputed_overlay:
            return False
    if pair in _water_dropped(coupling_standard):
        return False
    return True


def _filter_neighbors(
    code: str,
    neighbors: set[str],
    de_facto_borders: bool,
    coupling_standard: str,
) -> set[str]:
    """Drop overlay (strict view) and water-separated (coupling standard) neighbours."""
    overlay = (
        _adm1_disputed_overlay
        if (not de_facto_borders and _adm1_disputed_overlay)
        else None
    )
    dropped = _water_dropped(coupling_standard)
    if not overlay and not dropped:
        return neighbors
    result: set[str] = set()
    for n in neighbors:
        pair = frozenset({code, n})
        if overlay and pair in overlay:
            continue
        if pair in dropped:
            continue
        result.add(n)
    return result


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def lookup_adm1_pericoupling(
    adm1_a: str,
    adm1_b: str,
    de_facto_borders: bool = True,
    coupling_standard: str = "moderate",
) -> Adm1PericouplingResult:
    """Look up whether two ADM1 regions are pericoupled or telecoupled.

    Parameters
    ----------
    adm1_a:
        ADM1 code of the first region (e.g., ``"MEX001"``).
    adm1_b:
        ADM1 code of the second region (e.g., ``"USA035"``).

    Returns
    -------
    An :class:`Adm1PericouplingResult` with the lookup outcome.

    Examples
    --------
    >>> result = lookup_adm1_pericoupling("AFG001", "PAK005")
    >>> result.pair_type
    <Adm1PairType.PERICOUPLED: 'pericoupled'>
    >>> result.cross_country
    True
    """
    _ensure_loaded()
    coupling_standard = _check_standard(coupling_standard)
    assert _adm1_pairs is not None
    assert _adm1_country is not None

    adm1_a = adm1_a.strip()
    adm1_b = adm1_b.strip()

    # Check if codes are in the database
    a_known = adm1_a in _adm1_country
    b_known = adm1_b in _adm1_country

    if not a_known or not b_known:
        return Adm1PericouplingResult(
            pair_type=Adm1PairType.UNKNOWN,
            code_a=adm1_a if a_known else None,
            code_b=adm1_b if b_known else None,
            cross_country=False,
            confidence="unresolved",
        )

    if adm1_a == adm1_b:
        return Adm1PericouplingResult(
            pair_type=Adm1PairType.SAME_REGION,
            code_a=adm1_a,
            code_b=adm1_b,
            cross_country=False,
            confidence="same_region",
        )

    cross_country = _adm1_country[adm1_a] != _adm1_country[adm1_b]
    pair = frozenset({adm1_a, adm1_b})

    if _pair_in_db(pair, de_facto_borders, coupling_standard):
        return Adm1PericouplingResult(
            pair_type=Adm1PairType.PERICOUPLED,
            code_a=adm1_a,
            code_b=adm1_b,
            cross_country=cross_country,
            confidence="database",
        )

    return Adm1PericouplingResult(
        pair_type=Adm1PairType.TELECOUPLED,
        code_a=adm1_a,
        code_b=adm1_b,
        cross_country=cross_country,
        confidence="database",
    )


def get_adm1_neighbors(
    adm1_code: str,
    de_facto_borders: bool = True,
    coupling_standard: str = "moderate",
) -> set[str]:
    """Return all ADM1 codes adjacent to the given region.

    Includes both within-country and cross-border neighbors.

    Parameters
    ----------
    adm1_code:
        ADM1 code (e.g., ``"AFG001"``).

    Returns
    -------
    Set of adjacent ADM1 codes.  Empty set if the code is not in the
    database.

    Examples
    --------
    >>> neighbors = get_adm1_neighbors("AFG001")
    >>> "PAK005" in neighbors
    True
    >>> "AFG024" in neighbors
    True
    """
    _ensure_loaded()
    coupling_standard = _check_standard(coupling_standard)
    assert _adm1_neighbors is not None
    code = adm1_code.strip()
    return _filter_neighbors(
        code, set(_adm1_neighbors.get(code, set())), de_facto_borders,
        coupling_standard,
    )


def get_cross_border_neighbors(
    adm1_code: str,
    de_facto_borders: bool = True,
    coupling_standard: str = "moderate",
) -> set[str]:
    """Return ADM1 codes adjacent to the given region in *different* countries.

    Parameters
    ----------
    adm1_code:
        ADM1 code (e.g., ``"AFG001"``).

    Returns
    -------
    Set of adjacent ADM1 codes that belong to a different country.

    Examples
    --------
    >>> cross = get_cross_border_neighbors("AFG001")
    >>> "PAK005" in cross
    True
    >>> "AFG024" in cross  # same country
    False
    """
    _ensure_loaded()
    coupling_standard = _check_standard(coupling_standard)
    assert _adm1_neighbors is not None
    assert _adm1_country is not None

    adm1_code = adm1_code.strip()
    if adm1_code not in _adm1_country:
        return set()

    focal_iso = _adm1_country[adm1_code]
    candidates = _filter_neighbors(
        adm1_code, set(_adm1_neighbors.get(adm1_code, set())), de_facto_borders,
        coupling_standard,
    )
    result: set[str] = set()
    for neighbor in candidates:
        if _adm1_country.get(neighbor) != focal_iso:
            result.add(neighbor)
    return result


def get_adm1_codes_for_country(country: str) -> set[str]:
    """Return all ADM1 codes belonging to a country.

    Parameters
    ----------
    country:
        Country name (e.g., ``"Afghanistan"``) or ISO alpha-3 code
        (e.g., ``"AFG"``).

    Returns
    -------
    Set of ADM1 codes for that country.  Empty set if the country
    cannot be resolved or has no ADM1 regions in the database.

    Examples
    --------
    >>> codes = get_adm1_codes_for_country("Afghanistan")
    >>> "AFG001" in codes
    True
    """
    _ensure_loaded()
    assert _adm1_country is not None

    # Try direct ISO code match first
    country_stripped = country.strip().upper()
    if len(country_stripped) == 3:
        # Check if it's a valid ISO code in our data
        matching = {
            code
            for code, iso in _adm1_country.items()
            if iso == country_stripped
        }
        if matching:
            return matching

    # Try resolving via countries.py
    iso_code = resolve_country_code(country)
    if iso_code is None:
        return set()

    return {
        code for code, iso in _adm1_country.items() if iso == iso_code
    }


def get_adm1_info(adm1_code: str) -> dict[str, str] | None:
    """Return metadata for an ADM1 region.

    Parameters
    ----------
    adm1_code:
        ADM1 code (e.g., ``"AFG001"``).

    Returns
    -------
    A dictionary with keys ``name``, ``country_name``, ``iso_a3``,
    ``wb_region``, or ``None`` if the code is not in the database.

    Examples
    --------
    >>> info = get_adm1_info("AFG001")
    >>> info["name"]
    'Badakhshan'
    >>> info["country_name"]
    'Afghanistan'
    """
    _ensure_loaded()
    assert _adm1_metadata is not None
    return _adm1_metadata.get(adm1_code.strip())


def get_adm1_aliases(adm1_code: str) -> list[str]:
    """Return alias keys for an ADM1 region from the exonym table.

    Parameters
    ----------
    adm1_code:
        ADM1 code (e.g., ``"DEU002"``).

    Returns
    -------
    List of lowercase alias keys (e.g., ``["bavaria"]``), or ``[]`` if
    the code has no aliases or ``adm1_aliases.csv`` is absent.  Keys are
    the normalized lookup strings, not display-form strings.
    """
    _ensure_loaded()
    assert _adm1_aliases is not None
    return [
        key
        for key, entries in _adm1_aliases.items()
        if any(code == adm1_code.strip() for code, _ in entries)
    ]


def is_adm1_pericoupled(
    adm1_a: str,
    adm1_b: str,
    de_facto_borders: bool = True,
    coupling_standard: str = "moderate",
) -> bool | None:
    """Convenience function for quick ADM1 pericoupling checks.

    Returns
    -------
    ``True`` if the two regions share a border, ``False`` if they do
    not, ``None`` if one or both codes are not in the database.

    See :func:`lookup_adm1_pericoupling` for ``de_facto_borders`` (default
    ``True`` treats disputed land as part of its de-facto administrator).
    """
    result = lookup_adm1_pericoupling(
        adm1_a, adm1_b, de_facto_borders, coupling_standard
    )
    if result.pair_type == Adm1PairType.PERICOUPLED:
        return True
    if result.pair_type in (Adm1PairType.TELECOUPLED, Adm1PairType.SAME_REGION):
        return False
    return None


def get_adm1_country(adm1_code: str) -> str | None:
    """Return the ISO alpha-3 country code for an ADM1 region.

    Parameters
    ----------
    adm1_code:
        ADM1 code (e.g., ``"MEX001"``).

    Returns
    -------
    ISO alpha-3 code (e.g., ``"MEX"``), or ``None`` if the code is
    not in the database.
    """
    _ensure_loaded()
    assert _adm1_country is not None
    return _adm1_country.get(adm1_code.strip())


# ---------------------------------------------------------------------------
# ADM1 name-to-code resolution
# ---------------------------------------------------------------------------

# Known suffixes in World Bank ADM1 naming conventions.
# These are stripped during fuzzy matching so "Anhui" matches "Anhui Sheng".
_ADM1_NAME_SUFFIXES: tuple[str, ...] = (
    "sheng",
    "oblast",
    "okrug",
    "kray",
    "pradesh",
    "laen",
    "novads",
    "maakond",
    "region",
    "province",
    "state",
    "zizhiqu",
    "rep.",
    "giang",
)

# Connector / article words that may pad a query around a region name without
# changing which region it denotes ("State of Bavaria").  Combined with the
# administrative suffixes above, these are the ONLY extra words tolerated when
# a *query* contains a canonical name (Direction A, see ``_substring_match``).
_DIRECTION_A_CONNECTORS: frozenset[str] = frozenset({
    "the", "of", "de", "del", "la", "el", "da", "do",
    "und", "and", "du", "des", "van", "von",
})
_DIRECTION_A_ADMIN_NOUNS: frozenset[str] = frozenset({
    # Romance / other-language equivalents of the English nouns already in
    # _ADM1_NAME_SUFFIXES.  These appear in caller queries (e.g. LLM output
    # in Spanish: "Estado de Jalisco") but NOT in the WB canonical names for
    # most regions, so they end up as Direction-A leftovers.  'ciudad' is
    # deliberately excluded so "Mexico City" → None stays intact.
    "estado", "provincia", "comunidad", "departamento",
    "prefectura", "gobernacion", "distrito",
})
_IGNORABLE_SUBSTRING_WORDS: frozenset[str] = (
    frozenset(_ADM1_NAME_SUFFIXES) | _DIRECTION_A_CONNECTORS | _DIRECTION_A_ADMIN_NOUNS
)


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-word (word-boundary) containment of ``needle`` in ``haystack``."""
    pattern = rf"(?<![a-z]){re.escape(needle)}(?![a-z])"
    return re.search(pattern, haystack) is not None


# Split a name into words on whitespace AND hyphens, so a hyphen-joined
# qualifier (the French "nouveau-brunswick" = New Brunswick) is treated the
# same as a space-separated one ("new brunswick").
_WORD_SPLIT_RE = re.compile(r"[\s\-]+")


def _name_words(text: str) -> list[str]:
    return [w for w in _WORD_SPLIT_RE.split(text) if w]


def _substring_match(query: str, db_name: str) -> bool:
    """Direction-aware substring match between a query and a DB name.

    **Direction B** (the canonical ``db_name`` contains the ``query`` as a
    whole phrase) is a legitimate short / common-name match -- e.g.
    ``"michoacan"`` inside ``"michoacan de ocampo"`` -- and is accepted only
    when the words *preceding* the query in ``db_name`` are all ignorable
    (connectors / admin nouns).  A meaningful preceding word makes the
    canonical a *different*, more-specific region: ``"york"`` ⊂ ``"new york"``
    (preceded by ``new``) and ``"brunswick"`` ⊂ ``"nouveau-brunswick"``
    (preceded by ``nouveau``) are rejected, so a stray token does not
    confidently resolve to New York / New Brunswick.  Trailing words are fine
    (they are the official long-form suffix, ``"… de ocampo"``).

    **Direction A** (the ``query`` contains the ``db_name``) is symmetric:
    accepted only when every leftover query word is ignorable.  A meaningful
    extra word means the query denotes a different place -- ``"mexico city"`` ⊃
    ``"mexico"`` leaves ``{"city"}`` and is rejected.

    Both directions reject the "qualifier + name" form, which is the class of
    confident-wrong-answer this guard exists to prevent.
    """
    if _contains_phrase(db_name, query):
        d = _name_words(db_name)
        q = _name_words(query)
        n = len(q)
        for i in range(len(d) - n + 1):
            if d[i:i + n] == q:
                # Accept iff only ignorable words precede the query (head match).
                return all(w in _IGNORABLE_SUBSTRING_WORDS for w in d[:i])
        # Matched as a substring but not as a clean token run (punctuation
        # quirk) -- preserve the prior permissive behaviour for that edge.
        return True
    if _contains_phrase(query, db_name):
        db_words = set(_name_words(db_name))
        leftover = [w for w in _name_words(query) if w not in db_words]
        return bool(leftover) and all(
            w in _IGNORABLE_SUBSTRING_WORDS for w in leftover
        )
    return False


_adm1_name_index: dict[str, list[tuple[str, str]]] | None = None


def _build_adm1_name_index() -> dict[str, list[tuple[str, str]]]:
    """Build a reverse index from lowercase name → [(adm1_code, iso_a3), ...].

    Also indexes suffix-stripped and slash-split variants.
    """
    _ensure_loaded()
    assert _adm1_metadata is not None

    index: dict[str, list[tuple[str, str]]] = {}

    for code, meta in _adm1_metadata.items():
        raw_name = meta["name"]
        iso = meta["iso_a3"]
        entry = (code, iso)

        names_to_index: set[str] = set()

        # 1. Full name (lowercased)
        names_to_index.add(raw_name.lower())

        # 2. Suffix-stripped variants
        name_lower = raw_name.lower()
        for suffix in _ADM1_NAME_SUFFIXES:
            if name_lower.endswith(" " + suffix):
                stripped = name_lower[: -(len(suffix) + 1)].strip()
                if stripped and len(stripped) >= 2:
                    names_to_index.add(stripped)

        # 3. Slash-split variants (e.g., "Cataluña/Catalunya")
        if "/" in raw_name:
            for part in raw_name.split("/"):
                part = part.strip().lower()
                if part and len(part) >= 2:
                    names_to_index.add(part)

        for n in names_to_index:
            index.setdefault(n, []).append(entry)

    return index


def _get_adm1_name_index() -> dict[str, list[tuple[str, str]]]:
    """Lazily build and cache the ADM1 name-to-code index."""
    global _adm1_name_index
    if _adm1_name_index is None:
        _adm1_name_index = _build_adm1_name_index()
    return _adm1_name_index


# Letters NFKD cannot decompose (they are distinct letters, not
# letter + combining mark), mapped to their conventional ASCII
# transliterations so native-script queries such as "Łódź" or
# "Đắk Lắk" fold to the ASCII names the database stores.
_NON_DECOMPOSABLE = str.maketrans({
    "Đ": "D", "đ": "d",
    "Ł": "L", "ł": "l",
    "Ø": "O", "ø": "o",
    "Ð": "D", "ð": "d",
    "Þ": "Th", "þ": "th",
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "ß": "ss",
    "ı": "i",
})


def _fold_diacritics(text: str) -> str:
    """NFKD-normalize, strip combining marks, and transliterate
    the handful of letters NFKD cannot decompose.

    Used by the accent-folded fallback in ``resolve_adm1_code``
    (PR #45) so that unaccented user input — common in English
    text and many LLM outputs — still resolves accented region
    names in the database (e.g. ``"Michoacan"`` → ``MEX016``
    even though the DB name is ``"Michoacán de Ocampo"``), and
    conversely so that native-script input with non-decomposable
    letters (``"Łódź"``, ``"Đắk Lắk"``) still resolves the ASCII
    names the database stores.
    """
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    ).translate(_NON_DECOMPOSABLE)


_adm1_folded_index: dict[str, list[tuple[str, str]]] | None = None


def _get_adm1_folded_name_index() -> dict[str, list[tuple[str, str]]]:
    """Lazily build and cache an accent-folded version of the
    name index for the PR #45 fallback.

    Built by folding every key in the accented index; entries
    are deduplicated per folded key so that a single resolved
    region doesn't appear twice when both its full and stripped
    forms fold to the same string.
    """
    global _adm1_folded_index
    if _adm1_folded_index is not None:
        return _adm1_folded_index
    base = _get_adm1_name_index()
    folded: dict[str, list[tuple[str, str]]] = {}
    for name, entries in base.items():
        f = _fold_diacritics(name)
        bucket = folded.setdefault(f, [])
        for entry in entries:
            if entry not in bucket:
                bucket.append(entry)
    _adm1_folded_index = folded
    return _adm1_folded_index


def _resolve_country_filter(country: str) -> str | None:
    """Resolve a country hint to an ISO alpha-3 code for filtering.

    Mirrors :func:`get_adm1_codes_for_country`: a 3-letter token that is
    itself an ISO code present in the loaded ADM1 data is honored
    directly, so hints such as ``"COD"`` (DR Congo) constrain correctly
    even when ``countries.py`` lacks that mapping.  Falls back to
    :func:`resolve_country_code` for country names and other forms.
    """
    _ensure_loaded()
    assert _adm1_country is not None
    stripped = country.strip().upper()
    if len(stripped) == 3 and stripped in set(_adm1_country.values()):
        return stripped
    return resolve_country_code(country)


def _is_cross_country_folded_collision(name_lower: str) -> bool:
    """Return ``True`` when ``name_lower`` is an *unaccented* query whose
    accent-folded spelling is shared by ADM1 regions in more than one
    country.

    This is the cross-country analogue of the within-country ambiguity
    that ``_pick_best_candidate`` already contains (PR #45).  The
    motivating case: the Democratic Republic of the Congo stores its
    province *with* the accent as ``"Équateur"`` (``COD013``) while the
    Central African Republic stores its as the bare ``"Equateur"``
    (``CAF002``).  A query of ``"Equateur"`` therefore found an exact
    Strategy-1 match for the CAR province and silently resolved to
    ``CAF002`` — an artifact of which country happens to omit the accent,
    not a real signal of which region the caller meant.

    The guard fires only when the query itself carries no diacritics
    (folding is a no-op).  An *accented* query such as ``"Équateur"``
    changes under folding, so this returns ``False`` and that query
    keeps its exact accented Strategy-1 match (``COD013``).  A resolved
    country hint also bypasses this guard (checked at the call site),
    letting the strategies disambiguate.
    """
    if _fold_diacritics(name_lower) != name_lower:
        return False
    entries = _get_adm1_folded_name_index().get(name_lower)
    if not entries:
        return False
    return len({iso for _, iso in entries}) > 1


def resolve_adm1_code(
    name: str,
    country: str | None = None,
) -> str | None:
    """Resolve an ADM1 region name to its World Bank ADM1 code.

    Four resolution strategies, applied in order, returning the
    first successful match:

    0. **(PR #60)** English-exonym alias table (``adm1_aliases.csv``):
       exact match against the pre-validated alias index.  Covers
       common English names that differ from the WB canonical form
       (e.g. ``"Mexico City"`` → ``MEX009``, ``"Bavaria"`` → ``DEU02``).
       No-op when the CSV is absent (graceful fallback to strategies 1–3).
    1. Direct lookup against the accented index built from the DB
       canonical names plus suffix-stripped and slash-split
       variants.
    2. Substring containment (either direction, both sides ≥ 4
       chars) against the same accented index.
    3. **(PR #45)** Accent-folded fallback: NFKD-fold the query
       and retry strategies 1/2 against a folded version of the
       index.  Recovers common English / unaccented surface forms
       of accented DB names (``"Michoacan"`` → ``MEX016`` for
       ``"Michoacán de Ocampo"``, ``"Sao Paulo"`` → ``BRA029``
       for ``"São Paulo"``, ``"Nuevo Leon"`` → ``MEX019``).
       Ambiguity from folding is contained by reusing
       ``_pick_best_candidate`` (single ISO or matching country
       filter required).

    Cross-country ambiguity
    -----------------------
    If the query is an *unaccented* name whose accent-folded spelling is
    shared by ADM1 regions in more than one country, it cannot identify a
    single region and ``None`` is returned unless a ``country`` hint is
    given.  The motivating case is ``"Equateur"``: the DRC stores its
    province accented as ``"Équateur"`` (``COD013``) while the CAR stores
    its unaccented as ``"Equateur"`` (``CAF002``), so a bare
    ``"Equateur"`` used to resolve arbitrarily to ``CAF002``.  An exact
    *accented* query (``"Équateur"``) is unambiguous and still resolves
    (``COD013``); pass ``country`` to pick a side for the unaccented
    form.

    Parameters
    ----------
    name:
        Region name (e.g., ``"Michigan"``, ``"Anhui"``, ``"Catalunya"``,
        or an unaccented variant like ``"Michoacan"``).
    country:
        Optional country name or ISO alpha-3 code to disambiguate when
        the name matches regions in multiple countries (e.g., ``"Georgia"``
        exists in both the USA and as a country).

    Returns
    -------
    The ADM1 code (e.g., ``"USA023"``), or ``None`` if not found.

    Examples
    --------
    >>> resolve_adm1_code("Michigan")
    'USA023'

    >>> resolve_adm1_code("Anhui")
    'CHN001'

    >>> resolve_adm1_code("Georgia", country="United States")
    'USA011'

    >>> resolve_adm1_code("Michoacan", country="MEX")   # PR #45 fallback
    'MEX016'

    >>> resolve_adm1_code("Équateur")                   # exact accented
    'COD013'

    >>> resolve_adm1_code("Equateur") is None           # ambiguous, no hint
    True

    >>> resolve_adm1_code("Equateur", country="COD")    # hint disambiguates
    'COD013'
    """
    if not name or not name.strip():
        return None

    name_lower = name.strip().lower()

    # Skip if the name resolves as a country (e.g. "Ethiopia", "China")
    # UNLESS a country filter is provided — that signals the caller knows
    # this is a subnational region (e.g. "Georgia" in "USA").
    if country is None and resolve_country_code(name_lower):
        return None

    index = _get_adm1_name_index()

    # Resolve country filter if provided.  Honor a 3-letter ISO code
    # present in the data directly (see ``_resolve_country_filter``), so
    # hints such as ``"COD"`` work; fall back to ``resolve_country_code``
    # for names.
    country_iso: str | None = None
    if country is not None:
        country_iso = _resolve_country_filter(country)

    # --- Cross-country folded-collision guard (Équateur/Equateur) ---
    # When no usable country hint is available and the query is an
    # unaccented form whose folded spelling is shared across countries,
    # it cannot identify a single region.  Refuse to guess rather than
    # return whichever country happens to store its name without the
    # accent (see ``_is_cross_country_folded_collision``).  A resolved
    # country hint routes past this guard and is disambiguated by the
    # strategies below; an exact *accented* query is unaffected because
    # folding changes it.
    if country_iso is None and _is_cross_country_folded_collision(name_lower):
        return None

    # --- Strategy 0: English-exonym alias table (PR #60) ---
    # Keys in the CSV are pre-validated lowercase NFKD-folded forms.  Try
    # the raw lower form first; if different, also try the NFKD-folded form
    # so accented callers (e.g. "Múnich") still resolve.  Country filter
    # respected via _pick_best_candidate (miss → fall through to 1-3).
    if _adm1_aliases:
        _folded_lower = _fold_diacritics(name_lower)
        for _alias_key in dict.fromkeys((name_lower, _folded_lower)):
            _alias_entries = _adm1_aliases.get(_alias_key)
            if _alias_entries:
                result = _pick_best_candidate(_alias_entries, country_iso)
                if result:
                    return result

    # --- Strategy 1: Direct index lookup ---
    candidates = index.get(name_lower)
    if candidates:
        result = _pick_best_candidate(candidates, country_iso)
        if result:
            return result

    # --- Strategy 2: Substring match (direction-aware, ambiguity-guarded) ---
    # ``_substring_match`` accepts a shorter query inside a longer canonical
    # name (Direction B) but rejects a query padded with meaningful extra words
    # (Direction A).  Collect ALL matches and resolve to distinct codes; return
    # only when exactly one region matches -- refuse to guess on ambiguity, and
    # never let the first matching name win over a conflicting second.  On 0 or
    # >1 matches, fall through so the folded strategies still get a chance.
    if len(name_lower) >= 4:
        matches: set[str] = set()
        for db_name, entries in index.items():
            if len(db_name) < 4:
                continue
            if _substring_match(name_lower, db_name):
                picked = _pick_best_candidate(entries, country_iso)
                if picked:
                    matches.add(picked)
        if len(matches) == 1:
            return next(iter(matches))

    # --- Strategy 3: Accent-folded fallback (PR #45) ---
    # Last resort: fold diacritics on both sides and retry direct
    # + substring lookup against a folded index.  Recovers the
    # very common "Michoacan / Yucatan / Nuevo Leon" pattern from
    # English LLM output (query is ASCII but the DB name carries
    # accents).  Always runs when Strategies 1/2 missed — even for
    # ASCII queries, since the *index* may still hold accented
    # forms that only the folded index can match.  Ambiguity is
    # contained by reusing ``_pick_best_candidate`` (requires a
    # single ISO or a matching country filter).
    folded_query = _fold_diacritics(name_lower)
    if folded_query:
        folded_index = _get_adm1_folded_name_index()
        # Strategy 3a: direct folded lookup.
        candidates = folded_index.get(folded_query)
        if candidates:
            result = _pick_best_candidate(candidates, country_iso)
            if result:
                return result
        # Strategy 3b: folded substring match (direction-aware + ambiguity
        # guard, same as Strategy 2).
        if len(folded_query) >= 4:
            matches = set()
            for db_name, entries in folded_index.items():
                if len(db_name) < 4:
                    continue
                if _substring_match(folded_query, db_name):
                    picked = _pick_best_candidate(entries, country_iso)
                    if picked:
                        matches.add(picked)
            if len(matches) == 1:
                return next(iter(matches))

    return None


def _pick_best_candidate(
    candidates: list[tuple[str, str]],
    country_iso: str | None,
) -> str | None:
    """Pick the best ADM1 code from a list of candidates.

    If a country filter is given, only candidates matching that country
    are considered.  Otherwise returns the first candidate.
    """
    if country_iso:
        for code, iso in candidates:
            if iso == country_iso:
                return code
        return None  # No match for the specified country

    if len(candidates) == 1:
        return candidates[0][0]

    # Multiple candidates, no country filter — return first but only
    # if all candidates are in the same country (unambiguous)
    isos = {iso for _, iso in candidates}
    if len(isos) == 1:
        return candidates[0][0]

    # Ambiguous — cannot determine without country hint
    return None
