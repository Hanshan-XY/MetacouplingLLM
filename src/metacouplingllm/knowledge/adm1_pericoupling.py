"""
ADM1 (subnational) pericoupling database.

Loads a curated CSV edge list of first-level administrative divisions (ADM1)
to determine which subnational regions share a border (pericoupled).  The
database uses World Bank ADM1 codes (e.g., ``"MEX001"``, ``"USA035"``).

The edge list contains **8,290 border pairs** covering **3,366 unique ADM1
regions** across **195 countries**, including both within-country and
cross-country borders.

Source
------
``pericoupled_adm1_edge_list.csv`` (research dataset using World Bank ADM1
codes and ISO 3166-1 alpha-3 country codes).
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


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _locate_csv() -> Path | None:
    """Find the ADM1 pericoupling CSV data file."""
    try:
        data_pkg = resources.files("metacoupling") / "data"
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


def _ensure_loaded() -> None:
    """Lazily load and cache all ADM1 data."""
    global _adm1_pairs, _adm1_neighbors, _adm1_country, _adm1_metadata
    if _adm1_pairs is None:
        _adm1_pairs, _adm1_neighbors, _adm1_country, _adm1_metadata = (
            _load_adm1_data()
        )


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def lookup_adm1_pericoupling(
    adm1_a: str,
    adm1_b: str,
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

    if pair in _adm1_pairs:
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


def get_adm1_neighbors(adm1_code: str) -> set[str]:
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
    assert _adm1_neighbors is not None
    return set(_adm1_neighbors.get(adm1_code.strip(), set()))


def get_cross_border_neighbors(adm1_code: str) -> set[str]:
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
    assert _adm1_neighbors is not None
    assert _adm1_country is not None

    adm1_code = adm1_code.strip()
    if adm1_code not in _adm1_country:
        return set()

    focal_iso = _adm1_country[adm1_code]
    result: set[str] = set()
    for neighbor in _adm1_neighbors.get(adm1_code, set()):
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


def is_adm1_pericoupled(adm1_a: str, adm1_b: str) -> bool | None:
    """Convenience function for quick ADM1 pericoupling checks.

    Returns
    -------
    ``True`` if the two regions share a border, ``False`` if they do
    not, ``None`` if one or both codes are not in the database.
    """
    result = lookup_adm1_pericoupling(adm1_a, adm1_b)
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


def resolve_adm1_code(
    name: str,
    country: str | None = None,
) -> str | None:
    """Resolve an ADM1 region name to its World Bank ADM1 code.

    Parameters
    ----------
    name:
        Region name (e.g., ``"Michigan"``, ``"Anhui"``, ``"Catalunya"``).
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

    # Resolve country filter if provided
    country_iso: str | None = None
    if country is not None:
        country_iso = resolve_country_code(country)

    # --- Strategy 1: Direct index lookup ---
    candidates = index.get(name_lower)
    if candidates:
        result = _pick_best_candidate(candidates, country_iso)
        if result:
            return result

    def _contains_phrase(haystack: str, needle: str) -> bool:
        pattern = rf"(?<![a-z]){re.escape(needle)}(?![a-z])"
        return re.search(pattern, haystack) is not None

    # --- Strategy 2: Substring match ---
    # Only if name is at least 4 chars and the matching DB name is also
    # at least 4 chars, to avoid false positives from short fragments.
    if len(name_lower) >= 4:
        for db_name, entries in index.items():
            if len(db_name) < 4:
                continue
            if (
                _contains_phrase(db_name, name_lower)
                or _contains_phrase(name_lower, db_name)
            ):
                result = _pick_best_candidate(entries, country_iso)
                if result:
                    return result

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
