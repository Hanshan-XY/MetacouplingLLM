"""
Pericoupling country-pair database.

Loads a curated CSV database of country pairs to determine which are
pericoupled (geographically adjacent, value=1) vs telecoupled (distant,
value=0).  The database uses ISO 3166-1 alpha-3 codes.

At runtime the full ``PeriTelecoupling_clean.csv`` is loaded if present;
otherwise the small ``PeriTelecoupling_subset.csv`` is used as a fallback
(useful for testing).

With ``maritime_adjacency=True`` the lookups also count country pairs whose
waters meet along a maritime boundary (``maritime_adjacency_pairs.csv``,
built from the Marine Regions Maritime Boundaries Geodatabase by
``scripts/build_maritime_adjacency.py``; see ``data/PROVENANCE.md``).

Source
------
PeriTelecoupling_clean.csv (research dataset using ISO 3166-1 alpha-3).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path

from metacouplingllm.knowledge.countries import resolve_country_code


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class PairCouplingType(str, Enum):
    """Result of a pericoupling database lookup."""

    PERICOUPLED = "pericoupled"
    TELECOUPLED = "telecoupled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PericouplingResult:
    """Result of looking up a country pair in the pericoupling database.

    Attributes
    ----------
    pair_type:
        Whether the pair is pericoupled, telecoupled, or unknown.
    sending_code:
        Resolved ISO alpha-3 code for the sending country, or ``None``.
    receiving_code:
        Resolved ISO alpha-3 code for the receiving country, or ``None``.
    confidence:
        ``"database"`` if looked up from the database, ``"unresolved"`` if
        one or both country codes could not be resolved, ``"same_country"``
        if both names resolved to the same country.
    """

    pair_type: PairCouplingType
    sending_code: str | None
    receiving_code: str | None
    confidence: str = "database"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

_pericoupled_pairs: frozenset[frozenset[str]] | None = None
_disputed_overlay_adm0: frozenset[frozenset[str]] | None = None
_water_all_adm0: frozenset[frozenset[str]] | None = None
_water_nobridge_adm0: frozenset[frozenset[str]] | None = None
_maritime_pairs: frozenset[frozenset[str]] | None = None
_maritime_disputed_only: frozenset[frozenset[str]] | None = None
_pairs_view_cache: dict[tuple[bool, str, bool], frozenset[frozenset[str]]] = {}

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


def _locate_csv() -> Path | None:
    """Find the pericoupling CSV data file.

    Tries the full dataset first, then falls back to the subset.
    """
    try:
        data_pkg = resources.files("metacouplingllm") / "data"
    except (TypeError, ModuleNotFoundError):
        return None

    for name in ("PeriTelecoupling_clean.csv", "PeriTelecoupling_subset.csv"):
        candidate = data_pkg / name
        # resources.files may return a Traversable; convert to Path if possible
        try:
            path = Path(str(candidate))
            if path.is_file():
                return path
        except Exception:
            continue
    return None


def _load_disputed_overlay(level: str) -> frozenset[frozenset[str]]:
    """Load the de-facto disputed-territory overlay pairs for a level.

    Reads ``disputed_overlay_pairs.csv`` (shipped beside the data) and returns
    the set of ISO/ADM1 code pairs whose adjacency exists only under the
    de-facto view.  These are subtracted when ``de_facto_borders=False``.
    Returns an empty set if the manifest is absent (older data) — so both
    views are then identical.
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
            if row.get("level", "").strip() == level:
                overlay.add(
                    frozenset({row["code_a"].strip(), row["code_b"].strip()})
                )
    return frozenset(overlay)


def _load_pairs() -> frozenset[frozenset[str]]:
    """Load pericoupled pairs from the CSV data file.

    Returns a frozenset of frozensets — each inner frozenset is a pair of
    ISO alpha-3 codes where Intracoupling == 1 (pericoupled).  Since the
    CSV contains both directions (A→B and B→A), this naturally deduplicates
    via frozenset.
    """
    csv_path = _locate_csv()
    if csv_path is None:
        return frozenset()

    pairs: set[frozenset[str]] = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("Intracoupling", "").strip() == "1":
                sending = row["Sending"].strip()
                receiving = row["Receiving"].strip()
                pairs.add(frozenset({sending, receiving}))
    return frozenset(pairs)


def _load_water_separated_adm0() -> tuple[
    frozenset[frozenset[str]], frozenset[frozenset[str]]
]:
    """Load water-separated ADM0 (country) pairs for ``coupling_standard``.

    Reads ``water_separated_pairs.csv`` (rows with ``level == "adm0"``): ISO
    country pairs whose entire shared border is water (rolled up from the ADM1
    crossings).  Returns ``(all_water, no_bridge)``; under ``moderate`` the
    ``no_bridge`` set is subtracted from the base, under ``stringent`` the full
    ``all_water`` set is.  Empty sets if the manifest is absent (older data).
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
            if row.get("level", "").strip() != "adm0":
                continue
            pair = frozenset({row["code_a"].strip(), row["code_b"].strip()})
            all_water.add(pair)
            if row.get("has_bridge", "").strip() != "True":
                no_bridge.add(pair)
    return frozenset(all_water), frozenset(no_bridge)


def _water_dropped_adm0(coupling_standard: str) -> frozenset[frozenset[str]]:
    """Water-only country pairs to subtract from the base under a standard."""
    if coupling_standard == "stringent":
        return _water_all_adm0 or frozenset()
    if coupling_standard == "moderate":
        return _water_nobridge_adm0 or frozenset()
    return frozenset()


def _load_maritime_pairs() -> tuple[
    frozenset[frozenset[str]], frozenset[frozenset[str]]
]:
    """Load the country pairs that share a maritime boundary.

    Reads ``maritime_adjacency_pairs.csv``: ISO country pairs whose waters
    meet along a boundary recorded in the Marine Regions Maritime Boundaries
    Geodatabase.  Returns ``(pairs, disputed_only)``; the second set holds the
    pairs that meet only through the waters of a disputed tract, which the
    strict view (``de_facto_borders=False``) leaves out.  Empty sets if the
    file is absent (older data).
    """
    try:
        data_pkg = resources.files("metacouplingllm") / "data"
        path = Path(str(data_pkg / "maritime_adjacency_pairs.csv"))
    except (TypeError, ModuleNotFoundError):
        return frozenset(), frozenset()
    if not path.is_file():
        return frozenset(), frozenset()
    pairs: set[frozenset[str]] = set()
    disputed: set[frozenset[str]] = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            pair = frozenset({row["code_a"].strip(), row["code_b"].strip()})
            pairs.add(pair)
            if row.get("disputed_only", "").strip() == "True":
                disputed.add(pair)
    return frozenset(pairs), frozenset(disputed)


def _get_pairs(
    de_facto_borders: bool = True,
    coupling_standard: str = "lenient",
    maritime_adjacency: bool = False,
) -> frozenset[frozenset[str]]:
    """Lazily load and cache the pericoupled pairs for a given view.

    Parameters
    ----------
    de_facto_borders:
        ``True`` (default) returns the de-facto view (disputed land folded into
        its de-facto administering country, e.g. China–Pakistan, Israel–Syria);
        ``False`` subtracts the disputed-territory overlay.  See
        ``data/disputed_overlay_pairs.csv`` and ``data/PROVENANCE.md``.
    coupling_standard:
        ``"lenient"`` (default) keeps every border, land or water;
        ``"moderate"`` drops water-only country pairs with no open fixed
        crossing; ``"stringent"`` drops every water-only pair.  See
        ``data/water_separated_pairs.csv``.
    maritime_adjacency:
        ``False`` (default) counts the borders between the countries'
        territories, land, river or lake; ``True`` also counts pairs whose
        waters meet at sea, along a boundary in the Marine Regions Maritime
        Boundaries Geodatabase (a treaty, a median line, a joint regime, an
        unsettled line).  A pair that meets only through the waters of a
        disputed tract follows ``de_facto_borders``.  See
        ``data/maritime_adjacency_pairs.csv``.
    """
    global _pericoupled_pairs, _disputed_overlay_adm0
    global _water_all_adm0, _water_nobridge_adm0
    global _maritime_pairs, _maritime_disputed_only
    coupling_standard = _check_standard(coupling_standard)
    maritime_adjacency = bool(maritime_adjacency)
    key = (de_facto_borders, coupling_standard, maritime_adjacency)
    if key in _pairs_view_cache:
        return _pairs_view_cache[key]
    if _pericoupled_pairs is None:
        _pericoupled_pairs = _load_pairs()
    if _disputed_overlay_adm0 is None:
        _disputed_overlay_adm0 = _load_disputed_overlay("adm0")
    if _water_all_adm0 is None:
        _water_all_adm0, _water_nobridge_adm0 = _load_water_separated_adm0()
    pairs = set(_pericoupled_pairs)
    if not de_facto_borders:
        pairs -= _disputed_overlay_adm0
    pairs -= _water_dropped_adm0(coupling_standard)
    if maritime_adjacency:
        if _maritime_pairs is None:
            _maritime_pairs, _maritime_disputed_only = _load_maritime_pairs()
        sea = set(_maritime_pairs)
        if not de_facto_borders:
            sea -= _maritime_disputed_only or frozenset()
        pairs |= sea
    view = frozenset(pairs)
    _pairs_view_cache[key] = view
    return view


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def lookup_pericoupling(
    country_a: str,
    country_b: str,
    de_facto_borders: bool = True,
    coupling_standard: str = "lenient",
    maritime_adjacency: bool = False,
) -> PericouplingResult:
    """Look up whether two countries are pericoupled or telecoupled.

    Parameters
    ----------
    country_a:
        Name or ISO code of the first country (e.g., ``"Mexico"`` or
        ``"MEX"``).
    country_b:
        Name or ISO code of the second country (e.g., ``"United States"``
        or ``"USA"``).
    de_facto_borders, coupling_standard, maritime_adjacency:
        Adjacency settings (see :func:`_get_pairs`).

    Returns
    -------
    A :class:`PericouplingResult` with the lookup outcome.

    Examples
    --------
    >>> result = lookup_pericoupling("Mexico", "United States")
    >>> result.pair_type
    <PairCouplingType.PERICOUPLED: 'pericoupled'>

    >>> result = lookup_pericoupling("Brazil", "China")
    >>> result.pair_type
    <PairCouplingType.TELECOUPLED: 'telecoupled'>
    """
    code_a = resolve_country_code(country_a)
    code_b = resolve_country_code(country_b)

    if code_a is None or code_b is None:
        return PericouplingResult(
            pair_type=PairCouplingType.UNKNOWN,
            sending_code=code_a,
            receiving_code=code_b,
            confidence="unresolved",
        )

    if code_a == code_b:
        return PericouplingResult(
            pair_type=PairCouplingType.UNKNOWN,
            sending_code=code_a,
            receiving_code=code_b,
            confidence="same_country",
        )

    pair = frozenset({code_a, code_b})
    pairs = _get_pairs(de_facto_borders, coupling_standard, maritime_adjacency)

    if pair in pairs:
        return PericouplingResult(
            pair_type=PairCouplingType.PERICOUPLED,
            sending_code=code_a,
            receiving_code=code_b,
            confidence="database",
        )

    return PericouplingResult(
        pair_type=PairCouplingType.TELECOUPLED,
        sending_code=code_a,
        receiving_code=code_b,
        confidence="database",
    )


def get_pericoupled_neighbors(
    country: str,
    de_facto_borders: bool = True,
    coupling_standard: str = "lenient",
    maritime_adjacency: bool = False,
) -> set[str]:
    """Return the set of ISO alpha-3 codes pericoupled with a country.

    Parameters
    ----------
    country:
        Country name or ISO alpha-3 code.
    de_facto_borders, coupling_standard, maritime_adjacency:
        Adjacency settings (see :func:`_get_pairs`).

    Returns
    -------
    A set of ISO alpha-3 codes for all countries that are pericoupled
    (geographically adjacent) with the given country.  Empty set if the
    country cannot be resolved or has no pericoupled neighbors.

    Examples
    --------
    >>> sorted(get_pericoupled_neighbors("Mexico"))
    ['BLZ', 'GTM', 'USA']
    """
    code = resolve_country_code(country)
    if code is None:
        return set()
    neighbors: set[str] = set()
    for pair in _get_pairs(de_facto_borders, coupling_standard, maritime_adjacency):
        if code in pair:
            others = pair - frozenset({code})
            if others:
                neighbors.add(next(iter(others)))
    return neighbors


def is_pericoupled(
    country_a: str,
    country_b: str,
    de_facto_borders: bool = True,
    coupling_standard: str = "lenient",
    maritime_adjacency: bool = False,
) -> bool | None:
    """Convenience function for quick pericoupling checks.

    Returns
    -------
    ``True`` if pericoupled, ``False`` if telecoupled, ``None`` if one or
    both countries could not be resolved.

    See :func:`_get_pairs` for the adjacency settings: ``de_facto_borders``
    (default ``True`` treats disputed land as part of its de-facto
    administrator), ``coupling_standard`` and ``maritime_adjacency``.
    """
    result = lookup_pericoupling(
        country_a, country_b, de_facto_borders, coupling_standard,
        maritime_adjacency,
    )
    if result.pair_type == PairCouplingType.PERICOUPLED:
        return True
    if result.pair_type == PairCouplingType.TELECOUPLED:
        return False
    return None
