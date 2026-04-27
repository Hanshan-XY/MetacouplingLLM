"""
Pericoupling country-pair database.

Loads a curated CSV database of country pairs to determine which are
pericoupled (geographically adjacent, value=1) vs telecoupled (distant,
value=0).  The database uses ISO 3166-1 alpha-3 codes.

At runtime the full ``PeriTelecoupling_clean.csv`` is loaded if present;
otherwise the small ``PeriTelecoupling_subset.csv`` is used as a fallback
(useful for testing).

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


def _get_pairs() -> frozenset[frozenset[str]]:
    """Lazily load and cache the pericoupled pairs."""
    global _pericoupled_pairs
    if _pericoupled_pairs is None:
        _pericoupled_pairs = _load_pairs()
    return _pericoupled_pairs


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def lookup_pericoupling(
    country_a: str,
    country_b: str,
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
    pairs = _get_pairs()

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


def get_pericoupled_neighbors(country: str) -> set[str]:
    """Return the set of ISO alpha-3 codes pericoupled with a country.

    Parameters
    ----------
    country:
        Country name or ISO alpha-3 code.

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
    for pair in _get_pairs():
        if code in pair:
            others = pair - frozenset({code})
            if others:
                neighbors.add(next(iter(others)))
    return neighbors


def is_pericoupled(country_a: str, country_b: str) -> bool | None:
    """Convenience function for quick pericoupling checks.

    Returns
    -------
    ``True`` if pericoupled, ``False`` if telecoupled, ``None`` if one or
    both countries could not be resolved.
    """
    result = lookup_pericoupling(country_a, country_b)
    if result.pair_type == PairCouplingType.PERICOUPLED:
        return True
    if result.pair_type == PairCouplingType.TELECOUPLED:
        return False
    return None
