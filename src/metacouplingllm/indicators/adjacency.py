"""Auto-fill the ``adjacent`` flag of an adjacency table from the
bundled pericoupling databases (ADM0 country level + ADM1 subnational
level).

This is an OPT-IN convenience that bridges the curated pericoupling
databases (:mod:`metacouplingllm.knowledge`) with the indicator
pipeline (:func:`metacouplingllm.indicators.classify_coupling`).  The
caller supplies only ``origin_id`` / ``destination_id`` pairs;
:func:`build_adjacency` fills ``adjacent`` with ``1`` (pericoupled /
bordering) or ``0`` (telecoupled / not bordering) by looking each pair
up in the bundled data.

The "no hardcoded geography" design principle of the indicators module
is preserved: this helper *generates a reviewable adjacency table* that
the caller still passes into ``classify_coupling``, and it never
guesses -- any pair the database cannot resolve is left as ``<NA>`` and
reported in a single ``UserWarning`` so the user can fix or fill it in.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from metacouplingllm.indicators.core import _require_pandas

if TYPE_CHECKING:  # pragma: no cover - type-only
    import pandas as pd


def _resolve_adm1_token(token: str) -> str | None:
    """Map one ADM1 cell to a World Bank ADM1 code.

    Accepts a code already (``"MEX008"``), a bare region name
    (``"Chihuahua"``), or a ``"Region, Country"`` form
    (``"Chihuahua, Mexico"``) -- the country suffix disambiguates names
    that repeat across countries.  Returns the code, or ``None`` when
    the name cannot be resolved.
    """
    from metacouplingllm.knowledge.adm1_pericoupling import (
        get_adm1_info,
        resolve_adm1_code,
    )

    text = str(token).strip()
    # Already a valid code?  (codes are unique keys in the ADM1 DB)
    if get_adm1_info(text) is not None:
        return text
    # "Region, Country" -> resolve with a country hint.
    if "," in text:
        name, country = text.rsplit(",", 1)
        return resolve_adm1_code(name.strip(), country=country.strip() or None)
    return resolve_adm1_code(text)


def build_adjacency(
    pairs: "pd.DataFrame",
    level: str = "adm0",
    *,
    origin_col: str = "origin_id",
    destination_col: str = "destination_id",
    flag_col: str = "adjacent",
    de_facto_borders: bool = True,
    coupling_standard: str = "moderate",
) -> "pd.DataFrame":
    """Fill an ``adjacent`` column (1 / 0) from the bundled pericoupling DB.

    Parameters
    ----------
    pairs:
        DataFrame with an origin and a destination column.  You can pass
        a dedicated pairs table, or your *flows* table directly (its
        ``origin_id`` / ``destination_id`` columns are reused).
    level:
        ``"adm0"`` (country) or ``"adm1"`` (subnational).  At ``adm0``
        the IDs may be country names or ISO alpha-3 codes; at ``adm1``
        they may be World Bank ADM1 codes (``"MEX008"``), region names
        (``"Chihuahua"``), or ``"Region, Country"`` (``"Chihuahua,
        Mexico"``).
    origin_col, destination_col:
        Column names to read in ``pairs``.
    flag_col:
        Name of the filled flag column in the returned table.
    de_facto_borders, coupling_standard:
        Forwarded to the bundled lookups (see
        :func:`metacouplingllm.knowledge.pericoupling.is_pericoupled`).

    Returns
    -------
    A new DataFrame with ``origin_col`` / ``destination_col`` /
    ``flag_col`` columns.  ``flag_col`` is a nullable integer:
    ``1`` = pericoupled (bordering), ``0`` = telecoupled (not
    bordering), ``<NA>`` = the database could not resolve one or both
    IDs.  Duplicate pairs are collapsed and self-pairs
    (``origin == destination``) are dropped -- intracoupling is handled
    by ``classify_coupling`` via ``focal_id``, not by adjacency.

    Notes
    -----
    Unresolved pairs are reported in a single ``UserWarning``; they are
    deliberately left as ``<NA>`` rather than assumed telecoupled, so an
    unknown ID is never silently misread as "distant".  Pass the result
    to :func:`classify_coupling`, which treats ``<NA>`` / ``0`` as
    not-adjacent.
    """
    pd = _require_pandas()

    if level not in ("adm0", "adm1"):
        raise ValueError(
            f"level must be 'adm0' or 'adm1', got {level!r}."
        )
    missing = [c for c in (origin_col, destination_col) if c not in pairs.columns]
    if missing:
        raise KeyError(
            f"pairs is missing required column(s): {missing}. "
            f"Available: {list(pairs.columns)}."
        )

    # Per level: `_canon` maps an ID to its canonical identity (ISO alpha-3
    # for adm0, WB ADM1 code for adm1) or None when unresolvable; `_lookup`
    # runs the pericoupling check on two canonical codes.
    if level == "adm0":
        from metacouplingllm.knowledge.countries import resolve_country_code
        from metacouplingllm.knowledge.pericoupling import is_pericoupled

        _canon = resolve_country_code

        def _lookup(code_a: str, code_b: str) -> bool | None:
            return is_pericoupled(
                code_a, code_b,
                de_facto_borders=de_facto_borders,
                coupling_standard=coupling_standard,
            )
    else:
        from metacouplingllm.knowledge.adm1_pericoupling import (
            is_adm1_pericoupled,
        )

        _canon = _resolve_adm1_token

        def _lookup(code_a: str, code_b: str) -> bool | None:
            return is_adm1_pericoupled(
                code_a, code_b,
                de_facto_borders=de_facto_borders,
                coupling_standard=coupling_standard,
            )

    origins: list[str] = []
    dests: list[str] = []
    flags: list[object] = []
    unresolved: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for _, row in pairs.iterrows():
        a_raw, b_raw = row[origin_col], row[destination_col]
        if pd.isna(a_raw) or pd.isna(b_raw):
            continue
        a, b = str(a_raw), str(b_raw)
        if a == b:  # literal self-pair -> intracoupling, drop
            continue
        if (a, b) in seen:  # collapse exact duplicates
            continue
        seen.add((a, b))

        # Resolve both IDs first, so the self-pair drop is *entity-level*,
        # not string-level: the same place written two ways ('MEX008' vs
        # 'Chihuahua, Mexico'; 'Mexico' vs 'MEX') resolves to one code and is
        # dropped, rather than being mislabelled telecoupled (adm1, where the
        # same-region lookup returns False) or warned as unresolved (adm0).
        canon_a, canon_b = _canon(a), _canon(b)
        if canon_a is not None and canon_b is not None and canon_a == canon_b:
            continue  # same entity -> self-pair

        if canon_a is None or canon_b is None:
            result = None  # an ID the database cannot resolve
        else:
            result = _lookup(canon_a, canon_b)

        origins.append(a)
        dests.append(b)
        if result is True:
            flags.append(1)
        elif result is False:
            flags.append(0)
        else:
            flags.append(pd.NA)
            unresolved.append((a, b))

    if unresolved:
        shown = ", ".join(f"{a!r}<->{b!r}" for a, b in unresolved[:10])
        more = "" if len(unresolved) <= 10 else f" (and {len(unresolved) - 10} more)"
        warnings.warn(
            f"build_adjacency: {len(unresolved)} pair(s) could not be "
            f"resolved in the bundled {level.upper()} database and are "
            f"left as <NA>: {shown}{more}. Review or fill these in "
            "before relying on the classification.",
            UserWarning,
            stacklevel=2,
        )

    return pd.DataFrame({
        origin_col: origins,
        destination_col: dests,
        flag_col: pd.array(flags, dtype="Int64"),
    })
