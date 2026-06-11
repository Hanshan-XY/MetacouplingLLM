"""Deterministic metacoupling indicator functions (PR #35).

Five public functions implementing the three indicator families
documented in the spec
``docs/metacoupling_codex_prompts_and_formulas_with_llm.md``:

1. ``classify_coupling`` -- in ``classify.py``; assigns I/P/T
   coupling type to each edge based on user-supplied adjacency.
2. ``compute_flow_shares`` -- Metacoupled Flow Shares (IFS / PFS /
   TFS) per focal system.
3. ``compute_mfe`` -- Metacoupled Flow Evenness (normalised
   Shannon entropy).
4. ``compute_mfci`` -- Metacoupled Flow Concentration Index
   (normalised HHI within each coupling type).
5. ``summarize_metacoupling`` -- one-shot combined indicator table.

All functions are DataFrame-in / DataFrame-out and accept
column-name kwargs so users with non-standard schemas don't have
to rename columns.

Pandas is imported lazily; users who don't install
``metacouplingllm[indicators]`` get a clear ImportError pointing
to the install command.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from metacouplingllm.indicators._math import (
    normalised_hhi,
    raw_hhi,
    shannon_entropy_normalised,
)

if TYPE_CHECKING:  # pragma: no cover - type-only
    import pandas as pd


def _require_pandas() -> "Any":
    """Lazy-import pandas with a helpful error when missing."""
    try:
        import pandas as pd

        return pd
    except ImportError as exc:  # pragma: no cover - exercised via test
        raise ImportError(
            "metacouplingllm.indicators requires pandas. Install with "
            "`pip install metacouplingllm[indicators]`."
        ) from exc


# Canonical coupling-type labels per spec §3.  Accept both single
# letter and full-word forms; normalise internally.
_COUPLING_ALIASES: dict[str, str] = {
    "i": "I",
    "intra": "I",
    "intracoupling": "I",
    "intracoupled": "I",
    "p": "P",
    "peri": "P",
    "pericoupling": "P",
    "pericoupled": "P",
    "t": "T",
    "tele": "T",
    "telecoupling": "T",
    "telecoupled": "T",
}


def _normalise_coupling_value(raw: object) -> str | None:
    """Normalise a coupling-type cell to ``"I"`` / ``"P"`` / ``"T"``
    or ``None`` if unrecognised."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    return _COUPLING_ALIASES.get(s)


# ---------------------------------------------------------------------------
# compute_flow_shares
# ---------------------------------------------------------------------------


def compute_flow_shares(
    data: "pd.DataFrame",
    system_col: str = "focal_system_id",
    coupling_col: str = "coupling_type",
    weight_col: str = "flow_value",
    group_cols: list[str] | None = None,
) -> "pd.DataFrame":
    """Compute Intracoupled / Pericoupled / Telecoupled Flow Shares.

    Per spec §6:
      IFS_i = F_iI / (F_iI + F_iP + F_iT)
      PFS_i = F_iP / (F_iI + F_iP + F_iT)
      TFS_i = F_iT / (F_iI + F_iP + F_iT)

    Parameters
    ----------
    data:
        Edge-level OR aggregated DataFrame.  At minimum needs
        ``system_col``, ``coupling_col``, ``weight_col`` columns.
    system_col, coupling_col, weight_col:
        Column names in ``data``.  Defaults match the canonical
        names from spec §14.2.
    group_cols:
        Optional extra grouping columns (e.g., ``["year"]`` for
        time-series output, or ``["year", "flow_type"]`` for the
        per-flow-type analysis described in spec §3).  One output
        row per (system, *group_cols).

    Returns
    -------
    pd.DataFrame with columns
        [system_col, *group_cols, F_I, F_P, F_T, F_total,
         IFS, PFS, TFS].

    Edge cases
    ----------
    - ``F_total == 0``: ``IFS / PFS / TFS = NaN`` plus a
      ``UserWarning`` (spec §13).
    - ``F_I == 0``: emits a ``UserWarning`` so users don't misread
      missing intracoupling data as "no intracoupling" (per scope
      decision: intracoupled flow data is required for meaningful
      indicators).
    - Unrecognised coupling-type labels are silently dropped from
      the totals but counted toward an internal validation that
      emits a one-shot warning per call.
    """
    pd = _require_pandas()

    _check_columns(data, [system_col, coupling_col, weight_col])
    if group_cols:
        _check_columns(data, group_cols)

    # Normalise coupling-type column to I / P / T.
    work = data.copy()
    work[coupling_col] = work[coupling_col].apply(_normalise_coupling_value)
    unknown_mask = work[coupling_col].isna()
    if unknown_mask.any():
        warnings.warn(
            f"compute_flow_shares: {int(unknown_mask.sum())} edge(s) had "
            "unrecognised coupling_type labels and were excluded from the "
            "totals.  Accepted: I/P/T (or intra / peri / tele).",
            UserWarning,
            stacklevel=2,
        )
        work = work.loc[~unknown_mask].copy()

    # Coerce weights to float; non-numeric becomes NaN -> dropped.
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    invalid_w = work[weight_col].isna() | (work[weight_col] < 0)
    if invalid_w.any():
        warnings.warn(
            f"compute_flow_shares: {int(invalid_w.sum())} row(s) had "
            "non-numeric or negative flow_value and were excluded.",
            UserWarning,
            stacklevel=2,
        )
        work = work.loc[~invalid_w].copy()

    group_keys = [system_col, *(group_cols or [])]
    # Pivot: one row per group, columns I/P/T.
    pivot = (
        work.groupby([*group_keys, coupling_col])[weight_col]
        .sum()
        .unstack(coupling_col)
        .reindex(columns=["I", "P", "T"])
        .fillna(0.0)
        .reset_index()
    )
    pivot = pivot.rename(columns={"I": "F_I", "P": "F_P", "T": "F_T"})
    pivot["F_total"] = pivot["F_I"] + pivot["F_P"] + pivot["F_T"]

    zero_mask = pivot["F_total"] == 0
    if zero_mask.any():
        warnings.warn(
            f"compute_flow_shares: {int(zero_mask.sum())} group(s) have "
            "zero total flow; IFS / PFS / TFS are undefined and returned "
            "as NaN.",
            UserWarning,
            stacklevel=2,
        )

    # NaN-safe division.
    with np.errstate(invalid="ignore", divide="ignore"):
        pivot["IFS"] = np.where(
            pivot["F_total"] > 0, pivot["F_I"] / pivot["F_total"], np.nan
        )
        pivot["PFS"] = np.where(
            pivot["F_total"] > 0, pivot["F_P"] / pivot["F_total"], np.nan
        )
        pivot["TFS"] = np.where(
            pivot["F_total"] > 0, pivot["F_T"] / pivot["F_total"], np.nan
        )

    # User-decision warning: intracoupling data is required for
    # meaningful indicators (per PR #35 scope discussion).  Don't
    # double-warn if total is also zero.
    intra_zero_mask = (
        (pivot["F_I"] == 0) & (pivot["F_total"] > 0)
    )
    if intra_zero_mask.any():
        warnings.warn(
            f"compute_flow_shares: {int(intra_zero_mask.sum())} group(s) "
            "have F_I = 0 but non-zero peri- or telecoupled flow.  "
            "Confirm that intracoupling absence reflects the system, "
            "not missing data -- IFS = 0 is shown but may be a data "
            "artifact.",
            UserWarning,
            stacklevel=2,
        )

    return pivot


# ---------------------------------------------------------------------------
# compute_mfe
# ---------------------------------------------------------------------------


def compute_mfe(
    data: "pd.DataFrame",
    share_cols: tuple[str, str, str] = ("IFS", "PFS", "TFS"),
) -> "pd.DataFrame":
    """Compute Metacoupled Flow Evenness (MFE) for each row.

    Per spec §7: normalised Shannon entropy over (IFS, PFS, TFS).

    Adds an ``MFE`` column in-place-style (returns a copy).
    Rows whose three shares are all NaN or all zero get
    ``MFE = NaN``.  Uses the ``0 * ln(0) = 0`` convention.
    """
    pd = _require_pandas()
    _check_columns(data, list(share_cols))

    result = data.copy()
    mfe_values: list[float] = []
    for _, row in result.iterrows():
        shares = np.array(
            [row[share_cols[0]], row[share_cols[1]], row[share_cols[2]]],
            dtype=float,
        )
        if np.isnan(shares).all():
            mfe_values.append(float("nan"))
        else:
            # Treat NaNs as 0 for the entropy calculation; entropy
            # of a degenerate distribution returns NaN.
            shares = np.nan_to_num(shares, nan=0.0)
            mfe_values.append(shannon_entropy_normalised(shares))
    result["MFE"] = mfe_values
    return result


# ---------------------------------------------------------------------------
# compute_mfci
# ---------------------------------------------------------------------------


def compute_mfci(
    data: "pd.DataFrame",
    system_col: str = "focal_system_id",
    partner_col: str = "destination_id",
    coupling_col: str = "coupling_type",
    weight_col: str = "flow_value",
    group_cols: list[str] | None = None,
) -> "pd.DataFrame":
    """Compute Metacoupled Flow Concentration Index per coupling type.

    Per spec §8: normalised HHI computed from partner-level shares
    within each coupling type.

    Returns
    -------
    pd.DataFrame with one row per (system, *group_cols, coupling_type)
    carrying [n_partners, HHI, MFCI, effective_n_partners].
    Coupling-type values are ``"I"`` / ``"P"`` / ``"T"`` (always
    all three rows per group, even when ``F_ic = 0``, with numeric
    columns set to ``NaN`` per spec §8 edge cases).

    Edge cases
    ----------
    - ``F_ic == 0``: ``MFCI = NaN`` + ``UserWarning`` (spec §8).
    - ``n_ic == 1``: ``MFCI = 1`` by convention + ``UserWarning``
      (spec §8 + §14.6 -- intracoupling self-loop case).
    """
    pd = _require_pandas()
    _check_columns(
        data, [system_col, partner_col, coupling_col, weight_col]
    )
    if group_cols:
        _check_columns(data, group_cols)

    work = data.copy()
    work[coupling_col] = work[coupling_col].apply(_normalise_coupling_value)
    work = work.loc[work[coupling_col].notna()].copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work = work.loc[
        work[weight_col].notna() & (work[weight_col] >= 0)
    ].copy()

    group_keys = [system_col, *(group_cols or [])]
    rows: list[dict[str, Any]] = []

    # Iterate over every (group, coupling_type) we observe AND emit
    # placeholder NaN rows for the three coupling types that aren't
    # represented in a given group -- so the output schema is
    # predictable.
    for group_vals, group_df in work.groupby(group_keys, dropna=False):
        # group_vals is a tuple when group_keys has multiple cols,
        # else a scalar.
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        for cat in ("I", "P", "T"):
            cat_df = group_df.loc[group_df[coupling_col] == cat]
            partner_totals = (
                cat_df.groupby(partner_col)[weight_col]
                .sum()
                .loc[lambda s: s > 0]
            )
            f_ic = float(partner_totals.sum())
            n_ic = int(partner_totals.size)
            row = {k: v for k, v in zip(group_keys, group_vals)}
            row[coupling_col] = cat
            if f_ic == 0:
                row["n_partners"] = 0
                row["HHI"] = float("nan")
                row["MFCI"] = float("nan")
                row["effective_n_partners"] = float("nan")
                warnings.warn(
                    f"compute_mfci: no {cat}-coupling flow for group "
                    f"{group_vals}; MFCI = NaN.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                shares = partner_totals.to_numpy()
                hhi = raw_hhi(shares)
                mfci_val = normalised_hhi(shares)
                row["n_partners"] = n_ic
                row["HHI"] = hhi
                row["MFCI"] = mfci_val
                row["effective_n_partners"] = (
                    1.0 / hhi if hhi and not np.isnan(hhi) else float("nan")
                )
            rows.append(row)

    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# summarize_metacoupling
# ---------------------------------------------------------------------------


def summarize_metacoupling(
    data: "pd.DataFrame",
    system_col: str = "focal_system_id",
    partner_col: str = "destination_id",
    coupling_col: str = "coupling_type",
    weight_col: str = "flow_value",
    group_cols: list[str] | None = None,
) -> "pd.DataFrame":
    """One-shot combined indicator table per spec §12.5.

    Combines ``compute_flow_shares``, ``compute_mfe``, and
    ``compute_mfci`` outputs into a single wide table.

    Returns
    -------
    pd.DataFrame with columns
        [system_col, *group_cols, F_I, F_P, F_T, F_total,
         IFS, PFS, TFS, MFE,
         IFCI, PFCI, TFCI,
         n_I, n_P, n_T,
         ENP_I, ENP_P, ENP_T]
    """
    pd = _require_pandas()

    shares_df = compute_flow_shares(
        data,
        system_col=system_col,
        coupling_col=coupling_col,
        weight_col=weight_col,
        group_cols=group_cols,
    )
    shares_with_mfe = compute_mfe(shares_df)

    mfci_df = compute_mfci(
        data,
        system_col=system_col,
        partner_col=partner_col,
        coupling_col=coupling_col,
        weight_col=weight_col,
        group_cols=group_cols,
    )

    # Pivot MFCI long->wide: one column per (coupling type x metric).
    group_keys = [system_col, *(group_cols or [])]
    mfci_wide = mfci_df.pivot_table(
        index=group_keys,
        columns=coupling_col,
        values=["n_partners", "MFCI", "effective_n_partners"],
        aggfunc="first",
    )
    # Flatten the MultiIndex columns to IFCI / PFCI / TFCI / n_I / ...
    rename_map = {
        ("MFCI", "I"): "IFCI",
        ("MFCI", "P"): "PFCI",
        ("MFCI", "T"): "TFCI",
        ("n_partners", "I"): "n_I",
        ("n_partners", "P"): "n_P",
        ("n_partners", "T"): "n_T",
        ("effective_n_partners", "I"): "ENP_I",
        ("effective_n_partners", "P"): "ENP_P",
        ("effective_n_partners", "T"): "ENP_T",
    }
    mfci_wide.columns = [
        rename_map.get(col, "_".join(str(c) for c in col))
        for col in mfci_wide.columns
    ]
    mfci_wide = mfci_wide.reset_index()

    merged = shares_with_mfe.merge(mfci_wide, on=group_keys, how="left")

    # Stable column order matching spec §12.5.
    ordered = [
        *group_keys,
        "F_I", "F_P", "F_T", "F_total",
        "IFS", "PFS", "TFS", "MFE",
        "IFCI", "PFCI", "TFCI",
        "n_I", "n_P", "n_T",
        "ENP_I", "ENP_P", "ENP_T",
    ]
    # Keep only columns that actually exist (e.g., merge fills with
    # NaN if some coupling type was absent entirely).
    ordered = [c for c in ordered if c in merged.columns]
    return merged[ordered]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_columns(data: "pd.DataFrame", required: list[str]) -> None:
    """Raise KeyError listing any missing columns up front."""
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(data.columns)}."
        )
