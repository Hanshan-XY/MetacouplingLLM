"""Classify each flow edge as intracoupled / pericoupled /
telecoupled.

Per spec §4 and §12.1:
- I (intracoupling): flow occurs within the focal system
- P (pericoupling): focal connects to an adjacent system
- T (telecoupling): focal connects to a non-adjacent / distant system

This implementation uses a USER-SUPPLIED adjacency table (no
hardcoded geography per spec §4).  Distance-threshold classification
is documented as out of scope for PR #35; users wanting that today
can pre-compute their own adjacency from a distance matrix.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from metacouplingllm.indicators.core import _require_pandas

if TYPE_CHECKING:  # pragma: no cover - type-only
    import pandas as pd


def classify_coupling(
    edges: "pd.DataFrame",
    focal_id: str | list[str],
    adjacency: "pd.DataFrame | None" = None,
    origin_col: str = "origin_id",
    destination_col: str = "destination_id",
    coupling_col_out: str = "coupling_type",
    adjacency_origin_col: str = "origin_id",
    adjacency_destination_col: str = "destination_id",
    adjacency_flag_col: str = "adjacent",
) -> "pd.DataFrame":
    """Add a coupling-type column to an edge table.

    Parameters
    ----------
    edges:
        Edge-level DataFrame with origin / destination columns.
    focal_id:
        Focal system identifier.  Either a single string (e.g.,
        ``"Brazil"``) or a list of internal subunit IDs (e.g.,
        ``["Mato Grosso", "Goias", ...]``) for a sub-national
        intracoupling analysis.
    adjacency:
        DataFrame with at least 3 columns: origin id, destination
        id, and an ``adjacent`` flag (0/1 or bool).  When ``None``,
        only self-loop intracoupling edges can be classified;
        every cross-system edge raises ``ValueError``.
    origin_col, destination_col:
        Column names in ``edges``.
    coupling_col_out:
        Name of the new column added to the returned DataFrame.
        Default ``"coupling_type"``.
    adjacency_origin_col, adjacency_destination_col, adjacency_flag_col:
        Column names in the adjacency table.

    Returns
    -------
    Copy of ``edges`` with a new ``coupling_type`` column whose
    values are ``"I"`` / ``"P"`` / ``"T"`` / ``NaN``.  ``NaN``
    is used when the adjacency table doesn't cover that edge --
    a ``UserWarning`` is emitted in that case.

    Notes
    -----
    Adjacency is treated as **symmetric**: if ``(A, B, adjacent=1)``
    is in the table, both ``A -> B`` and ``B -> A`` get classified
    as P.  Users who need directional adjacency should pre-process
    their adjacency table accordingly.
    """
    pd = _require_pandas()

    required_edge_cols = [origin_col, destination_col]
    missing = [c for c in required_edge_cols if c not in edges.columns]
    if missing:
        raise KeyError(
            f"edges is missing required column(s): {missing}. "
            f"Available: {list(edges.columns)}."
        )

    focal_ids: set[str] = (
        {focal_id} if isinstance(focal_id, str) else set(focal_id)
    )

    # Build a fast O(1) lookup for adjacency: {(origin, dest)} for
    # all rows where the flag is truthy.  Treat as symmetric.
    adjacency_set: set[tuple[str, str]] = set()
    if adjacency is not None:
        adj_required = [
            adjacency_origin_col,
            adjacency_destination_col,
            adjacency_flag_col,
        ]
        adj_missing = [c for c in adj_required if c not in adjacency.columns]
        if adj_missing:
            raise KeyError(
                f"adjacency is missing required column(s): {adj_missing}. "
                f"Available: {list(adjacency.columns)}."
            )
        for _, arow in adjacency.iterrows():
            if not bool(arow[adjacency_flag_col]):
                continue
            a = str(arow[adjacency_origin_col])
            b = str(arow[adjacency_destination_col])
            adjacency_set.add((a, b))
            adjacency_set.add((b, a))

    out = edges.copy()
    coupling_values: list[Any] = []
    unclassifiable_count = 0
    for _, row in out.iterrows():
        origin = str(row[origin_col])
        dest = str(row[destination_col])
        coupling_values.append(
            _classify_one(
                origin, dest, focal_ids, adjacency_set,
                adjacency_provided=adjacency is not None,
            )
        )
        if coupling_values[-1] is None:
            unclassifiable_count += 1

    if unclassifiable_count > 0:
        if adjacency is None:
            raise ValueError(
                f"classify_coupling: {unclassifiable_count} cross-system "
                "edge(s) cannot be classified because no adjacency table "
                "was provided.  Pass an adjacency DataFrame with columns "
                f"[{adjacency_origin_col!r}, {adjacency_destination_col!r}, "
                f"{adjacency_flag_col!r}]."
            )
        warnings.warn(
            f"classify_coupling: {unclassifiable_count} edge(s) had "
            "no adjacency information and are returned as NaN.",
            UserWarning,
            stacklevel=2,
        )

    out[coupling_col_out] = coupling_values
    return out


def _classify_one(
    origin: str,
    destination: str,
    focal_ids: set[str],
    adjacency_set: set[tuple[str, str]],
    adjacency_provided: bool,
) -> str | None:
    """Single-edge classifier.  Returns ``"I"`` / ``"P"`` / ``"T"`` /
    ``None`` (unclassifiable)."""
    in_focal_origin = origin in focal_ids
    in_focal_dest = destination in focal_ids
    # Intracoupling: edge stays within the focal system (or set of
    # subunits).
    if in_focal_origin and in_focal_dest:
        return "I"
    # If neither endpoint is the focal system, we can still ask:
    # is this even an edge we should classify?  The spec frames
    # classification as "for focal system i", so non-focal edges
    # are not the package's job -- return None.
    if not (in_focal_origin or in_focal_dest):
        return None
    # Cross-system edge involving the focal system: check adjacency.
    if not adjacency_provided:
        # Caller will turn the None into ValueError aggregated.
        return None
    pair = (origin, destination)
    if pair in adjacency_set:
        return "P"
    return "T"
