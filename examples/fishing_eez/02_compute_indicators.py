#!/usr/bin/env python
"""
02_compute_indicators.py  --  Compute per-EEZ metacoupling indicators by calling
the metacouplingllm package.

INPUT  : data/fishing_edge_table.csv  (pinned, produced by 00_build_edge_table.py)
OUTPUT : outputs/indicators_by_eez.csv

This step is PACKAGE functionality.  We do NOT re-implement any indicator -- we
hand the edge table to `summarize_metacoupling`, which returns one row per focal
EEZ with the flow totals, the Metacoupled Flow Shares, the Metacoupled Flow
Evenness (MFE), the Metacoupled Flow Concentration Index per coupling type
(MFCI), and the Equivalent Number of Partners (ENP):

    F_I, F_P, F_T, F_total,
    IFS, PFS, TFS, MFE,
    IFCI, PFCI, TFCI,
    ENP_I, ENP_P, ENP_T
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# >>> PACKAGE FUNCTIONALITY <<<
from metacouplingllm.indicators import summarize_metacoupling

HERE = Path(__file__).resolve().parent
EDGES = HERE / "data" / "fishing_edge_table.csv"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    edges = pd.read_csv(EDGES)
    print(f"Loaded {len(edges)} edges across {edges['focal_system_id'].nunique()} EEZs")

    # One package call does all the indicator maths.
    indicators = summarize_metacoupling(
        edges,
        system_col="focal_system_id",
        partner_col="destination_id",
        coupling_col="coupling_type",
        weight_col="flow_value",
        group_cols=None,
    )

    out = OUT_DIR / "indicators_by_eez.csv"
    indicators.to_csv(out, index=False)
    print(f"Wrote {len(indicators)} EEZ indicator rows -> {out}")

    # Quick sanity print: global total catch (sum of F_total) in Mt and Bt.
    total_t = indicators["F_total"].sum()
    print(f"Global total catch: {total_t:,.0f} t = {total_t/1e9:.3f} Bt")


if __name__ == "__main__":
    main()
