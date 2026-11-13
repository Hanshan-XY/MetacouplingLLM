#!/usr/bin/env python
"""
03_analysis.py  --  Reproduce the paper's regional tables (Tables 4 and 5) and the
summary figures of its fishing example (Section 5.2).

INPUT
    outputs/indicators_by_eez.csv  -- per-EEZ indicators from the package (step 02)
    outputs/classified_edges.csv   -- every flow with the database's label and Carlson's (step 02)
    data/eez_region.json           -- per-EEZ region map  {EEZ: {grp, region}}

OUTPUT
    outputs/regional_catch.csv       -- Table 4: regional catch totals and proportions
    outputs/regional_indicators.csv  -- Table 5: catch-weighted regional indicators
    outputs/global_totals.csv        -- global shares and the Section 5.2 figures
    outputs/carlson_comparison.csv   -- flows and catch by Carlson's label and the database's
    outputs/telecoupled_share_timeseries.csv   (optional; needs the source tables)

ALL the statistics in this file are USER-SUPPLIED analysis built ON TOP of the
package's labels and per-EEZ indicator table.  The package classifies the flows and
computes the indicators; this script aggregates them by region.

  * Regions follow the 'grp' field of eez_region.json (Carlson et al.'s Table S16
    scheme, joined by 01_build_region_map.py).  The eleven regions cover 276 of the
    280 EEZs; four EEZs absent from Carlson's nation list are unassigned (Taiwan,
    Cape Verde, Cook Islands, Ascension) and are left out of both tables.
  * Regional indicator means are CATCH-WEIGHTED (weight = F_total).  An EEZ with no
    flow in a coupling type has no PFCI/TFCI/ENP for it, so those means cover only
    EEZs with positive catch in that type, with the weights renormalized.
  * Rows are ordered by telecoupled share, highest first, as in the paper.
"""

from __future__ import annotations

import json
import os
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IND = HERE / "outputs" / "indicators_by_eez.csv"
EDGES = HERE / "outputs" / "classified_edges.csv"
REGION_JSON = HERE / "data" / "eez_region.json"
OUT_DIR = HERE / "outputs"
SRC_DIR = Path(os.environ.get("CARLSON2020_SUPPLEMENT", HERE / "source"))

# Columns reported in Table 5 (the catch-weighted regional means).
INDICATOR_COLS = ["IFS", "PFS", "TFS", "MFE", "PFCI", "TFCI", "ENP_P", "ENP_T"]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Catch-weighted mean, ignoring NaN indicator values (and their weight)."""
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    mask = ~np.isnan(v)
    if not mask.any() or w[mask].sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def main() -> None:
    ind = pd.read_csv(IND)
    edges = pd.read_csv(EDGES)

    region_map = json.loads(REGION_JSON.read_text(encoding="utf-8"))
    grp = {eez: meta.get("grp") for eez, meta in region_map.items()}
    ind["region"] = ind["focal_system_id"].map(grp)
    reg = ind[ind["region"].notna()].copy()

    # ----- Table 4: regional catch totals and proportions (million metric tons) ---
    catch_rows = []
    for region, sub in reg.groupby("region"):
        total, t1, t2, t3 = (sub[c].sum() / 1e6 for c in ("F_total", "F_I", "F_P", "F_T"))
        catch_rows.append({
            "region": region, "n_eez": len(sub), "total_MMT": total,
            "type1_MMT": t1, "type2_MMT": t2, "type3_MMT": t3,
            "type1_pct": 100 * t1 / total, "type2_pct": 100 * t2 / total,
            "type3_pct": 100 * t3 / total,
        })
    table4 = pd.DataFrame(catch_rows).sort_values("type3_pct", ascending=False)
    table4.to_csv(OUT_DIR / "regional_catch.csv", index=False)

    # ----- Table 5: catch-weighted regional indicators ---------------------------
    ind_rows = []
    for region, sub in reg.groupby("region"):
        row = {"region": region, "n_eez": len(sub)}
        for col in INDICATOR_COLS:
            row[col] = weighted_mean(sub[col], sub["F_total"])
        ind_rows.append(row)
    table5 = pd.DataFrame(ind_rows).sort_values("TFS", ascending=False)
    table5.to_csv(OUT_DIR / "regional_indicators.csv", index=False)

    # ----- The database's labels against Carlson et al.'s -------------------------
    comparison = (edges.groupby(["carlson_type", "coupling_type"])["flow_value"]
                  .agg(n_flows="size", catch_MMT="sum").reset_index())
    comparison["catch_MMT"] /= 1e6
    comparison.to_csv(OUT_DIR / "carlson_comparison.csv", index=False)
    foreign = edges[edges["carlson_type"] != "I"]          # Carlson's adjacent and distant flows
    same = foreign["coupling_type"] == foreign["carlson_type"]

    # ----- Global figures --------------------------------------------------------
    total_t = ind["F_total"].sum()
    distant = ind[ind["F_T"] > 0]
    nominal = (edges[edges["coupling_type"] == "T"]
               .groupby("focal_system_id")["partner"].nunique()
               .reindex(distant["focal_system_id"]))
    global_rows = pd.DataFrame([
        {"metric": "n_eez", "value": ind["focal_system_id"].nunique()},
        {"metric": "total_catch_Bt", "value": total_t / 1e9},
        {"metric": "own_plus_adjacent_share", "value": (ind["F_I"].sum() + ind["F_P"].sum()) / total_t},
        {"metric": "telecoupled_share", "value": ind["F_T"].sum() / total_t},
        {"metric": "n_eez_region_assigned", "value": len(reg)},
        {"metric": "region_assigned_catch_share", "value": reg["F_total"].sum() / total_t},
        {"metric": "n_eez_with_distant_catch", "value": len(distant)},
        {"metric": "mean_ENP_T_unweighted", "value": distant["ENP_T"].mean()},
        {"metric": "mean_nominal_distant_nations", "value": nominal.mean()},
        {"metric": "n_foreign_flows", "value": len(foreign)},
        {"metric": "carlson_agreement_flows", "value": same.mean()},
        {"metric": "carlson_agreement_catch",
         "value": foreign.loc[same, "flow_value"].sum() / foreign["flow_value"].sum()},
    ])
    global_rows.to_csv(OUT_DIR / "global_totals.csv", index=False)

    g = dict(zip(global_rows["metric"], global_rows["value"]))
    print("=== Global ===")
    print(f"  EEZs: {g['n_eez']:.0f}; region-assigned: {g['n_eez_region_assigned']:.0f} "
          f"({100 * g['region_assigned_catch_share']:.2f}% of catch)")
    print(f"  total catch {g['total_catch_Bt']:.3f} Bt; own + adjacent {100 * g['own_plus_adjacent_share']:.1f}%")
    print(f"  EEZs with distant catch: {g['n_eez_with_distant_catch']:.0f}; unweighted mean ENP_T "
          f"{g['mean_ENP_T_unweighted']:.2f} vs {g['mean_nominal_distant_nations']:.2f} nominal distant nations")
    print(f"  Carlson's adjacent and distant flows: the database gives the same label to "
          f"{100 * g['carlson_agreement_flows']:.1f}% of the {g['n_foreign_flows']:.0f} flows and "
          f"{100 * g['carlson_agreement_catch']:.1f}% of their catch")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n=== Table 4: regional catch (MMT, %) ===")
        print(table4.set_index("region").round(1))
        print("\n=== Table 5: catch-weighted regional indicators ===")
        print(table5.set_index("region").round(2))
        print("\n=== Carlson's label -> the database's ===")
        print(comparison.set_index(["carlson_type", "coupling_type"]).round(1))

    _telecoupled_timeseries(edges)


def _telecoupled_timeseries(edges: pd.DataFrame) -> None:
    """Per-year global telecoupled share = T / (I + P + T), with the database's labels.

    Re-reads the source tables, year by year (the pinned edge table sums over
    years), with step 00's parsers, and gives each year's catch the label its
    (EEZ, fishing nation) flow got in step 02.  This is a supplementary,
    descriptive figure -- not part of the regional tables.
    """
    if not (SRC_DIR / "Table S4.xlsx").exists():
        print(f"\n(Source tables not found in {SRC_DIR}; skipping temporal series.)")
        return

    step00 = runpy.run_path(str(HERE / "00_build_edge_table.py"))   # its parsers, not its main()
    label = dict(zip(zip(edges["focal_system_id"], edges["destination_id"]), edges["coupling_type"]))
    totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for year, _eez, catch in step00["iter_own"](SRC_DIR / "Table S4.xlsx"):
        totals[int(float(year))]["I"] += catch
    for table in ("Table S6.xlsx", "Table S8.xlsx"):
        for year, eez, nation, catch in step00["iter_blocks"](SRC_DIR / table):
            if catch:
                totals[int(float(year))][label[(eez, nation)]] += catch

    rows = []
    for y in sorted(totals):
        tot = sum(totals[y].values())
        rows.append({"year": y, "telecoupled_share": totals[y]["T"] / tot if tot else np.nan})
    ts = pd.DataFrame(rows)
    ts.to_csv(OUT_DIR / "telecoupled_share_timeseries.csv", index=False)

    peak = ts.loc[ts["telecoupled_share"].idxmax()]
    last = ts.iloc[-1]
    print("\n=== Telecoupled share over time (optional) ===")
    print(f"  peak: {peak['telecoupled_share']:.3f} in {int(peak['year'])}")
    print(f"  {int(last['year'])}: {last['telecoupled_share']:.3f}")


if __name__ == "__main__":
    main()
