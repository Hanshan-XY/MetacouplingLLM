#!/usr/bin/env python
"""
02_compute_indicators.py  --  Label every flow with the package's country database,
then compute the per-EEZ metacoupling indicators.

INPUT  : data/fishing_edge_table.csv  (pinned, produced by 00_build_edge_table.py)
         data/sau_country_codes.csv   (pinned: the country of each EEZ and fishing nation)
OUTPUT : outputs/classified_edges.csv (every flow with the database's label and Carlson's)
         outputs/indicators_by_eez.csv

This step is PACKAGE functionality.  We do NOT re-implement the classification or
any indicator:

  1. build_adjacency(level="adm0", maritime_adjacency=True) says which countries
     border each other, on land or at sea (the bundled country database and its
     maritime table);
  2. classify_coupling labels each flow of an EEZ: intracoupled (I) when the fleet
     belongs to the EEZ's own country, pericoupled (P) when the two countries
     border each other, telecoupled (T) otherwise;
  3. summarize_metacoupling returns one row per EEZ:

    F_I, F_P, F_T, F_total,
    IFS, PFS, TFS, MFE,
    IFCI, PFCI, TFCI,
    ENP_I, ENP_P, ENP_T

Partners are countries.  A fleet counts as its country (data/sau_country_codes.csv)
and an EEZ's own catch as its country's fleet, so a country fishing the waters of
its own islands (Portugal in the Azores) is intracoupled.  The Channel Islands EEZ
spans two countries (Guernsey and Jersey) and borders what either borders; the
South Orkney Islands EEZ and one unknown fleet have no country and border nothing.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

# >>> PACKAGE FUNCTIONALITY <<<
from metacouplingllm.indicators import (
    build_adjacency,
    classify_coupling,
    summarize_metacoupling,
)

HERE = Path(__file__).resolve().parent
EDGES = HERE / "data" / "fishing_edge_table.csv"
CODES = HERE / "data" / "sau_country_codes.csv"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_codes() -> dict[str, list[str]]:
    """{Sea Around Us name: [ISO alpha-3 codes of its ADM0 units]} (empty: no country)."""
    codes: dict[str, list[str]] = defaultdict(list)
    with CODES.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            codes[row["name"]] += [row["adm0"]] if row["adm0"] else []
    return dict(codes)


def label_flows(edges: pd.DataFrame, codes: dict[str, list[str]]) -> pd.DataFrame:
    """Add eez_country, partner and the database's coupling_type to every flow."""
    names = set(edges["focal_system_id"]) | set(edges.loc[edges["carlson_type"] != "I", "destination_id"])
    missing = sorted(names - set(codes))
    if missing:
        raise SystemExit(f"names without a row in {CODES.name}: {missing}")

    def country(eez: str) -> str:
        # The EEZ's country as one partner ID ("GGY+JEY" for the Channel Islands).
        return "+".join(codes[eez]) or eez

    edges = edges.copy()
    edges["eez_country"] = edges["focal_system_id"].map(country)
    fleet = edges["destination_id"].map(lambda name: (codes.get(name) or [name])[0])
    edges["partner"] = fleet.where(edges["carlson_type"] != "I", edges["eez_country"])

    # Which countries border each other, on land or at sea (PACKAGE).
    known = {c for cs in codes.values() for c in cs}
    pairs = sorted({(a, b) for eez, b in zip(edges["focal_system_id"], edges["partner"])
                    for a in codes[eez] if b in known and b != a})
    adjacency = build_adjacency(pd.DataFrame(pairs, columns=["origin_id", "destination_id"]),
                                level="adm0", maritime_adjacency=True)
    hit = adjacency["adjacent"].fillna(0).eq(1)
    borders = {frozenset(p) for p in zip(adjacency.loc[hit, "origin_id"], adjacency.loc[hit, "destination_id"])}

    # Label each EEZ's flows (PACKAGE): the EEZ's countries are the focal system.
    labelled = []
    for eez, flows in edges.groupby("focal_system_id", sort=False):
        focal = [country(eez), *codes[eez]]
        others = sorted(set(flows["partner"]) - set(focal))
        table = pd.DataFrame({
            "origin_id": country(eez),
            "destination_id": others,
            "adjacent": [int(any(frozenset((a, b)) in borders for a in codes[eez])) for b in others],
        })
        labelled.append(classify_coupling(flows, focal_id=focal, adjacency=table,
                                          origin_col="eez_country", destination_col="partner"))
    return pd.concat(labelled, ignore_index=True)


def main() -> None:
    edges = pd.read_csv(EDGES)
    print(f"Loaded {len(edges)} edges across {edges['focal_system_id'].nunique()} EEZs")

    classified = label_flows(edges, load_codes())
    classified.to_csv(OUT_DIR / "classified_edges.csv", index=False)
    moved = classified[classified["coupling_type"] != classified["carlson_type"]]
    print(f"Labelled every flow with the country database: {len(moved)} of {len(classified)} flows "
          f"({moved['flow_value'].sum() / 1e6:,.1f} Mt) differ from Carlson's labels")

    # One package call does all the indicator maths.
    indicators = summarize_metacoupling(
        classified,
        system_col="focal_system_id",
        partner_col="partner",
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
