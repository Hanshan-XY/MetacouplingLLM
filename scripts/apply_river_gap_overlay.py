"""Apply the reviewed river-gap overlay to the shipped ADM1 adjacency data.

The river-gap overlay (``data/river_gap_overlay_pairs.csv``) is a reviewed
static layer applied on top of the geometry build, analogous to the
disputed-territory overlay and the OSM bridge classification. Its pairs are
cross-border, river-separated ADM1 units that the strict rook-contiguity build
omits because the World Bank source digitizes the two opposite banks as
non-touching geometry. They were identified by the near-miss audit
(``paper/SUPPLEMENT_river_border_audit.md``), ground-truthed against
authoritative border geography, and bridge-classified for ``coupling_standard``.

This script is **idempotent**: it appends each overlay pair to
``pericoupled_adm1_edge_list.csv`` and ``water_separated_pairs.csv`` only if it
is not already present. Re-running it (or running it after a fresh geometry
build) restores the overlay; running it on already-overlaid data is a no-op.

Usage:  python scripts/apply_river_gap_overlay.py
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "src" / "metacouplingllm" / "data"
EDGE = DATA / "pericoupled_adm1_edge_list.csv"
WATER = DATA / "water_separated_pairs.csv"
MANIFEST = DATA / "river_gap_overlay_pairs.csv"
NARROW_KM = 5.0


def _load_edge_meta() -> tuple[dict, set, list]:
    """Return (code -> (name, country, iso, region)), present-pair set, columns."""
    meta: dict[str, tuple[str, str, str, str]] = {}
    present: set[frozenset[str]] = set()
    with open(EDGE, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames
        for r in rd:
            meta[r["ADM1_code_A"]] = (r["ADM1_name_A"], r["country_A"],
                                      r["ISO_A3_A"], r["WB_region_A"])
            meta[r["ADM1_code_B"]] = (r["ADM1_name_B"], r["country_B"],
                                      r["ISO_A3_B"], r["WB_region_B"])
            present.add(frozenset({r["ADM1_code_A"], r["ADM1_code_B"]}))
    return meta, present, cols


def main() -> None:
    meta, present, cols = _load_edge_meta()
    water_present: set[frozenset[str]] = set()
    with open(WATER, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        wcols = rd.fieldnames
        for r in rd:
            if r.get("level") == "adm1":
                water_present.add(frozenset({r["code_a"], r["code_b"]}))

    edge_rows, water_rows = [], []
    with open(MANIFEST, newline="", encoding="utf-8-sig") as fh:
        for e in csv.DictReader(fh):
            if e.get("level") != "adm1":
                continue
            ca, cb = e["code_a"].strip(), e["code_b"].strip()
            pair = frozenset({ca, cb})
            if ca not in meta or cb not in meta:
                raise SystemExit(f"overlay unit not in edge list: {ca} or {cb}")
            km = float(e.get("border_km") or 0.0)
            if pair not in present:
                na, cna, ia, ra = meta[ca]
                nb, cnb, ib, rb = meta[cb]
                edge_rows.append({
                    "ADM1_code_A": ca, "ADM1_name_A": na, "country_A": cna,
                    "ISO_A3_A": ia, "WB_region_A": ra,
                    "ADM1_code_B": cb, "ADM1_name_B": nb, "country_B": cnb,
                    "ISO_A3_B": ib, "WB_region_B": rb,
                    "cross_country": ia != ib, "border_length_km": f"{km:.4f}",
                    "narrow_border": km < NARROW_KM, "potential_artifact": False,
                })
            if pair not in water_present:
                water_rows.append({
                    "level": "adm1", "code_a": ca, "code_b": cb,
                    "has_bridge": e.get("has_bridge", "False").strip(),
                    "water_type": "river", "water_body": e.get("river", ""),
                    "note": "river-gap overlay (audited near-miss)",
                })

    if edge_rows:
        with open(EDGE, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=cols).writerows(edge_rows)
    if water_rows:
        with open(WATER, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=wcols).writerows(water_rows)
    print(f"river-gap overlay applied: {len(edge_rows)} edge row(s), "
          f"{len(water_rows)} water row(s) added "
          f"(0 means already present / idempotent).")


if __name__ == "__main__":
    main()
