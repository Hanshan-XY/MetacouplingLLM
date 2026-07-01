r"""Apply the lake overlay: restore the audited lake-only ADM1 pairs as edges.

The geometry build's lake filter removes shared-boundary segments lying inside
Natural Earth lake polygons, dropping pairs that meet only across a lake.  That
made ``coupling_standard`` river-only in practice: a *bridged* lake pair (e.g.
Flevoland <-> Noord-Holland via the Houtribdijk) could never be pericoupled,
because the loaders only subtract from the base graph.

This overlay restores all audited lake-only pairs (``lake_overlay_pairs.csv``,
the Natural Earth lake-filter candidate set, each with an OSM + web + manual
``has_bridge`` review) as base edges, so the toggle governs lakes exactly like
rivers with NO loader change:

  * ``lenient``   - every water border (river or lake) is pericoupled;
  * ``moderate``  - only water borders with a fixed crossing open to traffic;
  * ``stringent`` - no water-only border is pericoupled.

It also patches the ADM0 matrix for country pairs whose ONLY contact is a lake
(COD <-> TZA across Lake Tanganyika) and recomputes the deterministic ADM0
roll-up in ``water_separated_pairs.csv``.  Idempotent: re-running adds 0 rows.
"""
from __future__ import annotations
import collections
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "src" / "metacouplingllm" / "data"
EDGE_LIST = DATA / "pericoupled_adm1_edge_list.csv"
WATER = DATA / "water_separated_pairs.csv"
MANIFEST = DATA / "lake_overlay_pairs.csv"
ADM0 = DATA / "PeriTelecoupling_clean.csv"

# WB country display names for units absent from the edge list is not needed:
# country names are looked up per ISO from existing rows (every ISO in the
# manifest already has edges).


def main() -> None:
    manifest = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8-sig")))

    # --- lookups from the existing edge list: ISO -> (region, country name) ---
    iso_region: dict[str, str] = {}
    iso_country: dict[str, str] = {}
    existing: set[frozenset[str]] = set()
    with open(EDGE_LIST, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames
        for r in rdr:
            existing.add(frozenset({r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()}))
            iso_region.setdefault(r["ISO_A3_A"].strip(), r["WB_region_A"].strip())
            iso_region.setdefault(r["ISO_A3_B"].strip(), r["WB_region_B"].strip())
            iso_country.setdefault(r["ISO_A3_A"].strip(), r["country_A"].strip())
            iso_country.setdefault(r["ISO_A3_B"].strip(), r["country_B"].strip())

    # --- 1. append missing lake edges ---
    added = 0
    with open(EDGE_LIST, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for m in manifest:
            pair = frozenset({m["code_a"], m["code_b"]})
            if pair in existing:
                continue
            km = float(m["border_km"])
            w.writerow([
                m["code_a"], m["name_a"], iso_country[m["iso_a"]], m["iso_a"], iso_region[m["iso_a"]],
                m["code_b"], m["name_b"], iso_country[m["iso_b"]], m["iso_b"], iso_region[m["iso_b"]],
                str(m["iso_a"] != m["iso_b"]), round(km, 4),
                str(km < 5.0), str(km < 1.0),
            ])
            existing.add(pair)
            added += 1

    # --- 2. ADM0 matrix: country pairs whose only contact is a lake ---
    adm0_rows = list(csv.DictReader(open(ADM0, newline="", encoding="utf-8-sig")))
    adj: dict[tuple[str, str], dict] = {}
    for r in adm0_rows:
        adj[(r["Sending"].strip(), r["Receiving"].strip())] = r
    patched = []
    for m in manifest:
        ia, ib = m["iso_a"], m["iso_b"]
        if ia == ib:
            continue
        for a, b in ((ia, ib), (ib, ia)):
            row = adj.get((a, b))
            if row is not None and row["Intracoupling"].strip() != "1":
                row["Intracoupling"] = "1"
                patched.append((a, b))
    if patched:
        with open(ADM0, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["Sending", "Receiving", "Intracoupling"])
            w.writeheader()
            w.writerows(adm0_rows)

    # --- 3. recompute the ADM0 roll-up in water_separated_pairs.csv ---
    adm1_rows, water_all, has_b = [], set(), {}
    with open(WATER, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["level"] != "adm1":
                continue
            pair = frozenset({r["code_a"].strip(), r["code_b"].strip()})
            water_all.add(pair)
            has_b[pair] = str(r.get("has_bridge", "")).strip() == "True"
            adm1_rows.append((r["code_a"].strip(), r["code_b"].strip(),
                              str(r.get("has_bridge", "")).strip(),
                              r.get("water_type", "").strip(),
                              r.get("water_body", "").strip(),
                              r.get("note", "").strip()))
    edges, iso = set(), {}
    with open(EDGE_LIST, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            a, b = r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()
            edges.add(frozenset({a, b}))
            iso[a], iso[b] = r["ISO_A3_A"].strip(), r["ISO_A3_B"].strip()
    by_country = collections.defaultdict(list)
    for e in edges:
        a, b = tuple(e)
        ia, ib = iso.get(a), iso.get(b)
        if ia and ib and ia != ib:
            by_country[frozenset({ia, ib})].append(e)
    adm0_out = []
    for ctypair, crossings in by_country.items():
        if all(c in water_all for c in crossings):
            ia, ib = sorted(ctypair)
            anyb = "True" if any(has_b.get(c, False) for c in crossings) else "False"
            adm0_out.append((ia, ib, anyb))
    with open(WATER, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "code_a", "code_b", "has_bridge", "water_type", "water_body", "note"])
        for ca, cb, br, wt, wb, note in adm1_rows:
            w.writerow(["adm1", ca, cb, br, wt, wb, note])
        for ia, ib, br in sorted(adm0_out):
            w.writerow(["adm0", ia, ib, br, "", "", "adm1-rollup"])

    nob = sum(1 for r in adm1_rows if r[2] != "True")
    print(f"lake overlay applied: +{added} adm1 edge(s); adm0 matrix cells patched: {patched or 'none'}")
    print(f"  adm1 water-only rows: {len(adm1_rows)}  (bridge {len(adm1_rows)-nob} / no-bridge {nob})")
    print(f"  adm0 roll-up rows:    {len(adm0_out)}")


if __name__ == "__main__":
    main()
