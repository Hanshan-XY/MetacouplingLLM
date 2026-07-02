r"""Apply the wide-river water-only overlay to ``water_separated_pairs.csv``.

These ADM1 pairs are *already edges* in ``pericoupled_adm1_edge_list.csv`` (they
share a border), but the 2.4 km river-centerline candidate screen under-counted
them because the river (chiefly the Danube) is very wide and the World-Bank
border weaves >2.4 km from the Natural Earth centerline.  A 5 km re-screen plus
per-pair LLM ground-truth + adversarial verification recovered them as genuine
water-only borders (the other 5 km candidates were rejected as mixed river+land).

So this overlay only adds a *water-only flag* (it does NOT add edges): it appends
ADM1 rows to ``water_separated_pairs.csv`` and recomputes the deterministic ADM0
roll-up (a country pair is water-only iff *all* its ADM1 crossings are, with a
bridge iff any).  Idempotent: re-running adds 0 rows.

Source of the set: ``wide_river_overlay_pairs.csv`` (manifest).  Method + audit:
``paper/SUPPLEMENT_river_border_audit.md`` / ``docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md``.
"""
from __future__ import annotations
import collections
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "src" / "metacouplingllm" / "data"
EDGE_LIST = DATA / "pericoupled_adm1_edge_list.csv"
WATER = DATA / "water_separated_pairs.csv"
MANIFEST = DATA / "wide_river_overlay_pairs.csv"
NOTE = "wide-river overlay (5km re-screen + ground-truth)"

# (code_a, code_b, has_bridge, water_body) -- has_bridge from OSM + web verification
OVERLAY = [
    ("BGR018", "ROU012", False, "Danube"),
    ("BGR016", "ROU020", True,  "Danube"),          # Friendship Bridge (Ruse-Giurgiu)
    ("BGR013", "ROU037", False, "Danube"),
    ("BGR010", "ROU018", False, "Danube"),
    ("BGR028", "ROU018", False, "Danube"),
    ("BGR013", "ROU031", False, "Danube"),
    ("BGR026", "ROU037", False, "Danube"),
    ("BGR016", "ROU037", False, "Danube"),
    # Niassa<->Ruvuma: public documentation of Unity Bridge 2 (Mkwenda /
    # Matchedje) is conflicting, and the audit's initial conservative call was
    # has_bridge=False; True rests on domain confirmation (2026-06-29) that at
    # least one fixed crossing exists here, not on OSM + web verification alone.
    ("MOZ008", "TZA025", True,  "Ruvuma"),          # Unity Bridge 2 (Mkwenda), domain-confirmed
    ("ALB031", "MNE020", True,  "Bojana"),          # Muriqan-Sukobin bridge
    ("DEU011", "LUX002", True,  "Moselle; Sauer"),  # Wasserbillig road bridge
    ("PRK002", "RUS060", True,  "Tumen"),           # Korea-Russia Friendship rail bridge
    ("BWA002", "ZMB109", True,  "Chobe; Zambezi"),  # Kazungula Bridge
]


def _edge_index():
    info, iso = {}, {}
    with open(EDGE_LIST, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            a, b = r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()
            info[frozenset({a, b})] = r
            iso[a], iso[b] = r["ISO_A3_A"].strip(), r["ISO_A3_B"].strip()
    return info, iso


def main() -> None:
    info, iso = _edge_index()

    # 1. write manifest (provenance)
    mcols = ["code_a", "name_a", "iso_a", "code_b", "name_b", "iso_b",
             "water_body", "border_km", "has_bridge", "source"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(mcols)
        for ca, cb, hb, wb in OVERLAY:
            r = info.get(frozenset({ca, cb}))
            if r is None:
                raise SystemExit(f"{ca}<->{cb} not found in edge list (must be an existing edge)")
            # orient names to the (ca, cb) order
            if r["ADM1_code_A"].strip() == ca:
                na, ia, nb, ib = r["ADM1_name_A"], r["ISO_A3_A"], r["ADM1_name_B"], r["ISO_A3_B"]
            else:
                na, ia, nb, ib = r["ADM1_name_B"], r["ISO_A3_B"], r["ADM1_name_A"], r["ISO_A3_A"]
            w.writerow([ca, na, ia.strip(), cb, nb, ib.strip(), wb,
                        round(float(r["border_length_km"]), 1), hb,
                        "5km re-screen + LLM ground-truth + adversarial verify (2026-06)"])

    # 2. load existing adm1 water rows (drop adm0 -- recomputed; drop overlay
    #    rows -- re-added fresh below so has_bridge edits in OVERLAY take effect)
    overlay_keys = {frozenset({a, b}) for a, b, *_ in OVERLAY}
    adm1_rows, seen = [], set()
    with open(WATER, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["level"] != "adm1":
                continue
            key = frozenset({r["code_a"].strip(), r["code_b"].strip()})
            if key in overlay_keys:
                continue
            seen.add(key)
            adm1_rows.append((r["code_a"].strip(), r["code_b"].strip(),
                              str(r.get("has_bridge", "")).strip(),
                              r.get("water_type", "").strip(),
                              r.get("water_body", "").strip(),
                              r.get("note", "").strip()))

    # 3. append overlay (dedupe)
    added = 0
    for ca, cb, hb, wb in OVERLAY:
        if frozenset({ca, cb}) in seen:
            continue
        adm1_rows.append((ca, cb, "True" if hb else "False", "river", wb, NOTE))
        seen.add(frozenset({ca, cb}))
        added += 1

    # 4. recompute ADM0 roll-up
    water_all = {frozenset({a, b}) for a, b, *_ in adm1_rows}
    has_b = {frozenset({a, b}): (br == "True") for a, b, br, *_ in adm1_rows}
    by_country = collections.defaultdict(list)
    for e in {frozenset({a, b}) for a, b, *_ in [(r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip())
              for r in csv.DictReader(open(EDGE_LIST, encoding="utf-8-sig"))]}:
        a, b = tuple(e)
        ia, ib = iso.get(a), iso.get(b)
        if ia and ib and ia != ib:
            by_country[frozenset({ia, ib})].append(e)
    adm0_rows = []
    for ctypair, crossings in by_country.items():
        if all(c in water_all for c in crossings):
            ia, ib = sorted(ctypair)
            anyb = "True" if any(has_b.get(c, False) for c in crossings) else "False"
            adm0_rows.append((ia, ib, anyb))

    # 5. write water_separated_pairs.csv
    with open(WATER, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "code_a", "code_b", "has_bridge", "water_type", "water_body", "note"])
        for ca, cb, br, wt, wb, note in adm1_rows:
            w.writerow(["adm1", ca, cb, br, wt, wb, note])
        for ia, ib, br in sorted(adm0_rows):
            w.writerow(["adm0", ia, ib, br, "", "", "adm1-rollup"])

    nob = sum(1 for r in adm1_rows if r[2] != "True")
    print(f"overlay applied: +{added} adm1 row(s)")
    print(f"  adm1 water-only total: {len(adm1_rows)}  (bridge {len(adm1_rows)-nob} / no-bridge {nob})")
    print(f"  adm0 roll-up total:    {len(adm0_rows)}")


if __name__ == "__main__":
    main()
