r"""Apply the reviewed overlay correction layer to the shipped adjacency data.

The pericoupling database is constructed in two stages: a deterministic
geometry build (``scripts/build_pericoupling_db.py``) and a **reviewed
correction layer** -- nine manifest CSVs in ``src/metacouplingllm/data/``
holding 445 pair-rows (28 edge-restoring + 417 water-flag-only), each pair
individually audited and human-verified (provenance: ``data/PROVENANCE.md``).
This engine applies the whole layer in one pass:

  ==================== ====================================== ========================
  registry             manifest                               effect
  ==================== ====================================== ========================
                                                              +2 water rows
  land_gap             land_gap_overlay_pairs.csv             +4 land edges
  hydro_water          hydro_water_overlay_pairs.csv          water flags on 18 edges
                                                              (incl. the Uruguay River,
                                                              geodesic 500 m)
  hydro_lakes          hydro_lakes_overlay_pairs.csv          water flags on 12 edges
                                                              (incl. the Dead Sea,
                                                              geodesic 500 m)
  rescreen_gap         rescreen_gap_overlay_pairs.csv         +16 edges, +16 water rows
                                                              (water-screen rebuild,
                                                              per-row water_type)
  rescreen_water       rescreen_water_overlay_pairs.csv       water flags on 386 edges
                                                              (water-screen rebuild
                                                              b1-b6 + 20km holds +
                                                              identity-audit shore
                                                              flags + the ru1-folded
                                                              river rows, per-row
                                                              water_type)
  ==================== ====================================== ========================

The manifests are the single source of truth for the correction data; this
file holds only behavior.  Water rows are matched to their overlay by the
``note`` constant in the registry (do not edit those strings -- they identify
the shipped rows); rows are updated in place, so a ``has_bridge`` edit in a
hydro_water/rescreen_water manifest propagates on the next run.  Overlay pairs
whose water row comes from the base bridge classification (empty note -- the
63 pre-classified lake pairs) are left untouched: the bridge CSV stays
authoritative for them.  The ADM0 roll-up is recomputed once, at the end
(a country pair is water-only iff *all* its ADM1 crossings are, with a
bridge iff any).

Idempotent and byte-stable: running on already-overlaid data changes no
file (each output is composed in memory and written only if its bytes
differ).  ``--check`` exits 2 instead of writing.

Usage:  python scripts/apply_overlays.py [--data-dir PATH] [--check]
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "src" / "metacouplingllm" / "data"
NARROW_KM = 5.0

# (name, manifest filename, adds_edges, edge_style, water_type, water_body_col, note)
# edge_style: "iso_lookup" = names from manifest, km as round(.,4), artifact=km<1;
#             None = flags-only (no edges added).
# water_body_col None = no water rows (ordinary land edges).
REGISTRY = [
    ("land_gap", "land_gap_overlay_pairs.csv", True, "iso_lookup",
     None, None, None),
    ("hydro_water", "hydro_water_overlay_pairs.csv", False, None,
     "river", "water_body", "hydro-water overlay (HydroRIVERS full-database cross-check + human map verification)"),
    ("hydro_lakes", "hydro_lakes_overlay_pairs.csv", False, None,
     "lake", "water_body", "hydro-lakes overlay (HydroLAKES full-database sweep + human map verification)"),
    ("rescreen_gap", "rescreen_gap_overlay_pairs.csv", True, "iso_lookup",
     None, "water_body", "rescreen-gap overlay (water-screen rebuild: recovered non-touching water border, two-pass adjudication + verification)"),
    ("rescreen_water", "rescreen_water_overlay_pairs.csv", False, None,
     None, "water_body", "rescreen-water overlay (water-screen rebuild full-ladder audit, two-pass adjudication + verification)"),
]


def _read_manifest(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    # legacy manifests may carry a level column; only adm1 rows are pair entries
    return [r for r in rows if r.get("level", "adm1").strip() == "adm1"]


def _detect_eol(path: Path) -> str:
    """Match the file's existing line ending (CRLF on autocrlf checkouts,
    LF on Linux/CI), so the engine is byte-stable on both."""
    return "\r\n" if b"\r\n" in path.read_bytes()[:4096] else "\n"


def _serialize(rows: list[list], header: list[str] | None = None,
               eol: str = "\r\n") -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=eol)
    if header is not None:
        w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _write_if_changed(path: Path, content: bytes, check: bool) -> bool:
    if path.read_bytes() == content:
        return False
    if not check:
        with open(path, "wb") as fh:
            fh.write(content)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true",
                    help="exit 2 if any file would change; write nothing")
    args = ap.parse_args(argv)
    data = args.data_dir
    edge_path = data / "pericoupled_adm1_edge_list.csv"
    water_path = data / "water_separated_pairs.csv"
    adm0_path = data / "PeriTelecoupling_clean.csv"
    edge_eol = _detect_eol(edge_path)
    water_eol = _detect_eol(water_path)
    adm0_eol = _detect_eol(adm0_path)

    manifests = {name: _read_manifest(data / fname)
                 for name, fname, *_ in REGISTRY}

    # ---- edge list: per-code meta, ISO lookups, present pairs ----------------
    meta: dict[str, tuple[str, str, str, str]] = {}
    iso_region: dict[str, str] = {}
    iso_country: dict[str, str] = {}
    present: set[frozenset[str]] = set()
    pair_iso: dict[frozenset[str], tuple[str, str]] = {}
    with open(edge_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            a, b = r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()
            meta[a] = (r["ADM1_name_A"], r["country_A"], r["ISO_A3_A"], r["WB_region_A"])
            meta[b] = (r["ADM1_name_B"], r["country_B"], r["ISO_A3_B"], r["WB_region_B"])
            ia, ib = r["ISO_A3_A"].strip(), r["ISO_A3_B"].strip()
            iso_region.setdefault(ia, r["WB_region_A"].strip())
            iso_region.setdefault(ib, r["WB_region_B"].strip())
            iso_country.setdefault(ia, r["country_A"].strip())
            iso_country.setdefault(ib, r["country_B"].strip())
            pair = frozenset({a, b})
            present.add(pair)
            pair_iso[pair] = (ia, ib)

    # ---- 1. edge additions (registry order, manifest row order) -------------
    new_edge_rows: list[list] = []
    for name, _fname, adds_edges, _style, *_ in REGISTRY:
        if not adds_edges:
            continue
        for m in manifests[name]:
            ca, cb = m["code_a"].strip(), m["code_b"].strip()
            pair = frozenset({ca, cb})
            if pair in present:
                continue
            km = float(m["border_km"])
            ia, ib = m["iso_a"].strip(), m["iso_b"].strip()
            if ia not in iso_country or ib not in iso_country:
                raise SystemExit(f"overlay ISO not in edge list: {ia} or {ib}")
            new_edge_rows.append([
                ca, m["name_a"], iso_country[ia], ia, iso_region[ia],
                cb, m["name_b"], iso_country[ib], ib, iso_region[ib],
                str(ia != ib), round(km, 4), str(km < NARROW_KM), str(km < 1.0),
            ])
            present.add(pair)
            pair_iso[pair] = (ia, ib)

    # ---- 2. flags-only overlays must be existing edges ----------------------
    for name, _fname, adds_edges, *_ in REGISTRY:
        if adds_edges:
            continue
        for m in manifests[name]:
            pair = frozenset({m["code_a"].strip(), m["code_b"].strip()})
            if pair not in present:
                raise SystemExit(
                    f"{name}: {m['code_a']}<->{m['code_b']} not found in edge "
                    f"list (must be an existing edge)")

    # ---- 3. water rows: update-in-place / append-if-missing -----------------
    adm1_rows: list[tuple] = []
    row_index: dict[frozenset[str], int] = {}
    with open(water_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["level"] != "adm1":
                continue
            key = frozenset({r["code_a"].strip(), r["code_b"].strip()})
            row_index[key] = len(adm1_rows)
            adm1_rows.append((r["code_a"].strip(), r["code_b"].strip(),
                              str(r.get("has_bridge", "")).strip(),
                              r.get("water_type", "").strip(),
                              r.get("water_body", "").strip(),
                              r.get("note", "").strip()))

    for name, _fname, _adds, _style, wtype, body_col, note in REGISTRY:
        if body_col is None:
            continue
        for m in manifests[name]:
            ca, cb = m["code_a"].strip(), m["code_b"].strip()
            key = frozenset({ca, cb})
            br = "True" if str(m.get("has_bridge", "")).strip() == "True" else "False"
            # registry water_type None = mixed manifest, type carried per row
            wt = wtype or m["water_type"].strip()
            row = (ca, cb, br, wt, m[body_col].strip(), note)
            i = row_index.get(key)
            if i is None:                      # fresh build: append
                row_index[key] = len(adm1_rows)
                adm1_rows.append(row)
            elif adm1_rows[i][5] == note:      # this overlay's row: sync in place
                adm1_rows[i] = (adm1_rows[i][0], adm1_rows[i][1], br,
                                wt, m[body_col].strip(), note)
            # else: base bridge-classification row (or another overlay's) wins

    # ---- 4. ADM0 roll-up, computed once --------------------------------------
    water_all = {frozenset({a, b}) for a, b, *_ in adm1_rows}
    has_b = {frozenset({a, b}): (br == "True") for a, b, br, *_ in adm1_rows}
    by_country = collections.defaultdict(list)
    for pair, (ia, ib) in pair_iso.items():
        if ia != ib:
            by_country[frozenset({ia, ib})].append(pair)
    adm0_rows = []
    for ctypair, crossings in by_country.items():
        if all(c in water_all for c in crossings):
            ia, ib = sorted(ctypair)
            anyb = "True" if any(has_b.get(c, False) for c in crossings) else "False"
            adm0_rows.append((ia, ib, anyb))

    # ---- 5. ADM0 matrix: a restored cross-country edge implies adjacency ----
    matrix_rows = list(csv.DictReader(open(adm0_path, newline="", encoding="utf-8-sig")))
    cells = {(r["Sending"].strip(), r["Receiving"].strip()): r for r in matrix_rows}
    patched = []
    for name, _fname, adds_edges, *_ in REGISTRY:
        if not adds_edges:
            continue
        for m in manifests[name]:
            ia, ib = m["iso_a"].strip(), m["iso_b"].strip()
            if ia == ib:
                continue
            for a, b in ((ia, ib), (ib, ia)):
                row = cells.get((a, b))
                if row is not None and row["Intracoupling"].strip() != "1":
                    row["Intracoupling"] = "1"
                    patched.append((a, b))

    # ---- 6. compose outputs; write only what changed -------------------------
    changed = []

    if new_edge_rows:  # append-only, exactly like the superseded scripts
        content = edge_path.read_bytes() + _serialize(new_edge_rows, eol=edge_eol)
        if _write_if_changed(edge_path, content, args.check):
            changed.append(edge_path.name)

    water_out = [["adm1", *row] for row in adm1_rows]
    water_out += [["adm0", ia, ib, br, "", "", "adm1-rollup"]
                  for ia, ib, br in sorted(adm0_rows)]
    header = ["level", "code_a", "code_b", "has_bridge", "water_type", "water_body", "note"]
    if _write_if_changed(water_path, _serialize(water_out, header, eol=water_eol), args.check):
        changed.append(water_path.name)

    if patched:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["Sending", "Receiving", "Intracoupling"],
                           lineterminator=adm0_eol)
        w.writeheader()
        w.writerows(matrix_rows)
        if _write_if_changed(adm0_path, buf.getvalue().encode("utf-8"), args.check):
            changed.append(adm0_path.name)

    nob = sum(1 for r in adm1_rows if r[2] != "True")
    print(f"overlay layer applied: +{len(new_edge_rows)} edge row(s); "
          f"adm0 matrix cells patched: {patched or 'none'}")
    print(f"  adm1 water-only rows: {len(adm1_rows)}  "
          f"(bridge {len(adm1_rows) - nob} / no-bridge {nob})")
    print(f"  adm0 roll-up rows:    {len(adm0_rows)}")
    print(f"  files {'needing change' if args.check else 'changed'}: "
          f"{changed or 'none (byte-stable no-op)'}")
    return 2 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
