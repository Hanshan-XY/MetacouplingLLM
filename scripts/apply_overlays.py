r"""Apply the reviewed inputs of Stages 3 and 4 to the geometry build's output.

The pericoupling database is built in four stages.  The geometry build
(``scripts/build_pericoupling_db.py``) runs Stages 1 and 2: exact-contact
contiguity with the source-relabel and the unit merge, then the de facto
overlay across disputed areas.  This engine runs the other two, each from
reviewed files in ``src/metacouplingllm/data/`` (provenance:
``data/PROVENANCE.md``):

  ===== ================================== =========================================
  stage file                               effect
  ===== ================================== =========================================
  3     water_classification_pairs.csv     the water-only classification, 803 rows:
                                           779 existing edges flagged water-only and
                                           24 water borders between non-touching
                                           units added (``adds_edge``), each with its
                                           water row
  4     land_gap_overlay_pairs.csv         +4 sub-tolerance land borders
  4     denylist_pairs.csv                 -5 contacts found not to be borders
  ===== ================================== =========================================

The files are the single source of truth for the reviewed data; this script
holds only behavior.  It composes the whole water table from the water file
(the ``note`` column names the instrument class that nominated the row),
computes the ADM0 roll-up once, after Stage 4 (a country pair is water-only
iff *all* its ADM1 crossings are, with a bridge iff any), and marks a country
pair adjacent in the ADM0 matrix when an added cross-country edge implies it.

Idempotent and byte-stable: running on already-processed data changes no
file (each output is composed in memory and written only if its bytes
differ).  ``--check`` exits 2 instead of writing.  Row order is canonical:
the native rows in geometry-build order (the denylisted contacts removed),
then the Stage 3 edges in water-file order, then the Stage 4 edges in
manifest order; the water table in water-file order, then the ADM0 roll-ups.
A fresh ``--full`` build produces the same order.

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

WATER_FILE = "water_classification_pairs.csv"   # Stage 3
LAND_GAP_FILE = "land_gap_overlay_pairs.csv"    # Stage 4: adds edges
DENYLIST_FILE = "denylist_pairs.csv"            # Stage 4: removes edges
INPUT_FILES = (WATER_FILE, LAND_GAP_FILE, DENYLIST_FILE)
NOTE_ON_EDGE = "water-only border on a shared edge (edge screens; two-model adjudication)"
NOTE_NON_TOUCHING = ("water-only border between non-touching units (corridor census; "
                     "two-model adjudication)")
WATER_HEADER = ["level", "code_a", "code_b", "has_bridge", "water_type", "water_body",
                "note", "adjudication", "verification_tier"]


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _key(row: dict) -> frozenset[str]:
    return frozenset({row["code_a"].strip(), row["code_b"].strip()})


def _detect_eol(path: Path, default: str = "\r\n") -> str:
    """Match the file's existing line ending (CRLF on autocrlf checkouts,
    LF on Linux/CI), so the engine is byte-stable on both; ``default`` for a
    file the engine creates (the water table on a fresh ``--full`` build)."""
    if not path.exists():
        return default
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
    if path.exists() and path.read_bytes() == content:
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
    water_eol = _detect_eol(water_path, default=edge_eol)
    adm0_eol = _detect_eol(adm0_path)

    water = _read(data / WATER_FILE)
    land_gap = _read(data / LAND_GAP_FILE)
    denylist = _read(data / DENYLIST_FILE)
    water_keys = [_key(m) for m in water]
    if len(set(water_keys)) != len(water_keys):
        raise SystemExit(f"{WATER_FILE}: a pair appears twice")
    deny_keys = {_key(m) for m in denylist}
    clash = deny_keys & (set(water_keys) | {_key(m) for m in land_gap})
    if clash:
        raise SystemExit(f"{DENYLIST_FILE}: pair also added or classified: {sorted(map(sorted, clash))}")

    # ---- edge list: per-code meta, ISO lookups, present pairs ----------------
    iso_region: dict[str, str] = {}
    iso_country: dict[str, str] = {}
    present: set[frozenset[str]] = set()
    pair_iso: dict[frozenset[str], tuple[str, str]] = {}
    with open(edge_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            a, b = r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()
            ia, ib = r["ISO_A3_A"].strip(), r["ISO_A3_B"].strip()
            iso_region.setdefault(ia, r["WB_region_A"].strip())
            iso_region.setdefault(ib, r["WB_region_B"].strip())
            iso_country.setdefault(ia, r["country_A"].strip())
            iso_country.setdefault(ib, r["country_B"].strip())
            pair = frozenset({a, b})
            present.add(pair)
            pair_iso[pair] = (ia, ib)

    new_edge_rows: list[list] = []

    def _add_edge(m: dict) -> None:
        """A reviewed addition, named and measured by its file row."""
        pair = _key(m)
        if pair in present:
            return
        km = float(m["border_km"])
        ia, ib = m["iso_a"].strip(), m["iso_b"].strip()
        if ia not in iso_country or ib not in iso_country:
            raise SystemExit(f"added edge's ISO not in the edge list: {ia} or {ib}")
        new_edge_rows.append([
            m["code_a"].strip(), m["name_a"], iso_country[ia], ia, iso_region[ia],
            m["code_b"].strip(), m["name_b"], iso_country[ib], ib, iso_region[ib],
            str(ia != ib), round(km, 4), str(km < NARROW_KM), str(km < 1.0),
        ])
        present.add(pair)
        pair_iso[pair] = (ia, ib)

    # ---- Stage 3: the water-only classification ------------------------------
    stage3_added = [m for m in water if m["adds_edge"].strip() == "True"]
    for m in stage3_added:
        _add_edge(m)
    n_added3 = len(new_edge_rows)
    for m in water:
        if _key(m) not in present:
            raise SystemExit(f"{WATER_FILE}: {m['code_a']}<->{m['code_b']} is not an edge "
                             f"(a water flag needs an existing edge)")
    adm1_rows = [(m["code_a"].strip(), m["code_b"].strip(),
                  "True" if m["has_bridge"].strip() == "True" else "False",
                  m["water_type"].strip(), m["water_body"].strip(),
                  NOTE_NON_TOUCHING if m["adds_edge"].strip() == "True" else NOTE_ON_EDGE,
                  m["adjudication"].strip(), m["verification_tier"].strip())
                 for m in water]

    # ---- Stage 4: edge corrections -------------------------------------------
    for m in land_gap:
        _add_edge(m)
    n_added4 = len(new_edge_rows) - n_added3
    removed = [k for k in deny_keys if k in present]
    for k in deny_keys:
        present.discard(k)
        pair_iso.pop(k, None)

    # ---- ADM0 roll-up, computed once, after Stage 4 ---------------------------
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

    # ---- ADM0 matrix: an added cross-country edge implies adjacency ----------
    matrix_rows = _read(adm0_path)
    cells = {(r["Sending"].strip(), r["Receiving"].strip()): r for r in matrix_rows}
    patched = []
    for m in [*stage3_added, *land_gap]:
        ia, ib = m["iso_a"].strip(), m["iso_b"].strip()
        if ia == ib:
            continue
        for a, b in ((ia, ib), (ib, ia)):
            row = cells.get((a, b))
            if row is not None and row["Intracoupling"].strip() != "1":
                row["Intracoupling"] = "1"
                patched.append((a, b))

    # ---- compose outputs; write only what changed -----------------------------
    changed = []

    # Edge list, canonical order: the native rows (geometry-build order) stay
    # verbatim, the denylisted contacts removed; the added rows follow, Stage 3's
    # in water-file order, then Stage 4's in manifest order.
    added_order = {_key(m): i for i, m in enumerate([*stage3_added, *land_gap])}
    new_edge_line = {frozenset({row[0], row[5]}): _serialize([row], eol="").decode("utf-8")
                     for row in new_edge_rows}
    raw_edge = edge_path.read_bytes()
    bom = b"\xef\xbb\xbf" if raw_edge.startswith(b"\xef\xbb\xbf") else b""
    edge_lines = raw_edge.decode("utf-8-sig").split(edge_eol)
    if edge_lines and edge_lines[-1] == "":
        edge_lines.pop()
    header_line, body_lines = edge_lines[0], edge_lines[1:]
    native_lines: list[str] = []
    existing_added_line: dict[frozenset[str], str] = {}
    for line in body_lines:
        fields = next(csv.reader([line]))
        key = frozenset({fields[0].strip(), fields[5].strip()})
        if key in deny_keys:
            continue
        if key in added_order:
            existing_added_line[key] = line
        else:
            native_lines.append(line)
    added_lines = [existing_added_line.get(key) or new_edge_line[key]
                   for key, _ in sorted(added_order.items(), key=lambda kv: kv[1])]
    content = bom + (edge_eol.join([header_line, *native_lines, *added_lines]) + edge_eol).encode("utf-8")
    if _write_if_changed(edge_path, content, args.check):
        changed.append(edge_path.name)

    # Water table: every ADM1 row from the water file, in its order; the ADM0
    # roll-ups are derived, not adjudicated, so both provenance columns stay blank.
    water_out = [["adm1", *row] for row in adm1_rows]
    water_out += [["adm0", ia, ib, br, "", "", "adm1-rollup", "", ""]
                  for ia, ib, br in sorted(adm0_rows)]
    if _write_if_changed(water_path, _serialize(water_out, WATER_HEADER, eol=water_eol), args.check):
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
    print(f"stage 3 (water): {len(adm1_rows)} water-only rows (bridge {len(adm1_rows) - nob} / "
          f"no-bridge {nob}); +{n_added3} edge row(s)")
    print(f"stage 4 (edge corrections): +{n_added4} land-gap edge row(s); "
          f"-{len(removed)} denylisted contact(s)")
    print(f"  adm0 roll-up rows: {len(adm0_rows)}; adm0 matrix cells patched: {patched or 'none'}")
    print(f"  files {'needing change' if args.check else 'changed'}: "
          f"{changed or 'none (byte-stable no-op)'}")
    return 2 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
