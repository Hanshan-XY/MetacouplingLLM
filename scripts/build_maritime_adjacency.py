r"""Build the maritime adjacency table: country pairs that share a maritime boundary.

Source: the Maritime Boundaries Geodatabase, version 12, of the Flanders Marine
Institute (VLIZ), distributed by Marine Regions (Flanders Marine Institute, 2023,
https://doi.org/10.14284/632; licence CC BY 4.0).  The inputs are two of its
attribute tables, read without geometry as CSV from the VLIZ WFS service,
pinned by SHA-256 (``PINNED``) and committed in ``build_data/marine_regions/``,
where ``--boundaries`` and ``--eez`` find them by default::

  https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=2.0.0
      &request=GetFeature&outputFormat=csv&typeNames=<layer>&propertyName=<fields>

  --boundaries  MarineRegions:eez_boundaries, BOUNDARY_FIELDS  (2,349 lines)
  --eez         MarineRegions:eez,            EEZ_FIELDS       (285 records)

Rules, applied to every record alike:

1. A boundary line between two territories makes them maritime neighbours,
   whatever its type (treaty, median line, court ruling, joint regime, an
   unsettled line, and so on).
2. An overlapping-claim area counts as bordering for every claimant its record
   lists, and its claimants share it.  A joint-regime area's parties are the
   territories on the lines that carry it (``mrgid_jreg``), compared by
   territory ID; its record's parties stand only when no line carries it.
3. A territory's country is its own ISO 3166-1 code where Marine Regions gives
   one (Greenland, the Faroe Islands, Reunion), else its sovereign's (Alaska,
   Hawaii, the Andaman and Nicobar Islands): the units of the ADM0 database.
4. Units the ADM0 database does not have follow its own rule: a World Bank
   disputed tract goes to the administrator Natural Earth records for it
   (``_NDLSA_TRACT_ADMIN`` in ``scripts/build_pericoupling_db.py``), and a unit
   the World Bank ADM0 layer draws inside another country goes to that country
   (``WB_ADM0``).
5. A pair that exists only through the waters of a disputed tract is marked
   ``disputed_only``; the loaders drop it from the strict World Bank view
   (``de_facto_borders=False``), as they drop the land overlay.
6. Pairs within one country are not pairs.

Output: ``src/metacouplingllm/data/maritime_adjacency_pairs.csv`` with
``code_a, name_a, code_b, name_b, line_types, boundary_km, disputed_only``.
``boundary_km`` sums the lengths Marine Regions records for the lines between
the two countries' own territories; it is 0 for a pair joined only through a
shared area.  Byte-stable (each run writes only if the bytes differ); ``--check``
exits 2 instead of writing.

Usage:  python scripts/build_maritime_adjacency.py [--boundaries PATH] [--eez PATH] [--out PATH] [--check] [--no-verify]
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "metacouplingllm" / "data"
DEFAULT_OUT = DATA / "maritime_adjacency_pairs.csv"
INPUTS = REPO / "build_data" / "marine_regions"
HEADER = ["code_a", "name_a", "code_b", "name_b", "line_types", "boundary_km", "disputed_only"]

PINNED = {
    "boundaries": "7cc30de5e68c238659f43b4e8608550bb001a9e61a2bdcfa65566d0947d39adb",
    "eez": "e68c97e7aa1ebe774a3329f6bf73d16774b9fbfceeaab9b9c4ddddfbe651f187",
}
BOUNDARY_FIELDS = ("line_id,line_name,line_type,mrgid_sov1,mrgid_ter1,territory1,sovereign1,mrgid_ter2,"
                   "territory2,mrgid_sov2,sovereign2,mrgid_eez1,eez1,mrgid_eez2,eez2,origin,doc_date,"
                   "mrgid_jreg,joint_reg,length_km")
EEZ_FIELDS = ("mrgid,geoname,mrgid_ter1,pol_type,mrgid_sov1,territory1,iso_ter1,sovereign1,mrgid_ter2,"
              "mrgid_sov2,territory2,iso_ter2,sovereign2,mrgid_ter3,mrgid_sov3,territory3,iso_ter3,"
              "sovereign3,mrgid_eez,area_km2,iso_sov1,iso_sov2,iso_sov3")

# Marine Regions territories that are World Bank disputed tracts -> the tract's name in _NDLSA_TRACT_ADMIN
DISPUTED_TRACTS = {
    "Western Sahara": "Western Sahara",
    "Falkland / Malvinas Islands": "Falkland Islands",
    "South Georgia and the South Sandwich Islands": "South Georgia and South Sandwich Islands",
    "Chagos Archipelago": "British Indian Ocean Territory",
}
# Units the World Bank ADM0 layer draws inside another country's polygon
WB_ADM0 = {"TWN": "CHN"}


def _tract_admins() -> dict[str, str | None]:
    """The land database's administrator of each disputed tract, read without importing the build."""
    src = (REPO / "scripts" / "build_pericoupling_db.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "_NDLSA_TRACT_ADMIN")
    return ast.literal_eval(node.value)


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _adm0_codes() -> set[str]:
    codes: set[str] = set()
    for row in _read(DATA / "PeriTelecoupling_clean.csv"):
        codes |= {row["Sending"].strip(), row["Receiving"].strip()}
    return codes


def build(boundaries: list[dict], eez: list[dict]) -> dict[tuple[str, str], dict]:
    """Country pairs with their line types, boundary length and disputed-only flag."""
    admins = _tract_admins()
    tract_country = {t: admins[tract] for t, tract in DISPUTED_TRACTS.items()}
    missing = [t for t, c in tract_country.items() if not c]
    if missing:
        raise SystemExit(f"no administrator recorded for the tracts of {missing}")

    # IDs to names and ISO codes, ordinary 200 NM records first: claim records reuse their claimants' IDs.
    ter_name: dict[str, str] = {}
    ter_iso: dict[str, str] = {}
    for r in sorted(eez, key=lambda r: r["pol_type"] != "200NM"):
        for i in "123":
            tid = r["mrgid_ter" + i]
            if tid:
                ter_name.setdefault(tid, r["territory" + i])
                ter_iso.setdefault(tid, r["iso_ter" + i] or r["iso_sov" + i])

    def country(territory: str, iso: str) -> str:
        if territory in tract_country:
            return tract_country[territory]
        return WB_ADM0.get(iso, iso)

    parties = {r["mrgid_eez"]: [(r["territory" + i], country(r["territory" + i], r["iso_ter" + i] or r["iso_sov" + i]))
                                for i in "123" if r["territory" + i]] for r in eez}
    line_tids: dict[str, set[str]] = defaultdict(set)
    for r in boundaries:
        if r["mrgid_jreg"]:
            line_tids[r["mrgid_jreg"]] |= {r["mrgid_ter" + i] for i in "12" if r["mrgid_ter" + i]}
    records = {r["mrgid_eez"]: r for r in eez}
    for area, tids in line_tids.items():
        rec = records.get(area)
        if rec is None or tids != {rec["mrgid_ter" + i] for i in "123" if rec["mrgid_ter" + i]}:
            parties[area] = [(ter_name[t], country(ter_name[t], ter_iso[t])) for t in sorted(tids)]

    pairs: dict[tuple[str, str], dict] = defaultdict(lambda: {"types": set(), "km": 0.0, "via": set()})

    def add(p: tuple[str, str], q: tuple[str, str], kind: str, km: float) -> None:
        (t1, c1), (t2, c2) = p, q
        if not c1 or not c2 or c1 == c2:
            return
        e = pairs[tuple(sorted((c1, c2)))]
        e["types"].add(kind)
        e["km"] += km
        e["via"].add(bool({t1, t2} & set(DISPUTED_TRACTS)))

    for r in boundaries:
        if not (r["territory1"] and r["territory2"]):
            continue
        own = []
        for i in "12":
            tid = r["mrgid_ter" + i]
            t = ter_name.get(tid, r["territory" + i])
            own.append((t, country(t, ter_iso.get(tid, ""))))
        add(own[0], own[1], r["line_type"], float(r["length_km"] or 0))
        for p in parties.get(r["mrgid_eez1"], []) + [own[0]]:
            for q in parties.get(r["mrgid_eez2"], []) + [own[1]]:
                if (p, q) != (own[0], own[1]):
                    add(p, q, r["line_type"], 0.0)
    for r in eez:
        ps = parties[r["mrgid_eez"]]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                add(ps[i], ps[j], r["pol_type"] + " area", 0.0)
    return dict(pairs)


def _detect_eol(path: Path, default: str = "\r\n") -> str:
    """Match the file's line ending (CRLF on autocrlf checkouts, LF on Linux/CI)."""
    if not path.exists():
        return default
    return "\r\n" if b"\r\n" in path.read_bytes()[:4096] else "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundaries", type=Path, default=INPUTS / "mr_eez_boundaries_v12.csv")
    ap.add_argument("--eez", type=Path, default=INPUTS / "mr_eez_v12.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="exit 2 if the output would change; write nothing")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check of the inputs")
    args = ap.parse_args(argv)
    for key, path in (("boundaries", args.boundaries), ("eez", args.eez)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != PINNED[key] and not args.no_verify:
            raise SystemExit(f"{path}: SHA-256 {digest} is not the pinned {PINNED[key]} (--no-verify to override)")

    pairs = build(_read(args.boundaries), _read(args.eez))
    unknown = sorted({c for pair in pairs for c in pair} - _adm0_codes())
    if unknown:
        raise SystemExit(f"country codes the ADM0 database does not have: {unknown}")

    sys.path.insert(0, str(REPO / "src"))
    from metacouplingllm.knowledge.countries import get_country_name

    rows = [[a, get_country_name(a), b, get_country_name(b), "; ".join(sorted(e["types"])),
             f"{e['km']:.1f}", str(e["via"] == {True})] for (a, b), e in sorted(pairs.items())]
    eol = _detect_eol(args.out)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=eol)
    w.writerow(HEADER)
    w.writerows(rows)
    content = buf.getvalue().encode("utf-8")
    changed = not args.out.exists() or args.out.read_bytes() != content
    if changed and not args.check:
        args.out.write_bytes(content)
    n_disputed = sum(1 for r in rows if r[-1] == "True")
    print(f"{len(rows)} country pairs ({n_disputed} only through disputed waters); "
          f"{'would change' if args.check and changed else 'written' if changed else 'unchanged'}: {args.out}")
    return 2 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
