r"""Apply the land-gap overlay: dry-land borders the offset corridor hides.

Along straight-surveyed borders the two countries' polygons can be digitized
from different renderings of the same line, leaving an offset corridor wider
than the ~55 m snap tolerance, so a genuine land border drops out of the strict
rook-contiguity build.  The snap-tolerance extras audit
(``build_data/snap_extras_audit/``) surfaced two such cases on the
Kenya-Tanzania survey line, both confirmed by human map review:

  * Kajiado (KEN010) <-> Kilimanjaro (TZA011), the Loitokitok-Rombo sector
  * Narok (KEN033) <-> Mara (TZA016), the Maasai Mara-Serengeti sector

Unlike the river-gap / lake overlays these are ordinary LAND borders: the
restored edges are pericoupled under every ``coupling_standard`` and touch no
water manifest.  Manifest: ``land_gap_overlay_pairs.csv``.  Idempotent.
"""
from __future__ import annotations
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "src" / "metacouplingllm" / "data"
EDGE_LIST = DATA / "pericoupled_adm1_edge_list.csv"
MANIFEST = DATA / "land_gap_overlay_pairs.csv"


def main() -> None:
    manifest = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8-sig")))

    iso_region: dict[str, str] = {}
    iso_country: dict[str, str] = {}
    existing: set[frozenset[str]] = set()
    with open(EDGE_LIST, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            existing.add(frozenset({r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()}))
            iso_region.setdefault(r["ISO_A3_A"].strip(), r["WB_region_A"].strip())
            iso_region.setdefault(r["ISO_A3_B"].strip(), r["WB_region_B"].strip())
            iso_country.setdefault(r["ISO_A3_A"].strip(), r["country_A"].strip())
            iso_country.setdefault(r["ISO_A3_B"].strip(), r["country_B"].strip())

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
    print(f"land-gap overlay applied: +{added} adm1 edge(s)")


if __name__ == "__main__":
    main()
