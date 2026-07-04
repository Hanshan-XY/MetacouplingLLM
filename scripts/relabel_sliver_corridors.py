r"""Source-geometry relabel stage: reassign WB sliver-corridor artifacts.

Some World Bank Admin-1 polygons carry a thin border-tracing "sliver
corridor" -- a ribbon of one unit's territory mislabeled as a neighbour's.
The archetype is ``TZA001`` Arusha, whose polygon grows a ~151 km x ~166 m
tentacle along the Kenya-Tanzania border all the way to Lake Victoria; the
tentacle's land is really Mara (``TZA016``).  These artifacts corrupt the
adjacency graph two ways: they **fabricate** edges (a unit "borders" the
tentacle, not its true owner) and **starve** the real edges the tentacle
blocks (recorded length collapses).

This stage fixes the artifacts *at the source geometry*, before contiguity
runs, so the whole downstream graph is computed from corrected polygons --
one deterministic operation replacing a pile of post-hoc edge patches.

Method (pure geometry; no LLM, no manual step at build time):

1. For each reviewed host polygon (``data/sliver_corridor_relabel.csv``),
   decompose it by a **morphological opening** (``buffer(-D).buffer(+D)``,
   D = 2e-3 deg ~ 220 m): ``main = opening``; ``corridors = polygon - main``
   restricted to parts above a small area floor.
2. Assign each corridor to the reviewed owner unit whose boundary it
   touches (within the build snap tolerance).  Multi-corridor hosts (only
   Arusha) resolve automatically: its NW corridor touches Mara, its E
   corridor touches Kilimanjaro.
3. Rebuild geometries area-conservingly: the host loses exactly the assigned
   corridor polygons (``host - union(corridors)``); each owner gains its
   corridor (``owner + corridor``).

The reviewed list is the authority for *which* polygons are artifacts (that
verdict came from cross-geodata ground-truth, ``build_data/
arusha_sliver_audit/scan_ground_truth.md`` -- three genuine panhandles,
Vennbahn / Courantyne / Dhekelia, were deliberately excluded).  This module
only executes the reassignment the review authorized.

Usage (standalone verification):
  python scripts/relabel_sliver_corridors.py --adm1-gpkg <WB Admin 1 .gpkg>
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import shapely
from pyproj import Geod
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
RELABEL_CSV = REPO / "src" / "metacouplingllm" / "data" / "sliver_corridor_relabel.csv"
OPENING_D_DEG = 2e-3      # ~220 m: same as the sliver-corridor audit
MIN_AREA_KM2 = 0.5        # ignore opening residue below this (rounding noise)
SNAP_TOL_DEG = 5e-4       # build snap tolerance (corridor-owner touch test)
_GEOD = Geod(ellps="WGS84")


def _area_km2(geom: BaseGeometry) -> float:
    a, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(a) / 1e6


def load_relabel_manifest(path: Path = RELABEL_CSV) -> dict[str, dict]:
    """host_code -> {"owners": [owner_code, ...], "d_deg": float}.

    ``opening_d_deg`` is the per-host morphological-opening radius (the
    smallest that fully detaches that host's corridor; default 2e-3 deg).
    """
    hosts: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            h = r["host_code"].strip()
            hosts.setdefault(h, {"owners": [], "d_deg": OPENING_D_DEG})
            hosts[h]["owners"].append(r["owner_code"].strip())
            if r.get("opening_d_deg", "").strip():
                hosts[h]["d_deg"] = float(r["opening_d_deg"])
    return hosts


def _corridors(geom: BaseGeometry, d_deg: float = OPENING_D_DEG) -> list[BaseGeometry]:
    """Opening decomposition: the thin appendage parts above the area floor."""
    main = geom.buffer(-d_deg).buffer(d_deg)
    residue = geom.difference(main)
    parts = (list(residue.geoms) if residue.geom_type == "MultiPolygon"
             else [] if residue.is_empty else [residue])
    return [p for p in parts if _area_km2(p) > MIN_AREA_KM2]


def relabel(gdf: gpd.GeoDataFrame, code_col: str = "ADM1CD_c",
            manifest: dict[str, list[str]] | None = None,
            verbose: bool = False) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """Return (relabeled copy of gdf, per-corridor reassignment log).

    Pure geometry; conserves total area (corridors are moved, not deleted).
    """
    manifest = manifest or load_relabel_manifest()
    gdf = gdf.copy()
    geom_col = gdf.geometry.name
    idx = {c: i for i, c in enumerate(gdf[code_col])}
    log: list[dict] = []

    for host, spec in manifest.items():
        owners = spec["owners"]
        if host not in idx:
            raise SystemExit(f"relabel: host {host} not in layer")
        hi = idx[host]
        host_geom = gdf.geometry.iloc[hi]
        owner_geoms = {o: gdf.geometry.iloc[idx[o]] for o in owners if o in idx}
        missing = [o for o in owners if o not in idx]
        if missing:
            raise SystemExit(f"relabel: owner(s) {missing} for host {host} not in layer")

        moved = defaultdict(list)   # owner_code -> [corridor geoms]
        for c in _corridors(host_geom, spec["d_deg"]):
            cbuf = c.buffer(1e-4)
            # assign to the reviewed owner whose boundary this corridor touches
            hits = [o for o, g in owner_geoms.items()
                    if g.boundary.dwithin(cbuf, SNAP_TOL_DEG)]
            if len(hits) > 1:
                # a corridor touching two reviewed owners is genuinely
                # ambiguous -> stop rather than guess
                raise SystemExit(
                    f"relabel: corridor of {host} (area {_area_km2(c):.1f} km2, "
                    f"bounds {[round(v,3) for v in c.bounds]}) touches multiple "
                    f"reviewed owners {hits}")
            if not hits:
                # a thin part that touches no reviewed owner is not the
                # artifact this fix targets (e.g. an unrelated panhandle) --
                # leave it on the host untouched
                continue
            moved[hits[0]].append(c)
            log.append({"host": host, "owner": hits[0],
                        "corridor_km2": round(_area_km2(c), 2),
                        "bounds": [round(v, 3) for v in c.bounds]})

        # every reviewed owner must actually have received a corridor
        unfilled = [o for o in owners if o not in moved]
        if unfilled:
            raise SystemExit(
                f"relabel: host {host} has no corridor touching reviewed "
                f"owner(s) {unfilled} -- manifest/geometry mismatch")

        # area-conserving rebuild
        all_moved = unary_union([c for cs in moved.values() for c in cs])
        gdf.iat[hi, gdf.columns.get_loc(geom_col)] = host_geom.difference(all_moved)
        for o, cs in moved.items():
            oi = idx[o]
            new = unary_union([gdf.geometry.iloc[oi], *cs])
            gdf.iat[oi, gdf.columns.get_loc(geom_col)] = new
        if verbose:
            for o, cs in moved.items():
                print(f"  {host} -> {o}: moved {len(cs)} corridor(s), "
                      f"{sum(_area_km2(c) for c in cs):.1f} km2")
    return gdf, log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adm1-gpkg", required=True)
    ap.add_argument("--adm1-layer", default="WB_GAD_ADM1")
    args = ap.parse_args(argv)

    print("loading layer...", flush=True)
    gdf = gpd.read_file(args.adm1_gpkg, layer=args.adm1_layer).to_crs(4326)
    bad = ~gdf.geometry.is_valid
    if bad.any():
        gdf.loc[bad, gdf.geometry.name] = gdf.geometry[bad].buffer(0)
    gdf = gdf.reset_index(drop=True)

    before = {c: _area_km2(g) for c, g in zip(gdf["ADM1CD_c"], gdf.geometry)}
    relabeled, log = relabel(gdf, verbose=True)
    after = {c: _area_km2(g) for c, g in zip(relabeled["ADM1CD_c"], relabeled.geometry)}

    print(f"\n{len(log)} corridor(s) reassigned:")
    for e in log:
        print(f"  {e['host']} -> {e['owner']}: {e['corridor_km2']} km2  {e['bounds']}")
    print("\narea change (should be corridor-sized, host loses = owners gain):")
    for c in sorted({e["host"] for e in log} | {e["owner"] for e in log}):
        d = after[c] - before[c]
        if abs(d) > 0.01:
            print(f"  {c}: {before[c]:,.1f} -> {after[c]:,.1f} km2  ({d:+.1f})")
    total = sum(after.values()) - sum(before.values())
    print(f"\ntotal area drift (must be ~0): {total:+.4f} km2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
