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
3. Close each corridor at a junction: where the opening's outline meets the
   host's raw outline it leaves a **cut point** inside a raw segment.  When
   that segment faces a third unit and ends at a junction (a raw vertex where
   the unit across the outline changes), the cut point is replaced by the
   junction, so the corridor carries all of the host's frontage with that
   neighbour or none of it (Salta's corridor otherwise stops 304.2 m short of
   the Potosi-Tarija point, Branicevo's runs 13.8 m past the Caras-Severin-
   Mehedinti point).
4. Node: every remaining cut point is inserted as a vertex into each polygon
   whose outline passes through it (host, owner, the unit across the border),
   so the new three-unit junction is a shared vertex.  Without it the host
   keeps the raw segment beyond the cut point as a zero-width spike, and the
   owner's new frontage matches no neighbour's line exactly (the exact-contact
   measure then gives Kitgum<->Eastern Equatoria 12.92 km for 21.28).
5. Rebuild geometries area-conservingly: the host loses exactly the assigned
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
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
RELABEL_CSV = REPO / "src" / "metacouplingllm" / "data" / "sliver_corridor_relabel.csv"
OPENING_D_DEG = 2e-3      # ~220 m: same as the sliver-corridor audit
MIN_AREA_KM2 = 0.5        # ignore opening residue below this (rounding noise)
SNAP_TOL_DEG = 5e-4       # build snap tolerance (corridor-owner touch test)
NODE_TOL_DEG = 1e-9       # a cut point lies this close to its raw segment (it is ~1e-16 off the line)
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


def _rings(geom: BaseGeometry):
    for p in getattr(geom, "geoms", [geom]):
        yield p.exterior
        yield from p.interiors


def _cut_points(corridor: BaseGeometry, host_geom: BaseGeometry) -> list[tuple]:
    """The corridor's vertices on the host's outline that are not raw vertices of the host."""
    raw = {v for r in _rings(host_geom) for v in r.coords}
    return [v for r in _rings(corridor) for v in r.coords[:-1]
            if v not in raw and host_geom.boundary.distance(Point(v)) < NODE_TOL_DEG]


def _close_at_junctions(corridor, host_i, owner_is, geoms, tree):
    """Replace each cut point that falls on a raw segment facing a third unit and ending at a junction by that junction.

    ``geoms``: the layer's polygons (``tree`` indexes them); ``host_i`` / ``owner_is``: the indexes of the host and of its
    reviewed owners (a cut on the host<->owner line is left alone).  Returns the corridor and [(cut, junction, metres)].
    """
    host_geom = geoms[host_i]
    segs = [(a, b) for r in _rings(host_geom) for a, b in zip(r.coords[:-1], r.coords[1:])]

    def across(pt):  # the other units whose outline passes through pt
        q = Point(pt)
        return {int(j) for j in tree.query(q, predicate="dwithin", distance=NODE_TOL_DEG)
                if int(j) != host_i and geoms[int(j)].boundary.distance(q) < NODE_TOL_DEG}

    cuts = set(_cut_points(corridor, host_geom))
    closed, out = [], []
    for v in corridor.exterior.coords[:-1]:
        if v in cuts:
            a, b = next(s for s in segs if LineString(s).distance(Point(v)) < NODE_TOL_DEG)
            xv = across(v)
            ends = [] if xv & owner_is else [e for e in (a, b) if across(e) - xv]
            if len(ends) > 1:
                raise SystemExit(f"relabel: cut point {v} lies on a segment between two junctions")
            if ends:
                closed.append((v, ends[0], _GEOD.inv(v[0], v[1], ends[0][0], ends[0][1])[2]))
                v = ends[0]
        if not out or out[-1] != v:
            out.append(v)
    if out[0] == out[-1]:
        out.pop()
    return Polygon(out, [r.coords for r in corridor.interiors]), closed


def _insert_vertices(geom: BaseGeometry, pts: list[tuple]) -> BaseGeometry:
    """``geom`` with every point of ``pts`` that lies inside one of its segments inserted there as a vertex."""
    def ring(coords):
        out = [coords[0]]
        for a, b in zip(coords[:-1], coords[1:]):
            seg = LineString([a, b])
            on = [p for p in pts if p != a and p != b and seg.distance(Point(p)) < NODE_TOL_DEG]
            out += sorted(on, key=lambda p: (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) + [b]
        return out
    parts = [Polygon(ring(list(p.exterior.coords)), [ring(list(r.coords)) for r in p.interiors])
             for p in getattr(geom, "geoms", [geom])]
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


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

        geoms = list(gdf.geometry)
        tree = STRtree(geoms)
        owner_is = {idx[o] for o in owners}
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
            c, closed = _close_at_junctions(c, hi, owner_is, geoms, tree)
            if not c.is_valid:
                raise SystemExit(f"relabel: corridor of {host} invalid after closing it at a junction")
            moved[hits[0]].append(c)
            log.append({"host": host, "owner": hits[0],
                        "corridor_km2": round(_area_km2(c), 2),
                        "bounds": [round(v, 3) for v in c.bounds],
                        "closed_at_junction_m": [round(m, 1) for _, _, m in closed]})

        # every reviewed owner must actually have received a corridor
        unfilled = [o for o in owners if o not in moved]
        if unfilled:
            raise SystemExit(
                f"relabel: host {host} has no corridor touching reviewed "
                f"owner(s) {unfilled} -- manifest/geometry mismatch")

        # node: each cut point becomes a vertex of every polygon whose outline
        # passes through it, so host, owner and neighbour share the new junction
        cuts = [v for cs in moved.values() for c in cs for v in _cut_points(c, host_geom)]
        hit = {int(j) for v in cuts for j in tree.query(Point(v), predicate="dwithin", distance=NODE_TOL_DEG)
               if geoms[int(j)].boundary.distance(Point(v)) < NODE_TOL_DEG}
        for j in sorted(hit):
            gdf.iat[j, gdf.columns.get_loc(geom_col)] = _insert_vertices(geoms[j], cuts)

        # area-conserving rebuild
        all_moved = unary_union([c for cs in moved.values() for c in cs])
        gdf.iat[hi, gdf.columns.get_loc(geom_col)] = gdf.geometry.iloc[hi].difference(all_moved)
        for o, cs in moved.items():
            oi = idx[o]
            new = unary_union([gdf.geometry.iloc[oi], *cs])
            gdf.iat[oi, gdf.columns.get_loc(geom_col)] = new
        if verbose:
            for o, cs in moved.items():
                print(f"  {host} -> {o}: moved {len(cs)} corridor(s), "
                      f"{sum(_area_km2(c) for c in cs):.1f} km2")
            for e in log:
                if e["host"] == host and e["closed_at_junction_m"]:
                    print(f"  {host} -> {e['owner']}: corridor closed at a junction "
                          f"{e['closed_at_junction_m']} m from its cut")
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
