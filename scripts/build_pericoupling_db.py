#!/usr/bin/env python
"""Build pericoupling adjacency databases from World Bank Official Boundaries.

Regenerates the two bundled adjacency datasets used by ``metacouplingllm``:

  * ``pericoupled_adm1_edge_list.csv`` — ADM1 (subnational) land-border edges
  * ``PeriTelecoupling_clean.csv``     — ADM0 (country) land-border matrix

Method (land-border definition — PR #50)
----------------------------------------
1. Load the WB admin layer + the WB ocean mask (same vintage, so the clip
   aligns with the boundaries — no cross-dataset mismatch).
2. Clip each polygon to land by subtracting the ocean mask.
3. Two regions are adjacent iff their land polygons share a boundary
   (shared-boundary length > 0, with a small snapping tolerance to bridge
   clip-induced slivers).
4. **Lake filter** — subtract Natural Earth lake shores; a pair whose shared
   boundary is *only* lake-shore (Great Lakes, Caspian, …) is dropped.
5. **River measurement** — subtract Natural Earth river buffers to measure the
   *true* land-border length.  Rivers are FLAG-ONLY; pairs are never dropped
   because of a river border (bridges/crossings are real coupling interfaces).
6. Border length is measured in **kilometres** (geodesic, ``pyproj.Geod``) —
   not degrees — so the ``narrow_border`` / ``potential_artifact`` flags mean
   the same physical length at every latitude.

Flags (advisory only — never remove a pair):
  ``narrow_border``       : true land border < 5 km
  ``potential_artifact``  : true land border < 1 km

Sources (record in data/PROVENANCE.md with checksums):
  * World Bank Official Boundaries (GeoPackage, 2026-05-14)
  * Natural Earth 10m lakes + rivers_lake_centerlines

Usage
-----
    python scripts/build_pericoupling_db.py \
        --adm1-gpkg "<...>/World Bank Official Boundaries - Admin 1 (1).gpkg" \
        --adm0-gpkg "<...>/World Bank Official Boundaries - Admin 0 (1).gpkg" \
        --ocean-gpkg "<...>/World Bank Official Boundaries - Ocean Mask.gpkg" \
        --out-dir src/metacouplingllm/data \
        --ne-cache build_data/naturalearth

Run with ``--dry-run`` to write to a scratch dir for diffing before replacing
the shipped CSVs.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Geod
from shapely import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

SNAP_TOL_DEG = 5e-4          # ~55 m: bridge clip-induced gaps in shared borders
NARROW_KM = 5.0              # narrow_border flag threshold
ARTIFACT_KM = 1.0           # potential_artifact flag threshold
RIVER_BUFFER_DEG = 2e-3     # ~220 m buffer around river centerlines (flag calc)
LAKE_SHORE_BUFFER_DEG = 1e-3  # ~110 m: treat shared border within this of a
                              # lake polygon boundary as "lake shore"

_GEOD = Geod(ellps="WGS84")

# ADM0 disputed-border allowlist (PR #50).  The standard 264-unit WB ADM0
# layer excludes the NDLSA disputed-areas layer, which drops a few real,
# well-established international land borders that run through contested
# tracts.  Rather than fold the overlapping NDLSA polygons in (non-standard
# ISO codes, double-counting), we re-add these specific pairs explicitly.
# Each entry is a frozenset of ISO-3 codes.  Verified missing-in-new /
# present-in-old and confirmed as genuine land borders:
#   CHN/PAK — Khunjerab Pass / Gilgit-Baltistan (Kashmir)
#   ISR/SYR — Golan Heights
# (India already retains the Kashmir geometry, so IND/PAK and IND/CHN are
# present without a patch; ARE/QAT is correctly absent — no shared land
# border; Western Sahara ESH/* pairs are superseded by Morocco's borders.)
_ADM0_DISPUTED_ALLOWLIST: set[frozenset[str]] = {
    frozenset({"CHN", "PAK"}),
    frozenset({"ISR", "SYR"}),
}

# Natural Earth 10m downloads (used only when --ne-cache has no local copy)
_NE_LAKES_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_lakes.zip"
)
_NE_RIVERS_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_physical/"
    "ne_10m_rivers_lake_centerlines.zip"
)


def log(msg: str) -> None:
    print(f"[build_pericoupling_db] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _make_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair invalid geometries (buffer(0) fallback) and drop empties."""
    geom = gdf.geometry
    bad = ~geom.is_valid
    if bad.any():
        log(f"  repairing {int(bad.sum())} invalid geometries")
        gdf = gdf.copy()
        gdf.loc[bad, gdf.geometry.name] = geom[bad].buffer(0)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf


def _geodesic_km(geom: BaseGeometry | None) -> float:
    """Geodesic length (km) of a line/multiline in EPSG:4326."""
    if geom is None or geom.is_empty:
        return 0.0
    try:
        return _GEOD.geometry_length(geom) / 1000.0
    except Exception:
        return 0.0


def _shared_border(a: BaseGeometry, b: BaseGeometry) -> BaseGeometry | None:
    """Return the shared-boundary geometry of two polygons, or None.

    Uses the boundary∩boundary; if that is empty but the polygons lie within
    the snapping tolerance, fall back to the intersection of one boundary with
    the other polygon buffered by ``SNAP_TOL_DEG`` (bridges clip slivers).
    """
    shared = a.boundary.intersection(b.boundary)
    if not shared.is_empty:
        return shared
    if a.distance(b) <= SNAP_TOL_DEG:
        shared = a.boundary.intersection(b.buffer(SNAP_TOL_DEG))
        if not shared.is_empty:
            return shared
    return None


def _clip_to_land(
    gdf: gpd.GeoDataFrame, ocean: BaseGeometry
) -> gpd.GeoDataFrame:
    """Subtract the ocean mask from polygons that intersect it (coastal)."""
    gdf = gdf.copy()
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    touched = set(int(i) for i in tree.query(ocean, predicate="intersects"))
    log(f"  clipping {len(touched)} coastal polygons against ocean mask")
    out = []
    col = gdf.geometry.name
    for i, g in enumerate(geoms):
        if i in touched:
            try:
                g2 = g.difference(ocean)
                g = g2 if (g2 is not None and not g2.is_empty) else g
            except Exception:
                pass
        out.append(g)
    gdf[col] = out
    return _make_valid(gdf)


def _load_ne(url: str, cache_dir: Path, name: str) -> gpd.GeoDataFrame:
    """Load a Natural Earth layer from a local cache or download once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / name
    if local.exists():
        log(f"  Natural Earth: using cached {local}")
        return gpd.read_file(local)
    log(f"  Natural Earth: downloading {url}")
    gdf = gpd.read_file(url)
    try:
        gdf.to_file(local, driver="GPKG")
    except Exception:
        pass
    return gdf.to_crs(4326)


# ---------------------------------------------------------------------------
# Core adjacency build
# ---------------------------------------------------------------------------

def build_edges(
    gdf: gpd.GeoDataFrame,
    code_col: str,
    lakes_gdf: gpd.GeoDataFrame | None,
    rivers_gdf: gpd.GeoDataFrame | None,
) -> list[dict]:
    """Compute land-border adjacency edges for a polygon layer.

    Returns one dict per undirected adjacent pair with shared/true border
    lengths (km) and the lake/river/artifact flags.

    Lake/river handling is spatially indexed: a global union of all Natural
    Earth lakes/rivers is huge and differencing every edge against it is
    prohibitively slow.  Instead we build an STRtree over the lake/river
    geometries and, per edge, only difference against features whose bbox is
    near the (small) shared-border geometry — so most edges (nowhere near
    water) skip the expensive op entirely.
    """
    gdf = gdf.reset_index(drop=True)
    geoms = list(gdf.geometry)
    n = len(geoms)
    log(f"  building adjacency over {n} polygons (STRtree)")
    tree = STRtree(geoms)

    lake_geoms = (
        list(lakes_gdf.geometry) if lakes_gdf is not None else []
    )
    lake_tree = STRtree(lake_geoms) if lake_geoms else None
    river_geoms = (
        list(rivers_gdf.geometry) if rivers_gdf is not None else []
    )
    river_tree = STRtree(river_geoms) if river_geoms else None

    def _drop_water(border: BaseGeometry) -> tuple[BaseGeometry, BaseGeometry]:
        """Return (non_lake_border, true_land_border) for a shared border.

        WB admin polygons INCLUDE lake water (e.g. US states meet down the
        middle of the Great Lakes; Caspian littoral states meet mid-sea), so
        a shared boundary running through a lake interior is a false land
        border.  Subtract the lake **polygon area** (not the shoreline): a
        mid-lake border lies inside the lake polygon and is removed, while a
        real land border that merely runs *along* a lake's edge stays (it's
        outside the polygon).  Pairs whose remaining land border is ~0 are
        dropped by the caller.
        """
        non_lake = border
        if lake_tree is not None:
            near = lake_tree.query(border)
            if len(near):
                lake_area = unary_union([lake_geoms[int(k)] for k in near])
                try:
                    non_lake = border.difference(lake_area)
                except Exception:
                    non_lake = border
        true_land = non_lake
        if river_tree is not None and not non_lake.is_empty:
            near = river_tree.query(non_lake)
            if len(near):
                rbuf = unary_union(
                    [river_geoms[int(k)] for k in near]
                ).buffer(RIVER_BUFFER_DEG)
                try:
                    true_land = non_lake.difference(rbuf)
                except Exception:
                    true_land = non_lake
        return non_lake, true_land

    seen: set[tuple[int, int]] = set()
    edges: list[dict] = []
    for i, g in enumerate(geoms):
        if i and i % 250 == 0:
            log(f"    {i}/{n} polygons, {len(edges)} edges so far")
        cand = tree.query(g.buffer(SNAP_TOL_DEG))
        for jx in cand:
            j = int(jx)
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            h = geoms[j]
            shared = _shared_border(g, h)
            if shared is None:
                continue
            shared_km = _geodesic_km(shared)
            if shared_km <= 0:
                continue

            non_lake, true_land = _drop_water(shared)
            non_lake_km = _geodesic_km(non_lake)
            if non_lake_km <= 0:
                # contact is ONLY lake shore -> not a land border; drop
                continue
            true_km = _geodesic_km(true_land)

            ra, rb = gdf.iloc[i], gdf.iloc[j]
            edges.append({
                "i": i, "j": j,
                "code_a": ra[code_col], "iso_a": ra["ISO_A3"],
                "name_a": ra.get("NAM_1", ra.get("NAM_0", "")),
                "country_a": ra.get("NAM_0", ""),
                "wb_a": ra.get("WB_REGION", ""),
                "code_b": rb[code_col], "iso_b": rb["ISO_A3"],
                "name_b": rb.get("NAM_1", rb.get("NAM_0", "")),
                "country_b": rb.get("NAM_0", ""),
                "wb_b": rb.get("WB_REGION", ""),
                "border_length_km": round(true_km, 4),
                "shared_km": round(non_lake_km, 4),
            })
    log(f"  -> {len(edges)} adjacency edges")
    return edges


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_adm1_csv(edges: list[dict], out_path: Path) -> None:
    cols = [
        "ADM1_code_A", "ADM1_name_A", "country_A", "ISO_A3_A", "WB_region_A",
        "ADM1_code_B", "ADM1_name_B", "country_B", "ISO_A3_B", "WB_region_B",
        "cross_country", "border_length_km", "narrow_border",
        "potential_artifact",
    ]
    rows = []
    for e in edges:
        km = e["border_length_km"]
        rows.append({
            "ADM1_code_A": e["code_a"], "ADM1_name_A": e["name_a"],
            "country_A": e["country_a"], "ISO_A3_A": e["iso_a"],
            "WB_region_A": e["wb_a"],
            "ADM1_code_B": e["code_b"], "ADM1_name_B": e["name_b"],
            "country_B": e["country_b"], "ISO_A3_B": e["iso_b"],
            "WB_region_B": e["wb_b"],
            "cross_country": e["iso_a"] != e["iso_b"],
            "border_length_km": km,
            "narrow_border": km < NARROW_KM,
            "potential_artifact": km < ARTIFACT_KM,
        })
    rows.sort(key=lambda r: (r["ADM1_code_A"], r["ADM1_code_B"]))
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log(f"  wrote {out_path} ({len(rows)} edges)")


def write_adm0_matrix(
    edges: list[dict], iso_codes: list[str], out_path: Path
) -> None:
    """Full directed country matrix: Sending,Receiving,Intracoupling (1=adj)."""
    adj: set[frozenset[str]] = set()
    for e in edges:
        a, b = e["iso_a"], e["iso_b"]
        if a and b and a != b:
            adj.add(frozenset({a, b}))
    codes = sorted(set(iso_codes))
    # PR #50: re-add well-established land borders dropped by excluding the
    # NDLSA disputed-areas layer (only pairs whose BOTH codes exist here).
    for pair in _ADM0_DISPUTED_ALLOWLIST:
        a, b = tuple(pair)
        if a in codes and b in codes:
            adj.add(pair)
            log(f"  +allowlist disputed border: {a}/{b}")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Sending", "Receiving", "Intracoupling"])
        for a in codes:
            for b in codes:
                if a == b:
                    continue
                w.writerow([a, b, 1 if frozenset({a, b}) in adj else 0])
    log(f"  wrote {out_path} ({len(codes)} ISO units, "
        f"{len(adj)} adjacent pairs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adm1-gpkg", required=True)
    ap.add_argument("--adm0-gpkg", required=True)
    ap.add_argument("--ocean-gpkg", required=True)
    ap.add_argument("--adm1-layer", default="WB_GAD_ADM1")
    ap.add_argument("--adm0-layer", default="WB_GAD_ADM0")
    ap.add_argument("--ocean-layer", default="WB_GAD_ocean_mask")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ne-cache", default="build_data/naturalearth")
    ap.add_argument("--skip-ne", action="store_true",
                    help="skip lake/river processing (no NE download)")
    ap.add_argument("--skip-rivers", action="store_true",
                    help="load lakes (needed to drop mid-lake edges) but skip "
                         "the river-buffer flag pass (advisory only)")
    ap.add_argument("--clip-ocean", action="store_true",
                    help="subtract the WB ocean mask before adjacency. OFF by "
                         "default: WB official boundaries are already land-only "
                         "(islands isolate, no maritime adjacencies), and the "
                         "clip is the build's main bottleneck.")
    ap.add_argument("--levels", default="adm0,adm1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = {x.strip() for x in args.levels.split(",")}

    ocean = None
    if args.clip_ocean:
        log("loading ocean mask")
        ocean_gdf = _make_valid(
            gpd.read_file(args.ocean_gpkg, layer=args.ocean_layer).to_crs(4326)
        )
        ocean = unary_union(list(ocean_gdf.geometry))

    lakes_gdf = rivers_gdf = None
    if not args.skip_ne:
        ne = Path(args.ne_cache)
        lakes_gdf = _load_ne(_NE_LAKES_URL, ne, "ne_10m_lakes.gpkg").to_crs(4326)
        if not args.skip_rivers:
            rivers_gdf = _load_ne(
                _NE_RIVERS_URL, ne, "ne_10m_rivers.gpkg"
            ).to_crs(4326)

    def _prep(path: str, layer: str) -> gpd.GeoDataFrame:
        g = _make_valid(gpd.read_file(path, layer=layer).to_crs(4326))
        if ocean is not None:
            g = _clip_to_land(g, ocean)
        return g

    if "adm1" in levels:
        log("=== ADM1 ===")
        a1 = _prep(args.adm1_gpkg, args.adm1_layer)
        e1 = build_edges(a1, "ADM1CD_c", lakes_gdf, rivers_gdf)
        write_adm1_csv(e1, out_dir / "pericoupled_adm1_edge_list.csv")

    if "adm0" in levels:
        log("=== ADM0 ===")
        a0 = _prep(args.adm0_gpkg, args.adm0_layer)
        e0 = build_edges(a0, "ISO_A3", lakes_gdf, rivers_gdf)
        write_adm0_matrix(
            e0, list(a0["ISO_A3"]), out_dir / "PeriTelecoupling_clean.csv"
        )

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
