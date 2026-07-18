#!/usr/bin/env python
"""Build pericoupling adjacency databases from World Bank Official Boundaries.

Regenerates the bundled adjacency datasets used by ``metacouplingllm``:

  * ``pericoupled_adm1_edge_list.csv`` — ADM1 (subnational) border edges
  * ``PeriTelecoupling_clean.csv``     — ADM0 (country) border matrix
  * ``water_separated_pairs.csv``      — water-only flags (with ``--bridge-csv``)
  * ``disputed_overlay_pairs.csv``     — de-facto overlay manifest (with
    ``--ndlsa-gpkg``)

Method (full detail: ``docs/METHODS_adjacency.md``)
---------------------------------------------------
1. Load the WB admin layer + the WB ocean mask (same vintage, so the clip
   aligns with the boundaries — no cross-dataset mismatch); clip each polygon
   to land by subtracting the ocean mask (``--clip-ocean``).
2. **Source-relabel** — reassign the reviewed WB sliver-corridor artifacts to
   their true owner units before contiguity
   (``scripts/relabel_sliver_corridors.py`` + the reviewed manifest
   ``data/sliver_corridor_relabel.csv``).
3. **Exact-contact rook contiguity** (``TOPOLOGY_TOL_DEG = 0``): two units are
   adjacent iff their boundaries share a segment of non-zero geodesic length.
   No snapping tolerance, no lake filter (pairs meeting across a lake are
   native edges; ``coupling_standard`` governs them downstream).  The reviewed
   ``_ADM1_FALSE_POSITIVE_DENYLIST`` drops confirmed fake-touch edges.
4. **De-facto disputed overlay** — derived from the WB NDLSA disputed-areas
   layer and validated against its geometry at build time.
5. ``border_length_km`` is the **full** shared-boundary length in kilometres
   (geodesic, ``pyproj.Geod``) — no river/lake subtraction; water status is a
   separate descriptive layer (``water_separated_pairs.csv``) that never adds
   or drops an edge.

The reviewed correction overlays (river-gap, lake-gap, land-gap, and the
flags-only water overlays) are applied afterwards by
``scripts/apply_overlays.py``; ``scripts/build_all.py`` orchestrates both
steps and verifies every headline count.

Flags (advisory only — never remove a pair):
  ``narrow_border``       : shared border < 5 km
  ``potential_artifact``  : shared border < 1 km

Sources (recorded in data/PROVENANCE.md with checksums):
  * World Bank Official Boundaries (GeoPackage, 2026-05-14)

Usage
-----
    python scripts/build_pericoupling_db.py \
        --adm1-gpkg "<...>/World Bank Official Boundaries - Admin 1 (1).gpkg" \
        --adm0-gpkg "<...>/World Bank Official Boundaries - Admin 0 (1).gpkg" \
        --ocean-gpkg "<...>/World Bank Official Boundaries - Ocean Mask.gpkg" \
        --ndlsa-gpkg "<...>/World Bank Official Boundaries - NDLSA (1).gpkg" \
        --bridge-csv build_data/bridge_classified_authoritative.csv \
        --out-dir src/metacouplingllm/data --clip-ocean
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
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# The main ADM1/ADM0 topology uses EXACT contact (tolerance 0) -- a
# parameter-free geometry core.  Every deviation from exact geometry that a
# non-zero snap would bridge is instead a reviewed manifest row (band-audit
# pairs) or a source-relabel (sliver corridors).  SNAP_TOL_DEG is retained
# ONLY for the disputed-overlay derivation's tract-touch validation, where a
# small tolerance is needed to bridge the NDLSA carve-out gaps.
TOPOLOGY_TOL_DEG = 0.0       # exact-contact main topology (parameter-free)
SNAP_TOL_DEG = 5e-4          # disputed-overlay tract-touch tolerance only
NARROW_KM = 5.0              # narrow_border flag threshold
ARTIFACT_KM = 1.0           # potential_artifact flag threshold

_GEOD = Geod(ellps="WGS84")

# De-facto administrator map for the NDLSA disputed-areas tracts.
#
# The standard WB ADM0 (264-unit) and ADM1 (3,591-unit) layers EXCLUDE the
# 24-feature NDLSA disputed-areas layer, carving each contested tract out of
# *both* neighbours and opening a gap — so the flanking units are recorded as
# non-adjacent even where they meet across the de-facto line of control.  The
# de-facto view (default) re-adds those borders by folding each tract into its
# de-facto administering country; the resulting overlay pairs are *derived from
# geometry* by ``derive_disputed_overlay`` (not hand-listed), so a mis-labelled
# tract fails loudly instead of silently dropping a pair.
#
# IMPORTANT — these tract→administrator assignments are AUTHORED.  The NDLSA
# layer carries NO administering-country field (``SOVEREIGN`` is null for all 24
# tracts; ``WB_STATUS`` is uniformly "Non-determined legal status area"), so each
# mapping reflects *de-facto control* for a connectivity dataset and is NOT a
# legal or endorsed sovereignty claim.  ``None`` = no single de-facto
# administrator (islands, no-man's-lands, UN zones); such tracts are not folded.
# Keyed by the tract's ``NAM_0`` (note "Kauirik" carries a stray newline in the
# source, normalised on load).
_NDLSA_TRACT_ADMIN: dict[str, str | None] = {
    # India–China LoAC tracts (already-adjacent ISO pair → produce no new pair):
    "Aksai Chin": "CHN", "Kauirik": "CHN", "Lapthal": "CHN", "Shipki Pass": "CHN",
    "Chumar East": "IND", "Chumar West": "IND", "Demchok": "IND",
    "Jadh Ganga Valley": "IND", "Arunachal Pradesh": "IND",
    "Jammu and Kashmir": "IND", "Kalapani": "IND", "Doklam": "BTN",
    # Pair-producing tracts (sole land link between the two flanking countries):
    "Gilgit Baltistan": "PAK", "Karakoram Range": "PAK",   # → CHN/PAK
    "Golan Heights": "ISR", "Shebaa Farms Dispute": "ISR",  # → ISR/SYR
    "Western Sahara": "MAR",                                # → MAR/MRT
    # Ilemi Triangle: de-facto Kenya, but Kenya and South Sudan ALREADY share an
    # ~80 km border in the standard layer (SW of the tract), so folding it adds
    # no new pair — only lengthens an existing border. Left assigned to KEN; the
    # geometric derivation correctly emits no overlay pair for it.
    "Ilemi Triangle": "KEN",
    # No single de-facto administrator → not folded:
    "Abyei": None, "No Man's Land": None, "UN Buffer Zone": None,
    "British Indian Ocean Territory": None,
    "South Georgia and South Sandwich Islands": None, "Falkland Islands": None,
}

# De-facto administering ADM1 PROVINCE(s) per tract, for the subnational overlay.
# Authored (same neutral-framing caveat as _NDLSA_TRACT_ADMIN above) and
# GEOMETRY-VALIDATED: derive_disputed_overlay raises if a listed province does
# not touch its tract.  A tract may list SEVERAL provinces — a large tract spans
# more than one (e.g. Western Sahara) — and the tract↔neighbour frontier is split
# among them by nearest province.  An EMPTY list means the de-facto administering
# sub-unit is NOT a WB ADM1 province and the relationship is carried at ADM0 only:
# Gilgit-Baltistan and Ladakh/Jammu & Kashmir are disputed territories EXCLUDED
# from WB's standard layer, not provinces (verified — Pakistan has only 5 WB ADM1
# units: Balochistan, Federal Capital Territory, Khyber Pakhtunkhwa, Punjab,
# Sindh; none is Gilgit-Baltistan), so attributing them to the nearest province
# (Khyber Pakhtunkhwa / Himachal Pradesh) would be geographically false.  Tracts
# whose flanking provinces are ALREADY adjacent in the strict layer are listed
# too (completeness/auditability); the geometric derivation emits no row for them.
# Every non-None _NDLSA_TRACT_ADMIN tract MUST appear here (completeness guard).
_NDLSA_TRACT_ADM1: dict[str, list[str]] = {
    # Pair-producing — the tract is the sole subnational land link:
    "Golan Heights": ["ISR004"],          # Northern District administers the Golan
    "Shebaa Farms Dispute": ["ISR004"],
    "Arunachal Pradesh": ["IND003"],      # the state itself — NOT Assam (which is
                                          # merely the nearest non-tract polygon)
    "Doklam": ["BTN005"],                 # Haa Dzongkhag
    "Western Sahara": ["MAR005", "MAR007"],  # Guelmim-Oued Noun + Laâyoune-Sakia
                                          # al Hamra (Moroccan "Southern Provinces")
    # Already-adjacent flanks (no new overlay row — listed for completeness):
    "Aksai Chin": ["CHN029", "CHN028"], "Kauirik": ["CHN029"],
    "Lapthal": ["CHN029"], "Shipki Pass": ["CHN029"],
    "Jadh Ganga Valley": ["IND035", "IND014"], "Kalapani": ["IND035"],
    "Ilemi Triangle": ["KEN043"],
    # De-facto admin sub-unit is NOT a WB ADM1 province → ADM0-only (empty):
    "Gilgit Baltistan": [], "Karakoram Range": [],
    "Jammu and Kashmir": [], "Chumar East": [], "Chumar West": [], "Demchok": [],
}

# Frontier-sampling step (~1.1 km at the equator) used to split a tract↔neighbour
# border among the authored admin provinces by nearest province (ADM1 overlay).
_ADM1_SAMPLE_DEG = 0.01

# ADM1 false-positive denylist.  The snapping tolerance (SNAP_TOL_DEG) bridges
# sub-tolerance gaps between independently-digitised polygons, which is correct
# for genuine borders drawn with a small cross-source offset but occasionally
# fabricates an edge between two units that do not actually share a frontier.
# These pairs were manually verified (against imagery) to be NON-adjacent — the
# polygons are separated by a gap the tolerance spuriously closes — and are
# removed from the ADM1 edge list.  Each entry is a frozenset of ADM1CD_c codes.
# Verified not-adjacent:
#   MLT002/MLT019 — Balzan / Iklin (Malta): ~31 m apart, no shared frontier.
# The snap-bridged class is EMPTY under tolerance-0 topology.  The Malta
# artifact (MLT002/MLT019, ~31 m apart) only appears when a non-zero snap
# bridges the gap; at exact contact it never becomes an edge, so no removal is
# needed.  The five genuine sub-tolerance borders the old snap recovered
# (Egypt-Libya etc.) now ship as reviewed land-gap manifest rows instead.
#
# A SECOND, distinct class can survive at tolerance 0: exact-contact
# exclave-simplification slivers.  Where a fragmented territory has a compact
# exclave, the source's polygon generalisation can make that exclave share a
# short frontier with a NON-neighbour across a narrow corridor that a third
# same-country unit actually occupies.  Like the cross-country sliver corridors
# of §10 this is a fake *touch* with no mislabelled ribbon to reassign, so it is
# removed here rather than source-relabelled — but ONLY when it carries the
# thin-sliver signature AND is confirmed non-adjacent on an authoritative map.
# A geometry/OSM adjacency verdict alone is NOT sufficient to drop a WB edge:
# an OSM or gazetteer neighbour list describes a different boundary dataset, and
# a substantial WB frontier at a tripoint is usually genuine (see §10.6 and
# build_data/arusha_sliver_audit/domestic_scan_ground_truth.md for the three
# candidates — Dubai/Fujairah, Chaguanas/San Juan-Laventille, Chișinău/Dubăsari —
# that were screened this way but confirmed GENUINE on the official maps and are
# therefore NOT denylisted).
#
# Verified not-adjacent (denylisted):
#   LIE001/LIE005 — Balzers/Planken (Liechtenstein): a 0.42 km sliver; the
#     communes do not share a frontier — Schaan/Vaduz/Triesenberg lie between,
#     Balzers's alpine exclaves sit in the upper Valorsch/Gapfahl zone and
#     Planken's Garselli on the lower Saminatal (Historisches Lexikon).
#   LBR006/LBR014 — Grand Gedeh/Rivercess (Liberia): the four-county corner at
#     the Cestos–Gwen Creek confluence is an exact quadripoint (OSM: one shared
#     node; GADM 4.1: point intersection); the WB's 1.12 km contact is two
#     straight cardinal construction legs bridging offset river-boundary
#     termini — a sliver wedge, not a border (docs/FUTURE_EDGE_AUDITS.md #7;
#     maintainer removal decision 2026-07-18).
#   VEN001/VEN003 — Apure/Amazonas (Venezuela): Amazonas' territorial-division
#     law (Art. 59 constitution, 1994 DPT law) enumerates its perimeter with no
#     Apure segment — Bolívar's east-bank Orinoco frontage plus Colombia
#     separate the states; the WB's 2.35 km contact is a mid-river seam where
#     Bolívar's sliver frontage was dropped (docs/FUTURE_EDGE_AUDITS.md #8;
#     maintainer removal decision 2026-07-18).
_ADM1_FALSE_POSITIVE_DENYLIST: set[frozenset[str]] = {
    frozenset({"LIE001", "LIE005"}),
    frozenset({"LBR006", "LBR014"}),
    frozenset({"VEN001", "VEN003"}),
}

# Reviewed ADM1 unit merges, applied before contiguity (source-data artifacts
# where the WB/GAUL lineage splits one real unit into two). RUS050 ("Name
# Unknown", GAUL_1 2537) is the western salient of the Republic of Kalmykia —
# the Gorodovikovsky + Yashaltinsky raions (district capitals geocode inside
# it; areas match; 1943-57 deportation-era transfer explains the upstream
# split) — merged into RUS024 so the internal raion line stops shipping as an
# ADM1 edge and the salient's Rostov/Stavropol frontages accrue to Kalmykia
# (docs/FUTURE_EDGE_AUDITS.md #10; maintainer merge decision 2026-07-18).
_ADM1_UNIT_MERGES: dict[str, str] = {
    "RUS050": "RUS024",
}

# Populated by ``derive_disputed_overlay`` (geometry-derived from the NDLSA
# tracts + _NDLSA_TRACT_ADMIN).  _ADM0_DISPUTED_ALLOWLIST: set of ISO frozensets
# re-added to the ADM0 matrix; _ADM1_DISPUTED_OVERLAY: list of full ADM1 edge
# rows appended to the edge list; _DISPUTED_OVERLAY_MANIFEST: the shipped sidecar
# rows (both levels).  Empty until derive_disputed_overlay() runs.
_ADM0_DISPUTED_ALLOWLIST: set[frozenset[str]] = set()
_ADM1_DISPUTED_OVERLAY: list[dict] = []
_DISPUTED_OVERLAY_MANIFEST: list[dict] = []


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


def _shared_border(a: BaseGeometry, b: BaseGeometry,
                   tol: float = SNAP_TOL_DEG) -> BaseGeometry | None:
    """Return the shared-boundary geometry of two polygons, or None.

    Uses the boundary∩boundary; if that is empty but the polygons lie within
    ``tol``, fall back to the intersection of one boundary with the other
    polygon buffered by ``tol``.  At ``tol=0`` (the main topology) only the
    exact boundary∩boundary is used.
    """
    shared = a.boundary.intersection(b.boundary)
    if not shared.is_empty:
        return shared
    if tol > 0 and a.distance(b) <= tol:
        shared = a.boundary.intersection(b.buffer(tol))
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


# ---------------------------------------------------------------------------
# Core adjacency build
# ---------------------------------------------------------------------------

def build_edges(gdf: gpd.GeoDataFrame, code_col: str) -> list[dict]:
    """Compute shared-border adjacency edges for a polygon layer (Stage 1).

    Pure World Bank geometry: rook contiguity at **exact contact**
    (``TOPOLOGY_TOL_DEG = 0``) -- two units are adjacent iff their boundaries
    share a segment of non-zero geodesic length.  ``border_length_km`` is the
    **full** shared-boundary length (no Natural Earth lake/river subtraction);
    the water classification of a border is a separate, purely descriptive
    layer (Stage 3, ``water_separated_pairs.csv``) that never adds or drops an
    edge.  Because WB admin polygons include lake water, pairs that meet across
    a lake are native edges here (e.g. the Great Lakes and Lake-Tanganyika
    pairs) -- no lake filter removes them, so ``coupling_standard`` governs
    lakes and rivers uniformly with no restoration overlay.
    """
    gdf = gdf.reset_index(drop=True)
    geoms = list(gdf.geometry)
    n = len(geoms)
    log(f"  building adjacency over {n} polygons (STRtree, exact contact)")
    tree = STRtree(geoms)

    seen: set[tuple[int, int]] = set()
    edges: list[dict] = []
    for i, g in enumerate(geoms):
        if i and i % 250 == 0:
            log(f"    {i}/{n} polygons, {len(edges)} edges so far")
        for jx in tree.query(g):          # bbox candidates (exact test below)
            j = int(jx)
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            shared = _shared_border(g, geoms[j], tol=TOPOLOGY_TOL_DEG)
            if shared is None:
                continue
            km = _geodesic_km(shared)
            if km <= 0:                   # single-vertex/point contact -> not an edge
                continue

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
                "border_length_km": round(km, 4),
                "shared_km": round(km, 4),
            })
    log(f"  -> {len(edges)} adjacency edges")
    return edges


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_adm1_csv(edges: list[dict], out_path: Path,
                   de_facto_borders: bool = True) -> None:
    cols = [
        "ADM1_code_A", "ADM1_name_A", "country_A", "ISO_A3_A", "WB_region_A",
        "ADM1_code_B", "ADM1_name_B", "country_B", "ISO_A3_B", "WB_region_B",
        "cross_country", "border_length_km", "narrow_border",
        "potential_artifact",
    ]

    def _row_from_edge(e: dict) -> dict:
        km = e["border_length_km"]
        return {
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
        }

    rows = []
    dropped = 0
    for e in edges:
        if frozenset({e["code_a"], e["code_b"]}) in _ADM1_FALSE_POSITIVE_DENYLIST:
            dropped += 1
            continue
        rows.append(_row_from_edge(e))

    # De-facto disputed-territory overlay (default).  Re-adds the ADM1 borders
    # that the NDLSA-exclusion opened a gap across; omitted for the strict
    # standard-layer view.  See _ADM1_DISPUTED_OVERLAY for the authored
    # attribution caveat.
    overlay_added = 0
    if de_facto_borders:
        present = {frozenset({r["ADM1_code_A"], r["ADM1_code_B"]}) for r in rows}
        for e in _ADM1_DISPUTED_OVERLAY:
            if frozenset({e["code_a"], e["code_b"]}) in present:
                continue
            rows.append(_row_from_edge(e))
            overlay_added += 1

    rows.sort(key=lambda r: (r["ADM1_code_A"], r["ADM1_code_B"]))
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log(f"  wrote {out_path} ({len(rows)} edges; "
        f"{dropped} false-positive edge(s) dropped; "
        f"{overlay_added} de-facto overlay edge(s) added)")


def _norm(s: object) -> str:
    """Normalise a tract name (the source has stray newlines, e.g. 'Kauirik')."""
    return " ".join(str(s).split())


def derive_disputed_overlay(
    a0: gpd.GeoDataFrame,
    a1: gpd.GeoDataFrame,
    ndlsa: gpd.GeoDataFrame,
) -> None:
    """Derive the de-facto disputed-territory overlay from geometry.

    For each NDLSA tract with an assigned de-facto administrator
    (``_NDLSA_TRACT_ADMIN``), fold the tract into that administrator's polygon
    and re-measure adjacency.  A pair is added to the overlay only when the two
    flanking units are NON-adjacent in the strict layer but adjacent after the
    fold (i.e. the tract is their sole land link) — this automatically excludes
    already-adjacent ISO pairs and multi-claimant artifacts.

    Populates the module globals ``_ADM0_DISPUTED_ALLOWLIST`` (ISO pairs),
    ``_ADM1_DISPUTED_OVERLAY`` (full ADM1 edge rows) and
    ``_DISPUTED_OVERLAY_MANIFEST`` (sidecar rows, both levels).

    Validation: every assigned tract must touch (within SNAP_TOL of) at least
    one polygon of its administrator at the level being processed; a tract that
    touches NONE of its administrator's geometry raises ``ValueError`` (this is
    the guard that makes a mis-labelled tract fail loudly instead of silently
    dropping a pair).
    """
    global _ADM0_DISPUTED_ALLOWLIST, _ADM1_DISPUTED_OVERLAY
    global _DISPUTED_OVERLAY_MANIFEST

    # tract geometry + admin, keyed by normalised name
    tracts: dict[str, BaseGeometry] = {}
    for _, row in ndlsa.iterrows():
        tracts[_norm(row["NAM_0"])] = row.geometry
    # admin -> [tract names]
    admin_tracts: dict[str, list[str]] = {}
    for name, adm in _NDLSA_TRACT_ADMIN.items():
        if adm is None:
            continue
        if name not in tracts:
            raise ValueError(f"NDLSA admin map names unknown tract: {name!r}")
        admin_tracts.setdefault(adm, []).append(name)

    def _union_iso(gdf, iso):
        sub = gdf[gdf["ISO_A3"] == iso]
        return unary_union(list(sub.geometry)) if len(sub) else None

    def _shared_km(a, b):
        sh = _shared_border(a, b)
        return _geodesic_km(sh) if (sh is not None and not sh.is_empty) else 0.0

    adm0_pairs: set[frozenset[str]] = set()
    adm1_rows: list[dict] = []
    manifest: list[dict] = []
    tract_label = {  # de-facto admin -> human tract list for the manifest
        adm: "; ".join(sorted(names)) for adm, names in admin_tracts.items()
    }

    for adm, names in sorted(admin_tracts.items()):
        adm_geom0 = _union_iso(a0, adm)
        if adm_geom0 is None:
            continue
        merged = unary_union([adm_geom0] + [tracts[n] for n in names])
        # validation: at least one tract must touch this admin's polygon
        if not any(adm_geom0.distance(tracts[n]) <= SNAP_TOL_DEG for n in names):
            raise ValueError(
                f"NDLSA validation: no tract of admin {adm} ({names}) touches "
                f"its ADM0 polygon — check _NDLSA_TRACT_ADMIN"
            )
        # which OTHER ISO becomes newly adjacent?
        for other in sorted(set(a0["ISO_A3"]) - {adm}):
            og = _union_iso(a0, other)
            if og is None:
                continue
            if _shared_km(adm_geom0, og) > 0:
                continue  # already adjacent in strict layer
            km0 = _shared_km(merged, og)
            if km0 <= 0:
                continue
            adm0_pairs.add(frozenset({adm, other}))
            a_name = a0[a0["ISO_A3"] == adm]["NAM_0"].iloc[0]
            o_name = a0[a0["ISO_A3"] == other]["NAM_0"].iloc[0]
            manifest.append({
                "level": "adm0", "code_a": adm, "name_a": a_name, "iso_a": adm,
                "code_b": other, "name_b": o_name, "iso_b": other,
                "defacto_admin_iso": adm, "tracts": tract_label[adm],
                "defacto_border_km": round(km0, 1),
            })
    # ---- ADM1 overlay: authored de-facto admin province(s) per tract ----
    # Country-level adjacency (ADM0, above) cannot stand in for province-level
    # adjacency: two countries adjacent elsewhere can still have their flanking
    # PROVINCES meet only across a disputed tract.  So the ADM1 overlay is
    # derived independently from an authored, geometry-validated tract->province
    # map.  Completeness guard: every folded tract must carry an assignment.
    for _nm, _adm in _NDLSA_TRACT_ADMIN.items():
        if _adm is not None and _nm not in _NDLSA_TRACT_ADM1:
            raise ValueError(f"_NDLSA_TRACT_ADM1 is missing tract {_nm!r}")
    a1g = list(a1.geometry)
    a1code = list(a1["ADM1CD_c"]); a1iso = list(a1["ISO_A3"])
    a1name = list(a1["NAM_1"]); a1ctry = list(a1["NAM_0"]); a1reg = list(a1["WB_REGION"])
    code2i = {c: i for i, c in enumerate(a1code)}
    a1tree = STRtree(a1g)

    def _attribute(seg, ap_idx):
        """Credit each ~1 km of `seg` to the nearest authored province index."""
        acc: dict[int, float] = {}
        sub = STRtree([a1g[i] for i in ap_idx])
        parts = list(seg.geoms) if seg.geom_type == "MultiLineString" else [seg]
        for ln in parts:
            if ln.length == 0:
                continue
            steps = max(1, int(ln.length / _ADM1_SAMPLE_DEG))
            prev = ln.interpolate(0.0, normalized=True)
            for k in range(1, steps + 1):
                cur = ln.interpolate(k / steps, normalized=True)
                mid = LineString([prev, cur]).interpolate(0.5, normalized=True)
                owner = ap_idx[int(sub.nearest(mid))]
                acc[owner] = acc.get(owner, 0.0) + _geodesic_km(LineString([prev, cur]))
                prev = cur
        return acc

    adm1_acc: dict[tuple[str, str], float] = {}
    adm1_src: dict[tuple[str, str], set[str]] = {}
    adm1_owner: dict[tuple[str, str], tuple[int, int]] = {}
    for tname, provs in _NDLSA_TRACT_ADM1.items():
        if not provs:
            continue  # de-facto admin sub-unit is not a WB province -> ADM0-only
        T = tracts[tname]
        ap_idx: list[int] = []
        for c in provs:
            if c not in code2i:
                raise ValueError(f"_NDLSA_TRACT_ADM1[{tname!r}]: unknown ADM1 code {c!r}")
            gi = code2i[c]
            if a1g[gi].distance(T) > SNAP_TOL_DEG:
                raise ValueError(
                    f"_NDLSA_TRACT_ADM1 validation: {c} does not touch tract {tname!r}"
                )
            ap_idx.append(gi)
        adm_isos = {a1iso[i] for i in ap_idx}
        for jx in a1tree.query(T.buffer(SNAP_TOL_DEG)):
            oi = int(jx)
            if a1iso[oi] in adm_isos or a1g[oi].distance(T) > SNAP_TOL_DEG:
                continue
            seg = _shared_border(T, a1g[oi])
            if seg is None or seg.is_empty:
                continue
            for owner, kmv in _attribute(seg, ap_idx).items():
                if kmv <= 0:
                    continue
                # NEW pairs only: the two provinces are non-adjacent in strict
                if _shared_km(a1g[owner], a1g[oi]) > 0:
                    continue
                key = (a1code[owner], a1code[oi])
                adm1_acc[key] = adm1_acc.get(key, 0.0) + kmv
                adm1_src.setdefault(key, set()).add(tname)
                adm1_owner[key] = (owner, oi)
    for key, kmv in adm1_acc.items():
        owner, oi = adm1_owner[key]
        kmv = round(kmv, 1)
        adm1_rows.append({
            "code_a": a1code[owner], "name_a": a1name[owner],
            "country_a": a1ctry[owner], "iso_a": a1iso[owner], "wb_a": a1reg[owner],
            "code_b": a1code[oi], "name_b": a1name[oi],
            "country_b": a1ctry[oi], "iso_b": a1iso[oi], "wb_b": a1reg[oi],
            "border_length_km": kmv,
        })
        manifest.append({
            "level": "adm1", "code_a": a1code[owner], "name_a": a1name[owner],
            "iso_a": a1iso[owner], "code_b": a1code[oi], "name_b": a1name[oi],
            "iso_b": a1iso[oi], "defacto_admin_iso": a1iso[owner],
            "tracts": "; ".join(sorted(adm1_src[key])),
            "defacto_border_km": kmv,
        })

    _ADM0_DISPUTED_ALLOWLIST = adm0_pairs
    _ADM1_DISPUTED_OVERLAY = adm1_rows
    # stable sort: adm0 rows first, then adm1, each by code_a/code_b
    manifest.sort(key=lambda r: (r["level"], r["code_a"], r["code_b"]))
    _DISPUTED_OVERLAY_MANIFEST = manifest
    log(f"  derived disputed overlay: {len(adm0_pairs)} ADM0 pairs, "
        f"{len(adm1_rows)} ADM1 pairs")


def write_disputed_overlay_manifest(out_path: Path) -> None:
    """Write the shipped sidecar listing the de-facto disputed-territory pairs.

    This documents the authored ADM0 + ADM1 overlay pairs (the user's manual-
    verification artifact) and is also read at runtime to subtract the overlay
    for the strict (``de_facto_borders=False``) view.
    """
    cols = [
        "level", "code_a", "name_a", "iso_a", "code_b", "name_b", "iso_b",
        "defacto_admin_iso", "tracts", "defacto_border_km",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in _DISPUTED_OVERLAY_MANIFEST:
            w.writerow({c: r.get(c, "") for c in cols})
    log(f"  wrote {out_path} ({len(_DISPUTED_OVERLAY_MANIFEST)} overlay pairs)")


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


def write_water_separated_manifest(
    bridge_csv: Path, edge_list_path: Path, out_path: Path
) -> None:
    """Write the water-separated manifest (``coupling_standard`` filter source).

    ``bridge_csv`` is the curated, web+geometry-verified ADM1 bridge
    classification (water-only pairs with ``has_bridge``) — a *reviewed static
    artifact* produced by the bridge-classification pipeline and NOT regenerated
    here (it embeds web + manual verification; see
    ``docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md``).  Only the ADM0 roll-up is
    computed: a country pair is water-only iff *all* its ADM1 crossings (from
    ``edge_list_path``) are water-only, and has a bridge iff *any* does.
    """
    import collections

    water_all: set[frozenset[str]] = set()
    has_b: dict[frozenset[str], bool] = {}
    adm1_rows: list[tuple[str, str, str, str, str]] = []
    with open(bridge_csv, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            ca, cb = r["code_a"].strip(), r["code_b"].strip()
            br = "True" if str(r["has_bridge"]).strip() == "True" else "False"
            pair = frozenset({ca, cb})
            water_all.add(pair)
            has_b[pair] = br == "True"
            adm1_rows.append(
                (ca, cb, br, r.get("water_type", ""), r.get("water_body", ""))
            )

    edges: set[frozenset[str]] = set()
    iso: dict[str, str] = {}
    with open(edge_list_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            a, b = r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()
            edges.add(frozenset({a, b}))
            iso[a] = r["ISO_A3_A"].strip()
            iso[b] = r["ISO_A3_B"].strip()
    by_country: dict[frozenset[str], list] = collections.defaultdict(list)
    for e in edges:
        a, b = tuple(e)
        ia, ib = iso.get(a), iso.get(b)
        if ia and ib and ia != ib:
            by_country[frozenset({ia, ib})].append(e)
    adm0_rows: list[tuple[str, str, str]] = []
    for ctypair, crossings in by_country.items():
        if all(c in water_all for c in crossings):
            ia, ib = sorted(ctypair)
            anyb = "True" if any(has_b.get(c, False) for c in crossings) else "False"
            adm0_rows.append((ia, ib, anyb))

    cols = ["level", "code_a", "code_b", "has_bridge",
            "water_type", "water_body", "note"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for ca, cb, br, wt, wb in adm1_rows:
            w.writerow(["adm1", ca, cb, br, wt, wb, ""])
        for ia, ib, br in sorted(adm0_rows):
            w.writerow(["adm0", ia, ib, br, "", "", "adm1-rollup"])
    log(f"  wrote {out_path} ({len(adm1_rows)} adm1 + "
        f"{len(adm0_rows)} adm0 water-separated pairs)")


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
    ap.add_argument("--ndlsa-gpkg",
                    help="NDLSA disputed-areas GeoPackage. If given, the de-facto "
                         "disputed-territory overlay is derived from geometry "
                         "(default view); if omitted, no overlay is applied "
                         "(strict standard-layer build).")
    ap.add_argument("--ndlsa-layer", default="WB_GAD_ADM0_NDLSA")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clip-ocean", action="store_true",
                    help="subtract the WB ocean mask before adjacency. OFF by "
                         "default: WB official boundaries are already land-only "
                         "(islands isolate, no maritime adjacencies), and the "
                         "clip is the build's main bottleneck.")
    ap.add_argument("--no-relabel-sliver", action="store_true",
                    help="skip the source-relabel stage (reassigning reviewed WB "
                         "sliver-corridor artifacts to their true units before "
                         "contiguity; on by default). See "
                         "scripts/relabel_sliver_corridors.py.")
    ap.add_argument("--levels", default="adm0,adm1")
    ap.add_argument("--bridge-csv", default=None,
                    help="curated ADM1 bridge classification CSV; if given, "
                         "writes water_separated_pairs.csv (coupling_standard "
                         "source). See docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md")
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

    # Stage 1 is pure World Bank geometry -- Natural Earth is NOT used in
    # the topology build (no lake filter, no river-buffer length).  Water
    # classification is a separate descriptive layer (Stage 3,
    # water_separated_pairs.csv) that never adds or drops an edge.

    def _prep(path: str, layer: str) -> gpd.GeoDataFrame:
        g = _make_valid(gpd.read_file(path, layer=layer).to_crs(4326))
        if ocean is not None:
            g = _clip_to_land(g, ocean)
        return g

    # Derive the de-facto disputed-territory overlay from geometry (before the
    # writers, which consume the module-level overlay globals).  Needs the
    # unclipped ADM0 + ADM1 polygons + the NDLSA tracts.  Skipped (strict build)
    # if --ndlsa-gpkg is not supplied.
    if args.ndlsa_gpkg:
        log("=== disputed overlay (de-facto) ===")
        a0_raw = _make_valid(
            gpd.read_file(args.adm0_gpkg, layer=args.adm0_layer).to_crs(4326)
        )
        a1_raw = _make_valid(
            gpd.read_file(args.adm1_gpkg, layer=args.adm1_layer).to_crs(4326)
        )
        ndlsa = _make_valid(
            gpd.read_file(args.ndlsa_gpkg, layer=args.ndlsa_layer).to_crs(4326)
        )
        derive_disputed_overlay(a0_raw, a1_raw, ndlsa)

    if "adm1" in levels:
        log("=== Stage 1: ADM1 topology (WB only, exact contact) ===")
        a1 = _prep(args.adm1_gpkg, args.adm1_layer)
        if not args.no_relabel_sliver:
            from relabel_sliver_corridors import relabel as _relabel_sliver
            log("  source-relabel: reassigning reviewed sliver-corridor artifacts")
            a1, _rl_log = _relabel_sliver(a1, code_col="ADM1CD_c", verbose=True)
            log(f"  source-relabel: {len(_rl_log)} corridor(s) reassigned")
        for _src, _dst in _ADM1_UNIT_MERGES.items():
            _si = a1.index[a1["ADM1CD_c"] == _src]
            _di = a1.index[a1["ADM1CD_c"] == _dst]
            if len(_si) and len(_di):
                log(f"  unit merge: {_src} -> {_dst} (reviewed; see denylist block)")
                a1.loc[_di[0], a1.geometry.name] = unary_union(
                    [a1.loc[_di[0]].geometry, a1.loc[_si[0]].geometry])
                a1 = a1.drop(index=_si)
        e1 = build_edges(a1, "ADM1CD_c")
        write_adm1_csv(e1, out_dir / "pericoupled_adm1_edge_list.csv")

    if "adm0" in levels:
        log("=== Stage 1: ADM0 topology (WB only, exact contact) ===")
        a0 = _prep(args.adm0_gpkg, args.adm0_layer)
        e0 = build_edges(a0, "ISO_A3")
        write_adm0_matrix(
            e0, list(a0["ISO_A3"]), out_dir / "PeriTelecoupling_clean.csv"
        )

    # Shipped manifest of the de-facto disputed-territory overlay pairs
    # (verification artifact + runtime strict-mode subtraction source).
    write_disputed_overlay_manifest(out_dir / "disputed_overlay_pairs.csv")

    # Shipped water-separated manifest (coupling_standard filter source).  The
    # ADM1 bridge data is a reviewed static artifact (web+geometry verified);
    # only the ADM0 roll-up is computed here.  See
    # docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md.
    if args.bridge_csv:
        write_water_separated_manifest(
            Path(args.bridge_csv),
            out_dir / "pericoupled_adm1_edge_list.csv",
            out_dir / "water_separated_pairs.csv",
        )

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
