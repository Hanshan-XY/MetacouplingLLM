#!/usr/bin/env python
"""Build pericoupling adjacency databases from World Bank Official Boundaries.

Runs Stages 1 and 2 of the four-stage build and writes:

  * ``pericoupled_adm1_edge_list.csv`` — ADM1 (subnational) border edges
  * ``PeriTelecoupling_clean.csv``     — ADM0 (country) border matrix
  * ``disputed_overlay_pairs.csv``     — de-facto overlay manifest (with
    ``--ndlsa-gpkg``)

Method (full detail: ``docs/METHODS_adjacency.md``)
---------------------------------------------------
Stage 1 — geometry:

1. Load the WB admin layer + the WB ocean mask (same vintage, so the clip
   aligns with the boundaries — no cross-dataset mismatch); clip each polygon
   to land by subtracting the ocean mask (``--clip-ocean``).
2. The two reviewed corrections that change polygons act before contiguity:
   the **source-relabel** reassigns the WB sliver-corridor artifacts to their
   true owner units (``scripts/relabel_sliver_corridors.py`` + the reviewed
   manifest ``data/sliver_corridor_relabel.csv``), and the **unit merge**
   rejoins a unit the source split (``_ADM1_UNIT_MERGES``).
3. **Exact-contact rook contiguity** (``TOPOLOGY_TOL_DEG = 0``): two units are
   adjacent iff their boundaries share a segment of non-zero geodesic length.
   No snapping tolerance, no lake filter (pairs meeting across a lake are
   native edges; ``coupling_standard`` governs them downstream).
4. ``border_length_km`` is the **full** shared-boundary length in kilometres
   (geodesic, ``pyproj.Geod``) — no river/lake subtraction.

Stage 2 — **de-facto disputed overlay**: each WB NDLSA disputed-area tract is
folded into the administrator Natural Earth records for most of it
(``_NDLSA_TRACT_ADMIN``); the overlay pairs are derived from the geometry and
validated against it at build time.

Stages 3 (the water-only classification) and 4 (the reviewed edge
corrections: the land-gap borders and the denylist of non-adjacent contacts)
are applied afterwards by ``scripts/apply_overlays.py``;
``scripts/build_all.py`` runs all four stages in order and verifies every
headline count.

Flags (advisory only — never remove a pair):
  ``narrow_border``       : shared border < 5 km
  ``potential_artifact``  : shared border < 1 km

Sources (recorded in data/PROVENANCE.md with checksums):
  * World Bank Official Boundaries (GeoPackage, 2026-05-14)
  * Natural Earth v5.1.2 disputed areas + countries — the tract administrators,
    applied as the constant ``_NDLSA_TRACT_ADMIN`` (not read at build time)

Usage
-----
    python scripts/build_pericoupling_db.py \
        --adm1-gpkg "<...>/World Bank Official Boundaries - Admin 1 (1).gpkg" \
        --adm0-gpkg "<...>/World Bank Official Boundaries - Admin 0 (1).gpkg" \
        --ocean-gpkg "<...>/World Bank Official Boundaries - Ocean Mask.gpkg" \
        --ndlsa-gpkg "<...>/World Bank Official Boundaries - NDLSA (1).gpkg" \
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
# IMPORTANT — the NDLSA layer carries NO administering-country field (``SOVEREIGN``
# is null for all 24 tracts; ``WB_STATUS`` is uniformly "Non-determined legal status
# area").  Each tract is therefore assigned, whole, to the administrator that Natural
# Earth v5.1.2 (the latest release: ne_10m_admin_0_disputed_areas, then
# ne_10m_admin_0_countries for the rest of the tract) records for the largest part of
# it, a dependency counting for its sovereign; a tract whose largest part has no
# administrator the World Bank lists as a country is left unassigned (``None``, not
# folded).  No exceptions: a smaller area Natural Earth records inside a tract does not
# override the majority (maintainer rulings 2026-09-25, campaign da1; every share is
# recorded in build_data/ndlsa_ne/SPEC_da1_defacto_administrators.md).  The mapping
# describes de-facto administration for a connectivity dataset and is NOT a legal or
# endorsed sovereignty claim.  Keyed by the tract's ``NAM_0`` (note "Kauirik" carries a
# stray newline in the source, normalised on load).
_NDLSA_TRACT_ADMIN: dict[str, str | None] = {
    # India–China tracts (already-adjacent ISO pair → produce no new pair).  Natural
    # Earth: Aksai Chin CHN 99%; Kauirik CHN 57% (IND 43%); Lapthal IND 83%; Shipki
    # Pass IND 55% (CHN 45%); Kalapani IND 75%; the others IND or BTN, 95-100%:
    "Aksai Chin": "CHN", "Kauirik": "CHN", "Lapthal": "IND", "Shipki Pass": "IND",
    "Chumar East": "IND", "Chumar West": "IND", "Demchok": "IND",
    "Jadh Ganga Valley": "IND", "Arunachal Pradesh": "IND",
    "Jammu and Kashmir": "IND", "Kalapani": "IND", "Doklam": "BTN",
    # Pair-producing tracts (sole land link between the two flanking countries):
    "Gilgit Baltistan": "PAK",                              # → CHN/PAK
    "Golan Heights": "ISR", "Shebaa Farms Dispute": "ISR",  # → ISR/SYR (the Golan;
                                                            # Shebaa Farms touches
                                                            # Lebanon and Israel only)
    "Western Sahara": "MAR",                                # → MAR/MRT (Natural Earth:
                                                            # MAR 66%, self-admin. 34%)
    # Karakoram Range is the Shaksgam Valley (Trans-Karakoram Tract): Natural Earth
    # "Admin. by China; Ceded to China by Pakistan; Claimed by India" (99.9%).  It
    # meets Gilgit-Baltistan along the Karakoram crest through K2, which the de-facto
    # China–Pakistan border therefore includes.
    "Karakoram Range": "CHN",
    # Already-adjacent ISO pairs (folding adds no country pair):
    "Ilemi Triangle": "KEN",   # Kenya and South Sudan share ~80 km in the standard layer
    "Abyei": "SDN",            # Natural Earth "Admin. by Sudan; Claimed by South Sudan"
    "No Man's Land": "ISR",    # Latrun; Natural Earth "Admin. By Israel; Claimed by Palestine"
    # Island tracts: they touch no unit, so folding them adds no pair.
    "British Indian Ocean Territory": "GBR",
    "South Georgia and South Sandwich Islands": "GBR", "Falkland Islands": "GBR",
    # Not folded: Natural Earth records no administrator for most of it (the UN buffer
    # zone and Northern Cyprus, 89%).
    "UN Buffer Zone": None,
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
    "Shebaa Farms Dispute": ["ISR004"],   # links no pair: its Lebanese flank already
                                          # borders ISR004
    "Arunachal Pradesh": ["IND003"],      # the state itself — NOT Assam (which is
                                          # merely the nearest non-tract polygon)
    "Doklam": ["BTN005"],                 # Haa Dzongkhag
    "Western Sahara": ["MAR005", "MAR007"],  # Guelmim-Oued Noun + Laâyoune-Sakia
                                          # al Hamra (Moroccan "Southern Provinces")
    # The administrator's provinces that touch the tract (da1):
    "Abyei": ["SDN012", "SDN013"],        # Southern Darfur, Southern Kordofan
    "No Man's Land": ["ISR001", "ISR003"],  # Central District, Jerusalem
    # Already-adjacent flanks (no new overlay row — listed for completeness):
    "Aksai Chin": ["CHN029", "CHN028"], "Kauirik": ["CHN029"],
    "Lapthal": ["IND035"], "Shipki Pass": ["IND014"],  # Uttarakhand; Himachal Pradesh
    "Jadh Ganga Valley": ["IND035", "IND014"], "Kalapani": ["IND035"],
    "Ilemi Triangle": ["KEN043"],
    "Karakoram Range": ["CHN028"],        # Xinjiang; no other province touches it
    # De-facto admin sub-unit is NOT a WB ADM1 province → ADM0-only (empty):
    "Gilgit Baltistan": [],
    "Jammu and Kashmir": [], "Chumar East": [], "Chumar West": [], "Demchok": [],
    # Island tracts touch no province:
    "British Indian Ocean Territory": [],
    "South Georgia and South Sandwich Islands": [], "Falkland Islands": [],
}

# Frontier-sampling step (~1.1 km at the equator) used to split a tract↔neighbour
# border among the authored admin provinces by nearest province (ADM1 overlay).
_ADM1_SAMPLE_DEG = 0.01

# The reviewed non-adjacent contacts (the denylist) are removed in Stage 4 by
# scripts/apply_overlays.py from data/denylist_pairs.csv, each row with its
# evidence and ruling.  The geometry build keeps every exact contact; a
# denylist entry whose contact the geometry no longer produces is caught by
# scripts/build_all.py --full.

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
    the water-only classification is applied in Stage 3
    (``scripts/apply_overlays.py``), which never changes an edge's length.
    Because WB admin polygons include lake water, pairs that meet across
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

    rows = [_row_from_edge(e) for e in edges]

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

    Validation (campaign da1): every assigned tract must touch (within SNAP_TOL
    of) its administrator's territory, the administrator's own polygon or another
    tract assigned to it, unless the tract touches no unit at all (an island
    tract, which adds no pair); otherwise it raises ``ValueError`` (the guard
    that makes a mis-labelled tract fail loudly instead of silently dropping a
    pair).  A restored country border is measured between the two countries'
    de-facto territories (each country's polygon with the tracts assigned to it).
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

    # each administrator's de-facto territory: its own polygon with its tracts
    std0 = {adm: _union_iso(a0, adm) for adm in admin_tracts}
    merged_of = {adm: unary_union([g] + [tracts[n] for n in admin_tracts[adm]])
                 for adm, g in std0.items() if g is not None}
    a0tree = STRtree(list(a0.geometry))
    for adm, names in sorted(admin_tracts.items()):
        adm_geom0 = std0[adm]
        if adm_geom0 is None:
            continue
        merged = merged_of[adm]
        # validation: each tract touches its administrator's territory (its polygon
        # or another of its tracts), unless it touches no unit at all (an island)
        for n in names:
            own = [adm_geom0] + [tracts[m] for m in names if m != n]
            if any(g.distance(tracts[n]) <= SNAP_TOL_DEG for g in own):
                continue
            if len(a0tree.query(tracts[n], predicate="dwithin", distance=SNAP_TOL_DEG)) == 0:
                continue  # an island tract: folding it adds no pair
            raise ValueError(
                f"NDLSA validation: tract {n!r} does not touch the territory of its "
                f"administrator {adm} — check _NDLSA_TRACT_ADMIN"
            )
        # which OTHER ISO becomes newly adjacent?
        for other in sorted(set(a0["ISO_A3"]) - {adm}):
            og = _union_iso(a0, other)
            if og is None:
                continue
            if _shared_km(adm_geom0, og) > 0:
                continue  # already adjacent in strict layer
            if _shared_km(merged, og) <= 0:
                continue
            # the restored border, measured between the two de-facto territories
            km0 = _shared_km(merged, merged_of.get(other, og))
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_ocean_mask(ocean_gpkg: str, ocean_layer: str = "WB_GAD_ocean_mask"):
    """The ocean mask as one geometry, exactly as the build uses it for ``--clip-ocean``."""
    ocean_gdf = _make_valid(gpd.read_file(ocean_gpkg, layer=ocean_layer).to_crs(4326))
    return unary_union(list(ocean_gdf.geometry))


def load_adm1_build_geometry(adm1_gpkg: str, adm1_layer: str = "WB_GAD_ADM1", ocean=None,
                             relabel_sliver: bool = True, log=None) -> gpd.GeoDataFrame:
    """The ADM1 polygons the build derives its edges from: validity-repaired, in EPSG:4326, clipped to land
    when ``ocean`` is given (the ``--clip-ocean`` build), the reviewed source-relabel applied, the reviewed
    unit merges applied.  The build's own Stage 1 calls this; the water screens read the same polygons
    through it (``build_data/water_screen_rebuild/border_arc.py``), so the two cannot drift apart."""
    log = log or (lambda *_a, **_k: None)
    a1 = _make_valid(gpd.read_file(adm1_gpkg, layer=adm1_layer).to_crs(4326))
    if ocean is not None:
        a1 = _clip_to_land(a1, ocean)
    if relabel_sliver:
        from relabel_sliver_corridors import relabel as _relabel_sliver
        log("  source-relabel: reassigning reviewed sliver-corridor artifacts")
        a1, _rl_log = _relabel_sliver(a1, code_col="ADM1CD_c", verbose=True)
        log(f"  source-relabel: {len(_rl_log)} corridor(s) reassigned")
    for _src, _dst in _ADM1_UNIT_MERGES.items():
        _si = a1.index[a1["ADM1CD_c"] == _src]
        _di = a1.index[a1["ADM1CD_c"] == _dst]
        if len(_si) and len(_di):
            log(f"  unit merge: {_src} -> {_dst} (reviewed; see _ADM1_UNIT_MERGES)")
            a1.loc[_di[0], a1.geometry.name] = unary_union(
                [a1.loc[_di[0]].geometry, a1.loc[_si[0]].geometry])
            a1 = a1.drop(index=_si)
    return a1


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
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = {x.strip() for x in args.levels.split(",")}

    ocean = None
    if args.clip_ocean:
        log("loading ocean mask")
        ocean = load_ocean_mask(args.ocean_gpkg, args.ocean_layer)

    # Stage 1 is pure World Bank geometry -- Natural Earth is NOT used in
    # the topology build (no lake filter, no river-buffer length).  The water
    # classification (Stage 3) and the edge corrections (Stage 4) are applied
    # afterwards by scripts/apply_overlays.py.

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
        a1 = load_adm1_build_geometry(args.adm1_gpkg, args.adm1_layer, ocean=ocean,
                                      relabel_sliver=not args.no_relabel_sliver, log=log)
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

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
