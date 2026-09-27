# Reproducing the pericoupling database — step-by-step manual

This manual walks through regenerating and verifying the two bundled adjacency
datasets — the ADM1 edge list (8,464 subnational shared-border pairs) and the
ADM0 country matrix (326 pairs) with their water-only classification (805
ADM1 pairs / 26 ADM0 roll-ups) — from scratch. It is written for a reader who
has never touched the pipeline. Companion documents:
`src/metacouplingllm/data/PROVENANCE.md` (what the data is, sources, known
limitations) and `docs/METHODS_adjacency.md` (why each methodological choice
is defensible).

## 0. What "reproducing" means here — three distinct claims

The pipeline separates three things a reader might want to check, and it is
important to know which one you are testing:

1. **Build reproducibility** (fully automatic). The shipped CSVs are a pure
   function of (a) four pinned World Bank GeoPackages and (b) four reviewed
   input files shipped in the repo (the source-relabel manifest, a Stage 1
   input; the water classification file, Stage 3; the land-gap and denylist
   files, Stage 4; the de-facto overlay manifest is
   written by the build from the NDLSA layer, the tract administrators
   taken from Natural Earth v5.1.2 and the province attributions authored in
   `build_pericoupling_db.py`). Re-running the build replays these inputs deterministically —
   **no AI, no network, no judgment calls at build time**. Sections 2–4.
2. **Audit traceability** (read, don't re-run). The reviewed inputs of Stages 3
   and 4 were *discovered* by deterministic Python screens and *adjudicated* by
   frozen two-pass AI research + human map review. Those verdicts are frozen
   history: every water row carries its provenance in a `source` column, every
   denylist row its evidence and ruling,
   and the discovery record lives in PROVENANCE.md / METHODS. Reproducing
   the database does **not** require re-running any audit. Section 5.
3. **Screen re-derivability** (optional, heavier). The deterministic screens
   that *nominate* the audit candidates (the edge screens on one border arc —
   Natural Earth ladders, HydroRIVERS, HydroLAKES, the combined river and lake
   screens and the cross-type union — and the non-touching corridor census) can
   be re-run from public data to confirm the candidate lists. Section 6.

## 1. Prerequisites

- Python ≥ 3.11 with `geopandas`, `shapely`, `pyproj`, `pandas` (the shipped
  lengths were produced with geopandas 1.1.2 / shapely 2.1.2 / pyproj 3.7.2 —
  see the byte-identity note in §3).
- The repository checkout (the reviewed input files and the engine ship in it).
- For the **full** regeneration only: the four pinned GeoPackages (~1 GB)
  below. The quick verification path (§2) needs no downloads.

**Pinned inputs** (SHA-256 also hard-coded in `scripts/build_all.py`, which
refuses to build from a mismatching file):

| input | source | SHA-256 |
|---|---|---|
| WB Admin 1 GeoPackage (layer `WB_GAD_ADM1`, 3,591 features) | World Bank Official Boundaries, **2026-05-14 release** — datacatalog.worldbank.org/search/dataset/0038272 | `dbac29f4…b64b45` |
| WB Admin 0 GeoPackage (layer `WB_GAD_ADM0`, 264 features) | same distribution | `97f0c8a0…f4b117e` |
| WB Ocean Mask GeoPackage | same distribution | `c2b074fd…c88d702` |
| WB NDLSA GeoPackage (24 disputed-area features) | same distribution | `159ef2d1…55d2fa4` |

(Full 64-character hashes: `data/PROVENANCE.md` → "Sources (pinned)".)

> The GeoPackage digests are of the raw bytes. The reviewed input files live in
> the repository under version control and are not pinned; `build_all.py`
> reports whether the regenerated files are byte-identical to the committed ones.

**What a clean clone contains.** Everything §2 and §4 need ships in the
repository, including the reviewed input files (in `src/metacouplingllm/data/`).
The four GeoPackages for §3 must be downloaded from the
World Bank (their hashes are verified before any build). The wider
`build_data/` audit-evidence tree (screen outputs, frozen AI verdicts,
worksheets) is a local working archive, NOT shipped in the repository — its
content is summarized in PROVENANCE.md/METHODS and in each water row's
`source` column, and §6 re-derives the candidate screens from the public
datasets directly rather than from those local artifacts.

## 2. Quick path — verify the shipped data (minutes, no downloads)

```
python scripts/build_all.py
```

This (a) re-applies Stages 3 and 4 (the reviewed inputs) to the shipped data
via `scripts/apply_overlays.py`, (b) recomputes every headline count from the
resulting CSVs, and (c) compares them to the expected values. On an untouched
checkout it must print `build_all: OK` with all twelve counts `OK` and report
the engine as a **byte-stable no-op** — i.e. applying Stages 3 and 4 again
changes zero bytes. This is the day-to-day reproducibility check; CI runs the
same property as a test (`tests/test_apply_overlays.py`).

Expected counts (current):

| count | value |
|---|---|
| ADM1 edges (lenient) | 8,464 (3,374 regions, 196 countries) |
| ADM1 moderate / stringent | 8,067 / 7,659 |
| water-only ADM1 | 805 = 408 with a fixed crossing / 397 without |
| water-only provenance | `adjudication` all `cross-vendor`; `verification_tier` A 270 / B 238 / C 297 |
| ADM0 pairs (lenient / moderate / stringent) | 326 / 320 / 300 |
| ADM0 water roll-ups | 26 |

## 3. Full regeneration from the pinned sources (~1–2 h)

```
python scripts/build_all.py --full ^
    --adm1-gpkg  <path>\WB_ADM1.gpkg ^
    --adm0-gpkg  <path>\WB_ADM0.gpkg ^
    --ocean-gpkg <path>\WB_ocean_mask.gpkg ^
    --ndlsa-gpkg <path>\WB_NDLSA.gpkg
```

What happens, stage by stage (S1–S2 inside `build_pericoupling_db.py`, S3–S4
in `scripts/apply_overlays.py`):

- **S1 — geometry.** SHA-256 of every GeoPackage is verified against the pins
  (hard error on mismatch). A **source-relabel** step first reassigns 10
  reviewed WB sliver-corridor artifacts to their true owner units
  (`scripts/relabel_sliver_corridors.py` + manifest
  `sliver_corridor_relabel.csv`; each corridor's cut points become vertices
  of every outline through them, so an owner takes exactly the frontage its
  host gives up). Then rook contiguity at **exact contact
  (tolerance 0)** over the land-clipped polygons: two units are adjacent iff
  their boundaries share a segment of non-zero geodesic length (a shared
  corner does not count). No lake filter — units meeting across a lake are
  native edges. Border lengths are full geodesic
  shared-boundary lengths (WGS84, `pyproj.Geod`).
  → 8,427 raw − 2 relabel − 3 unit merge (RUS050 → RUS024, the reviewed
  rejoining of Kalmykia's split salient) = **8,422** native pairs.
- **S2 — de-facto connectivity.** Each NDLSA disputed tract is folded into
  its de-facto administrator (the administrator Natural Earth v5.1.2 records for
  most of the tract; the table `_NDLSA_TRACT_ADMIN`) and adjacency re-measured;
  +16 ADM1 pairs whose sole link is a tract (+3 pairs at ADM0). Geometry-validated:
  an assigned tract that does not touch its administrator's territory (unless it
  touches no unit at all) or an authored province that does not touch its tract
  fails loudly.
  → **8,438**; this is the base file the geometry build writes (the five
  denylisted contacts still in it; `build_all.py` checks that each is present).
- **S3 — water-only classification.** `scripts/apply_overlays.py` reads
  `water_classification_pairs.csv` (805 rows; the build reads no Natural Earth
  layer): it flags 779 existing edges water-only (`water_type`, `water_body`,
  `has_bridge`) and adds the 26 water borders between non-touching units, each
  with its water row → **8,464**; water rows **805**.
- **S4 — edge corrections.** +5 land-gap edges (`land_gap_overlay_pairs.csv`)
  and −5 denylisted contacts (`denylist_pairs.csv`) → **8,464**; the ADM0
  roll-up is then recomputed once (a country pair is water-only iff *all* its
  ADM1 crossings are; bridged iff *any* is). Stages 3–4 are idempotent, one pass
  (§5).

The command then verifies all twelve headline counts and reports whether the
regenerated files are byte-identical to the committed ones.

**Byte-identity scope (important, honest caveat).** The adjacency *pair set*
reproduces exactly on any toolchain (verified by clean-room regenerations,
the last on 2026-09-25). The advisory `border_length_km` column can differ in low-order
digits under a different GEOS/Shapely version, because the geometry-cleaning
and ocean-clip operations preceding the length measurement are sensitive to
the buffer implementation — lengths never add or drop an edge, so this does
not affect the pair set or any count. Exact byte-identity of lengths
additionally requires the original toolchain (geopandas 1.1.2 / shapely 2.1.2
/ pyproj 3.7.2). Stages 3–4 are toolchain-independent, and their row order is
canonical: the native rows keep the geometry-build order (the denylisted
contacts removed), then the Stage 3 edges in water-file order and the Stage 4
edges in manifest order; the water table follows the water file's order —
exactly what a fresh `--full` run produces.

## 4. Run the test suite

```
python -m pytest tests/ -q
```

The suite (1,389 tests at the time of writing) includes: the expected-count assertions, the engine
byte-stability guard, a fresh-build check that Stages 3–4 reproduce the
shipped files, the check that the engine's inputs are exactly the reviewed
files, loader behavior for
`de_facto_borders` × `coupling_standard`, and **doc-drift guards** that parse
`docs/METHODS_adjacency.md`, `INTRODUCTION.md`, and `MANUAL.md` and fail if
any headline count in the prose disagrees with the live data.

## 5. The reviewed inputs — what they are and how to change them

Five reviewed files in `src/metacouplingllm/data/` govern the build: the
source-relabel manifest is a Stage 1 input, the de-facto overlay
manifest is written by S2 from the NDLSA layer, the tract administrators taken
from Natural Earth v5.1.2 and the province attributions authored in the build
script, and the other **three** are the inputs of Stages 3 and 4 (815 rows:
805 water rows, 26 of them adding an edge, 5 land edges and 5 removals). The
engine (`scripts/apply_overlays.py`) holds only behavior; editing a file and
re-running the engine is the supported way to change Stages 3 and 4:

| file | effect |
|---|---|
| `sliver_corridor_relabel.csv` | S1 input: 10 polygon relabels before contiguity |
| `disputed_overlay_pairs.csv` | S2 output: 16 ADM1 + 3 ADM0 de-facto pairs (the loaders drop them when `de_facto_borders=False`) |
| `water_classification_pairs.csv` | S3: water flags on 779 existing edges and 26 water borders between non-touching units added as edges (`adds_edge`); per-row `water_type`, `water_body`, `has_bridge`, `adjudication`, `verification_tier` and discovery provenance in `source` (composition by origin: `data/PROVENANCE.md`) |
| `land_gap_overlay_pairs.csv` | S4: +5 land edges (four sub-tolerance borders, and Samukh↔Yevlakh, whose polygons meet only at a point) |
| `denylist_pairs.csv` | S4: −5 contacts found not to be borders, each with its evidence and ruling; checked present in the geometry build's output before removal |

Engine semantics worth knowing:

- **Idempotent and byte-stable**: outputs are composed in memory and written
  only if bytes differ; `--check` exits 2 instead of writing. Running on
  already-corrected data is a no-op.
- **The water table is composed from the water file**: every run writes it
  from `water_classification_pairs.csv` (its `note` column names the
  instrument class, the edge screens or the corridor census), so the water
  file is the one place to edit a water row.
- **Editing propagates**: change `has_bridge` in a water-file row, re-run the
  engine, and the water CSV row plus the ADM0 roll-up update on the next
  pass.
- **Every row carries provenance**: the water file's `source` column states
  how the row was discovered, adjudicated, verified, and how its bridge flag
  was classified; each denylist row carries its evidence and ruling.

**What is deliberately NOT re-run.** The AI adjudication (two-pass research +
adversarial judgment) and the human map reviews that *validated* each
row are frozen history — re-running a live model would make the
build non-deterministic. The design principle throughout: *deterministic
screens nominate, frozen audits decide, the build replays manifests.* If you
distrust a row, its `source` string plus PROVENANCE.md tell you exactly which
screen found it, which audit design judged it, and who verified it.

## 6. Optional: re-derive the candidate screens (public data)

The audits' *candidate lists* came from deterministic screens over public
datasets; each can be re-run to confirm no candidate was hand-picked:

- **The border arc** (`build_data/water_screen_rebuild/border_arc.py`; specification
  `build_data/arc_and_combined_screens/SPEC_ba1_border_arc_and_screens.md`): every
  edge screen measures the same set of points. For an edge A↔B, on the build's
  own polygons (`load_adm1_build_geometry` in `scripts/build_pericoupling_db.py`),
  the arc is A's outline where it coincides with B's (merged into continuous
  lines), where it runs inside B (overlapping polygons), within 5×10⁻⁴° of B for
  the four sub-tolerance land-gap pairs only, and where it faces B across a gap of at most
  1,000 m (not on or across a third unit's outline, reciprocal, the chord's
  midpoint inside neither unit; a chord shorter than 1 m is contact and stays
  in the arc unless it only repeats the end of a stretch already in it). Each
  continuous piece of geodesic length L gets n + 1 points at equal geodesic
  intervals, n = max(2, ⌊L/500 m⌋ + 1). Every distance is measured on the
  ground (`build_data/geodesic_rescreen/geodist.nearest_on_ground`; specification
  `build_data/geodesic_distances/SPEC_gd1_geodesic_distances.md`): the nearest
  point is located in a local frame in which a metre east equals a metre north
  (beyond 2.5 km, in an azimuthal-equidistant frame centred on the point) and
  measured with `GEOD.inv`. `build_arc_cache.py --out arc_cache_gd1.jsonl`
  stores, per point, the distance to each layer;
  `build_data/geodesic_distances/run_screens_gd1.py` computes the exact shares
  and nominations (`gd1_screens.csv`, report `gd1_report.txt`), and
  `build_data/water_screen_rebuild/rule_rejections/run_screens_mr1.py` the current record,
  with the cross-type union at every ladder width (`mr1_screens.csv`).
- **Edge screens on the arc:** Natural Earth river ladder
  (`ne_10m_rivers_lake_centerlines`, 1:10M, named rivers): ≥ 0.50 within 2.5 km,
  rungs 5/10/15/20 km; Natural Earth lake ladder (`ne_10m_lakes`): ≥ 0.40 within
  125 m, rungs to 1,500 m; HydroRIVERS v10 (hydrosheds.org): ≥ 0.50 within 500 m
  of any reach (discharge tiers 0 / 10 / 100 m³/s recorded); HydroLAKES v10:
  ≥ 0.40 within 500 m; combined river (a point within 2.5 km of a Natural Earth
  river or 500 m of a HydroRIVERS reach): ≥ 0.50; combined lake (125 m of a
  Natural Earth lake or 500 m of a HydroLAKES polygon): ≥ 0.40; cross-type union
  (any of the four layers at its widest width: Natural Earth rivers 20 km, lakes 1,500 m,
  HydroRIVERS and HydroLAKES 500 m): ≥ 0.80. An edge is nominated
  when any screen reaches its bar.
- **Non-touching corridor census**
  (`build_data/water_screen_rebuild/corridor_census_exact/census_cc1.py`, on
  the build's polygons): for unit pairs within 100 km of each other on the
  ground that are neither edges nor denylisted contacts, those whose polygons
  meet only at a point included, every distance on the ground in an azimuthal-equidistant frame
  centred on each pair's nearest approach. Transects start at most 250 m apart
  along each unit's outline where it lies within the facing band of the other
  unit (max(2 × gap, gap + 1 km), at most 100 km) and run to the nearest point
  of the other outline. The share of their length inside the union of the edge
  screens' four layers at their base widths (named Natural Earth rivers 2.5 km,
  Natural Earth lakes 125 m, HydroRIVERS and HydroLAKES 500 m; the river layers
  for gaps up to 5 km) nominates at ≥ 0.80; a gap of at most 1 km nominates when
  a transect meets the HydroRIVERS buffer; and a gap of at most 5 km nominates
  when crossings at most 5 km long that each lie at least 80% inside the layers
  follow one another along an outline for more than 1 km.
- **Bridge screen** (OpenStreetMap Overpass): any way tagged as a bridge on a
  road, path or railway (or `man_made=bridge`), not under construction or
  proposed, coming within 100 m of both units' polygons measured on the ground
  (`build_data/water_screen_rebuild/crossing_width/cw1_screen.py`; an answer
  that reports an Overpass error counts as a failed query) — layer 1 of the
  four-layer `has_bridge` classification
  (`docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md`). The screen does not filter by
  way class: the rule that only a road or rail bridge, causeway, dam-top road
  or tunnel counts (ferries, fords and footbridges never do) is applied by the
  later layers — web verification, adversarial recheck, maintainer rulings.

Thresholds are anchored, not tuned, and they are screening tolerances, not
positional accuracies: 2.5 km ≈ ½ × the NMAS tolerance at 1:10M (0.5 mm of map
distance ≈ 5 km on the ground), a width that follows from the map scale;
500 m ≈ one 15 arc-second HydroSHEDS cell at the equator, the grid of
HydroRIVERS (whose technical documentation states no positional accuracy),
and HydroLAKES takes the same width; the rung ladders
were extended until the capture pattern was fully characterized; the border
arc's 1,000 m facing reach is the corridor census's presence gap, and the census's three 1 km
constants (the presence gap, the facing band's margin and the covered stretch, each twice the
HydroRIVERS width) are a convention; 0.80 is the union bar the census shares with the edge
screens. The bridge
screen's 100 m has no standard to anchor it: it is a round value inside the range over which
the screen's agreement with the reviewed flags is flat (25–250 m on the ground;
`crossing_width/SPEC_cw1_crossing_width.md`). Screens
only ever **nominate** — no threshold ships a row by itself.

## 7. Loader-level reproduction (how users consume the data)

```python
from metacouplingllm.knowledge.adm1_pericoupling import is_adm1_pericoupled
is_adm1_pericoupled("USA044", "MEX028")                       # default views
is_adm1_pericoupled("ROU008", "ROU039",
                    coupling_standard="stringent")            # water policy
```

Two orthogonal toggles select the view: `de_facto_borders` (default `True`;
`False` removes the 16 ADM1 / 3 ADM0 disputed-overlay pairs) and
`coupling_standard` (`"lenient"` keeps every water border; `"moderate"`,
the default, keeps water-only pairs only when a fixed crossing open to
traffic links the two units; `"stringent"` drops all water-only pairs).
All counts in §2 derive from these two toggles and nothing else.
