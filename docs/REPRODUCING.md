# Reproducing the pericoupling database — step-by-step manual

This manual walks through rebuilding and verifying the two bundled adjacency
datasets — the ADM1 edge list (8,460 subnational shared-border pairs) and the
ADM0 country matrix (326 pairs) with their water-only classification (747
ADM1 pairs / 28 ADM0 roll-ups) — from scratch. It is written for a reader who
has never touched the pipeline. Companion documents:
`src/metacouplingllm/data/PROVENANCE.md` (what the data is, sources, known
limitations) and `docs/METHODS_adjacency.md` (why each methodological choice
is defensible).

## 0. What "reproducing" means here — three distinct claims

The pipeline separates three things a reader might want to check, and it is
important to know which one you are testing:

1. **Build reproducibility** (fully automatic). The shipped CSVs are a pure
   function of (a) four pinned World Bank GeoPackages, (b) one reviewed
   bridge-classification CSV, and (c) eleven reviewed manifest CSVs shipped in
   the repo (two build-stage inputs plus the nine forming the engine's
   correction layer). Re-running the build replays these inputs deterministically —
   **no AI, no network, no judgment calls at build time**. Sections 2–4.
2. **Audit traceability** (read, don't re-run). The nine manifests were
   *discovered* by deterministic Python screens and *adjudicated* by frozen
   two-pass AI research + human map review. Those verdicts are frozen
   history: every manifest row carries its provenance in a `source` column,
   and the discovery record lives in PROVENANCE.md / METHODS. Reproducing
   the database does **not** require re-running any audit. Section 5.
3. **Screen re-derivability** (optional, heavier). The deterministic screens
   that *nominated* the audit candidates (Natural Earth ladders,
   HydroRIVERS/HydroLAKES sweeps, the non-touching corridor census) can be
   re-run from public data to confirm the candidate lists. Section 6.

## 1. Prerequisites

- Python ≥ 3.11 with `geopandas`, `shapely`, `pyproj`, `pandas` (the shipped
  lengths were produced with geopandas 1.1.2 / shapely 2.1.2 / pyproj 3.7.2 —
  see the byte-identity note in §3).
- The repository checkout (the nine manifest CSVs and the engine ship in it).
- For the **full** rebuild only: the four pinned GeoPackages (~1 GB) and the
  bridge CSV below. The quick verification path (§2) needs no downloads.

**Pinned inputs** (SHA-256 also hard-coded in `scripts/build_all.py`, which
refuses to build from a mismatching file):

| input | source | SHA-256 |
|---|---|---|
| WB Admin 1 GeoPackage (layer `WB_GAD_ADM1`, 3,591 features) | World Bank Official Boundaries, **2026-05-14 release** — datacatalog.worldbank.org/search/dataset/0038272 | `dbac29f4…b64b45` |
| WB Admin 0 GeoPackage (layer `WB_GAD_ADM0`, 264 features) | same distribution | `97f0c8a0…f4b117e` |
| WB Ocean Mask GeoPackage | same distribution | `c2b074fd…c88d702` |
| WB NDLSA GeoPackage (24 disputed-area features) | same distribution | `159ef2d1…55d2fa4` |
| Bridge classification CSV (309 rows, reviewed static artifact) | `build_data/bridge_classified_authoritative.csv` (in-repo) | `87f03c1d…5c2abaa` |

(Full 64-character hashes: `data/PROVENANCE.md` → "Sources (pinned)".)

**What a clean clone contains.** Everything §2 and §4 need ships in the
repository, including the bridge classification CSV (tracked at its
`build_data/` path). The four GeoPackages for §3 must be downloaded from the
World Bank (their hashes are verified before any build). The wider
`build_data/` audit-evidence tree (screen outputs, frozen AI verdicts,
worksheets) is a local working archive, NOT shipped in the repository — its
content is summarized in PROVENANCE.md/METHODS and in each manifest row's
`source` column, and §6 re-derives the candidate screens from the public
datasets directly rather than from those local artifacts.

## 2. Quick path — verify the shipped data (minutes, no downloads)

```
python scripts/build_all.py
```

This (a) re-applies the entire reviewed correction layer to the shipped data
via `scripts/apply_overlays.py`, (b) recomputes every headline count from the
resulting CSVs, and (c) compares them to the expected values. On an untouched
checkout it must print `build_all: OK` with all twelve counts `OK` and report
the engine as a **byte-stable no-op** — i.e. applying the corrections again
changes zero bytes. This is the day-to-day reproducibility check; CI runs the
same property as a test (`tests/test_apply_overlays.py`).

Expected counts (current):

| count | value |
|---|---|
| ADM1 edges (lenient) | 8,460 (3,374 regions, 196 countries) |
| ADM1 moderate / stringent | 8,062 / 7,710 |
| water-only ADM1 | 750 = 352 with a fixed crossing / 398 without |
| ADM0 pairs (lenient / moderate / stringent) | 326 / 320 / 298 |
| ADM0 water roll-ups | 28 |

## 3. Full rebuild from the pinned sources (~1–2 h)

```
python scripts/build_all.py --full ^
    --adm1-gpkg  <path>\WB_ADM1.gpkg ^
    --adm0-gpkg  <path>\WB_ADM0.gpkg ^
    --ocean-gpkg <path>\WB_ocean_mask.gpkg ^
    --ndlsa-gpkg <path>\WB_NDLSA.gpkg
```

What happens, stage by stage (all inside `build_pericoupling_db.py` except
S4):

- **S1 — topology.** SHA-256 of every input is verified against the pins
  (hard error on mismatch). A **source-relabel** step first reassigns 10
  reviewed WB sliver-corridor artifacts to their true owner units
  (`scripts/relabel_sliver_corridors.py` + manifest
  `sliver_corridor_relabel.csv`). Then rook contiguity at **exact contact
  (tolerance 0)** over the land-clipped polygons: two units are adjacent iff
  their boundaries share a segment of non-zero geodesic length (a shared
  corner does not count). No lake filter — units meeting across a lake are
  native edges. One reviewed denylist entry removes a fabricated domestic
  exclave-contact sliver (Balzers↔Planken). Border lengths are full geodesic
  shared-boundary lengths (WGS84, `pyproj.Geod`).
  → 8,427 raw − 2 relabel − 3 unit merge (RUS050 → RUS024) − 3 denylist =
  **8,419** native pairs.
- **S2 — de-facto connectivity.** Each NDLSA disputed tract is folded into
  its de-facto administrator and adjacency re-measured; +13 ADM1 pairs whose
  sole link is a tract (+3 pairs at ADM0). Geometry-validated: an authored
  tract→unit attribution that does not touch its tract fails loudly.
  → **8,432**; this is the base file the geometry build writes.
- **S3 — water classification (descriptive).** Natural Earth lakes/rivers +
  the reviewed bridge CSV label borders `water_type`/`has_bridge` (309 base
  water-only rows). This stage never adds or drops an edge.
- **S4 — reviewed correction layer.** `scripts/apply_overlays.py` applies
  the nine manifests in registry order (idempotent, one pass; see §5).
  → +6 river-gap, +2 lake-gap, +4 land-gap, +16 rescreen-gap edges =
  **8,460**; water rows 309 → **750**; ADM0 roll-up recomputed once (a
  country pair is water-only iff *all* its ADM1 crossings are; bridged iff
  *any* is).

The command then verifies all twelve headline counts and reports whether the
rebuilt files are byte-identical to the committed ones.

**Byte-identity scope (important, honest caveat).** The adjacency *pair set*
reproduces exactly on any toolchain (verified by a clean-room rebuild
2026-07-02). The advisory `border_length_km` column can differ in low-order
digits under a different GEOS/Shapely version, because the geometry-cleaning
and ocean-clip operations preceding the length measurement are sensitive to
the buffer implementation — lengths never add or drop an edge, so this does
not affect the pair set or any count. Exact byte-identity of lengths
additionally requires the original toolchain (geopandas 1.1.2 / shapely 2.1.2
/ pyproj 3.7.2). The correction layer is toolchain-independent.

## 4. Run the test suite

```
python -m pytest tests/ -q
```

The suite (1,374 tests at the time of writing) includes: the expected-count assertions, the engine
byte-stability guard, registry-covers-all-manifests, loader behavior for
`de_facto_borders` × `coupling_standard`, and **doc-drift guards** that parse
`docs/METHODS_adjacency.md`, `INTRODUCTION.md`, and `MANUAL.md` and fail if
any headline count in the prose disagrees with the live data.

## 5. The reviewed correction layer — what it is and how to change it

Eleven reviewed manifest CSVs in `src/metacouplingllm/data/` govern the
build: the first two below are build-stage inputs (S1 relabel, S2 disputed)
and the other **nine** form the engine's correction layer (443 pair-rows:
29 edge-restoring + 414 water-flag-only). The engine
(`scripts/apply_overlays.py`) holds only behavior; editing a manifest and
re-running the engine is the supported way to change the correction layer:

| manifest | effect |
|---|---|
| `sliver_corridor_relabel.csv` | S1 input: 10 polygon relabels before contiguity |
| `disputed_overlay_pairs.csv` | S2 input: 13 ADM1 + 3 ADM0 de-facto pairs |
| `river_gap_overlay_pairs.csv` | +6 edges (non-touching river banks) |
| `lake_gap_overlay_pairs.csv` | +2 edges (Peipus, Skadar) |
| `land_gap_overlay_pairs.csv` | +5 land edges (sub-tolerance survey lines) |
| `wide_river_overlay_pairs.csv` | water flags on 13 edges |
| `audit_water_overlay_pairs.csv` | water flags on 3 edges |
| `hydro_water_overlay_pairs.csv` | water flags on 18 edges |
| `hydro_lakes_overlay_pairs.csv` | water flags on 12 edges |
| `rescreen_gap_overlay_pairs.csv` | +16 edges (2026-07 water-screen rebuild; per-row `water_type`) |
| `rescreen_water_overlay_pairs.csv` | water flags on 371 edges (rebuild batches b1–b6 + holds + the 2026-07-18 identity audit; per-row `water_type`) |

Engine semantics worth knowing:

- **Idempotent and byte-stable**: outputs are composed in memory and written
  only if bytes differ; `--check` exits 2 instead of writing. Running on
  already-corrected data is a no-op.
- **`note` strings are frozen identifiers**: each water row's `note` column
  ties it to its overlay; the engine syncs rows to their manifest by that
  string. Never edit the note constants in the registry.
- **Editing propagates**: change `has_bridge` in a manifest row, re-run the
  engine, and the water CSV row plus the ADM0 roll-up update on the next
  pass.
- **Every manifest row carries provenance**: the `source` column states how
  the row was discovered, adjudicated, verified, and (for water rows) how
  its bridge flag was classified.

**What is deliberately NOT re-run.** The AI adjudication (two-pass research +
adversarial judgment) and the human map reviews that *validated* each
manifest row are frozen history — re-running a live model would make the
build non-deterministic. The design principle throughout: *deterministic
screens nominate, frozen audits decide, the build replays manifests.* If you
distrust a row, its `source` string plus PROVENANCE.md tell you exactly which
screen found it, which audit design judged it, and who verified it.

## 6. Optional: re-derive the candidate screens (public data)

The audits' *candidate lists* came from deterministic screens over public
datasets; each can be re-run to confirm no candidate was hand-picked:

- **Natural Earth screens** (`ne_10m_lakes`, `ne_10m_rivers_lake_centerlines`,
  1:10M): river candidates = ≥ 0.50 of a shared border within geodesic
  2.5 km of a named NE river centerline (ladder rungs 2.5/5/10/15/20 km);
  lake candidates = ≥ 0.40 within 125 m of an NE lake polygon (rungs to
  1,500 m). Sampling every ~500 m geodesic along the border.
- **HydroRIVERS v10 / HydroLAKES sweeps** (hydrosheds.org): every border
  sampled against a geodesic **500 m** buffer (the datasets' positional
  accuracy); river nomination ≥ 0.5 coverage at discharge ≥ 10 m³/s plus the
  full creek band (≥ 0.5 at any discharge); lake bar 0.5.
- **Non-touching recovery census**: for unit pairs whose polygons do not
  touch, nominate when ≥ 0.80 of the corridor between the facing boundaries
  lies inside the union water mask (NE lakes ∪ 500 m HydroRIVERS buffer),
  with a short-corridor proximity amendment for sub-kilometre gaps.
- **Bridge screen** (OpenStreetMap Overpass): road/rail bridge, causeway,
  dam-top road, or tunnel intersecting both units' polygons (ferries and
  footbridges excluded) — layer 1 of the four-layer `has_bridge`
  classification (`docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md`).

Thresholds are anchored, not tuned: 2.5 km ≈ ½ × the NMAS horizontal
accuracy at 1:10M (0.5 mm map distance ≈ 5 km ground); 500 m = the
HydroSHEDS-derived datasets' stated positional accuracy; the rung ladders
were extended until the capture pattern was fully characterized. Screens
only ever **nominate** — no threshold ships a row by itself.

## 7. Loader-level reproduction (how users consume the data)

```python
from metacouplingllm.knowledge.adm1_pericoupling import is_adm1_pericoupled
is_adm1_pericoupled("USA044", "MEX028")                       # default views
is_adm1_pericoupled("ROU008", "ROU039",
                    coupling_standard="stringent")            # water policy
```

Two orthogonal toggles select the view: `de_facto_borders` (default `True`;
`False` removes the 13 ADM1 / 3 ADM0 disputed-overlay pairs) and
`coupling_standard` (`"lenient"` keeps every water border; `"moderate"`,
the default, keeps water-only pairs only when a fixed crossing open to
traffic links the two units; `"stringent"` drops all water-only pairs).
All counts in §2 derive from these two toggles and nothing else.
