# Methods: ADM1/ADM0 land-border adjacency construction

Companion to `src/metacouplingllm/data/PROVENANCE.md`. PROVENANCE records *what
the data is and its known flaws*; this document records *why the methodological
choices are defensible* — the contiguity rule, the snapping tolerance, the
prior-practice context, and the sensitivity analyses that back the parameters.

All numeric results below were computed from the World Bank Official Boundaries
2026-05-14 GeoPackages (ADM1 `WB_GAD_ADM1`, 3,591 polygons; ADM0 `WB_GAD_ADM0`,
264 polygons) with `scripts/build_pericoupling_db.py`.

## Construction and the source of every number (four-stage build)

The database is built by one command (`scripts/build_all.py --full`) in four stages, run in order
and each reproducible. The stages give the order in which the build applies its inputs, not the
order of the reviews that produced those inputs; the numbered sections below are the *discovery and
validation record* of those reviews, not steps a reader re-runs.

1. **Geometry (World Bank only).** Rook contiguity at **exact contact**
   (tolerance 0 — a parameter-free core; two units are adjacent iff their
   boundaries share a segment of non-zero geodesic length) over the
   SHA-256-pinned WB Admin-1/Admin-0 polygons, land-clipped by the WB ocean
   mask. `border_length_km` is the **full** shared length (no Natural Earth
   subtraction). Because WB polygons include lake water, pairs meeting across a
   lake are native edges here — **no lake filter removes them**, so no
   restoration overlay is needed. The two reviewed corrections that change
   polygons act before contiguity: a **source-relabel** reassigns 10 reviewed WB
   sliver-corridor artifacts (thin border-tracing ribbons of one unit's land
   mislabeled onto a neighbour) to their true owners
   (`scripts/relabel_sliver_corridors.py`; §10), and a **unit merge** rejoins
   Kalmykia's split western salient (RUS050 → RUS024; §10).
2. **De-facto connectivity (NDLSA).** Folds each disputed tract into the
   administrator Natural Earth records for most of it and re-measures adjacency;
   adds the pairs whose sole land link is a tract (§5). Geometry-derived and
   validated per tract. The geometry build (`scripts/build_pericoupling_db.py`)
   ends here and writes the base edge list.
3. **Water-only classification.** One reviewed file,
   `data/water_classification_pairs.csv` (803 rows), flags **779** existing edges
   water-only — each with its water type, water body and crossing flag — and
   adds the **24** water borders between units whose banks the source digitizes
   apart, each with its water row; `coupling_standard` reads the result. Every row
   was nominated by a **deterministic Python screen** before any human or AI
   adjudication — the 779 by the edge screens, which run on one definition of the
   shared border, the border arc of §8 (the Natural Earth river and lake ladders,
   HydroRIVERS, HydroLAKES, a combined river and a combined lake screen, and a
   cross-type union), the 24 by the corridor census of non-touching pairs (§8) — so
   the discovery is auditable and the verdicts are frozen (whole-graph
   attributability 803/803).
4. **Edge corrections.** Reviewed corrections that add or remove non-water edges:
   **+4** genuine sub-tolerance land borders the exact-contact rule misses
   (`data/land_gap_overlay_pairs.csv`; found by the tolerance-band audit, §2.4/§4)
   and **−5** contacts found in the water adjudication not to be borders
   (`data/denylist_pairs.csv`; each checked present in the geometry before it is
   removed). The ADM0 water roll-up is then computed once. Stages 3 and 4 are
   applied by `scripts/apply_overlays.py`.

**ADM1 provenance chain (every number reproducible from the shipped CSVs):**

Every pipeline step appears in **execution order**, and each row shows the edge count *before → after*; the *water-only* column is the running count that drives the three views (803 = 407 with a fixed crossing / 396 without). S1–S2 = the geometry build (`scripts/build_pericoupling_db.py`, which writes the 8,438-row base file, the five denylisted contacts still in it); S3–S4 = the reviewed inputs applied by `scripts/apply_overlays.py`.

| step (execution order) | ADM1 edges | water-only | source / reason |
|---|---|---|---|
| **S1** exact-contact contiguity, raw WB polygons | — → 8,427 | — | tolerance 0, ocean-clip, no lake filter, full geodesic lengths — before the source-relabel |
| **S1** − source-relabel of cross-country sliver corridors | 8,427 → 8,425 | — | §10: removes 4 fabricated cross-country edges (Migori↔Arusha, Taita-Taveta↔Arusha, Salta↔Potosí, Braničevo↔Mehedinți), makes 2 Kenya survey-line pairs native (Kajiado↔Kilimanjaro, Narok↔Mara) |
| **S1** − unit merge (reviewed source-data artifact) | 8,425 → 8,422 | — | §10 / `docs/FUTURE_EDGE_AUDITS.md` #10: RUS050 ("Name Unknown", GAUL-split western salient of Kalmykia — Gorodovikovsky + Yashaltinsky raions) merged into RUS024 — a net −3 edge rows: the internal 19.8 km raion-line edge dissolves, and the salient's Rostov (187.3 km) and Stavropol (143.6 km) frontage rows fold into the existing RUS024 edges (388.7 → 576.0 km; 282.9 → 426.6 km) instead of remaining separate (maintainer decision 2026-07-18) |
| **S2** + de-facto disputed overlay (NDLSA) | 8,422 → 8,438 | — | +16 province pairs whose sole link is a disputed tract; **the geometry build writes this base file** |
| **S3** water-only classification of existing edges | 8,438 → 8,438 | — → 779 | 779 rows of `data/water_classification_pairs.csv` on a shared edge (668 river / 111 lake; 401 with a fixed crossing; tiers A 244 / B 238 / C 297), each nominated by the edge screens on its border arc (§8) and adjudicated by the two-model design; discovery provenance per row in `source`, composition in `data/PROVENANCE.md` |
| **S3** + water borders between non-touching units | 8,438 → 8,462 | 779 → 803 | +24 edges with their water rows (`adds_edge`; 21 river / 3 lake; 6 with a fixed crossing; tier A), nominated by the corridor census (§8) and adjudicated |
| **S4** + land-gap borders (*discovered* by the S1 tolerance-band audit, §2.4/§4) | 8,462 → 8,466 | 803 → 803 | +4 genuine sub-tolerance land borders (Egypt–Libya 0.4 m … a domestic Anguilla pair); no water rows |
| **S4** − denylist of reviewed non-adjacent contacts | 8,466 → 8,461 | 803 → 803 | `data/denylist_pairs.csv`, 5 contacts found in the water adjudication not to be borders (register `docs/FUTURE_EDGE_AUDITS.md` #7, #8, #11–#13): Grand Gedeh↔Rivercess (a quadripoint stretched into a 1.12 km cardinal-leg connector, a World Bank shape signature) and Apure↔Amazonas (a 2.35 km mid-river seam contradicted by Amazonas' territorial-division law), maintainer decisions 2026-07-18; the Kasba Lake four-corners point CAN003↔CAN006, a point contact on the Lake Edward boundary COD009↔UGA102 and a diagonal non-adjacency in Lake Victoria TZA016↔UGA040, whose World Bank arcs are stable across tolerances and which rest on the maintainer's official-map check (2026-07-25) |
| **shipped (lenient)** | **8,461** | 803 | 3,374 regions, 196 countries — every edge kept |
| moderate (default) | **8,065** | −396 | 8,461 − 396 water-only pairs with no fixed crossing |
| stringent | **7,658** | −803 | 8,461 − all 803 water-only pairs |

**What Stage 1 changed vs a naïve build (add/remove reasons):** removed 4
fabricated cross-border edges (Salta↔Potosí, Braničevo↔Mehedinți, and the two
Arusha edges) that were pure sliver-corridor artifacts; made native the
lake-meeting pairs and the placeholder Nyasa pair (no lake filter) and the two
Kenya survey-line pairs (Kajiado↔Kilimanjaro, Narok↔Mara — the relabel gives
their border territory back to Mara/Kilimanjaro); the Malta false edge
(Balzan↔Iklin, ~31 m apart) never appears at tolerance 0, so it needs no
denylist entry.

**Water-only set = 803 ADM1** (407 with a fixed crossing / 396 without) + **26
ADM0** roll-ups, from one file (`data/water_classification_pairs.csv`): 779 on a
shared edge and 24 between non-touching units. Every one of the 803 is nominated
by the screens run over every edge: the 779 by the edge screens on their border
arc (§8), the 24 by the corridor census (whole-graph attributability 803/803).
("Every edge" means every edge with a shared border arc to sample: 8,421 of the
8,461. The 16 de-facto overlay edges meet only across a disputed tract, so they
have no arc in the standard layer, and none is water-only; the 24 non-touching
pairs are the corridor census's domain.) Earlier discovery campaigns — the ~1 km
near-miss net and the 1–100 km lake band (6 of the 24 non-touching rows), the
5 km/10 km widening re-screens (15 rows on existing edges), and the 2026-07-02/04
HydroRIVERS/HydroLAKES geodesic 500 m cross-checks (18 + 12 rows on existing
edges, incl. the Uruguay River ARG008↔URY012 with the San Martín bridge and the
Dead Sea ISR005↔JOR007, no crossing) — are retired as nomination steps
(2026-09-09): their rows are independently re-nominated by the current screens,
and each keeps its discovery provenance in `source`. The 298 rows of the first
classification round (238 river / 60 lake) each carry a cross-vendor
re-adjudication — the river rows from ru1 (2026-07-21), the lake-class rows from
wu1 (2026-07-25) — and the correction history is in `CHANGELOG.md`. Twelve
borders were **reclassified land→water** by the HydroLAKES sweep (e.g. the Great
Lakes, Lake Malawi median, Lake Chad).

**ADM0 = 326** shipped (323 native + 3 disputed; COD↔TZA and other lake-only
country pairs are now native, no matrix patch), **320** moderate, **300**
stringent. Five country borders are all-water because every ADM1 crossing
between the two countries is water-only: DEU↔LUX (the Our–Sauer–Moselle chain,
every crossing bridged), BEN↔NER (the Niger + Mékrou, Malanville bridge), CMR↔GAB
(the Ntem, bridged), MWI↔TZA (the bridged Songwe plus the Lake Malawi/Nyasa
segments) — these four stay in the moderate view — and GUY↔SUR (the Corentyne,
ferry-only, no fixed crossing), which drops from moderate; a sixth, MOZ↔TZA, is
all-water through the three Rovuma pairs plus the Lake Nyasa corner
MOZXXX↔TZA025 (the Unity Bridge pairs keep it in moderate); stringent drops all
26 roll-ups.

---

> **Note.** Sections 2–4 below are the *validation record* for the shipped parameter-free design: why **exact contact (tolerance 0)** is safe (the tolerance sensitivity sweep and the sub-55 m band audit), and why `border_length_km` is the full geodesic shared-boundary length. The audits described here produced the reviewed correction manifests the build replays; the authoritative counts are the provenance ledger above (shipped 8,461 / moderate 8,065 / water-only 803).

## 1. Contiguity rule: rook, not queen

Two units are adjacent **iff their polygon boundaries share a segment of
non-zero (line) length**. A contact at a single shared vertex/corner does not
count. In the spatial-analysis vocabulary this is **rook contiguity**; the
looser **queen** rule additionally counts single-vertex contacts (Anselin,
GeoDa workbook; Rey, Arribas-Bel & Wolf, *Geographic Data Science*, CRC Press,
2023).

This choice is deliberate and is **not** the default of common geometry
predicates. The OGC/DE-9IM `touches` predicate (PostGIS `ST_Touches`, JTS/GEOS,
Shapely `.touches`) is **queen-inclusive**: it returns true for a single shared
corner. A rook rule must be enforced explicitly — e.g. via a DE-9IM pattern
requiring a line-dimension boundary/boundary intersection (`ST_Relate(...,
'F***1****')`). We enforce it by taking the boundary∩boundary intersection and
keeping the pair only when its geodesic length is > 0.

**Why rook here.** Single-vertex contacts at administrative boundaries are
predominantly **quadripoint artifacts** — places where two nearby tripoints are
drawn at one coordinate (see PROVENANCE, "collapsed-node artifacts"; e.g. the
Congo Pedicle). Counting them as adjacency would manufacture neighbor relations
between units that only meet at a mathematical point. Rook contiguity excludes
these by construction.

---

## 2. Why exact contact (no snapping tolerance)

### 2.1 The failure mode a tolerance would address

Adjacent administrative polygons digitized from independent sources sometimes
do **not** share identical vertices along their common boundary; the two
renderings of the same line sit a small distance apart. An exact-intersection
test then returns *empty* and misses a real border. A snapping tolerance would
bridge this sub-tolerance offset — but the build deliberately does **not** use
one for topology: the main contiguity test is exact contact (`TOPOLOGY_TOL_DEG
= 0`), and the few genuine sub-tolerance borders are added explicitly as
reviewed manifest rows (the land-gap overlay, §4) instead of by loosening a
global threshold. (`SNAP_TOL_DEG = 5×10⁻⁴°` ≈ 55 m survives in the code for two
narrow purposes only: the disputed-overlay tract-touch validation of §5 and the
source-relabel's test of which reviewed owner a detached corridor touches
(§10.3), never the main contiguity.)

Concrete illustration (verified): the Egypt–Libya border between Matrouh
(`EGY016`) and Ajdabiya (`LBY001`) is a ~49 km straight segment whose two
renderings are offset by **~0.4 m** and share **zero** vertices. Exact
intersection recovers **no** shared boundary there — which is exactly why that
pair ships as a reviewed land-gap overlay row. The sections below show why this
explicit-manifest design beats a tuned tolerance: no threshold value separates
the genuine sub-tolerance borders from the artifacts (§2.4, §4).

### 2.2 There is no canonical value to cite — and that is the documented state of practice

A literature/tool survey (GIS-topology and spatial-data-quality sources) found
**well-cited standards for the *mechanism* of contiguity but no canonical
numeric snapping-tolerance value** — a key reason the build avoids a tuned
tolerance entirely:

| Tool / standard | Tolerance | Notes |
|---|---|---|
| ArcGIS Pro "XY / cluster tolerance" | **0.001 m (1 mm)** default; = 10× the XY *resolution* | A vertex-*coincidence* threshold; docs say it "should never approach your data capture accuracy" and should exceed ~2× resolution (Esri ArcGIS Pro topology docs). |
| GRASS GIS `v.clean` (snap) | **no default** (user-supplied `thresh`) | In map units; **degrees** for lat-long data (GRASS manual). |
| PostGIS `ST_Snap` | **no default** (mandatory argument) | (PostGIS docs). |
| GeoDa "precision threshold" | ~`0.0001` desktop / `0` API | "fuzzy" band used only when coordinate precision is insufficient for exact match; in layer coordinate units → degrees for unprojected data (GeoDa workbook; rgeoda/pygeoda). |
| NMAS (1947), NSSDA (FGDC-STD-007.3-1998) | **set no threshold** | NSSDA §3.1.2 verbatim: "This standard does not define threshold accuracy values." Reporting standards (NSSDA: Accuracy_r = 1.7308·RMSE_r at 95%), not tolerance prescriptions. |

Two important cautions from the survey:
- ArcGIS's 1 mm default and the ~55 m band examined here are **different
  quantities**: ArcGIS's is a *coincidence* threshold (snap vertices meant to
  be identical), the band is a *cross-source-discrepancy* scale (two renderings
  of the same border). They are not comparable.
- The intuition "set tolerance ≈ source positional accuracy (RMSE)" is **not**
  supported: ArcGIS guidance is the opposite (tolerance ≪ capture accuracy), and
  the epsilon-band/RMSE link was not corroborated in the surveyed sources.

### 2.3 Data-relative anchor

The WB ADM1 coordinates are stored at full double precision (no fixed grid;
median ~14 decimal places). Their **median inter-vertex spacing is ≈5.4×10⁻³°
(~600 m; ≈5.9×10⁻³° along shared borders)**
(`build_data/doc_checks/vertex_spacing.py`), far coarser than the offsets
between two renderings of one border that exact contact can miss (0.4–39 m
for the genuine borders of §4). The band audit (§4) examines the [0, 55 m]
band (5×10⁻⁴°) pair by pair, and the sensitivity sweep below extends the net
an order of magnitude further, to 555 m, about one median vertex spacing.

### 2.4 Sensitivity analysis (the primary justification)

Because no canonical tolerance value exists, the defensible justification for
shipping exact contact is **robustness**: sweep a hypothetical snapping
tolerance and show the graph barely moves. The sweep runs on the shipped
geometry (source-relabelled, ocean-clipped, no lake filter), with the candidate
search widened by each tolerance — as a tolerance build's must be, so no
within-tolerance pair escapes the test. Counts are native (before the unit
merge, the denylist and the correction overlays); measured 2026-09-22 by
`build_data/snap_sweep/snap_sweep_tol.py`, record
`build_data/snap_sweep/snap_sweep_fixed_relabel.txt` (the script reproduces the
earlier record `build_data/snap_sweep_tol0_report.txt` exactly on the polygons
that record was measured on; the ADM0 column, which the relabel does not touch,
is that record's):

| tolerance (°) | ≈ metres | ADM1 edges | Δ vs 0 | ADM0 pairs |
|---|---|---|---|---|
| **0 (recorded)** | **0** | **8,425** | — | **323** |
| 1×10⁻⁴ | 11 | 8,428 | +3 | 323 |
| 2×10⁻⁴ | 22 | 8,428 | +3 | 323 |
| 5×10⁻⁴ | 55 | 8,431 | +6 | 323 |
| 1×10⁻³ | 111 | 8,436 | +11 (+0.13%) | 323 |
| 2×10⁻³ | 222 | 8,451 | +26 (+0.31%) | 324 |
| 5×10⁻³ | 555 | 8,515 | +90 (+1.07%) | 324 |

Over **[0, 10⁻³°]** the ADM1 edge count rises by at most 11 edges (~0.13%) and
the ADM0 matrix is **exactly invariant at 323** (rising to 324 only at
≥2×10⁻³°: one Namibia–Zimbabwe pair, gap ~135 m). Counts climb materially only
above ~2×10⁻³°, where a tolerance grows large enough to fuse genuinely separate
units across rivers and straits. Exact contact therefore sits at the foot of a
low-sensitivity plateau, an order of magnitude below the width at which
spurious edges appear.

The **[0, 55 m] band holds exactly six pairs**: the four genuine land-gap
borders (Egypt–Libya 0.4 m, Anguilla 1.3 m, Ethiopia–Sudan ~31 m, Dominican
~39 m — shipped as the reviewed land-gap overlay, §4) plus two **rejected
artifacts** — the placeholder–Malawi lake corner (0.9 m), first restored and
then reversed on 2026-07-18 as a true point contact; and the Malta Balzan↔Iklin
councils (31 m), which exact contact excludes for free. (Braničevo↔Mehedinți,
the fabricated corridor edge the source-relabel removes, meets at a single
point where the corridor closes, §10, and no tolerance turns a point into a
shared segment.) Genuine and artifact gaps **interleave**
(0.4 / 0.9 / 1.3 / 30.9 / 31.3 / 39.2 m), so no tolerance value separates
them — the per-pair audit, not a threshold, is load-bearing.

The **84 pairs the sweep adds beyond the 55 m band** (+5 at 1×10⁻³, +15 at
2×10⁻³, +64 at 5×10⁻³) were audited per pair with the §4
recovered-length-vs-tolerance diagnostic: **every one shows the growing
gap-corridor signature** and none the flat genuine-border signature. Every
cross-border addition falls within the ground-truthed near-miss census set:
three are river recoveries restored to the shipped graph (non-touching water
borders of Stage 3, folded in 2026-07-22), one — Cahul↔Vaslui —
is the four-corner artifact the rg1 re-adjudication demoted, and the
remaining additions are confirmed artifact corridors. The per-pair record
below holds 87 pairs, three of which the sweep no longer adds: a fourth river recovery,
Bor↔Caraș-Severin, which now meets at a single point where the relabel closes
Braničevo's corridor (§10.3), and the two long survey-line corridors on the
straight-surveyed Kenya-Tanzania border — Kajiado↔Kilimanjaro (47.9 km as a native edge; ~54 km on the recovered-length diagnostic at 1×10⁻²°) and Narok↔Mara (67.5 km native; ~70 km at 1×10⁻²°) — which were
map-verified as genuine borders and are **native** edges after the
source-relabel (§10).
Per-pair results: `build_data/snap_extras_audit/extras.csv` (local audit
artifact, not shipped).

### 2.5 Units policy: degrees for geometry operations, geodesic metres for measurements

The pipeline deliberately mixes two unit systems, under one rule: **degrees for
geometry *operations* on the native-CRS polygons; geodesic metres/kilometres
for any quantity that *measures* the real world.**

*Why the geometry operations run in degrees.* The WB polygons are stored in
longitude/latitude on the WGS84 datum (EPSG:4326 — the layers declare it, and
the build's `.to_crs(4326)` normalises any input that did not), so every
global geometry pass — the contiguity
test, spatial indexing, the morphological opening, the tract-touch test — runs
on those native coordinates. This is a limit of the geometry *engine*, not of
WGS84: shapely is **planar** (it treats coordinates as flat *x*, *y*, with no
geodesic-aware buffer or intersection), so to run its operations "in metres"
you would first have to flatten the globe onto a metre plane — a projected CRS
— and no single projection is accurate for the whole planet (UTM alone needs
60 zones; per-pair local projections cannot back one global spatial index).
Note this constrains *operations* only: metric *measurement* is global and
exact on WGS84 via geodesics (next paragraph), needing no projection — which is
exactly why the measured quantities can be metric while the operations stay in
degrees. This is doubly safe here because the
operations are *topological or artifact-scale*: the edge test is **exact
contact (tolerance 0), which is unit-free** — the same in degrees, metres, or
anything else — so the "how many metres is a degree here?" question vanishes
from the shipped build. The surviving degree constants (`2×10⁻³°` opening
radius, `5×10⁻⁴°` disputed-tract and relabel-owner touch, and the water screens' land-gap band of
`5×10⁻⁴°`, used for the four land-gap pairs only — every other edge's border arc
starts from its exact shared line, 8,417 of the 8,421 edges the screens cover) are
artifact-scale thresholds whose
east–west metric width shrinks by cos(latitude) (for 5×10⁻⁴°, ≈55 m N–S everywhere; ≈39 m
E–W at 45°, ≈28 m at 60°); at tens-of-metres scale that anisotropy never flips
a verdict, and it is stated rather than hidden.

*Metric quantities split by role, not all geodesic.* The metric quantities are
**measurements compared against physical scales** (river widths, dataset
positional accuracy, digitization offsets), which are isotropic — so they must
not depend on latitude. But the pipeline computes them two ways, and the split
is deliberate:

- **Load-bearing metrics are geodesic and exact.** Where the number itself
  decides a shipped result, it is computed on the WGS84 ellipsoid with
  `pyproj.Geod` (latitude-independent, no projection): the reported
  `border_length_km` (`Geod.geometry_length`) and the ~1 km near-miss census
  cutoff (`GEOD.inv` between the two nearest points). The census's coarse
  candidate pre-filter is a generous ~7 km *degree* buffer — over-inclusive by
  ~7×, so its distortion can never drop a true sub-1 km pair — after which the
  geodesic gap does the actual filtering. (Using a flat `°→km` constant for the
  cutoff itself is exactly the bug that transiently produced a census of 38
  instead of 41; see the cautionary example below.)

- **Why the hydro widths are single-valued while the Natural Earth widths are
  laddered (diagnostic 2026-09-10).** HydroRIVERS is derived from 15 arc-second
  HydroSHEDS, so the geodesic 500 m sample-to-reach width is one cell of the
  dataset's own positional accuracy; Natural Earth's 1:10M centerlines can sit
  kilometres from the bank, which is why that width is wide and laddered. A
  wider hydro width would nominate on proximity to neighbouring rivers rather
  than on the border's own water, and the one case a fixed 500 m under-measures
  — a wide channel whose modelled reach lies more than 500 m from a median- or
  bank-line border — is already carried by the Natural Earth rung, wide rivers
  being named rivers: 32 of the 668 shipped river borders on existing edges
  fall below the 0.50 bar at 500 m (Orinoco, Paraguay, Elbe estuary,
  Yellow River, Danube, Rhône, Potomac, Uruguay), every one nominated by the
  Natural Earth river ladder on its border arc (`gd1_screens.csv`). Re-running the edge screen
  with the corridor census's discharge-scaled ladder (1,000 / 2,500 m for
  reaches ≥ 1,000 m³/s) raises the big-river nominations from 176 to 250 and
  adds no unaudited candidate — the 26 added non-water edges were all
  adjudicated in the 2026-07 run of the water screens (22 from its audit queue, 4 earlier rejections re-adjudicated on 2026-09-01);
  `build_data/water_screen_rebuild/hydro_bigreach_ladder_edges_2026-09-10.txt`.
  The ladder is therefore reserved for the non-touching corridor census, where
  a bank-line rendering places the reach half a channel width from each
  polygon edge by construction.
- **The water-only screens are nomination screens (thresholds nominate, audits
  decide), now measured geodesically.** They do *not* decide membership — they
  nominate candidates adjudicated per pair. The June base run's river-centerline
  screen used a ~2.5 km degree approximation (`d° · 111.32 · cos φ`); every current
  screen — the Natural Earth river and lake ladders and the HydroRIVERS and
  HydroLAKES 500 m buffers — uses a **geodesic** metric (`GEOD.inv`, a uniform
  width in metres with a cos-φ-widened candidate query), because a
  raw-degree buffer reaches only `X · cos φ` metres east–west and so
  *under*-measures E–W distance at latitude — the one direction a completeness
  screen must not err (a planar control reproduces the frozen numbers bit-for-bit,
  proving only the metric changed). Since 2026-09-23 the nearest point is also *located* on the
  ground (`gd1`, §8): the earlier helper located it in plain degrees, which weighs an east–west
  offset 1/cos²φ too heavily and overstated the distance to an oblique feature — the same failure
  direction, a feature within the width read as outside it — by up to 88–106% at 60–70°
  (`build_data/geodesic_distances/validate_rule.txt`). Each of these screens samples each continuous piece of the
  border arc (§8, `ba1`) at equal geodesic intervals, n = max(2, ⌊L/500 m⌋ + 1)
  intervals for a piece of geodesic length L, so every interval is shorter than
  500 m. The geodesic buffer feeds both hydro
  row families (carried in the water file since the 2026-07-28 consolidation;
  both cross-checks are retired as nomination steps, every row being
  re-nominated by the current cross-border hydro rung). The **HydroLAKES** sweep is the hydro-lakes row family: 19 cross-border lake
  candidates — the 15 borders not yet water-only among the first sweep's 27 flags (11
  accepted, 2 mixed, 2 left as they were and accepted by the later audit) and 4 more
  from the geodesic 500 m re-run (3 rejected as mixed: Burundi–Rwanda; two
  Norway–Sweden) — 12 shipped, including the **Dead Sea** (Southern District↔Karak,
  no crossing). The **HydroRIVERS**
  sweep is the hydro-water row family (18 borders); the geodesic metric recovered one the
  raw-degree screen had missed — the **Uruguay River** (hidden by ~0.04 of E–W
  anisotropy: planar coverage 0.46 vs geodesic 0.54). There is no separate
  hydro-geodesic overlay. The 2.4→2.5 km invariance and
  the 5 km/10 km completeness re-screens bound the river screen; extending it to
  **25/50/100 km** confirms the bound — the candidate net grows to 1,399 but adds
  no genuine water-only border (96% of the wide-buffer-only candidates have < 20%
  of their border within 2.5 km of a river — *near*, not *on*, the border, so they
  cannot be water-only). Completeness is fixed by the buffer-independent
  database-wide HydroRIVERS/HydroLAKES sweeps, not the NE buffer width
  (`build_data/_archive_pre_rebuild/wide_river_audit/screen_ladder_to_100km.py`).

*Cautionary example (why the boundary between the two matters).* During the
census re-verification, one diagnostic converted degree distances with a flat
`× 111.32` — pretending 1° ≈ 111 km in every direction. That inflated
east–west gaps at mid-latitudes and transiently produced a census of 38
instead of the correct **41**: three genuine pairs with geodesic gaps of
0.85–0.99 km (Austria↔Liechtenstein, Azerbaijan↔Turkey, Moldova↔Romania) read
as >1 km and were wrongly excluded. The error was caught by validating the
method against the frozen original 48-pair roster (it must regenerate 48/48)
before trusting its output on the current geometry
(`build_data/build_record/census_definitive.txt`). Rule of thumb:
convert degrees to metres only through a geodesic, never through a constant.

---

## 3. Border length: full geodesic shared-boundary length (advisory)

`border_length_km` is the **full geodesic length of the shared boundary**
(boundary∩boundary, measured on the WGS84 ellipsoid with
`pyproj.Geod.geometry_length`) — no Natural Earth river or lake subtraction of
any kind. Water status is a separate, purely descriptive layer (§8,
`water_separated_pairs.csv`) that never edits a length and never adds or drops
an edge, so a river border's full frontage is reported even where the border
runs mid-channel. It is the exact shared line only, so it understates the
borders where the two polygons overlap or face each other across a channel
(the four Saint-Louis borders by an order of magnitude); computing it from the
border arc of §8 is deferred (`docs/FUTURE_WORK.md`).

The length column is **advisory**: it feeds only the two flags
`narrow_border` (< 5 km) and `potential_artifact` (< 1 km), which nominate
pairs for audit attention and never remove a pair. Adjacency itself is decided
solely by the exact-contact rook test of §1 — a pair with a 40 m shared segment
and a pair with a 4,000 km one are equally edges; the flags simply record which
deserve scrutiny. (The sliver-corridor audit of §10 is the systematic follow-up
those flags motivated.)

---

## 4. Auditing the sub-55 m band (the land-gap overlay)

Exactly **6 ADM1 pairs** sit in the [0, 55 m] band — present the moment a
snapping tolerance reaches their gap, absent at exact contact (§2.4). Because
this set is small it was **manually reviewed against imagery**, with a
recovered-length-vs-tolerance diagnostic:

| pair | min gap | recovered length vs tolerance | verdict |
|---|---|---|---|
| EGY016 Matrouh ↔ LBY001 Ajdabiya | 0.4 m | ~constant (49.4 km) → real border | genuine — land-gap overlay row |
| **MOZXXX ↔ MWI003 (MOZ placeholder unit)** | **0.9 m** | grows with tol | **artifact — a point contact at a four-unit lake corner** (first restored as a land-gap row; reversed 2026-07-18 by maintainer map ruling) |
| AIA001 ↔ AIA003 (Anguilla, domestic) | 1.3 m | grows with tol | genuine — land-gap overlay row |
| ETH013 Tigray ↔ SDN004 Kassala | ~31 m | grows with tol | genuine — land-gap overlay row |
| DOM011 Independencia ↔ DOM026 San Juan | ~39 m | grows with tol | genuine — land-gap overlay row |
| **MLT002 Balzan ↔ MLT019 Iklin** | **31 m** | grows with tol | **artifact — excluded at exact contact** (the councils share no frontier) |

A **flat** recovered-length-vs-tolerance curve indicates a true shared border
(the whole length is at sub-metre offset, so loosening a tolerance adds
nothing); a **growing** curve indicates a gap corridor being progressively
swallowed. By this diagnostic only EGY/LBY is unambiguously a real border; the
others required human judgment — which is exactly why the band is audited
rather than trusted to any threshold value. Note that min-gap alone does
**not** separate genuine from artifact (Malta's artifact gap of 31 m is
*smaller* than the genuine Dominican pair's ~39 m; the placeholder–Malawi
corner's 0.9 m is smaller than Anguilla's genuine 1.3 m), so **no tolerance
value can get all six right** — the per-pair audit is load-bearing.
Braničevo↔Mehedinți, the fabricated corridor edge the source-relabel removes,
is not in the band: the corridor closes at the Caraș-Severin–Mehedinți
junction, so the two units meet at a single point (§10.3).

The four genuine pairs ship as the reviewed **land-gap overlay**
(`land_gap_overlay_pairs.csv`, applied by `scripts/apply_overlays.py`) —
ordinary land edges, pericoupled under every `coupling_standard`; the two
artifacts never enter the graph at exact contact, so they need no denylist
entry. The full count chain is the provenance ledger at the top of this
document.

**Takeaway.** A snapping tolerance is neither necessary nor sufficient here:
the genuine sub-tolerance borders enter as four explicit reviewed manifest
rows, and the artifacts are excluded for free by exact contact.

---

## 5. Disputed territories (de-facto vs strict)

WB's standard ADM0 (264-unit) and ADM1 (3,591-unit) layers exclude the
24-feature NDLSA disputed-areas layer. That exclusion carves each contested
tract out of **both** neighbouring units, opening a multi-km gap, so the
flanking units are recorded as non-adjacent even where they meet along the
de-facto line of control (e.g. China–Pakistan across Gilgit-Baltistan;
Israel–Syria across the Golan; Morocco–Mauritania across Western Sahara).

**Default = de-facto view.** Because metacoupling concerns connection, the
shipped data treats disputed land as part of its de-facto administrator. The
overlay is **derived from geometry at build time** by `derive_disputed_overlay()`
and ships (all 19 rows) in `data/disputed_overlay_pairs.csv`; the runtime loaders
accept `de_facto_borders` (default `True`) and, with `False`, subtract the
overlay to reproduce the WB standard-layer adjacency.

**Administrators.** The NDLSA layer carries **no**
administering-country field — `SOVEREIGN` is null for all 24 tracts and
`WB_STATUS` is uniformly "Non-determined legal status area". Each tract is
therefore assigned, **whole**, to the administrator that **Natural Earth v5.1.2**
(the latest release: its disputed-areas layer, then its countries layer for the
rest of the tract) records for the largest part of it, a dependency counting for
its sovereign; a tract whose largest part has no administrator the World Bank
lists as a country stays unassigned. There are no exceptions: an area Natural
Earth records separately inside a tract does not override the majority (Western
Sahara goes to Morocco whole, although Natural Earth records its eastern third,
33.5%, as self-administered; Jammu and Kashmir to India whole, although Natural
Earth records the Siachen Glacier, 1.9%, with claims only; the Ilemi Triangle to
Kenya whole, although Natural Earth records a 17 km² area inside it, 0.5%, as South
Sudanese). **23 of the 24 tracts are assigned**; the UN Buffer Zone in Cyprus
(Natural Earth: UN zone or Northern Cyprus for 89%) is not. The three island tracts
(British Indian Ocean Territory, the Falklands, South Georgia & the South Sandwich
Islands) go to the United Kingdom and, touching no unit, add no pair. The mapping
describes de-facto administration for a connectivity dataset and is **not** a legal
or endorsed sovereignty claim. Per-tract shares and records:
`build_data/ndlsa_ne/SPEC_da1_defacto_administrators.md` (the specification, with
the maintainer's rulings verbatim) and `docs/ndlsa_tract_audit.csv`. The tract the
layer names Karakoram Range is the Shaksgam Valley (Natural Earth: "Admin. by China;
Ceded to China by Pakistan; Claimed by India"). History: `CHANGELOG.md`.

**Validation.** Every assigned tract must touch its administrator's territory — the
administrator's own polygon or another tract assigned to it — unless it touches no
unit at all (the three islands); every authored province must touch its tract. A
mislabel fails the build loudly rather than silently dropping a pair.

**ADM0 (country level)** — three pairs, each the de-facto administrator country
re-joined to the neighbour it meets only across a disputed tract:

| pair | de-facto admin | tracts of the admin (row label) | restored border |
|---|---|---|---|
| `CHN`/`PAK` | PAK | Gilgit-Baltistan | ~455 km |
| `ISR`/`SYR` | ISR | Golan Heights; No Man's Land; Shebaa Farms | ~79 km |
| `MAR`/`MRT` | MAR | Western Sahara | ~1,544 km |

A restored border is measured between the two countries' de-facto territories (each
country's polygon with the tracts assigned to it), so China–Pakistan includes the
Karakoram crest line between Gilgit-Baltistan and the Shaksgam Valley. A row's label
lists all the tracts of its administrator (No Man's Land lies on Israel's border with
Palestine, not with Syria).

**ADM1 (subnational level)** — derived **independently**, because country
adjacency does *not* imply province adjacency: two countries adjacent elsewhere
can still have the provinces flanking a tract meet only across it (India and
China are adjacent, but Arunachal Pradesh and Tibet are non-adjacent in the
strict layer — the tract sits between them). Driven by a geometry-validated
tract→**province** map (`_NDLSA_TRACT_ADM1`: each tract's administering
province(s); for Lapthal, Shipki Pass, the Karakoram Range, No Man's Land and Abyei,
the administrator's provinces that touch the tract); a tract spanning
several administering provinces has its frontier split among them by nearest
province. **16 pairs**:

| de-facto admin province | neighbour province(s) | tract | border |
|---|---|---|---|
| `IND003` Arunachal Pradesh | `CHN029` Tibet | Arunachal Pradesh | ~993 km |
| `IND003` Arunachal Pradesh | `BTN016`/`BTN015`/`BTN011` (E. Bhutan) | Arunachal Pradesh | 86/83/41 km |
| `ISR004` Northern | `SYR012` Quneitra | Golan | ~74 km |
| `ISR004` Northern | `SYR006` Dar'ā / `LBN004` Bekaa | Golan | 4.8 / 0.4 km |
| `ISR003` Jerusalem | `PSE013` Ramallah | No Man's Land (Latrun) | ~12 km |
| `BTN005` Haa | `CHN029` Tibet / `IND030` Sikkim | Doklam | 62 / 6.6 km |
| `MAR005` Guelmim + `MAR007` Laâyoune | `MRT012` Tiris-Zemmour | Western Sahara | 287 + 757 km |
| `MAR007` Laâyoune | `MRT004` Dakhlet-Nouadhibou / `MRT001` Adrar | Western Sahara | 405 / 96 km |
| `SDN012` Southern Darfur / `SDN013` Southern Kordofan | `SSD010` Warrap | Abyei | 48 / 34 km |

**`CHN`/`PAK` is ADM0-only — no ADM1 row.** Gilgit-Baltistan and Ladakh/Jammu &
Kashmir are disputed territories **excluded from WB's ADM1 layer** — not
provinces (WB lists only five Pakistani ADM1 units, none Gilgit-Baltistan; the
tracts overlap no province, i.e. true gaps). With no de-facto province to
attribute to, crediting the border to the nearest *existing* province (Khyber
Pakhtunkhwa / Himachal Pradesh) would be geographically false (e.g. "Assam ↔
Tibet"), so the relationship is carried at the country level only. The asymmetry
is honest: the Golan (Israel's Northern District) and Western Sahara (Morocco's
Laâyoune/Guelmim) *do* have administering provinces in WB; Gilgit-Baltistan and
Ladakh do not. The Shaksgam Valley (Xinjiang) has one, but no other province
touches it, so it adds no ADM1 pair.

**Tracts that change no pair** — the other India–China and India–Pakistan tracts
(among them the Karakoram Range, which lies on the restored China–Pakistan border and
adds to its length), Shebaa Farms (it touches Lebanon and Israel, not Syria; the Golan
restores Israel–Syria), the Ilemi Triangle (Kenya & South Sudan already share an
~80 km border) and the three islands — are documented in the full per-tract audit
shipped at `docs/ndlsa_tract_audit.csv`, with Natural Earth's record and share for
each tract.

---

## 6. Scope / honesty notes

- The prior-practice evidence (§2.2) is predominantly **tool and standards
  documentation**, not peer-reviewed GIS-science; it establishes *standard
  practice*, not a citable optimum.
- **Inland-water (lake) handling** has **no standard-practice citation** found
  in the survey. The build uses **no lake filter** — because WB admin polygons include
  lake water, mid-lake contacts are native edges — and `ne_10m_lakes` feeds
  only the water screens, which nominate lake borders for adjudication (§8),
  never adjacency.
  Treating lake water as native adjacency is a pragmatic choice, documented in
  PROVENANCE.
- No standard **minimum shared-border-length** threshold exists in the
  literature; the rook rule is qualitative ("non-zero length"). The
  `narrow_border` (<5 km) and `potential_artifact` (<1 km) flags are advisory
  labels, not adjacency criteria, and have no canonical cutoff.

## 7. Reproducibility

One command reproduces the shipped database:

```
# full regeneration from the pinned sources (SHA-256-verified before running):
python scripts/build_all.py --full \
    --adm1-gpkg <WB Admin 1 .gpkg> --adm0-gpkg <WB Admin 0 .gpkg> \
    --ocean-gpkg <WB Ocean Mask .gpkg> --ndlsa-gpkg <WB NDLSA .gpkg>

# re-derive + verify from the shipped data (fast; no GeoPackages needed):
python scripts/build_all.py
```

The full mode verifies the inputs' SHA-256 against the pins (in
`scripts/build_all.py` and `data/PROVENANCE.md`; a mismatch is a hard error),
runs the geometry build (Stages 1–2), checks that every denylisted contact is in its
output, applies Stages 3–4 with `scripts/apply_overlays.py`, checks every headline count (edges, regions,
countries, water set, roll-ups, all three standard views at both levels), and
reports byte-identity against the committed CSVs. The refresh mode is a
byte-stable no-op on an untouched checkout — the day-to-day reproducibility
check, also enforced by `tests/test_apply_overlays.py`. The sensitivity sweep
is re-run by `build_data/snap_sweep/snap_sweep_tol.py` (the build's own edge
test at each tolerance, with the candidate search widened by the tolerance;
§2.4); `SNAP_TOL_DEG` plays no part in contiguity, so varying it changes no
edge.

## 8. Coupling standard (water-separated pairs)

A subset of cross-border pairs share **only** a river or lake border — no land
segment. Whether such a pair counts as "pericoupled" depends on the question
being asked, so the loaders expose a `coupling_standard` (default `"moderate"`),
orthogonal to `de_facto_borders`:

| standard | a water-only pair is pericoupled iff… |
|---|---|
| `lenient` | always (any shared water counts — the prior behaviour) |
| `moderate` *(default)* | a **fixed crossing open to traffic** links the two units |
| `stringent` | never (water never counts) |

**Data.** Since 2026-07-25 each row also carries two **structured provenance**
columns appended after `note` — `adjudication` (the process class; uniformly
`cross-vendor` across all 803 rows after ru1/wu1/wu2/rj2/nt2/dc1/pr1) and `verification_tier`
(the evidence strength: **A 268 / B 238 / C 297**, tier B pinned to the
preregistered validation study's measured frame). ADM0 roll-up rows carry both
blank, being derived arithmetic rather than adjudicated verdicts. Full
semantics: `data/PROVENANCE.md`.

`data/water_separated_pairs.csv` lists the **803 ADM1** water-only
pairs with a `has_bridge` flag (779 on a shared edge + 24 between non-touching units,
all from `data/water_classification_pairs.csv`, Stage 3), plus **26 ADM0** country
pairs rolled up from them (a
country pair is water-only iff *all* its ADM1 crossings are, and has a bridge
iff *any* does). 298 rows come from the first classification round (238 river / 60 lake), every row
carrying a cross-vendor re-adjudication — river rows from ru1 (2026-07-21),
lake-class rows from wu1 (2026-07-25) — correction history in `CHANGELOG.md`.
The other 505 — 481 existing edges classified water-only and 24 non-touching
water borders added as edges — come from the 2026-07 run of the water screens over
every edge (audit batches b1–b6 and the 20 km-hold tranche) and the campaigns after
it (composition: `data/PROVENANCE.md`), every row nominated by the
full-ladder screens (river bar 0.50 at geodesic 2.5/5/10/15/20 km rungs; lake
bar 0.40 at geodesic 125/250/500/1,000/1,500 m; HydroRIVERS bar 0.50 at a fixed
geodesic 500 m sample-to-reach width (one 15 arc-second HydroSHEDS cell), nomination floor
10 m³/s plus the creek band — ≥ 0.5 of the border within 500 m of any reach — whose 715 domestic creek-only edges (714 set aside on 2026-07-10 as document-only, and one edge no screen population held) were adjudicated on 2026-09-14 (54 accepted, 50 of them shipped); HydroLAKES at geodesic 500 m with the same 0.40 lake bar (0.5 until 2026-09-19; the 21 borders the lower bar newly nominated were adjudicated, none water-only); since 2026-09-22 every edge screen runs on the border arc, with a combined river screen (bar 0.50), a combined lake screen (0.40) and a cross-type union (0.80) added (`ba1` below; the union since 2026-09-26 at each layer's widest width, `mr1` below), and since 2026-09-23 every distance is located and measured on the ground (`gd1` below); the corridor
census for non-touching pairs (re-derived 2026-09-10 and measured on the ground since 2026-09-23, below); domestic borders in scope for the first time),
every candidate Tier-2 adjudicated and
every shipped verdict human- or dual-AI-verified with per-row provenance in
the water file's `source` column —
all of it applied in one pass by the idempotent engine
`scripts/apply_overlays.py` (Stage 3; full provenance in
`data/PROVENANCE.md`). Fifty-one of these 505 rows were first found by
earlier campaigns now retired as nomination steps (2026-09-09) and folded
in with their discovery provenance retained in `source`: the 18 hydro-water and
12 hydro-lakes rows (the 2026-07-02/04 HydroRIVERS/HydroLAKES geodesic 500 m
cross-checks — every one of the then-1,800 cross-border edges sampled at three
discharge tiers, the 34 un-adjudicated flags two-stage ground-truthed and every
confirmation human map-verified; incl. the Oder/Neisse, Rio Hondo, Cavally,
Mano, Kagera, Alazani, the Uruguay River and the Dead Sea; 2026-07-28 fold),
the 15 river rows of the 5 km/10 km widening re-screens (ru1 fold,
2026-07-21), the 5 river rows of the ~1 km near-miss net (rg1 fold,
2026-07-22) and the Skadar lake row of the 1–100 km lake band (lg1 fold,
2026-07-23) — all independently re-nominated by the current screens
(whole-graph attributability 803/803). Under
the default, ADM1 pericoupled edges fall 8,461 → **8,065** and ADM0 country
pairs 326 → **320** (the hydro-water rows' GUF↔SUR roll-up — the Maroni
system, ferry only — joins COD↔TZA, MRT↔SEN, CAF↔COD, NGA↔TCD across
Lake Chad, and the roll-up GUY↔SUR — the ferry-only Corentyne — as
default-view subtractions; the roll-ups DEU↔LUX, BEN↔NER,
CMR↔GAB, MWI↔TZA, and MOZ↔TZA (completed by the Lake Nyasa corner
MOZXXX↔TZA025) are bridged, so they move only stringent; the stringent view ships at
**300**).

**Adjudication design.** Every water-only verdict on record — the 803 shipped
rows and every rejected candidate in the audit record — was set by one design
run *downstream* of the deterministic screens: a **cross-vendor two-pass** in
which the research pass ran on OpenAI's GPT-5.6 Sol, as a web-enabled agent in
the Codex app, and the adversarial judgment pass on Anthropic's Claude Sonnet 5,
blind to the repository's prior verdicts, so no single model family sets a
verdict alone. Both passes are labelled by their model throughout — `GPT-5.6 Sol
research`, `Sonnet-5 adversarial judgment` — in the per-row `source` strings and in
the campaign paragraphs below. Medium-
confidence verdicts were human map-verified, high-confidence ones dual-AI
cross-checked, and every new *edge* additionally passed the four-layer bridge
pipeline (OSM Overpass screen, agent web verification, adversarial recheck of
disagreements, geocode + province-polygon cross-check) before shipping.
Candidate coverage is fixed by the buffer-independent full-database
HydroRIVERS/HydroLAKES sweeps and the whole-graph screens — complete
relative to those datasets' documented floors, a coverage claim rather than a
recall claim against the unknown true adjacency set — and correctness by the
human/dual-AI verification; the LLM passes only nominate and cross-check
within that frame. Two single-family designs ran before the 2026-07 run of the
water screens — a
two-stage ground truth for the 2026-07-02 hydro sweeps and a two-pass
adversarial adjudication of the 57 lake water-band candidates (2026-07-04) —
and survive only as discovery history in each row's `source` (evidence
archived under `build_data/_archive_pre_rebuild/`): every verdict they produced
was re-adjudicated under the cross-vendor design (the shipped rows on
2026-07-21/25; of the 201 rejected candidates, 199 on 2026-09-01 and the two that
were first closed mechanically, because no screen then nominated them, on
2026-07-22 (Cahul↔Vaslui, river-gap retirement) and 2026-09-20 (Vorarlberg↔Vaduz,
`nt4`)) and upheld.
The earlier *discovery nets* are likewise retired as nomination steps
(2026-09-09): the ~1 km near-miss net and the 1–100 km lake band, the 5 km/10 km
widening re-screens, the 2026-07-02/04 HydroRIVERS/HydroLAKES cross-checks and,
for the first-round classification, the original ~60%-within-2.5 km river screen.
Re-running the whole-graph attributability check on the shipped data
(`build_data/water_screen_rebuild/hydro_fold/attributability_check.py`) shows
every shipped water row nominated by the current screens alone — all 779 rows on
a shared border by the edge screens on their border arc (the 298 first-round rows
and the 481 later ones alike, none by a combined screen or the union alone) and
the 24 non-touching rows by the corridor census — so no shipped row depends on a retired net. The corridor census was itself
re-derived on 2026-09-10 (v2: HydroLAKES joins NE lakes; the HydroRIVERS buffer is
geodesic and discharge-laddered, where the 2026-07 run buffered in planar degrees;
transect corridors along the facing frontage replace the single nearest-approach
segment; third-unit and wedge flags are recorded). It re-nominates all 22 earlier
non-touching rows and produced 80 new nominations (8 by the 0.80 share rule, 30 only
under the wide ladder, 42 by the short-gap presence rule), all adjudicated by the
cross-vendor two-pass and maintainer map rulings in two campaigns: nt2 (46 pairs — 7
share-rule nominations not already ruled, 30 ladder, 9 presence-rule with ≥ 0.25
transect support): two shipped (Équateur↔Cuvette, Entre Ríos↔Artigas), 44 rejected;
nt3 (2026-09-11; the 33 presence-rule nominations a post-hoc 0.25 transect-support
floor had set aside): all 33 rejected on convergent verdicts (15 sea-separated, 18 dry
gaps), the floor withdrawn — the presence rule nominates, it does not drop (record:
`build_data/water_screen_rebuild/corridor_census_v2/`). Its population, since 2026-09-23
(`gd1` below), is every pair within 100 km of each other on the ground that do not touch and
are not already edges, 23,407 pairs (a point contact has no corridor:
Jõgeva↔Pskov and Salta↔Potosí are outside it, as are the 16 de-facto edges; a pair the build's
polygons join at a single point while the World Bank polygons keep it apart is measured on the
World Bank polygons); the 100 km reach is checked against the
lakes — of the 357 Natural Earth lakes ≥ 500 km², five have surfaces the WB layer
leaves unassigned, none wider than 60 km
(`corridor_census_v2/lake_surface_coverage_2026-09-10.txt`). The rule as implemented
(`build_data/geodesic_distances/census_gd1.py`, on the ground: frontage, transects and samples
in an azimuthal-equidistant frame centred on each pair's nearest approach): transects at most
250 m apart along the facing frontage, samples at most 100 m apart; lake share = share of
samples within 125 m of a lake polygon (the larger of
NE and HydroLAKES ≥ 0.25 km²); river share = share within 500 m of a reach (gaps
≤ 5 km only); nominated when lake share + river share ≥ 0.80 (a sum capped at 1 — a
generous stand-in for the union), or, in the wide variant, with lakes at 1,500 m and
reaches ≥ 1,000 m³/s at 2,500 m (a 1,000 m rung is recorded alongside), or, for gaps
≤ 1,000 m, when any sample lies within 500 m of a reach. Of its thresholds, the
500 m river width (one HydroSHEDS cell), the 250 m transect spacing and the 5 km river
cap (recovered river gaps 0.2–1.7 km) are anchored to measured quantities; the 0.80
bar and the 125 m lake rung are inherited conventions and the 1,000 m³/s big-reach
threshold a judgment value introduced with the re-derivation, all reported for
sensitivity: every bar from 0.60 to 0.85 re-nominates all 24 shipped recoveries; on the
2026-09-10 census, 0.75 → 0.85 left the new share-rule nominations almost unchanged (10 / 8 / 8)
and moved the share-or-ladder nominations 50 → 32. Recomputed on the full ladder
(`corridor_census_v2/ladder_profile_2026-09-11.csv`), the 42 ladder-only nominations first
cross the bar at 250 m (1), 500 m (3), 1,000 m (16) and 1,500 m (14) on the lake axis and at
1,000 m (4) and 2,500 m (3) on the big-reach axis, one only with both wide rungs; the 34
lake-axis pairs are all adjudicated rejections and the one genuine ladder-only border, the
Congo pair, crosses at the 1,000 m big-reach rung. Each row keeps its discovery
provenance in `source`, and the nets' audit records (the 48-pair near-miss
funnel, the 10 km double-verification, the Natural Earth omissions that
motivated the hydro rungs) remain validation evidence.
Ten verdicts the maintainer could not at first resolve from available map
evidence — the pre-2021 Latvia/Lithuania subdivision-vintage pairs and the two
`Area under National Administration` placeholder units — were **ruled by the
maintainer on 2026-09-19 on the shipped WB geometry itself** (a deterministic
overlay of OSM waterways and HydroRIVERS along each arc, read at 100/250/500 m,
and a rendered map per pair): nine accepted, one rejected. The rulings replace a
delegated final-arbitration pass of 2026-07-17 and reach the same verdicts; the
two placeholder polygons are the Malawian and Mozambican Lake Malawi/Niassa
water-surface units (placeholder-identity audit, 2026-07-18). Every
accepted pair passed the same four-layer bridge pipeline (the
province-polygon layer corrected one off-border citation, the Kerio
crossing, to the on-reach Rorok structure).

**Acceptance standard (Standard M, maintainer ruling 2026-09-01).** A pair is
accepted as water-only only when its dry component is digitization noise, not a
**genuine land corridor**: a named surveyed straight line, ridge/watershed
stretch, or overland connector defeats `water_only` regardless of its share of
the arc, and the "~20% dry" figure in the adjudication prompts is an AI-pass
screening tolerance, never an acceptance license. Every maintainer ruling had
already applied this standard (the register precedent `BRA025`↔`URY014` was
ruled not water-only at 10.2% dry, over a judge verdict that applied the
numeric bar), and no shipped water-only row carries an accepted verdict with a
quantified 4–25% dry share (checked live 2026-09-01). The ruling closed the
`rj1` campaign, which re-adjudicated the 48 rejected candidates in the
cross-border hydrography screen record that rested solely on the earlier
designs — **all 48 upheld, zero data change** — and the `rj2` campaign the same
day re-adjudicated the remaining **153** design-era rejections in the
contact-screen and non-touching records (**all 153 upheld**), so no verdict on
record, shipped or rejected, rests on the earlier designs alone. rj2's
completeness scan also surfaced **37 domestic large-river nominations**
(HydroRIVERS ≥ 0.5 in the ≥ 100 m³/s band) for which the 2026-07-10 queue
registry had no domestic source — its medium-band source stopped at 100 m³/s,
and large rivers were assumed covered by the Natural Earth rungs, which miss
exactly these; adjudicated the same way, **15 shipped as water-only**
(11 bridged / 4 not; 14 ship since 2026-09-22, 11 / 3, after Montevideo↔San
José left the water-only set on the border arc, `ba1` below) — the 736 → 751 change
(`build_data/water_screen_rebuild/rejection_unification/rj1_rulings.json`,
`rj2_rulings.json`).

The registry's HydroRIVERS **creek band** — ≥ 0.5 of the border within geodesic 500 m of *any*
reach but < 0.5 at ≥ 10 m³/s — had been audited cross-border and wherever another rung
co-nominated the edge, but the 2026-07-10 gate set its 714 domestic creek-only edges aside as
document-only. That gate was withdrawn on 2026-09-12 (thresholds nominate, audits decide) and
the band went through the same design as every other nomination band (`dc1`), together with
the one edge restored on 2026-07-19 after the screen populations froze and therefore never
screened (Balzers↔Planken, which screens into the band): 715 edges, GPT-5.6 Sol research (65
water-only) → Sonnet-5 adversarial judgment (55; no repository access, one structured return
per agent), gate A 30 / B 60 / C 175 / D 450. Because the judges, lacking web access, mostly
derived their dry component from the complement of the HydroRIVERS share, a deterministic gap
measurement (`domestic_creek/measure_dc1.csv`: kilometres of border more than 500 m from any
reach, the longest contiguous such run at 500 and 1,000 m, its position on the arc)
accompanied the 90 pairs with at least one water-only verdict to the maintainer, whose map rulings of 2026-09-14 accepted
**54** (20 dual-true, 21 research-only, 13 judge-only) and upheld 36 as land; the 54 passed the
four-layer bridge pipeline (OSM screen with the in-both-units test, 19 settled; GPT-5.6 Sol adversarial
recheck of the 35 disagreements and unknowns; coordinate cross-check against the shared WB arc;
47 bridged / 7 not) — the 753 → 807 change (`build_data/water_screen_rebuild/domestic_creek/`,
`dc1_rulings.json`). The forward completeness check (`rejection_unification/completeness_check.py`)
now reports zero open nominations in every band of both screen populations.

**Crossing unification (`bu1`, 2026-09-16).** Water-only *status* had been
unified across all rows (ru1/wu1/wu2), but the crossing flags had come from three
generations: the June 2026 base pipeline (OSM classifier with a ~130 m unit
buffer and a 400 m sweep; 57 huge-lake rows never queried), the 2026-07 and later
four-layer runs (a faithful copy of that classifier in July; a 50 m/300 m fork
with a tunnel/dam query in the September campaigns; location tolerances of ~3 km,
1.7 km and 500 m), and the folded earlier rows with no screen or location
record at all (45 on existing edges, 6 non-touching, 3 shore contacts). The unified
screen was re-run over all 807 rows: it agreed with the shipped flag on 691, found
no bridge way for 54 bridged rows and a bridge way for 56 unbridged rows, and only
a tunnel or dam-top way for 6 bridged rows. Those 116, plus the 15 bridged rows
without a location record, went to the GPT-5.6 Sol adversarial recheck (88 yes / 43 no),
every *yes* to the location test (26 of the 88 fell outside 500 m of the arc; 18
of those are mid-span pins within 750 m of both units, 6 bank-line gaps within
3 km, 2 wrong-unit citations — Yacyretá lands in a different Paraguayan
department, the Queen Louise Bridge in a different Lithuanian unit), giving 20
proposed upgrades and 7 proposed downgrades. The maintainer's map rulings of
2026-09-16 accepted 10 upgrades (among them the Pont de Vonkoro, the Niangoloko
and Aghband–Kalaleh bridges completed since the June run, the Cernavodă complex,
Cầu Sài Gòn 1) and 4 downgrades (the Bojana, Oti, Paranaíba and Alima crossings),
and kept 13 shipped flags (a pontoon bridge on the Prut and the Fifth Friendship
Bridge on the pre-2011 Nong Khai polygon among them): **409 bridged / 398 not**
(was 403/404), moderate 8,054 → **8,060**, ADM0 roll-ups unchanged. Every row's
provenance now ends with a `bu1` clause stating the unified screen's verdict and,
where run, the recheck, the location distance and the ruling; the base CSV is
re-pinned. Record: `build_data/water_screen_rebuild/bridge_unification/`
(`bu1_screen.csv`, `bu1_gate.json`, `bu1_maintainer_rulings_2026-09-16.txt`,
`bu1_rulings.json`).

**Pilot re-adjudication (`pr1`, 2026-09-18).** The nine-pair pilot of the 2026-07
audit queue had been adjudicated by the original audit design — a Sonnet-5
research pass feeding a Fable-5 judgment pass, both on one vendor — before the
research pass moved to GPT-5.6 Sol, and was then frozen. All nine were rejected and
none shipped, but they were the one set of verdicts on record that the
cross-vendor design had never covered: the 2026-09-01 statement above that no
verdict rests on an earlier design overlooked them, because
`completeness_check.py` counted membership of `audit_queue.csv`, which lists the
pilot, as a cross-vendor record (fixed 2026-09-18). All nine were still nominated
by a current screen, so all nine went through the standard process, blind to the
pilot verdicts: GPT-5.6 Sol research (9/9 valid; 8 not water-only, 1 water-only),
Sonnet-5 judgment (9 agents, one structured return each), gate A 0 / B 2 / C 4 /
D 3, a deterministic gap measurement of the 8 contact pairs, and maintainer map
rulings on the two disagreements. **Kapisa↔Parwan was accepted** as water-only
along the Panjshir River (research water-only; judge land, on an inferred ridge
segment the measurement did not find — longest uncovered run 1.0 km of 97 km, none
at 1,000 m); Kuçovë↔Lushnjë was upheld as land (the line runs along the Thana
reservoir's dam embankment). Eight rejections stand, Kukës↔Tropojë again found not
adjacent across the Fierza Reservoir. The accepted pair passed the four-layer
bridge pipeline without a recheck — the unified OSM screen found 7 open bridge
ways in both units, research and judge both reported a crossing, and the
trunk-road bridge lies on the shared arc — and ships as a tier-A water-only
row: the 807 → 808 change (410 with a fixed crossing / 398 without; moderate
unchanged at 8,060, stringent 7,650)
(`build_data/water_screen_rebuild/pilot_readjudication/`).

**Arbitration review (`ar1`, 2026-09-19).** Ten verdicts of the 2026-07 audit
(the pre-2021 Latvia/Lithuania subdivision-vintage pairs and the two lake-surface
placeholder contacts) had been delegated by the maintainer, for want of map
evidence, to a final-arbitration pass on a third model (Fable 5, 2026-07-17),
which shipped nine and rejected one. They were the only verdicts in the database
resting on that model. The maintainer ruled on all ten on 2026-09-19, on the
shipped WB geometry itself — the Latvian units are municipalities abolished in
2021, so current maps do not show these lines — using a worksheet built for the
purpose: the existing GPT-5.6 Sol research and Sonnet-5 judgment verdicts (both
water-only on all ten), a deterministic overlay of OSM waterways along each arc
(50 m sampling; share within 100, 250 and 500 m of a river or stream, longest dry
run and its position), the HydroRIVERS/HydroLAKES instrument of the creek-band
campaign, and a rendered map per pair. **All ten rulings match the arbitration**
(Telšiai↔Saldus not water-only; nine water-only), so no flag and no count
changes; the nine rows' `source` now cites the maintainer ruling, their tier A is
literally human-verified, and no shipped provenance string names the third model.
The adjudication design is therefore two models, one per vendor, plus maintainer
map rulings (`build_data/water_screen_rebuild/arbitration_review/`).

**HydroLAKES band (`hl1`, 2026-09-19).** The two lake rungs had carried different
bars for no recorded reason. The Natural Earth lake bar, 0.40 at geodesic 125 m,
has a derivation (`build_data/lake_screen_analysis/lake_ladders.py`, 2026-07-09):
lake coverage is sharply bimodal — of the 1,772 cross-border edges measured,
1,659 score zero, 40 score at least 0.8 and only 16 fall between 0.2 and 0.6
(on the border arc of 2026-09-22: 1,663 of 1,774 zero, 38 at 0.8 or more, 13 between) —
so the bar sits in the empty valley, at the first round value below the weakest
genuine lake border Natural Earth captures (0.48), and the candidate set barely
moves with it (53–55 candidates across widths of 110–500 m). The HydroLAKES bar,
0.5 at geodesic 500 m, had only been lowered from the 0.6 of the 2026-07-04
cross-check to match the river bars, and had no derivation of its own.
On 2026-09-19 it was lowered to the same 0.40 and the newly nominated band
adjudicated, thresholds nominating and audits deciding. The band [0.40, 0.50)
then held 55 edges: 14 shipped water-only, every one also nominated by a river rung
(rivers with a reservoir or lake stretch); 20 had been rejected by earlier
audits; **21 were new** (20 domestic, 1 cross-border), every one below 0.5 on
HydroRIVERS and without a Natural Earth rung. GPT-5.6 Sol research, blind: 20 not
water-only, 1 water-only; Sonnet-5 adversarial judgment: 21 not water-only. The
one disagreement (St. Paul↔St. Peter, Antigua: a 5 km line whose eastern ~2 km
crosses the Potworks Dam reservoir and whose western ~3 km runs straight over
land) went to a maintainer map ruling: not water-only. **All 21 rejected; no flag
and no count changes.** At 0.40 the HydroLAKES rung nominates 291 edges of the
edge list, 164 of them shipped water-only, and each of the other 127 carries a
cross-vendor rejection (`build_data/water_screen_rebuild/hydrolakes_band/`); the
sweep's records hold two more pairs that are not edges and are not counted
(RUS024↔RUS050, dissolved by the unit merge; Manitoba↔Northwest Territories, a
corner contact). On the border arc, with distances on the ground (`gd1` below), the rung
nominates 290 edges, 162 of them shipped and each of the other 128 carrying a cross-vendor
rejection; the band holds 59, 14 of them shipped, every one also nominated by a river screen.

**Census residue (`nt4`, 2026-09-20).** A code-level audit of the papers found that
one of the 245 nominations of the 2026-09-10 corridor census, Vorarlberg↔Vaduz
(`AUT008<->LIE011`; gap 492 m, short-corridor hit), carried no two-model record. It
had a near-miss-era verdict and the *mechanical* closure of 2026-09-01 ("no current
screen nominates this pair"), a premise that stopped holding when the re-derived
census nominated the pair nine days later; it was never queued for the census
campaigns because it sat in the predecessor's file and was counted as already
adjudicated, and the completeness check had hidden it by counting the mechanical
closure as a record (it no longer does). The pair went through the standard two-pass,
blind: GPT-5.6 Sol research and Sonnet-5 adversarial judgment both returned not
water-only at high confidence — the units are not adjacent (other Liechtenstein
municipalities hold the Austrian line) and the gap is dry alpine ridge — so no
maintainer ruling was needed. **Rejection upheld; no flag and no count changes**;
every nomination of every screen now carries a two-model record, and the completeness
check shows 0 open (`build_data/water_screen_rebuild/corridor_census_v2/`, `nt4_*`).

**Way-class follow-up (`wc1`, 2026-09-20).** Layer 1 of the crossing pipeline accepts
any OpenStreetMap way tagged as a bridge on a road, path or railway; the rule that
only a road or rail crossing counts is applied by the later layers, and a row on which
the screen and the web verification agreed never reached the adversarial recheck, the
one layer that states the footbridge exclusion. The screen had kept way names, not
classes, so a diagnostic re-query recorded the class of every qualifying way for the
354 bridged rows with a layer-1 "bridge way found" (OpenStreetMap as of 2026-09-20;
4,923 ways, 899 of them foot or cycle ways): **340 rows have a road or rail way in
both units**; 6 have foot or cycle ways only, 5 a service road, farm track or
access-restricted way only, 3 a disused railway or a bridge outline only. Of those 14,
11 rest on a road or rail structure named by the web verification or an earlier
recheck, or on a maintainer ruling (the screen misses such a structure when the
generalized polygons leave it outside the 130 m test, or when it is a dam-top road or
a tunnel rather than a bridge way). The 3 with no such structure on record went to the
GPT-5.6 Sol adversarial recheck: Sankt Gallen↔Gamprin was confirmed (the Haag–Bendern
road bridge, 370 m from the farther unit), and two downgrades were proposed. The
maintainer's rulings of 2026-09-20 accepted one — **Basel-Landschaft↔Baden-Württemberg,
whose only crossing is the foot and cycle way over the Augst–Wyhlen barrage** — and
kept Telšiai↔Vaiņodes novads, whose only structure is a disused railway bridge with
its track in place (convention below): **409 bridged / 399 not** (was 410/398),
moderate 8,060 → **8,059**, ADM0 roll-ups unchanged
(`build_data/water_screen_rebuild/bridge_unification/`, `bridge_way_class_*`, `wc1_*`).

**The HydroRIVERS screens take the other screens' sampling rule (`gs1`,
2026-09-20/21).** The HydroRIVERS-family screens (the domestic and cross-border sweeps,
the latter also computing the cross-border HydroLAKES share, and the creek-band
enrichment) set the number of samples on each part of the arc from its coordinate
length, one per 4.5×10⁻³° — about 500 m along a north–south line but only about
500·cos(latitude) m along an east–west one — so how many points a border received, and
with it the resolution of its coverage share, depended on its latitude and orientation;
the Natural Earth screens and the domestic HydroLAKES sweep already set it from the
geodesic length. On the maintainer's decision all three now use that rule: n = max(2,
⌊L/500 m⌋ + 1) intervals for a part of geodesic length L. Both rules space the points
evenly along a part in coordinates, so the average spacing is below 500 m while single
intervals vary with direction and latitude (median 493 m, 5th–95th percentile 403–545 m
on the shipped borders, `sample_spacing_shipped.txt`); the switch changes how many
points fall on a border and where, not how a share is weighted along a part.
Re-run on the pinned inputs, the change reshuffles bar-edge cases in both directions
(median coverage change about zero) with two consequences. **Nineteen creek-band pairs
were newly nominated** with no adjudication record (17 domestic, 2 cross-border); they
went through the standard two-pass — blind GPT-5.6 Sol research (19/19 valid; one
water-only) and Sonnet-5 adversarial judgment (none water-only) — gate A 0 / B 1 / C 6 /
D 12, and the maintainer ruled the one disagreement (Vitebsk↔Zilupes novads) not
water-only after a gap measurement: all 19 rejected. **Three shipped rows were no
longer nominated by any screen** — Lekoumou↔Niari (0.50 → 0.49), Kyegegwa↔Ssembabule
(0.50 → 0.33 on a 0.65 km border sampled at three points) and Hai Duong↔Quang Ninh
(0.51 → 0.49), all creek-band rows ruled water-only on maps on 2026-09-14. On the
maintainer's decision they **leave the water-only set** (the edges stay, as ordinary land
borders): the database is complete relative to its screens, and a row no recorded screen
nominates falls outside the method however it was ruled — which is not a finding that
they are land; they were listed as known omissions in `docs/FUTURE_WORK.md` (the border-arc rule of 2026-09-22 nominates
two of them again; both were re-adjudicated, `ba1` below). A 100 m step on
all four screens was measured before the decision and not adopted: it would nominate
Lekoumou↔Niari again but leave four other shipped tier-A rows with no nominating screen and
open 19 new candidates without a record among the edges near a bar
(`step_sensitivity_all_screens.py`). Water-only
808 → **805** (409/399 → **408/397**), moderate 8,059 → **8,061**, stringent
7,650 → **7,653**, tier A 273 → **270**, ADM0 unchanged (all three domestic); completeness
0 open, attributability 805/805. Record: `build_data/geodesic_sampling/` (`gs1_*`; the
degree-step predecessors frozen as `*_degree-step_pre-gs1.csv`).

**One border arc for every edge screen (`ba1`, 2026-09-21/22).** The edge screens had each
defined the shared border their own way. The Natural Earth ladders sampled the exact shared line
as the geometry engine returns it, unmerged (a median of 70 pieces per edge, each with its own
three or more points), so a border's share was weighted by how finely the line happened to be cut;
the HydroRIVERS and HydroLAKES screens sampled unit A's outline within 10⁻³° (about 111 m) of unit B
(widened to 5×10⁻³° and 2×10⁻²° where that found nothing), a band that added 4,932 km beyond the
exact line: 57.7% of it stretches of a third unit's border near the tripoints and 12.5% coastline,
but also 11.8% river channel that the Ocean Mask cuts out as sea between two facing banks (273
edges); all of them read the raw World Bank polygons, whereas the build
relabels the sliver corridors and merges RUS050 into RUS024, so the two borders that exist only
after the relabel (Kajiado↔Kilimanjaro, Narok↔Mara) had never been screened; and the records
rounded shares to two decimals, which made the bars 0.495 and 0.395 in practice. A specification
written and frozen before any measurement
(`build_data/arc_and_combined_screens/SPEC_ba1_border_arc_and_screens.md`, maintainer approval
2026-09-21) replaces them with one rule, implemented once
(`build_data/water_screen_rebuild/border_arc.py`). The screens read the build's own polygons
(`load_adm1_build_geometry` in `scripts/build_pericoupling_db.py`, the loader the geometry build
itself uses). The **border arc** of an edge is the set of points of A's outline where A bounds the
contact with B: the exact shared line, its pieces merged; A's outline inside B where the polygons
overlap (the four Saint-Louis borders); A's outline within 5×10⁻⁴° of B for the four land-gap pairs
only; and **facing stretches**, where A's outline faces B across a gap no wider than 1,000 m (the
corridor census's short-gap presence rule), lies on no third unit's outline and its chord crosses
none, is reciprocal (the point of A's outline nearest to B's nearest point lies within one
sampling interval) and the chord's midpoint lies inside neither unit (in the Ocean Mask or a strip
that belongs to no unit). Each continuous piece is sampled at n + 1 points at equal geodesic
intervals, n = max(2, ⌊L/500 m⌋ + 1); shares are stored exact. The four screens keep their widths
and bars, and three screens join them: a **combined river** screen (a point counts as water within
2.5 km of a Natural Earth river or 500 m of a HydroRIVERS reach; bar 0.50), a **combined lake**
screen (within 125 m of a Natural Earth lake or 500 m of a HydroLAKES polygon; bar 0.40) and a
**cross-type union** of all four layers at their operating widths with the corridor census's bar,
0.80. Over the 8,421 edges with a border arc, the arc holds 1,227,377 km of exact line (8,417
edges), 5,903 km of facing stretches (954 edges; median width 286 m), 444 km of overlap (the four
Saint-Louis borders) and 50 km of land-gap band (4 edges). Measured blind on the frozen rule, 3,248 edges are nominated
(the earlier records nominated 3,257); 75 nominations are new, 14 of them with a two-model record
and 61 without (15 only through a combined screen, none only through the union); three shipped
water-only rows are nominated by no screen; and two rows that left the set on 2026-09-21 are
nominated again. Where the rule coincides with the earlier one it reproduces the HydroRIVERS share
and point count on 7,430 of the 7,464 exact-only edges (the others lie where the build's polygons
differ from the raw file); the shares computed on B's outline differ from A's by a median of 0.002;
and a reach of 500 or 2,000 m or a reciprocity tolerance of 250 or 1,000 m moves between 5 and 12
nominations (gained plus lost), a union bar of 0.70 or 0.90 none, and no shipped row loses its
screen under any of them. The maintainer's map check of a
96-row sample (facing, overlap, land-gap and excluded stretches, and the rows the rule drops or
nominates again) found nothing to correct. The 63
pairs (the 61 without a record, and the two re-nominated rows, re-adjudicated rather than restored
on their earlier rulings) went through the standard two-pass, blind: GPT-5.6 Sol research (63/63
valid; water-only on 2) and Sonnet-5 adversarial judgment (63 agents; water-only on none), gate
A 0 / B 2 / C 16 / D 45. The two disagreements were measured (`measure_ba1.py`, `map_ba1.py`) and
their cited sources read. **Lekoumou↔Niari returns as water-only** by maintainer map ruling (the
research's legal source is the 2002 forestry decree, whose zone limit follows the Mpoukou from its
source on the Gabon border to the Louessé, and the Louessé to the Niari; the World Bank line lies within 500 m
of a HydroRIVERS reach along 50% of its length and within 1 km along 73%), without a fixed crossing
(the road and rail bridges at Makabana that the research named, the P1 and Comilog bridges over the
Louessé in OpenStreetMap, lie 2.3 km inside Niari on the World Bank map; maintainer ruling). **Garissa↔Wajir was ruled mixed** (not
water-only). Ten further nominations, closed by the auto-reject rule on the earlier shares but no longer
matching it on the border arc (one layer at 0.20–0.28 each; the completeness check now re-tests the rule instead of
counting the ledger, whose rows the audit queue also lists), went through the same two-pass as a second tranche:
all ten rejected, the maintainer ruling the one disagreement (Neretas↔Vecumnieku, whose line meets the Mēmele only at its southwest end) mixed. The three rows no screen nominates (Chiba↔Tokyo on the Edo River, Montevideo↔San José
on the Santa Lucía, Haiphong↔Quảng Ninh on the Đá Bạch–Bạch Đằng) leave the water-only set under the
removal rule the maintainer set with the specification (the edges stay; `docs/FUTURE_WORK.md` §4
lists them with their shares); Hai Duong↔Quang Ninh, re-adjudicated, stays out (both passes not
water-only). Water-only 805 → **803** (408/397 → **406/397**), moderate unchanged at **8,061**,
stringent 7,653 → **7,655**, tier A 270 → **268**, ADM0 unchanged (all four rows domestic).
`ba1_screens.csv` became the edge screens' record (until `gd1`, below): the completeness check read its bands (0 open)
and the attributability check required every shipped row with a border arc to be nominated by it
(779/779; the 24 non-touching rows by the corridor census; 803/803). Deferred
(`docs/FUTURE_WORK.md`): `border_length_km` is still the exact shared line; a border whose water is
split between a river and a lake with a union share between 0.50 and 0.80 is not nominated. Record:
`build_data/arc_and_combined_screens/` (`SPEC_ba1_*`, `ba1_*`) and `build_data/water_screen_rebuild/`
(`border_arc.py`, `build_arc_cache.py`, `run_screens_ba1.py`, `ba1_screens.csv`, `ba1_report.txt`).

**Distances on the ground (`gd1`, 2026-09-23).** The screens measured every distance on the
ellipsoid but located the nearest point of a feature in plain longitude/latitude, where a degree
east–west counts as much as a degree north–south although it is cos φ times shorter: the located
point is then not the nearest one when the feature runs obliquely, and the distance reads long —
by up to 106% (HydroLAKES), 96% (HydroRIVERS) and 88% (Natural Earth rivers) at 60–70° on a
13,372-pair test (`build_data/geodesic_distances/validate_rule.txt`). The corridor census chose its candidates within 0.9°
(100.2·cos φ km east–west), measured its gap between the points nearest in degrees and drew its
facing frontage with a degree buffer. The specification
(`build_data/geodesic_distances/SPEC_gd1_geodesic_distances.md`, maintainer approval 2026-09-23,
after the maintainer asked why the census reach is 0.9°) measures every distance on the ground: the
nearest point is located in a local frame in which a metre east equals a metre north (longitude
scaled by cos φ·N/M) and, beyond 2.5 km, again in an azimuthal-equidistant frame centred on the
point; the distance is measured with `GEOD.inv` (`geodist.nearest_on_ground`; within 0.013 m of an
exact computation on the test pairs, and no cached distance grows over 2,586,484 border samples).
The border arc's facing test measures its width and reciprocity the same way, and its midpoint
test no longer depends on rounding: a chord shorter than 1 m is contact and stays in the arc unless
its point lies within 2×10⁻⁷° of a stretch already in the arc (a repeat of that stretch's end), and a
longer chord's midpoint counts as on an outline within 10⁻⁸° (maintainer decisions; the exact test
first adopted had dropped 389 km of outlines that coincide within rounding from 45 arcs); the arcs
now hold 5,595 km of facing stretches on 630 edges (5,903 km on 954 before, the difference being
junction points the earlier code counted twice), the other classes unchanged. The
corridor census (`census_gd1.py`) reads the build's polygons and takes every pair within 100 km of
each other on the ground (candidates by a query widened by 1/cos φ, with the shorter degree length
near the equator); per pair it works in an azimuthal-equidistant frame centred on the nearest
approach (frontage, transects and samples at equal intervals of at most 250 m and 100 m); a pair
the build's polygons join at a single point while the World Bank polygons keep it apart is measured
on the World Bank polygons (maintainer decision; two pairs: Caraș-Severin↔Bor, the shipped Danube
row, nominated, and Tarija↔Jujuy, not nominated). Measured blind, the edge screens nominate 3,285
edges (3,248 before), 43 of them new (14 with a two-model record, 29 without) and 6 no longer (each
with a record, none shipped); every shipped row keeps its nomination (779/779); four auto-reject
ledger rows no longer satisfy the rule (a base share at 0.20 or more). The census holds 23,410 pairs
(20,653) and nominates 243 (245): 12 new (1 with a record), 14 no longer (four Arusha pairs the
source-relabel moves apart, ten whose shares fall below the bar when measured on the ground), all 24
shipped rows again. The maintainer's map check of a 30-map sample found no measurement issue. The
44 pairs without a record (29 + 4 + 11) went through the standard two-pass, blind: GPT-5.6 Sol
research (44/44 valid; water-only on none) and Sonnet-5 adversarial judgment (44 agents; water-only
on none), gate A 0 / B 0 / C 8 / D 36, maintainer: no override. **No shipped row changes.**
`gd1_screens.csv` and `census_gd1.csv` are now the screens' record: the completeness check reads
their bands (0 open) and the attributability check holds (803/803: 779 by the edge screens, 24 by
the census on the ground). Record: `build_data/geodesic_distances/` (`SPEC_gd1_*`, `gd1_*`,
`census_gd1*`).

**The crossing screen's width on the ground (`cw1`, 2026-09-24).** Layer 1 of the crossing pipeline
counted a bridge way when it intersected both units' polygons buffered by 0.0012°, a default carried
over from the June classifier (whose first version used 0.0009°, marked as not calibrated): 133 m
north–south but 133·cos φ m east–west, 55 m on the northernmost shipped row (the Torne, 65.7°). A
read-only dry run (2026-09-23, `build_data/water_screen_rebuild/bridge_unification/bridge_width_*`)
re-ran the screen's queries over the 803 rows with the search widened to 1 km and measured the
distance on the ground from every bridge way to both units (26,831 ways): on 295 of the 406 bridged
rows a bridge way meets both polygons; on the 95 whose nearest bridge way falls short of a unit
within 1 km, the median shortfall is 94 m; the screen's disagreement with the reviewed flags is flat
from 25 m to 250 m (100–108 rows) and rises on both sides (129 at 0 m and at 300 m, 155 at 500 m).
No mapping standard sets such a width (§2.2). On the maintainer's request for a width measured on
the ground, the value left to the coordinating model, the specification
(`build_data/water_screen_rebuild/crossing_width/SPEC_cw1_crossing_width.md`, frozen 2026-09-24) sets
**100 m on the ground**: a round value inside that flat range, near the median shortfall, and of the
round values tested the one that reopens the fewest settled rows (6), measured in an
azimuthal-equidistant frame centred on the way. The dry run also showed that the 2026-09-15 screen
had reported no bridge way on four rows whose ways existed unedited and which its own query returns
today: its client accepted any HTTP-200 answer, and Overpass reports a query that ran out of time
inside such an answer. The screen now counts an answer only if it reports no error (on the re-run it
rejected one, Illinois↔Iowa, 0 elements, where the dry run's client had accepted one and lacked 105
ways), and the location test locates its two distances on the ground (no outcome changes on the 89
located rechecks on record). Re-run over the 803 rows (OpenStreetMap as of 2026-09-24; `cw1_screen.py`)
and validated way by way against the dry run (every difference an OSM edit or that incomplete
answer; the row verdicts identical), the screen agrees with the shipped flag on 700 rows; 97 of the
other 103 already carry an adversarial recheck. Seven rows went to the GPT-5.6 Sol adversarial
recheck, the six without one and Panamá↔Panamá Oeste, which the way-class rule sends back on the new
record (within 100 m only the outline of the Puente de las Américas and two footways; its trunk-road
way lies 117 m from one unit): six confirmed their flags (the Hollandse Brug, the Rāmnieki bridge,
the Puente de las Américas, two Ugandan road bridges; Amazonas↔Loreto has only a footway and stays
unbridged), and one proposed an upgrade, Gambela↔Pibor Administrative Area across the Akobo: the Raad
Bailey road bridge (100 m, built in 2010 by Ethiopia's road authority; WFP Logistics Cluster), its
cited coordinates 16 m from the shared border, accepted by maintainer ruling (2026-09-24). Bridged
406 → **407**, unbridged 397 → **396**, moderate 8,061 → **8,062**; stringent, lenient and the ADM0
roll-ups unchanged; the seven rechecked rows carry a `cw1` clause after their `bu1` clause. On the new record 348 bridged rows have a bridge way within 100 m of both units,
337 of them a road or rail way; of the other 11, 9 rest on a structure on record and 2 were confirmed
by a recheck. Record: `build_data/water_screen_rebuild/crossing_width/` (`SPEC_cw1_*`, `cw1_*`).

**The de facto administrators (`da1`, 2026-09-25; §5).** Not a water change: each disputed-area tract
is now assigned whole to the administrator Natural Earth v5.1.2 records for the largest part of it,
which adds three de facto province pairs (Jerusalem↔Ramallah; Southern Darfur↔Warrap and Southern
Kordofan↔Warrap across Abyei). A de facto edge has no border arc, so the edge screens' population
stays 8,421 and no water row changes. The corridor census takes only pairs that are not edges: the
three pairs leave its population (23,410 → 23,407), and Jerusalem↔Ramallah, nominated by its
short-gap presence rule (a 422 m gap) and rejected in the nt2 two-pass as a dry gap, leaves its
nominations (243 → 242). The census computes each pair on its own, so dropping the three pairs from
its record gives what a re-run on the new edge list gives. Record: `build_data/ndlsa_ne/`.

**The rule rejections adjudicated; the union at every ladder width (`mr1`, 2026-09-26).** The July hybrid auto-reject
rule closed, without any model looking, an edge nominated only by a wider Natural Earth river rung (5 to 20 km) whose
share at every layer's base width is below 0.20; it held for 269 edges on the current record (207 domestic, 62
cross-border). It could miss a border that only the wider rungs see: the Orinoco border Anzoátegui↔Bolívar, a shipped
water-only row first nominated at 10 km, reaches 0.24 at best at a base width (HydroRIVERS). The rule is dropped, and the
269 were adjudicated by the standing two-pass (blind GPT-5.6 Sol research, Sonnet-5 adversarial judgment): water-only on
none (gate A 0 / B 0 / C 22 / D 247; maintainer: no override). The cross-type union now takes the widest rung of each
Natural Earth ladder (rivers 20 km, lakes 1,500 m; HydroRIVERS and HydroLAKES 500 m; bar 0.80), so it combines the
layers at every width the ladders measure: 1,144 → 2,045 nominations (723 → 744 of them accepted), one new candidate
(Plužine↔Šavnik, already adjudicated, not water-only); edges nominated by any screen 3,285 → 3,286. Every nominated
candidate now carries a two-model verdict (the completeness check no longer lets the rule's ledger, or the July audit
queue that lists its rows, close a nomination); attributability 803/803. No shipped row changes. Record:
`build_data/water_screen_rebuild/rule_rejections/`.

**`has_bridge` classification.** A pair is `True` iff a road/rail **bridge,
causeway, dam-top road, or tunnel** (not a ferry — ferries are OSM relations and
are excluded) lies in **both** units. Every one of the 803 rows now carries the
same **four-layer** record (crossing unification, 2026-09-16, campaign `bu1`; layer 1 re-run over every row on 2026-09-24 with its width measured on the ground, campaign `cw1`; the row added on 2026-09-18 went through the same pipeline, and the row that returned on 2026-09-22 keeps its 2026-09-16 record;
before it the first-round rows, the 2026-07 rows and the folded
earlier rows had come through three implementations of the same design):
(1) a deterministic **OSM Overpass screen** — a bridge/tunnel/causeway way counts
only if it comes within **100 m of both units' polygons, measured on the ground**
(the OSM↔WB registration tolerance; `cw1` above), construction/proposed tags are
dropped, the class of the way is not tested (a footpath bridge passes; the classes
are recorded separately, `wc1` and `cw1` above), non-touching pairs use the
nearest-approach corridor, the search reaches every way within 1 km of the border
(the border's bounding box padded by 1 km, or, on borders wider than 0.5 deg², an
around-sweep of 1,300 m radius at points 0.005° apart along the line), and an
Overpass answer that reports an error counts as a failed query and is repeated; (2) the row's
**independent web verification** (the research pass of the campaign that
nominated it); (3) an **adversarial recheck** (run on GPT-5.6 Sol, the research
model) of every row the screen does not
settle — a disagreement with the shipped flag, a bridged row with only a
tunnel/dam-top way, or a bridged row with no location on record — which must
name the structure, its coordinates, and answer separately whether it is open to
traffic and lands in both units; (4) a deterministic **location test** — the
coordinates must lie within 500 m of the shared WB arc or within 750 m of both
units, both distances located and measured on the ground (a mid-span pin over a
water strip the source assigns to neither unit),
750 m–3 km flagged as a bank-line gap for maintainer judgment, farther treated
as a wrong-unit citation — and **maintainer map rulings** on every proposed
flip. The flag is therefore a **reviewed static artifact**, shipped directly;
the Stage 3–4 engine (`scripts/apply_overlays.py`) copies it from the reviewed
water file and computes only the deterministic ADM0 roll-up. Full method, error taxonomy,
and per-pair sources: `BRIDGE_CLASSIFICATION_METHODOLOGY.md` (the June run);
the unified run's record: `build_data/water_screen_rebuild/bridge_unification/`; the
current screen's: `build_data/water_screen_rebuild/crossing_width/`.

**Two boundary conventions worth stating.**

- *Open to traffic* means the fixed link is **structurally complete**. A
  finished bridge on a **politically closed** border (e.g. Armenia–Turkey,
  closed since 1993; Tajikistan–Afghanistan) still counts — pericoupling is a
  *structural* relation and closures are transient; only
  **under-construction/proposed** links are dropped. By the same logic a
  structurally complete railway bridge on a **disused** line with its track in
  place counts, because the line can reopen (maintainer ruling 2026-09-20 on
  Telšiai↔Vaiņodes novads, the one row of the way-class diagnostic whose only
  structure is such a bridge); a dismantled line would not. Political openness is not
  checked for any other (land-border) pair, so applying it selectively here
  would be inconsistent.
- *Mid-lake "median-line" meetings* (two units meeting in open water — e.g. Lake
  Victoria, the Great Lakes, Lake Constance) are **native edges** in the
  build — no lake filter removes them — so `coupling_standard` governs them
  directly, exactly like river borders (the water file classifies 51 existing edges as
  lake borders beyond the first round, twelve of them first found by the HydroLAKES
  cross-check, and adds the three non-touching lake
  borders — Malësi e Madhe↔Bar across Lake Skadar, the one cross-border
  case, Kampong Thom↔Pursat across Tonlé Sap and Jura↔Neuchâtel across the Lac de
  Biaufond; the former Peipus restoration was removed 2026-07-22): `lenient` keeps every audited water contact, `moderate`
  keeps only the twelve lake pairs with a fixed crossing (Flevoland↔Noord-Holland
  via the Hollandse Brug, Flevoland↔Gelderland via the Nijkerkerbrug,
  Södermanland↔Uppsala via the Hjulstabron, Sud-Kivu↔Rwanda's Western Province
  via the Ruzizi bridge, three Ontario↔USA Great Lakes crossings — Michigan,
  Minnesota, New York — and the five later additions ALB011↔ALB017,
  IRL002↔IRL024, NLD011↔NLD012, SLV003↔SLV004, TUR026↔TUR055), and
  `stringent` keeps none.
  `lenient` therefore equals the shipped base adjacency (8,461 edges).

## 9. Name resolution (lookup layer)

Not part of the geometry build, but part of using the shipped data: the
runtime lookup layer resolves free-text region names to World Bank ADM1 codes
(`resolve_adm1_code`) through ordered strategies, each designed to fail to
`None` rather than guess:

1. **Alias table** (`data/adm1_aliases.csv`; 1,145 validated English exonyms /
   alternative spellings for 863 regions across 136 countries, PR #60/#61) —
   `"Bavaria"` → `DEU002`, `"Tuscany"` → `ITA016`. Additions-only,
   deterministically validated, with a curated review denylist.
2. **Exact / normalized match** against the official WB names, tolerant of
   possessives, hyphens, and administrative suffixes.
3. **Accent-folded fallback** (PR #45): lookup keys and queries are
   NFKD-normalized with combining marks stripped, so unaccented input matches
   accented names in either direction (`"Michoacan"` → `Michoacán de Ocampo`;
   `"Jõgeva"` ≡ `"Jogeva"` → `EST006`). The fold also **transliterates the
   handful of standalone letters NFKD cannot decompose** (Ł/ł, Đ/đ, Ø/ø, Ð/ð,
   Þ/þ, Æ/æ, Œ/œ, ß, ı), so native-script queries resolve the ASCII names the
   database stores — `"Łódź"` → `POL003`, `"Đắk Lắk"` → `VNM016`. The same
   fold builds the index and folds queries, so matching stays symmetric.
4. **Token-based substring match** with direction guards (PR #61): a query
   must match whole name tokens, so `"york"` does not match `New York` and
   grammatically declined stems (`"Krāslava"` vs `Krāslavas novads`) do not
   stem-match — deliberate strictness; such cases are handled by adding an
   alias-table entry, not by loosening the matcher.

Ambiguous names (several candidate regions, or a name denoting a different
place, e.g. `"Mexico City"` vs the State of México) return `None`. Full usage
documentation: `MANUAL.md` §8 and §12.

## 10. Source-relabel: fixing sliver-corridor artifacts at the source

Some WB Admin-1 polygons carry a **sliver corridor** — a thin ribbon of one
unit's territory mislabeled onto a neighbour, tracing an international border
for tens of kilometres. These artifacts corrupt the graph in a way no snapping
tolerance can detect or repair: the ribbon *genuinely touches* the units across
the border, so the fabricated edge is present at every tolerance (a fake
*touch*, not a gap). Stage 1 therefore applies a dedicated correction — the
source-relabel — before contiguity is computed.

### 10.1 Detection (deterministic shape screen)

The archetype is **TZA001 Arusha**, whose polygon grows two ribbons tracing the
Kenya–Tanzania border: a ~151 km × ~166 m NW tentacle reaching Lake Victoria
(really Mara's land) and a ~68 km × ~175 m E tentacle (really Kilimanjaro's) —
these are exactly the two Kenya "offset corridors" the land-gap audit (§4)
flags. Opening decomposition reproduces every affected shipped border length to
four decimals, confirming the mechanism.

The class was then swept exhaustively with a deterministic shape-screen
(morphological opening; not currently shipped as a re-runnable `scripts/`
tool — the frozen output is `build_data/arusha_sliver_audit/scan_all_units.csv`,
directly verifiable) over all 3,591 polygons, flagging appendages **≥ 10 km
long with ≤ 500 m mean width**. The signature is diagnostic because real units administer
territory (towns, farms) while a 10–150 km ribbon a few hundred metres wide is
the shape of a *digitization offset* — the band left between two renderings of
the same border, glued to whichever polygon was drawn outermost (the
Kenya–Tanzania offsets measure 73–101 m; Arusha's ribbon 166 m). The scan
found **144** candidates, of which **14** touch a cross-country border across
**13** host polygons (Arusha carries two corridors).

### 10.2 Adjudication (independent geodata, two passes)

The screen only nominates, and every flagged host is treated identically. Each
was ground-truthed against **independent geodata** — GADM 4.1, geoBoundaries,
Natural Earth, OSM/Nominatim, and official government border open-data — in a
research pass plus an adversarial judgment pass (unanimous): **10 confirmed
artifacts** (their strips belong to a same-country neighbour — the reviewed
relabel manifest) and **3 genuine** narrow territories, correctly kept — the
Vennbahn treaty corridor (BEL), the Courantyne west-bank strip (SUR), and the
Dhekelia road corridor (GBR). Thresholds nominate, audits decide. Evidence:
`build_data/arusha_sliver_audit/scan_ground_truth.md`.

**Re-adjudicated under the current design (`sr1`, 2026-09-20).** That audit ran
before the research pass moved to GPT-5.6 Sol — the first Codex handoff is
2026-07-10 — and its workflow script and output were not kept, so no model was
on record for either of its passes while every other adjudicated input of the
database carries one. The shape screen was therefore re-run on the pinned
GeoPackage (144 candidates, 14 touching a cross-country border across 13 hosts —
identical to the frozen scan) and all 14 strips went through the standard
two-vendor pass, blind to the shipped treatment: GPT-5.6 Sol research, then a
Sonnet-5 adversarial judgment pass with no repository access. Both passes
confirmed every attribution, owner codes included — 11 strips to a same-country
neighbour, 3 kept as the host's own territory — so the gate proposed no change
(K 12 / C 2, the two being the judge's requests for a human glance, closed by
the maintainer the same day; one asks whether the Dhekelia link road is
sovereign territory or a right of way, the other whether GADM and OSM are
independent sources for Uganda — where the source's own geometry leaves Kitgum a
residual 0.89 km of frontier under the ribbon, and the reassignment conserves the
two units' frontage, 171.02 + 0.89 → 150.63 + 21.28 km). **No data change**; the relabel manifest
now rests on a model-labelled adjudication
(`build_data/sliver_readjudication/`, `sr1_*`).

### 10.3 The fix (pure geometry, area-conserving)

For each reviewed host (`data/sliver_corridor_relabel.csv`, 10 hosts / 11
rows — Arusha is the only multi-owner host):

1. **Detach** the ribbon by morphological opening: `buffer(-D).buffer(+D)`
   removes anything thinner than ~2D and regrows the body; `corridor =
   polygon − opening` (parts > 0.5 km²). D = 2×10⁻³° (~220 m) for every
   host; the manifest's `opening_d_deg` column would record a larger D for a
   ribbon the default does not detach, and none needs one.
2. **Assign** each detached piece to the reviewed owner whose boundary it
   touches (Arusha resolves automatically: NW piece → Mara, E pieces →
   Kilimanjaro). Safety rails: a thin part touching *no* reviewed owner is
   left on the host (not the targeted artifact); touching *two* owners is a
   hard error; an owner receiving *nothing* is a hard error.
3. **Close at a junction**: the opening's outline meets the host's raw
   outline at a *cut point* inside a raw segment. Where that segment faces a
   third unit and ends at a junction — a World Bank vertex where the unit
   across the outline changes — the corridor is closed at the junction, so it
   carries all of the host's frontage with that neighbour or none of it. Two
   corridors close this way (maintainer ruling, 2026-09-22): Salta's, whose cut
   falls 304.2 m short of the Potosí–Tarija point, and Braničevo's, whose cut
   falls 13.8 m past the Caraș-Severin–Mehedinți point. Salta↔Potosí and
   Braničevo↔Mehedinți then meet at a single point, and Bor meets
   Caraș-Severin at that point too, leaving their Danube border to the
   non-touching water borders of Stage 3 (§8).
4. **Node**: every other cut point is inserted as a vertex into each polygon
   whose outline passes through it — host, owner and the unit across the
   border — so the new three-unit junction is a shared vertex. Without it the
   host would keep the rest of the raw segment as a zero-width spike and the
   owner's new frontage would coincide with no neighbour's line: the
   exact-contact measure would credit the host and miss the owner
   (Lamwo↔Eastern Equatoria would keep 8.36 km that is Kitgum's, and
   Kitgum↔Eastern Equatoria read 12.92 km for 21.28 km).
5. **Move, never delete**: host loses exactly the pieces, owners gain exactly
   them. Measured global area drift: below 0.0001 km² (the standalone run
   below), and for every group of relabels whose units touch, each outside
   unit's frontage with the group is conserved
   (`build_data/sliver_remnant/check_relabel_fix.py`).

Contiguity then runs on the corrected polygons, so every downstream number
(edges, lengths, water rows, the §2.4 sweep) inherits the fix consistently —
one deterministic operation instead of a pile of post-hoc edge patches.

### 10.4 Impact ledger (what the relabel changed, pair by pair)

| effect | pairs |
|---|---|
| **4 fabricated cross-country edges removed** | Migori↔Arusha (83.2 km → 0), Taita-Taveta↔Arusha (22.4 → 0), Salta↔Potosí (19.1 → 0), Braničevo↔Mehedinți (10.6 → 0). All four would otherwise ship as eligible pericoupled pairs (three under every standard; Braničevo↔Mehedinți, water-only unbridged, under `lenient`). |
| **1 orphaned water row deleted** | Braničevo↔Mehedinți's Danube row classified an edge that no longer exists and was removed from the first-round water classification with it. |
| **Starved true borders recovered** | Migori↔Mara 20 → 103 km (raw), Kitgum↔E. Equatoria 0.9 → 21.3, Taita-Taveta↔Kilimanjaro 148 → 170. |
| **Rightful owners absorbed the strips** | Jujuy↔Potosí 302 → 321, Bor↔Mehedinți 153 → 164. |
| **7 length-only artifacts corrected** | Gedo, Wajir, Galgaduud, Lamwo, ʿAsīr, Atyrau, Béchar — the true adjacency existed via the correct unit; only `border_length_km` was starved or inflated (e.g. Wajir↔Lower Juba 69.7 → 85.1, Garissa↔Lower Juba 198.7 → 255.6, Mudug↔Ethiopian-Somali 12.9 → 46.2 — all *increases*, since the ceded corridor lengthens the rightful owner's true frontage). |
| **2 pairs upgraded overlay → native** | Kajiado↔Kilimanjaro and Narok↔Mara touch at exact contact once the ribbon moves — no land-gap overlay row is needed for them. |

Net effect on the exact-contact native count: **8,427 → 8,425** (−2 — four
fabricated cross-country edges removed, two Kenya survey-line pairs made
native); 30 border lengths change; one orphaned water row deleted
(records: `build_data/build_record/relabel_edge_delta.txt` for the edge set,
`build_data/sliver_remnant/relabel_length_delta.txt` for every length).

### 10.5 Verification and reproduction

`tests/test_relabel_sliver_corridors.py` asserts the manifest's integrity in
CI and — when the pinned GeoPackage is present — area conservation, all 12
corridor moves, the four bogus edges at zero contact, the starved-edge
recoveries, no zero-width spike in any host or owner, the owner taking the
host's frontage exactly (Kitgum↔Eastern Equatoria 21.28 km, the two units'
frontage conserved), and the two junction closings. Standalone re-run:

    python scripts/relabel_sliver_corridors.py --adm1-gpkg "<WB Admin 1 .gpkg>"

Full verification record: `build_data/build_record/relabel_RESULTS.md`; scan +
adjudication evidence: `build_data/arusha_sliver_audit/`.
