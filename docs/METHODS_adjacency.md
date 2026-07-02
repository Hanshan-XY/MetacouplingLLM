# Methods: ADM1/ADM0 land-border adjacency construction

Companion to `src/metacouplingllm/data/PROVENANCE.md`. PROVENANCE records *what
the data is and its known flaws*; this document records *why the methodological
choices are defensible* — the contiguity rule, the snapping tolerance, the
prior-practice context, and the sensitivity analyses that back the parameters.

All numeric results below were computed from the World Bank Official Boundaries
2026-05-14 GeoPackages (ADM1 `WB_GAD_ADM1`, 3,591 polygons; ADM0 `WB_GAD_ADM0`,
264 polygons) with `scripts/build_pericoupling_db.py`.

**Construction at a glance.** The shipped database is the output of exactly two
stages, both committed: **(a)** one deterministic geometry build
(`scripts/build_pericoupling_db.py`, from SHA-256-pinned inputs — see
`data/PROVENANCE.md` — including its internal one-entry Malta denylist and the
geometry-derived 13-pair disputed overlay), and **(b)** one reviewed
**correction layer**: five manifest CSVs holding **88 pair-rows** (72
edge-restoring + 16 water-flag-only), each pair individually audited and
human-verified, plus the 314-row bridge classification, applied in a single
pass by the idempotent engine `scripts/apply_overlays.py`. One command
reproduces the whole database (§7): `python scripts/build_all.py --full ...`
from the pinned sources, or `python scripts/build_all.py` to re-derive and
verify from the shipped data. Everything else in this document — the
sensitivity sweeps, the near-miss and completeness audits, the buffer
re-screens — is the **discovery and validation record** that produced and
defends the correction layer; a reader reproducing the database never needs to
re-run it.

---

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

## 2. Snapping tolerance

### 2.1 What it does

Adjacent administrative polygons from independent sources frequently do **not**
share identical vertices along their common boundary; the two renderings of the
same line sit a small distance apart. An exact-intersection test then returns
*empty* and misses a real border. The snapping tolerance `SNAP_TOL_DEG = 5×10⁻⁴`
degrees (≈55 m at the equator, shrinking E–W toward the poles) bridges this
sub-tolerance offset: if the boundaries do not intersect exactly but lie within
the tolerance, one boundary is intersected against the other polygon buffered by
the tolerance.

Concrete illustration (verified): the Egypt–Libya border between Matrouh
(`EGY016`) and Ajdabiya (`LBY001`) is a ~49 km straight segment whose two
renderings are offset by **~0.4 m** and share **zero** vertices. Exact
intersection (tolerance 0) recovers **no** shared boundary; a tolerance ≳ 1 m
recovers the full 49 km. This shows the tolerance is *necessary*, not cosmetic —
a strict zero-tolerance rule omits genuine borders.

### 2.2 There is no canonical value to cite — and that is the documented state of practice

A literature/tool survey (GIS-topology and spatial-data-quality sources) found
**well-cited standards for the *mechanism* of contiguity but no canonical
numeric snapping-tolerance value**:

| Tool / standard | Tolerance | Notes |
|---|---|---|
| ArcGIS Pro "XY / cluster tolerance" | **0.001 m (1 mm)** default; = 10× the XY *resolution* | A vertex-*coincidence* threshold; docs say it "should never approach your data capture accuracy" and should exceed ~2× resolution (Esri ArcGIS Pro topology docs). |
| GRASS GIS `v.clean` (snap) | **no default** (user-supplied `thresh`) | In map units; **degrees** for lat-long data (GRASS manual). |
| PostGIS `ST_Snap` | **no default** (mandatory argument) | (PostGIS docs). |
| GeoDa "precision threshold" | ~`0.0001` desktop / `0` API | "fuzzy" band used only when coordinate precision is insufficient for exact match; in layer coordinate units → degrees for unprojected data (GeoDa workbook; rgeoda/pygeoda). |
| NMAS (1947), NSSDA (FGDC-STD-007.3-1998) | **set no threshold** | NSSDA §3.1.2 verbatim: "This standard does not define threshold accuracy values." Reporting standards (NSSDA: Accuracy_r = 1.7308·RMSE_r at 95%), not tolerance prescriptions. |

Two important cautions from the survey:
- ArcGIS's 1 mm default and our 5×10⁻⁴° (~55 m) are **different quantities**:
  ArcGIS's is a *coincidence* threshold (snap vertices meant to be identical),
  ours is a *cross-source-discrepancy* threshold (bridge two renderings of the
  same border). They are not comparable, and ours is appropriately larger.
- The intuition "set tolerance ≈ source positional accuracy (RMSE)" is **not**
  supported: ArcGIS guidance is the opposite (tolerance ≪ capture accuracy), and
  the epsilon-band/RMSE link was not corroborated in the surveyed sources.

### 2.3 Data-relative anchor

The WB ADM1 coordinates are stored at full double precision (no fixed grid;
median ~14 decimal places). Their **median inter-vertex spacing is ≈3.9×10⁻⁴°
(~43 m)**. The chosen tolerance (5×10⁻⁴°) is therefore ≈1.3× the data's own
native vertex spacing — large enough to bridge cross-source offsets at the scale
of the source's own resolution, small enough not to reach across genuine gaps.

### 2.4 Sensitivity analysis (the primary justification)

Because no canonical value exists, the defensible justification is **robustness**.
Re-running the full adjacency build (lake filter retained) at a range of
tolerances:

| tolerance (°) | ≈ metres | ADM1 edges | Δ vs 5×10⁻⁴ | ADM0 pairs |
|---|---|---|---|---|
| 0 | 0 | 8,363 | −6 (−0.07%) | 324 |
| 1×10⁻⁴ | 11 | 8,366 | −3 | 324 |
| 2×10⁻⁴ | 22 | 8,366 | −3 | 324 |
| **5×10⁻⁴** | **55** | **8,369*** | — | **324** |
| 1×10⁻³ | 111 | 8,376 | +7 (+0.08%) | 324 |
| 2×10⁻³ | 222 | 8,391 | +22 (+0.26%) | 325 |
| 5×10⁻³ | 555 | 8,456 | +87 (+1.04%) | 325 |

\* Raw build figure. The shipped ADM1 count is **8,453** (8,369 − 1 §4 Malta
removal + 13 §5 de-facto overlay + 6 river-gap overlay + 64 lake overlay + 2 land-gap overlay, §8-9);
the shipped ADM0 count is **326** (the §5 overlay's net addition MAR/MRT plus
the lake overlay's COD/TZA are tolerance-independent constants — this sweep's
ADM0 column reflects the 2-pair disputed allowlist in force when it was run).

Over **[0, 10⁻³°]** the ADM1 edge count varies by **<0.1%** and the ADM0 matrix
is **invariant** (324 pairs in this raw sweep; **326** shipped\*). Counts climb only above ~2×10⁻³°, where loosening
the tolerance begins merging genuinely separate units (e.g. across rivers and
straits); the additions there are increasingly cross-country with multi-km
"shared" lengths. The result is therefore **insensitive to the precise tolerance
within the plateau**, and 5×10⁻⁴° lies in its interior.

The **87 pairs added above the shipped tolerance** (7 at 1×10⁻³, +15 at 2×10⁻³,
+65 at 5×10⁻³; 23 cross-border + 64 domestic) were audited per pair with the §4
recovered-length-vs-tolerance diagnostic: **all 87 show the growing gap-corridor
signature** and none the flat genuine-border signature. Every cross-border
addition falls within the ground-truthed 48-pair near-miss set: five are the
river-gap overlay pairs already restored to the shipped graph, and **two long
survey-line corridors on the straight-surveyed Kenya-Tanzania border —
Kajiado↔Kilimanjaro (~54 km) and Narok↔Mara (~70 km), whose dry-land adjacency
lay outside the river audit's scope — were map-verified as genuine borders and
restored as the land-gap overlay** (`land_gap_overlay_pairs.csv`, applied by
`scripts/apply_overlays.py`; ordinary land edges, pericoupled under
every standard). The remaining cross-border additions and all 64 domestic
additions are confirmed artifact corridors. Per-pair results:
`build_data/snap_extras_audit/extras.csv` (local audit artifact, not shipped).

---

## 3. River buffer (`RIVER_BUFFER_DEG`) — affects reported length only

Natural Earth rivers are **centerlines** (1-D, zero width); the shared border is
also a line, so subtracting a zero-width line removes nothing. `RIVER_BUFFER_DEG`
= 2×10⁻³° (≈220 m N–S) inflates each centerline into a thin ribbon so that the
stretch of border running *along* a river falls inside the ribbon and is removed,
leaving the *true dry-land* length in `border_length_km`. **This subtraction
feeds only the length column and the advisory flags derived from it; adjacency is
decided earlier, from the lake-subtracted length, so the river buffer cannot add
or drop an edge.**

Sensitivity (ADM1; snap tolerance and lake filter held fixed):

| river buffer (°) | ≈ m (N–S) | edges | total length (km) | `narrow`<5 km | `artifact`<1 km |
|---|---|---|---|---|---|
| 0 | 0 | 8,369 | 1,207,370 | 561 | 156 |
| 5×10⁻⁴ | 56 | 8,369 | 1,201,945 | 565 | 157 |
| 1×10⁻³ | 111 | 8,369 | 1,196,527 | 566 | 158 |
| **2×10⁻³** | **222** | **8,369** | **1,185,809** | **573** | **158** |
| 4×10⁻³ | 444 | 8,369 | 1,165,559 | 596 | 166 |
| 8×10⁻³ | 888 | 8,369 | 1,132,042 | 640 | 205 |

The **edge count is invariant (8,369) at every width** — empirical confirmation
that the river buffer is a length-only parameter. (This harness reads the raw
edge set before the §4 removal and the §5/§8 overlays; the shipped count is 8,453.) Over
[0, 2×10⁻³°] total border length moves only ~1.8% and the advisory-flag counts
barely change; only at 8×10⁻³° (≈4× the chosen value) does the effect become
large (length −6%, and 36 short borders erased entirely). About **25% of edges
(2,064/8,369) follow a river** and are affected at all; for those the median
length shift from the chosen value to zero buffer is ≈2.2 km. The other ~75% of
borders touch no river and are identical at every width.

**Caveats.** (i) Like the snap tolerance, the buffer is specified in *degrees*,
so the ribbon is **anisotropic** — ≈220 m N–S but only ≈220·cos(lat) m E–W
(≈110 m at 60°). High-latitude river borders therefore have slightly *less*
length subtracted and read marginally long. (ii) A single 220 m ribbon cannot
match the true range of river widths (≪10 m streams to multi-km rivers such as
the Congo/Amazon), so it under-captures wide rivers and over-captures narrow
ones. Both effects perturb only `border_length_km`, never adjacency. A future
revision could buffer in a projected/equidistant CRS (uniform metric width) if
border lengths are used quantitatively.

---

## 4. Auditing the tolerance-sensitive band

Only **6 ADM1 pairs** are adjacency-sensitive to the tolerance (present at
5×10⁻⁴°, absent at 0); the change is purely additive (no edge is lost). Because
this set is small it was **manually reviewed against imagery**:

| pair | min gap | recovered length vs tolerance | verdict |
|---|---|---|---|
| EGY016 Matrouh ↔ LBY001 Ajdabiya | 0.4 m | ~constant (49.4 km) → real border | adjacent (keep) |
| DOM011 Independencia ↔ DOM026 San Juan | 39 m | grows with tol (gap corridor) | adjacent (keep) |
| ETH013 Tigray ↔ SDN004 Kassala | 31 m | grows with tol | adjacent (keep) |
| **MLT002 Balzan ↔ MLT019 Iklin** | **31 m** | grows with tol | **not adjacent (removed)** |
| AIA001 ↔ AIA003 (Anguilla) | 1.3 m | grows with tol | left as-is (domestic, unnamed) |
| MOZXXX ↔ MWI003 | 0.9 m | grows with tol | left as-is (MOZ placeholder unit) |

A **flat** recovered-length-vs-tolerance curve indicates a true shared border
(the whole length is at sub-metre offset, so loosening the tolerance adds
nothing); a **growing** curve indicates a gap corridor being progressively
swallowed. By this diagnostic only EGY/LBY is unambiguously a real border; the
others are gap-bridged and require human judgment — which is exactly why the band
is audited rather than trusted to any single threshold value. Note that min-gap
alone does **not** separate keep from drop (Malta's 31 m gap is smaller than
DOM's 39 m), so no tolerance value can get both right; the manual call is
load-bearing.

`MLT002`/`MLT019` was confirmed non-adjacent (the two councils are ~31 m apart
with no shared frontier; the tolerance fabricated the edge) and is removed via
`_ADM1_FALSE_POSITIVE_DENYLIST` in the build script (8,369 → 8,368). The
shipped edge list then adds the **13 de-facto disputed-territory overlay edges**
(§5), the **6 river-gap overlay edges**, the **64 lake overlay edges** (§8), and the
**2 land-gap overlay edges** (below), for a final count of **8,453**.

**Takeaway.** The snapping tolerance is *necessary* (it recovers real borders
that exact matching misses) yet *insufficient alone* (it can fabricate edges
across true gaps); the small tolerance-sensitive band is therefore audited
edge-by-edge rather than governed by the threshold value.

---

## 5. Disputed territories (de-facto vs strict)

WB's standard ADM0 (264-unit) and ADM1 (3,591-unit) layers exclude the
24-feature NDLSA disputed-areas layer. That exclusion carves each contested
tract out of **both** neighbouring units, opening a multi-km gap, so the
flanking units are recorded as non-adjacent even where they meet along the
de-facto line of control (e.g. China–Pakistan across Gilgit-Baltistan/Karakoram;
Israel–Syria across the Golan; Morocco–Mauritania across Western Sahara).

**Default = de-facto view.** Because metacoupling concerns connection, the
shipped data treats disputed land as part of its de-facto administrator. The
overlay is **derived from geometry at build time** by `derive_disputed_overlay()`
and ships (all 16 rows) in `data/disputed_overlay_pairs.csv`; the runtime loaders
accept `de_facto_borders` (default `True`) and, with `False`, subtract the
overlay to reproduce the WB standard-layer adjacency.

**ADM0 (country level)** — three pairs, each the de-facto administrator country
re-joined to the neighbour it meets only across a disputed tract:

| pair | de-facto admin | source tract(s) | shared border |
|---|---|---|---|
| `CHN`/`PAK` | PAK | Gilgit-Baltistan; Karakoram | ~491 km |
| `ISR`/`SYR` | ISR | Golan Heights; Shebaa Farms | ~79 km |
| `MAR`/`MRT` | MAR | Western Sahara | ~1,544 km |

**ADM1 (subnational level)** — derived **independently**, because country
adjacency does *not* imply province adjacency: two countries adjacent elsewhere
can still have the provinces flanking a tract meet only across it (India and
China are adjacent, but Arunachal Pradesh and Tibet are non-adjacent in the
strict layer — the tract sits between them). Driven by an authored, geometry-
validated tract→**province** map (`_NDLSA_TRACT_ADM1`); a tract spanning several
administering provinces has its frontier split among them by nearest province.
**13 pairs**:

| de-facto admin province | neighbour province(s) | tract | border |
|---|---|---|---|
| `IND003` Arunachal Pradesh | `CHN029` Tibet | Arunachal Pradesh | ~993 km |
| `IND003` Arunachal Pradesh | `BTN016`/`BTN015`/`BTN011` (E. Bhutan) | Arunachal Pradesh | 86/83/41 km |
| `ISR004` Northern | `SYR012` Quneitra | Golan | ~74 km |
| `ISR004` Northern | `SYR006` Dar'ā / `LBN004` Bekaa | Golan | 4.8 / 0.4 km |
| `BTN005` Haa | `CHN029` Tibet / `IND030` Sikkim | Doklam | 62 / 6.6 km |
| `MAR005` Guelmim + `MAR007` Laâyoune | `MRT012` Tiris-Zemmour | Western Sahara | 287 + 757 km |
| `MAR007` Laâyoune | `MRT004` Dakhlet-Nouadhibou / `MRT001` Adrar | Western Sahara | 405 / 96 km |

**`CHN`/`PAK` is ADM0-only — no ADM1 row.** Gilgit-Baltistan and Ladakh/Jammu &
Kashmir are disputed territories **excluded from WB's ADM1 layer** — not
provinces (WB lists only five Pakistani ADM1 units, none Gilgit-Baltistan; the
tracts overlap no province, i.e. true gaps). With no de-facto province to
attribute to, crediting the border to the nearest *existing* province (Khyber
Pakhtunkhwa / Himachal Pradesh) would be geographically false (e.g. "Assam ↔
Tibet"), so the relationship is carried at the country level only. The asymmetry
is honest: the Golan (Israel's Northern District) and Western Sahara (Morocco's
Laâyoune/Guelmim) *do* have administering provinces in WB; Gilgit-Baltistan and
Ladakh do not.

**Authored attribution (neutral framing).** The NDLSA layer carries **no**
administering-country field — `SOVEREIGN` is null for all 24 tracts and
`WB_STATUS` is uniformly "Non-determined legal status area" — so every
tract→administrator (ADM0) and tract→province (ADM1) mapping is hand-authored.
Each is **geometry-validated** (`derive_disputed_overlay` raises if an authored
unit does not touch its tract, so a mislabel fails loudly rather than silently
dropping a pair). It records effective/physical coupling across the de-facto line
for a connectivity dataset and is **not** a legal or endorsed sovereignty claim.

**Tract screening.** Of the 24 NDLSA tracts, **6** have no single de-facto
administrator (No Man's Land, the UN Buffer Zone, Abyei, and three island/EEZ
tracts — British Indian Ocean Territory, the Falklands, South Georgia & the South
Sandwich Islands) and are excluded. Of the remaining 18, the ADM0 overlay keeps
the four (→ 3 distinct country pairs) whose claimant countries are non-adjacent
under the strict layer; the ADM1 overlay independently keeps the 13 province
pairs above. Tracts that change no pair — every other India–China / India–
Pakistan tract, and the Ilemi Triangle (Kenya & South Sudan already share an
~80 km border) — are documented in the full per-tract candidate audit shipped at
`docs/ndlsa_tract_audit.csv`.

---

## 6. Scope / honesty notes

- The prior-practice evidence (§2.2) is predominantly **tool and standards
  documentation**, not peer-reviewed GIS-science; it establishes *standard
  practice*, not a citable optimum.
- The **lake filter** (Natural Earth `ne_10m_lakes` used to drop mid-lake
  contacts) has **no standard-practice citation** found in the survey; it is a
  pragmatic choice (documented in PROVENANCE) motivated by WB admin polygons
  including lake water.
- No standard **minimum shared-border-length** threshold exists in the
  literature; the rook rule is qualitative ("non-zero length"). The
  `narrow_border` (<5 km) and `potential_artifact` (<1 km) flags are advisory
  labels, not adjacency criteria, and have no canonical cutoff.

## 7. Reproducibility

One command reproduces the shipped database:

```
# full rebuild from the pinned sources (SHA-256-verified before running):
python scripts/build_all.py --full \
    --adm1-gpkg <WB Admin 1 .gpkg> --adm0-gpkg <WB Admin 0 .gpkg> \
    --ocean-gpkg <WB Ocean Mask .gpkg> --ndlsa-gpkg <WB NDLSA .gpkg>

# re-derive + verify from the shipped data (fast; no GeoPackages needed):
python scripts/build_all.py
```

The full mode verifies the inputs' SHA-256 against the pins (in
`scripts/build_all.py` and `data/PROVENANCE.md`; a mismatch is a hard error),
runs the geometry build, applies the correction layer with
`scripts/apply_overlays.py`, checks every headline count (edges, regions,
countries, water set, roll-ups, all three standard views at both levels), and
reports byte-identity against the committed CSVs. The refresh mode is a
byte-stable no-op on an untouched checkout — the day-to-day reproducibility
check, also enforced by `tests/test_apply_overlays.py`. The sensitivity sweep
and band audit remain reproducible from the build script by varying
`SNAP_TOL_DEG` and diffing the resulting edge sets.

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

**Data.** `data/water_separated_pairs.csv` lists the **337 ADM1** water-only
pairs with a `has_bridge` flag (314 land-classified + 6 river-gap overlay + 13
wide-river overlay + 1 lake-gap overlay + 3 audit-water overlay), plus **21 ADM0** country pairs rolled up from them (a
country pair is water-only iff *all* its ADM1 crossings are, and has a bridge
iff *any* does). Four reviewed overlays feed this set — the **river-gap
overlay** (6 near-miss river pairs restored as edges), the **wide-river
overlay** (13 existing edges reclassified water-only after a 5 km candidate
re-screen), the **lake overlay** (63 audited lake-only pairs plus 1 audited
lake-gap near-miss — Jõgeva↔Pskov across Lake Peipus, whose shores the source
digitizes as non-touching — restored as edges so the standard governs lakes
exactly like rivers), and the **audit-water overlay** (3 existing edges the
Natural Earth screens missed — Shirak↔Kars on the Akhurian, Vratca↔Olt on a
Danube main-stem span absent from the NE river list, Kagera↔Ntungamo on a
402 m Kagera-thalweg arc — flagged by the 10 km completeness audit's two-pass
verification and confirmed by human map review 2026-07-02; flags only, no new
edges) — each shipped as a
reviewed manifest (`data/*_overlay_pairs.csv`) and applied in one pass by the
idempotent engine `scripts/apply_overlays.py` (full provenance in
`data/PROVENANCE.md`). Under
the default, ADM1 pericoupled edges fall 8,453 → **8,232** and ADM0 country
pairs 326 → **323**.

**`has_bridge` classification.** A pair is `True` iff a road/rail **bridge,
causeway, dam-top road, or tunnel** (not a ferry — ferries are OSM relations and
are excluded) lies in **both** units. The flag was derived from OpenStreetMap
and then **independently verified** — a parallel web-search pass per pair, an
adversarial recheck of every disagreement, a deterministic geocode +
province-polygon cross-check, and maintainer review. It is therefore a
**reviewed static artifact**, shipped directly; the build script
(`write_water_separated_manifest`) regenerates only the deterministic ADM0
roll-up, not the bridge flags. Full method, error taxonomy, and per-pair
sources: `BRIDGE_CLASSIFICATION_METHODOLOGY.md`.

**Two boundary conventions worth stating.**

- *Open to traffic* means the fixed link is **structurally complete**. A
  finished bridge on a **politically closed** border (e.g. Armenia–Turkey,
  closed since 1993; Tajikistan–Afghanistan) still counts — pericoupling is a
  *structural* relation and closures are transient; only
  **under-construction/proposed** links are dropped. Political openness is not
  checked for any other (land-border) pair, so applying it selectively here
  would be inconsistent.
- *Mid-lake "median-line" meetings* (two units meeting in open water — e.g. Lake
  Victoria, the Great Lakes, Lake Constance) are removed from the geometry build
  by the inland-water (lake) filter (§6; PROVENANCE Method step 3) and restored
  by the reviewed **lake overlay**, so all three standards govern them exactly
  like river borders: `lenient` keeps every audited water contact, `moderate`
  keeps only the three lake pairs with a fixed crossing (Flevoland↔Noord-Holland
  via the Houtribdijk, Flevoland↔Gelderland via the Nijkerkerbrug,
  Södermanland↔Uppsala via the Hjulstabron), and `stringent` keeps none.
  `lenient` therefore equals the shipped base adjacency (8,453 edges).

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
