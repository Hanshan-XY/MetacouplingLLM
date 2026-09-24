# Specification: distances on the ground in the edge screens and the corridor census (campaign `gd1`)

Status: **APPROVED and FROZEN, 2026-09-23** (maintainer: "Approve as drafted"). Maintainer decisions of 2026-09-23:
the corridor census becomes fully geodesic ("C, please"); the same correction applies to the edge screens' distance step
("Census + edge screens"); the census reach is 100 km on the ground ("100 km"); section 5.1, the census reads the build's
polygons, approved with the draft. Nothing below had been run against the shipped rows except the dry runs of section 2.
After the freeze only implementation defects are corrected, each logged at the end of the file.

## 1. Purpose

Every distance a screen uses is to be a distance on the ground. Today several steps measure on the ellipsoid but choose
their points in plain longitude/latitude degrees, where a degree east–west counts as much as a degree north–south although
it is cos(latitude) times shorter:

- **The shared distance helper** (`build_data/geodesic_rescreen/geodist.nearest_geod_m`, and the census's `gdist_m`)
  locates the nearest point of a feature in degrees, then measures to it. The located point is not the nearest one when the
  feature runs obliquely: for a straight river 400 m away the helper reads 3–6% long at 45° latitude, 8–24% at 60° and
  11–60% at 70° (`helper_locate_error.txt`). A point within a screen's width can read outside it. Bound: the true distance
  is at least cos(latitude) times the helper's.
- **The border arc's facing test** (`border_arc.classify_facing`) measures the facing width w and the reciprocity distance
  between points located the same way.
- **The corridor census** (`corridor_census_v2/recovery_census_v2.py`) in seven places: (a) the candidate query,
  `dwithin 0.9` degree, whose east–west reach is 100.2·cos(latitude) km; (b) the gap, measured between the points nearest
  in degrees; (c) the facing frontage, a buffer of facing_m / 111,320 degrees, narrower east–west; (d) the transect end,
  the point nearest in degrees; (e) sample placement, even in coordinate length (the rule the edge screens left on
  2026-09-21); (f) the search pads of the lake and river tests, a fixed factor 1.6 that covers east–west only up to about
  51° at the outer widths; (g) the no-lake prefilter, a degree buffer around the nearest-approach chord only.

The census also reads the raw World Bank file, while the edge screens read the build's own polygons (section 5.1).

## 2. Dry runs already measured (read-only; `build_data/water_screen_rebuild/corridor_census_v2/reach_geodesic_*`)

- The census re-derived exactly as coded reproduces its record: 20,653 pairs, no difference.
- A geodesic 100.2 km reach adds 2,801 pairs, all 48–100 km apart (median 89 km); the census's code nominates none of
  them (highest lake share 0.57, 0.71 on the wide variant).
- Measuring the gap on the ground moves 26 existing pairs across the census's 1 km or 5 km gap limits; with the gap so
  measured, 3 of them are nominated (Malew–Rushen, Jaunjelgava–Lielvārde, Flanders–North Rhine-Westphalia), each with
  third units on at least 96% of its transects.
- The planar locate step of the helper (section 1) is not in these dry runs; it affects both screen families.

## 3. The distance rule (all screens)

The distance from a sample point p to a feature is the geodesic distance (WGS84, `GEOD.inv`) from p to the feature's
nearest point, the nearest point being located in a local frame centred on p in which a metre east equals a metre north
(longitude differences multiplied by cos(latitude of p)). Candidate features are found by a query widened by
1/cos(latitude) (`query_radius_deg`, cosine floored at 0.2), as the edge screens already do.

Validation, before any screen is run (a failure is an implementation defect, fixed and logged):
1. On at least 10,000 (point, feature) pairs spread over latitudes 0–80° and over the four layers, the distance agrees with
   an exact computation (nearest point in an azimuthal-equidistant frame centred on p, where distances from the centre are
   exact) within 0.5 m up to 2.5 km and within 0.1% up to 25 km.
2. On every point of the rebuilt arc cache that keeps its position, the new distance to each layer is at most the old one
   plus 0.01 m (the old distance was to an actual point of the feature, so it can only shrink).

## 4. Edge screens

The border arc keeps its definition (ba1 §4) and its numbers. Inside the facing test, the facing width w (from m to B's
outline) and the reciprocity distance (from B's nearest point back to A's outline) are measured by the distance rule, and
the nearest points are located by it. The chord remains the straight segment between m and B's nearest point: at 1,000 m
or less it departs from the geodesic by centimetres. Unchanged: the classes, the 1,000 m reach, the 500 m reciprocity
tolerance, the 1 m third-unit tolerance, the midpoint test, the land-gap band of 5×10⁻⁴° (the land-gap audit's own band,
four pairs only), sampling at equal geodesic intervals shorter than 500 m, the stored distance limits, the seven screens,
their widths and their bars.

The per-point distances to the four layers use the distance rule. The arc cache is rebuilt in full, because the facing
classification can change.

## 5. Corridor census

### 5.1 Geometry (for approval)
The census reads the polygons the build uses: the loader of ba1 §3 (validity-repaired, clipped to land by the Ocean Mask,
source-relabelled, RUS050 merged). Why: the census measures gaps between units as the database draws them, as the edge
screens have since 2026-09-22. Two consequences, both measured in section 7 and reported, not acted on beforehand:
(i) pairs that touch in the raw file only through water the Ocean Mask removes (an estuary or tidal channel drawn as sea)
are skipped today as point or line contacts and are never tested by either screen family; on the build's polygons they
have a gap and enter the census; (ii) pairs around the relabelled units follow the relabel (for example Jujuy–Tarija and
Braničevo–Mehedinți, point contacts on the build's polygons).

### 5.2 Population
Every pair of units whose geodesic gap is at most **100 km**, whose polygons do not touch, and which is not a shipped edge,
plus the shipped rescreen-gap rows (the rediscovery check). Candidates come from a query widened by 1/cos(latitude). The
gap is the geodesic minimum distance between the two outlines, located in an azimuthal-equidistant frame centred on the
midpoint of the nearest approach (re-centred once on the result) and measured with `GEOD.inv`. Point contacts (gap 0)
are skipped, as now.

### 5.3 Per pair
Computed in the pair's azimuthal-equidistant frame (scale error below 0.05% within 400 km of the centre):
- facing frontage: each unit's outline within facing_m of the other, facing_m = min(max(2·gap, gap + 750 m), 100 km)
  (the formula unchanged; its cap is the reach);
- frontage samples at equal intervals of at most 250 m, at most 200 per side (as now);
- transects from each frontage sample to the nearest point of the other outline, kept when the midpoint lies inside
  neither unit (as now);
- transect samples at equal intervals of at most 100 m, at most 3,000 per pair (as now);
- the third-unit share and the wedge ratio as now, on these transects.

Lake and river shares use the distance rule of section 3 with the widths, ladders and limits of today: lake 125 m (NE, and
HydroLAKES of at least 0.25 km²), river 500 m, evaluated for gaps of 5 km or less; the wide variant (lakes 1,500 m, reaches
of at least 1,000 m³/s at 2,500 m); the presence rule for gaps of 1,000 m or less. The no-lake shortcut applies only when no
lake polygon lies within 1,500 m on the ground of any transect (today: of the nearest-approach chord only). Nomination
rule unchanged: share sum at least 0.80, the wide variant, or the presence rule.

## 6. Every number in this specification

| Number | Used for | Source |
|---|---|---|
| 100 km | census reach | maintainer decision 2026-09-23 (the census's documented reach, "0.9 deg (~100 km)") |
| cos(latitude) | local frame; widened queries | geometry of the ellipsoid, as in `query_radius_deg` |
| 0.2 | floor of the cosine in queries | `query_radius_deg`, unchanged |
| 0.5 m, 0.1%, 0.01 m | validation tolerances | numerical |
| 0.05% within 400 km | frame scale error | property of the azimuthal-equidistant projection |
| every width, bar, rung, step and cap | screens and census | unchanged (ba1 spec §7–8; census METHODS §8) |

## 7. Freeze, then measure

1. **Freeze** on approval; later corrections are implementation defects only, logged below.
2. **Validation** of section 3.
3. **Blind measurement**, reported in full before any data change:
   - edge screens: per screen, nominated edges; new nominations with and without a two-model record; shipped rows no screen
     nominates; change of the per-point distances by latitude band; facing points gained and lost;
   - census: population; nominations; new nominations with and without a record; every shipped rescreen-gap row re-nominated
     (a failure is reported, not tuned); pairs and nominations in the 100–100.2 km band that leaves; pairs added by the
     build's polygons (section 5.1 (i) and (ii));
   - checks: at latitudes below 10° shares move by at most one sample in the edge screens; the census pairs of the record
     keep their gap within 1% where both unit outlines lie below 10° latitude.
4. **Map check** by the maintainer of a sample: facing points gained or lost, new census pairs from section 5.1 (i), and
   every new nomination with a large distance change.
5. **Campaign**: regenerate the records; completeness (every nominated edge or pair carries a two-model record) and
   attributability (every shipped water-only row is nominated); adjudicate every new nomination without a record by the
   standing design (blind GPT-5.6 Sol research in the maintainer's Codex run, Sonnet-5 adversarial judgment on the
   maintainer's "launch", maintainer map rulings on disagreements); one data pull request; the documents once.

## 8. Rows

As ba1 §10: a shipped water-only row that no screen nominates leaves the water-only set and is recorded in
`docs/FUTURE_WORK.md` §4 (not a finding that the border is land); a row nominated again after leaving is re-adjudicated;
every new nomination without a two-model record goes through the two-pass before any row is added.

## 9. Documents (once, after the campaign)

The manuscript (the census sentence: pairs within 100 km of each other on the ground, the population count; the lake
sentence rests on the pair check of the five partly unassigned lakes; Table 2 counts), METHODS, PROVENANCE, REPRODUCING,
FUTURE_WORK, CHANGELOG, the drafts, the process notes, the supplement's status note; the manuscript guard gains the census
population and the census-reach statement.

## Change log

- 2026-09-23 draft.
- 2026-09-23 approved as drafted (section 5.1 included). Frozen.
- 2026-09-23 implementation defect (section 3), fixed before any screen run: the local frame scaled longitude by
  cos(latitude) alone, which makes a metre east equal a metre north only on a sphere; on the WGS84 ellipsoid the factor is
  cos(latitude)·N/M (prime-vertical over meridional radius of curvature: 1.0067 at the equator, 1 at the poles). Found by
  a six-edge test of the rebuilt cache code, in which one distance at 2°N read 1 m longer than before. The first
  validation run, on the spherical factor, was stopped at 1,000 of 4,344 points and is discarded.
- 2026-09-23 validation 2 (section 3) reads the ba1 cache's stored distances, which are whole metres; the comparison is
  therefore made with a tolerance of one rounding step (1 m) in place of 0.01 m.
- 2026-09-23 validation 1 (section 3), reference corrected: the exact computation projected only a feature's vertices into
  the azimuthal-equidistant frame and joined them there by straight lines, while the data's edges are straight in
  longitude/latitude; on long generalized segments (Natural Earth rivers) the two lines part by metres, and 68 of 13,372
  pairs failed, several with the rule shorter than the 'exact' value. The reference now densifies the feature (0.0001
  degree) before projecting it. The rule itself is unchanged.
- 2026-09-23 implementation defect (section 3), found by validation 2 on the rebuilt arc cache: the local frame is
  exactly isotropic only at the sample point and its scale drifts with distance, so beyond a few kilometres it can locate a
  point up to 18 m (0.07%) past the true nearest one. That met validation 1's 0.1% at 25 km, but 11 of 2,586,484 cached
  Natural Earth river distances (13–25 km; none across a ladder rung) came out longer than the ba1 helper's. Fix: beyond
  2.5 km the nearest point is located again in an azimuthal-equidistant frame centred on the sample point, where every
  distance from it is exact (the feature densified to 0.0005 degree); the 11 cases now match the exact distance within
  1 cm. Only the Natural Earth river ladder measures beyond 2.5 km. Validation 1 is re-run and the arc cache rebuilt.
- 2026-09-23 implementation defect (section 5.3), found in the census run: a valid unit polygon projected into a pair's
  azimuthal-equidistant frame can self-intersect where two of its edges nearly meet (the vertices are projected and joined
  by straight lines), and GEOS then refuses the third-unit overlay (first seen on LVA087–LVA005, a self-intersection in a
  third unit's projection). Such a pair is computed again with every projected polygon made valid (`shapely.make_valid`,
  structure method); this path runs only when GEOS refuses, so no other pair's result depends on it. The pairs so computed
  are listed in the census log.
- 2026-09-23 maintainer decision (section 5.2), on the blind measurement's finding that the shipped rescreen-gap row
  Caraș-Severin↔Bor (ROU013↔SRB001, the Danube at the Iron Gates) touches at one point on the build's polygons (Bor's
  relabelled strip ends at the Caraș-Severin–Mehedinți junction; ruling of 2026-09-22, "the overlay row stands"), so the
  census skipped it and section 8 would have removed it. Question as put: "How should the census treat Caraș-Severin↔Bor,
  the shipped Danube row that touches at one point on the build's polygons?" Answer: "Raw polygons (Recommended)". Rule: a
  pair that is not a shipped edge (or is a shipped rescreen-gap row), whose build polygons touch at a single point while
  its World Bank polygons do not touch, is measured on the World Bank polygons (`point_contacts_gd1.py` lists the pairs,
  `census_gd1.py --raw --pairs` measures them). Measured: 2 pairs of the 79 point contacts — Caraș-Severin↔Bor (gap
  316 m, nominated) and Tarija↔Jujuy (gap 549 m, not nominated, as on 2026-09-10); the example of section 5.1 (ii) is
  amended accordingly (Jujuy–Tarija stays in the population).
- 2026-09-23 maintainer decision and implementation defect (section 4), on the blind measurement's finding that the
  facing test's midpoint test was decided by floating-point rounding where the chord's midpoint lies on an outline: a
  sample on B's outline (a zero-length chord; the midpoint is the sample itself, on A's outline) and a chord running along
  an outline to a junction. Computed exactly, the test counts such a midpoint as covered (the sample is excluded); the code
  let rounding decide (in the ba1 and gd1 caches about 2,080 zero-length chords were facing, and 292 samples changed class
  between the two for this reason alone). Question as put: "Should the facing test stop letting rounding decide
  zero-length chords (and chords whose midpoint lies on an outline)?" Answer: "Fix in this campaign (Recommended)". Fix: a
  midpoint within 10⁻⁸ degree (about 1 mm) of either outline counts as on it (`border_arc.MID_TOL_DEG`; the ba1 record's
  locate keeps its test). The arc cache is rebuilt; positions and distances are unchanged (checked on eight edges).
- 2026-09-23 implementation defect (section 5.2), found by the blind measurement: the census's candidate query took
  111,320 m for a degree in every direction (widened by 1/cos(latitude)), but a degree of latitude is shorter near the
  equator (110,574 m), so the query could miss pairs up to 100 km apart that lie mostly north–south of each other with both
  units below about 6.6° latitude; four pairs of the 2026-09-10 census at 99.9–100.1 km (Kenya and Uganda, one of them
  nominated then: Buikwe–Masaka) were not candidates. The query now takes the shorter of the two degree lengths, with 1% of
  margin; the census resumes, measuring only the pairs the query adds (the result of every pair already measured does not
  depend on the query). The edge screens are not affected: every lon/lat search they make carries a margin of 1.2–1.5.
- 2026-09-23 implementation defect in the midpoint fix above (section 4), found on the cache it produced, and maintainer
  decision: computed exactly, the midpoint test excludes every zero-length chord, but a zero-length chord is not only the
  end of a stretch already in the arc. Where two outlines coincide within rounding without sharing vertices, the exact line
  misses the stretch and the facing test meets it as a run of zero-length chords (in the ba1 cache mostly facing, by
  rounding): the exact test removed 389 km of such contact from 45 arcs (9 of them shipped water-only rows; Saint-Louis–
  Matam lost 160 km, Gorgol–Saint-Louis 62 km). Question as put: "The approved midpoint fix also removed 389 km of real
  contact (outlines coinciding within rounding) from 45 arcs. Which rule should the facing test use?" Answer: "Corrected
  rule (Recommended)". Rule: a chord shorter than 1 cm is contact and stays in the arc (class facing, reason "contact"),
  unless the sample lies within 10⁻⁷ degree (the arc's own tolerance, EPS_DEG, about 1 cm) of a stretch already in the arc,
  which it would only repeat (excluded, reason "duplicate"); a longer chord keeps the midpoint test with the 1 mm
  tolerance (`border_arc.CONTACT_M`). Checked on 13 edges: positions and distances unchanged. The cache made with the
  exact test is kept as a record (arc_cache_gd1_exact-midpoint.jsonl); the arc cache is rebuilt.
- 2026-09-23 implementation defect in the contact rule above, found on the cache it produced, and maintainer decision: the
  arc removes the stretches it already holds with a margin of 10⁻⁷ degree (`border_arc.subtract`, EPS_DEG, about 1 cm), so
  the first sample of a piece that continues such a stretch lies about 1 cm from it, with a chord to B of up to about
  1 cm; both limits of the rule (contact below 1 cm, duplicate within 10⁻⁷ degree) sat on that scale, and rounding still
  decided junction samples (of the 159 runs removed by the exact test, 60 came back whole, 47 in part). Question as put:
  "The contact rule's 1 cm limit sits on the same scale as the arc's own 1 cm cut at junctions, so rounding still decides
  junction samples. Change the two tolerances?" Answer: "1 m contact, 2 cm duplicate (Recommended)". Rule: a chord shorter
  than 1 m is contact (`CONTACT_M` = `THIRD_TOL_M`, the arc's own "on an outline" tolerance); a contact sample within
  2 × 10⁻⁷ degree (about 2 cm) of a stretch already in the arc is a duplicate. Checked on 16 edges: contact stretches kept,
  junction repeats dropped, positions and distances unchanged. The cache made with the 1 cm rule is kept as a record
  (arc_cache_gd1_contact-1cm.jsonl); the arc cache is rebuilt.
