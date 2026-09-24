# Specification: the crossing screen's width on the ground (campaign `cw1`)

Status: **APPROVED and FROZEN, 2026-09-24** (maintainer: "Sure, go!"). Maintainer, 2026-09-23: "Both 0.0012 degrees and 133 m and seems
to be weird number. Do you have any suggestion about threshold to make it reasonable? I prefer the geodesic length but the
final decision is up to you." Nothing below has been run against the shipped rows except the read-only measurements of
section 2. After the freeze only implementation defects are corrected, each logged at the end.

## 1. Purpose

Layer 1 of the fixed-crossing pipeline (`bridge_unification/bu1_screen.py`, the June classifier's rule) counts an OSM bridge
way as a crossing when it intersects both units' polygons buffered by `UNIT_BUF = 0.0012` degree. The value was a default
(the first classifier used 0.0009 degree, "Default, not calibrated", `build_data/ADJACENCY_DATABASE_DEVELOPMENT.md` §5.8;
the planned calibration is not in the records). On the ground the buffer is 133 m north–south and 133·cos(latitude) m
east–west: 94 m at 45°, 67 m at 60°, 55 m on the northernmost shipped row (Torne, 65.7°). The screen is to use a width
measured on the ground, the same in every direction and at every latitude, with a stated basis.

## 2. Measurements already made (read-only, 2026-09-23; `bridge_unification/bridge_width_*`)

`bridge_width_dryrun.py` re-ran the screen's query and tag rules over all 803 shipped water-only rows with the search widened
to 1 km, and measured for every bridge way its distance on the ground to each unit's polygon (26,831 ways). Report:
`bridge_width_report.txt`.

- **Validation.** Its degree test agrees with the 2026-09-15 screen on 798 of the 802 rows that screen covered. On the other
  four (ZAF005–BWA001, ZMB104–COD014, ETH006–SSD006, BRA004–PER016) the 2026-09-15 screen reported no bridge way, but the ways
  it missed were last edited in OSM (way or nodes) between 2018 and February 2026, and on the two re-queried (ETH006–SSD006,
  BRA004–PER016) the 2026-09-15 screen's own sweep query returns them today (237 m and 284 m from a sweep point, radius
  400 m). Its row records show no failed query. The client accepts any HTTP-200 JSON answer as complete; Overpass reports a
  query that ran out of time or memory inside such an answer (a `remark`), which is the likely cause. Two of the four rows
  are bridged and were rechecked anyway (the screen disagreed with their flag); the other two (ETH006–SSD006,
  BRA004–PER016) are unbridged rows that were marked as agreeing and never rechecked.
- **What the width absorbs.** On 295 of the 406 bridged rows a bridge way meets both polygons (0 m). The width only matters
  where the World Bank line (median vertex spacing about 600 m) sits off the bank the bridge lands on. On the 95 bridged rows
  whose nearest bridge way falls short of a unit, within 1 km, the median shortfall is 94 m.
- **Sensitivity against the reviewed flags** (803 rows; any open bridge way; bridged rows found / unbridged rows called
  bridged / total disagreement): 0 m 295/18/129; 25 m 323/21/104; 50 m 335/29/100; 75 m 340/34/100; 100 m 347/44/103;
  150 m 357/57/106; 200 m 364/66/108; 250 m 376/77/107; 300 m 379/102/129; 500 m 384/133/155; the degree test 352/49/103.
  The disagreement is flat from 25 m to 250 m and rises on both sides. No mapping standard sets such a width (NMAS and NSSDA
  set no threshold; `docs/METHODS_adjacency.md` §2.2).
- **Rows each width sends back.** A row needs an adversarial recheck when its new layer-1 verdict disagrees with its shipped
  flag and it carries no recheck on record (bu1 or wc1). 50 m: 16 rows; 75 m: 12; **100 m: 6**; 150 m: 9; 200 m: 18;
  250 m: 29. Every count includes ETH006–SSD006 and, from 85 m, BRA004–PER016 (the missed ways above).
- **The polygons.** On the 14 water-only rows whose units the build's source-relabel, unit merge or validity repair changes,
  the nearest bridge way lies at the same distance from the build's (unclipped) polygons as from the screen's: no verdict
  changes.
- **The location test** (layer 4) measures with `GEOD.inv` but locates its two nearest points in plain degrees, the step
  campaign gd1 replaced in the edge screens. Re-measured with the gd1 rule (`geodist.nearest_on_ground`) on the 89 located
  rechecks on record: no outcome changes (largest shift: ROU008–ROU039, 586 → 475 m from the farther unit, located either
  way). No located outcome on record depends on the arc: all 89 lie within 750 m of both units or fail both tests.

## 3. The rule

1. **Width.** A way counts as lying in both units when it comes within **100 m** of each unit's polygon, measured on the
   ground: the larger of its two distances is at most 100 m. Each distance is measured in an azimuthal-equidistant frame
   centred on the way's midpoint (the frame the corridor census measures each pair in), with the way and the polygon
   densified to 0.0005 degree first; 0 when the way meets the polygon. The same width applies to the tunnel, dam-top,
   dyke and embankment ways the screen records for bridged rows.
   *Basis:* a round value inside the flat range of section 2 (25–250 m), close to the median shortfall of the bridged rows
   (94 m), and the round value that sends the fewest rows back (6). It lies inside the range the degree buffer spanned on
   the ground across the shipped rows (55–134 m), which is why so few verdicts move.
2. **Overpass answers.** An answer counts only if it is HTTP-200 JSON with an `elements` list and no `remark`; anything
   else is a failed query (logged with its remark), retried with mirror rotation as now. A row with any failed query is
   recorded as failed and re-run until every query of the row has answered.
3. **Search.** The screen's own border arc (unchanged) locates the search: its bounding box padded by 1 km on the ground,
   or, when that box exceeds 0.5 square degrees, an around-sweep of radius 1,300 m (1 km plus half the sample spacing) at
   points every 0.005 degree along every part of the arc: every way within 1 km of the arc is returned. This is the dry
   run's search, validated in section 2.
4. **Record.** Every way returned within 1 km of both units: id, tags, class (`bridge_way_class_check.way_class`), whether
   it passes the tag rule, its two distances, and its geometry (WKT), so that a later question can be answered from the
   record without a new query.
5. **Location test.** The two distances (to the shared arc, to the farther unit) are located and measured by the gd1 rule
   (`geodist.nearest_on_ground`). Thresholds unchanged: within 500 m of the shared arc or within 750 m of both units;
   750 m – 3 km flagged as a bank-line gap for maintainer judgment; farther, a wrong-unit citation.

## 4. Unchanged

The query tag set (`way[highway][bridge]`, `way[railway][bridge]`, `man_made=bridge`), the tag rule (construction and
proposed dropped; the class of a way is not tested at layer 1), the polygons (raw World Bank ADM1, validity-repaired), the
border arc's construction and the nearest-approach corridor for non-touching pairs, the location thresholds, the recheck
question and response schema (bu1, verbatim), the gate buckets, and the convention of what counts as a crossing.

## 5. Procedure

1. Run the screen over all 803 rows (two workers; about 45 minutes). **Validation:** compare with the dry run of 2026-09-23
   way by way; every difference is explained (an OSM edit between the two dates, or a query the dry run's client accepted
   incomplete) before the record is used.
2. **Recheck set:** rows whose new verdict disagrees with the shipped flag, or bridged rows with only a tunnel/dam-top way,
   that carry no adversarial recheck on record. From the dry run: ETH006–SSD006 (Gambela ↔ Pibor, Akobo; unbridged; a
   tertiary-road bridge way meets both polygons), BRA004–PER016 (Amazonas ↔ Loreto; unbridged; a footway 85 m from Loreto),
   NLD002–NLD008 (Flevoland ↔ Noord-Holland; bridged; nearest bridge way 103 m), LVA075–LVA080 (Pārgaujas ↔ Priekuļu;
   bridged; 113 m), UGA083–UGA107 (Mubende ↔ Ssembabule; bridged; 104 m), UGA007–UGA039 (Amuria ↔ Kaberamaido; bridged;
   129 m), plus any row the re-run adds.
3. **Layer 3:** GPT-5.6 Sol adversarial recheck (the maintainer runs it in the Codex app; bu1 question and schema verbatim,
   with a row-specific conflict line). **Layer 4:** the location test of section 3.5. **Gate:** the bu1 buckets.
   **Maintainer map rulings** on proposed flag changes only.
4. The location test re-run on every recheck on record with the gd1 rule; any change in outcome goes to the gate.
5. Documentation, once: `paper/METACOUPLINGLLM_EMS_MANUSCRIPT_V3.md` (the layer-1 sentence and the way-class counts that
   follow it, recomputed on the new record), `docs/METHODS_adjacency.md` (`has_bridge` paragraph; campaign paragraph),
   `docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md` (status note), `src/metacouplingllm/data/PROVENANCE.md`,
   `docs/REPRODUCING.md`, `CHANGELOG.md`; the drafts and the supplement. One pull request.

## 6. Change log

- **2026-09-24, section 5.2 (the recheck set), clarified.** The way-class rule of 2026-09-20 (campaign wc1, the manuscript's
  account of the classes of the ways behind the bridged flags) applies to the new record as it did to the old: a bridged row
  whose bridge ways within 100 m of both units include no road or rail way goes to the recheck, unless a road or rail
  structure on record carries it (`bridge_unification/wc1_rulings.json`, `reconciled_on_record`) or a recheck on record
  answers it. On the dry run this adds one row, Panamá ↔ Panamá Oeste (PAN011–PAN012, Panama Canal): within 100 m only the
  outline of the Puente de las Américas (`man_made=bridge`) and two footways reach both units; the bridge's trunk-road way
  lies 117 m from one unit; the row's record names the bridge only through the screen. Implemented in `make_recheck_cw1.py`.
