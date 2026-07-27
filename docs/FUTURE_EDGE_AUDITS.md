# Future-edge-audit list — consolidated register

Pairs where an audit, adjudication, or validation-study rater questioned an
**edge's existence** (not merely its water status), plus placeholder-unit
pairs the validation study had to set aside on unit identity. Standing rule:
an edge is removed only on **World-Bank-internal geometric evidence** or the
**maintainer's official-map check** — never on an AI verdict or another
dataset's disagreement alone (independently digitized datasets prove nothing
about each other). Items are therefore *recorded* here with their evidence
and stay shipped until the maintainer decides.

Items 1-10 were audited **2026-07-18** (deterministic geometry passes + five
web-research adjudications); items 11-14 were raised **2026-07-25** by the wu1
water-screen unification. Status codes: CLOSED (no action needed), RETAINED
(edge confirmed genuine), DECISION (maintainer call pending).

| # | pair | question [raised] | outcome | status |
|---|---|---|---|---|
| 1 | MLT009<->MLT055 | rater: "non-adjacent" (validation study) | Never an edge: WB polygons do not touch (~0.008 deg gap); it was a non-touching recovery *candidate*, correctly rejected. Rater note confirms the correct absence. | CLOSED |
| 2 | KHM000<->KHM018 | placeholder unit identity (study NA) | KHM000 **is Cambodia's Tonle Sap water-surface polygon**: 95% of its 2,551 km2 lies inside Natural Earth's Tonle Sap polygon (2,538 km2). Shipped lake flag validated by identity. | CLOSED |
| 3 | MWI002<->MWI004 | placeholder unit identity (study NA) | MWI004 proven to be **Malawi's Lake Malawi water-surface polygon** (22,766 km2, mid-lake probes inside, zero land, Likoma/Chizumulu correctly in MWI002; WB adjacency signature). Shipped lake flag validated. | CLOSED |
| 4 | MWI003<->MWI004 | placeholder unit identity (study NA) | Same identity proof; shipped lake flag validated. | CLOSED |
| 5 | MWI004<->TZA022 | placeholder unit identity (study NA) | Same identity proof; shipped lake flag validated. Note: sits on the Malawi-Tanzania lake-boundary question (shoreline vs median-line claims); the WB rendering encodes the shoreline-side reading — keep the project's neutral descriptive framing. | CLOSED (note) |
| 6 | HRV010<->HUN001 | AI audit verdict "not adjacent" (b3-b6) | **ADJACENT — edge genuine.** The Baranya/Bacs-Kiskun county line meets the state border on the *east* bank of the Danube (45.9147N, 18.8216E); the WB arc matches OSM way 72864666 (8.17 km, overland, the demarcated HU-YU line) almost exactly. The contact depends on attributing the Croatia-Serbia disputed tripoint pocket to Croatia (Croatia's 1891-cadastre claim — the same line drawn by Hungary's national spatial plan OTrT); under de facto Serbian-thalweg control the pair would not touch. The AI verdict read the de facto frame. **Edge retained; annotated dispute-contingent.** The 2026-07-02 water-flag demotion (mixed) is confirmed correct (~8.2 of 8.6 km dry demarcation). | RETAINED |
| 7 | LBR006<->LBR014 | AI audit verdict "not adjacent" (b3-b6) | **Confirmed NOT-ADJACENT.** Reality is a river-confluence **quadripoint** (Cestos-Gwen Creek junction, ~9.094W 5.829N): OSM's four county relations share exactly one node; GADM 4.1 reduces both diagonal intersections to the same point; no Liberian source lists the pair as neighbours. WB's 1.12 km segment is two straight *cardinal* legs (~640 m S, ~480 m W) bridging offset river-boundary termini — a WB-computable shape signature (rivers do not make L-shaped doglegs). Inverse of the collapsed-node artifact class (quadripoint stretched into an edge). | RESOLVED 2026-07-18: **removed** (denylisted) |
| 8 | VEN001<->VEN003 | AI audit verdict "not adjacent" (b3-b6) | **NOT-ADJACENT (moderate-high confidence).** Amazonas' constitution (Art. 59 = 1994 territorial-division law) enumerates its perimeter with **no Apure segment** (north: Bolivar from the Cano Orera mouth; west: Colombia back to Cano Orera); Bolivar's short east-bank Orinoco frontage plus Colombia separate the pair (corroborated by Bolivar and Puerto Carreno limit descriptions). WB's 2.35 km contact is a mid-river water-only seam where Bolivar's sliver frontage was dropped; worst case degenerates to a point contact. | RESOLVED 2026-07-18: **removed** (denylisted) |
| 9 | IRN013<->IRN020 | inter-rater disagreement on the water flag | **Flag UPHELD, disagreement resolved quantitatively:** 88.4% of the exact 311.14 km WB border lies within 500 m of the Seymareh/Karkheh (91.8% within 2 km); the only dry run is ~26 km near the Khuzestan tripoint (~8%) — far under the ~20% threshold. OSM corroborates (179 km of boundary ways are the tagged river way). | CLOSED |
| 10 | RUS024<->RUS050 | "Name Unknown" unit identity (study NA) [2026-07-18] | RUS050 identified (>95% confidence) as **Kalmykia's own western salient** — Gorodovikovsky + Yashaltinsky districts (3,625 km2; district capitals geocode inside it; areas sum to ~3,515 km2; neighbour topology matches), split off and left unnamed upstream (FAO GAUL "Name Unknown", GAUL_1 2537, inherited by WB GAD). The 19.76 km RUS024<->RUS050 edge is really the internal Yashaltinsky-Priyutnensky raion line. Options: (a) keep the unit, identity documented (as done for the lake-surface placeholders); (b) reviewed merge into RUS024 — dissolves the edge, re-attributes its Rostov (187.3 km) and Stavropol (143.6 km) segments, regions 3,375 -> 3,374 (source-relabel-class change). | RESOLVED 2026-07-18: **merged** (`_ADM1_UNIT_MERGES`) |
| 11 | CAN003<->CAN006 | mid-lake contact flagged thin-arc by the wu1 measurement gate [2026-07-25] | **Maintainer map ruling: NOT ADJACENT — a four-corners point on Kasba Lake** (Manitoba / Northwest Territories). Note the WB arc is **stable**, not collapsing: 0.563 km at 1e-3 / 1.232 at 5e-3 / 3.740 at 2e-2, one component throughout (ratio 0.151) — so unlike items 7-8 and the 2026-07-22 Peipus/Prut removals there is no WB-internal collapse signature. This rests on the maintainer's official-map check alone, which the standing rule permits as its top tier. Water measurement is non-discriminating here (lake@500 m = 1.00 for a genuine mid-lake border and a corner artifact alike). | RESOLVED 2026-07-25: **removed** (`_ADM1_FALSE_POSITIVE_DENYLIST`) |
| 12 | COD009<->UGA102 | mid-lake contact flagged thin-arc by the wu1 measurement gate [2026-07-25] | **Maintainer map ruling: NOT ADJACENT — a point contact on the Lake Edward boundary** (Nord-Kivu / Rukungiri). Arc likewise stable: 1.081 / 2.064 / 5.749 km, one component (ratio 0.188). Same basis and same caveat as item 11. | RESOLVED 2026-07-25: **removed** (`_ADM1_FALSE_POSITIVE_DENYLIST`) |
| 13 | TZA016<->UGA040 | mid-lake pair challenged by the wu1 re-adjudication [2026-07-25] | **Maintainer map ruling: NOT ADJACENT — diagonal non-adjacency on Lake Victoria** (Mara / Kalangala): the mid-lake median segment WB assigns to this pair belongs to a different unit pair. Not a thin arc — 14.019 / 14.906 / 18.235 km, one component (ratio 0.769) — so the instrument gave no signal at all; the ruling is purely the map check. The maintainer separately **upheld** the neighbouring TZA016<->UGA088 (Mara / Namayingo, arc 30.8 km) as a genuine water-only border, which is what distinguishes the two. | RESOLVED 2026-07-25: **removed** (`_ADM1_FALSE_POSITIVE_DENYLIST`) |
| 14 | BOL004<->PER007 | WB source artifact surfaced while measuring wu1 [2026-07-25] | **OPEN — edge existence questioned, no action taken.** WB's `PER007` polygon is named *Callao* — Lima's Pacific port — but is 4-part with centre at lat -15.25 / lon -71.02, i.e. in the Puno altiplano ~800 km from Callao, and its contact with La Paz (`BOL004`) falls at lat -16.198 **inside Lake Titicaca**. La Paz already borders the real Puno (`PER021`) natively over 761.7 km. The polygon is 4-part and the parts are ~900 km apart: the LARGEST (by area) sits at Lake Titicaca (centre lon -68.97 / lat -16.36), while the other three are on the Pacific coast at the genuine Callao (lon -77.1 to -77.2, lat -11.9 to -12.1), so this looks like an upstream mislabel of a Puno fragment rather than a genuine La Paz-Callao border. Recorded, not removed: the failure mode is *unit identity* (a source-relabel-class question), not a water verdict, so it is outside wu1's scope and needs its own identity-fingerprinting pass against the pinned GeoPackage. The pair ships unchanged. | DECISION pending |

## Method notes

- **Placeholder-identity fingerprinting** (items 2-5, 10): deterministic —
  geodesic area, representative point, point-in-polygon probes for known
  settlements/water, land-content and Natural-Earth-lake overlap, and the
  WB adjacency signature (which neighbours, over how many km). No judgment
  calls; reproducible from the pinned GeoPackage.
- **Arc-tolerance ladder** (items 11-13, 2026-07-25): the shared arc is
  re-extracted at 1e-3 / 5e-3 / 2e-2 degrees and its length and component
  count reported at each. A genuine segment stays stable and single-component;
  a collapsed-node artifact shrinks toward zero at the tightest tolerance and
  inflates at the loosest. This replaced ru1's `frac_500 >= 0.80` rule for
  lake-class contests, where that rule is **non-discriminating** — a genuine
  mid-lake border and a corner artifact both score 1.00. None of items 11-13
  showed the collapse signature, so all three are map-ruling removals.
- **Edge-existence adjudications** (items 6-8): WB contact geometry
  (shared-arc extraction) + independent web research against authoritative
  sources (state law, county statutes, boundary-treaty records, OSM/GADM
  topology as cross-checks only). Verdicts here are evidence for the
  maintainer's decision, not removals.
- Research records: agent reports of 2026-07-18 (session artifacts);
  identity computations reproducible from the pinned WB GeoPackage +
  `build_data/naturalearth/ne_10m_lakes.gpkg`.

## Decisions (maintainer, 2026-07-18) — all resolved

1. **LBR006<->LBR014** — REMOVED (added to `_ADM1_FALSE_POSITIVE_DENYLIST`).
2. **VEN001<->VEN003** — REMOVED (added to `_ADM1_FALSE_POSITIVE_DENYLIST`).
3. **RUS050** — MERGED into RUS024 via the reviewed `_ADM1_UNIT_MERGES` step
   (regions 3,375 -> 3,374; the salient's Rostov/Stavropol frontages fold
   into the existing RUS024 rows: 388.66+187.30 = 575.96 km and
   282.93+143.65 = 426.58 km).

Count effects of all resolutions (incl. the shore-contact nominations
below, ruled 2026-07-18): edges 8,467 -> 8,461; regions 3,375 -> 3,374;
water-only 747 -> 750 (352/398); moderate 8,072 -> 8,063; stringent
7,720 -> 7,711; ADM0 layer unchanged.

## Decisions (maintainer, 2026-07-25) — wu1 water-screen unification

1. **CAN003<->CAN006** — REMOVED (added to `_ADM1_FALSE_POSITIVE_DENYLIST`).
2. **COD009<->UGA102** — REMOVED (added to `_ADM1_FALSE_POSITIVE_DENYLIST`).
3. **TZA016<->UGA040** — REMOVED (added to `_ADM1_FALSE_POSITIVE_DENYLIST`).
4. **TZA016<->UGA088** — RETAINED, water-only upheld (the Lake Victoria border
   the removal in 3 is *not*).
5. **BOL004<->PER007** — recorded as item 14, no action; needs a unit-identity
   pass, not a water re-adjudication.

**These three differ in kind from items 7-8.** There the removal rested on a
WB-computable signature (a stretched quadripoint; a seam contradicted by
statute). Here the WB arc is stable across the whole tolerance ladder, so the
dataset offers no internal objection at all — the removals rest solely on the
maintainer's official-map check. The standing rule permits that (it is the top
evidence tier), and this is the first time the project has exercised it against
stable WB geometry, so it is recorded explicitly rather than folded into the
earlier precedent.

Count effects: denylist 2 -> 5 pairs; edges 8,459 -> 8,456; water-only
739 (345/394) -> 736 (345/391); base classification 301 -> 298 rows.
Unchanged: moderate 8,065, stringent 7,720, regions 3,374, ADM0 326/320/300
and its 26 roll-ups (none of the three removals was a country pair's only
crossing).

## Placeholder-unit identities (complete enumeration, 2026-07-18)

All 14 placeholder units in the edge list are now identified: the four
water-surface polygons (KHM000 Tonle Sap; MWI004 + MOZXXX Lake
Malawi/Niassa; ZWE011 Lake Kariba — 83% inside NE's Kariba polygon,
contains mid-lake, excludes shore towns), OMN009 = Oman's **Madha exclave**
(77.8 km2, contains Madha town; its UAE borders are genuine land borders),
LCA001 = Saint Lucia's interior crown-land/forest-reserve tract (65.7 km2,
no quarter name upstream; benign land unit), and the seven AIA units =
Anguilla's real districts, unnamed upstream.

## Shore-contact nominations — ruled 2026-07-18

- KHM000<->KHM004 (72.7 km) and KHM000<->KHM006 (97.3 km) — Tonle Sap
  shorelines — and ZWE011<->ZWE008 (417.8 km, Lake Kariba's southwestern
  shore): **maintainer-confirmed water-only (lake, no fixed crossing)**;
  shipped as rescreen-water rows with identity-audit provenance
  (manifest 368 -> 371; water-only 747 -> 750).
- MOZXXX<->MWI003 (the 0.11 km lake-corner contact): **maintainer ruling —
  the true contact is a POINT**, a non-edge under the rook rule (the WB arc
  collapses to a single coordinate at 34.867E 13.483S under tight
  tolerance). This reverses the pair's 2026-07 land-gap recovery: the row
  is removed from `land_gap_overlay_pairs.csv` (5 -> 4) and the edge from
  the shipped list.
