# Future-edge-audit list — consolidated register

Pairs where an audit, adjudication, or validation-study rater questioned an
**edge's existence** (not merely its water status), plus placeholder-unit
pairs the validation study had to set aside on unit identity. Standing rule:
an edge is removed only on **World-Bank-internal geometric evidence** or the
**maintainer's official-map check** — never on an AI verdict or another
dataset's disagreement alone (independently digitized datasets prove nothing
about each other). Items are therefore *recorded* here with their evidence
and stay shipped until the maintainer decides.

Full audit executed **2026-07-18** (deterministic geometry passes + five
web-research adjudications). Status codes: CLOSED (no action needed),
RETAINED (edge confirmed genuine), DECISION (maintainer call pending).

| # | pair | question | outcome (2026-07-18) | status |
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
| 10 | RUS024<->RUS050 | "Name Unknown" unit identity (study NA) | RUS050 identified (>95% confidence) as **Kalmykia's own western salient** — Gorodovikovsky + Yashaltinsky districts (3,625 km2; district capitals geocode inside it; areas sum to ~3,515 km2; neighbour topology matches), split off and left unnamed upstream (FAO GAUL "Name Unknown", GAUL_1 2537, inherited by WB GAD). The 19.76 km RUS024<->RUS050 edge is really the internal Yashaltinsky-Priyutnensky raion line. Options: (a) keep the unit, identity documented (as done for the lake-surface placeholders); (b) reviewed merge into RUS024 — dissolves the edge, re-attributes its Rostov (187.3 km) and Stavropol (143.6 km) segments, regions 3,375 -> 3,374 (source-relabel-class change). | RESOLVED 2026-07-18: **merged** (`_ADM1_UNIT_MERGES`) |

## Method notes

- **Placeholder-identity fingerprinting** (items 2-5, 10): deterministic —
  geodesic area, representative point, point-in-polygon probes for known
  settlements/water, land-content and Natural-Earth-lake overlap, and the
  WB adjacency signature (which neighbours, over how many km). No judgment
  calls; reproducible from the pinned GeoPackage.
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
below, ruled 2026-07-18): edges 8,466 -> 8,460; regions 3,375 -> 3,374;
water-only 747 -> 750 (352/398); moderate 8,071 -> 8,062; stringent
7,719 -> 7,710; ADM0 layer unchanged.

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
