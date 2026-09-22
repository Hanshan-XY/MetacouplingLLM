# Specification: the border arc and the edge screens (campaign `ba1`)

Status: **APPROVED and FROZEN, 2026-09-21** (maintainer: "1. geodesic placement; 2. approve; 3. re-adjudication"). Nothing
below has been run against the shipped rows; the measurement of its effect follows the freeze (section 9). After the freeze
only implementation defects are corrected, and each correction is logged at the end of this file.

Maintainer decisions (2026-09-21): the full rule for facing stretches (not the exact line alone); rows that no screen
nominates leave the water-only set and are recorded in `docs/FUTURE_WORK.md` (the removal rule); `border_length_km` deferred;
the cross-type union included at the 0.80 bar; sample points at equal geodesic intervals (section 5); rows nominated again
after leaving the set are re-adjudicated, not restored on their earlier ruling (section 10).

## 1. Purpose

One definition of "the shared border" and one sampling rule for every edge screen, derived from the build's own definition of
adjacency and from numbers the project already uses. It replaces the two code lineages whose rules diverged unnoticed (the
Natural Earth caches sample the exact shared line, unmerged; the hydrography sweeps sample unit A's outline within 0.001
degree of unit B) and adds the type-combined screens and the cross-type union. All records are regenerated once; one
adjudication campaign; one data change.

Findings that motivate each element are in `analysis.txt`, `analysis_merged.txt`, `analysis_buffer_extra.txt` (this
directory) and `paper/REPO_FACTS.md` (2026-09-21 entries).

## 2. Scope

| In scope | Out of scope (unchanged) |
|---|---|
| The four single-layer edge screens (Natural Earth river ladder, Natural Earth lake ladder, HydroRIVERS, HydroLAKES) | The corridor census for non-touching pairs (`corridor_census_v2`) |
| Two type-combined screens and one cross-type union (new) | The fixed-crossing (`has_bridge`) pipeline |
| The border arc, the sampling rule, the stored records, the completeness and attributability checks | The bars and widths of the existing screens |
| The rows the new records add to or remove from the water-only set | `border_length_km` in the edge list (deferred; see section 11) |
| | The 500 m sampling step (maintainer decision 2026-09-21) |

Population: every edge of the shipped ADM1 edge list except the 24 rescreen-gap pairs (the census's) and the 13 de facto
edges (no shared arc): 8,421 edges today.

## 3. Geometry

The screens read the polygons **the build uses**: the World Bank ADM1 layer after the source-relabel
(`scripts/relabel_sliver_corridors.py`, manifest `data/sliver_corridor_relabel.csv`) and the reviewed unit merge
(`_ADM1_UNIT_MERGES`, RUS050 -> RUS024), exposed by the build script as one loader so that the screens cannot drift from
it. The Ocean Mask layer is read for the facing test (section 4.4).

Why: the screens have read the raw file. Two shipped borders that exist only after the relabel (Kajiado<->Kilimanjaro
44.7 km, Narok<->Mara 62.6 km) were therefore never screened, and RUS024<->RUS050, dissolved by the merge, appears in the
sweep records.

## 4. The border arc

For an edge A<->B (A = `ADM1_code_A` of the edge row), the arc is the set of sample points on **A's outline** where A bounds
the contact with B. Every stretch is assigned to exactly one class below; the record keeps the class of every point.

### 4.1 Exact line
Where A's outline and B's outline coincide: `boundary(A) ∩ boundary(B)`, its pieces **merged** into continuous lines
(`shapely.line_merge`; identical length, median 70 pieces -> 1). No number.

Why merge: each unmerged piece receives its own three or more sample points, so a border's share is otherwise weighted by
how finely the overlap happened to be cut (Natural Earth screens today: 4 points per km on average instead of 2).

### 4.2 Land-gap pairs
For the four pairs of `data/land_gap_overlay_pairs.csv` only, which the build adds as edges although their outlines stop
0.4-41 m short of each other: A's outline within 5e-4 degree of B. This is the sub-tolerance band the land-gap audit
used; the Natural Earth screens already apply it. No other pair gets a tolerance.

### 4.3 Overlap
Where A's outline runs **inside** B (the polygons overlap instead of sharing a line; Saint-Louis with Trarza, Brakna,
Louga and Matam): those stretches of A's outline. No number.

### 4.4 Facing stretches
A point m on A's outline that is in none of the classes above is a facing point when all of the following hold, with
nb the nearest point of B's outline to m and w = geodesic distance(m, nb):

1. **Reach:** w <= 1,000 m. This is the distance of the corridor census's short-gap presence rule, the number the project
   already uses to say two units can be water neighbours across a gap. Not a tolerance for coincidence: it defines how far
   apart two facing banks may be.
2. **Not a third unit's border:** m lies on no third unit's outline (geodesic distance > 1 m, a numerical tolerance), and
   the chord m-nb crosses no third unit. Removes tripoint spill-over (58% of what the 0.001-degree buffer adds today).
3. **Facing (reciprocity):** the nearest point of A's outline to nb lies within one sampling interval (500 m) of m.
   Two banks of a channel return to each other; A's coastline near the sea end of the border returns to the border's end
   instead. The interval is the sampling step, not a new number; it admits at most one interval of coastline per sea end.
4. **Across water or across nobody's land:** the midpoint of the chord lies inside neither A nor B (it lies in the Ocean
   Mask, which is how the World Bank layer draws tidal rivers and estuaries, or in a strip that belongs to no unit).

Why: 12% of what the buffer adds today is river channel cut out as sea between two facing banks (273 edges; HydroRIVERS runs
along 78% of it) and is real water border; 13% is coastline and 58% tripoint spill-over, which are not. The exact line alone
would silently drop the channel borders (three shipped tier-A rows among them).

Symmetry: A's outline is a proxy for a border that lies between the two banks. The blind measurement reports the share
computed on B's outline as well (section 8); the two banks lie within reach of the same water, so the difference is expected
to be small and is reported, not tuned.

### 4.5 Excluded
Everything else on A's outline: coastline, third-unit borders, stretches farther than the reach. Recorded as excluded with
its reason so the classification can be audited.

## 5. Sampling

Every continuous piece of the arc of geodesic length L receives n + 1 points, n = max(2, floor(L / 500 m) + 1), placed at
**equal geodesic intervals** along the piece (cumulative geodesic length over the piece's vertices).

Today the points are placed evenly in coordinate length (only their number comes from the geodesic length), so single
intervals vary from 403 to 545 m (5th-95th percentile) with direction and latitude, and a border's share is weighted
toward its east-west stretches. Equal geodesic intervals weight every stretch alike and make the statement "every 500 m"
true. No threshold, no data dependence. Measured on all 8,421 edges (`compare_placement.txt`): shares move by a median of
0.000-0.002; 25 edges are nominated only under the coordinate placement and 21 only under the geodesic one; no shipped row
is affected either way. Approved 2026-09-21.

The step stays 500 m (maintainer decision 2026-09-21; the aliasing of shares at the bar is a documented property).

## 6. Per-point record

For every sample point, the geodesic distance (WGS84) to the nearest feature of each layer, stored once in one cache:

| Layer | Stored to | Detail |
|---|---|---|
| Natural Earth named river centerlines | 25 km | covers the ladder to 20 km |
| Natural Earth lake polygons | 1,500 m | 0 inside a polygon |
| HydroRIVERS reaches | 500 m | at discharge tiers 0 / 10 / 100 m3/s (the creek band and the queue registry use the tiers) |
| HydroLAKES polygons | 500 m | 0 inside a polygon |

Shares are computed from this cache and **stored exact** (no rounding; the sweeps' two-decimal records made the bars
0.495 and 0.395 in practice, 20 nominations). The cache carries the point's arc class (section 4).

## 7. Screens and nomination

An edge is **nominated** when any screen reaches its bar. Every width and bar is an existing one.

| Screen | Point counts as water when | Bar | Status |
|---|---|---|---|
| Natural Earth river ladder | within 2.5 km of a named river; rungs 5 / 10 / 15 / 20 km recorded | 0.50 | existing |
| Natural Earth lake ladder | within 125 m of a lake polygon; rungs 250 / 500 / 1,000 / 1,500 m recorded | 0.40 | existing |
| HydroRIVERS | within 500 m of a reach (all tiers; >= 10 and >= 100 m3/s recorded) | 0.50 | existing |
| HydroLAKES | within 500 m of a polygon | 0.40 | existing |
| Combined river | within 2.5 km of a named Natural Earth river **or** 500 m of a HydroRIVERS reach | 0.50 | new (existing widths and bar) |
| Combined lake | within 125 m of a Natural Earth lake **or** 500 m of a HydroLAKES polygon | 0.40 | new (existing widths and bar) |
| Cross-type union | any of the four layers at its operating width | 0.80 | new; the bar is the corridor census's bar for a mixed river-and-lake water share |

The union at 0.80 nominated no additional edge in the 2026-09-21 measurement; it is included so that a border whose water is
split between a river and a lake is nominated by rule rather than left as a documented gap. A union bar of 0.50 was
considered and not adopted (a new use of the river bar; 48 candidates without a record).

## 8. Every number in this specification

| Number | Used for | Source |
|---|---|---|
| 5e-4 degree | the four land-gap pairs' band | the land-gap audit band already in the build and in the Natural Earth screens |
| 1,000 m | facing reach | the corridor census's short-gap presence rule |
| 500 m | sampling step; reciprocity tolerance | existing step |
| 1 m | "on a third unit's outline" | numerical tolerance |
| 25 km, 1,500 m, 500 m, 500 m | stored distance limits | the widest rung of each screen |
| 2.5 / 5 / 10 / 15 / 20 km; 125 / 250 / 500 / 1,000 / 1,500 m; 500 m; 500 m | widths | existing (METHODS section 6) |
| 0.50, 0.40, 0.50, 0.40 | bars | existing |
| 0.80 | union bar | the corridor census's union-share bar |

## 9. Freeze, then measure

1. **Freeze.** On approval this file is committed as is. After the freeze no number and no rule is changed to alter which
   edges are nominated. Implementation defects (a test that does not compute what section 4 says) are fixed and logged in
   this file's change log; design changes reopen the specification and are approved again.
2. **Blind measurement** on the frozen rule, reported in full before any data change:
   - reproduction: where the new rule coincides with today's (the exact line for the Natural Earth screens; the hydro shares
     on edges with no facing or overlap stretch), today's nominations must be reproduced up to the rounding of section 6;
   - per screen: nominated edges; **new** nominations (not nominated by today's records) with and without a two-model
     adjudication record; shipped water-only rows that no screen nominates;
   - arc composition: km and points per class per edge; the distribution of facing widths (does the 1,000 m reach bind);
     the A-outline vs B-outline difference on overlap and facing stretches;
   - sensitivity, reported and not acted on: nominations that change if the reach were 500 or 2,000 m, the reciprocity
     tolerance 250 or 1,000 m, the union bar 0.70 or 0.90.
3. **Map check by the maintainer** of a sample of facing stretches (largest, random, and each excluded class), the
   overlap edges, and the four land-gap pairs. This validates that the geometric tests find what a map shows. A correction
   is written as a general rule (section 4) and the specification is re-approved; no correction is a per-row decision.
4. **Campaign.** Regenerate the caches and records; run the completeness check (every nominated edge carries a two-model
   record) and the attributability check (every shipped on-border water-only row is nominated); adjudicate every new
   nomination without a record by the standing design (blind GPT-5.6 Sol research in the maintainer's Codex run, Sonnet-5
   adversarial judgment on the maintainer's "launch", maintainer map rulings on disagreements); one data PR; the
   documents once (section 12).

## 10. Rows

- A shipped water-only row that no screen nominates under the frozen rule **leaves the water-only set** (maintainer
  decision 2026-09-21, the rule applied on 2026-09-21 to three rows) and is recorded in `docs/FUTURE_WORK.md` section 4 with
  its shares; this is not a finding that the border is land. Measured on the draft rule: one row (Montevideo<->San Jose).
- A row that left the set on 2026-09-21 and is nominated again is **re-adjudicated** through the two-pass design like any
  new nomination (maintainer decision 2026-09-21: re-adjudication, not restoration on the 2026-09-14 ruling); measured on
  the draft rule: Lekoumou<->Niari and Hai Duong<->Quang Ninh. Kyegegwa<->Ssembabule stays in section 4.
- Every new nomination without a two-model record goes through the two-pass adjudication before any row is added.

## 11. Deferred (recorded in `docs/FUTURE_WORK.md`)

- `border_length_km` is the exact shared line and so understates overlap and channel borders (the four Saint-Louis borders
  by an order of magnitude; Tamaulipas<->Veracruz 29.8 km against 74 km of channel). When acted on, it should be computed
  from the arc of section 4.
- A border whose water is split between a river and a lake with a union share between 0.50 and 0.80 is not nominated.
- The 500 m step's aliasing at the bar (measured 2026-09-21: 100 m would move about 70 nominations in both directions).

## 12. Documents to update after the campaign (once)

METHODS (the arc, the sampling rule, Table of screens, counts), PROVENANCE, REPRODUCING, FUTURE_WORK, CHANGELOG; the
manuscript (the arc sentence, Table 2 gains the combined screens and the union; current counts; no history), the drafts,
the process notes, the ledger, the supplement's status note. The manuscript guard's procedure-source map points at the
shared arc module. Screen parameters live in one module that the documentation tests read.

## Change log

- 2026-09-21 draft.
- 2026-09-21 approved: section 5 geodesic placement kept; section 10 changed from restoration to re-adjudication of
  rows nominated again (maintainer decision). Frozen.
- 2026-09-22 blind measurement on the frozen rule (build_data/water_screen_rebuild/ba1_report_blind_2026-09-22.txt): 3,248 nominated; 75 new,
  61 without a record; 3 shipped rows nominated by no screen; 2 rows nominated again. Map check of the 96-row sample by the
  maintainer: "All fine"; no correction to section 4. Implementation: border_arc.py, build_arc_cache.py, run_screens_ba1.py.
- 2026-09-22 implementation defect, fixed (section 9.1): `border_arc.build_arc` recorded a facing candidate part's length as
  one interval per sample point, (n + 1) L / n instead of L, so the per-class kilometres of the facing and excluded classes
  over-counted one interval per part (A's outline: facing 6,210.6 -> 5,902.6 km, excluded 44,769.9 -> 36,269.1 km;
  `arc_km_check.py` aligned all 8,384 edges with candidate points). Points, classes, shares and nominations were unaffected;
  the research and judgment prompts had carried the over-counted facing lengths. Each point now weighs the half-intervals on
  either side of it; the cache was rebuilt with the fixed code and every other field reproduces (`compare_cache_kmfix.py`).
  The blind-measurement outputs are kept as `*_blind_2026-09-22.*`. The screen record now also carries every rung of both
  ladders and `nominated_by_ge10` (the screens that nominate when reaches below 10 m3/s are ignored), for the documents.
- 2026-09-22 campaign: the 63 pairs (61 without a record + COG006<->COG008 and VNM025<->VNM049, re-adjudicated):
  GPT-5.6 Sol research 63/63, Sonnet-5 judgment 63 agents; gate A 0 / B 2 / C 16 / D 45; maintainer rulings
  (`ba1_maintainer_rulings_2026-09-22.txt`): COG006<->COG008 water-only with no fixed crossing, KEN007<->KEN046 mixed,
  bucket C no disagreement. Section 10 applied (`execute_ba1.py`): JPN038<->JPN040, URY010<->URY016 and VNM026<->VNM049
  leave the water-only set; COG006<->COG008 returns. Record `ba1_rulings.json`.
- 2026-09-22 completeness: the completeness check had counted every row of the hybrid auto-reject ledger as closed, because
  all 288 rows are also listed in the audit queue. A ledger closure is a rule, not an adjudication, and holds only while the
  edge still satisfies the rule on the current screen record; on this record ten nominated ledger rows no longer did (one
  layer at 0.20-0.28 each) and none had a two-model record. The check now re-tests the rule (and counts a later research
  input only once its campaign's rulings are frozen); the ten went through the same two-pass as a second tranche (`ba1r_*`:
  gate A 0 / B 1 / C 3 / D 6; maintainer ruling "mixed" on LVA070<->LVA113): all ten rejected. Completeness 0 open;
  attributability 803/803 (779 on the border arc, 24 by the corridor census).
