# Specification: the corridor census on the water screens' widths, measured by length, with a stretch rule, pairs that meet at a point included (campaign `cc1`)

Status: **APPROVED, 2026-09-27** (maintainer: "Yes, approve as drafted"). Frozen; later corrections are implementation
defects only, logged in section 15.

## 1. Origin

The maintainer asked for the census's steps and thresholds to be reviewed, and for the census to be run again if any is
unreasonable (section 13). The review traced every step to `build_data/geodesic_distances/census_gd1.py` and measured it
on the census record (`census_gd1.csv`: 23,407 pairs, 242 nominated, 24 accepted). It found four defects, and the
maintainer found a fifth:

1. **Pairs that meet at a point are never tested.** On the build's polygons 85 pairs touch without being edges: 80 meet
   only at points and 5 share a line (the contacts of `data/denylist_pairs.csv`). The gd1 listing records one of the 80,
   Gorgol↔Saint-Louis, as an overlap; its area is zero.
   - The census skips any pair at distance 0, and rook contiguity keeps a point contact out of the edge list, so no
     instrument examines these pairs. The exception is the rule of 2026-09-23, which measures two of them
     (Caraș-Severin↔Bor and Tarija↔Jujuy) on the World Bank polygons.
   - Rook contiguity rejects the contact itself, usually a corner where four units meet. It says nothing about water
     between the two units elsewhere.
   - Measured with the census's own code at distance 0 (`touch_pairs.py`), 42 of the other 78 would be nominated: 30 have
     a third unit across at least 95 percent of their transects, and 12 do not.
2. **The caps make the stated spacing untrue for most pairs.** Samples were capped at 3,000 per pair and transect starts
   at 200 per side, both thinned evenly beyond that.
   - The sample cap binds for 95 of the 242 nominated pairs, and by estimate for 16,017 of the 16,065 pairs with a lake
     near their transects.
   - The shares were not distorted. On the census's own transects, shares computed from lengths differ from the record by at most 0.008
     on 30 pairs with gaps over 5 km. On 57 short-gap pairs with water they differ by −0.015 to +0.027, because samples
     count both ends of every transect (the banks) at full weight (`pilot_exact.py`, 93 pairs).
   - The manuscript's "at most 250 m … at most 100 m apart" nonetheless holds only for short corridors, and the presence
     rule's "any sample" was thinned too.
3. **The combined share is not the union.** The census adds the river share to the larger of the two lake layers' shares
   and caps the sum at 1. The sum counts water twice where a reach runs through a lake (Entre Ríos↔Artigas 1.000 where the
   union itself is 0.959), and the larger-of misses water the two lake layers draw in different places. The edge screens
   count a point as water when any layer reaches it (`geodesic_distances/run_screens_gd1.py`, line 65).
4. **HydroLAKES is limited to lakes of at least 0.25 km² without a stated reason.** That keeps 689,234 of its 1,427,688
   polygons; the edge screens use all of them.
5. **The facing stretch is set by the pair's single narrowest point.** This defect was found by the maintainer: take two
   units that touch at a corner at one end and face each other across a 2 km river farther along. Their gap is 0, so
   the facing band is 750 m (1 km under this specification), and the river is never measured. The same happens whenever
   the narrowest point is a land
   near-touch, and covered transects inside the band can be outweighed by uncovered ones elsewhere in it. Measured on the
   1,622 pairs with gaps of 5 km or less or a point contact, with crossings of at most 5 km as specified in section 3
   (`facing_band_check.py --fixed-reach`; "covered" as in section 4):
   - 18 pairs have a covered stretch longer than 1 km outside their band and are nominated by no other rule, among them
     two Icelandic point contacts covered along about 2–4 km;
   - a further 78 have such a stretch inside the band, but uncovered transects elsewhere in it keep their share below
     0.80;
   - each of the 23 accepted pairs with gaps of 5 km or less has a covered stretch of 3.1 km or more.
   
   The stretch rule (section 5) nominates both kinds.

**The maintainer's questions on the first draft** (section 13) were each measured before this rewrite:

- **Transect spacing, 250 m against 500 m** (`review_questions_cc1.py`, 1,890 pairs):
  - one nomination differs (Ruse↔Călăraşi, 0.802 against 0.797, at the bar);
  - shares differ by a median of 0.002 (95 percent within 0.017);
  - 250 m gives short corridors 60–70 percent more transects (point contacts a median of 13 against 8; gaps up to 1 km,
    24 against 14).
  
  The maintainer had no preference; the spacing stays 250 m.
- **Natural Earth rivers:** the census has never used them.
  - Named centerlines at the edge screens' 2.5 km nominate Usulután↔La Paz on the lower Lempa: a gap of 2.7 km, 0.20
    within 500 m of HydroRIVERS, a river below 1,000 m³/s, and never adjudicated.
  - No rule of the first draft could reach it.
- **The water screens' thresholds in the census** ("can we just use same threshold in water screens?"). Measured on the
  census corridors (`edge_thresholds_on_census.py`, `base_width_union_on_census.py`, `far_lakes_on_census.py`):
  - **Their widths and their union's bar transfer.** With the census's presence rule, on the gaps up to 5 km and the point
    contacts, they nominate 210 pairs (55 never adjudicated) against the first draft's 208 (44). They nominate none of 600
    sampled pairs farther apart, and every accepted pair.
  - **Their ladder rungs do not transfer.** Natural Earth rivers at 5–20 km and lakes at 250–1,500 m would nominate 32 of
    129 sampled pairs more than 5 km apart, about 5,400 over the 21,863 such pairs. Their median gap is 58 km, and almost
    all come through the 10–20 km river rungs.
    - On an edge a wide rung only sends an existing border to adjudication.
    - In the census the transects already span the gap, so a wide band counts any land near a river as covered, and
      each nomination proposes a new neighbour.
  - **Their single-layer bars do not transfer.** Rivers at 0.50 and lakes at 0.40 would add 84 more short-gap pairs to
    adjudicate, gaps half of whose transect length lies outside the buffered layers. On an edge those bars ask whether
    half of an existing border lies near mapped water. A gap between units that do not touch is screened by the union's
    0.80.

- **The stretch rule's reach** (`lake_stretch_reach_check.py`): taking lake crossings out to the census reach (100 km)
  adds nothing.
  - The 10 point contacts facing a lake beyond 5 km are nominated by the other rules already.
  - Among 300 sampled pairs more than 5 km apart, no lake stretch lies beyond the facing band.
  - The 6 sampled pairs with a lake stretch inside the band are, by their flags, diagonals across Lake Victoria, Lake
    Kyoga and the Caspian, with another unit across 81–100 percent of their transects. That is about 2 percent of such
    pairs, so the stretch rule stays with gaps of 5 km or less.
  - Taking the crossings out to the facing band where that is wider than 5 km (gaps of 2.5–5 km) would add 30 pairs.
    All 30 have another unit across at least 95 percent of their transects, and their crossings are mostly 5–8.5 km
    long. So no crossing of the stretch rule is longer than 5 km (section 3).

The maintainer asked for the rewrite ("Yes, please rewrite the spec that way") and the stretch rule ("Sure, please
revise the specification"). The census script cannot be re-run as it stands, because it reads
`rescreen_gap_overlay_pairs.csv`, which the stage campaign (`st1`) retired. The 24 shipped non-touching rows are now the
`adds_edge` rows of `data/water_classification_pairs.csv`.

## 2. Population

- **Polygons:** the build's (the ba1 loader: validity-repaired, clipped to land by the Ocean Mask, source-relabelled, RUS050
  merged), from `water_screen_rebuild/arc_cache_geometry.pkl`, as gd1. Every pair is measured on them. The rule of
  2026-09-23, which measured pairs the build closes to a point on the World Bank polygons, is retired: such a pair is a
  point contact measured like any other.
- **Pairs:** every pair whose gap on the ground is at most 100 km and which is not a shipped edge. This includes pairs
  that meet only at points (gap 0). The 24 shipped rows that add an edge are included as the rediscovery check.
- **Excluded:** the five contacts of `data/denylist_pairs.csv`, which share a line and which the maintainer ruled not to
  be borders. Any other line-sharing pair that is not an edge stops the run; none is expected.
- **Candidates:** from gd1's query, widened by 1/cos(latitude). Expected population: 23,485 (23,407 + the 78 point
  contacts not measured today).

## 3. Per pair: gap, facing stretch, transects

Computed in the pair's azimuthal-equidistant frame, as gd1 (section 5.3 of its specification):

- **Gap:** the geodesic nearest approach, located in the frame centred on it (re-centred once) and measured with
  `GEOD.inv`. It is 0 for a pair that meets at a point; the frame is then centred on the contact point the search returns.
- **Facing stretch:** each unit's outline within facing_m of the other unit, facing_m = min(max(2·gap, gap + 1 km),
  100 km). At gap 0 it is 1 km.
  - The 1 km is twice the HydroRIVERS width, so the band switches from "gap + 1 km" to "2 × gap" exactly at the 1 km
    presence gap (section 5). Below it, the band reaches one full HydroRIVERS band-width beyond the gap.
  - It replaces 750 m, which the census of 2026-09-10 introduced without a recorded reason ("Yes, replace 750 m with
    1 km"; measured in `threshold_sensitivity_check.py`: 4 more nominations, 3 of them never adjudicated, and no
    accepted pair affected).
  - The facing stretch serves the share and presence rules. The stretch rule has its own crossings (below).
- **Transect starts:** along each facing stretch, per part, n + 1 points at equal intervals L/n with
  n = max(2, floor(L / 250 m) + 1), so at most 250 m apart. There is no cap.
- **Transects:** from each start to the nearest point of the other unit's outline. Zero-length transects are dropped, and a
  transect is kept when its midpoint lies inside neither unit (as now). If none is kept, the nearest-approach chord is the
  only transect (as now); at gap 0 the chord has no length, so the pair has no transect and is recorded unnominated.
- **Crossings for the stretch rule** (gaps of 5 km or less, point contacts included): the same construction, with starts
  every 250 m or less and no cap, along the part of each outline that lies within **D = 5 km** of the other unit, kept in
  their order along each outline part.
  - 5 km is a maximum crossing distance: every crossing is at most 5 km long, the river limit, for every gap, whatever
    the facing band.
  - The stretch rule looks wherever the two units come within 5 km of each other, not only near their narrowest point.
  - The facing band above serves the share and presence rules only.
- **Flags, never rejections** (definitions unchanged):
  - the third-unit share: the share of transects with more than 10 percent of their length in a third unit;
  - the third-unit midpoint share;
  - the wedge ratio, the median transect length over the gap (left blank at gap 0).
  
  As a filter, the third-unit share would reject 6 of the 24 accepted pairs.

## 4. Coverage by the water screens' buffered layers, measured by length

**What the score is.** The buffered hydrographic layers give a water-proximity indicator: they show where a gap lies
close to mapped water. A pair that meets a threshold is nominated for review. The score does not measure the area
actually covered by water, and it does not establish that two units are neighbours. A river drawn as a line and
buffered by 2.5 km covers a band 5 km wide, so a transect can be fully covered while crossing a channel 50 m wide.
Adjacency is decided afterwards, by the two-model adjudication and the maintainer's map ruling (section 9).

A point of a transect is **covered** when it lies within the edge screens' base width of any of their four layers. The
widths are read from `water_screen_rebuild/border_arc.py`, so the census cannot drift from the screens:

| Layer | Width | Constant |
|---|---|---|
| Named Natural Earth river centerlines (`ne_10m_rivers`, non-empty name, as the edge screens) | 2.5 km | `NE_RIVER_RUNGS_M[0]` |
| Natural Earth lakes (`ne_10m_lakes`) | 125 m | `NE_LAKE_RUNGS_M[0]` |
| HydroRIVERS v1.0, every reach | 500 m | `HR_WIDTH_M` |
| HydroLAKES v1.0, every polygon (1,427,688) | 500 m | `HL_WIDTH_M` |

- **The widths are screening tolerances, not error bounds.**
  - HydroRIVERS' 500 m matches the grid it was derived from: 15 arc-seconds, "approximately 500 m at the equator" (its
    technical documentation), and narrower east–west away from it. The documentation states no positional accuracy, so
    500 m does not guarantee that a mapped river lies within 500 m of the real one.
  - Natural Earth's 2.5 km follows from the map scale: half the 5 km that the National Map Accuracy Standards allow at
    1:10,000,000. Natural Earth is not certified against that standard.
  - The lake widths are the edge screens' base widths.
  
  All four are the widths the edge screens have used and adjudicated on, and nomination leads only to review.
- **River limit:** the two river layers count only for gaps of 5 km or less, twice the widest width (2.5 km). A bank-line
  river gap is the channel's width, and the accepted river gaps run 0.17–1.74 km.
- **Masks:** built in the pair's frame from the features near its transects, each clipped to the transects' envelope
  widened by the width. Each feature is densified to 0.0005° before it is projected, so its edges stay straight in
  longitude/latitude as in the data (gd1's rule), then buffered by its width.
- **Mask:** W is the union of the four buffered layers.
- **Share (coverage):** Σ length(t ∩ W) / Σ length(t) over the pair's transects, the share of the transects' total length
  that lies in W. It is calculated directly from lengths: no points are sampled along a transect, and there is no sample
  spacing and no cap. The transects themselves remain discrete, every 250 m or less along the outlines (section 3).
- **Covered crossing** (for the stretch rule): a crossing is covered when at least 0.80 of its own length lies in W, the
  share rule's bar applied to one crossing.
- **Approximations.** The shares are measured against constructed shapes, and three numerical approximations remain:
  1. **Polygonal buffers.** The round parts of a buffer (around line ends, bends and polygon corners) are drawn with q
     straight segments per quarter circle. Between vertices they lie up to R(1 − cos(π/4q)) inside the true circle of
     radius R. q is set per width so that this is at most 1 m:
     - q = 8 for 125 m (0.60 m);
     - q = 16 for 500 m (0.60 m);
     - q = 32 for 2.5 km (0.75 m).
     
     With q = 8 throughout, the pilot's setting, the 2.5 km buffer's round parts lie up to 12 m inside. Raising q from 8
     to 32 on the 1,622 pairs with gaps of 5 km or less or a point contact changed no nomination; shares changed by at
     most 0.0055, and by at most 0.0009 within 0.05 of the bar (`buffer_precision_check.py`).
  2. **Projection.** The frame is azimuthal equidistant, centred on the pair's nearest approach. It keeps distances from
     its centre exact, but between other points its scale error is about (d/R_E)²/6 at distance d from the centre:
     0.01 percent at 150 km, 0.07 percent at 400 km. On those 1,622 pairs no transect end lies more than 94 km from its
     centre, under 0.004 percent. The dry run reports the largest distance over all pairs.
  3. **Discrete transects.** Along the outlines the transects are samples, every 250 m or less. Halving their density
     changed 1 nomination in 1,890 (section 1).
  
  The layers' own positional errors are not errors of the calculation. The widths are tolerances chosen with those errors
  in mind, not bounds on them.

## 5. Nomination

A pair is nominated when:

1. **Share rule:** its share is at least **0.80**, the bar of the edge screens' cross-type union (`UNION_BAR`); or
2. **Presence rule:** its gap is **1,000 m** or less and some transect passes within 500 m of a HydroRIVERS reach.
   1,000 m is twice the HydroRIVERS width: within it, where the reach happens to be drawn decides the share; or
3. **Stretch rule:** its gap is **5 km** or less (point contacts included), and consecutive covered crossings along one
   outline span more than **1 km**. The crossings are those of section 3, each at most 5 km long. The span is measured
   along the outline from the first to the last start of the run: (m − 1) × the part's interval for m crossings.
   - **Why:** the facing stretch is set by the narrowest point (section 1, defect 5). The stretch rule asks where the two
     units face each other within the buffered layers, wherever that is. It finds covered crossings beyond a corner or a
     land near-touch, and covered crossings that uncovered ones elsewhere in the facing stretch outweigh.
   - **1 km is a screening convention**, set at twice the HydroRIVERS width, the same length as the presence gap. It is
     not a geometric bound, and one stream can cover a longer stretch. The idealized geometry below illustrates this;
     it is an explanation, not part of the nomination algorithm.
     - **Conditions:** two parallel straight outlines G apart; crossings perpendicular to them; one straight stream,
       much longer than the area, at angle a from the crossing direction; its buffer of full width w (1,000 m for
       HydroRIVERS); no other covered feature.
     - **When a crossing can be covered:** along a crossing the buffer extends w / sin a (at a = 0 it holds the whole
       crossing or none of it). A crossing can reach 80 percent coverage only if w / sin a ≥ 0.8 G. Otherwise no
       crossing is covered and the span is zero. A 3 km gap, for example, has no covered crossing at any angle but 0°.
     - **Span when it can:** w / cos a − 0.6 G tan a along the outline. At right angles that is w, about 1 km. It grows
       with the angle, and without limit as the stream turns parallel to the outlines (a river along the gap), but only
       when w ≥ 0.8 G, that is for gaps up to w / 0.8. That limit is 1.25 km for a HydroRIVERS reach and 6.25 km for a
       named Natural Earth river (w = 5 km), so every gap the stretch rule examines.
     - **Example:** in a 1.1 km gap a straight stream covers 1.7 km at 80° and 3.7 km at 85° (`oblique_stream_check.py`).
     
     So 1 km does not separate streams crossing the gap from rivers along it; adjudication does.
   - Each of the 23 accepted pairs with gaps of 5 km or less has a covered stretch of 3.1 km or more.

These three rules are the whole nomination. Everything else the census used to nominate by is dropped:

- **The census's own wide variant** (lakes at 1,500 m; reaches of at least 1,000 m³/s at 2,500 m) and its judgment value
  1,000 m³/s:
  - all 24 accepted pairs are nominated without it (Équateur↔Cuvette through the Natural Earth river width);
  - its lake part only ever nominated pairs that were rejected (34);
  - with every HydroLAKES polygon, its 1,500 m band nominates lake-district pairs 40–50 km apart (2 of 600 sampled pairs
    more than 5 km apart, one of them new).
- **The census's own lake width** (125 m for HydroLAKES as well): replaced by the screens' 500 m.
- **The presence rule's second clause** (a lake share of 0.80 on its own): the share rule covers it.

## 6. Recorded for sensitivity, not nominating

- **Per-layer shares** (Natural Earth rivers 2.5 km, Natural Earth lakes 125 m, HydroRIVERS 500 m, HydroLAKES 500 m) and the
  combined river and combined lake shares. These give layer attribution, the values the prompts state, and the
  single-layer screens' counts from the record.
- **Bar:** nominations at every bar from 0.55 to 0.90 in steps of 0.05.
- **Facing width:** for every pair with a share of at least 0.50, or nominated by the presence rule, the share with facing_m
  halved and doubled.
- **The dropped wide variant,** for gaps of 5 km or less: its share, to show what dropping it changes.
- **HydroLAKES at 125 m,** for every pair with a share of at least 0.50: the census's former lake width.
- **Buffer precision,** for every pair near a decision: those whose share lies within 0.05 of 0.80, and those whose
  longest covered stretch lies within 500 m of 1 km. Their rules are computed again with q doubled for every width, and any
  change of nomination is reported.
- **Covered stretches,** for every pair of gaps of 5 km or less:
  - the longest covered stretch and the part of it outside the facing stretch;
  - nominations with the stretch set at 2 km instead of 1 km (on the review's measurement: 47 added pairs instead of 96,
    25 never adjudicated instead of 59).

## 7. Implementation checks, before the dry run's nominations are read

1. **Old definition reproduced.** On the census's own transects (gd1's construction, caps included), the length method with
   the old definition (larger lake layer plus river, capped at 1; HydroLAKES of at least 0.25 km² at 125 m; the wide
   variant) reproduces `census_gd1.csv` over its whole population. Differences are reported in full; any pair differing
   by more than 0.05 is traced to its transects and explained before the run proceeds (pilot maximum 0.027).
2. **Transect starts reproduced.** Without the caps, the transect starts equal the census's wherever the caps did not bind
   (same count, same points within 1 cm).
3. **Gap reproduced.** The gap equals `census_gd1.csv`'s, to the metre, for every pair it measured on the build's polygons.
4. **Synthetic cases.** The masks and shares are checked on constructed cases:
   - a straight river crossed by a transect at a known angle;
   - a lake polygon;
   - a named Natural Earth centerline;
   - a pair that meets at a point;
   - a reach running through a lake, counted once;
   - the maintainer's example: two units that touch at a corner and face each other across a 2 km river farther along
     (nominated by the stretch rule, not by the share rule);
   - a straight stream crossing a gap under the idealized conditions of section 5: a 1.1 km gap at 0°, 45°, 80° and 85°,
     and a 3 km gap at 0°, 30° and 60°. The covered stretch must be zero where w / sin a < 0.8 G (the 3 km gap at 30°
     and 60°). Otherwise it must lie within two transect spacings below w / cos a − 0.6 G tan a. The references from
     `oblique_stream_check.py` are 1.0, 0.75, 2.0 and 3.9 km for the 1.1 km gap, and 1.0 km, 0 and 0 for the 3 km gap;
   - a pair 4 km apart whose only covered crossings are 6 km long: the stretch rule does not examine them (crossings are
     at most 5 km long).

## 8. The dry run, reported in full before any adjudication

- **Run totals:** the population (expected 23,485), pairs beyond the reach, errors (none may remain).
- **Nominations** by rule.
- **Rediscovery:** the 24 shipped non-touching rows nominated. A failure is reported, not tuned.
- **Previous nominations:** each of the 242 nominations of the gd1 record is re-nominated, or its loss explained. The 242
  are those of `census_gd1.csv` and `census_gd1_closed_raw.csv`, less Jerusalem↔Ramallah, an edge since 2026-09-25.
  Expected from the measurements: 23 lost (15 more than 5 km apart, 8 closer), all adjudicated, none accepted. The 8
  closer ones were nominated only by the dropped wide variant: three Singapore pairs, Amuru↔Moyo, Kaliro↔Serere,
  Humacao↔San Juan, Burgenland↔Wien and Bucureşti↔Călăraşi.
- **New nominations,** each with its gap, per-layer shares, flags, and whether it carries a two-model record.
- **Sensitivity:** the tables of section 6.
- **Maps** of the new nominations for the maintainer, in the style of `map_census_gd1.py`.

## 9. Campaign

1. **Scope:** every new nomination without a two-model record goes through the standing design. A nomination that already
   carries a verdict keeps it.
2. **Research:** GPT-5.6 Sol in the maintainer's Codex run, fresh and blind. The prompt is the standing non-touching prompt
   of `corridor_census_v2/make_handoff_nt2.py`, verbatim by AST, except its geometric context:
   - it states the share of the transects' length within each layer's width and all four together, and the longest
     covered stretch, along which the two outlines face each other within the buffered layers. The prompt names these
     as proximity to mapped water, not as water coverage;
   - for a pair that meets at a point, its two sentences on the geometry say that the two polygons meet only at a single
     point and do not otherwise touch.
   
   The defined-line question is unchanged: a meeting at a point or corner is not adjacency. The two sentences live in
   `water_screen_rebuild/standing_prompts.py`, as the edge screens' nomination sentence does.
3. **Judgment:** Claude Sonnet 5, adversarial, on the maintainer's "launch". The judge is the standing census judge of
   `corridor_census_v2/make_judge_nt4.py`, verbatim, with the same geometric context.
4. **Gate:**
   - A: both passes water-only, ruling needed;
   - B: the passes disagree, ruling needed;
   - C: both not water-only but flagged, for a glance;
   - D: both not water-only, closed.
   
   Only the maintainer's map ruling accepts a pair; the two models alone never do. The rulings are frozen.
5. **An accepted pair** adds an edge with its water row (Stage 3, `adds_edge`). Its verification tier is A, human
   map-verified (PROVENANCE, "Verification tiers"), because its acceptance is the maintainer's map ruling; the 24 shipped
   non-touching rows are tier A for the same reason. Tier B cannot be given: it is pinned to the validation study's frame
   (`tests/test_apply_overlays.py::test_tier_b_is_exactly_the_validation_study_frame`). The pair's crossing flag comes
   from the standing four-layer pipeline, and the ADM0 roll-up is computed again.

## 10. Rows, records, documents

- **Rows:**
  - Nothing ships except through section 9.5.
  - A shipped non-touching row that the new census does not nominate is reported to the maintainer first; no row changes
    without the maintainer's ruling.
- **Records:** `census_cc1.csv` becomes the census record. The completeness check
  (`rejection_unification/completeness_check.py`), the attributability check (`hydro_fold/attributability_check.py`) and
  the manuscript guard (`paper/verify_v3_section3.py`) read it; `census_gd1*` stays as the gd1 record.
- **Documents,** once after the campaign:
  - the manuscript's census paragraph:
    - the population, with the point contacts;
    - coverage by the screens' buffered layers, calculated from transect lengths, with its approximations, described as
      a proximity indicator that nominates pairs for review;
    - the bar of the screens' union, and the presence and stretch rules;
    - the counts and the bar sentence;
  - the explanations of the edge screens' widths, wherever they call the widths accuracies: the manuscript's screen
    paragraph, METHODS, REPRODUCING, the drafts and the process notes. They are restated as screening tolerances:
    - HydroRIVERS' 500 m matches its 15 arc-second grid and is not a stated positional accuracy;
    - Natural Earth's 2.5 km follows from the map scale;
  - its guard;
  - METHODS §8 (its current description and a `cc1` record), REPRODUCING, CHANGELOG;
  - the drafts EN/ZH and the Chinese checklist, the process notes, the reproduction ledger, the fact sheet;
  - a status note in the supplement.

## 11. Every number in this specification

| Number | Used for | Source |
|---|---|---|
| 100 km | census reach | maintainer decision 2026-09-23; the five large lakes the World Bank layer leaves partly unassigned are at most 60 km wide |
| 2·gap, gap + 1 km | facing width | 1 km = twice the HydroRIVERS width, so the band switches regime at the 1 km presence gap. It replaces 750 m, which had no recorded reason. Measured, the added constant at 0 / 375 m / 750 m / 1 km / 1.5 km / 5 km gives 18 fewer, 12 fewer, the baseline, 4 more, 20 more and 52 more nominations, and no accepted pair is affected. Halving and doubling are recorded (section 6) |
| 250 m | transect spacing | a sampling density along the outlines, half the 500 m screening width; it does not make the data more precise than their sources. Against 500 m, one nomination in 1,890 differs (maintainer: no preference) |
| 2.5 km | Natural Earth river width | the edge screens' base width, a screening tolerance following the map scale: half the 5 km the National Map Accuracy Standards allow at 1:10,000,000 (Natural Earth is not certified against it) |
| 125 m | Natural Earth lake width | the edge screens' base width, a screening tolerance |
| 500 m | HydroRIVERS width | the edge screens' width, a screening tolerance matching the source grid (15 arc-seconds, about 500 m at the equator); not a positional-error bound |
| 500 m | HydroLAKES width | the edge screens' width, a screening tolerance |
| 0.80 | share bar | the edge screens' cross-type union bar, a screening convention: a pair above it is nominated for review; the share is buffer coverage, not water coverage or a probability of adjacency. Bar sensitivity re-measured in the dry run |
| 5 km | river limit; the stretch rule's gap limit and its maximum crossing distance | twice the widest width (2.5 km): the widest gap one river line of the most generous layer can cover completely, so no river layer is cut off early; accepted river gaps 0.17–1.74 km; crossings beyond it add only diagonals (section 1). Measured: 4 km gives 29 fewer nominations, 6.25 km gives 35 more (31 never adjudicated); no accepted pair is affected |
| 1,000 m | presence gap | twice the HydroRIVERS width: within it, a HydroRIVERS line in the gap covers at least half of every crossing. It is the finest river layer's scale, because the presence rule tests HydroRIVERS. Measured: 500 m gives 37 fewer, 1.5 km gives 53 more (50 never adjudicated); 5 km gives 847 more, since 1,140 of the 1,622 pairs within 5 km have a crossing within 500 m of some reach; no accepted pair is affected |
| 0.80 | covered crossing | the share bar, applied to one crossing |
| 1 km | stretch length | a screening convention, twice the HydroRIVERS width (the presence gap's length); not a geometric bound, since an oblique stream can cover a longer stretch (section 5); each accepted pair with a gap of 5 km or less has a covered stretch of 3.1 km or more. Measured: 500 m gives 39 more nominations, 2 km 50 fewer, 3 km 74 fewer, 5 km 86 fewer (5 km would drop the two Icelandic point contacts); no accepted pair is affected; 2 km also recorded (section 6) |
| 10 percent | third-unit flag | unchanged definition |
| 0.0005° | densification | gd1's rule |
| q = 8, 16, 32 | buffer segments per quarter circle for 125 m, 500 m, 2.5 km | round parts at most 1 m inside the true circle: R(1 − cos(π/4q)) = 0.60, 0.60, 0.75 m |
| 0.05; 1 cm; 1 m | implementation checks | the census's sampling error (pilot maximum 0.027); numerical |

**Measured sensitivities** (`threshold_sensitivity_check.py`): the 2,000 pairs with gaps up to 6.25 km or a point
contact, each constant moved alone. Counts are relative to the specified values (303 nominations on those pairs), and
all 23 accepted pairs among them stay nominated at every value tested. In "twice a width" the width is measured on each
side of a line, so a river line covers a band twice its width across:
- the river limit takes the most generous layer (Natural Earth, 5 km), so that no river layer is cut off early;
- the settings for narrow gaps and short stretches (presence gap, stretch length, facing constant) take the finest river
  layer (HydroRIVERS, 1 km). Only HydroRIVERS locates small rivers at that scale, and at Natural Earth's scale those
  settings either flood adjudication (presence gap at 5 km: 847 more) or stop catching short water frontages (stretch
  length at 5 km).

**Removed:**
- the 100 m sample spacing and the 3,000-sample cap;
- the 200-per-side cap on transect starts;
- the 0.25 km² HydroLAKES limit;
- the larger-of rule for the two lake layers, and the capped sum;
- the census's own lake width for HydroLAKES (125 m);
- the wide variant (lakes 1,500 m; reaches of at least 1,000 m³/s at 2,500 m) and its discharge limit;
- the presence rule's lake clause;
- the World Bank polygons for pairs the build closes to a point;
- the no-lake shortcut, an efficiency step;
- the facing band's 750 m, replaced by 1 km.

**Measured, not adopted:** the edge screens' ladder rungs (Natural Earth rivers 5–20 km, lakes 250–1,500 m) and
single-layer bars (0.50, 0.40), for the reasons and with the counts of section 1.

## 12. Expected effect and running time

- **Population:** 23,407 → 23,485.
- **Nominations:** about 363 (242 − 23 + 144). That is about 310 among the gaps of 5 km or less and the point contacts
  (210 by the share and presence rules, 96 more by the stretch rule, and about 4 more from the facing band's 1 km), and 53
  farther apart, all nominations today.
- **To adjudicate:** about 117 pairs without a two-model record. That is 43 point contacts (the two Icelandic pairs among
  them) and 71 others (Usulután↔La Paz among them), plus about 3 from the 1 km facing band. The measurements found none
  among 600 sampled pairs more than 5 km apart; the dry run gives the list.
- **Data:** if none is accepted, no data change and one documents pull request. If k are accepted, edges 8,461 + k and
  water-only rows 803 + k, in one data pull request with the documents.
- **Edge screens:** unaffected.
- **Running time:** about 3 hours for the census on 6 workers, and about 2 hours for check 7.1.
  - The review measured 0.25 s per short-gap pair and 2.8 s per distant pair with five lake masks; this census builds two.
  - The finer buffers (q = 16 and 32) double the mask time where they apply (measured on the 1,622 short-gap pairs).
  - The stretch rule's crossings took under a minute for the 1,622 pairs it applies to.

## 13. Maintainer (2026-09-26 and 2026-09-27, verbatim)

- "Explain the reasons of steps and threshold used in the corridor census: …" (the manuscript's census paragraph)
- "Can you review the full procedure again? I need a reasonable plan for steps and related threshold. If there is anything
  unreasonable, we can do the  corridor census again."
- "Yes", to: "Shall I write this up as a specification for your approval, with point contacts included (my
  recommendation), and then run the dry run?"
- "I have some questions: 1. why "Half of one 500 m HydroSHEDS cell" is reasonable, how about 500m; 2. The current step
  didn't consider NE rivers? 3. What's the estimated running time for your plan?"
- Transect spacing and Natural Earth rivers, asked as choices: "[No preference]" to both.
- "I'm still confused. You provide some solutions like 2.5 km for NE rivers. But we use more ladders for the water only
  borders adjudication, so can we just use same threshold in water screens?"
- "Yes, please rewrite the spec that way"
- "One quick question: I remembered the river data in NE rivers is just center line, so all pairs with water borders don't
  have shared line?"
- "For your facing-distance rule, …" with a worked example: A and B touch at one corner at their northern ends; farther
  south their banks face each other across a 2 km-wide river; with a minimum gap of 0 m the facing-distance limit is
  750 m, so the southern banks, 2,000 m apart, are excluded before the program checks the river data. "Do I understand
  correctly?"
- "Sure, please revise the specification." (to adding the stretch rule at 1 km)
- "Another question: What's the rule of verification tier A? I remembered it's only for human reviewed pairs in water
  screens."
- A reviewer's comment, passed on by the maintainer ("Is this comment reasonable?"):
  - "measured exactly" overstates the accuracy: the shares are measured against constructed shapes, and the transects
    remain discrete samples along the outlines;
  - "under 2 m" covers the projection only: the buffers' rounded parts are polygons, up to about 12 m inside the true
    circle at 2.5 km with eight segments per quarter circle;
  - recommended: word it as calculated directly from lengths with stated approximations, specify the buffer precision,
    and check near the 0.80 bar whether a finer precision changes any classification.
- The same reviewer's second comment, passed on by the maintainer ("What do you think?"):
  - some threshold explanations confuse proxies with physical properties;
  - resolution is not accuracy: HydroRIVERS derives from a 15 arc-second grid, which does not make every river lie
    within 500 m of its true position;
  - buffer coverage is not water coverage: a 2.5 km buffer around a 50 m river can cover a whole 1 km transect;
  - so "at most a fifth dry" overstates what 0.80 establishes;
  - recommended: describe the thresholds as screening choices and the score as a water-proximity indicator that
    nominates for review.
- The same reviewer's third comment, passed on by the maintainer:
  - the 1 km stretch rule's geometric justification is wrong for oblique crossings. In a synthetic test one straight
    stream crossing a 1.1 km gap covered about 1.73 km of consecutive crossings;
  - keep 1 km as a screening convention, remove the guarantee, and revise the synthetic test;
  - the stretch rule's search distance is inconsistent: section 3 said D = max(5 km, facing_m), section 5 "within 5 km".
    Decide whether 5 km is a maximum crossing distance or a minimum search reach, and align the formula, the wording and
    the tests.
- The same reviewer's fourth comment, passed on by the maintainer:
  - the oblique-stream formula needs its applicability condition. The span grows without limit as the stream turns
    parallel only while a crossing can still reach 80 percent coverage;
  - in the synthetic script a 3 km gap at 60° measures zero, while the unrestricted formula gives about −1.12 km;
  - restrict the formula to the idealized conditions, or handle the cases where 80 percent is impossible. This affects
    the explanation, not the nomination algorithm.
- "Why the twice the widest width (5km) and twice the HydroRIVERS (1km) are reasonable? Btw, why using 750m in the facing
  distance = max(2 × minimum gap, minimum gap + 750 m)?"
- "Why use the twice the HydroRIVERS rather than twice the widest width?"
- "Yes, replace 750 m with 1 km"
- "Yes, approve as drafted"

## 14. Record

This folder (`build_data/water_screen_rebuild/corridor_census_exact/`) holds:

- this specification;
- the review's measurements:
  - `touch_pairs.py` / `.csv`: the touching pairs at distance 0;
  - `census_caps.py` / `.csv`: the caps on the nominated pairs;
  - `pilot_exact.py` / `.csv`: length-based shares on 93 pairs;
  - `uncapped_timing.py`;
  - `review_questions_cc1.py` / `.csv` / `_summary.py` / `.txt`: spacing, Natural Earth rivers, running time;
  - `edge_thresholds_on_census.py` / `.csv` / `.log` / `_summary.py`: the seven edge screens on the census corridors. The
    run was stopped after 129 of its 600 distant pairs, which were slow under the 20 km river band;
  - `base_width_union_on_census.py` / `.csv`: the screens' union at base widths;
  - `far_lakes_on_census.py` / `.csv` / `.log`: 600 distant pairs, lake rules;
  - `far_nominated_b.py` / `.csv`: the 68 distant nominations of the gd1 record under the rules of sections 4–5;
  - `facing_band_check.py` / `.csv` / `_d5.csv`: covered stretches, inside and outside the facing band (1,622 pairs).
    The `.csv` run takes crossings out to max(5 km, facing band); `_d5.csv` (`--fixed-reach`) takes them to 5 km, as
    specified;
  - `oblique_stream_check.py`: the covered stretch of one straight stream crossing a gap at various angles;
  - `threshold_sensitivity_check.py` / `.csv`: the river limit, the presence gap, the stretch length and the facing
    constant, each moved alone (2,000 pairs);
  - `buffer_precision_check.py` / `.csv`: the three rules with q = 8 and q = 32, and each transect's distance from its
    frame centre (1,622 pairs);
  - `lake_stretch_reach_check.py` / `.csv` / `.log`: lake stretches out to the census reach (80 point contacts, 300
    distant pairs);
- `census_cc1.py`, `census_cc1.csv`, `census_cc1_report.txt`;
- the comparison with the gd1 record;
- the maps;
- the campaign files.

## 15. Change log

- 2026-09-26 first draft: the census's own widths, with Natural Earth rivers not used.
- 2026-09-26 the maintainer's questions on spacing, Natural Earth rivers and running time, and whether the census can use
  the water screens' thresholds, measured (section 1). Rewritten on "Yes, please rewrite the spec that way":
  - water is the edge screens' four layers at their base widths, nominated at their union's bar (sections 4–5);
  - the census's own wide variant and HydroLAKES width are dropped;
  - the spacing stays 250 m.
- 2026-09-27 the maintainer found that the facing stretch is set by the single narrowest point (section 1, defect 5),
  measured (`facing_band_check.py`, `lake_stretch_reach_check.py`). Revised on "Sure, please revise the specification":
  the stretch rule (section 5, rule 3) at 1 km for gaps of 5 km or less; its crossings (section 3), the covered crossing
  (section 4), its records (section 6), two synthetic cases (section 7), the expected numbers (sections 8 and 12).
- 2026-09-27 on the maintainer's question about verification tier A: sections 9.4–9.5 state that only the maintainer's
  map ruling accepts a pair, which is why an accepted pair is tier A (human map-verified), and that tier B cannot be
  given.
- 2026-09-27 on a reviewer's comment passed on by the maintainer:
  - "measured exactly" becomes "measured by length" (title, section 4, section 10);
  - "no samples" now means no points sampled along a transect;
  - the approximations are stated with their bounds (section 4): polygonal buffers, projection and discrete transects;
  - the buffer precision is set per width (q = 8, 16, 32, at most 1 m inside the true circle);
  - a precision check near the bar is added (section 6);
  - the running time is revised (section 12).
  
  Measured first (`buffer_precision_check.py`): q = 8 against q = 32 changes no nomination on the 1,622 short-gap pairs.
- 2026-09-27 on the reviewer's second comment (proxies described as physical properties). No rule or number changes;
  the wording does:
  - section 4 opens with what the score is, a water-proximity indicator that nominates for review;
  - "water" becomes "covered" wherever it meant inside the buffered layers ("covered crossing", "covered stretch");
  - the widths are stated as screening tolerances. The HydroRIVERS technical documentation gives the 15 arc-second grid
    (about 500 m at the equator) and states no positional accuracy;
  - the 0.80 bar loses "at most a fifth dry";
  - section 10 adds the same correction to the documents that describe the edge screens' widths as accuracies.
- 2026-09-27 on the reviewer's third comment:
  - **1 km** is a screening convention, not a geometric bound (section 5). An oblique stream covers longer stretches:
    `oblique_stream_check.py` measures 1.7 km at 80° and 3.7 km at 85° in a 1.1 km gap, against w/cos a − 0.6 G tan a.
    The synthetic test now checks that relation (section 7).
  - **5 km is the stretch rule's maximum crossing distance** (D = 5 km, section 3), not a minimum reach. Taking crossings
    out to the facing band for gaps of 2.5–5 km added 30 pairs, all with another unit across at least 95 percent of
    their transects. The numbers are re-measured with 5 km (`facing_band_check.py --fixed-reach`):
    - the stretch rule adds 96 pairs (59 never adjudicated) instead of 126 (83);
    - about 359 nominations instead of 389, and 114 pairs to adjudicate instead of 138;
    - 23 of today's nominations lost instead of 19, all adjudicated and none accepted;
    - the smallest covered stretch among the accepted pairs is 3.1 km instead of 3.5 km;
    - a synthetic case for the maximum is added (section 7).
- 2026-09-27 on the reviewer's fourth comment. The oblique-stream explanation (section 5) states its idealized conditions
  and its applicability condition:
  - a crossing can be covered only if w / sin a ≥ 0.8 G, and the span is zero otherwise;
  - the span grows without limit toward parallel only for gaps up to w / 0.8 (1.25 km for HydroRIVERS, 6.25 km for
    Natural Earth rivers).
  
  `oblique_stream_check.py` computes the idealized span piecewise and checks every case: 18 of 18 lie within two
  spacings, and the 3 km gap is zero at every angle but 0°. Its unrestricted formula had given −1.12 km at 60° and a
  spurious +115 m at 30°. The synthetic test (section 7) adds the zero cases. The nomination algorithm is unchanged.
- 2026-09-27 on the maintainer's questions about the 5 km river limit, the 1 km constants and the facing band's 750 m:
  - `threshold_sensitivity_check.py` moves each constant alone on 2,000 pairs. No accepted pair is affected at any
    value tested. Section 11 records the counts, and why the river limit takes the widest layer while the narrow-gap
    settings take HydroRIVERS.
  - On "Yes, replace 750 m with 1 km", the facing band is max(2·gap, gap + 1 km) (section 3). Its regime switch now
    falls at the 1 km presence gap. It adds about 4 nominations (3 never adjudicated), and section 12 is revised to
    match.
  - Section 3 also states that the facing band serves the share and presence rules. It had said "the share rule only",
    but the presence rule tests the band's transects.
- 2026-09-27 approved as drafted ("Yes, approve as drafted"). Frozen.
- 2026-09-27 implementation (`census_cc1.py`); no rule, threshold or number changes:
  - A pair whose facing band keeps no transect has share 0 and no presence, as section 3 says ("recorded unnominated").
    The stretch rule still examines the pair's crossings, which section 3 defines apart from the band and with point
    contacts included. Otherwise the maintainer's corner example would depend on whether the corner leaves a transect.
    The dry run reports how many pairs this touches.
  - HydroLAKES is read from `hydrolakes_all.gpkg` (all 1,427,688 lakes, the shapefile's count), written by
    `make_hydrolakes_gpkg.py`.
  - Synthetic cases (section 7.4): 16 of 16 pass (`synthetic_checks_cc1.txt`).
- 2026-09-27 checks 7.1–7.3 over all 23,408 pairs of `census_gd1.csv` (`check_old_definition_cc1.py`, summary
  `check_old_definition_cc1.txt`):
  - 7.2: the starts are identical (same kept transects, within 1 cm) for all 7,179 pairs where gd1's caps did not bind.
  - 7.3: all 23,408 gaps are equal to the metre.
  - 7.1: the median difference is 0.0003, the 99th percentile 0.0077 and the maximum 0.0673. Two pairs differ by more
    than 0.05, BFA006↔GHA007 (gap 133 m, 12 transects, 0.5195 → 0.4522) and LTU004↔POL014 (gap 169 m, 15 transects,
    0.5385 → 0.4854). One pair crosses 0.80, MNE014↔MNE015 (0.8038 → 0.7918).
  - Traced transect by transect (`trace_check1_cc1.py`, `trace_check1_cc1.txt`): gd1's own samples, pooled, reproduce its
    record to four decimals on the same transects and masks, so the difference is gd1's sampling. It placed
    floor(L / 100 m) + 2 points on each transect, both ends included, and pooled them. In the two pairs, the
    transects shorter than the median, most of them fully covered, hold 35 and 30 percent of the samples but 29 and 25
    percent of the length (−0.033 and −0.035). The rest comes from having only 3 to 11 points per transect (−0.034 and −0.018).
  - The check first called gd1's transects without gd1's repair retry and stopped at the first pair that needs it. It
    resumed with the retry, and the three pairs it repaired are the three gd1 repaired (LVA087↔LVA005, MDA011↔ROU042,
    SWE011↔SWE015).
- 2026-09-27 the dry run: `census_cc1.py` (340.7 min on 4 workers, no errors; the same three pairs repaired), reported by
  `report_cc1.py` (`census_cc1_report.txt`, and `census_cc1_new.csv` for the new nominations) and mapped by
  `map_census_cc1.py` (`maps/`). No pair's facing band kept no transect, so the first implementation note above touched
  no pair. Two defects of this specification were found, neither in the census's rules:
  - **Section 6's halved facing band.** Above a 1 km gap the band is twice the gap, so half of it is the gap itself and
    keeps only the nearest approach. There the halved test is void: 49 of its 81 losses and all 6 of its gains. At gaps of
    1 km or less it measures a narrower band: 32 losses, no gains, and no shipped row among them.
  - **Sections 8 and 12 expected 23 of the gd1 record's nominations lost; 22 are.** The review's `far_nominated_b.csv`
    has no value for Buikwe↔Masaka (gap 99,962 m) because its transects lacked section 3's chord fallback, so the
    forecast counted the pair as lost. The census measures it on the chord (share 0.94), and it stays nominated.
- 2026-09-27 the campaign started ("Yes, start the campaign on the 122 pairs"):
  - `build_queue_cc1.py` writes `cc1_queue.csv`, `cc1_research_input.jsonl` and `CODEX_INSTRUCTIONS_cc1.md`: 122 pairs
    (45 point contacts and 77 pairs that do not touch; 101 domestic, 21 cross-border). Each pair was measured again, and
    its row equals `census_cc1.csv`'s.
  - The research prompt is nt2's, verbatim by AST, except the geometric context. `standing_prompts.census_context`
    replaces the passage from "Geometric context:" to the NOTE, including its sentence on mapped lakes (HydroLAKES within
    1.5 km of a transect midpoint, as nt3 and gd1). For a point contact, `CENSUS_TOUCH_POINT` replaces the sentence that
    says the polygons do not touch.
  - The judge (`make_judge_cc1.py`) is nt4's, verbatim by AST, with the same geometric context. For a point contact, the
    first sentence of its frame is `CENSUS_FRAME_POINT`.
  - `validate_research_cc1.py`, `extract_judge_cc1.py` and `compile_gate_cc1.py` are forked from mr1's and gd1's.
- 2026-09-27 the judgment ran on "launch" (122 Sonnet 5 agents, one tool use each; the 22 prior-citation matches are the
  judges repeating the prompt's own words), and the gate is A 2 / B 4 / C 27 / D 89. The maintainer's rulings are quoted
  verbatim in `cc1_maintainer_rulings_2026-09-27.txt` and frozen in `cc1_rulings.json`:
  - Flores↔Río Negro is accepted as a water-only border within the Río Negro, about 1.7 miles, with no fixed crossing
    (the bridge links the two through Soriano). Bến Tre↔Trà Vinh is accepted, with a fixed crossing (Cầu Cổ Chiên).
  - Samukh↔Yevlakh is adjacent across a mixed land and reservoir border, so it is not water-only. Section 9.5 covers only
    water-only acceptances; on the maintainer's choice ("Add an ordinary edge (Recommended)") the pair is added as an
    ordinary edge in Stage 4 (`land_gap_overlay_pairs.csv`), its length from the ruling ("About 15km").
  - Jõgeva↔Pskov keeps the 2026-07-22 ruling (a corner contact). The census queued it because its record check counts
    only two-model verdicts; none of the other 121 pairs has an earlier ruling.
  - Kalangala↔Kalungu (a point contact) and Uttarakhand↔Haryana are not adjacent; bucket C is not overridden.
- 2026-09-27 the data change (`ship_cc1.py`, branch `data/cc1-corridor-census`):
  - The crossing flags come from the four layers. `crossing_cc1.py` runs cw1's layer-1 search on the two rows (its
    helpers taken by AST; it first reproduced cw1's verdicts on two shipped rows). There is no bridge way within 1 km of
    both Flores and Río Negro, and Cầu Cổ Chiên touches both provinces; the two passes and the rulings complete the record.
  - `border_km` is the facing arc on the World Bank polygons for Bến Tre↔Trà Vinh (16.6 km, the shipped rows' measure).
    For the two pairs whose polygons meet only at a point, it is the maintainer's length (2.7 km and 15 km): there the
    facing arc measures only 0.54 and 0.76 km around the point, which would flag a genuine border as a potential artifact.
  - Edges 8,461 → 8,464; water-only 803 → 805 (408/397); moderate 8,067; stringent 7,659; tier A 270; ADM0 unchanged.
    `build_all.py` passes 12/12.
  - `census_cc1.csv` is the census record of `completeness_check.py` (0 open), `attributability_check.py` (805/805) and
    `paper/verify_v3_section3.py` (71/71, with `census_cc1.py`'s constants and `crossing_cc1_pairs.csv`).
  - Documents: METHODS (construction, ledger, section 8's census description, a `cc1` record, and the widths as
    screening tolerances), PROVENANCE, REPRODUCING, INTRODUCTION, CHANGELOG, the loader and engine docstrings. Papers:
    V3 (text, both tables, DOCX rebuilt), the drafts EN/ZH, the Chinese checklist, the process notes EN/ZH (§9.22),
    the reproduction ledger, the fact sheet, and a status note in the supplement.

- 2026-09-27 the documents checked again against the data:
  - Samukh↔Yevlakh, now an edge, leaves the census's population and nominations, as a re-run on the new edge list
    gives (`census_cc1.py` skips the shipped edges except the rows that add one): 23,485 → 23,484 pairs (80 → 79 point
    contacts), 368 → 367 nominations (share rule 161 → 160, stretch rule 237 → 236), bar sensitivity 467–357 →
    466–356. The 26 accepted pairs and every other figure are unchanged. The guard reads `census_cc1.csv` less the
    current edges other than the Stage 3 water borders, and the documents state the current figures.
  - `build_arc_cache.py` read `rescreen_gap_overlay_pairs.csv`, deleted by st1 (2026-09-25); it now takes the
    non-touching water borders from the water file's `adds_edge` rows. Its land-gap band covers the four sub-tolerance
    rows only, and the land-gap row whose polygons meet only at a point gets no arc; the population it would build
    equals the screen record's 8,421 edges, with the band on the four sub-tolerance pairs. `border_arc.py`'s comments
    say the same.
  - The guard's band-offset check, which the fifth land-gap row had turned into a skip, runs on the four sub-tolerance
    rows: 72/72.
  - `paper/SECTION3_REVISED_EMS.md` and the submission checklist state the current counts; the drafts' Stage 3 and 4
    sentences and tier A (270) are brought up to date.

- 2026-09-30 the census's counts stated as run (maintainer: "I don't think 'counted on the database as it stands now'
  is reasonable"):
  - The documents had stated the census less Samukh↔Yevlakh (the entry above). That pair is one of the census's own
    nominations, and the 26 water borders the census led to were never taken out of its counts in the same way. The
    documents now state `census_cc1.csv` as written: 23,485 pairs (80 point contacts) and 368 nominations (share rule
    161, presence 174, stretch 237), of which 26 are water-only borders, one is the land border Samukh↔Yevlakh and 341
    were rejected; bar sensitivity 467–357.
  - `census_cc1.py` is unchanged: a re-run on the present edge list still skips Samukh↔Yevlakh (it skips the shipped
    edges except the rows that add one), so the record of the run, not a re-run, is the census's result.
  - The guard reads the record whole and checks that the census pairs that are edges now are the 26 water borders and
    the land border, all of them nominated.
  - Documents: METHODS (section 8 and the `cc1` record), CHANGELOG. Papers: V3 (DOCX rebuilt), the drafts EN/ZH, the
    process notes EN/ZH (§4.6, §9.22, a new §9.24), the condensed Section 3, the supplement's status note, the fact
    sheet, the Chinese checklist (item 74). No data change.

- 2026-09-30 the bar's sensitivity, both uses (maintainer, reading the manuscript: "What's the real threshold used in the
  corridor census?"; then "You can state both results, but can you also add the numbers of nominated pairs from 0.55 to
  0.90?"):
  - The threshold is 0.80, `border_arc.UNION_BAR`. `census_cc1.py` uses it twice: the share rule's bar, and the share of a
    crossing's length that must lie inside the layers for the crossing to count as covered (the stretch rule,
    `longest_stretch`). Section 6's bar sensitivity moves the first alone.
  - `bar_sensitivity_cc1.py` (new; deterministic) re-measures, for the 1,622 pairs of the record within the stretch rule's
    reach (gap of 5 km or less), the longest covered stretch at each bar from 0.55 to 0.90, with `census_cc1.py`'s own
    measurement (`rules` wrapped, nothing else changed). At 0.80 every pair reproduces its row of the record (share,
    presence, longest stretch).
  - Nominations at 0.55, 0.60, ..., 0.90: share bar alone 467, 439, 411, 393, 379, 368, 362 and 357; both together 593, 535, 481, 440, 396, 368, 346 and 321.
    Accepted pairs nominated: 26 of 26 at every bar in the first case; 26 of 26 up to 0.85 and 25 of 26 at 0.90 in the
    second. The pair lost is Bến Tre↔Trà Vinh (share 0.4859; gap 1,151 m, so the presence rule does not apply; covered
    stretch 3,741 m at 0.80, 1,997 m at 0.85, 750 m at 0.90).
  - The manuscript's "membership does not depend on it" is replaced by both results with the nomination counts; METHODS
    section 8, the drafts, the process notes (§4.6, §9.24), the condensed Section 3, the supplement's note, the fact sheet
    and the Chinese checklist say the same. Guard claims: the range, the share rule alone, both uses.
