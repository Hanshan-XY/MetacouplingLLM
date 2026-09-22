# Source-relabel stage — verification (2026-07-04)

Backs `scripts/relabel_sliver_corridors.py` + the reviewed manifest
`data/sliver_corridor_relabel.csv`. Pure geometry, deterministic, reproducible
from the pinned WB Admin-1 GeoPackage. Toolchain geopandas 1.1.2 / shapely
2.1.2 / pyproj 3.7.2.

## What it does

Reassigns 10 reviewed WB sliver-corridor artifacts to their true owner units
**before contiguity** (morphological opening `buffer(-D).buffer(+D)`; each
corridor moved to the reviewed same-country owner whose boundary it touches;
area-conserving). The 3 genuine panhandles (Vennbahn BEL002, Courantyne
SUR005, Dhekelia GBR001) are deliberately excluded — they are real territory.

## Result

- **12 corridors moved across 10 hosts** (Arusha = 3 parts: NW→Mara 25.0 km²,
  E→Kilimanjaro 12.0 + 0.53 km²).
- **Total area drift +0.0002 km²** — corridors are moved, not deleted; every
  host loses exactly what its owners gain (Arusha 38,141→38,104; Mara +25.0;
  Kilimanjaro +12.5).
- **All 4 fabricated cross-country edges removed** (tolerance-0 contact):
  Migori↔Arusha 83.2→0, Taita-Taveta↔Arusha 22.4→0, Salta↔Potosí 19.1→0,
  Braničevo↔Mehedinți 10.6→0.
- **Starved edges recovered**: Migori↔Mara 20.0→103.2 km (raw, lake-inclusive;
  ≈86 km land after NE-lake subtraction, matching the sliver audit),
  Kitgum↔E.Equatoria 0.9→12.9, Taita-Taveta↔Kilimanjaro 147.9→170.4.
- **Real neighbours absorbed the territory**: Jujuy↔Potosí 302→321,
  Bor↔Mehedinți 153→164.

## Per-host opening radius

Default D = 2e-3 deg (~220 m) detaches every corridor except ARG017 Salta,
whose slightly wider ribbon needs D = 3e-3 (recorded in the manifest's
`opening_d_deg` column; at 2e-3 it under-detaches, leaving 8.2 km residual
Salta↔Potosí contact). Each host uses the smallest D that fully detaches its
reviewed corridor.

## Reproduce

    python scripts/relabel_sliver_corridors.py \
        --adm1-gpkg "<WB Official Boundaries - Admin 1 .gpkg>"

Ground-truth for *which* polygons are artifacts (vs the 3 genuine panhandles):
`build_data/arusha_sliver_audit/scan_ground_truth.md`.

## Correction (2026-07-07)

The "Jujuy↔Potosí 302→313" line above is **wrong** — independently re-verified
by a fresh live re-run of `relabel_sliver_corridors.py` against the pinned WB
Admin-1 geopackage (before=302.0308 km confirmed correct; after=320.9505 km,
not 313), cross-checked against the actual shipped
`pericoupled_adm1_edge_list.csv` row (ARG010/BOL007 = 320.9505 km, exact
match). The shipped build output was always correct; only this prose summary
was wrong. Also wrong in the same original write-up (not shown above but
carried into `sliver_corridor_relabel.csv`'s note fields, now fixed): the
Wajir↔Lower-Juba, Garissa↔Lower-Juba, and Ethiopian-Somali↔Mudug "starves"
notes had before/after reversed and cited numbers that don't match any real
geometry. All four length-only corrections are *increases*, not decreases —
see the corrected `sliver_corridor_relabel.csv` notes and
`docs/METHODS_adjacency.md` §10.4 for the true values.

## Correction (2026-09-22): zero-width spikes; cut points shared; two corridors closed at a junction

The overlay `host - union(corridors)` left a zero-width spike wherever a corridor's cut point
(where the opening's outline meets the host's raw outline) lay inside a raw segment: the unit
across that segment had no vertex at the cut point, so the host kept the rest of the segment,
which the exact-contact measure counted as host frontage, and the owner's new edge matched no
neighbour's line, so its frontage went unmeasured. On the pinned geometry: Kitgum<->E.Equatoria
read 12.92 km (21.28 with the frontage measured), Lamwo<->E.Equatoria 158.99 (150.63),
Garissa<->Wajir 265.10 (229.70: a 17.7 km spike and slit counted twice), and 9.4 km of the
Kenya-Somalia border sat in no edge. The "Kitgum<->E.Equatoria 0.9->12.9" and "+0.0002 km2"
lines above describe that relabel.

The relabel now inserts each cut point as a vertex into every outline through it, and closes a
corridor whose cut falls on a raw segment facing a third unit and ending at a junction at that
junction (maintainer ruling: ARG017, 304.2 m from its cut, and SRB002, 13.8 m). ARG017's opening
radius returns to the default 2e-3 deg (maintainer ruling): the "8.2 km residual Salta<->Potosi
contact" at 2e-3 recorded above was such a spike -- the old relabel at 2e-3 leaves Salta a needle
along an 8.18 km raw segment, while the ribbon detaches at both radii (1.90 vs 1.91 km2).
Result: area drift below 0.0001 km2; no spike in any host or owner; no edge added or removed;
30 border lengths differ from the raw polygons (29 from the relabel before the fix); every
group of touching relabels conserves each neighbour's frontage exactly. Record:
`build_data/sliver_remnant/` (check_relabel_fix.log, relabel_length_delta.txt,
maintainer_rulings_2026-09-22.txt).

