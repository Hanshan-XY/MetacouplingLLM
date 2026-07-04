# v2 source-relabel stage — verification (2026-07-04)

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
- **All 3 fabricated cross-country edges removed** (tolerance-0 contact):
  Migori↔Arusha 83.2→0, Taita-Taveta↔Arusha 22.4→0, Salta↔Potosí 19.1→0,
  Braničevo↔Mehedinți 10.6→0.
- **Starved edges recovered**: Migori↔Mara 20.0→103.2 km (raw, lake-inclusive;
  ≈86 km land after NE-lake subtraction, matching the sliver audit),
  Kitgum↔E.Equatoria 0.9→12.9, Taita-Taveta↔Kilimanjaro 147.9→170.4.
- **Real neighbours absorbed the territory**: Jujuy↔Potosí 302→313,
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
