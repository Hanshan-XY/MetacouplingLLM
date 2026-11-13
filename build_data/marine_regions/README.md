# Marine Regions input tables

The two CSV files in this folder are attribute tables of the **Maritime
Boundaries Geodatabase, version 12**, of the Flanders Marine Institute
(VLIZ), distributed by Marine Regions: Flanders Marine Institute (2023),
https://www.marineregions.org/, https://doi.org/10.14284/632.

They are licensed under the Creative Commons Attribution 4.0 International
licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/) and are
included here unmodified: the attribute fields of two layers, without
geometry, as the VLIZ WFS service returned them on 2026-11-12.

| file | WFS layer | rows | SHA-256 |
|---|---|---|---|
| `mr_eez_boundaries_v12.csv` | `MarineRegions:eez_boundaries` | 2,349 boundary lines | `7cc30de5e68c238659f43b4e8608550bb001a9e61a2bdcfa65566d0947d39adb` |
| `mr_eez_v12.csv` | `MarineRegions:eez` | 285 zone records | `e68c97e7aa1ebe774a3329f6bf73d16774b9fbfceeaab9b9c4ddddfbe651f187` |

`scripts/build_maritime_adjacency.py` builds
`src/metacouplingllm/data/maritime_adjacency_pairs.csv` from them (its
defaults point here); the commands that fetched them are in
`docs/REPRODUCING.md` §8. `.gitattributes` turns off line-ending conversion
for both files, so their bytes, and the hashes above, are the same on every
platform.
