# Pericoupling data provenance

Regenerated in **PR #50** (2026-05-30) from the refreshed World Bank Official
Boundaries dataset. This file documents the sources, method, and known
limitations of the two bundled adjacency datasets.

## Datasets

| File | Shape | Content |
|---|---|---|
| `pericoupled_adm1_edge_list.csv` | 8,369 edges · 3,373 ADM1 regions · 196 countries | Subnational (ADM1) shared-land-border adjacency |
| `PeriTelecoupling_clean.csv` | 324 adjacent pairs · 264 units · 244 ISO codes | Country (ADM0) shared-land-border adjacency matrix |
| `PeriTelecoupling_subset.csv` | small | Test fallback for the country matrix |

Both use **current ISO 3166-1 alpha-3** codes (e.g. `COD`, `ROU`, `SRB`,
`TLS`). ADM1 codes are World Bank `ADM1CD_c` (e.g. `MEX014`).

## Sources (pinned)

- **World Bank Official Boundaries** — GeoPackage distribution,
  **2026-05-14 release**. https://datacatalog.worldbank.org/search/dataset/0038272
  - ADM1 layer `WB_GAD_ADM1` (3,591 features); ADM0 layer `WB_GAD_ADM0`
    (264 features, standard layer — the 24-feature NDLSA disputed-areas layer
    is **excluded**); `WB_GAD_ocean_mask`.
- **Natural Earth** 10m physical — `ne_10m_lakes` (inland-water filter).
  Rivers (`ne_10m_rivers_lake_centerlines`) are used only for advisory flags.

## Method

Built by `scripts/build_pericoupling_db.py` (committed; reproducible —
re-running on the same inputs yields byte-identical CSVs). Summary:

1. **Land-border definition.** Two units are adjacent iff their polygons share
   a boundary segment of non-zero length. Sea/strait separation ⇒ not adjacent
   (islands such as Japan, Cuba, the Philippines have no neighbours).
2. **Spatial index.** STRtree filter-and-refine; snap tolerance 5×10⁻⁴°
   (~55 m N–S, shrinking E–W toward the poles) bridges digitisation slivers
   without merging genuinely-separate units.
3. **Inland-water filter.** A shared border lying inside a Natural Earth lake
   polygon (two units meeting in open water — e.g. U.S. states mid-Great-Lakes,
   Caspian littoral states) is removed. A border merely running *along* a shore
   or *through* a river channel is kept (rivers are flag-only, never removed).
4. **Border length** measured **geodesically in kilometres** (WGS84,
   `pyproj.Geod`) — `border_length_km`. Advisory flags: `narrow_border` (<5 km),
   `potential_artifact` (<1 km). Flags never remove a pair.
5. **ADM0** computed directly from the ADM0 polygon layer (not aggregated from
   ADM1), for complete coverage and a uniform method.

## Known limitations

- **Disputed-area allowlist.** Excluding the WB NDLSA layer drops a few real
  land borders that run through contested tracts. These are re-added explicitly:
  **`CHN`/`PAK`** (Kashmir/Khunjerab) and **`ISR`/`SYR`** (Golan). India retains
  the Kashmir geometry natively (so `IND/PAK`, `IND/CHN` need no patch). Western
  Sahara (`ESH`) is folded into Morocco in the source, so former `ESH/*` pairs
  are superseded by Morocco's borders.
- **Correctly-absent pairs.** `ARE/QAT`, `KEN/SDN`, `SDN/UGA` are *not* land
  neighbours (separated by Saudi Arabia / South Sudan respectively) and are
  rightly absent — these differ from the older dataset, which listed them.
- **River-separated province pairs.** Under the strict land-border rule, ~29
  cross-border ADM1 pairs lie within ~1 km without being adjacent because the WB
  source digitised opposite river banks as separate polygons (e.g. *Cuvette,
  R. Congo ↔ Équateur, DR Congo*; *Tulcea, Romania ↔ Odeska, Ukraine* across the
  Danube). Genuine river-following borders where the polygons share a line are
  retained (e.g. the DRC–Uganda Semliki-corridor districts).
- **Flevoland (Netherlands).** Three **domestic** NLD↔NLD edges
  (Flevoland↔Gelderland / Noord-Holland / Utrecht) are dropped because Natural
  Earth maps the IJsselmeer as a lake and the polder meets its neighbours across
  it; in reality dikes/bridges connect them. Domestic-only, so it does not
  affect cross-border pericoupling. (Also one 0.1 km MOZ/TZA sliver.)
- **ADM1 granularity varies by country** (WB "first-order unit"): micro-states
  appear at municipal granularity (e.g. Latvia 119 units), so per-country region
  counts are not directly comparable.
