# Reproduction bundle: global marine fishing across EEZs (1950-2014)

This bundle regenerates the metacoupling results for global marine fishing across
Exclusive Economic Zones (EEZs), 1950-2014, **through the `metacouplingllm`
package**. It reproduces the paper's **Tables 4 and 5** (regional catch by coupling
type, and catch-weighted regional means of the indicators) and the other figures
of its fishing example (Section 5.2).

The catch data are Carlson et al.'s (2020) reconstruction of *who caught it,
where*: for each EEZ, the catch of its own fleet and of every foreign nation. The
package labels each of these flows with its country (ADM0) database, counting
borders on land and at sea (`maritime_adjacency=True`):

| Label | When | Indicator family |
|------|------|------------------|
| **I** intracoupling | the fleet belongs to the EEZ's own country | `IFS`, `F_I` |
| **P** pericoupling | the fleet's country borders the EEZ's country, on land or at sea | `PFS`, `PFCI`, `ENP_P`, ... |
| **T** telecoupling | neither | `TFS`, `TFCI`, `ENP_T`, ... |

Carlson et al. sorted the same flows into their Type 1 (own), Type 2 (adjacent
nation) and Type 3 (distant nation) tables. The database gives the same label to
**97.3%** of their adjacent and distant flows, holding **97.4%** of their catch
(see *Comparison with Carlson et al.'s labels* below).

## What it reproduces (verified)

Run end-to-end, this bundle reproduces every cell of the paper's Tables 4 and 5,
and:

- **280 EEZs**, of which **276** are assigned to a region (**99.53%** of catch);
- **269 EEZs** with distant catch, whose unweighted mean equivalent number of
  distant fishing countries (`ENP_T`) is **3.15**, against **17.60** nominal ones;
- total reconstructed catch **5.733 Bt**, own + adjacent share **88.4%**.

**Table 5 — catch-weighted regional means** (weight = `F_total`), spot-check rows
(ordered by `TFS`; `n` = number of EEZs):

| region | n | IFS | PFS | TFS | MFE | PFCI | TFCI | ENP_P | ENP_T |
|--------|--:|----:|----:|----:|----:|-----:|-----:|------:|------:|
| Africa (West)        | 12 | 0.41 | 0.04 | 0.54 | 0.54 | 0.63 | 0.20 | 1.50 | 8.21 |
| Oceania              | 25 | 0.44 | 0.03 | 0.53 | 0.55 | 0.76 | 0.26 | 1.34 | 3.55 |
| North America        | 19 | 0.73 | 0.05 | 0.22 | 0.49 | 0.69 | 0.42 | 1.43 | 3.64 |
| Europe (North)       | 24 | 0.53 | 0.33 | 0.14 | 0.69 | 0.36 | 0.27 | 2.65 | 3.38 |
| Central America      | 14 | 0.97 | 0.02 | 0.01 | 0.12 | 0.97 | 0.52 | 1.02 | 1.88 |

(All **eleven** regions are written to `outputs/regional_indicators.csv` (Table 5)
and `outputs/regional_catch.csv` (Table 4). Regions follow Carlson et al.'s Table
S16 nation-level world-region scheme, joined to each EEZ by
`01_build_region_map.py`; four EEZs absent from Carlson's nation list are
unassigned (Taiwan, Cape Verde, Cook Islands, Ascension). Carlson's two African
regions are kept separate, so West Africa — the most telecoupled region — is not
diluted by North/East/South Africa.)

**Optional temporal series** — global telecoupled share of catch peaks at
**0.203 in 1972** and falls to **0.077 by 2014**.
Written to `outputs/telecoupled_share_timeseries.csv`.

## How to run

Requirements: `pip install "metacouplingllm[indicators]" openpyxl` (or, from a
checkout of the repository, `pip install -e ".[indicators]" openpyxl`). On
Windows, prefix with `PYTHONUTF8=1` for clean Unicode handling.

One command (runs the steps in order):

```bash
python run_all.py
```

Or step by step:

```bash
python 00_build_edge_table.py     # parse S4/S6/S8     -> data/fishing_edge_table.csv
python 01_build_region_map.py     # nation->region join -> data/eez_region.json
python 02_compute_indicators.py   # package calls      -> outputs/classified_edges.csv, outputs/indicators_by_eez.csv
python 03_analysis.py             # Tables 4 and 5, global figures, comparison with Carlson, temporal series
```

### Source tables and the offline path
Step **00** and the optional temporal series in step 03 read three source tables
(`Table S4.xlsx`, `Table S6.xlsx`, `Table S8.xlsx`) from the supplementary
material of Carlson et al. (2020), on the article page
(https://doi.org/10.3390/su12114714). Put them in a `source/` folder next to these
scripts, or set the `CARLSON2020_SUPPLEMENT` environment variable to the folder
that holds them. Without them, `run_all.py` skips step 00 and steps 01-03 run
entirely off the **pinned** inputs in `data/` (`fishing_edge_table.csv`,
`sau_country_codes.csv`, `s16_country_region.json`), reproducing both tables;
only the temporal series is skipped.

## PACKAGE functionality vs USER-SUPPLIED analysis

The paper distinguishes the package's machinery from the user's domain analysis.
In this bundle:

- **PACKAGE** (`metacouplingllm`): `02_compute_indicators.py` makes three calls.
  `build_adjacency(level="adm0", maritime_adjacency=True)` says which countries
  border each other, on land or at sea; `classify_coupling(...)` labels each flow
  of an EEZ I, P or T; `summarize_metacoupling(...)` computes every per-EEZ
  indicator (`F_I/F_P/F_T/F_total`, `IFS/PFS/TFS`, `MFE`, `IFCI/PFCI/TFCI`,
  `ENP_I/ENP_P/ENP_T`). **No classification or indicator is re-implemented here.**
- **USER-SUPPLIED**:
  - `00_build_edge_table.py` — data wrangling: parse the Sea Around Us
    reconstruction tables into the package's edge-table input format.
  - `data/sau_country_codes.csv` — the country of each EEZ and fishing nation
    (see *Labelling the flows* below).
  - `01_build_region_map.py` — data wrangling: join each EEZ to its nation and
    then to that nation's Carlson Table S16 world region (`data/eez_region.json`).
  - `03_analysis.py` — the statistical analysis that reproduces the paper:
    regional catch totals (Table 4), catch-weighted regional means (Table 5),
    global shares, the comparison with Carlson's labels, and the temporal
    telecoupled-share series. This builds *on top of* the package's labels and
    per-EEZ indicator output.

## Labelling the flows

`data/sau_country_codes.csv` gives the country of each name in the edge table,
as the ISO alpha-3 code of a unit of the package's country (ADM0) database:
the unit that holds it in the World Bank's country layer, the source of that
database.

- A country's coasts and islands count as the country: `USA (Alaska, Subarctic)`
  and `Hawaii Main Islands (USA)` as the United States, `Azores Islands
  (Portugal)` as Portugal, `Andaman & Nicobar Isl. (India)` as India.
- A territory that is a unit of its own keeps its code: Greenland (`GRL`), the
  Faroe Islands (`FRO`), Réunion (`REU`), Puerto Rico (`PRI`), Hong Kong (`HKG`).
- Units the country database does not have follow its own rule: Western Sahara's
  waters (`Morocco (South)`) go to Morocco; the Falkland Islands, South Georgia
  and the Chagos Archipelago to the United Kingdom; Taiwan to China. Tromelin and
  the Glorieuse Islands lie in the French Southern and Antarctic Lands (`ATF`).
- The Channel Islands EEZ spans two units, Guernsey (`GGY`) and Jersey (`JEY`),
  and borders what either borders. The South Orkney Islands lie in no unit of the
  database, and one fleet is recorded as `Unknown Fishing Country`: both border
  nothing, so their flows are telecoupled.

Step 02 then labels each flow. An EEZ's own catch belongs to its country's fleet,
so it is intracoupled, and so is a fleet of the same country (Portugal in the
Azores). Otherwise the flow is pericoupled when the two countries share a land
border or a maritime boundary in the package's tables, and telecoupled when they
do not. Partners are countries, so `ENP_T` counts distant fishing countries.

### Comparison with Carlson et al.'s labels

`outputs/carlson_comparison.csv` cross-tabulates the two labels. Of Carlson's
5,486 adjacent and distant flows, 5,339 (97.3%, holding 97.4% of their catch)
keep their label:

- **Distant → adjacent**, 115 flows, 24.5 Mt: Russian catch off the United
  States' coasts (8.8 Mt; the country borders Russia through Alaska), South Korean
  catch in Russia's Far East (6.4 Mt; the two zones meet along a median line),
  Faroese catch in British, Icelandic and Norwegian waters (4.9 Mt; Carlson labels
  the reverse flows adjacent), Taiwanese catch in the waters of countries that
  border China (3.2 Mt), and 95 small flows (1.3 Mt).
- **Adjacent → distant**, 26 flows, 6.7 Mt: mostly catch between Indonesia and
  China or Taiwan, as Marine Regions records no boundary between China's and
  Indonesia's zones.
- **Adjacent or distant → intracoupled**, 6 flows, 5.7 Mt: Taiwan and China,
  Portugal in the Azores, the United Kingdom in the Falkland Islands, and two
  small flows.

## Data provenance

Source: **Carlson, A.K., Taylor, W.W., Rubenstein, D.I., Levin, S.A., Liu, J.,
2020.** Global marine fishing across space and time. *Sustainability* 12, 4714.
https://doi.org/10.3390/su12114714. Supplementary material (Sea Around Us catch
reconstructions, 1950-2014).

| File | Content | Used for |
|------|---------|----------|
| `Table S4.xlsx` (`catch` sheet) | Type 1 own-EEZ catch, wide (EEZ columns) | the EEZs' own catch |
| `Table S6.xlsx` (`catch` sheet) | Type 2 adjacent-nation catch, block layout | foreign catch |
| `Table S8.xlsx` (`catch` sheet) | Type 3 distant-nation catch, block layout | foreign catch |
| `Table S16.xlsx` (`GDP` sheet) | nation → "World region" (168 nations) | the regional grouping |

In the catch tables, row 3 is the header and data start at row 4; column A is the
year (1950-2014). S4 is wide; S6/S8 are repeating `(EEZ-name, 'Catch')` column-pair
blocks where the left column of a pair holds the *fishing-nation* name. The build
step sums catch over **all years** (1950-2014) and over repeated nation rows.

**EEZ names.** Two EEZs are named differently in the own-catch table (S4) and in
the foreign-catch tables (S6/S8): `Aruba (Netherlands)` / `Aruba` and
`Congo, R. of` / `Congo (Republic of)`. Step 00 merges each pair under the second
name; left apart, each EEZ would split into an all-domestic and an all-foreign
focal system.

**Rows without a nation.** In Table S8's block for `Clipperton Isl. (France)`,
five 2013 rows hold a number where the fishing nation's name belongs (2,314 t in
all). Their fleets cannot be known, so step 00 leaves them out and reports them.

`data/sau_country_codes.csv` — the country of each EEZ and fishing nation (332
names; *Labelling the flows*).

`data/s16_country_region.json` — the `{nation: "World region"}` map extracted
from Carlson's Table S16 (168 nations, 11 regions), pinned for offline use.

`data/eez_region.json` — per-EEZ region map `{EEZ: {grp, region}}`, **built by
`01_build_region_map.py`** by joining each EEZ to its harvesting nation (handling
sub-EEZs such as `USA (Alaska, Subarctic)` and `Korea (South)`) and then to that
nation's S16 world region. Carlson's two African regions are kept distinct
(`"Africa (West)"` vs `"Africa (non-western)"`). EEZs whose nation is absent from
Carlson's S16 list (Taiwan, Cape Verde, Cook Islands, Ascension) carry
`grp = null` and are excluded from the regional tables, exactly as in Carlson's
nation-level scheme.

## Edge-table construction

`data/fishing_edge_table.csv` has one row per flow (**5,764 rows, 280 EEZs**):

| carlson_type | row meaning | `focal_system_id` | `destination_id` | `flow_value` |
|---------------|-------------|-------------------|------------------|--------------|
| `I` | own catch (Table S4) | EEZ | `EEZ (own)` | summed own catch |
| `P` | one adjacent nation (Table S6) | EEZ | adjacent nation | summed catch |
| `T` | one distant nation (Table S8) | EEZ | distant nation | summed catch |

`carlson_type` records Carlson et al.'s label; step 02 adds the database's
`coupling_type`.

## Outputs

| File | Contents |
|------|----------|
| `data/fishing_edge_table.csv` | pinned edge table (package input, built by step 00) |
| `data/sau_country_codes.csv` | pinned country of each EEZ and fishing nation |
| `data/eez_region.json` | per-EEZ region map (built by step 01) |
| `data/s16_country_region.json` | pinned nation → region map (Carlson Table S16) |
| `outputs/classified_edges.csv` | every flow with its countries, the database's label and Carlson's |
| `outputs/indicators_by_eez.csv` | per-EEZ indicators from the package |
| `outputs/regional_catch.csv` | Table 4 — regional catch totals (MMT) and proportions (%) by coupling type |
| `outputs/regional_indicators.csv` | Table 5 — catch-weighted regional means of the indicators |
| `outputs/global_totals.csv` | global catch and shares, region coverage, distant-partner figures, agreement with Carlson's labels |
| `outputs/carlson_comparison.csv` | flows and catch by Carlson's label and the database's |
| `outputs/telecoupled_share_timeseries.csv` | optional annual telecoupled share |
