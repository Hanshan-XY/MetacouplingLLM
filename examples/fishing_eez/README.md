# Reproduction bundle: global marine fishing across EEZs (1950-2014)

This bundle regenerates the metacoupling results for global marine fishing across
Exclusive Economic Zones (EEZs), 1950-2014, **through the `metacouplingllm`
package**. It reproduces the paper's **Tables 4 and 5** (regional catch by coupling
type, and catch-weighted regional means of the indicators) and the other figures
of its fishing example (Section 5.2).

The framework classifies each tonne of reconstructed catch by *who caught it,
where*:

| Type | Meaning | Coupling | Indicator family |
|------|---------|----------|------------------|
| Type 1 | an EEZ fishing its **own** waters | **I** intracoupling | `IFS`, `F_I` |
| Type 2 | an **adjacent** nation fishing the EEZ | **P** pericoupling | `PFS`, `PFCI`, `ENP_P`, ... |
| Type 3 | a **distant** nation fishing the EEZ | **T** telecoupling | `TFS`, `TFCI`, `ENP_T`, ... |

## What it reproduces (verified)

Run end-to-end, this bundle reproduces every cell of the paper's Tables 4 and 5,
and:

- **280 EEZs**, of which **276** are assigned to a region (**99.53%** of catch);
- **271 EEZs** with distant catch, whose unweighted mean equivalent number of
  distant fishing nations (`ENP_T`) is **3.18**, against **18.20** nominal ones;
- total reconstructed catch **5.733 Bt**, own + adjacent share **88.1%**.

**Table 5 — catch-weighted regional means** (weight = `F_total`), spot-check rows
(ordered by `TFS`; `n` = number of EEZs):

| region | n | IFS | PFS | TFS | MFE | PFCI | TFCI | ENP_P | ENP_T |
|--------|--:|----:|----:|----:|----:|-----:|-----:|------:|------:|
| Africa (West)        | 12 | 0.41 | 0.04 | 0.55 | 0.54 | 0.63 | 0.20 | 1.48 | 8.23 |
| Oceania              | 25 | 0.44 | 0.03 | 0.53 | 0.55 | 0.78 | 0.26 | 1.32 | 3.58 |
| Europe (North)       | 24 | 0.53 | 0.32 | 0.15 | 0.69 | 0.43 | 0.22 | 2.52 | 3.89 |
| North America        | 19 | 0.73 | 0.03 | 0.24 | 0.49 | 0.77 | 0.41 | 1.18 | 3.29 |
| Central America      | 14 | 0.97 | 0.02 | 0.01 | 0.12 | 0.97 | 0.53 | 1.02 | 1.88 |

(All **eleven** regions are written to `outputs/regional_indicators.csv` (Table 5)
and `outputs/regional_catch.csv` (Table 4). Regions follow Carlson et al.'s Table
S16 nation-level world-region scheme, joined to each EEZ by
`01_build_region_map.py`; four EEZs absent from Carlson's nation list are
unassigned (Taiwan, Cape Verde, Cook Islands, Ascension). Carlson's two African
regions are kept separate, so West Africa — the most telecoupled region — is not
diluted by North/East/South Africa.)

**Optional temporal series** — global telecoupled share of catch peaks at
**0.216 in 1972** and falls to **0.078 by 2014**.
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
python 02_compute_indicators.py   # package call       -> outputs/indicators_by_eez.csv
python 03_analysis.py             # Tables 4 and 5, global figures, temporal series
```

### Source tables and the offline path
Step **00** and the optional temporal series in step 03 read three source tables
(`Table S4.xlsx`, `Table S6.xlsx`, `Table S8.xlsx`) from the supplementary
material of Carlson et al. (2020), on the article page
(https://doi.org/10.3390/su12114714). Put them in a `source/` folder next to these
scripts, or set the `CARLSON2020_SUPPLEMENT` environment variable to the folder
that holds them. Without them, `run_all.py` skips step 00 and steps 01-03 run
entirely off the **pinned** inputs in `data/` (`fishing_edge_table.csv`,
`s16_country_region.json`), reproducing both tables; only the temporal series is
skipped.

## PACKAGE functionality vs USER-SUPPLIED analysis

The paper distinguishes the package's indicator machinery from the user's
domain analysis. In this bundle:

- **PACKAGE** (`metacouplingllm`): `02_compute_indicators.py` calls
  `summarize_metacoupling(...)`, which computes every per-EEZ indicator
  (`F_I/F_P/F_T/F_total`, `IFS/PFS/TFS`, `MFE`, `IFCI/PFCI/TFCI`,
  `ENP_I/ENP_P/ENP_T`). **No indicator is re-implemented here.**
- **USER-SUPPLIED**:
  - `00_build_edge_table.py` — data wrangling: parse the Sea Around Us
    reconstruction tables into the package's edge-table input format.
  - `01_build_region_map.py` — data wrangling: join each EEZ to its nation and
    then to that nation's Carlson Table S16 world region (`data/eez_region.json`).
  - `03_analysis.py` — the statistical analysis that reproduces the paper:
    regional catch totals (Table 4), catch-weighted regional means (Table 5),
    global shares, and the temporal telecoupled-share series. This builds
    *on top of* the package's per-EEZ indicator output.

## Data provenance

Source: **Carlson, A.K., Taylor, W.W., Rubenstein, D.I., Levin, S.A., Liu, J.,
2020.** Global marine fishing across space and time. *Sustainability* 12, 4714.
https://doi.org/10.3390/su12114714. Supplementary material (Sea Around Us catch
reconstructions, 1950-2014).

| File | Content | Used for |
|------|---------|----------|
| `Table S4.xlsx` (`catch` sheet) | Type 1 own-EEZ catch, wide (EEZ columns) | intracoupling (I) edges |
| `Table S6.xlsx` (`catch` sheet) | Type 2 adjacent-nation catch, block layout | pericoupling (P) edges |
| `Table S8.xlsx` (`catch` sheet) | Type 3 distant-nation catch, block layout | telecoupling (T) edges |
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

`data/fishing_edge_table.csv` has one row per flow (**5,769 rows, 280 EEZs**):

| coupling_type | row meaning | `focal_system_id` | `destination_id` | `flow_value` |
|---------------|-------------|-------------------|------------------|--------------|
| `I` | own catch | EEZ | `EEZ (own)` | summed own catch |
| `P` | one adjacent nation | EEZ | adjacent nation | summed catch |
| `T` | one distant nation | EEZ | distant nation | summed catch |

## Outputs

| File | Contents |
|------|----------|
| `data/fishing_edge_table.csv` | pinned edge table (package input, built by step 00) |
| `data/eez_region.json` | per-EEZ region map (built by step 01) |
| `data/s16_country_region.json` | pinned nation → region map (Carlson Table S16) |
| `outputs/indicators_by_eez.csv` | per-EEZ indicators from the package |
| `outputs/regional_catch.csv` | Table 4 — regional catch totals (MMT) and proportions (%) by coupling type |
| `outputs/regional_indicators.csv` | Table 5 — catch-weighted regional means of the indicators |
| `outputs/global_totals.csv` | global catch and shares, region coverage, distant-partner figures |
| `outputs/telecoupled_share_timeseries.csv` | optional annual telecoupled share |
