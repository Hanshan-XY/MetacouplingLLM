# Specification: the build's stages (campaign `st1`)

Status: **APPROVED, 2026-09-25**, by the maintainer's rulings recorded verbatim in section 8. Only implementation
defects are corrected after this point, each logged in section 10.

## 1. Origin

Reading the English draft, the maintainer found the water work in Stage 4 and the word "rebuild" in the text: "Let's talk
about stages. It seems that water rebuild is in stage 4 when I read the draft.v2 (I remembered I said don't use
"rebuild"?), why?" The cause: the stages are the order in which the build applies its inputs, and the water-only rows
were stored in two places. The 298 rows of the first classification sat in `build_data/bridge_classified_authoritative.csv`,
which the geometry build read in Stage 3; the 481 + 24 rows found when the water screens were re-run over every edge sat
in two manifests, which the correction engine applied in Stage 4. The split recorded when the rows were found, not a
difference in method: every one of the 803 rows is nominated by the current screens, adjudicated by the two-model design
and flagged by the same four-layer crossing pipeline. The denylist of five non-adjacent contacts, found in the water
adjudication, was applied in Stage 1. The earlier instruction on "rebuild" (2026-09-19) was scoped to the manuscript.

## 2. The stages

| Stage | Run by | What it does | Inputs | ADM1 edges after |
|---|---|---|---|---|
| 1. Geometry | `scripts/build_pericoupling_db.py` | Exact-contact rook contiguity on the World Bank polygons, land-clipped; the two corrections that change polygons act before contiguity: the source-relabel of 10 sliver-corridor hosts and the Kalmykia unit merge | the four pinned GeoPackages, `data/sliver_corridor_relabel.csv`, `_ADM1_UNIT_MERGES` | 8,422 |
| 2. De facto | `scripts/build_pericoupling_db.py` | Each disputed tract folded into the administrator Natural Earth records for most of it; +16 province pairs meeting only across a tract (+3 country pairs) | the NDLSA GeoPackage, `_NDLSA_TRACT_ADMIN`, `_NDLSA_TRACT_ADM1`; writes `data/disputed_overlay_pairs.csv` | 8,438 |
| 3. Water | `scripts/apply_overlays.py` | The whole water-only classification from one file: 779 existing edges flagged water-only; the 24 water borders between non-touching units added with their water rows | `data/water_classification_pairs.csv` (803 rows) | 8,462 |
| 4. Edge corrections | `scripts/apply_overlays.py` | +4 sub-tolerance land borders; −5 contacts found not to be borders, each checked present in the geometry before removal | `data/land_gap_overlay_pairs.csv` (4 rows), `data/denylist_pairs.csv` (5 rows) | 8,461 |

The ADM0 water roll-up (26 country pairs) is computed once, after Stage 4, from the final edge list.

Rules: corrections that change polygons act in Stage 1; all water rows act in Stage 3; corrections that add or remove
non-water edges act in Stage 4. The stages give the order in which the build applies its inputs; the reviews that produced
the inputs are documented separately, with the population each ran on. `build_all.py --full` runs the stages in order,
1 → 2 → 3 → 4.

## 3. Files

- **New `data/water_classification_pairs.csv`, 803 rows**, the union of the three former stores in the order the shipped
  water table already has: the 298 rows of `build_data/bridge_classified_authoritative.csv` (its order), then the 24 rows of
  `data/rescreen_gap_overlay_pairs.csv`, then the 481 rows of `data/rescreen_water_overlay_pairs.csv` (manifest order).
  Columns: `code_a, name_a, iso_a, code_b, name_b, iso_b, water_type, water_body, has_bridge, adds_edge, border_km,
  adjudication, verification_tier, source`.
  - `adds_edge` is `True` for the 24 non-touching water borders, `False` for the 779 rows on an existing edge.
  - `border_km` carries the corridor length for the 24 rows (it becomes their `border_length_km`) and is blank for the 779
    (their length is the edge list's).
  - `source` is the per-row provenance record, kept verbatim: the manifests' `source`; for the 298 former base rows the
    base file's `note`, followed by `; crossings: <crossings>` where that column was filled. The base file's
    `country_a/b` (derivable from the codes), `border_km` and `ne_cov` (first-round screening values, superseded by the
    current screen record) are not carried; they remain in git history.
  - `iso_a/b` of the 298 former base rows are read from the edge list.
- **New `data/denylist_pairs.csv`, 5 rows**, the five reviewed non-adjacent contacts moved out of the build script's
  `_ADM1_FALSE_POSITIVE_DENYLIST`: `code_a, name_a, iso_a, code_b, name_b, iso_b, evidence, ruling, register`.
- **Unchanged:** `data/land_gap_overlay_pairs.csv`, `data/sliver_corridor_relabel.csv`, `data/disputed_overlay_pairs.csv`.
- **Deleted** (git history keeps them): `build_data/bridge_classified_authoritative.csv`,
  `data/rescreen_gap_overlay_pairs.csv`, `data/rescreen_water_overlay_pairs.csv`.

## 4. Code

- `build_pericoupling_db.py`: Stages 1 and 2 only. The denylist constant and its application are removed (Stage 4), as
  are `--bridge-csv` and `write_water_separated_manifest` (the water table is written by the engine in Stage 3).
- `apply_overlays.py`: Stage 3 (the water file: add the 24 edges, compose all 803 water rows) and Stage 4 (add the land-gap
  edges, remove the denylisted contacts), then the ADM0 roll-up and the ADM0-matrix patch for added cross-country edges.
  Still idempotent and byte-stable on shipped data. Canonical order: native rows in geometry-build order (the denylisted
  ones removed), then the Stage 3 edges in water-file order, then the Stage 4 edges in manifest order; the water table in
  water-file order, then the ADM0 roll-ups sorted. The water table's `note` column is set by the engine:
  `water-only border on a shared edge (edge screens; two-model adjudication)` for the 779 rows and `water-only border
  between non-touching units (corridor census; two-model adjudication)` for the 24; `adm1-rollup` for ADM0 rows as before.
- `build_all.py`: the bridge-CSV pin and `--bridge-csv` removed; `--full` stages the three review files into the out-dir,
  checks that every denylisted contact is present in the geometry build's output (a stale entry would otherwise do
  nothing), then runs the engine.
- `scripts/draw_validation_sample.py` reads the water file (population P unchanged: the same rows, the same draw);
  `scripts/verify_denylist_geometry.py` reads `data/denylist_pairs.csv`.

## 5. Expected effect on the shipped data

- `pericoupled_adm1_edge_list.csv`: the same 8,461 rows, every value unchanged; the last 28 rows reorder (the 24 water
  borders now precede the 4 land borders).
- `water_separated_pairs.csv`: the same 803 + 26 rows in the same order, every value unchanged except `note`.
- `PeriTelecoupling_clean.csv`, `disputed_overlay_pairs.csv`: byte-identical.
- Headline counts unchanged: 8,461 / 3,374 / 196; water-only 803 = 407 / 396; 26 roll-ups; moderate 8,065, stringent
  7,658; ADM0 326 / 320 / 300. Tiers unchanged: A 268 / B 238 / C 297 (a tier records the evidence behind a row, not where
  it is stored; tier B stays the preregistered validation study's frame).
- Intermediate counts in the documents: 8,427 raw → 8,425 relabel → 8,422 merge (Stage 1); 8,438 (Stage 2); 8,462
  (Stage 3); 8,466 → 8,461 (Stage 4).

## 6. Documents

Stage descriptions and Table 1 in the manuscript, the drafts (EN/ZH), the revised Section 3, METHODS, PROVENANCE,
REPRODUCING, the process notes, the ledger, the fact sheet, the tests and the manuscript guard. "Rebuild" is removed from
prose (the drafts, the revised Section 3, METHODS, PROVENANCE, REPRODUCING, script docstrings); folder names
(`build_data/water_screen_rebuild/`, `_archive_pre_rebuild/`), older CHANGELOG entries and the per-row `source` records
keep it.

## 7. Verification

1. Full rebuild from the pinned GeoPackages into a scratch folder: 12/12 counts; Stage 2 output 8,438 rows containing the
   five denylisted contacts; the rebuilt files equal the shipped ones as in section 5 (edge-list row set identical, the
   tail reordered; water table identical but for `note`; the other two files byte-identical).
2. The engine is a byte-stable no-op on the new shipped data; `build_all.py` OK.
3. Full test suite; manuscript guard; DOCX QA.

## 8. Maintainer rulings (2026-09-25, verbatim)

- "Let's talk about stages. It seems that water rebuild is in stage 4  when I read the draft.v2(I remembered I said don't
  use "rebuild"?), why?"
- "I prefer the option B, and please don't forget to move the 5 pairs to stage 4, also change the the repository docs. Is
  this plan OK? Btw, after the plan is done, will tier B and C merge?"
- Where the 24 non-touching water borders are added: "Stage 3 (Recommended)".
- The water table's note column: "Rewrite without 'rebuild' (Recommended)".
- The old base classification file: "Delete it (Recommended)".
- Scope of "rebuild": "I prefer prose only. But can you summarize the what future stages are?"
- After a reviewer's-view discussion that offered a three-stage alternative (adjacency / de facto / water): "Keep option B".
- The earlier instruction (2026-09-19): "if possible, don't use "rebuild" in the manuscript as it may confuse readers."

## 9. Record

`build_data/stage_order/`: this specification; `make_st1_files.py` (writes the two new files from the three former
stores and checks the implied water table against the shipped one); `st1_full_build.log` (`build_all.py --full` from
the pinned GeoPackages, 12/12 counts, geometry step 8,438 edges with every denylisted contact present);
`compare_full_regeneration.py` and `st1_full_regeneration_compare.txt` (the regenerated files equal the shipped ones
byte for byte; against main before st1 the effect is exactly section 5); `st1_pytest.log` (the full test suite).

## 10. Change log

(empty)
