# Specification: one rule for every shipped water-only row (campaign `ua1`)

Status: **APPROVED, 2026-10-02** (maintainer, asked to approve the draft: "Approve (Recommended)"). The draft is frozen as
written except where section 10 records the maintainer's answers on its open decisions. What has happened since, and
every correction, is logged in section 10.

## 1. Origin

The maintainer asked how many of the current water-only pairs the two models accepted automatically and how many they
rejected automatically. From the records:

- Tier B: the 2026-07 run shipped 238 rows on the two passes' agreement without a map check. In the verdict files, 225
  have both passes water-only at high confidence. 10 have a pass at medium. 3 have no "yes" from the research pass
  (BRA007↔BRA013, BRA012↔BRA013, CMR009↔GAB009), the rows the maintainer chose on 2026-10-01 to leave as they are.
- Tier C: the July re-checks of the 296 first-round rows (`ru1`, `wu1`) confirmed 250 with no maintainer step. 37 of those
  had a pass at medium; `ru1`'s "clean" label excluded only a low-confidence or flagged judgment.
- Rejections: of the 2,786 rejected nominations, both passes rejected 1,864 at high confidence with nothing flagged
  (`jr1`).

The maintainer then asked that the shipped rows and the fixed-crossing flags follow the same rules, and that this be done
once (section 9). This campaign checks every shipped row against one rule and puts every row that fails it before the
maintainer in one sheet.

## 2. The rule, and the population

### 2.1 Water-only status

The rule applies to every row, however it entered (the maintainer's answer "One rule for all", 2026-10-02):

- A row stands on its **latest two-model record** if both passes said water-only at high confidence and nothing is
  flagged.
- Otherwise it needs a **maintainer ruling** made no earlier than the latest challenge (a pass saying not water-only). A
  later confirmation does not undo a ruling.
- The judge's request for a human check is a flag, except in the July run, whose judgment prompt set it on every
  water-only verdict ("Set needs_human=true if your confidence is low OR your verdict is water_only=true").

This is the manuscript's rule without its incumbency clause. The manuscript sends a disagreement or medium-confidence
verdict to the maintainer only when it "would change the shipped set", which exempts a shipped row confirmed at medium
confidence.

Records read (`population_ua1.py`):

- the July audit (`verdicts/b1.json`, `b2_rerun.json`, `b3b4.json`, `b5b6.json`, `holds20km.json`), each file dated
  before the human verification it fed, b2 as re-run (2026-07-16/17);
- the b2 re-run triage sheet and the maintainer's answers on it;
- `ru1`, `rg1`, `wu1` and `wu2`, with their rulings;
- the new-candidate gate and the September gates, with their rulings (the index of `audit_all_rejections_jr1.py`);
- `jr1` and its second tranche;
- a human map verification recorded in a row's own provenance: the earlier overlays, the validation study and the
  placeholder-identity audit.

On 2026-10-02 the 866 rows give **450 automatic, 317 ruled and 99 open**:

| tier | automatic | ruled | open |
|---|---|---|---|
| A | 12 | 300 | 20 |
| B | 225 | 0 | 13 |
| C | 213 | 17 | 66 |

### 2.2 Fixed-crossing flags

All 866 flags meet the four-layer rule, so the campaign does nothing to them:

- each flag has one record: `ca1` (buckets A 48, B 48, C 285, D 422), or `jr1` for the 63 flags the maintainer set
  (31 yes and 31 no adopted from the passes' common answer, one split ruled);
- each record's final flag is the shipped flag;
- each of the 422 flags closed without the maintainer has both passes confirming at high confidence with nothing
  flagged;
- every other flag carries the maintainer's check or ruling.

Departures: 0.

### 2.3 The sheet: 111 rows

The sheet holds the 99 open rows and 12 tier-A rows that pass the rule on the two models alone. Tier A is described as
human map-verified, and the maintainer chose to rule on these 12 too ("Add them to the sheet"):

- 9 rows of the wide-river overlay, which a two-stage LLM check accepted in June 2026. The human adjudication of
  2026-07-02 covered only the 10 km net's 12 uncertain pairs (`_archive_pre_rebuild/wide_river_audit/README.md`).
- 3 rows of the river-gap overlay.

The sheet's rows by tier and group:

| tier | group | rows |
|---|---|---|
| A | challenged, kept on a deterministic measurement (`ru1` 4, `wu1` 8) | 12 |
| A | passes on the two models alone, no human map check on record | 12 |
| A | challenged by the research under `wu1`'s one-feature rubric defect, kept on the judgment | 4 |
| A | both passes water-only, a pass at medium confidence, no ruling | 2 |
| A | a disagreement with no ruling: BEN005↔NGA024 (its b2 re-run research said no; not on the triage sheet) | 1 |
| A | split, held in the audit register as a unit-identity question (BOL004↔PER007, #14) | 1 |
| B | both passes water-only, a pass at medium confidence | 10 |
| B | a disagreement with no ruling (the research said no) | 3 |
| C | both passes water-only, a pass at medium confidence or flagged, no ruling | 48 |
| C | challenged, kept on `ru1`'s measurement | 17 |
| C | challenged by the research under the rubric defect, kept on the judgment (NLD002↔NLD004) | 1 |

- 98 cross-border and 13 domestic.
- 4 non-touching: ROU013↔SRB001, DEU012↔LUX002, GUF001↔SUR004, ROU039↔UKR016.
- The deciding record is `ru1` for 74 rows, `wu1` for 19, the July audit for 14 and `rg1` for 4.
- The 3 tier-B disagreements are on the sheet because the maintainer answered "Put them on the sheet". For these three,
  that replaces the 2026-10-01 "Leave everything".

The rows are listed in `ua1_population.csv`.

Not on the sheet:

- the other 755 rows (438 automatic, 317 ruled);
- the crossing flags;
- every rejected nomination, which `jr1` closed.

## 3. Design: no model is run

### 3.1 What the maintainer gets

- **A worksheet**, one table row per pair: the units, the water body, the group, the deciding record's verdicts with
  both reasons shortened, the measurement, and a ruling cell.
- **A gate page** with:
  - both passes' reasons in full;
  - the water part of the row's `source`;
  - the earlier rulings on record.
- **The measurement**, by `measure_jr1.py`'s method on the current border arc:
  - the share of wet points and the dry kilometres;
  - the longest dry run, with map links at its two ends;
  - the 1,000 m offset reading.

  The 4 non-touching rows have no border arc and get no measurement.
- **Maps**, one page per pair: `map_jr1.py`'s design for the edges, `map_census_cc1.py`'s for the non-touching rows.

### 3.2 Rulings

- **Water-only:** the row stays, with the ruling recorded in its `source`.
- **Not water-only:** the row leaves the water-only set. An existing edge becomes an ordinary edge. A non-touching row's
  edge leaves with it, as the `wu1` and `rg1` removals did.
- **Not adjacent:** the contact goes to the denylist with an entry in the audit register, as #7 to #20 did.
- Every row needs a ruling, and the campaign is not closed until all 111 have one.
- A research file the maintainer makes as a reference is the maintainer's, not a model pass.
- The rulings are frozen verbatim (`ua1_maintainer_rulings_<date>.txt`) and in `ua1_rulings.json`.

### 3.3 Tiers

The maintainer chose "the first option" on 2026-10-02, read here as: a tier-B row ruled on a map moves to tier A. The
alternative was to keep its tier and record the ruling in `source`.

- **Tier B** becomes the rows of the validation study's frame that still rest on the July run's automatic acceptance:
  225 if the 13 are upheld. PROVENANCE and the tier-B test change with it. The study's report stays a dated record: its
  98.7% was measured on the 238-row frame.
- **Tier A** rows stay A. The 12 two-model-only rows are then map-verified, as the tier says.
- **First-round rows** keep tier C when ruled, as `ru1`'s seven did (section 8).

## 4. What the campaign does not change

- The screens, the nominations, the crossing flags and the rejected nominations.
- The two-model verdicts themselves.
- The validation study's report (2026-07-14).

## 5. Expected effect

- **All 111 upheld:**
  - no change to water-only status, edges or views;
  - the ruling recorded in 111 `source` strings;
  - tiers 332 / 238 / 296 → 345 / 225 / 296;
  - the documents change once.
- **k rows ruled not water-only:**
  - water-only 866 → 866 − k;
  - an existing edge enters the stringent view, and the moderate view too if it has no fixed crossing;
  - a non-touching row's edge leaves the edge list, and the moderate view too if it has a fixed crossing;
  - ADM0 changes only if a country pair's status changes.

No figure is predicted.

## 6. Documents (once, after the rulings)

- **Manuscript, section 3.4:** the rule sentence loses its incumbency clause and then holds for every shipped row.
- **METHODS:**
  - the adjudication-design paragraph, with the shipped rows' tally (automatic, ruled);
  - a `ua1` record;
  - the exception to "every verdict was set by the two-model design": the two placeholder-unit rows settled by the
    identity audit (MOZXXX↔MWI002, MOZXXX↔MWI004). A blind agent cannot adjudicate the second; the first never had the
    two passes.
- **PROVENANCE:** the tiers, the tier-A definition, the counts.
- **Also:** CHANGELOG, the drafts, the process notes, the fact sheet, the Chinese checklist, the tier-B test,
  `build_all.py` EXPECTED if a count changes, and the guard.

## 7. Verification

1. `population_ua1.py`: 866 rows, automatic 450 / ruled 317 / open 99, a sheet of 111, crossing departures 0.
2. The measurement has one row per edge on the sheet (107), with the HydroRIVERS share equal to the screen record's.
3. The worksheet, the gate page and the maps hold 111 rows or pages, in one order.
4. After the rulings, `population_ua1.py` gives open 0 and no tier-A row passing on the two models alone.
5. With a data change: `build_all.py`, the completeness and attributability checks, a full regeneration, the full test
   suite and the guard.

## 8. Decisions left to the maintainer

1. **The tier of a first-round row ruled water-only.** Keep C (this draft, as `ru1`'s seven), or move it to A. Moving it
   would also move the 17 first-round rows already ruled, and tier C would then mean first-round rows both passes
   confirmed at high confidence (213).
2. **The order of the sheet.** By group (this draft) or by country.

## 9. Maintainer (2026-10-02, verbatim)

- "For the current water only pairs, how many pairs were accepted automatically by 2 models and how many pairs were
  rejected automatically by 2 models?"
- "We should follow the current design,  so "10 have a pass at medium" and "3 have no "yes"" don't count the
  automatically accepted pairs, but 250 confirmations should count, right? Btw, according to current design, should we
  re-adjudicate the "10 have a pass at medium" and "the other 46"?"
- "I  prefer the first option. But I wanna sure sure that if we can see the current water only and fix crossing pairs in
  the database follow the same rules. I don't wanna adjudicate pairs again and again."
- Answers: "One rule for all (Recommended)"; "Put them on the sheet (Recommended)"; "Add them to the sheet
  (Recommended)".
- On the draft: "Approve (Recommended)". On the tier of a first-round row ruled water-only: "Move to tier A". On the
  order of the sheet: "By group (Recommended)".

## 10. Change log

- **2026-10-02, the maintainer's answers on section 8.**
  - **Tier.** A first-round row ruled water-only moves to tier A, and so do the 17 first-round rows already ruled (`ru1`
    7, `wu1` 10). Section 3.3's third point is replaced: tier A is every row with a maintainer ruling on record; tier C
    is the first-round rows both passes confirmed at high confidence (213). If all 111 are upheld, section 5's tiers
    read 332 / 238 / 296 → **428 / 225 / 213**.
  - **Order.** The sheet runs by group, the same issue together whatever the tier, by pair code within a group; labels
    U1 to U111:
    1. both passes water-only, a pass at medium confidence or flagged, no ruling (60: A 2, B 10, C 48);
    2. a disagreement with no ruling (4: A 1, B 3);
    3. challenged, kept on a deterministic measurement (29: A 12, C 17);
    4. challenged by the research under the rubric defect, kept on the judgment (5: A 4, C 1);
    5. split, held in the audit register (1: A);
    6. passing on the two models alone, no human map check on record (12: A).
- **2026-10-02, the package built (no model run).**
  - `population_ua1.py`: the same result (automatic 450 / ruled 317 / open 99; a sheet of 111; crossing departures 0);
    it also writes `ua1_gate.json`, the sheet's rows in `jr1_gate.json`'s shape.
  - `measure_ua1.py` (`measure_jr1.py` run unchanged on the 107 edges): the HydroRIVERS share equals the screen record's
    on all 107. 35 rows have no dry point; 49 have a dry run of 1 km or more.
  - Maps:
    - `map_ua1.py` (`map_jr1.py` run unchanged): `maps/ua1_map_check.pdf`, 107 pages;
    - the 4 non-touching rows: `corridor_census_exact/map_census_cc1.py` on `ua1_census_rows.csv`, the re-measured
      rows equal to the census record;
    - `combine_maps_ua1.py`: `maps/ua1_111_pairs.pdf`, 111 pages, page *n* = U*n*.
  - `make_worksheet_ua1.py`, with `make_worksheet_jr1.py`'s formatting by AST: `ua1_ruling_worksheet_2026-10-02.md`
    and `GATE_ua1.md`, 111 rows each.
  - The July judgment's request for a human check, which its prompt set on every water-only verdict, is not shown on
    the sheet or the maps (section 2.1).
- **2026-10-02, after the package: Standard M and blind judgment.** The maintainer asked: "After this change, will the
  current water only and fix crossing pairs in the database follow the same rules?"
  - **Two gaps in the 438 rows that stand on both passes at high confidence (tier B 225, tier C 213).** Both were judged
    under the July, `ru1` and `wu1` designs:
    - **Standard M** (maintainer ruling of 2026-09-01) was never applied to them. Under it, a named dry segment sends a
      pair to the maintainer's map check whatever its share, and no pair is accepted on "dry share under 20%". Reading
      both passes' reasons gives 19 that fail it:
      - 15 accept a named dry piece inside the pair;
      - 4 are accepted on a stated non-water share of 8 to 13%.
      Also, METHODS' statement that no shipped row carries an accepted verdict with a quantified 4–25% dry share
      (checked 2026-09-01) is wrong: at least 7 do.
    - **Judge blindness.** Today's judgment pass is blind to the repository's prior verdicts (NO_REPO). The July judges
      were not barred from the repository, the `ru1` and `wu1` judges were told the shipped verdict, and about 50 of the
      438 cite repository files.
  - **Answers.**
    - "Add them (Recommended)": the 19 join the sheet as group 7.
    - "Blind re-judgment first": a fresh Sonnet 5 judgment of all 438, in today's design, run only on the maintainer's
      "launch" and before the rulings, so that any new flag joins the same sheet.
  - **The re-judgment.**
    - The judge is given:
      - the research verdict of each row's deciding record, verbatim (GPT-5.6 Sol, July audit with b2 as re-run, `ru1`
        or `wu1`; that research was blind);
      - the deterministic screen context.
    - It is `mr1`'s judge, by AST: `make_judge_dc1.py`'s NO_REPO evidence rule, its rule reminder (Standard M:
      "a genuine dry segment of that kind defeats water_only REGARDLESS of its share"; name any dry component) and its
      tests and template.
    - The per-pair context is `build_queue_mr1.py`'s enrichment, by AST. No prompt states a prior verdict, the shipped
      status, the campaign or a date.
    - Two workflows: tier B (225) and tier C (213).
    - **The gate.** The blind record becomes each row's latest two-model record under section 2.1's rule. A row it does
      not close (a pass below high confidence, a judgment not water-only, the judge's needs_human, or a dry component the
      judge names inside the pair) joins the sheet as group 8.
  - **Labels.** U1 to U111 stay; groups 7 and 8 follow from U112. The package (measurement, maps, worksheet) is then
    built once for the whole sheet.
- **2026-10-02, the blind re-judgment prepared, NOT RUN.**
  - **`standard_m_ua1.py` → `ua1_standard_m.csv`.** The 19 group-7 rows, each with the sentence that names its piece,
    asserted verbatim in the deciding record's reason: named 15, share 4; tier B 12, tier C 7. They were found by term
    scans of both passes' reasons on all 438, and every hit was read in full. Reasons with no hit were not read in full;
    the blind judgment re-reads them.
  - **`build_rejudge_ua1.py` → `ua1_rejudge_queue.csv`, `ua1_rejudge_research.jsonl`.** 438 edges (235 domestic, 203
    cross-border); b1 = tier B 225, b2 = tier C 213. `build_queue_mr1.py`'s enrichment by AST.
  - **`make_judge_ua1.py` → `judge_ua1_b1.wf.js` (225 agents), `judge_ua1_b2.wf.js` (213).** `make_judge_mr1.py`'s
    method by AST, with three ua1 edits to the template, which was written for pairs that ship as land borders:
    1. "This pair IS an edge in the shipped graph and ships today as a LAND border ..." (false here, and a disclosure)
       becomes "This pair IS an edge of the graph: the two units share the border described above."
    2. The edge frame loses the shipped status and the population base rate ("Most members of this population are
       borders that follow a river or lake for only part of their length ...").
    3. needs_human is no longer set on every water_only=true; otherwise every confirmation would be flagged, as in July.
       It is now set for low confidence, disagreement with the research, marine water, or a reason the judge states.

    Checks:
    - node syntax check of both files;
    - a dry run of each with a stub agent (225 and 213 calls, as many results);
    - the judge's instructions hold no shipped status, base rate or earlier verdict;
    - the research reasons' only watch-list words are "report(s)" and one "ICJ's ... navigation ruling".
  - **`extract_judge_ua1.py`** is written (`extract_judge_mr1.py`'s parser by AST).
  - **`population_ua1.py`:**
    - reads the blind record once it exists (group 8) and Standard M (group 7);
    - writes its files only when run itself;
    - `standard_m_ua1.py` and `build_rejudge_ua1.py` read the records as they stood before the blind record
      (`UA1_PRE_BLIND`).

    The sheet is now 130 rows (groups 1 to 7); U1 to U111 are unchanged. The package scripts no longer fix the sheet's
    size.
  - **On the maintainer's "launch".**
    1. Run the Workflow tool with `judge_ua1_b1.wf.js` and with `judge_ua1_b2.wf.js`.
    2. Then, once: `extract_judge_ua1.py`, `population_ua1.py`, `measure_ua1.py`, `map_ua1.py`, `combine_maps_ua1.py`
       and `make_worksheet_ua1.py`. The corridor maps of the 4 non-touching rows stand, since every row of groups 7 and
       8 is an edge.
- **2026-10-02, launched.** The maintainer: "launch". Workflow runs:
  - `judge_ua1_b1.wf.js`, 225 agents: run `wf_a557f04e-1e9`;
  - `judge_ua1_b2.wf.js`, 213 agents: run `wf_f31ffe3e-5b8`.
- **2026-10-03, the judge model.** The maintainer: "not sonnet 5？"
  - **The finding, from the run transcripts' model field:**
    - the template's agent option `model: 'sonnet'` is an alias, not a model;
    - it was served as `claude-sonnet-5` in every judgment workflow of 2026-07-23 to 2026-09-28 (`wu1`, `rj1`/`rj2`,
      the census runs, `dc1`, `pr1`, `hl1`, `gs1`, `ba1`, `cw1`, `mr1`, `cc1`, the four `ca1` batches);
    - it was served as `claude-sonnet-5-5` from 2026-10-02.
  - **The two runs above** were therefore on Sonnet 5.5. They also stopped when the session ended, with 37 and 54 judges
    started. They are discarded and none of their verdicts is used.
  - **Template edit (4) in `make_judge_ua1.py`:** the model is pinned as `model: 'claude-sonnet-5'`. A one-agent probe
    (`wf_2efc63a2-7f6`) showed the full ID is accepted and served as `claude-sonnet-5`.
  - **`extract_judge_ua1.py`** now reads the model each judge was served from its transcript, and refuses any other.
  - **Relaunched under the maintainer's "launch"** (a blind Sonnet 5 judgment):
    - b1: run `wf_e6992c47-18a`;
    - b2: run `wf_605bdb7b-819`.
  - **Also found:** the `jr1` crossing judgments of 2026-10-02 (`wf_450ce7db-005`, 59 agents; `wf_863544a8-e82`, 4) were
    served `claude-sonnet-5-5`. The 63 rows' `source` strings and METHODS, CHANGELOG and the manuscript name them
    "Sonnet 5". This is put to the maintainer.
- **2026-10-03, the 63 `jr1` crossing judgments again, on Sonnet 5.** The maintainer: "Re-judge on Sonnet 5
  (Recommended)".
  - **`make_judge_crossing_ua1.py`** copies the two `jr1` crossing workflows unchanged except the model, pinned to
    `claude-sonnet-5`, and the run name. Both copies render the prompts their 2026-10-02 runs sent, word for word (59 of
    59, 4 of 4, checked against the run transcripts).
  - **Runs:**
    - `judge_crossing_ua1_t1.wf.js` (59): run `wf_67905d6b-015`;
    - `judge_crossing_ua1_t2.wf.js` (4): run `wf_4d361c45-e00`.
  - **The check after the runs.**
    - The gate of `compile_gate_crossing_jr1.py` is applied to the same research verdicts and location tests with the
      Sonnet 5 judgments.
    - A row whose group and common answer are unchanged keeps the flag the maintainer adopted ("As proposed"), and its
      record then cites the Sonnet 5 judgment.
    - A row where they change goes to the maintainer's crossing ruling on the same sheet.
    - The one row the maintainer ruled on a split (Căușeni↔Transnistria) keeps its ruling.
  - **t2 done (4 of 4).** The Sonnet 5 answers equal the 2026-10-02 ones: B133 yes; B134, B148 and B153 no.
  - **The usage limit.** The maintainer's session usage limit stopped the other three runs partway: t1 36 of 59, ua1 b1
    35 of 225, ua1 b2 36 of 213. The rest failed at the API ("session limit"), before any judging.
  - **Resumed after the limit reset**, each under its own run ID with `resumeFromRunId`:
    - t1 first, alone. Its journal showed the 6 judges started since the resume were all among the 23 that had failed,
      and none of the 36 finished ones ran again.
    - then b1 and b2.
    The finished verdicts come back from cache, unchanged.
  - **t1 done (59 of 59), and the comparison** (`compare_crossing_ua1.py` → `ua1_crossing_rejudge.json` / `.txt`). All 63
    judges were served `claude-sonnet-5`; the 23 empty transcripts are the attempts the limit stopped. The Sonnet 5
    judgment answers as the 2026-10-02 one on 58 of 63:
    - **57 unchanged:** the flag the maintainer adopted stands.
    - **B78 Căușeni↔Transnistria:** still split (research no, judge yes); the maintainer's ruling stands.
    - **5 changed:** research yes, Sonnet 5.5 yes, Sonnet 5 **no**. Each is now a split, and its flag (yes today) goes
      to the maintainer's ruling as group 9 of the sheet:
      - B43 Alger↔Tipaza (Oued Mazafran): the cited viaduct is 819 m from the shared arc;
      - B104 Kayunga↔Mukono (Sezibwa): emergency reconstruction, temporary crossing;
      - B110 Cerro Largo↔Tacuarembó (Río Negro): Paso Mazangano is 2,138 m from the arc;
      - B112 Río Negro↔Tacuarembó (Arroyo Salsipuedes Grande): the Ruta 20 bridge way is 274 m from Tacuarembó;
      - B126 Manicaland↔Masvingo (Save): Birchenough Bridge is 2,966 m from the arc.

      B43, B110 and B126 were pre-flagged on 2026-10-02 for the same location test. B110 is one of the three
      structures METHODS and the manuscript count as confirmed by both passes, so that count changes with the ruling.
  - **Group 9 in the package.**
    - `map_crossing_ua1.py`: `map_crossing_jr1.py` run unchanged except for its records (the 5 rows, with the Sonnet 5
      judgment and the split group, labels X1 to X5), its output folder and file name.
    - `make_worksheet_ua1.py` and `combine_maps_ua1.py`: the 5 rows as section 9 of the worksheet and the gate page,
      with their map pages after the rows.
  - **The usage limit cut b1 and b2 a second time** (b1 160 of 225, b2 179 of 213). Both resumed again under the same
    run IDs.
- **2026-10-03, the blind re-judgment done and the sheet built.**
  - **`extract_judge_ua1.py` → `ua1_rejudge_results.json`.** 438 verdicts, one per row. Every judge that answered was
    served `claude-sonnet-5`; the empty transcripts are the attempts the limit stopped.
    - water_only true 389, false 49 (objection dry-land 47, other 2);
    - confidence high 253, medium 166, low 19;
    - needs_human 179.
  - **`population_ua1.py`.** The blind record is the latest two-model record of the 438. Under section 2.1's rule it
    closes 237 (both passes at high confidence, nothing flagged) and leaves 201 open:
    - 18 of the 19 Standard M rows. The 19th closes on the blind record but stays on the sheet, by the maintainer's
      answer.
    - 183 new: group 8, in order of first reason:
      - 38 where the judgment says not water-only;
      - 131 water-only at medium confidence, 111 of them with a request for a human check;
      - 5 at low confidence;
      - 9 at high confidence with a request for a human check.
  - **The sheet: 313 rows** (groups 1 to 6, 111; group 7, 19; group 8, 183), plus group 9, the 5 crossing flags X1 to
    X5. Crossing departures 0.
  - **The package:**
    - `measure_ua1.py`: 309 edges; the HydroRIVERS share equals the screen record's on all 309. 77 have no dry point;
      163 have a dry run of 1 km or more.
    - `map_ua1.py` and `combine_maps_ua1.py` → `maps/ua1_313_pairs.pdf`: 318 pages (309 edges, 4 corridor maps, 5
      crossing pages).
    - `make_worksheet_ua1.py` → `ua1_ruling_worksheet_2026-10-03.md` (313 rows + X1 to X5) and `GATE_ua1.md`.
    - The dated copy of 2026-10-02 (the 111-row draft) is replaced.
  - **Observation.** The workflow harness relays the message that launched a run, verbatim, to every agent. For these
    four runs that is the maintainer's "not sonnet 5？". One t2 judge commented on it in its reason (B133: "on the
    relayed question 'not sonnet 5': this judgment pass was in fact run by Claude Sonnet 5"). The text discloses no
    verdict, shipped status or pair, so blindness holds, and no verdict rests on it. Future blind runs are launched from
    a neutral message.
- **2026-10-03, the open confirmations judged again with the web allowed.** The maintainer: "Why no web access for 136
  flagged at medium/low confidence? You have the web search function, right?"
  - **The finding.** The judgment template of the September campaigns (`make_judge_dc1.py`, reused by `mr1` and here)
    says "You have no web access -- judge the evidence." The July, `ru1` and `wu1` judges were not told so, and they
    cite web searches. The blind run's transcripts show its judges used no tool but the structured answer. The effect
    of this on confidence was not raised before the launch.
  - **Answer:** "Re-judge with web (Recommended)": "Same prompts, Sonnet 5 pinned, the repository and earlier verdicts
    still off-limits, but the judge may search the web to check the research's sources. Pairs it then confirms at high
    confidence with nothing flagged leave the sheet; the rest stay. One judge runs first as a test (served model, web
    use), then the rest on your word."
  - **The rows: 145** of group 8, those where both passes say water-only. The blind judgment was at medium confidence on
    131, at low on 5, and at high with a request for a human check on 9. The 38 rows the blind judgment does not call
    water-only, and the 19 Standard M rows, stay on the sheet.
  - **`make_judge_web_ua1.py`** → `judge_ua1_web_canary.wf.js` (1 agent: AFG004↔TJK004, the first in sheet order, the
    test judge), `judge_ua1_web.wf.js` (144) and `ua1_web_queue.txt` (the 145). It takes the blind workflows' pair items
    unchanged, and their template with two edits:
    1. "You have no web access -- judge the evidence." becomes: the judge has web access, and before deciding checks the
       sources the research cites and any others it needs on the course of the border.
    2. The evidence rule adds (d): public web sources the judge finds and reads itself. Still barred: the repository and
       every file on the computer; any prior audit record, ledger, verdict file or shipped data; and the data, code or
       documentation of any database, dataset or package that lists or classifies administrative adjacencies or water
       borders, wherever published. The judge names the web sources it relied on.

    Checks:
    - the 145 rendered prompts differ from their blind prompts by these two passages alone (one difference pattern);
    - the new text names no earlier pass or outcome;
    - model `claude-sonnet-5`; node syntax check.
  - **`extract_judge_web_ua1.py`** collects the verdicts into `ua1_web_results.json`, with each judge's searches and
    fetched URLs read from its transcript.
    - The model served must be `claude-sonnet-5`.
    - A judge whose tool calls or tool results show this project's repository or package is void, and its row keeps the
      blind record.
    - Tested on the blind b1 run's transcripts: every transcript's pair id is read, and those judges used no tool but the
      structured answer.
  - **`population_ua1.py`.** The web record, once run, is these rows' latest two-model record (`UA1_PRE_WEB` reads the
    records without it). A dry component it names is a flag, as for the blind record. Without the file, its three
    outputs are unchanged (checked byte for byte).
  - **Launch.** The harness relays the message that launches a run to every judge (above). This session's latest messages
    discuss the earlier judgment's flags, so the test judge waits for a neutral message from the maintainer ("launch").
    The 144 then run on the maintainer's word, also given as a neutral message.
- **2026-10-03, the test judge, and the memory index.** The maintainer: "launch". Run `wf_0414a21e-2ac`
  (`judge_ua1_web_canary.wf.js`, AFG004↔TJK004).
  - **The checks pass.**
    - The judge was served `claude-sonnet-5`, and the relayed message was "launch".
    - It loaded the web tools and made 8 searches and 7 page fetches: Wikipedia, Tajikistan's foreign ministry, and the
      University of Nebraska module the research cites.
    - No local file was read, and nothing of this project was met.
  - **Its verdict:** water-only, high confidence, nothing flagged, dry component none identified. Its sources are named.
    (The blind judgment had asked for a map check on this pair.)
  - **Cost.** 5.4 minutes. Its transcript totals 248k tokens of cache creation, 1.78M of cache reads and 15k of output.
    The blind judges' medians were 62k, 127k and about 24k.
  - **Found: every workflow judge receives the memory index (MEMORY.md) as context.** This has been so since 2026-09-10
    (the transcripts' `instructions` attachment). The index's in-progress line described the run:
    - the blind judges (438) and the crossing judges (63) read "one rule for every shipped water-only row" and "blind
      Sonnet re-judgment of the 438 automatic rows";
    - the test judge read the same line. (First recorded here as "WEB re-judgment of the 145 open confirmations", the
      line on disk at its launch. That was wrong: a judge receives the copy of the index loaded at the session's last
      compaction; see the next entry.)

    The blind judges read it. At least 25 used words found only in the index: "Standard M", "the project's documented
    GPT-5.6 Sol research + Claude Sonnet 5 judgment design", "adjudication_models.md", "re-judgment". None cites the
    cohort words (shipped, automatic, 438), and no pair of the 438 is named in the index, so the effect on the verdicts
    cannot be measured.

    Earlier runs had the index of their day. In-progress lines describing the run's cohort are in the runs of
    2026-09-22 (naming COG006-COG008 and KEN007-KEN046 with outcomes), 2026-09-27 and 2026-10-02 (`jr1` crossing).
  - **Fixed for the runs to come.** The index line is replaced by a neutral pointer, and the old line is kept verbatim at
    the top of the memory note. The index now names no run, cohort or verdict. The test judge's verdict is not used, and
    AFG004↔TJK004 goes back into the main run.
  - **Put to the maintainer:** whether the 236 rows the blind record closed (Standard M rows aside) are also judged with
    the web and the clean index.
- **2026-10-03, the 381 rows: launched and stopped.** The maintainer's answer: "381 rows (Recommended)".
  - **`make_judge_web_ua1.py`, rows widened.** Every row whose blind record has both passes saying water-only, outside the
    Standard M rows: 236 the blind record closed and 145 it left open; ordered by pair code. `judge_ua1_web.wf.js` has
    381 agents. The canary file is regenerated byte-identical to the one run.
    Checks:
    - the 381 prompts differ from their blind prompts by the two web passages alone;
    - AFG004↔TJK004's prompt is identical to the test judge's;
    - model `claude-sonnet-5`.
  - **Run `wf_ed0fd6e2-569`.** The first transcripts were checked at once. The relayed message was "launch", and the
    model `claude-sonnet-5`, but the memory index was the old copy (12,329 characters, with "blind Sonnet re-judgment
    of the 438 automatic rows"), not the neutral file on disk. The run was stopped within seconds: 6 judges had started
    and none had answered (no tokens).
  - **Found: a judge receives the copy of the memory index that was loaded at the session's last compaction.** Edits on
    disk reach the judges only after the next compaction, or in a new session. The canary and the stopped run had the
    same copy, the one the compaction of 2026-10-03 loaded.
  - **Next.** The maintainer compacts the session (`/compact`), which reloads the neutral index, then sends "launch".
    The first transcripts are checked again (index, relayed message, model) before the run is let continue. Six judges
    run at a time.
- **2026-10-03, three workflows.** The Workflow tool runs at most min(16, CPUs − 2) agents at once per workflow. This
  machine has 8 logical CPUs, so that is 6, the peak of each blind run. The maintainer, asked whether to split the 381
  rows into three workflows launched together (18 at a time): "Yes".
  - `make_judge_web_ua1.py` writes `judge_ua1_web_p1.wf.js`, `_p2` and `_p3`: 127 rows each, contiguous in pair-code
    order.
  - Together they render exactly the 381 prompts checked before, in order, with the same model.
  - The single 381-agent file, whose run `wf_ed0fd6e2-569` was stopped, is removed.
  - Extraction: `extract_judge_web_ua1.py <p1 out> <p2 out> <p3 out> --runs <p1 dir> <p2 dir> <p3 dir>`.
- **2026-10-03, the 381 rows launched.** The maintainer compacted the session (`/compact`) and sent "launch". The three
  workflows were launched together:
  - `judge_ua1_web_p1.wf.js`: run `wf_0e41d54d-fb9`;
  - `judge_ua1_web_p2.wf.js`: run `wf_3b2d4248-8d1`;
  - `judge_ua1_web_p3.wf.js`: run `wf_3c675248-1cd`.
  - Checked at once, on the first three transcripts of each run: the relayed message was "launch" and the model
    `claude-sonnet-5`. The memory index had 6,688 characters, identical to the neutral file on disk, and was the same
    copy in all 18 judges then started. It held none of the words that described the run (IN PROGRESS, ua1, 438, 381,
    145, 236, automatic, open confirmation, re-judgment, blind, sheet).
  - The first 18 judges' answers: all 292 model replies from `claude-sonnet-5`; 115 web searches and 54 fetches. One
    judge (USA044↔MEX019) also used the app's built-in browser on public OpenStreetMap pages (25 calls: screenshots,
    pans, closing pop-ups), which the evidence rule allows as a public map. The pane is shared, so two judges browsing
    at once could see each other's tab; the extraction checks the browser judges' time windows for overlap.
  - **`extract_judge_web_ua1.py`, the project check corrected.** The app saves a fetched PDF or a screenshot under
    `...\projects\D--metacoupling\tool-results\`, and that path matched "metacoupling": a false hit. The pattern is now
    `(?<!D--)metacoupling|hanshan-xy` (the account name, not Hanshan the place). Tested: the app's folder does not
    match; the repository folder, the GitHub repository and the package name do.
  - **`population_ua1.py`, a void web judgment.** No web judgment stands for that row, so it does not leave the sheet:
    a row the blind record closed goes back on it (the maintainer's reason for the 381: every row that leaves the
    sheet without a ruling rests on a judge that had the web and saw no cohort line). Tested with a throwaway results
    file (one closed row void): the sheet went from 313 to 314 rows, that row with "the web judgment void"; the file was
    removed and the outputs are byte-identical to before. `extract_judge_web_ua1.py` also records each judge's use of
    the built-in browser and lists the judges whose use overlapped.
- **2026-10-03, the first launch stopped; the judges limited to web search and fetch.** The judges had the default tool
  set, and the prompt named WebSearch and WebFetch without excluding the rest. Found in the transcripts:
  - three judges (AGO012↔COD019, ALB011↔ALB029, ALB011↔ALB017) used the app's built-in browser at the same time. They
    navigated and closed each other's tabs, and two read or saw the other's page (a Congo map, an Albanian relation);
  - ALB011↔ALB017's judge also used the shell. It downloaded OpenStreetMap boundaries from Nominatim with `curl` (four
    requests carrying the maintainer's email address in the User-Agent; no other judge's tool call carries it) and
    measured them with Python in the session's scratchpad;
  - DEU004↔POL016's judge marked a chapter in the session's view.

  The three runs were stopped: 59 judges had started and 41 answered. 39 of those used only web search and fetch;
  AGO012↔COD019 and USA044↔MEX019 also used the browser.
  The maintainer, asked how the other 342 rows should run: "Web-only, enforced (Recommended)". That is a judge
  definition offering only web search and fetch, which also forbids personal information in a search or a URL. The
  prompt is unchanged, one test judge runs first, then the 342 on the maintainer's word, and the 39 clean verdicts are
  kept.
  - `.claude/agents/water-judge.md` (`tools: WebSearch, WebFetch`; the `.claude` folder is not tracked).
  - **`make_judge_web_ua1.py`.** It reads the first launch's journals and transcripts and writes `ua1_web_kept.json`:
    the 39 verdicts whose judge was served `claude-sonnet-5`, called no tool but WebSearch, WebFetch, ToolSearch and the
    verdict, met nothing of this project and put no email address in a tool call. Not kept: AGO012↔COD019 and
    USA044↔MEX019 (the browser).
    It writes `judge_ua1_web2_canary.wf.js` (the test judge, ALB011↔ALB017, whose first judge used the browser and
    the shell; its verdict is a test) and `judge_ua1_web2_p1/p2/p3.wf.js`, 114 rows each. Their template adds one
    thing: `agentType: 'water-judge'`.
    Checked:
    - the 342 prompts are identical to the first launch's prompts for the same pairs;
    - the 342 and the 39 make the 381 of `ua1_web_queue.txt` exactly once;
    - model `claude-sonnet-5`, judge type `water-judge`;
    - the test judge's prompt is identical to its pair's in p1.

    The first launch's files stay as written.
  - **`extract_judge_web_ua1.py`.** It merges the 39 kept verdicts (`launch: first`) with the second launch's
    (`launch: second`) and lists any email address in a tool call. Tested with stand-in outputs: 381 verdicts, 39 + 342;
    the test file was removed.
  - **The test judge, run `wf_6782735d-9a6`: failed at once** ("agent type 'water-judge' not found"; 0 tokens). Judge
    definitions are loaded when a session starts, and the file was written during this one. Next: a new session, then
    the test judge on the maintainer's neutral "launch".
- **2026-10-03, the second launch.**
  - **The test judge.** It ran in a new session (worktree `charming-poincare-a77e24`) as run `wf_1d35d9ee-c09`, on
    the maintainer's "launch". That session's Workflow tool could not open the main checkout's files, so it ran a
    copy from its scratchpad; the copies of the test judge and of the three parts are byte-identical to the files here.
    Re-checked from its transcript:
    - served `claude-sonnet-5`;
    - offered only WebSearch, WebFetch and the verdict tool; 9 searches and 4 fetches, nothing else;
    - relayed message "launch"; the memory index 6,936 characters, with none of the words that described the run;
    - no email address in a tool call and no hit on this project;
    - verdict water-only, high, nothing flagged (a test, not used).
  - **The 342 rows.** This session was restarted, so the judge type loads here too. On the maintainer's "launch" the
    three parts were launched together:
    - `judge_ua1_web2_p1.wf.js`: run `wf_25c3eab0-c1e`;
    - `judge_ua1_web2_p2.wf.js`: run `wf_ffc37e1d-647`;
    - `judge_ua1_web2_p3.wf.js`: run `wf_517e041f-2ce`.

    The other session launched nothing more. Checked at once on the first three transcripts of each run: relayed
    "launch", model `claude-sonnet-5`, memory index 6,936 characters with none of those words.
- **2026-10-03, the second launch cut off; the continuation.**
  - **Cut off.** The session closed at 13:31 with 30 of the 342 answered (10 per part) and 18 cut off. All 48
    transcripts are clean: served `claude-sonnet-5` only, with 246 web searches, 209 fetches and 30 verdicts, and
    nothing else.
  - **Tokens per judge.** The maintainer asked whether a web judge uses more tokens per pair. Medians from the
    transcripts:

    | judges | rounds | input read | output |
    |---|---|---|---|
    | blind (440) | 1 | 63k | 25k |
    | web, second launch (30) | 5 | 121k | at most 26k |
    | web, first launch (41) | 8 | 604k | at most 39k |

    Each round re-reads the conversation so far; the web-only judge starts from a 12k context, the default judge from
    about 60k. A transcript logs a round's output count before the round is written, except the last round's, so the
    web judges' output is bounded by the growth of their context.
  - **Resume refused.** The Workflow tool resumes a run only in the session that launched it, and these runs' journals
    are in the closed session's folder.
  - **`make_judge_web_ua1.py`.** One `harvest()` now keeps a stopped launch's verdicts, by the same tests, for both
    launches. `ua1_web_kept.json` holds 69, each marked with its launch (first 39, second 30).
    It writes `judge_ua1_web2c_p1/p2/p3.wf.js` (the continuation), 104 rows each: each launched part less its kept
    rows.
    Checked:
    - the same labels, prompts, model, judge type and order as the launched parts;
    - kept plus continuation make the 381 exactly once;
    - the launched files are byte-identical to before.
  - **`extract_judge_web_ua1.py`.** It takes each kept verdict's launch from the file. Its arguments are the
    continuation's outputs and transcript folders.
  - **Launched on the maintainer's "launch"** in the new session (`c7642173`):
    - p1: run `wf_f8b9286f-699`;
    - p2: run `wf_d39ba2ec-822`;
    - p3: run `wf_69a79694-138`.

    Checked at once: relayed "launch", served `claude-sonnet-5`, judge type `water-judge`, memory index 6,936
    characters with none of the words that described the run.
  - **The continuation cut by the session usage limit** (reset 18:10). Answered: p1 50, p2 40, p3 44, so 134 of 312.
    The other 178 failed with "You've hit your session limit". All 312 transcripts are clean: served `claude-sonnet-5`
    only, with 432 web searches, 1,060 fetches and 134 verdicts, and nothing else. The runs were launched in this
    session, so they resume here: the 134 return from the record and the 178 run again, from the maintainer's neutral
    "launch".
  - **Resumed on the maintainer's "launch"** (same run ids, this session). The 134 verdicts return from the record and
    the 178 run again. Checked on the first new transcripts of each run: relayed "launch", served `claude-sonnet-5`,
    memory index 6,936 characters with none of those words.
- **2026-10-03, the web judgment's record.**
  - **Runs.** After the resume all three continuation runs finished with 104 of 104 each. The watcher saw 312 verdicts,
    served `claude-sonnet-5` only, and no alert.
  - **`extract_judge_web_ua1.py`.** Only a transcript that produced a verdict feeds that verdict's web record. A
    transcript cut off by the usage limit and run again is listed and checked, but is not part of any record.
    Result, `ua1_web_results.json`:
    - 381 verdicts: 69 kept (first launch 39, second 30) and 312 from the continuation; none void, none missing;
    - every judge served `claude-sonnet-5` and used only web search and fetch, 4 to 43 calls each (median 11);
    - no email address in a tool call;
    - the 178 cut-off transcripts show no hit on this project and no other tool;
    - verdicts: water-only 365, not water-only 16 (dry-land 15, undefined-line 1); confidence high 271, medium 109,
      low 1; needs_human 80.
  - **`population_ua1.py`.** Of the 381 rows:

    | before the web judgment | closed now | on the sheet now |
    |---|---|---|
    | on the sheet (145) | 79 | 66 |
    | closed by the blind record (236) | 184 | 52 |

    The 118 on the sheet, by the flags of their web record:
    - 16 the judgment not water-only;
    - 58 medium with needs_human;
    - 38 medium;
    - 5 needs_human;
    - 1 low with needs_human.

    The sheet is 286 rows (was 313): group 1 60, 2 4, 3 29, 4 5, 5 1, 6 12, 7 19, 8 156 (the 38 blind challenges
    and these 118), with X1 to X5 besides.
  - **The sheet rebuilt.**
    - `measure_ua1.py`: 282 border arcs; HydroRIVERS share at 500 m equal to the screen record's on 282 of 282.
    - `map_ua1.py` and `combine_maps_ua1.py`: `maps/ua1_286_pairs.pdf`, 291 pages (edges 282, non-touching 4, the
      five crossing flags after the rows).
    - `make_worksheet_ua1.py`: `ua1_ruling_worksheet_2026-10-03.md` and `GATE_ua1.md`, 286 rows (U1 to U286) plus X1
      to X5. Its introduction and group 8's note now say how the rows were judged again: blind, then with the web
      allowed.
    - `maps/ua1_313_pairs.pdf` and `maps/ua1_111_pairs.pdf` are earlier sheets' maps.
- **2026-10-04, the worksheet in two parts.** The maintainer, on the 286 rows: "286 pairs needing manual review?". By
  the deciding record:
  - 93 have a pass saying not water-only;
  - 19 are Standard M;
  - 12 are tier A with no human map check on record;
  - 162 have both passes saying water-only, flagged only by a confidence below high or a request for a human check.

  Asked how to work through them: "Two parts, same rule (Recommended)". The question counted 92 and 111/175, the split
  row (U99, the research not water-only) left out of the contested; it belongs to Part I, so the parts are 112 and 174.
  - `make_worksheet_ua1.py`:
    - Part I, decide row by row (112: groups 2, 3, 4, 5, 7, and group 8 where the latest record is a challenge),
      by group;
    - Part II, confirm by exception (174, both passes water-only, asserted), by the measurement (water within 500 m of
      every border point 74; longest dry run under 1 km 27; 1 to 5 km 44; 5 km or more 25; no border arc 4), then by
      label;
    - Part III, the crossing flags X1 to X5.

    The record cell now names the row's group and why the rule sends it. Labels and maps are unchanged (page n = Un);
    `GATE_ua1.md` is byte-identical; every row's other cells are unchanged.
- **2026-10-04, the rulings.** The maintainer: "Attached is my manual answers of Part 1 and X1-5, an no override for
  other pairs:", with the 117 answers, frozen verbatim in `ua1_maintainer_rulings_2026-10-04.txt`.
  - `rulings_ua1.py` → `ua1_rulings.json`. Each label and its units are checked against the gate.
    - Part I, 112 rows: 70 "Yes, their border is water-only."; 41 "No, their border isn't water-only."; U99
      BOL004↔PER007 "No, these units do not share a genuine boundary."
    - Part II, 174 rows: water-only ("no override for other pairs").
    - Crossings: X1, X3, X4 and X5 "Yes, there is at least one fixed crossing."; X2 Kayunga↔Mukono "No, there isn’t a
      fixed crossing."
  - The message's lines are also the body of `ua1_112_pairs_short_summary.docx`, which the maintainer placed in the
    folder with `ua1_partI_independent_adjudication_2026-10-04.txt`, the maintainer's reference while reviewing. Its
    tally, 54 yes / 41 no / 17 uncertain, is the docx's second line. The docx's 112 answers and five crossing answers
    equal the frozen rulings row for row. The reference is not a pass of the record (section 3.2): the rulings are the
    maintainer's.
- **2026-10-04, the ship (`ship_ua1.py`).**
  - The 244 rows ruled water-only keep their place and move to tier A. Each row's `source` gains a `ua1` clause: the
    deciding two-model record with both passes' verdicts and confidences, Standard M where it applies, and the ruling,
    row by row (70) or confirmed by exception (174).
  - The 17 first-round rows ruled earlier (`ru1` 7, `wu1` 10) move to tier A (the first entry of this log).
  - The 41 rows ruled not water-only leave the water file; their edges stay as ordinary edges (none is non-touching).
  - U99 leaves the water file, and BOL004↔PER007 joins `denylist_pairs.csv` (register #14).
  - The 63 `jr1` crossing clauses name Sonnet 5.5 ("served for the alias 'sonnet'") and add the Sonnet 5 re-run of
    2026-10-03. The five changed rows end "the passes now disagree; maintainer ruling 2026-10-04", with the flag ruled.
    X2's `has_bridge` becomes False.
  - `scripts/apply_overlays.py` regenerates the edge list and `water_separated_pairs.csv`.
  - **Counts:**
    - water file 866 → **824** rows (798 on edges, 26 non-touching) = 712 river (378 / 334) + 112 lake (10 / 102);
    - with a fixed crossing 418 → **388**, without 448 → **436**;
    - cross-border 374 → 344 (127 / 217), domestic 492 → 480 (261 / 219);
    - tiers 332 / 238 / 296 → **561 / 119 / 144**;
    - edges 8,463 → **8,462** (cross-country 1,798 → 1,797); moderate 8,015 → **8,026**; stringent 7,597 → **7,638**;
    - ADM0 326 / 320 / 300 → 326 / **321** / **307**. Roll-ups 26 (20 / 6) → **19** (14 / 5): CHN↔PRK, CMR↔GAB,
      DEU↔LUX, FIN↔SWE, GRC↔TUR, MOZ↔TZA and MRT↔SEN each keep an ADM1 border that is not water-only;
    - denylist 6 → 7.
  - **The 42 rows that left, by origin:** the first round 20 (19 river, 1 lake: Södermanland↔Uppsala); the 2026-07
    run 17; the 5 km re-screen 2 (Niassa↔Ruvuma, Shkodër↔Ulcinj); the HydroRIVERS cross-check 1 (Steiermark↔Pomurska);
    the HydroLAKES cross-check 2 (Ontario↔Minnesota, and U99). First-round rows 297 → 277.
- **2026-10-04, the checks after the ship.**
  - `build_all.py`: OK, with EXPECTED moved.
  - `population_ua1.py` in its after mode, which recognises the ship by the `ua1` clause (`ua1_population_after.txt`):
    - 824 rows: automatic 263, ruled 561, open 0;
    - tier A with no ruling: 0;
    - crossing flags: `ca1` A 45 / B 45 / C 270 / D 401; `jr1` 57 adopted, 1 split ruled, 5 ruled again here;
      departures 0.
    The gate and sheet files are unchanged (hashes).
  - Completeness 0 open in every band; attributability 824/824 (`completeness_ua1.txt`, `attributability_ua1.txt`).
  - `false_contacts/screens_false_contacts.py` re-run: seven contacts. La Paz↔Callao reproduces its row in the
    screens' record, as Franche-Comté↔Bern does.
  - The manuscript guard reads the Sonnet 5 crossing re-run and the rulings: 87 of 87.
  - Tests: the stated counts updated (`test_apply_overlays.py`, `test_adm1_pericoupling.py`, `test_pericoupling.py`).
    The tier-B test now pins the frame's rows that rest on no ruling: 119, the 102 ruled water-only being tier A. Full
    suite: 1,389 passed.
- **2026-10-04, the documents (once).**
  - **Manuscript and its companions:**
    - V3: the counts; the rule sentence without its incumbency clause; what the judges saw before October 2026; the
      seventh false contact;
    - its DOCX, rebuilt (`paper/build_ems_docx_v3.py`): every new figure present as often as in the Markdown, none of
      the earlier;
    - the revised Section 3 and the submission checklist.
  - **Repository documents:**
    - METHODS, PROVENANCE, REPRODUCING, INTRODUCTION and MANUAL (the default ADM0 count, 320 → 321);
    - CHANGELOG;
    - the edge-audit register (#14 resolved; the 2026-10-04 decisions);
    - the bridge methodology: the 2026-10-02 note's model, a 2026-10-04 note, §11's first-round count;
    - the engine and loader docstrings.
  - **The drafts EN/ZH:**
    - the counts; Tables 1 to 4;
    - Table 3 and the ladder sentences recomputed by `ladder_yield_ua1.py`, which runs
      `geodesic_distances/ladder_yield_gd1.py` unchanged into this folder: 2.5 km 475 of 690 shipped, 69%; all rungs
      553, 28%; the own river named for 436 of 475, the dominant one for 414; the other rungs unchanged;
    - 41 of the 689 shipped river borders on existing edges below the HydroRIVERS bar, every one nominated by the
      Natural Earth river ladder (the Rhône example left the set);
    - the rule sentence without its incumbency clause, with one dated sentence on this campaign;
    - the seventh false contact.
  - **Paper notes:**
    - the process notes EN/ZH: §4.1's first-round river rows, §6's crossing basis, §9.26, §10, §11, §12;
    - the reproduction ledger, the fact sheet and the supplement's status note;
    - the Chinese checklist: item 76.
  - **Corrected on the way:**
    - METHODS' rejected-nomination total read 2,786 where its parts sum to 2,827. The error was introduced in this
      campaign's first pass.
    - "Each row's `source` ends with its two-model verdict" no longer held: 244 rows now end with the `ua1` clause. It
      now reads "carries".
    - The process notes' "238 first-round river rows" was 237 before this campaign and is 218 now.
    - The scan's patterns were widened after a first pass. They had missed the crossing basis (422 / 285 / 96), the
      `jr1` rows' 62 + 1, the first-round and later rows (297 / 543) and the bare ADM0 count (320 in MANUAL).
  - **`doc_scan_ua1.py` → `doc_scan_ua1.txt`:** 489 mentions of an earlier figure, 55 of them outside a dated record.
    Each of the 55 was read; all are records or other quantities:
    - bullets and table rows that the scan cannot date (the fact sheet's campaign bullets, the ledger's campaign rows,
      the bridge methodology's status notes, PROVENANCE's waypoints);
    - the validation study's 238-row frame;
    - Table 3's first-nomination count at 15 km (374, unchanged);
    - an unrelated figure, the RAG corpus's 296.
- **2026-10-04, the second check of the documents.** The maintainer: "Please check the related documents to make sure
  that there isn't any stale or wrong information, and then commit and open PR."
  - **How.**
    - A scan of every document on text joined within paragraphs, so that a phrase broken across lines is still
      found. It covers the repository's documents, the paper notes, and the package's, scripts' and tests' code.
    - It looked for the figures of the state before the ship and their paraphrases, and for the names of the 42 rows
      that left the water-only set, in case any stays cited as a water-only example.
    - The diff of every tracked file was read.
    - METHODS' current sections and the manuscript's Section 3, limitations and conclusions were read end to end.
    - The new text's figures were recounted from the shipped data: Standard M 12 not water-only / 7 water-only; the
      ten lake rows with a fixed crossing, as METHODS lists them; the parts 112 / 174 by group; and the enumerations
      summed (origins 824; tier A 535 + 26).
  - **Found and corrected:**
    - PROVENANCE, two places: "six reviewed contacts" and "holds six reviewed entries" (now seven, with La
      Paz↔Callao). The roll-up sentence now reads "seven former roll-ups". The 2026-07 run's verification sentence
      now adds the `ua1` rulings. The wide-river paragraph now says that 11 of its 13 rows ship.
    - METHODS, two places: the census's "the 6 that meet along a line are the denylisted contacts" (now 7); and,
      after "Twelve borders were reclassified land→water", that ten of them ship.
    - `docs/FUTURE_WORK.md` §1 still read "recorded, no action. The pair ships unchanged" for BOL004↔PER007; it is now
      resolved.
    - A docstring in `tests/test_apply_overlays.py` (the fresh build: "the six denylisted contacts").
    - The drafts EN/ZH: "The 15 surviving rows ship" (13 since the rulings).
    - The process notes EN/ZH: the design invariant "humans rule on everything ship-affecting" and §5.2's
      auto-accept policy now state the rule without its exemption, and §5.2 notes when the judgment of 2026-10-03
      asks for a human check.
    - CHANGELOG: "incumbency clause" replaced by plain words; the web judges described as using web search and fetch
      only; FUTURE_WORK listed.
  - **Found, not changed:** the manuscript's program size, "approximately 33.1 MB". It matches the source tree with
    its `__pycache__` folders (34.1 MB today); without them the tree is 24.0 MB (23.4 MB tracked). It predates this
    campaign and is put to the maintainer.
  - **After the corrections:** the document and data tests pass (243), and `build_all.py` is OK. A re-run of the scan
    finds the earlier figures only in dated records and in other quantities.
