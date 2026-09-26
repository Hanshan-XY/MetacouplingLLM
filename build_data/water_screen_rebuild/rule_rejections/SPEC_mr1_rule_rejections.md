# Specification: the rule rejections adjudicated; the cross-type union at every ladder width (campaign `mr1`)

Status: **APPROVED, 2026-09-25** (maintainer: "Yes"). Frozen; later corrections are implementation defects only,
logged in section 11.

## 1. Origin

The manuscript reads: "A deterministic rule rejects 269 of the edges that are nominated only by a wider Natural Earth
river rung (5 to 20 km) and whose coverage is below 0.20 on all four layers." The rule is the July 2026 hybrid
auto-reject (`hybrid_autoreject_ledger.csv`, 288 rows; the "mechanism" rule): an edge nominated only by the Natural Earth
river ladder at 5, 10, 15 or 20 km, with coverage below 0.20 at every layer's base width (Natural Earth rivers 2.5 km,
Natural Earth lakes 125 m, HydroRIVERS 500 m, HydroLAKES 500 m), is rejected without any model looking at it.

The maintainer asked whether such an edge can be nominated at all, whether checking every width would be safer, and
whether the rule can miss a border that only the Natural Earth ladder sees (section 9). Measured on the current screen
record (`build_data/geodesic_distances/gd1_screens.csv`):

- The rule can miss such a border. Anzoátegui↔Bolívar (`VEN002<->VEN006`), a shipped water-only border on the Orinoco,
  was first nominated at the 10 km rung and its best coverage at any base width is 0.24 (HydroRIVERS): on a river that
  wide the modelled reach lies more than 500 m from a bank-line border. It reached the two models only because 0.24 is
  above 0.20. (Presidente Hayes↔San Pedro, `PRY016<->PRY017`, the Paraguay River: 0.29.)
- Of the 269 edges the rule rejects, 79 have a best base-width coverage below 0.05, 49 from 0.05 to 0.10, 66 from 0.10
  to 0.15 and 75 from 0.15 to 0.20.
- Against a miss so far: none of the 62 edges that satisfy the rule but were adjudicated by the two models for other
  reasons is water-only, and the preregistered validation study found no error in the 20 rule rejections it sampled
  (at most 13.9 percent, one-sided 95 percent). Neither rules a miss out; only adjudication does.

Checking the Natural Earth lake widths up to 1,500 m as well (option A) would have moved one edge out of the rule
(Vientiane↔Xaisômboun, `LAO013<->LAO016`: 0.14 at 125 m, 0.21 at 1,500 m) and leaves the risk above unchanged.
Requiring the Natural Earth river's narrower rungs to be low too (option B) would have moved 171, all near a river by
construction. The maintainer chose to drop the rule and adjudicate all 269.

The maintainer then asked what widening the cross-type union to the Natural Earth ladders would cost, noting that the
overlap of its nominations with the other screens is a result of today's data, not a property of the design, and asked
for the change to be added (section 4).

## 2. Population: 269 edges

The rows of `hybrid_autoreject_ledger.csv` that the current edge screens still nominate and for which the rule still
holds: nominated only by the Natural Earth river screen (`nominated_by == ne_river`), coverage at 2.5 km below 0.50, and
coverage below 0.20 at all four base widths (`ner2500`, `nel125`, `hr_all`, `hl`). 269 = 207 domestic + 62
cross-border (July batches b2 21, b3 85, b4 76, b5 80, b6 7). Computed live on `gd1_screens.csv` by `build_queue_mr1.py`
and frozen in `mr1_queue.csv`; the widened union of section 4 does not change it (all 269 are adjudicated in any case).

- None of the 269 carries a two-model verdict. This is checked against the campaigns' research inputs, not against
  membership of the July audit queue, which lists the rule rejections too (the screen records' `has_two_model_record`
  column counts that membership and so marks all 269 as reviewed; it is not used here).
- Every other nominated candidate carries one: the other 3,016 edges (779 accepted, 2,237 rejected; the 62
  rule-satisfying edges above among the rejected). The July 20 km holds, the second ledger, were all adjudicated as a
  tranche (`verdicts/holds20km.json`, 211 of 211).
- Not in the population: the other 19 ledger rows. Five are nominated by no current screen (not candidates); for 14
  the rule has lapsed and a later campaign adjudicated them.

## 3. Design: the standing two-pass, unchanged

1. **Research** on GPT-5.6 Sol in the maintainer's Codex run, fresh and blind: no prompt discloses a prior verdict, the
   campaign or a date; every verdict cites its sources. Prompts are the standing research prompts for existing edges,
   taken verbatim by AST as gd1 did (`arc_and_combined_screens/build_queue_ba1.py` prompt and nomination sentence;
   rubrics from `domestic_creek/make_handoff_dc1.py` and `rejection_unification/build_queue_rj2.py`), with the border
   geometry and screen shares from the gd1 arc cache. Outputs `mr1_research_input.jsonl` (269 lines) and
   `CODEX_INSTRUCTIONS_mr1.md`; the maintainer returns `mr1_research_results.jsonl`.
2. **Judgment** on Claude Sonnet 5, adversarial: it sees the research verdict and the screen coverages, is barred from
   the repository's prior verdicts, and tries to defeat each claim. The workflow is generated from `make_judge_gd1.py`
   (`make_judge_mr1.py`) and runs on the maintainer's "launch".
3. **Rubric**: the materiality standard. A named dry segment (surveyed or cadastral line, ridge, road, overland stretch)
   defeats water-only regardless of its share; the "about a fifth" figure is the passes' screening tolerance, never an
   acceptance licence; the answer defaults to not water-only when the evidence is uncertain.
4. **Gate** (`compile_gate_mr1.py`, as gd1): A both passes water-only, ruling needed; B the passes disagree, ruling
   needed; C both not water-only but flagged, for a glance; D both not water-only, closed.
5. **Rulings** frozen in `mr1_rulings.json` (`freeze_rulings_mr1.py`).
6. **An accepted border** is an existing edge reclassified water-only (nothing is added to the edge list): its row goes
   to the water file's later rows, verification tier A (maintainer map ruling), and its crossing flag comes from the
   standing four-layer pipeline (the crossing screen at 100 m on the ground, a GPT-5.6 Sol recheck where the screen does
   not settle, the location test, a maintainer ruling).

## 4. The cross-type union at every ladder width

Today the union counts a sample point as water when it lies within 2.5 km of a Natural Earth river, 125 m of a Natural
Earth lake, 500 m of any HydroRIVERS reach or 500 m of a HydroLAKES polygon, and nominates an edge at 0.80. It combines
the layers only at their base widths, although the two Natural Earth screens each run a ladder (rivers 2.5 to 20 km,
lakes 125 to 1,500 m).

**Change.** The union takes the widest rung of each Natural Earth ladder: 20 km for rivers, 1,500 m for lakes. A share
only grows with width, so the widest pair includes every narrower pair, and the union then combines the layers at every
width the ladders measure. HydroRIVERS (all reaches, its widest tier) and HydroLAKES keep their single 500 m width; the
bar stays 0.80.

**Measured on the gd1 record** (the per-point distances are already stored: rivers to 25 km, lakes to 1,500 m; no new
geometry and no download):

- union nominations 1,144 → 2,045 (1,235, 1,470 and 1,724 with the two widths stepped up together to 5 km / 250 m,
  10 km / 500 m and 15 km / 1,000 m); accepted among them 723 → 744 (Table 2's union row);
- one new candidate: Plužine↔Šavnik (`MNE015<->MNE018`, Montenegro), 0.36 within 20 km of a Natural Earth river and
  0.48 within 500 m of HydroRIVERS on different stretches, 0.82 together (0.48 on today's union). It already carries a
  two-model verdict, not water-only, and needs no adjudication. Candidates 3,285 → 3,286;
- widening the lakes alone adds no candidate.

The overlap is a property of today's data. An edge the widened union adds must stay below the Natural Earth ladder's own
bar and every other screen's bar yet reach 0.80 in total, which needs its water split across layers; with other data the
union may add more, and every such nomination goes through the two-pass like any other.

**Code.** `run_screens_mr1.py` computes the screens from the gd1 arc cache with `run_screens_gd1.py`'s share functions,
the union's two Natural Earth widths as named constants, and writes `mr1_screens.csv` (the columns of `gd1_screens.csv`).
From then on it is the current screen record: the completeness check and the manuscript guard read it.

## 5. The rule, afterwards

The rule is dropped. `rejection_unification/completeness_check.py` stops treating the ledger as a closure: its rule
exemption (the block that keeps a ledger row closed while the rule holds) is removed, the ledger rows and the audit
queue's ledger rows no longer count as records, and `mr1_research_input.jsonl` counts once `mr1_rulings.json` is frozen.
The ledger file stays as the July record. Every nominated candidate then carries a two-model verdict: 3,286 edges and
the 242 census pairs. No threshold rejects a candidate on its own.

## 6. Expected effect

- The union change alone changes no data (its one new candidate is already not water-only).
- Every verdict of the 269 "not water-only": no data change; the documents change once (section 7).
- k borders accepted: water-only 803 → 803 + k, moderate and stringent by their crossing flags; one data pull request
  with the documents.

## 7. Documents (once, after the campaign)

The manuscript (the rule sentence gives way to "every nominated candidate is adjudicated"; the union's definition;
Table 2's union row, 1,144 / 723 → 2,045 / 744, and any-screen row, 3,285 → 3,286 nominated; the validation paragraph's
sentence on the rule, to be proposed with the documents), the drafts EN/ZH and the Chinese checklist, METHODS,
PROVENANCE, the process notes, the reproduction ledger, the fact sheet, CHANGELOG, a status note in
`docs/VALIDATION_STUDY.md` (the study sampled the rule while it was in use), and the manuscript guard (the rule claim
replaced by the adjudicated count; its Table 2 claims read `mr1_screens.csv`).

## 8. Verification

1. `mr1_screens.csv`: union 2,045 nominated, 744 accepted; any screen 3,286; the one new candidate carries a two-model
   verdict; every column other than `union` and `nominated_by` equal to `gd1_screens.csv`; attributability unchanged
   (every shipped water-only row nominated).
2. `mr1_queue.csv` and `mr1_research_input.jsonl`: 269 rows each, equal to the population computed live.
3. Research results: 269 valid lines against the response schema; error lines, if any, are researched again.
4. Judgment: 269 results.
5. Gate, rulings, and the completeness check with the exemption removed, on `mr1_screens.csv`: 0 open.
6. With a data change: `build_all.py`, a full regeneration, the full test suite and the manuscript guard.

## 9. Maintainer (2026-09-25, verbatim)

- "There is "Every nominated candidate is disposed of in one of two ways. A deterministic rule rejects 269 of the edges
  that are nominated only by a wider Natural Earth river rung (5 to 20 km) and whose coverage is below 0.20 on all four
  layers.". If the  coverage is below 0.20 on all four layers, will it be nominated?"
- "Is that safer to change the base length to all ladders?"
- "1. If choosing option A, we might miss candidates only in NE ladders but not in Hydrorivers, right? 2. What are the
  other 3,016 candidates?"
- Ruling on "How should the 269 edges the deterministic rule rejects be handled?": "Adjudicate all 269 (Recommended)".
- "Before the approve, I have a quick question: What's the cost if change rules of Cross-type union that expand wider
  rungs with all ladders of NE rivers/lakes?"
- "You means only increase on candidates? But the change will make our screen cover most possible candidates."
- "answer my question first: What's the costs of covering all ladders besides revising documents?  I know most
  candidates are overlapped, but it's the results after sceens, right?"
- "Yes, please add it to the spec"

## 10. Record

`build_data/water_screen_rebuild/rule_rejections/`: this specification, `run_screens_mr1.py`, `mr1_screens.csv`,
`build_queue_mr1.py`, `mr1_queue.csv`, `mr1_research_input.jsonl`, `CODEX_INSTRUCTIONS_mr1.md`, the research results,
the judgment workflow and its output, the gate, the maintainer's rulings, `mr1_rulings.json`.

## 11. Change log

- 2026-09-25 draft.
- 2026-09-25 section 4 (the cross-type union at every ladder width) added at the maintainer's request, before approval.
- 2026-09-25 approved as drafted ("Yes"). Frozen.
- 2026-09-26 implementation detail (sections 4 and 8.1): the screen record also carries `nominated_by_ge10`, the
  diagnostic variant with HydroRIVERS reaches of at least 10 m³/s; it follows the widened union too (832 edges change), so
  section 8.1's "every column other than `union` and `nominated_by`" reads "other than `union`, `nominated_by` and
  `nominated_by_ge10`". `mr1_screens.csv` written and checked as specified (`mr1_screens_report.txt`).
- 2026-09-26 research handoff written (`build_queue_mr1.py`): 269 edges (207 domestic, 62 cross-border), none with a
  two-model verdict; `mr1_queue.csv`, `mr1_research_input.jsonl` (269 prompts, the standing text, no leak word) and
  `CODEX_INSTRUCTIONS_mr1.md`, which adds one line to the standing instructions: work from the prompts and the web only,
  opening no other file in the folder or the repository (the folder holds this specification).
- 2026-09-26 research returned ("Codex done"): 269 of 269 valid against the response schema, no error line
  (`validate_research_mr1.py`, `validate_research_mr1.txt`); water-only on none (confidence high 256, medium 13).
  Judgment workflow `judge_mr1_edges.wf.js` generated by `make_judge_mr1.py` (269 agents, Claude Sonnet 5; the standing
  templates verbatim; lake tests on 17), to run on the maintainer's "launch".
- 2026-09-26 judgment returned ("Go"; 269 agents, no error): water-only on none; gate A 0 / B 0 / C 22 / D 247
  (`GATE_mr1.md`); maintainer: "No override". Two observations, neither changing a verdict: the gate's prior-citation
  flag (22) caught the judges' own statements that they consulted no ledger or prior audit, not a prior verdict; and 178
  judges noted that the standing nomination sentence ("within a few kilometres of a named Natural Earth river
  centerline") understates the distance for edges first nominated at 5 to 20 km. They weighed the stated shares instead;
  the sentence errs toward water, and both passes rejected every edge. Recorded as a limitation of the standing prompt
  for wide-rung nominations.
- 2026-09-26 rulings frozen (`freeze_rulings_mr1.py`, `mr1_rulings.json`): 269 not water-only; no shipped row changes.
  The completeness check reads `mr1_screens.csv` and no longer lets the rule close a ledger row: 803 of 803 shipped rows
  with a record, the 283 nominated ledger rows each with a two-model record of its own, 0 open in every band
  (`completeness_mr1.txt`); attributability 803 of 803 on the mr1 record (`attributability_mr1.txt`).
- 2026-09-26 documents (section 7): the manuscript (the union's definition; Table 2's union row 2,045 / 744 and any-screen
  row 3,286; "Every nominated candidate, 3,286 edges and the 242 census pairs, is adjudicated ..."; the validation
  paragraph's sentence on the rule removed, the rule no longer being part of the method) and its guard (screen record
  `mr1_screens.csv`; the rule claim replaced by "every nominated edge adjudicated", checked against
  `records_mr1.two_model_records`; the rule's validation claim removed; 60 of 60); the revised Section 3; the drafts EN/ZH
  and Chinese checklist item 70; the process notes (§4.7, §5, §9.21); the reproduction ledger; the fact sheet; METHODS
  (§8's current description and an `mr1` record), PROVENANCE, REPRODUCING (the union's widths), a status note in
  `docs/VALIDATION_STUDY.md`, CHANGELOG. DOCX rebuilt, QA pass; documentation tests pass.
- 2026-09-26 stale check after the pull request (maintainer: "Can you check if there is any stale or wrong information in
  the docs?"): four statements the widened union had made stale, corrected. `docs/FUTURE_WORK.md` §4 named
  `gd1_screens.csv` as the current record and gave Chiba↔Tokyo a union share of 0.44 (0.53 on the mr1 record; still below
  the bar, none of the four omissions is nominated); §5 said a union bar of 0.70 or 0.90 changes no nomination, measured
  on the base-width union (on the widened union 0.70 adds 19 candidates and 0.90 drops Plužine↔Šavnik); REPRODUCING §6
  gave only `run_screens_gd1.py` for the current record; METHODS' mr1 record now names `mr1_screens.csv` as the edge
  screens' record. The supplement gains a status note.
- 2026-09-26 the limitation recorded above, fixed (maintainer: "Fix the "few kilometres" prompt wording"): the standing
  nomination sentence moves to `build_data/water_screen_rebuild/standing_prompts.py`, whose Natural Earth river clause states
  the width at which the edge reached the bar (over the 3,286 nominated edges: 2.5 km 688, 5 km 188, 10 km 337, 15 km 373,
  20 km 384; every other clause unchanged). The ba1 original stays in `build_queue_ba1.py` as the record of the ba1, ba1r,
  gd1 and mr1 prompts.
- 2026-09-26 the lake clause likewise (maintainer: "Yes, fix the lake clause too"): it states the Natural Earth lake width
  at which the edge reached the bar instead of "within a short distance" (over the 3,286 nominated edges: 125 m 137,
  250 m 4, 500 m 7, 1,000 m 11, 1,500 m 8 = the 167 lake nominations; nothing else changes).
- 2026-09-26 the union clause likewise (maintainer: "Yes, fix the union clause too"): it states the union's widths,
  20 km of a Natural Earth river, 1,500 m of a Natural Earth lake, 500 m of a HydroRIVERS reach or a HydroLAKES polygon
  (taken from the ladders' widest rungs and the hydrography width, not written in), instead of "within reach of some
  river or lake feature"; it applies to the one edge the union alone nominates, Plužine↔Šavnik.
