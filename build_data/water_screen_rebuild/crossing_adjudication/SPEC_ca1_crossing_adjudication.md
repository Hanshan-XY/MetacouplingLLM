# Specification: every fixed-crossing flag adjudicated by the two-model design (campaign `ca1`)

Status: **APPROVED, 2026-09-27** (maintainer: "Yes, switch to option 2, and approve the draft after checking again if the
design align with water-only judgment"), after the alignment check logged in section 12. Frozen; later corrections are
implementation defects only, logged in section 12. Nothing below has been run against the shipped rows except the
read-only measurement of section 2.

## 1. Origin

Reading the manuscript, the maintainer asked which models decide whether a water-only pair has a fixed crossing, and
whether that decision uses the same two-model design as the water-only judgment (section 10). The water-only status of
every shipped row rests on that design: GPT-5.6 Sol research, then Claude Sonnet 5 adversarial judgment, then maintainer
rulings. The crossing flag (`has_bridge`, which the default `moderate` view reads) does not: the four-layer procedure's web
verification (layer 2) and adversarial recheck (layer 3) were run differently depending on when a row entered the table,
and 540 of the 805 flags have no GPT-5.6 Sol input at all. The maintainer asked for the flags to be adjudicated with
the same logic as the water-only judgment. The precedent is the water-only re-adjudication of shipped rows (river
unification `ru1`, `river_unification/RIVER_UNIFICATION_SPEC.md`; water unification `wu1`), with the judge's evidence
rule as it stands since the rejection unification (`rj2`), and this specification follows it step by step (section 4).

## 2. What is on record (read-only, 2026-09-27; `crossing_evidence_ca1.py`, `crossing_evidence_ca1.txt`)

Layer 1, the OpenStreetMap screen, has been run over every row (the whole-table runs of 2026-09-16 and, on the ground at
100 m, 2026-09-24; the two rows of 2026-09-27 by `corridor_census_exact/crossing_cc1.py`). Layers 2 and 3 differ by group:

| Group (from the row's `source`) | Rows | Layer 2, web verification | Layer 3, adversarial recheck | Both passes' crossing answers on record |
|---|---|---|---|---|
| First round, June 2026 | 298 | multi-agent web check; its agent models are not recorded | agents of the same run, on disagreements with the June screen | 59 |
| July 2026 run | 387 | Claude Sonnet 5 web-research agents on every row but the three lake-shore contacts, which need none (`verdicts/bridge_*.wf.js`, `model:'sonnet'`); the July research pass on GPT-5.6 Sol had no crossing question (none of `codex_handoff/codex_research_results*.jsonl` carries a crossing field) | Claude Sonnet 5, on every disagreement with the July screen (`verdicts/recheck_*.wf.js`); Sonnet 5 coordinate lookups and disambiguation (`coords_*`, `disambig_*`) | 4 |
| Earlier checks folded in (widening, HydroRIVERS and HydroLAKES folds, near-miss and lake band) | 51 | their own records | their own records | 30 |
| Since September 2026 (rj2, dc1, census rows, pr1, cc1) | 69 | both passes, as one item of the water-only prompt (`has_fixed_crossing` yes/no/unknown, no coordinates) | GPT-5.6 Sol, where the screen and the two passes disagree (`bridge_screen_rj2.py`, `bridge_screen_dc1.py`); for the two rows of 2026-09-27, the maintainer's ruling | 69 |
| **Total** | **805** | | | **162** |

On top of this, the whole-table runs sent the rows the screen did not settle, unless a recheck on record answered them,
to a GPT-5.6 Sol recheck (131 rows on 2026-09-16; the way-class follow-up of 2026-09-20; the run of 2026-09-24). With
the September campaigns' rechecks, 154 rows carry a GPT-5.6 Sol crossing recheck. So:

- 162 rows carry both passes' crossing answers; 103 carry neither but a GPT-5.6 Sol recheck; **540 carry no GPT-5.6 Sol
  input**: for 330 July rows the only model behind the flag is Claude Sonnet 5, and 210 rest on the June or earlier
  checks.
- The answers on record are weighed differently from row to row. Of the 162 rows with both passes' answers, 132 agree with
  the flag, 28 include an "unknown", and two contradict it in both passes. Flevoland↔Utrecht (`NLD002<->NLD010`,
  IJsselmeer) ships without a crossing although both passes said yes; the whole-table screen agreed with the flag, so no
  recheck weighed their answer. Ayacucho↔Junín (`PER005<->PER012`, Mantaro) ships with one although both passes said
  no; a GPT-5.6 Sol recheck and the screen overrode them.
- Past whole-table runs changed few flags: 14 of 807 on 2026-09-16, one on 2026-09-20, one of 803 on 2026-09-24.

## 3. Population

All 805 shipped water-only ADM1 rows of `data/water_classification_pairs.csv`: 779 on a shared edge and 26 between
non-touching units. The 26 ADM0 roll-ups are derived, not adjudicated (`scripts/apply_overlays.py`: a country pair is
bridged when any of its ADM1 crossings is) and follow the ADM1 flags.

## 4. Design: the water-only re-adjudication logic, applied to the crossing flag

| Step | Water-only re-adjudication (ru1, wu1) | This campaign |
|---|---|---|
| Population | shipped rows, chosen deterministically | all 805 shipped rows |
| Research | GPT-5.6 Sol, "RESEARCH pass (1 of 2)"; fresh, prior verdicts not disclosed; sees the screen context; a RUBRIC block in each prompt; cites sources | the same, on the crossing question |
| Verdict | committed: `water_only` true/false with a confidence; "Default to water_only=false when uncertain" | committed: `has_fixed_crossing` yes/no with a confidence; "no" when uncertain |
| Judgment | Claude Sonnet 5, adversarial; sees the research verdict, the rubric and the screen context; told what the row currently ships and marks `needs_human` on an answer that would change it (ru1, wu1); `objection` is the single best reason for a negative answer, `none` for a positive one (wu1); no web (wu1 on) and no repository or prior-verdict files (the evidence rule since rj2) | the same, plus the location test |
| Gate | A both passes challenge the shipped value; B split; C both confirm but flagged, including a deterministic pre-flag (wu1: a thin arc); D clean confirmation | the same buckets; the pre-flag is the screen contradicting the confirmed answer |
| Rule | "EVERY ship-affecting disagreement or medium/low-confidence verdict goes to the maintainer ... flips ship only on the maintainer's ruling" | the same |
| Record | a clause citing the re-adjudication in the `source` of every row it re-checked (wu1: all 91 still shipped) | a note on every row (section 6) |

1. **Layer 1, the screen (unchanged, not re-run).** The record of 2026-09-24 (`crossing_width/cw1_screen.csv`,
   `cw1_ways*.csv`; 803 rows) and of 2026-09-27 (`corridor_census_exact/crossing_cc1_pairs.csv`, `crossing_cc1_ways.csv`;
   2 rows): whether an open bridge way lies within 100 m of both units, measured on the ground; every way within 1 km of
   both units with its class, name and two distances; for bridged rows the tunnel, dam-top and dyke ways. The flags stay
   pinned to the OpenStreetMap snapshots of June and September 2026, as the manuscript states. The screen is the
   deterministic context both passes see, as the water-only screen shares are.
2. **Research on GPT-5.6 Sol**, in the maintainer's Codex run, in the water-only research pass's form
   (`water_unification/make_water_unification_handoff.py`): the header "You are the RESEARCH pass (1 of 2) ... A second,
   independent adversarial pass will try to refute your verdict, so report what the evidence supports and mark
   uncertainty honestly rather than guessing", verbatim, with its subject changed to the fixed crossing; the pair
   (names, codes, ISO, domestic or international) and the row's water type and water body, which this campaign does not
   revisit; the screen context (the OpenStreetMap ways the screen found within 100 m of both units, with class and name,
   or none; the nearest bridge way's two distances; tunnel and dam-top ways), followed by the caution the
   water-unification prompts attach to their hint, "do NOT treat that as a verdict". The question is the recheck question
   of the whole-table runs, verbatim (`bridge_unification/make_bridge_recheck_bu1.py`, taken by AST and asserted): a road
   or rail bridge, causeway, dam-top road, culverted public road or tunnel, open to traffic, that crosses the water on the
   shared limit; name it, give its approximate coordinates, state the evidence that it links these two units; default to
   "no" when uncertain. The response schema is the recheck's (`structure`, `location` with approximate lat,lon, required
   for a "yes", `in_both_units`, `open_to_traffic`, `confidence`, `reason`, `sources`), with one change that the water-only
   logic requires: `has_fixed_crossing` is yes or no, never "unknown", as `water_only` is true or false. No prompt or
   input line discloses the shipped flag, a prior verdict, the campaign or a date. Each prompt ends, as the water-only
   prompts do (`RUBRIC_COMMON`), with a RUBRIC block: (1) what counts as a fixed crossing, the recheck's conventions
   verbatim (a road or rail bridge, causeway, culverted public road, dam-top road or tunnel, structurally complete and
   open to traffic, lying in both units; ferries, fords and footbridges never count; under construction and proposed
   never count; a completed bridge on a politically closed border counts; on very wide lakes a "no" is expected; on a
   small stream a numbered road crossing the stream on the unit line counts even where OpenStreetMap maps no bridge way),
   with the water-only rubric's note that dam-top roads count and are easy to miss on reservoirs; (2) the water-only
   rubric's source rule, verbatim: "Prefer official/authoritative geography (boundary treaties, IBS studies, national
   mapping agencies, UN/OCHA maps, established encyclopedic sources). An OSM-only absence proves nothing: OSM and the
   World Bank polygons are independently digitized datasets."; (3) its default, on the crossing: "no" when uncertain,
   "and say so in `reason`". The instructions repeat the rubric and add the mr1 line: work from the prompts and the web
   only, opening no other file. Four input files of about 200 lines.
3. **Layer 4, the location test (unchanged):** `crossing_width/cw1_location.py` on every "yes" with coordinates: located
   (within 500 m of the shared arc or 750 m of both units, on the ground), bank-line gap (750 m to 3 km), off-border
   (farther), or no coordinates. Its outcome is deterministic context for the judge and a flag for the gate.
4. **Judgment on Claude Sonnet 5**, adversarial, on the maintainer's "launch", in the water-only judges' form. It sees the
   research verdict (every field), the screen context and the location test's distances and outcome, and, as the
   water-unification judges were told "This pair currently SHIPS as water-only with water_type=...", it is told what the
   row currently ships (with or without a fixed crossing). Like the water-only judges it works without the web (its
   prompt says so), under their evidence rule (`NO_REPO` of `corridor_census_v2/make_judge_nt4.py`, verbatim), which bars
   repository files and prior audit records; the research pass is told nothing of the flag. It is given the research
   pass's RUBRIC block, as the ru1 judges were given theirs. It tries to defeat the research's answer: a structure that
   is not open to traffic, lands in a neighbouring unit or crosses a reach inside one unit; a ferry, a ford or a
   footbridge; a structure under construction; or a "no" that an open road or rail way in both units contradicts. As the water-only judge must name any dry component, it must name the structure it accepts or
   the defect it finds. Response: `has_fixed_crossing` (yes or no, "no" when uncertain), `structure`, `objection` (as the
   water-only judges give it for a negative verdict, the single best reason for a "no": not-open, not-in-both-units,
   not-on-the-shared-limit, ferry-ford-or-footbridge, under-construction, no-evidence, other; `none` for a "yes"),
   `confidence`, `needs_human`, `reason`. As in ru1 and wu1, `needs_human` is true when its confidence is low or its
   answer differs from what the row ships: such an answer would change shipped data and gets the maintainer's review. Generated by `make_judge_ca1.py`, one workflow per research file.
5. **Gate** (`compile_gate_ca1.py`), the water-only re-adjudication buckets, per row (the shipped flag F, research R,
   judgment J):
   - **A, both passes challenge the flag** (R = J ≠ F): a ruling.
   - **B, split** (R ≠ J): a ruling.
   - **C, both confirm the flag but a flag is raised:** either confidence below high; either `needs_human`; the judge cites
     a prior verdict; the two passes name different structures; or the deterministic pre-flag, as wu1's thin arc: the
     screen contradicts the confirmed answer (on a confirmed "no", an open road or rail bridge way within 100 m of both
     units; on a confirmed "yes", no such way and a structure the location test does not locate). The maintainer checks
     it.
   - **D, clean confirmation:** both confirm the flag at high confidence with no flag raised. Closed.
   The gate page shows the maintainer each row's record (earlier crossing answers, rechecks and rulings) and flags a
   proposed change that would reverse an earlier maintainer ruling. The models never see it.
6. **Rulings:** maintainer map rulings on A, B and C, with a map page per row as the recent water-only gates had (ba1,
   gd1, cc1; the shared border, the structure's coordinates, the screen's ways), quoted verbatim and frozen in
   `ca1_rulings.json`. A flag changes only on a ruling.
7. **Per-row record:** `ca1_verdicts.csv` (pair, research, judgment, location, screen, bucket, ruling, final flag).

## 5. Defaults

- Both passes commit to yes or no and answer "no" when uncertain, as the water-only passes answer "not water-only";
  uncertainty shows in the confidence, which sends the row to C.
- A flag never changes on a model answer alone: A and B need a ruling, and C a check.

## 6. Data change and documents

- **Data:** every row's `source` in `data/water_classification_pairs.csv` gains one note with its two-model crossing
  verdict (the two answers, the structure, the confidences, the location test) and its outcome: a clean two-model
  confirmation, or the maintainer's ruling (decision 3). `has_bridge` changes only on ruled rows. The campaign record
  holds the full verdicts, and PROVENANCE says so. The derived files carry no `source` and change only where a flag
  changes. `apply_overlays.py` regenerates the shipped files; `build_all.py` EXPECTED moves with the flags
  (`water_bridge`, `water_nobridge`, `adm1_moderate`; ADM0 if a country pair's roll-up changes). Verification tiers
  describe the water-only evidence and do not change. One data pull request with the documents.
- **Documents, once:** the manuscript's crossing paragraph (layers 2 and 3 become the two-model design; the way-class
  sentence recomputed on the new record) and its guard; METHODS (the `has_bridge` paragraph and the four-layer
  description, whose layer 2 is today "the research pass of the campaign that nominated it", which the July rows
  contradict; a `ca1` record); PROVENANCE; REPRODUCING; a status note in `docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md`;
  CHANGELOG; the drafts EN/ZH and the Chinese checklist; the process notes EN/ZH (section 6, whose layer 2 is today "the
  2026-07 and later GPT-5.6 Sol research passes", and a section 9 record); the revised Section 3; the reproduction ledger;
  the fact sheet; a status note in the supplement.

## 7. Expected effect and effort

- Research: 805 prompts in four files for the maintainer's Codex run. Judgment: 805 Claude Sonnet 5 agents in four
  workflows, each run on the maintainer's "launch" once its research file has been validated.
- Rulings: A and B are rulings, C a check. Of the 162 rows with both passes' answers on record, 30 differ from the flag
  (28 of them through an "unknown"), about one in five, so up to about 150 A and B rows is possible. C adds every
  confirmation below high confidence; the answers on record cannot measure it (their crossing item had no confidence of
  its own), so the gate counts it when the research returns.
- Flag changes: the whole-table runs changed 14, 1 and 1 flags; a handful is expected here.

## 8. Checks

1. Research input: 805 lines, one per shipped row; question, schema and conventions equal to the recheck's, and the
   RUBRIC block's source rule and default wording equal to the water-only rubric's (asserted);
   leak check: no line contains the flag, a prior verdict, a campaign code, a date, or words such as "shipped", "flag",
   "recheck", "ledger"; each line holds only `id`, `prompt`, `response_schema` and `batch`; the screen context equals
   the screen record.
2. The location test reproduces the outcomes of the rechecks on record (`crossing_width/cw1_location_ground.py`: no
   change).
3. Research results: 805 valid lines against the schema; `has_fixed_crossing` yes or no; every "yes" carries
   coordinates; error lines are researched again.
4. Judgment workflows syntax-checked and tested on synthetic results before the "launch"; 805 results.
5. A consistency report, not used by the gate: the new answers against the answers on record (the 162 two-pass rows, the
   154 rows with a GPT-5.6 Sol recheck, the July Sonnet 5 answers).
6. After the data change: `build_all.py`, a full regeneration, the full test suite, the manuscript guard, the completeness
   and attributability checks.

## 9. Decisions (maintainer, 2026-09-27)

The water-only logic settles the defaults, what each pass sees, what the judge is told and its lack of web access.
The four choices, as approved:

1. **Population:** all 805 rows.
2. **Layer 1:** the record of 2026-09-24 and 2026-09-27, not re-run.
3. **Source strings:** a note on every row (option 2; maintainer: "Yes, switch to option 2").
4. **The strength of each crossing verdict** (a clean two-model confirmation, or a ruling): in each row's note
   (decision 3) and the campaign record, with PROVENANCE stating it; no new column.

## 10. Maintainer (2026-09-27, verbatim)

- "I'm reading the manuscript. What's model used  in fixed crossing judgment? still GPT 5.6 Sol and Sonnet 5?"
- "Are you sure? We didn't use the 2 model run?"
- "Is that possible to use same 2 model run with water only judgement?"
- "Yes, please draft the specification"
- "I wanna use the same logic with water only border judgement. Do your suggestion meet the requirements?"
- "Explain "Edit only the rows you rule on, or add a note to every row.""
- "Yes, switch to option 2, and approve the draft after checking again if the design align with water-only judgment"

## 11. Record

`build_data/water_screen_rebuild/crossing_adjudication/`: this specification, `crossing_evidence_ca1.py` and its output,
`build_queue_ca1.py`, `ca1_queue.csv`, `ca1_research_input_b*.jsonl`, `CODEX_INSTRUCTIONS_ca1.md`, the research results,
`validate_research_ca1.py`, `make_judge_ca1.py` and the judgment workflows with their output, `extract_judge_ca1.py`,
`compile_gate_ca1.py`, `GATE_ca1.md`, `ca1_gate.json`, the maintainer's rulings, `freeze_rulings_ca1.py`,
`ca1_rulings.json`, `ca1_verdicts.csv`, `ship_ca1.py`.

## 12. Change log

- 2026-09-27 draft.
- 2026-09-27 revised before approval to follow the water-only re-adjudication logic (maintainer: "I wanna use the same
  logic with water only border judgement"). The first draft differed from it in five places: the research did not see
  the screen (the water-only research sees the screen context); the answers could be "unknown" (the water-only
  verdicts are committed, defaulting to the negative); a row closed on any agreement with its flag (the water-only gate
  closes only a clean, high-confidence confirmation and sends every medium or low-confidence verdict to the
  maintainer); its buckets were its own (now ru1's and wu1's A to D); and the judge was not told what the row ships
  (the water-unification judges were told each row's shipped value and flagged an answer that would change it).
- 2026-09-27 decision 3 switched to option 2 (maintainer: "Yes, switch to option 2"): every row's `source` gains the
  crossing verdict's note, as the water-only re-check of 2026-07-25 cites itself in all 91 of its rows still shipped.
- 2026-09-27 second alignment check, before approval (maintainer: "... after checking again if the design align with
  water-only judgment"), against the water-only re-adjudication's scripts (`river_unification/`
  `make_river_unification_handoff.py` and `make_judge_wf_ru1.py`; `water_unification/`
  `make_water_unification_handoff.py`, `make_judge_wf_wu1.py` and `compile_gate_wu1.py`). Four further alignments:
  (1) the rubric sits in each research prompt as a RUBRIC block, as `RUBRIC_COMMON` does, with its source rule and
  default wording verbatim, and the judge receives it, as the ru1 judges did; (2) `objection` is the single best reason
  for a negative answer and `none` for a positive one, as in wu1 (the draft's `contradicted-by-screen`, a reason for a
  "yes", is dropped); (3) C's deterministic pre-flag, as wu1's thin arc, is the screen contradicting the confirmed
  answer in either direction (the draft covered only a confirmed "yes"); (4) the rulings come with a map page per row,
  as the recent water-only gates had (ba1, gd1, cc1). Found aligned: the two models and their roles; the research
  blind to prior verdicts, framed as pass 1 of 2, with the screen context and its caution, committed answers, the
  negative default
  and cited sources; the judge adversarial, without the web, under the evidence rule, told what the row ships and
  marking an answer that would change it; the gate A to D with the clean-confirmation rule; rulings on every
  ship-affecting or uncertain answer; a note on every re-checked row. Outside the per-campaign logic and not included:
  a validation study of the D closures (the water-only tier B was measured once, by the preregistered study of July
  2026).
- 2026-09-27 approved; frozen.
- 2026-09-27 research handoff written on the maintainer's "Yes, build the research handoff" (`build_queue_ca1.py`):
  `ca1_queue.csv`, `ca1_research_input_b1..b4.jsonl` (202 / 201 / 201 / 201 prompts; 438 domestic, 367 cross-border)
  and `CODEX_INSTRUCTIONS_ca1.md`. The question and the definition are asserted against bu1's script, the header and
  the rubric's dam-top note, source rule and default against the water-unification script. The screen context
  agrees with the screen record on every row: 392 rows have an open bridge way within 100 m of both units (349 of
  the 408 shipped with a crossing, 43 of the 397 without), 230 have none within 1 km. Implementation corrections:
  - Section 4.1's tunnel, dam-top and dyke ways stay out of the research context. Layer 1 queried them only for rows
    shipped with a crossing (`cw1_screen.py`), so their presence would disclose the flag. The judge receives them
    (`ca1_queue.csv`, `extra_within_100m_judge_only`), since it is told the flag anyway.
  - One water-body text names a crossing: Cavan↔Westmeath (`IRL002<->IRL024`), "... the River Inny (past Finnea
    Bridge) to Lough Kinal". Its prompt copy drops the parenthesis; the data is unchanged. The dam names in six other
    texts are the reservoirs' own names and stay.
  - The leak check allows the definition's verbatim phrase "(the flag is structural)", which says nothing about a
    row. Its bare-year rule reads the prompt outside the screen-context line, where four-digit numbers are
    OpenStreetMap road numbers used as way names ("2033", "2034"); the full-date rule reads the whole prompt. No hit
    remains.
  - The pair line reads "a domestic" or "an international first-level border" (bu1's sentence had "a international").
- 2026-09-28 research returned (maintainer: "Codex done"): 805 answers in four files. Validation
  (`validate_research_ca1.py`): every id answered once, every field valid, except seven "yes" answers whose
  coordinates have two decimals (about 1 km), which the location test's pattern (three decimals) cannot read and which
  are too coarse for its 500 m and 750 m thresholds. As section 8.3 provides, they are researched again
  (`make_redo_ca1.py`: `ca1_research_input_redo.jsonl`, the seven prompts verbatim; `CODEX_INSTRUCTIONS_ca1_redo.md`,
  the standing instructions with the location's precision stated: at least three decimals); the validator merges
  the redo answers into `ca1_research_results_all.jsonl`. Answers: yes 372, no 433; confidence high 609, medium 173,
  low 23. Against the shipped flags (for the record; the gate compares after the judgment): 737 agree, 68 differ
  (52 shipped with a crossing answered no, 16 without answered yes). Three "yes" answers mark the structure as not
  open to traffic: bridges on closed borders, which the rubric counts (reported, not rejected).
- 2026-09-28 location test (`location_ca1.py`). Section 8.2: the seven located rechecks recorded in
  `crossing_width/cw1_gate.json` re-measured, none differs in a distance or an outcome. Section 4.3, the 372 "yes":
  located 286, bank-line gap 47, off-border 32, no usable coordinates 7 (the redo).
- 2026-09-28 judgment workflows generated and tested (`make_judge_ca1.py`): `judge_ca1_b1..b4.wf.js` (202 / 201 / 201 /
  201 Claude Sonnet 5 agents), each passing a node syntax check and a dry run with stub agents that renders every
  prompt (`judge_ca1_prompt_samples.txt`). To be regenerated once the redo answers are merged, and run on the
  maintainer's "launch".
- 2026-09-28 redo returned (maintainer: "Codex redo done"): the seven answers are all "yes", with three-decimal
  coordinates; two name a different structure than the first answer (Tamaulipas–Veracruz: Puente Tampico; Grand
  Gedeh–Nimba: the Cestos River Bridge on TAH 7). Validation passes on the merged answers (805, no problem; yes 372, no 433;
  confidence high 610, medium 172, low 23); `ca1_research_results_all.jsonl` written. Location test re-run: located
  291, bank-line gap 48, off-border 33 (the seven: five located, Burtnieku–Rūjienas off-border at 10 km,
  Durazno–Tacuarembó a bank-line gap at 987 m). Judgment workflows regenerated from the merged answers (202 / 201 / 201 /
  201 agents; node check and dry run pass; no first answer of the seven remains). Ready for the maintainer's
  "launch".
- 2026-09-28 judgment launched (maintainer: "launch"): the four workflows run, 805 Claude Sonnet 5 agents. Before the
  launch the RUBRIC block moved from each item to one constant per workflow (every research prompt ends with the same
  block, asserted; the rendered sample prompts are byte-identical), which brings each file under the Workflow tool's
  inline script limit (269,000 to 317,000 characters). Written and tested on synthetic verdicts meanwhile:
  `extract_judge_ca1.py` (the judge schema's checks, `objection` none exactly for a "yes"); `compile_gate_ca1.py`
  (A / B / C / D; the earlier gates' prior-citation pattern verbatim, "shipped" alone not counted since the judge is
  told what the pair ships; each row's record, which the models never saw: its `source` crossing clause, the earlier
  crossing answers on record including the July Sonnet 5 checks, the maintainer's earlier crossing rulings, and a mark
  on a proposed change that would reverse one); `map_ca1.py` (a page per A, B and C row: an overview of the shared
  border and a 5 km detail with the research's coordinates and the screen's ways by class).
- 2026-09-28 the judgment stopped at the session's usage limit: 372 of 805 judges returned (b1 83, b2 108, b3 88,
  b4 93), 433 failed with the limit message, none with another error. After the limit reset (maintainer: "Please
  continue from where you left off"), each workflow resumed from its run with the script unchanged: the journals
  record the returned judges as results and the failed ones as failures, so the 372 replay from cache and only the
  433 run again, under the same settings as the first 372.
- 2026-09-28 judgment complete: 805 of 805 judges after two resumes (the second pass ended at 733, its 72 failures
  60 at the usage limit and 12 request timeouts; the third ran the rest), every script unchanged, so every judge ran
  under the same prompt and settings. The task outputs are kept as `judge_ca1_b1..b4_output.json`. Extraction
  (`extract_judge_ca1.py`): 805 verdicts, all within the schema; yes 350, no 455; `needs_human` 122; confidence high
  473, medium 294, low 38; objections none 350, no-evidence 185, not-in-both-units 107, ferry-ford-or-footbridge 86,
  not-on-the-shared-limit 34, under-construction 23, not-open 19, other 1.
- 2026-09-28 gate (`compile_gate_ca1.py`, `GATE_ca1.md`, `ca1_gate.json`): A 50 (41 shipped with a crossing that both
  passes answer no, 9 without one that both answer yes), B 48, C 285, D 422. C's flags: research confidence medium
  142 or low 13, judge confidence medium 225 or low 21, the judge's `needs_human` 37, the pre-flag 17, different
  structures 14, prior citation 16; 221 of the 285 carry only confidence flags or the prior-citation flag, and all 16
  prior-citation flags are the judges' own statements that they consulted no record (as in cc1, false alarms). 15
  rows would reverse an earlier maintainer crossing ruling (A 11, B 4). Map pages (`map_ca1.py`,
  `maps/ca1_map_check.pdf`): 383, one per A, B and C row.
- 2026-09-28 the map pages re-rendered with a CJK fallback font (Microsoft YaHei after DejaVu Sans), so Chinese way
  and structure names render; 383 pages, no missing glyph. The ruling worksheet (`make_worksheet_ca1.py`,
  `ca1_ruling_worksheet_2026-09-28.md`) follows the 2026-09-16 format: A and B rows with an empty RULING column
  (Yes = accept the proposed change, No = keep the shipped flag), the 64 C rows with a flag other than confidence or
  the prior citation listed for a check, and a mark on the 15 rows where Yes would reverse an earlier ruling.
- 2026-09-29 rulings received (verbatim in `ca1_maintainer_rulings_2026-09-29.txt`, section 1): all 50 A and 48 B rows,
  each "They have a fixed crossing." or "There isn't a fixed crossing.", the first meaning at least one fixed crossing
  (maintainer's note). Mapped live to the gate's rows (labels = the worksheet's order; every pair's names checked against
  the gate's): 44 flags would change as written (33 to no crossing, 11 to a crossing); eight of them reverse an earlier
  ruling (A3, A7, A8, A12, A18, A19, A31, A34), and the other seven marked rows keep it (A5, A6, A21, B1, B7, B22, B45).
  Two rulings reach beyond the flag (A13, A32), and bucket C was not covered.
- 2026-09-29 checks before the freeze, each put to the maintainer (answers verbatim in section 2 of the rulings file):
  - A29 Hamgyong-bukto↔Primorskiy Kray, written as no fixed crossing: the World Bank layer has 11 North Korean units and
    no Rason unit; Rajin and Tumangang lie inside its Hamgyong-bukto, and the research's own coordinates for the new
    Tumangang–Khasan road bridge test located (215 m from the farther unit). Both passes had answered no only because
    Rason is a separate unit in reality — the case of Bueng Kan, which lies inside the World Bank's Nong Khai (A19). The
    pair is the North Korea–Russia pair's only crossing, so a "no" would have made that roll-up unbridged (ADM0 moderate
    320 → 319). Answer: "Has a fixed crossing (Recommended)".
  - A13 Franche-Comté↔Bern, ruled not adjacent: the World Bank polygons share 0.8232 km, two straight 0.41 km segments
    meeting at the Pont de Biaufond on the Doubs; the vertex spacing of the World Bank's Franco-Swiss line is 400–500 m
    throughout, so the straightness is no shape signature, and the arc is stable (0.989 / 1.667 / 7.869 km at 1e-3 / 5e-3
    / 2e-2 degrees), as for the three lake contacts removed on 2026-07-25 on the maintainer's map check (Saramacca↔
    Sipaliwini, 2026-09-01, was kept as "too early to remove"). Answer: "Remove it via the denylist (Recommended)"
    (`docs/FUTURE_EDGE_AUDITS.md` #16).
  - A32 Cuscatlán↔San Vicente, "Actually, their border is mixed with land on the northern.": it reverses the 2026-09-14
    ruling that accepted the pair as water-only (dc1, B17 "Yes"); the 2026-09-14 measurement's 4.0 km run more than 500 m
    from any mapped reach lies at the north end of the 14.9 km arc. Answer: "Take it out (Recommended)".
  - Bucket C (285 rows). Answer: "No override (Recommended)".
  - A scan of every pass's "no" for units the World Bank layer lacks found no further case: the Kangarli rows name the
    World Bank's AZE035 Kengerli (spelled so), where the Poldasht–Shahtakhti bridge lands; the Ugandan (Pakwach, Kwania,
    Kazo) and Gambian rows do not rest on a missing unit.
  - B32 Södermanland↔Uppsala, ruled no fixed crossing, is the example of a bridged lake pair in METHODS, PROVENANCE, the
    drafts and `tests/test_adm1_pericoupling.py`: the Hjulsta Bridge lies inside the World Bank's Uppsala polygon, 2.9 km
    from Södermanland (the record of 2026-09-16 had already called it off the shared line), so the ruling matches the
    polygons; the test and the examples were changed, not the ruling.
- 2026-09-29 frozen (`freeze_rulings_ca1.py` → `ca1_rulings.json`, `ca1_verdicts.csv`) and shipped (`ship_ca1.py`): 43
  flags changed (32 to no crossing, 16 of them between Ugandan districts; 11 to a crossing), FRA011↔CHE006 denylisted
  (denylist 5 → 6) and its water row removed, SLV004↔SLV011's water row removed (the edge stays ordinary); a `ca1` note
  on every row's `source` (the two answers, the research's structure — or, where the judge alone answers yes, a short name
  of the judge's structure, `JUDGE_SHORT` — the confidences, the objection, the location test and the outcome). Edges
  8,464 → 8,463; water-only 805 → 803 (386 / 417); moderate 8,067 → 8,046; stringent 7,659 → 7,660; tiers A 269 / B 238 /
  C 296; ADM0 326 / 320 / 300, roll-ups 26 (20 / 6); `build_all.py` EXPECTED moved; completeness 0 open in every band;
  attributability 803/803.
- 2026-09-29 found while updating the documents:
  - The crossing screen's width basis moved with the flags: on the adjudicated flags its disagreement is 105 / 82 / 80 /
    80 / 85 / 94 / 100 / 105 at 0 / 25 / 50 / 75 / 100 / 150 / 200 / 250 m (it was flat, 99–107, from 25 to 250 m on the
    earlier flags), so the manuscript's sentence now names the range 25 to 100 m, over which agreement is highest, and
    the guard checks that.
  - The edge screens' population is the edges with a border arc, which leaves denylisted contacts out: 8,421 → 8,420, so
    Table 2's nominated and accepted counts, the complementarity sentence and the small-stream counts were recomputed over
    it (the guard now takes the screen record minus the denylist). The small-stream limitation grows from 10 to 29 borders
    without a fixed crossing (7 cross-border, 22 domestic; 16 with no reach of 10 m³/s within 500 m).
  - The guard's crossing checks: the screen record need only cover the shipped rows (it holds the two rows taken out);
    the way-class sentence is recomputed on the new flags (346 bridged rows with a bridge way within 100 m, 336 with a
    road or rail way, the other 10 resting on a structure both passes confirm, 2, or on a ruling, 8); a claim on each
    flag's basis (422 clean confirmations, 285 checked, 96 ruled) is added. Guard 73/73.
  - Stale before this campaign and fixed on the way: PROVENANCE's way-class counts (337 of 348, from 2026-09-24),
    DRAFT_v2's first-round river split (101/137, where the data had 102/136), the ledger's land-gap manifest (4 rows, 5
    since 2026-09-27).
  - Documents: the manuscript (counts, Tables 1 and 2, the false contacts, the crossing paragraph as the two-model
    design, tiers, the limitation) and its DOCX; METHODS (current sections, the has_bridge paragraph, the lake list, a
    `ca1` record); PROVENANCE; REPRODUCING; INTRODUCTION; FUTURE_EDGE_AUDITS (#16); the bridge methodology's status note;
    CHANGELOG; the module and engine docstrings; the drafts EN/ZH; the process notes EN/ZH (section 6 rewritten, 9.23,
    10, 12); the revised Section 3; the ledger; the fact sheet; the supplement's status note; the V3 submission checklist;
    the Chinese checklist (item 73).
- 2026-09-29 second check of the documents (maintainer: "Can you check again to make sure there isn't any stale or wrong
  information in the related documents?"): every figure the change could move scanned across the documents, every
  citation of the 45 affected pairs, every description of the crossing method, and every new sentence traced to its script
  or data. Corrected:
  - Figures computed over the edge screens' population, which the denylisted contact left: the drafts' Table 3 (the 2.5 km
    rung 687 / 483 / 455 river / 464 also hydro; all rungs 1,969 / 543 / 489 / 516) and "446 of its 483 shipped captures
    (the dominant one for 424)" (`ladder_yield_gd1.py`'s rule, run on the population); the Natural Earth lake distribution
    "of 1,773 cross-border edges, 1,662" in the manuscript and the drafts (the guard's lake claim counted the whole record and
    now counts the population); "32 of the 666 shipped river borders on existing edges" below the 0.50 bar at 500 m
    (METHODS, drafts, process notes; the process notes' "34 of the 668" was already stale — the record gives 32).
  - The flags' snapshot: every flag now rests on the screen record of September 2026 (2026-09-24, and 2026-09-27 for the two
    rows added then) and web evidence of the same month, so the manuscript, the revised Section 3 and the drafts no longer
    name the June snapshot, which is only the first round's origin (section 4.1's "pinned to the OpenStreetMap snapshots of
    June and September 2026" no longer holds).
  - A29's note says "maintainer ruling": its flag was decided by the follow-up answer, not the map ruling (the freeze records
    the basis; the ship's note builder reads it). All 803 notes and flags reproduce from `ca1_rulings.json` and `ship_ca1.py`.
  - Smaller: the lists of the layers' origins name the folded rows' own checks; the process notes say the research's
    coordinates are required for a yes and read by the location test to three decimals (the instructions set no
    precision; the redo asked for three decimals); Cuscatlán↔San Vicente's run is "more than 500 m from any mapped reach";
    METHODS' rj2 record gives the large-river split since this campaign (12 / 2); the bridge methodology's closing note
    (297 rows); a test docstring (six denylisted contacts); the fact sheet's adjudication-models bullet points here.
  Guard 73/73; `build_all.py` 12/12; DOCX rebuilt, QA pass.

- 2026-09-30 the edge screens' figures counted over every pair with a border arc (maintainer: "I don't think 'counted on
  the database as it stands now' is reasonable"; of the three ways to count the six false contacts, "Recount Table 2"):
  - This campaign had taken Franche-Comté↔Bern out of the edge screens' population when it was denylisted (8,421 → 8,420,
    above), as the five contacts denylisted earlier were outside it. All six are contacts the screens nominated and the
    adjudication removed, so the screens' figures are now stated as run, over 8,426 pairs: the 8,420 edges and the six.
  - Five of the six are not in `mr1_screens.csv`. `false_contacts/screens_false_contacts.py` measures all six by the
    record's method (arc and per-point distances of `build_arc_cache.py`, shares and nominations of `run_screens_gd1.py`,
    the union at the widest rungs) and stops unless Franche-Comté↔Bern reproduces its row of the record. All six are
    nominated, each the same from either unit's outline.
  - Over the 8,426: any screen 3,291 nominated (3,285) = 777 water-only + 6 false contacts + 2,508 rejected; Natural
    Earth rivers 1,974 (1,969), lakes 170 (167), HydroRIVERS 2,048 (2,044), HydroLAKES 293 (289), combined rivers 2,269
    (2,265), combined lakes 299 (295), union 2,050 (2,044); accepted counts unchanged. Natural Earth lake distribution over
    cross-border pairs: 1,663 of 1,776 at zero, 40 at 0.8 or more, 13 between 0.2 and 0.6. Ladder first nominations
    690 / 188 / 338 / 374 / 384; yields unchanged. Coincident line 1,227,396 km.
  - The corridor census's counts were restated the same day (`SPEC_cc1_corridor_census.md`, section 15).
  - Guard 76/76 (new claims: Table 2's population, the six nominated, the nominations' outcomes). Documents: METHODS
    (the population sentence and a record), CHANGELOG, the edge-audit register's 2026-09-29 note. Papers: V3 (DOCX
    rebuilt), the drafts EN/ZH (Table 3), the process notes EN/ZH (§2, §9.24, §12), the condensed Section 3, a status note
    in the supplement, the fact sheet, the Chinese checklist (item 74). No data change.
