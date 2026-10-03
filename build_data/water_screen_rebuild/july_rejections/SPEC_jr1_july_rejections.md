# Specification: the July rejections the maintainer never saw, put through the September gate (campaign `jr1`)

Status: **APPROVED, 2026-10-02; CARRIED OUT, 2026-10-02** (maintainer, asked whether to record the draft of 2026-10-01 as approved and frozen:
"Yes, approved (Recommended)"), with the rulings of the same day. Frozen as drafted; what has happened since, and every
correction, is logged in section 11. The 63 borders accepted in the two tranches ship as water-only rows with their
fixed-crossing flags, in the working tree; nothing is committed yet (section 11).

## 1. Origin

The manuscript states the adjudication rule as the campaigns since September have applied it: "A candidate that both
passes reject at high confidence is rejected, and rejections at lower confidence are listed for the maintainer, who may
override them. [...] every disagreement or medium-confidence verdict that would change the shipped set goes to the
maintainer."

The maintainer asked how many of the 3,291 nominated pairs were accepted and rejected automatically (section 9).
Tabulated from the campaigns' gate files, the July verdict files and the July worksheets, 2,120 of the 2,508 rejected
pairs were closed with no maintainer involvement. For 1,660 both passes rejected at high confidence, as the rule says.
The other 460 were closed by the July audit (batches b1 to b6 and the 20 km holds) in two ways the rule does not allow:

- 334 pairs that both passes rejected, at least one of them below high confidence. The July audit did not list them for
  the maintainer.
- 126 pairs for which the research pass said water-only and the judgment pass did not. The July worksheets held only
  the pairs the judgment pass accepted, so these were closed on the judgment alone. On 56 of them the judgment pass
  itself asked for a human check.

The maintainer: "Let's deal with the 460 pairs to make it align with our standards in September."

The September gate (section 3.1) flags a rejection for two more reasons than confidence: the judge asks for a human
check, or the water body's name reads as marine. Four more July rejections, both passes at high confidence, carry such a
flag (Baja Verapaz↔Quiché: the judge asked for a human check; Chattogram↔Dhaka, "estuary"; Al Quds↔Ariha, "Dead Sea";
Michigan↔Wisconsin, "Green Bay"). The population is therefore 464, the pairs the September gate would not have closed
without the maintainer.

A first draft adjudicated the 464 again with both models before gating them. The maintainer: "Why needs rerun? According
the current rule, I should do manual review for 464 pairs based on the results on July, right?" That is what the rule
requires, and this specification does exactly that: no model is run again.

## 2. Population: 464 edges

A pair is in the population when all of the following hold (`population_jr1.py`, computed live from the records):

1. it has a border arc and the current edge screens nominate it (`rule_rejections/mr1_screens.csv`);
2. it is neither water-only nor a denylisted contact;
3. its only two-model record is the July audit's (`verdicts/b1.json`, `b2_rerun.json`, `b3b4.json`, `b5b6.json`,
   `holds20km.json`): no later campaign gated it (`rj1`, `rj2`, `dc1`, `pr1`, `hl1`, `gs1`, `ba1`, `ba1r`, `gd1`, `mr1`);
4. the maintainer never decided it: it is on no July worksheet (b1, b1 class 2, b2, the b2 re-run triage, b3 to b6), in
   no ship manifest's human list, and not among the ten verdicts the maintainer ruled on 2026-09-19;
5. the September gate's rules, applied to its July verdicts, put it in bucket B (the passes disagree) or C (both not
   water-only, flagged).

Counts on 2026-10-01: 2,508 nominated pairs are neither water-only nor denylisted; 1,315 of them have the July audit as
their only two-model record; the maintainer decided 43 of those; of the other 1,272 the September gate gives A 0 / B 126
/ C 338 / D 808.

- **464 = 126 split + 338 flagged** (334 with a pass below high confidence, 4 at high confidence with the judge's
  request or a marine name).
- 57 cross-border (21 split, 36 flagged), 407 domestic. By July batch: b1 56, b2 (re-run) 122, b3 and b4 130, b5 and b6
  98, the 20 km holds 58.
- 20 of the 464 were among the rejected candidates the validation study sampled; the maintainer rated all 20 not
  water-only there.

Not in the population:

- **The 808 clean July rejections** (both passes not water-only at high confidence, no flag). The September gate closes
  such a pair without the maintainer, so they already meet the standard the maintainer named.
  `jr1_clean_july_rejections.csv` lists them.
- The 43 pairs the maintainer decided, and every pair a later campaign gated.

## 3. Design: the September gate on the July verdicts, and the maintainer's review

### 3.1 The gate

The two passes' verdicts are the July audit's own, as frozen in `verdicts/`: research on GPT-5.6 Sol, judgment on Claude
Sonnet 5. The bucket rules are `compile_gate_mr1.py`'s, applied by `compile_gate_jr1.py`:

- **B, the passes disagree** (here always: research water-only, judgment not): the maintainer rules on each. 126 rows.
- **C, both not water-only but flagged** (a pass below high confidence, the judge's request for a human check, a marine
  water body): listed for the maintainer, who may override. 338 rows.

Rows are numbered within a bucket in the order of their pair codes (B1 to B126, C1 to C338).

### 3.2 What the July verdicts are

They were written to the July rubric: "water_only=true ONLY if essentially the whole border/gap follows one water
feature; more than ~20% dry land, watershed ridge, or surveyed line => water_only=false." The July judgment prompt did
not bar the judge from the repository, and some judges cite its files. The maintainer's ruling applies the current
standard, the materiality standard of 2026-09-01: a named dry segment (a surveyed or cadastral line, a ridge, a road, an
overland stretch) defeats water-only regardless of its share; the "about a fifth" figure is never a reason to accept; the
answer is not water-only when the evidence is uncertain.

### 3.3 The review package

- `jr1_ruling_worksheet_2026-10-01.md`: one table row per pair. For a B row: the units, the water body the research
  named, the shares of the screens that nominate the border, both passes' verdicts and reasons (shortened), the
  measurement and a ruling cell. For a C row: the units, the screens, the flags and both reasons (shortened).
- `GATE_jr1.md`: every row with both passes' reasons in full.
- `measure_jr1.py`, `measure_jr1.csv` (B rows): where the border leaves mapped water, on the current border arc with the
  screens' own distances. A border point is wet when a feature of any of the four layers lies within 500 m. Given: the
  share of wet points, the dry kilometres, the longest dry run with links to its two ends and its position on the arc,
  and the longest dry run when HydroRIVERS reaches up to 1,000 m count (a run that closes at 1,000 m is an offset between
  the modelled reach and the border; one that persists is a gap). Check: the HydroRIVERS share it computes equals the
  screen record's for all 126.
- `map_jr1.py`, `maps/jr1_map_check.pdf` (B rows; page M*n* = row B*n*): the two units and their neighbours, the four
  water layers, and the border points coloured wet, offset or dry, with the longest dry run ringed.

### 3.4 Rulings

- B: **Yes** = water-only; **No** = not water-only. A B row without a ruling stays not water-only.
- C: "No override", or the rows to accept or to look at again.
- Rulings are frozen verbatim (`jr1_maintainer_rulings_<date>.txt`) and in `jr1_rulings.json` (`freeze_rulings_jr1.py`).
  From then on the July verdicts with the maintainer's ruling or check are the 464 pairs' record.

### 3.5 An accepted border

It is an existing edge reclassified water-only; nothing is added to the edge list. Its row goes to the water file with
verification tier A (maintainer map ruling). Its fixed-crossing flag is decided as every flag has been since 2026-09-29,
which is the one place this campaign runs a model, and only for accepted rows: the crossing screen at 100 m on the
ground, blind GPT-5.6 Sol research with coordinates (the maintainer's Codex run), the location test, Sonnet-5 adversarial
judgment (on the maintainer's "launch"), and the maintainer's ruling on a map, which sets the flag of every new row.

## 4. What the campaign does not change

- No screen, width, bar or nomination: the 464 are already nominated.
- The July verdicts themselves, and the 808 clean July rejections.
- The validation study's report (a record of 2026-07-14). If a pair the study rated is ruled water-only here, the
  campaign's record says so.

## 5. Expected effect

- No B row accepted and no C row overridden: no data change. The documents change once (section 6): every rejection
  among the nominated pairs is then either both passes' clean high-confidence rejection or has been before the
  maintainer, as the manuscript's rule says.
- k borders accepted: water-only 803 → 803 + k and stringent 7,660 → 7,660 − k; moderate 8,046 falls by the accepted
  borders without a fixed crossing; one data pull request with the documents. ADM0 changes only if an accepted border
  completes a country pair as all-water.

No figure is predicted. The measurement finds water within 500 m of every border point on 12 of the 126 B rows and a dry
run of 5 km or more on 55.

## 6. Documents (once, after the rulings)

METHODS (a `jr1` record; the count of rejections by how they were decided, if the maintainer wants it stated),
CHANGELOG, the process notes, the fact sheet, the Chinese checklist. The manuscript's rule sentence needs no change: it
then holds for every nominated pair. With a data change: every count, both tables, the drafts, PROVENANCE, REPRODUCING,
INTRODUCTION, the tests' stated counts and the guard.

## 7. Verification

1. `jr1_population.csv`: 464 rows, computed live; `jr1_gate.json`: B 126 / C 338, every B row a research yes against a
   judgment no in the July files.
2. `measure_jr1.csv`: 126 rows; the HydroRIVERS share and the number of points equal the screen record's for each.
3. The worksheet and `GATE_jr1.md`: 126 B rows and 338 C rows; `maps/jr1_map_check.pdf`: 126 pages in the B order.
4. Rulings frozen: every B row has a ruling or is recorded as left not water-only; the completeness check and
   attributability unchanged without a data change.
5. With a data change: the crossing adjudication of the accepted rows, `build_all.py`, a full regeneration, the full test
   suite and the manuscript guard.

## 8. Decisions left to the maintainer

1. **The 808 clean July rejections.** Out of scope here because the gate standard already closes them. They were judged
   to the July rubric, whose "one water feature" wording differs from the current "essentially all of it is water". A
   text scan cannot tell how many rest on that wording alone (such a phrase appears in 283 of their reasons, mostly to
   say the border is land). This draft leaves them out.
2. **The counts in the manuscript.** After the rulings the rejections divide into those closed on both passes' clean
   rejection, those listed for the maintainer and those ruled. This draft states them in METHODS only.

The population of 464, rather than 460, follows the September gate exactly; the maintainer's remark above names 464.

## 9. Maintainer (2026-10-01, verbatim)

- "In the current 3,291 nominated pairs, How many pairs that automatically rejected and accapted?"
- "Let's deal with the 460 pairs to make it align with our standards in September."
- On "How should the 464 pairs be brought in line with the September standard?" (a fresh two-model run then the gate, or
  gating the July verdicts as they are): "[No preference]".
- "You said there is 334 pairs with Both models rejected, at least one below high confidence before." (The 334 stands;
  section 1 gives the four pairs that make the flagged set 338.)
- "Why needs rerun? According the current rule, I should do manual review for 464 pairs based on the results on July,
  right?"

## 10. Record

`build_data/water_screen_rebuild/july_rejections/`: this specification, `population_jr1.py`, `jr1_population.csv`,
`jr1_population.txt`, `jr1_clean_july_rejections.csv`, `compile_gate_jr1.py`, `jr1_gate.json`, `measure_jr1.py`,
`measure_jr1.csv`, `measure_jr1_points.json`, `map_jr1.py`, `maps/jr1_map_check.pdf`, `make_worksheet_jr1.py`,
`jr1_ruling_worksheet_2026-10-01.md`, `GATE_jr1.md`, `make_full_names_jr1.py` and `jr1_full_names_B.md` (each B row's
units with their kind and country, as the maintainer asked); after the rulings the maintainer's rulings, `freeze_rulings_jr1.py`
and `jr1_rulings.json`.

## 11. Change log

- 2026-10-01 first draft: the 464 adjudicated again by both models, then gated.
- 2026-10-01 redrafted on the maintainer's remark (section 9, last item): no model run again; the September gate on the
  July verdicts and the maintainer's review. The review package of section 3.3 built with the draft.
- 2026-10-01 the full names of the B rows, with their kind of unit and country, on the maintainer's request ("Can you
  provide the ADM0 and ADM1 name for Bucket B, like 'A27: Trarza region in Mauritania and Saint Louis region in
  Senegal.'"): `make_full_names_jr1.py`, `jr1_full_names_B.md`.
- 2026-10-02 rulings received (verbatim in `jr1_maintainer_rulings_2026-10-02.txt`, section 1): "Attached is my answer of
  Bucket B and no override for Bucket C". All 126 B rows: 59 "Yes, their border is water-only.", 66 "No, their border
  isn't water-only." and, for B97, "NO ADJACENCY (point contact only)". Mapped live to the gate's rows: every label's
  names equal the worksheet's. The message's lines are also the body of `jr1_bucket_b_short_summary.docx`, which the
  maintainer placed in the folder with `jr1_bucket_b_codex_adjudication.md`, the maintainer's reference while reviewing
  (its tally, 31 yes / 27 no / 68 uncertain, is the docx's second line). The reference is not a pass of the record
  (maintainer: "Not necessary to records it as it's just a reference. The answer is my reviewed result."): the rulings
  are the maintainer's.
- 2026-10-02 checks before the freeze:
  - B97 Gävleborg↔Västmanland, ruled "NO ADJACENCY (point contact only)". On today's map the two counties meet at one
    point, where four counties meet (OpenStreetMap's county relations: Gävleborg and Västmanland share no way, nor do
    Dalarna and Uppsala). The World Bank polygons share 38.2 km along the Dalälven (16.70°E to 17.19°E), because the
    layer's Västmanland still holds Heby municipality (Heby, Östervåla and Tärnsjö lie inside `SWE020`), which moved to
    Uppsala County on 1 January 2007. Removing the edge would have been the first removal resting on a boundary change
    the World Bank has not applied rather than on a drawing artifact, and the database keeps the World Bank's units
    where they differ from today's (Albania's pre-2015 districts, Latvia's pre-2021 municipalities, both ruled on as
    drawn in this campaign). Put to the maintainer; answer: "Keep the edge (Recommended)". The edge stays an ordinary
    border, not water-only; a note goes to the edge-audit register with the documents.
  - The validation study: of the B rows only B106 Ngora↔Soroti was in its sample of rejected candidates; it is ruled
    not water-only, as rated there. The other 19 sampled pairs are C rows, not overridden. The study's figures stand.
  - Country pairs: the seven accepted cross-border rows leave their country pairs mixed (water-only ADM1 edges then:
    Burundi–Tanzania 3 of 6, Bolivia–Peru 2 of 5, Colombia–Venezuela 4 of 10, Ecuador–Peru 1 of 11, Guinea–Senegal 1 of
    4, Guinea–Sierra Leone 2 of 6, Kenya–Uganda 7 of 17), so no ADM0 roll-up changes.
  - The measurement against the rulings (section 3.3; it shows where mapped water lies, not where the border runs, so a
    legal river line offset from the modelled reach reads as dry): water-only on 7 of the 12 rows with water within
    500 m of every point, on 13 of the 16 whose longest dry run is under 1 km, on 25 of the 43 with 1 to 5 km and on 14
    of the 55 with 5 km or more (B5, B25, B32, B33, B66, B99, B102, B109, B110, B111, B117, B118, B120, B126).
  - Water names: the row of an accepted border carries the July research's water type and name (as the creek-band rows
    of 2026-09-14 carry their research's name), except where the maintainer's reference notes show another or a further
    water body: B2 Ishëm (the research named the Droja), B4 Drin with the Koman and Fierza reservoirs, B46
    Puyango–Tumbes and Quebrada Cazaderos, B51 Sissili (Kulpawn), B53 Mitji and Koulountou, B55 Meli (Makona), B80 Shire
    (Zambezi), B83 Yalí and Coco, B85 Viguí and Tabasará, B95 Tamiš and Danube, B105 Aswa or Achwa (the research added
    "Moroto", a tributary), B117 Apure with its Ruende and Apurito channels, B121 Gansvlei Spruit, Klip and Vaal (type
    river; the research had typed the Vaal Dam a lake). The freeze asserts each replaced and each new name against its
    source.
- 2026-10-02 approved (maintainer: "Yes, approved (Recommended)") and frozen (`freeze_rulings_jr1.py` → `jr1_rulings.json`):
  59 borders accepted (7 cross-border, 52 domestic; 58 river, 1 lake), 66 not water-only, B97 kept as an ordinary edge;
  C 338, no override. No shipped file changes yet (section 3.5).
- 2026-10-02 the fixed-crossing flags of the 59 accepted borders (section 3.5), begun:
  - Layer 1 (`crossing_jr1.py`; `crossing_jr1_pairs.csv`, `crossing_jr1_ways.csv`): the screen of 2026-09-24 as
    `corridor_census_exact/crossing_cc1.py` holds it for pairs that are not shipped rows, used unchanged; its two
    reproduction checks pass; 59 pairs, no failed query. An open bridge way lies within 100 m of both units on 29
    borders; 16 have none within 1 km. OpenStreetMap as of 2026-10-02.
  - Research handoff (`build_queue_jr1.py`; `jr1_crossing_queue.csv`, `jr1_crossing_research_input.jsonl`,
    `CODEX_INSTRUCTIONS_jr1_crossing.md`): ca1's prompt, built with ca1's own code taken by AST; the builder reproduces
    all 805 prompts of ca1's research input from ca1's queue and screen record; 59 prompts (52 domestic, 7
    cross-border); leak check clean. The instructions are ca1's, with the three-decimal precision of its redo stated
    from the start. For the maintainer's Codex run.
  - Written for the steps after it: `validate_crossing_jr1.py` (ca1's rules), `location_jr1.py` (the location test,
    unchanged), `make_judge_crossing_jr1.py` (ca1's judge template by AST with three passages changed, because these
    borders are not shipped rows: the judge is told that the pair carries no flag yet; it receives the tunnel, dam-top,
    dyke and embankment ways for every pair, since layer 1 queried them for every pair; `needs_human` is set on an
    answer that differs from the research's or on low confidence) and `extract_judge_crossing_jr1.py`.
- 2026-10-02 crossing research returned (maintainer: "Codex done."): 59 answers in `jr1_crossing_research_results.jsonl`.
  Validation (`validate_crossing_jr1.py`, `validate_crossing_jr1.txt`): every id answered once, every field valid,
  every "yes" with coordinates the location test can read; nothing to research again;
  `jr1_crossing_research_results_all.jsonl` written. Answers: yes 30, no 29; confidence high 32, medium 21, low 6.
  Against layer 1: 27 "yes" with an open bridge way within 100 m of both units and 27 "no" without one; 3 "yes" without
  one (B43, B111, B112) and 2 "no" with one (B44, B78).
- 2026-10-02 location test (`location_jr1.py`, `jr1_crossing_location.csv`, `location_jr1.txt`): the seven located
  rechecks on record re-measured, none differs in a distance or an outcome. The 30 "yes": located 20, bank-line gap 5
  (B32 945 m, B43 819 m, B110 2,138 m, B111 1,686 m, B126 2,966 m from the farther unit), off-border 5 (B33 71.0 km: the
  coordinates are the N4 bridge at Ébebda over the Sanaga; B51 19.7 km; B69 13.6 km; B56 5.2 km; B117 3.6 km).
- 2026-10-02 judgment workflow generated and tested (`make_judge_crossing_jr1.py`): `judge_crossing_jr1.wf.js`, 59
  Claude Sonnet 5 agents; node syntax check and a dry run with stub agents that renders every prompt
  (`judge_crossing_jr1_prompt_samples.txt`). To be run on the maintainer's "launch".
- 2026-10-02 judgment launched (maintainer: "Launch."): the workflow run, 59 Claude Sonnet 5 agents. Written and tested
  on synthetic verdicts meanwhile, the synthetic files deleted afterwards: `compile_gate_crossing_jr1.py` (ca1's flags
  and helpers by AST; with no shipped flag to confirm or challenge the rows are grouped as both passes yes, the passes
  disagree, both passes no, and every row goes to the maintainer; rows keep their labels of the water-only worksheet),
  `map_crossing_jr1.py` (ca1's page, its drawing code by AST; the detail is centred on the research's coordinates only
  when the location test places them on the border or within its bank-line gap, otherwise on the screen's nearest
  open bridge way or the middle of the border, so that a structure cited far from the border does not take the border
  out of view) and `make_worksheet_crossing_jr1.py` (ca1's table cells by AST; one table per group, the agreed answer
  shown as the proposed flag).
- 2026-10-02 judgment complete: 59 of 59 judges in one run, none failed; 59 tool uses for the 59 agents, the
  structured-output calls only, so no judge read a file. The task output is kept as `judge_crossing_jr1_output.json`.
  Extraction (`extract_judge_crossing_jr1.py`, `jr1_crossing_judge_results.json`): 59 verdicts, all within the schema;
  yes 31, no 28; `needs_human` 7; confidence high 27, medium 30, low 2; objections none 31, no-evidence 23,
  not-in-both-units 5.
- 2026-10-02 gate (`compile_gate_crossing_jr1.py`, `jr1_crossing_gate.json`, `GATE_jr1_crossing.md`): both passes yes 30,
  the passes disagree 1 (B78 Căușeni↔Transnistria: the research found no bridge, the judge accepts the screen's
  tertiary-road bridge way touching both units), both passes no 28. Flags: research confidence medium 21 or low 6, judge
  confidence medium 30 or low 2, the judge's `needs_human` 7, the pre-flag 4 (an agreed "yes" with no road or rail way
  within 100 m and a structure in the bank-line gap: B43, B110, B111; an agreed "no" with a road bridge way within 100 m
  of both units: B44, Lita's main street at the three-province junction), prior citation 2 (B7 and B43: both are the
  judges' own statements that they consulted no record). On two agreed "yes" rows the judge rejects the research's
  structure and accepts the screen's bridge way on the border instead (B33: the N4 bridge over the Mbam, not the Pont
  d'Ébebda over the Sanaga; B126: a secondary-road bridge on the arc, not the Birchenough Bridge 3 km off it). Map pages
  (`map_crossing_jr1.py`, `maps/jr1_crossing_map_check.pdf`): 59, one per row, in the worksheet's order. The ruling
  worksheet: `jr1_crossing_ruling_worksheet_2026-10-02.md`.
- 2026-10-02 rulings on the flags (verbatim in `jr1_crossing_maintainer_rulings_2026-10-02.txt`): "For B78, yes; For
  other 58 pairs, as proposed." Frozen (`freeze_crossing_rulings_jr1.py` → `jr1_crossing_rulings.json`,
  `jr1_crossing_verdicts.csv`): 31 borders with a fixed crossing, 28 without.
- 2026-10-02 shipped to the working tree (`ship_jr1.py`; nothing committed): 59 rows appended to
  `data/water_classification_pairs.csv` (on existing edges, tier A, cross-vendor; each `source` gives the nominating
  screens, the July verdicts, the gap measurement, the water-only ruling and the crossing record; four notes name the
  structure the judge accepted where it is not the research's). Water-only 803 → **862** (386/417 → **417/445**; river
  747, 406 with a crossing; lake 115, 11 with one); moderate 8,046 → **8,018**; stringent 7,660 → **7,601**; tier A 269 →
  **328**; edges 8,463 and ADM0 326/320/300 unchanged, roll-ups 26. `build_all.py` verifies the counts (EXPECTED moved);
  completeness 0 open in every band and attributability 862/862 (`completeness_jr1.txt`, `attributability_jr1.txt`); the
  stated counts of `scripts/apply_overlays.py`, the module docstring and `tests/test_apply_overlays.py` moved. The
  manuscript's figures were recomputed and its guard extended to rows that had no flag (87/87). The other documents
  wait for the second tranche below, so that they change once.
- 2026-10-02 the second tranche. Tabulating how every nominated pair was decided (`tabulate_decisions_jr1.py`,
  `jr1_decisions.txt`) left two pairs unclassed, both of the cross-border extension's new-candidate gate of 2026-07-28.
  One, Diffa↔Lac across Lake Chad, had been rejected by both passes at medium confidence and closed with the note "no
  maintainer override needed"; the tabulation of 2026-10-01 had counted it with the clean rejections on that note,
  without reading the confidences. The maintainer, asked whether to override it: "Why missed that pairs? Is there other
  pairs left so we can do the adjudication together?" It was missed because the population of section 2 holds only the
  edge screens' nominations whose record is the July audit's five verdict files, and the pairs outside it had been
  classed by the bucket their campaign printed. `audit_all_rejections_jr1.py` therefore applies the rule to the verdicts
  of every rejected nomination, of the edge screens and of the corridor census alike (2,790 = 2,449 + 341), and reads
  the maintainer's part from each campaign's own rulings: 1,864 clean (both passes at high confidence, no flag), 217
  ruled, 436 listed with a recorded answer, and 273 left:
  - 35 with no ruling where the rule calls for one: two on which both passes said water-only and 33 on which they
    disagree. Nine are census pairs with the July audit as their record (the population of section 2 required a border
    arc); ten were closed on 2026-09-01 by applying the maintainer's materiality standard to the dry segment the passes
    named (`rj1`; the maintainer ruled the six contested pairs); sixteen were closed the same day by default, because only
    one pass said water-only (`rj2`; the maintainer ruled bucket A).
  - 14 flagged rejections never listed: 13 census pairs of the July audit (seven of them Maltese harbour pairs, six
    flagged only for a marine water body) and Diffa↔Lac.
  - 224 flagged rejections that stood on a gate page or worksheet given to the maintainer, with no answer on the list
    recorded: the first campaigns' gate pages (`rj1` 10, `rj2` 31), the creek band's "for a glance" list (`dc1` 167),
    and the lists of `pr1` (4), `hl1` (9) and `gs1` (3), recorded then as "no override stated" or "no ruling requested".
  Second-tranche package, no model run: `compile_gate_t2_jr1.py` → `jr1_t2_gate.json` (A 2 / B 33 / C 238; labels
  continue the first tranche's: A1, A2, B127 to B159, C339 to C576); `measure_jr1.py --t2` and `map_jr1.py --t2` for the
  21 A and B rows that are edges (`measure_jr1_t2.csv`, `maps/jr1_t2_map_check.pdf`); the census's own map for the 14
  that are census pairs (`corridor_census_exact/map_census_cc1.py` over `jr1_t2_census_rows.csv`: every pair reproduces
  its census row; `maps/jr1_t2_census_map_check.pdf`); `make_worksheet_t2_jr1.py` →
  `jr1_t2_ruling_worksheet_2026-10-02.md`, `GATE_jr1_t2.md`; `make_full_names_jr1.py --t2` → `jr1_t2_full_names.md`. A
  census pair ruled water-only would be added as an edge (Stage 3), as the census campaigns' rows were.
- 2026-10-02 the 35 ruling rows in one file (maintainer: "Can you combine the 35 pairs (21+14) in a file, which is easier
  for me to read."): `make_combined_t2_jr1.py` → `jr1_t2_35_pairs.pdf`, 71 pages: how to rule and a list of the rows,
  then for each row, in label order, a summary page (the units' full names, why the row is here, the water named, what
  nominates it, the measurement, both passes' verdicts and reasons in full, the dry segment named when it was closed)
  and its map. Found while assembling it: the PDF that `corridor_census_exact/map_census_cc1.py` writes places each
  130 dpi map image on a 100 dpi canvas, which cuts off the top (the title) and the right of every census map; the
  images themselves are whole. The combined file takes the images, and `maps/jr1_t2_census_map_check.pdf` is written
  again from them (14 pages). The census campaign's own `maps/cc1_map_check.pdf` of 2026-09-27 has the same cut; it is a
  record and is left as it is.
- 2026-10-02 second-tranche rulings received (verbatim in `jr1_t2_maintainer_rulings_2026-10-02.txt`, section 1):
  "Attached is my answer of 35 pairs and no override for the 238 pairs". The 35 rows: 6 "Yes, their border is
  water-only." (B133, B134, B137, B143, B148, B153; on B143 the maintainer adds "The claim of  Malawi or Tanzania doesn’t
  change water-only results."), 28 "No, their border isn't water-only." (both A rows among them, on which both passes
  had said water-only) and, for B150, "NO ADJACENCY (point contact).". Mapped live to the gate's rows: every label's two
  countries equal the gate's (the maintainer wrote the units' names in their usual spellings).
- 2026-10-02 checks before the freeze:
  - B143 Northern Region (Malawi)↔Ruvuma (Tanzania), a census pair, so a Yes would add an edge. In the World Bank
    polygons the lake between them is a unit of its own (Malawi's "Area under National Administration"): Northern
    Region–lake unit (357.5 km) and lake unit–Ruvuma (158.0 km) are already water-only edges, and on 2026-09-01 the
    maintainer re-ruled the two pairs of the same kind No for that reason (Niassa–Central Region, Niassa–Northern
    Region). Put to the maintainer; answer: "No direct edge (Recommended)". The Yes is recorded as water between the two
    units, not as adjacency; no edge is added.
  - B137 Faranah (Guinea)↔Eastern (Sierra Leone), a census pair: the polygons are 794 m apart, and Northern (Sierra
    Leone) and Nzérékoré (Guinea) meet between them along the Meli, the border accepted as B55; Faranah and Eastern are
    the diagonal pair of that four-unit corner. Answer: "No edge (Recommended)". No edge is added.
  - B150 East (Rwanda)↔Isingiro (Uganda), a census pair ruled not adjacent: there is no edge and none is added.
  - The validation study: none of the 35 ruled rows was in its sample of rejected candidates; one C row was (C455, two
    Maltese localities), not overridden. The study's figures stand.
  - Country pairs: the one accepted cross-border row (B153) leaves South Sudan–Uganda mixed (1 of 9 ADM1 edges
    water-only), so no ADM0 roll-up changes.
  - The measurement of the four accepted edges (`measure_jr1_t2.csv`; it shows where mapped water lies, not where the
    border runs): water within 500 m of every point on B153; longest dry run 0.5 km on B148, 1.4 km on B133 and 4.5 km on
    B134, each under 1 km once HydroRIVERS reaches up to 1,000 m count (0.0, 0.4 and 0.5 km).
  - Water names: B134 Loange River and B148 Yuat River headwaters are the research's names; B133 Rapel River and B153
    Unyama River are the judgment's, shortened (on both the research had found the border not water-only and named
    other water: "minor Maipo/Rapel headwaters", "Bahr el Jebel and Unyama at the eastern endpoint only"). The freeze
    asserts each against its pass's text.
- 2026-10-02 frozen (`freeze_rulings_t2_jr1.py` → `jr1_t2_rulings.json`): four borders accepted, all existing edges and
  all rivers (B133 O'Higgins↔Valparaíso, B134 Kasaï↔Kwango, B148 Enga↔Madang, B153 Eastern Equatoria↔Adjumani; three
  domestic, one cross-border); 28 not water-only; two with water between the units and no edge (B137, B143); one not
  adjacent (B150); C 238, no override. No edge is added or removed. No shipped file changes until the four flags are set
  (section 3.5).
- 2026-10-02 the audit closed (`close_audit_jr1.py` → `jr1_audit_closed.txt`; the audit's own two files stay as run and
  are now written only when the audit itself is run): with the second tranche's rulings every rejected nomination is
  closed by the rule: 2,790 = 1,864 clean + 252 ruled (35 of them here) + 674 listed with a recorded answer (238 of them
  here); none left. The four accepted borders are counted as ruled until their rows ship. This closing replaces the
  classes of `jr1_decisions.txt`, which followed the bucket each campaign printed.
- 2026-10-02 the fixed-crossing flags of the four accepted borders (section 3.5), begun. The crossing scripts take
  `--t2`: every input and output file name then reads `jr1_t2` for `jr1`, and the accepted rows are those of
  `jr1_t2_rulings.json`. Run without it they reproduce the first tranche's files byte for byte (checked by hash: the
  queue, the research input and its instructions, the validation report, the merged answers, the location test's two
  files, the workflow and its prompt samples, the judge verdicts, the gate's two files, the worksheet; the map file
  was not drawn again). `build_queue_jr1.py` no longer counts this campaign's own shipped rows as existing water rows,
  so it runs after the first tranche shipped.
  - Layer 1 (`crossing_jr1.py --t2`; `crossing_jr1_t2_pairs.csv`, `crossing_jr1_t2_ways.csv`, `crossing_jr1_t2_run.log`):
    both reproduction checks pass; four pairs, no failed query. An open bridge way lies within 100 m of both units on
    B133 (Puente Rapel, a primary road, touching both); B134 and B148 have none within 1 km; on B153 the nearest is the
    bridge way of an unclassified road, in Eastern Equatoria and 111 m from Adjumani. OpenStreetMap as of 2026-10-02.
  - Research handoff (`build_queue_jr1.py --t2`; `jr1_t2_crossing_queue.csv`, `jr1_t2_crossing_research_input.jsonl`,
    `CODEX_INSTRUCTIONS_jr1_t2_crossing.md`): four prompts (three domestic, one cross-border); the builder reproduces all
    805 prompts of ca1's research input; leak check clean. For the maintainer's Codex run.
- 2026-10-02 crossing research returned (maintainer: "Codex done"): four answers in
  `jr1_t2_crossing_research_results.jsonl`. Validation (`validate_crossing_jr1.py --t2`, `validate_crossing_jr1_t2.txt`):
  every id answered once, every field valid, the one "yes" with coordinates the location test can read; nothing to
  research again; `jr1_t2_crossing_research_results_all.jsonl` written. Answers: yes 1 (B133, the Puente Rapel on route
  G-80-I, high confidence), no 3 (B134 medium, B148 high, B153 medium). Against layer 1: the "yes" has an open bridge way
  within 100 m of both units, the three "no" have none.
- 2026-10-02 location test (`location_jr1.py --t2`, `jr1_t2_crossing_location.csv`, `location_jr1_t2.txt`): the seven
  located rechecks on record re-measured, none differs in a distance or an outcome. The one "yes" is located: 28 m from
  the shared arc and from the farther unit.
- 2026-10-02 the water names checked, because the research on B134 states that the law makes the Lushiko, not the
  Loange, the Kasaï–Kwango limit (`water_names_t2_jr1.py` → `jr1_t2_water_names.txt`: OpenStreetMap's waterways within
  500 m of each border, by name; a check of names, not a classification, and OpenStreetMap is not one of the screens'
  water layers):
  - B133: 'Río Rapel' along 100% of the border. The name stands.
  - B134: 'Lushiko' along 70.5% of the border, over its whole length; no way named Loange in the border's box. The law
    itself, read from the text layer of the Journal officiel's special issue of 28 March 2015 (Organic Law 15/006;
    spelling restored from the scan's text): Kwango, article 14, "A l'Est: [...] La rivière Lushiko depuis le confluent
    de la rivière Lusunu jusqu'à son intersection avec le 7ème parallèle Sud"; Kasaï, "A l'Ouest: [...] La rivière
    Lushiko depuis la frontière [...] jusqu'à son confluent avec la rivière Loange; celle-ci jusqu'à son confluent avec
    la rivière Kasaï"; Kwilu, article 15, "A l'Est: [...] La rivière Loange depuis son confluent avec la rivière Kasaï
    jusqu'au confluent de la rivière Lushiko; celle-ci jusqu'au confluent de la rivière Lusunu". The World Bank arc ends
    at 7°S exactly. The Loange is the Kasaï–Kwilu limit; both passes of 2026-09-01 had named it for this border.
  - B148: 'Yuat River' along 100% of the border ("headwaters" is the passes' word).
  - B153: a river without a name in OpenStreetMap along the eastern two thirds (sample points 0 to 21 of 32). It runs
    from the south-east to the Nile at 3.587°N 32.037°E, where OpenStreetMap's Albert Nile ends and its Bahr al Jabal
    begins: the Unyama. The Nile's ways lie within 500 m of points 16 to 27, where the World Bank line runs due west
    from the Unyama's mouth along 3.5869°N. The last four points (about 0.9 km at the western end) have no OpenStreetMap
    waterway within 500 m; the campaign's measurement, on the screens' four layers, has water within 500 m of every
    point, and the maintainer ruled on that map. Both passes of 2026-09-01 named the two rivers; the shortened name had
    kept only the Unyama.
  Proposed to the maintainer with the judgment: B134 "Lushiko River", B148 "Yuat River", B153 "Unyama River; Albert
  Nile (Bahr el Jebel)" (the neighbouring row Adjumani↔Moyo names the same reach "Albert Nile (White Nile)"). The
  prompts of both crossing passes stay as they were researched (the research on B134 worked on the Lushiko by its own
  finding); a changed name goes to the row.
- 2026-10-02 judgment workflow generated and tested (`make_judge_crossing_jr1.py --t2`): `judge_crossing_jr1_t2.wf.js`,
  four Claude Sonnet 5 agents; node syntax check and a dry run with stub agents that renders every prompt
  (`judge_crossing_jr1_t2_prompt_samples.txt`). To be run on the maintainer's "launch".
- 2026-10-02 the names decided and the judgment launched (verbatim in `jr1_t2_maintainer_rulings_2026-10-02.txt`,
  section 3): "Corrected names (Recommended)" and "Launch (Recommended)". The freeze now carries the row's name and,
  where it differs, the passes' name as `water_body_researched` (`freeze_rulings_t2_jr1.py`, each new name asserted
  against the answer and against the check's output): B134 Lushiko River (the passes: Loange River), B148 Yuat River
  (Yuat River headwaters), B153 Unyama River; Albert Nile (Bahr el Jebel) (Unyama River); B133 Rapel River unchanged.
  `build_queue_jr1.py` gives the crossing passes the passes' name, so the queue, the research input and the workflow are
  unchanged byte for byte (checked by hash, as are the first tranche's queue, research input and gate); the gate shows
  the row's name with the name the passes were given.
- 2026-10-02 judgment complete (workflow run `wf_863544a8-e82`): four of four judges, none failed; four tool uses for the
  four agents, the structured-output calls only, so no judge read a file. The task output is kept as
  `judge_crossing_jr1_t2_output.json`. Extraction (`extract_judge_crossing_jr1.py --t2`,
  `jr1_t2_crossing_judge_results.json`): four verdicts, all within the schema; yes 1, no 3; `needs_human` 0; confidence
  high 2, medium 2; objections none 1, no-evidence 1 (B134), ferry-ford-or-footbridge 1 (B148), not-in-both-units 1
  (B153).
- 2026-10-02 gate (`compile_gate_crossing_jr1.py --t2`, `jr1_t2_crossing_gate.json`, `GATE_jr1_t2_crossing.md`): both
  passes yes 1 (B133, the Puente Rapel, located), the passes disagree 0, both passes no 3 (B134, B148, B153). Flags:
  research confidence medium 2 and judge confidence medium 2 (B134 and B153 each); no pre-flag, no prior citation. On
  B153 the judge notes the screen's near miss, an unclassified road's bridge way in Eastern Equatoria 111 m from
  Adjumani, and finds nothing that makes it a public road across this limit. Map pages (`map_crossing_jr1.py --t2`,
  `maps/jr1_t2_crossing_map_check.pdf`): four, in the worksheet's order; a title line that would run off the page is
  broken after the water name. The ruling worksheet: `jr1_t2_crossing_ruling_worksheet_2026-10-02.md`. Checked before
  the ruling: the screen's near miss on B153 crosses a side stream 64 m from the Unyama, and the road it carries stays
  inside South Sudan; the one road that crosses the World Bank line ends at the Nile's bank 33 m south of it, with no
  bridge.
- 2026-10-02 ruling on the four flags (verbatim in `jr1_t2_crossing_maintainer_rulings_2026-10-02.txt`): "As proposed
  (Recommended)". Frozen (`freeze_crossing_rulings_t2_jr1.py` → `jr1_t2_crossing_rulings.json`,
  `jr1_t2_crossing_verdicts.csv`): B133 with a fixed crossing; B134, B148 and B153 without.
- 2026-10-02 shipped to the working tree (`ship_t2_jr1.py`; nothing committed): four rows appended to
  `data/water_classification_pairs.csv` (on existing edges, tier A, cross-vendor; each `source` gives the nominating
  screens, the verdicts of 2026-09-01 and how that campaign closed the pair without a ruling, the gap measurement, the
  water-only ruling, the name set after the check of names, and the crossing record; the screen and crossing notes are
  `ship_jr1.py`'s, by AST). Water-only 862 → **866** (417/445 → **418/448**; river 751, 407 with a crossing; lake 115,
  11 with one; 840 on shared edges and 26 not touching); moderate 8,018 → **8,015**; stringent 7,601 → **7,597**; tier A
  328 → **332**; cross-border 374 (148/226), domestic 492 (270/222); edges 8,463 and ADM0 326/320/300 unchanged, roll-ups
  26. `build_all.py` verifies the counts (EXPECTED moved); completeness 0 open in every band and attributability 866/866
  (`completeness_jr1_t2.txt`, `attributability_jr1_t2.txt`); the stated counts of `scripts/apply_overlays.py`, the module
  docstring and `tests/test_apply_overlays.py` moved.
- 2026-10-02 the audit closed on the shipped state (`close_audit_jr1.py` → `jr1_audit_closed.txt`): 2,786 rejected
  nominations (edge screens 2,445, corridor census 341) = 1,864 clean + 248 ruled + 674 listed with a recorded answer;
  none left. Of the second tranche's 273: 31 ruled and still rejected, 4 accepted and shipped, 238 listed.
- 2026-10-02 the documents, once, for both tranches (section 6). The manuscript (14 passages: the three views, the
  water-only composition and tiers, Table 2's accepted column, the nominations' outcome, the layers' complementarity, the
  way classes behind the bridged flags and the flags' basis, "the passes' common answer (62 pairs) or ruling where they
  disagree (one)") and its guard, which now reads both tranches' screen records and flag rulings: 87 of 87. The DOCX
  rebuilt (`paper/build_ems_docx_v3.py`) and its structural QA passed; it carries the new figures and none of the earlier.
  METHODS: the pipeline counts, the rule's tally of the rejections (2,786 = 1,864 + 674 + 248) in the adjudication
  design, a `jr1` record, the `has_bridge` paragraph (the rows that had no flag), a pointer from the 2026-09-01 campaigns
  to the four rejections accepted here, and two stale figures corrected on the way: the creek band's shipped rows (50
  where 49 have shipped since 2026-09-29) and the later rows given once as 507 beside 506 (now 569; the first-round rows
  are 297 as stated, one of them tier A). PROVENANCE: the counts, the tiers, the origins, the flags' basis and the way
  classes behind the bridged flags (364 of 375; the third structure both passes confirm is the Paso Mazangano bridge,
  Cerro Largo↔Tacuarembó). REPRODUCING, INTRODUCTION, CHANGELOG, the edge-audit register (#17 to #20 and the decisions),
  the bridge methodology's status note. The drafts EN/ZH (the counts; Table 3 and the ladder sentences recomputed on
  the shipped set by `ladder_yield_jr1.py`, which runs `geodesic_distances/ladder_yield_gd1.py` unchanged into this
  folder: 2.5 km 507 of 690 shipped, 73%; all rungs 585, 30%; own river named for 468 of 507, the dominant one for 445;
  at 5 and 10 km 30 of 47; at 15 and 20 km 10 of 31; the sentence "the river-to-lake mix of the shipped captures inverts
  at 15 km" no longer holds, 9 river and 9 lake at 15 km, and is replaced by the lake share; "43 of the 728 shipped river
  borders on existing edges" below the HydroRIVERS bar at 500 m, every one nominated by the Natural Earth river ladder;
  the creek-only borders without a crossing 31 = 9 cross-border + 22 domestic; the lake rows 115 = 60 + 52 + 3, Lake
  Lilaste the new one), the process notes EN/ZH (§6, §7, §8, §10, a new §9.25, §12; the screen disagrees with the flags
  at 100 m on 89 of 866 rows), the revised Section 3, the reproduction ledger, the fact sheet, the supplement's status
  note, the Chinese checklist (item 75) and the submission checklist. Every remaining mention of an earlier figure is a
  dated record.
- 2026-10-02 second check of the documents (maintainer: "Can you scan the related documents to see if there is any stale
  or wrong information?"). `doc_scan_jr1.py` → `doc_scan_jr1.txt`: every mention of a headline figure of an earlier state
  in 24 documents and the code's docstrings and tests, with its heading and paragraph: 550, 70 of them outside a
  paragraph or section marked by a date or a campaign; each of the 70 read. Every figure of the new texts recounted from
  the shipped data and the campaign records (the composition by origin, the tiers, the gates' buckets, the rulings, the
  crossing passes' answers and groups, the audits' tallies), and the 63 rows' `source` strings checked against their
  records (verdicts and confidences of both passes, the gap measurement, both crossing passes, the flag, the water name,
  tier, class): no difference. Corrected: PROVENANCE's lake bullet, "10 of the 114 lake pairs" with a fixed crossing (11
  of 115; Lake Lilaste); the July run's gate, which PROVENANCE described as "medium-confidence verdicts human
  map-verified" and METHODS' design paragraph as "medium-confidence verdicts were human map-verified": that held for its
  acceptances, while its disagreements and flagged rejections reached the maintainer only on 2026-10-01/02 (both now say
  so); the process notes' auto-accept policy (EN/ZH), which gave no rule for rejections (the listing rule added, with the
  date the July run's own reached the maintainer); the drafts' Table 2 cell on the judgment pass, "told the flag" (or
  that there is none yet; EN/ZH); and the fact sheet's "all upheld" for rj1 and rj2 (pointers to the four accepted
  here). The changelog entry and the Chinese checklist (item 75 f) record them.
