# Future work — deferred items

Items deliberately **not** acted on, with enough detail to pick up cold. Nothing
here affects the shipped database: every count in `data/PROVENANCE.md` is current
and every item below ships unchanged until a maintainer decides otherwise.

Deferred **2026-07-26** by maintainer decision, to finish the manuscript first.

---

## 1. `BOL004`↔`PER007` — a probable unit-identity artifact

**Status: recorded, no action. The pair ships unchanged.**
Full evidence lives in `docs/FUTURE_EDGE_AUDITS.md` **#14** — not duplicated here.

One-line summary: World Bank's `PER007` polygon is named *Callao* (Peru's Pacific
port province, next to Lima) but is 4-part with its parts ~900 km apart — the
largest by area sits in **Lake Titicaca** and carries a 62.5 km border with La Paz,
while the three genuine Callao parts sit on the Pacific coast. La Paz already
borders the real Puno (`PER021`) natively over 761.7 km, so the Titicaca frontage
is already accounted for by a legitimate unit.

**Why it was deferred rather than fixed:** this is a *unit-identity* question, not
a water verdict, so it was out of scope for wu1/wu2. Resolving it needs the
identity-fingerprinting method used for `RUS050` (the Kalmykia salient) and the
four water-surface placeholders — geodesic area, representative point,
point-in-polygon probes for known settlements, and the WB adjacency signature —
run against the pinned GeoPackage. Whatever it concludes is a source-relabel-class
or denylist-class change, i.e. a Stage-1 edit, not a manifest edit.

## 2. No detector exists for this failure class

**Status: proposed, never run.**

This is the part worth remembering, because it is a gap rather than a single bad
row. The audit instruments each catch a different thing:

| instrument | catches |
|---|---|
| arc-tolerance ladder (1e-3 / 5e-3 / 2e-2) | corner and point contacts |
| water-share fractions (`frac_500`, `frac_2000`) | mixed land vs water |
| union-mask corridor census | non-touching but genuinely adjacent pairs |

**None of them can detect "this polygon is the wrong unit."** `BOL004`↔`PER007`
measured *perfectly clean* — arc 62.693 / 63.565 / 67.350 km, one component
throughout, collapse ratio 0.931, `frac_500` and `frac_lake_500` both **1.00**.
It was caught only because a human read the pair name and recognised that Callao
cannot border Bolivia. Nothing in the pipeline would catch the next one.

**Proposed screen** (deterministic, no AI, ~minutes over the pinned Admin-1
GeoPackage, 3,591 polygons):

- For every multi-part ADM1 polygon, compute the geodesic distance between part
  centroids. Flag any unit whose parts are separated by more than a threshold
  (~200 km is a reasonable first cut; `PER007` is ~900 km).
- Secondary signal: flag units where the **largest part by area** lies far from
  the area-weighted centroid, which is the specific signature here — the centroid
  (−71.02 / −15.25) is a compromise between two clusters and belongs to neither.
- Triage only. Genuine multi-part units are common and legitimate (island groups,
  exclaves, the Anguilla districts), so output is a candidate list for identity
  fingerprinting, never an automatic removal — the standing rule that an edge
  falls only on a WB-internal signature or the maintainer's official-map check
  still applies.

Expected outcome is one of two useful answers: `PER007` is a one-off, or it is the
visible member of a family. Either way the screen is cheap and the result is
frozen evidence.

## 3. Cross-border extension of the rebuilt hydro rungs (screen unification)

**Status: EXECUTED and CLOSED 2026-07-28** — the extension ran over all 1,794
cross-border edges (`build_data/water_screen_rebuild/hydro_fold/crossborder_hydro_disposition.csv`):
14/14 previously-unrecorded rows re-nominate, whole-graph attributability
736/736, and the fold followed (registry 5 → 3; see CHANGELOG). The two NEW
cross-border nominations it surfaced were **adjudicated the same day** under the
standard cross-vendor mini-batch (GPT-5.6 Sol research → Sonnet-5 adversarial judge →
deterministic measurement → maintainer gate) and **both ruled NOT water-only —
zero data change** (`newcand_rulings.json`): `NER002`↔`TCD010` by dual-AI
agreement (IBS-73 surveyed segments + the Northern Pool's post-1970s dryness);
`BRA025`↔`URY014` by **maintainer official-map ruling** (land border, especially
the straight-line segment) — overruling the judge's water_only=true (10.2% dry
< the 20% bar) and the measurement lean (union water 0.861 @500 m), recorded
verbatim per the evidence hierarchy. **Every nomination in the whole-graph
screen record is now adjudicated; none pending.** *(Correction 2026-09-01: the rj2 completeness scan found 37 domestic large-river nominations — HydroRIVERS ≥ 0.5 in the ≥ 100 m³/s band — for which the 2026-07-10 queue registry had no domestic source — its medium-band source stopped at 100 m³/s, and large rivers were assumed covered by the Natural Earth rungs, which miss exactly these; all 37 were adjudicated cross-vendor with maintainer map rulings, 15 shipped water-only. The whole-graph claim holds as of that date.)* The section below is retained
as the original campaign spec.

**Original status: precondition verified 2026-07-27; campaign not run.**

The rebuild's HydroRIVERS/HydroLAKES rungs ran over **domestic borders only** —
cross-border hydro coverage was inherited from the already-complete full-database
sweeps (2026-07-02/06), so the rebuilt edge-screen disposition does not cover 14
shipped rows (13 `hydro_water` + 1 `hydro_lakes`; Skadar is non-touching and
covered by the recovery census). This is a division of labour, not a gap in the
shipped set — but it means "one screen disposition covers the whole graph" is
not yet a true sentence, and it is the reason the hydro manifests cannot be
folded away (maintainer question 2026-07-27: "just use the rebuilt ladder?").

**Precondition now verified:** shared-arc coverage was computed for all 14 rows
(`build_data/water_screen_rebuild/water_unification/check_crossborder_hydro_renomination.py`)
— **14/14 clear the rebuilt rung's 0.5 nomination bar** (river 0.776–1.000,
lake 0.709). So the extension would re-nominate every currently-shipped row.

**Campaign shape (ru1/rg1-class, ~one day):** run the hydro rungs over the
~1,798 cross-border edges, record dispositions (`already-shipped` expected for
all shipped rows, incl. re-nomination of the 3 HydroLAKES candidates whose
frozen adjudications *rejected* them — the rejections replay from the manifest
record, never re-adjudicated live), merge into one whole-graph disposition file.
**Zero expected data change** — nominations only; shipped verdicts are the
frozen adjudications either way. After it, folding `hydro_water`/`hydro_lakes`
into `rescreen_water` (registry 5 → 3) becomes safe if still wanted, since every
border would then be attributable to a recorded whole-graph screen.

*(2026-09-09: the whole-graph attributability check was re-run on the shipped data — 751/751 — and the pre-rebuild discovery nets (near-miss net, lake band, 5/10 km widenings, 2026-07-02/04 hydro cross-checks) were retired as nomination steps; see `docs/METHODS_adjacency.md` §8. "One screen disposition covers the whole graph" is now a true sentence.)*

---

## 4. Known water-only borders below every screen's bar (method-scope omissions)

**Status: recorded 2026-09-21, updated 2026-09-22 and 2026-09-23; no action planned.** Four borders ruled water-only on maps by the
maintainer are nominated by none of the edge screens, and the maintainer took them out of the water-only set: the database
is complete relative to its screens, and a row no recorded screen nominates falls outside the documented method however it
was ruled ("Even we take 100m, we may still miss pairs"; the removal rule of the border-arc specification,
`build_data/arc_and_combined_screens/SPEC_ba1_border_arc_and_screens.md` §10). **They are not found to be land**; the
edges stay in the edge list as ordinary land borders. Kyegegwa↔Ssembabule left when the HydroRIVERS screens took the other
screens' sampling rule (campaign `gs1`, `build_data/geodesic_sampling/`); the other three when every edge screen moved onto
one border arc (campaign `ba1`, `build_data/arc_and_combined_screens/`). Measuring every distance on the ground (campaign `gd1`,
`build_data/geodesic_distances/`) nominates none of them. The shares below are those of the border arc on the current screen
record (`build_data/geodesic_distances/gd1_screens.csv`; the 1,000 m column measured the same way, `omissions_gd1.txt`).

| pair | units | water body | border arc | HydroRIVERS within 500 m | within 1,000 m | Natural Earth river within 20 km | cross-type union | left the set |
|---|---|---|---|---|---|---|---|---|
| `UGA063`↔`UGA107` | Kyegegwa ↔ Ssembabule (Uganda) | River Katonga | 0.7 km, 3 points | 0.00 | 1.00 | 0.00 | 0.00 | 2026-09-21 |
| `JPN038`↔`JPN040` | Chiba ↔ Tokyo (Japan) | Edo River | 20.5 km, 43 points | 0.44 | 0.77 | 0.28 | 0.44 | 2026-09-22 |
| `URY010`↔`URY016` | Montevideo ↔ San José (Uruguay) | Santa Lucía River | 3.2 km, 8 points | 0.25 | 0.88 | 0.00 | 0.25 | 2026-09-22 |
| `VNM026`↔`VNM049` | Haiphong ↔ Quảng Ninh (Viet Nam) | Sông Đá Bạch–Bạch Đằng | 30.8 km, 69 points | 0.39 | 0.65 | 0.00 | 0.48 | 2026-09-22 |

Why the screens miss them: the coverage is a distance to HydroRIVERS' *modelled* centerlines, traced on a 15 arc-second
elevation grid, and on these borders the model runs between 500 m and 1 km from much of the line: on the three river
borders the share within 1,000 m (0.67-0.88) is well above the share within 500 m (0.25-0.44), and the whole 0.7 km
Kyegegwa↔Ssembabule border lies in that band. Those three lie on the lower course of a wide river (the Edo, the Santa Lucía
and the Bạch Đằng estuaries); Natural Earth carries part of one of them (the Edo, 0.28 within 20 km), and HydroLAKES 0.12 of
the Haiphong↔Quảng Ninh border, too little to lift the union to its bar. A screen measured against a surveyed river network (national hydrography, OpenStreetMap waterways) is the route to
them; it is a new screen with its own nominations to adjudicate. A finer step for every screen is not a remedy on its own:
measured over all four screens at 100 m (2026-09-21, `build_data/geodesic_sampling/step_sensitivity_all_screens.py`) it
moved nominations in both directions, and the maintainer kept the 500 m step.

Two borders listed here from 2026-09-21 are no longer omissions: the border arc nominates them again, and both were
re-adjudicated on 2026-09-22 — Lekoumou↔Niari returned to the water-only set by maintainer map ruling, and both passes found
Hai Duong↔Quang Ninh not water-only.

## 5. Deferred by the border-arc specification (2026-09-21)

**Status: recorded; no action planned.** Three items the specification (`build_data/arc_and_combined_screens/`
`SPEC_ba1_border_arc_and_screens.md` §11) set aside by maintainer decision:

- **`border_length_km` from the border arc.** The shipped length is the exact shared line (boundary ∩ boundary), so it
  understates the borders where the polygons overlap or face each other across a channel: Trarza↔Saint Louis ships
  6.4 km against a border arc of 225.3 km, Brakna↔Saint Louis 7.7 km against 223.5 km, Tamaulipas↔Veracruz 29.8 km against
  147.7 km. Computed from the arc, the lengths of those edges and some `narrow_border` / `potential_artifact` flags
  would change; nothing else would.
- **Water split between a river and a lake.** A border whose water is partly river and partly lake, so that neither the
  river screens nor the lake screens reach their bars, and whose cross-type union share lies between 0.50 and 0.80, is
  nominated by no screen (the union's bar is the corridor census's 0.80; a union bar of 0.70
  or 0.90 changed no nomination in the sensitivity run).
- **The 500 m step's aliasing at the bars.** A share is a count of sample points, so a border near a bar can cross it when
  the step changes (measured 2026-09-21 on the 2,456 edges near a bar, `build_data/geodesic_sampling/`
  `analyze_step_sensitivity.py`: a 100 m step moved 37, 8, 32 and 6 edges across the Natural Earth river, Natural Earth
  lake, HydroRIVERS and HydroLAKES bars, in both directions); the step stays at 500 m by maintainer decision.
