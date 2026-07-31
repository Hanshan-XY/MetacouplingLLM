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
standard cross-vendor mini-batch (Codex research → Sonnet-5 adversarial judge →
deterministic measurement → maintainer gate) and **both ruled NOT water-only —
zero data change** (`newcand_rulings.json`): `NER002`↔`TCD010` by dual-AI
agreement (IBS-73 surveyed segments + the Northern Pool's post-1970s dryness);
`BRA025`↔`URY014` by **maintainer official-map ruling** (land border, especially
the straight-line segment) — overruling the judge's water_only=true (10.2% dry
< the 20% bar) and the measurement lean (union water 0.861 @500 m), recorded
verbatim per the evidence hierarchy. **Every nomination in the whole-graph
screen record is now adjudicated; none pending.** The section below is retained
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
