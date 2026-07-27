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
