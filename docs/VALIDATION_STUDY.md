# Validation study — water-only classification (results)

Preregistered design: `docs/VALIDATION_SAMPLING_PLAN.md` (committed 2026-07-13,
before the sample was drawn; seed 20260713; sampler
`scripts/draw_validation_sample.py`, byte-identical across reruns). Rating:
primary rater (maintainer) over all 180 sampled rows, 2026-07-14; independent
second rater over the 30-row blind subsample (verdict-direction cues removed).
An error is a shipped positive rated *not* water-only, or a sampled negative
rated water-only. NA rows are excluded from denominators and counted below;
the preregistered NA rule covers rows the rater cannot determine or special
administrative geometry. The placeholder-unit exclusions follow from unit
identity alone — the rule was fixed at preregistration and these rows were
set NA on encountering the placeholder names, **before any validation
outcome for them could be inspected** — and all estimates therefore apply
to the non-placeholder population only; inference excludes placeholder-unit
pairs. **All six NAs are itemized:** the positive sample's
n = 80 reduces to the analyzed n = 76 because four sampled rows involve World
Bank placeholder units the rater cannot locate on authoritative maps
(`KHM000<->KHM018`, `MWI002<->MWI004`, `MWI003<->MWI004`, `MWI004<->TZA022` —
all "Area under National Administration"); the negative sample loses
`RUS024<->RUS050` ("Name Unknown" placeholder) and `MLT009<->MLT055`
(rater-identified non-adjacency, recorded to the future-edge-audit list).

## Results

| population | n rated | NA | errors | estimate | 95% interval¹ |
|---|---|---|---|---|---|
| **P** — shipped dual-AI rows (of 240) | 76 | 4 | 1 | false-positive rate **1.3%** → precision **98.7%** | error [0.03%, 7.1%]; precision **[92.9%, 99.97%]** |
| **N1-high** — both-false negatives, max screen fraction ≥ 0.50 (of 566) | 39 | 1 | 1 | missed-border rate **2.6%** | [0.07%, 13.5%] |
| **N1-rest** — both-false negatives, fraction < 0.50 (of 788) | 39 | 1 | 0 | **0%** | ≤ 7.4% (one-sided) |
| **N2** — deterministic mechanism auto-reject (of 288) | 20 | 0 | 0 | **0%** | ≤ 13.9% (one-sided) |
| N1 combined (population-weighted) | 78 | 2 | 1 | false-omission **1.07%** | conservative combined [0.03%, 10.9%] |

¹ Per-population intervals are exact Clopper–Pearson (two-sided; one-sided
upper bound where zero errors were observed). The N1-combined interval is a
**conservative combined bound** formed by weighting the per-stratum
Clopper–Pearson limits — it is *not* an exact interval; only the individual
per-stratum bounds are exact.

**Inter-rater agreement** (30 blind double-rated rows): 28/30 = 93.3%,
**Cohen's κ = 0.867**. The two disagreements, resolved by the preregistered
primary-rater rule and recorded:

- `IRN013<->IRN020` (Ilam–Lorestan, Seymareh): primary Yes (ships), second
  rater No — flagged for future re-audit.
- `LVA024<->LVA053` (Dagdas–Krāslavas): primary No (stays unshipped), second
  rater Yes — attributed to the Latvia subdivision-vintage issue below.

## Errors found and corrected

Both errors were fixed through the normal reviewed-manifest path (the study
estimates above are computed on the *pre-fix* verdicts, as preregistered):

1. **False positive — `NIC016<->NIC017`** (Río San Juan–Rivas, "Lake
   Nicaragua"): the two departments share a land border south of Lago
   Cocibolca, so the border is mixed, not water-only. Row **demoted** —
   removed from `rescreen_water_overlay_pairs.csv` and
   `water_separated_pairs.csv`; the pair remains an ordinary land edge,
   pericoupled under every `coupling_standard`.
2. **False negative — `ITA005<->ITA020`** (Emilia-Romagna–Veneto): the border
   follows the **Po River** almost in its entirety; both AI passes had called
   it not water-only. Row **added** to `rescreen_water_overlay_pairs.csv`
   (human-map-verified), with the four-layer bridge protocol: OSM screen and
   agent verification agree `has_bridge = True` (SS16 Pontelagoscuro and A13
   Occhiobello crossings, both landing in the two regions).

Net count effect: water-only total unchanged at **698**; bridge split
319/379 → **320/378**; ADM1 moderate 8,087 → **8,088**; everything else
unchanged. After the corrections the shipped rescreen rows verified by
dual-AI agreement only number **238** (of 384); the study population was the
240 at draw time.

## Estimators and intervals (as computed)

- **Per-population rates**: point estimate k/n on the analyzed denominator;
  exact two-sided 95% Clopper–Pearson intervals (`scipy.stats.beta.ppf`);
  for k = 0 the table shows the one-sided 95% upper bound (preregistered).
- **Precision** is reported both ways: error rate 1/76 = 1.32% with exact CI
  [0.03%, 7.11%], hence precision 75/76 = **98.68%** with exact CI
  **[92.89%, 99.97%]** (the complement interval).
- **Weighted false-omission** applies to the **sampled candidate-negative
  frame only** — the 1,354 screen-nominated pairs both AI passes adjudicated
  not-water-only — *not* to overall database recall. Estimator:
  p̂ = w_high·p̂_high + w_rest·p̂_rest with population weights
  w_high = 566/1354 = 0.418 and w_rest = 788/1354 = 0.582, giving
  p̂ = 0.418·(1/39) + 0.582·0 = **1.07%**. Because the rest stratum observed
  zero events, a normal-approximation variance would be degenerate; we
  report a **conservative combined interval** (not exact — only the
  per-stratum Clopper–Pearson bounds are exact) formed by weighting the
  stratum bounds: [0.418·0.06% + 0.582·0, 0.418·13.48% +
  0.582·9.03%] = **[0.03%, 10.9%]** (labelled post-hoc: the plan
  prespecified per-stratum CIs and the weighted point estimate; the combined
  interval method is an addition; for the zero-event rest stratum the
  combination deliberately uses the two-sided 97.5% upper limit 9.03% rather
  than the one-sided 7.4% shown in the table — the more conservative choice,
  consistent with the bound's "conservative" label). Inference beyond this frame — to borders
  never nominated by any screen — is a *coverage* claim governed by the
  datasets' documented floors (`docs/REPRODUCING.md` §6), not by this
  estimator; the mechanism-rejected stratum is bounded separately (0/20,
  ≤ 13.9%).
- **Inter-rater reliability**: 2×2 agreement table over the 30 blind
  double-rated rows —

  | | second: Yes | second: No |
  |---|---|---|
  | **primary: Yes** | 14 | 1 |
  | **primary: No** | 1 | 14 |

  Observed agreement 28/30 = 93.3%; chance agreement 0.50 (balanced
  marginals); **κ = 0.867**, asymptotic SE 0.091, 95% CI **[0.69, 1.00]**.
  n = 30 is adequate for a reliability check but small — the interval is
  wide, and κ here characterizes this rater pair on this sample rather than
  a general rater population.

## Interpretation

- The dual-AI auto-accept tier's measured precision (**98.7%**, 95% CI lower
  bound ≈ 92.9%) is consistent with the human-review outcomes on the
  disagreement/medium tier during the audit itself, and the single confirmed
  false positive was found and removed.
- The false-omission estimate (**1.07%** weighted, conservative 95% bound
  [0.03%, 10.9%]; concentrated in the high-signal stratum, as designed)
  quantifies what the candidate-coverage framing predicts: within the
  audited candidate frame, misses are rare and cluster where screen signal
  was high but both AI passes read the border as mixed.
- The mechanism auto-reject (sub-0.20 on all four datasets) showed **no
  error** in its sample, supporting its use as a deterministic disposal rule.
- **NA concentration:** 5 of 6 NAs are World Bank placeholder units ("Area
  under National Administration", "Name Unknown") — a documented naming
  artifact of the source, not a classification failure; one NA is a
  rater-identified non-adjacency (Malta, `MLT009<->MLT055`) recorded to the
  future-edge-audit list.
- **Subdivision vintage (Latvia):** the WB 2026-05-14 release carries
  Latvia's pre-2021 subdivisions (119 novadi); Latvia's 2021 reform
  consolidated these to 43 (e.g. Dagdas novads was merged into Krāslavas
  novads). Ratings were made in the dataset's frame. This is a general
  caveat: ADM1 vintages follow the pinned WB release, not later national
  reforms.

## Provenance of this record

Worksheets, drawn-id manifest, and rater files:
`build_data/water_screen_rebuild/validation/` (local audit archive; summary
here is the citable record). Analysis code: exact Clopper–Pearson intervals
(`scipy.stats.beta`); kappa on the 30-row overlap.
