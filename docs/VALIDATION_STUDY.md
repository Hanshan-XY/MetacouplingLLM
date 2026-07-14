# Validation study — water-only classification (results)

Preregistered design: `docs/VALIDATION_SAMPLING_PLAN.md` (committed 2026-07-13,
before the sample was drawn; seed 20260713; sampler
`scripts/draw_validation_sample.py`, byte-identical across reruns). Rating:
primary rater (maintainer) over all 180 sampled rows, 2026-07-14; independent
second rater over the 30-row blind subsample (verdict-direction cues removed).
An error is a shipped positive rated *not* water-only, or a sampled negative
rated water-only. NA rows (placeholder units such as "Area under National
Administration"/"Name Unknown", or non-adjacent-by-rater) are excluded from
denominators and counted below.

## Results

| population | n rated | NA | errors | estimate | exact 95% CI |
|---|---|---|---|---|---|
| **P** — shipped dual-AI rows (of 240) | 76 | 4 | 1 | false-positive rate **1.3%** → precision **98.7%** | [0.03%, 7.1%] |
| **N1-high** — both-false negatives, max screen fraction ≥ 0.50 (of 566) | 39 | 1 | 1 | missed-border rate **2.6%** | [0.07%, 13.5%] |
| **N1-rest** — both-false negatives, fraction < 0.50 (of 788) | 39 | 1 | 0 | **0%** | ≤ 7.4% (one-sided) |
| **N2** — deterministic mechanism auto-reject (of 288) | 20 | 0 | 0 | **0%** | ≤ 13.9% (one-sided) |
| N1 combined (population-weighted 566/788) | 78 | 2 | 1 | false-omission **≈ 1.1%** | — |

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
dual-AI agreement only number **239** (of 335); the study population was the
240 at draw time.

## Interpretation

- The dual-AI auto-accept tier's measured precision (**98.7%**, 95% CI lower
  bound ≈ 92.9%) is consistent with the human-review outcomes on the
  disagreement/medium tier during the audit itself, and the single confirmed
  false positive was found and removed.
- The false-omission estimate (**≈ 1.1%** weighted; concentrated in the
  high-signal stratum, as designed) quantifies what the candidate-coverage
  framing predicts: misses are rare and cluster where screen signal was high
  but both AI passes read the border as mixed.
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
