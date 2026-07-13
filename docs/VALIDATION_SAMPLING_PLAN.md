# Preregistered sampling plan — validation study of the water-only classification

**Status: preregistration.** This plan is committed BEFORE any sample is
drawn; the commit timestamp is the preregistration record. The sampler
(`scripts/draw_validation_sample.py`) is committed alongside it and is fully
deterministic (fixed seed), so the sample is reproducible from this plan
alone. No sampled row has been seen by the rater at plan-commit time.

## 1. Objective

Estimate, with exact binomial confidence intervals, the error rates of the
2026-07 water-screen rebuild's AI-adjudicated verdicts that shipped without
individual human review:

- **Precision side (finding 2):** the error rate among the **240** shipped
  rescreen rows verified by dual-AI agreement only (`source` contains
  "dual-AI"), i.e. how often a shipped water-only flag is wrong.
- **Recall side (finding 4):** the false-omission rate among adjudicated
  negatives, i.e. how often a genuine water-only border was missed by
  (a) the two-pass AI adjudication and (b) the deterministic mechanism
  auto-reject.

## 2. Populations and strata

- **P (positives, N = 240):** rescreen manifest rows whose `source` contains
  "dual-AI". Stratified by batch (b1 / b2 / b3–b6 / holds) × population
  (cross-border / domestic) × water type (river / lake), **proportional
  allocation** by largest remainder, minimum 1 per non-empty stratum.
  **n = 80.**
- **N1 (AI-adjudicated negatives):** audited pairs whose research AND
  judgment passes both returned `water_only = false` (both-false pairs never
  reached a human worksheet by construction). Two strata:
  - **N1-high** (risk stratum): maximum water-screen fraction ≥ 0.50 —
    where a missed genuine border is most likely to hide. **n = 40.**
  - **N1-rest**: maximum fraction < 0.50. **n = 40.**
- **N2 (mechanism-rejected):** the 288 pairs auto-rejected by the
  deterministic <0.20 rule (`hybrid_autoreject_ledger.csv`). Simple random
  sample, **n = 20.**

Total rater workload: **180 rows.**

## 3. Randomization

Single fixed seed **20260713** (Python `random.Random(20260713)`), applied
after sorting each stratum by pair id (deterministic input order). The
sampler emits the two worksheets and a manifest of drawn ids; re-running it
must reproduce the identical sample byte-for-byte.

## 4. Rating protocol

The rater judges each row against authoritative map evidence (the same
protocol as the audit worksheets), **blind to the AI verdict direction where
possible** (worksheets present the pair and geometry context first; the AI
verdict is recorded in a trailing column for reference). Verdict vocabulary:

- `genuine_water_only` = **Yes** — essentially the entire shared border (or
  gap) follows one water feature (the shipped rubric, >~20% dry ⇒ No);
- **No** — mixed or land border;
- **NA** — cannot determine / special administrative geometry (recorded,
  excluded from the denominator, count reported).

An error is: a **P** row rated No (false positive), or an **N1/N2** row
rated Yes (false negative / wrong auto-reject).

**Inter-rater subsample:** 15 P rows + 15 N rows (seeded draw from the
sampled sets) may be independently rated by a second rater; Cohen's kappa is
reported if a second rater is available, otherwise the study reports
single-expert rating under this published protocol.

## 5. Analysis (prespecified)

- Per population: error count k of n → point estimate k/n and **exact 95%
  Clopper–Pearson interval**. Zero errors in n reports the one-sided 95%
  upper bound (≈ 3.7% at n = 80; ≈ 13.9% at n = 20).
- N1 strata are reported separately AND combined with stratum weights
  proportional to population sizes.
- No pass/fail gate: this is an estimation study. Every confirmed error is
  additionally fixed through the normal reviewed-manifest path and reported
  in the study record (`docs/VALIDATION_STUDY.md`, written after rating).

## 6. What this study does and does not claim

It estimates the error rates of the *audit machinery* on the rebuild's
populations, turning the graded provenance tiers (human-verified / dual-AI /
mechanism) into quantified uncertainty. It does not certify the upstream
World Bank geometry, the pre-rebuild overlays (100% human-verified at ship
time), or borders outside the candidate screens' documented dataset floors
(see `docs/REPRODUCING.md` §6).
