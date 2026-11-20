# Addendum 3 to registration https://osf.io/b2vc9 — disagreement components in place of kappa (post hoc)

Written: 20 November 2026, before `followup3/disagreement.py` was run on real data, but **after** the results of the
registered run (1 October 2026), the first follow-up (2 October 2026) and the second follow-up (17 November 2026)
were known. Every analysis below is **exploratory and post hoc**, and will be reported as such.

## Why
Addendum 2 (https://osf.io/8vr7f) summarised the agreement of S-NPP and NOAA-20 on monthly fire occurrence with
Cohen's kappa, positive agreement and related statistics. After submission, an editor pointed out that kappa should
not be used to compare maps (Pontius and Millones, 2011, Int. J. Remote Sens. 32, 4407–4429,
https://doi.org/10.1080/01431161.2011.552923; Foody, 2020, Remote Sens. Environ. 239, 111630,
https://doi.org/10.1016/j.rse.2019.111630). The same 2 × 2 tables are therefore re-expressed as disagreement
components. The kappa values of Addendum 2 remain in `results/followup2/intersensor.json` and will be reported in
the supplementary material, marked as superseded.

## What is computed
For each 2 × 2 table of Addendum 2, part B (fire in both records *a*, in S-NPP only *b*, in NOAA-20 only *c*, in
neither *d*, *n* = *a* + *b* + *c* + *d*):

- total disagreement D = (b + c) / n,
- quantity disagreement Q = |b − c| / n,
- allocation disagreement A = 2 min(b, c) / n, so that D = Q + A (Pontius and Millones, 2011),
- the share of the disagreement due to allocation, A / D,
- the disagreement among cell-months with fire in at least one record, (b + c) / (a + b + c).

The same quantities are computed for the comparison of the registered 2026 hold-out target (S-NPP NRT) with NOAA-20
(Addendum 2, part E).

## Data, domains, strata, uncertainty
Unchanged from Addendum 2: April 2018 to December 2025, forecast domain and union domain, strata all / peat / non-peat
/ calendar year / registered regime, 95 % percentile intervals from a block bootstrap over 0.5° blocks with 1,000
replicates. For part B the bootstrap uses the seed and the order of draws of Addendum 2, so the replicates are those
of Addendum 2; the script verifies this by recomputing the Addendum-2 intervals of kappa, positive agreement and
proportion of agreement. For the hold-out comparison a separate fixed seed is used.

## Reading
No threshold is applied to D, Q or A; the values and intervals are reported in full. The reading bands of Addendum 2
continue to apply to positive agreement, CCC and ARI. No stratum, domain or statistic is added beyond those listed here.

## Code
`followup3/disagreement.py` (imports the registered `grid_features.cell_center`; copies the follow-up-2 functions it
needs without changing them). Output: `results/followup3/disagreement.json`. Nothing else under `results/` changes.
