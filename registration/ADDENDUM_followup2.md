# Addendum 2 to registration https://osf.io/b2vc9 — exploratory inter-sensor agreement analyses

Written: 17 November 2026, after the registered results (1 October 2026) and the first follow-up (2 October 2026)
were known. Nothing below changes the registered analyses or their results; every analysis here is
**exploratory** and will be reported as such, whichever way it comes out.

## Why
In the registered run, the partition obtained from NOAA-20 detections (fingerprints over 2018–2025) agreed with the
reference partition obtained from S-NPP detections (fingerprints over 2012–2025) at ARI 0.293 on 5,319 cells. That
comparison changed three things at once: the sensor, the length of the record, and the fitted mixture. It cannot say
how much of the disagreement is due to the sensor. The manuscript reports this figure as a cross-sensor check of
VIIRS-derived fire-regime classes, so it has to separate the effect of the sensor from that of the record and to
state at which level of the processing chain agreement is lost. In addition, the comparison of standard-processing (SP) and near-real-time (NRT)
detections planned in the manuscript could not be made in the registered run because the two archives did not
overlap; the SP archive may now cover the hold-out months.

## Data
- The S-NPP and NOAA-20 cell-month cubes built for the registered run, unchanged. Overlap period: April 2018 to
  December 2025 (93 months), both from standard processing, same filters (nominal and high confidence, type 0,
  duplicates removed) and the same 0.01° grid.
- For part E only: detections for the hold-out target months of the registered run (August and September 2026)
  from S-NPP standard processing and from NOAA-20, downloaded from the same FIRMS API and gridded with the unchanged
  `grid_features.py`. A month is used only if the archive covers it completely according to the FIRMS
  data-availability service. For NOAA-20, standard processing is used if it covers the month, otherwise NRT;
  processing levels are never mixed within a month, and the level used is reported.
  The registered S-NPP NRT record ends on 29 September 2026, whereas the alternative targets cover the whole
  month, so part of any disagreement in September can come from that missing day.

## A. Consistency check
The ARI between the registered reference partition and the registered NOAA-20 partition is recomputed from the saved
labels. It must equal the registered value (absolute difference < 1e-9). If it does not, the analysis stops.

## B. Occurrence of fire per cell-month (nominal property)
- y = 1 if the cell has at least one retained detection in the month, for each sensor.
- Domains: (1) the *forecast domain*, the cells fingerprinted in the registered run; (2) the *union domain*, cells
  inside Indonesia with at least 20 retained detections from **either** sensor during the overlap period.
  The forecast domain was selected on S-NPP detections (2012–2025), which can favour S-NPP in the count of
  cell-months seen by one sensor only. The union domain is selected symmetrically. Statements about which sensor
  detects more are taken from the union domain only.
- Statistics from the 2 × 2 table of cell-months: Cohen's kappa, positive agreement 2a/(2a + b + c), negative
  agreement, proportion of agreement, ratio of NOAA-20 to S-NPP positive cell-months, and the share of each sensor's
  positive cell-months also seen by the other.
- Strata: all cells; peat fraction ≥ 0.5 and < 0.5; each calendar year; each registered regime (forecast domain).

## C. Cell-month indicators (cell-months in which both sensors have at least one detection)
- Indicators: number of detections (log), days with detections, longest run of consecutive detection days, night
  fraction, median FRP and summed FRP (log(1 + x)).
- Statistics: mean difference NOAA-20 − S-NPP, standard deviation of differences, 2.5th and 97.5th percentiles of
  differences, Pearson and Spearman correlation, Lin's concordance correlation coefficient (CCC).

## D. Fingerprint features and regime class
- Fingerprints are recomputed with the registered code (`regime_features.fingerprint`, ≥ 20 detections, static
  attributes attached as in the registered run) for S-NPP and for NOAA-20 over the **same** period, April 2018 to
  December 2025.
- Feature agreement on cells present in both: CCC, mean difference and correlation for each fire feature
  (intensity, recurrence, mean and maximum persistence, night fraction, FRP, season sine, cosine and concentration,
  ENSO sensitivity).
- Regime class, with the registered pre-processing and clustering code (`cluster_regimes.Prep`, `fit_one`, seed 0),
  for two partitions: the registered one (method and k from the registered selection) and k-means with k = 4
  (Supplement S1). Comparisons, each as ARI on common cells:
  1. reference (S-NPP, 2012–2025) vs NOAA-20 (2018–2025) — the registered comparison: sensor and period changed;
  2. reference (S-NPP, 2012–2025) vs S-NPP (2018–2025) — period changed, same sensor;
  3. S-NPP vs NOAA-20, both 2018–2025, fitted separately — sensor changed, same period;
  4. the S-NPP model (2018–2025) applied to NOAA-20 fingerprints of the same cells, and the reverse — sensor
     changed, same fitted model;
  5. repeatability under resampling for the 2018–2025 partitions of each sensor (50 subsamples of 80 %, as registered).

## E. The 2026 hold-out under alternative definitions of the measurand
The registered predictions for the 2026 hold-out at L = 1 (saved by the registered run; no model is refitted) are
scored against fire occurrence defined (i) by S-NPP standard processing and (ii) by NOAA-20, in place of the
registered target (S-NPP NRT), on the hold-out months covered completely. Reported: agreement between each alternative
target and the registered target (as in B); AUC-PR, ROC-AUC, Brier score and BSS per model under each target; the
difference in AUC-PR between targets per model; and ΔAUC-PR (MoE − global) with its one-sided p-value under each
target. If an archive does not yet cover a hold-out month completely, that part is reported as not available.

## Evaluation of uncertainty
Percentile intervals (2.5th to 97.5th) from a block bootstrap over 0.5° blocks, 1,000 replicates, with the random
seed fixed in the script. These are Type A evaluations of the component due to spatial sampling.

## Pre-specified reading
- An agreement coefficient (kappa, positive agreement, CCC, ARI) is read as **reproducible** if the lower bound of
  its 95 % interval is ≥ 0.80, as **not reproducible** if the upper bound is < 0.60, and as **partly reproducible**
  otherwise. These thresholds are conventions fixed here; the coefficients and intervals are reported in full.
- The level at which reproducibility is lost is the first level, in the order occurrence → cell-month indicators →
  fingerprint features → regime class, at which a coefficient is not read as reproducible in the forecast domain.
  The union domain is reported alongside.
- The effect of the sensor and the effect of the period on the regime class are reported side by side
  (comparisons 2 and 3) with their intervals; no threshold is applied to their difference.
- H4 is read as not supported under an alternative target if the 95 % interval of ΔAUC-PR (MoE − global) under that
  target includes zero or lies below it, and as supported if it lies above zero. Either outcome is reported.
- No stratum, indicator, threshold or comparison is added after the results are seen without being labelled post hoc.

## Code
`followup2/intersensor.py` (analysis) and `followup2/smoke_followup2.py` (synthetic software test; its output is
not a result). Registered code is imported unchanged.

## Code hashes (fill in at upload)
- followup2/intersensor.py: `0516e746f389b6356b51d13d5e969eda717a009731b66a045da426272cb08395`
- followup2/smoke_followup2.py: `99faf9d09f7d9529456a27a9d9fb38bb5add2b2d5e773c6ab16df17b9db495de`
