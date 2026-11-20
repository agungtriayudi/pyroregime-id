# Addendum to registration https://osf.io/b2vc9 — exploratory follow-up analyses

Written: 1 October 2026, after the registered results were known. Nothing below changes the registered
analyses or their results; every analysis here is **exploratory** and will be reported as such.

## Why
The registered run showed that hydroclimate added substantial skill beyond AlphaEarth embeddings for monthly
peat-fire forecasting (H2), while fire regimes were weakly separated and regime conditioning did not improve
forecasts (H4 not supported). The paper is framed around the embedding finding. Reviewers can reasonably ask
whether embeddings fail only for within-year timing, or also for the static susceptibility task in which they
matched physical variables in Victoria (Zhuang et al., 2026). The analyses below answer that question and
complete reporting promised in the manuscript.

## A. Static susceptibility (exploratory; design fixed here)
- Unit: 0.01° cells in the registered regions with LSIB Indonesia fraction ≥ 0.5 and WorldCover water < 0.5.
  Sample: all cells with fire detections used in the registered run (candidate cells) plus a systematic hash-based
  10 % sample of all other land cells: keep a cell if `((cell × 2654435761) mod 2^32) mod 10 = 0`.
- Target: y = 1 if the cell has ≥ 1 retained VIIRS S-NPP detection between January 2019 and December 2025.
- Predictor sets (all precede the target period):
  - P (physical, static): peat fraction (same peat map as the registered run), WorldCover class fractions,
    SRTM elevation and slope, CHIRPS mean annual rainfall 2012–2018, mean number of Aug–Oct days < 1 mm 2012–2018,
    ERA5-Land mean Aug–Oct top-layer soil moisture 2012–2018, ERA5-Land mean 2 m temperature 2012–2018.
  - E (embeddings): mean of AlphaEarth 2017 and 2018 annual embeddings (64 bands), aggregated to the cell with the
    same method as the registered run.
  - F (fire history): log(1 + detections 2012–2018) and number of years with detections 2012–2018.
  - Sets compared: P, E, P+E, F, F+P, F+E, F+P+E.
- Learner and evaluation: histogram gradient boosting with the registered hyperparameters (300 iterations,
  learning rate 0.05, 31 leaves, L2 = 1.0, early stopping on 10 %); 5-fold cross-validation grouped by 0.5° blocks;
  AUC-PR (primary) and ROC-AUC; block bootstrap over 0.5° blocks, 1,000 replicates.
- Reported differences: E − P, (P+E) − P, (P+E) − E, (F+P+E) − (F+P), (F+E) − (F+P); all cells and peat cells
  (peat fraction ≥ 0.5) separately.
- Pre-specified reading: embeddings are judged comparable to physical static predictors for susceptibility if the
  95 % interval of AUC-PR(E) − AUC-PR(P) includes zero or lies above zero. Either outcome will be reported.
- Code: `followup/susceptibility.py` (SHA-256 recorded below at upload).

## B. Supplement S1 — k-means with k = 4 (exploratory)
The prose summary of the k-selection rule in the original registration (highest silhouette among solutions with
bootstrap stability ≥ 0.80) differed from the registered code, which was followed. Under the prose rule, k-means with
k = 4 would have been selected. We report that partition: fingerprints, map, seasonality, stability (50 subsamples),
ARI with the registered partition, and the 2026 hold-out at L = 1 for global, global + regime and mixture-of-experts
models built on it, with the registered forecasting code and settings. Code: `followup/supplement_s1.py`.

## C. Figure redraws (descriptive; no new inference)
Figures 3, 5 and 6 are redrawn from saved registered outputs with a 12-colour palette, per-regime panels, readable
transition matrices and a 2023 → 2026 change map; newly burning cells are tabulated by province (ADM1).
Code: `followup/fig_redraw.py`.

## Code hashes (SHA-256, computed 2026-10-02 before upload)
- followup/susceptibility.py: `bc4c799d8cb5e070a3742bb01526c864aa7818a38c6a6a4b7c53702cc7b3eb90`
- followup/supplement_s1.py: `082b81e061217b354eb80d7915b81fd749edd34c0ba6233a307ca338b2ef8fd0`
- followup/fig_redraw.py: `715ae177f92aff0047cac4cfb1d29c791c1de379344c2805ce0fff2cd7d26608`
