# Pre-registration — Indonesian peatland fire regimes

Registered (UTC): 2026-10-01T03:25:19+00:00
Parameter hash (SHA-256): `2fe77e61dab9ff5ac2d3faeae0aaf0cc7b2983fb7e9c2df65d09e7cd18bf69e0`
Analysis-code hash (SHA-256): `1d3d980960838a577d4892f607380a366cc69fc057dbdd1dd16d82e07643ea86`

## Primary confirmatory hypothesis
H4: on the 2026 hold-out (training targets up to 2026-07, evaluation targets after it), the
regime-conditioned mixture of experts has higher AUC-PR than a single global model at lead L = 1. Test: paired
block bootstrap over 0.5° blocks (1000 replicates); one-sided p = share of replicates with difference ≤ 0.
All other comparisons are exploratory.

## Fixed analysis choices
- Reference period 2012–2025; cells with ≥ 20 retained VIIRS detections.
- Number of regimes k in 3-12: among solutions with bootstrap stability ARI ≥ 0.80
  (50 subsamples of 80 %), the highest silhouette. Algorithms: kmeans, gmm, ae_kmeans, hdbscan.
- Shift analysis: El Niño years [2015, 2019, 2023, 2026], months 1-10, ≥ 3 detections;
  neutral-year baseline [2017, 2018, 2021, 2022].
- Leave-one-El-Niño-out years: [2015, 2019, 2023]. Lead times: [1, 2, 3] months.
- Gradient boosting: 300 iterations, learning rate 0.05, 31 leaves, L2 = 1.0, early stopping on 10 %; no re-weighting.

## Full parameter set
```json
{
 "ref_years": [
  2012,
  2025
 ],
 "min_det": 20,
 "k_range": "3-12",
 "methods": [
  "kmeans",
  "gmm",
  "ae_kmeans",
  "hdbscan"
 ],
 "cluster_boot": 50,
 "shift_years": [
  2015,
  2019,
  2023,
  2026
 ],
 "neutral_years": [
  2017,
  2018,
  2021,
  2022
 ],
 "event_months": "1-10",
 "event_min_det": 3,
 "enso_years": [
  2015,
  2019,
  2023
 ],
 "holdout_train_end": "2026-07",
 "leads": [
  1,
  2,
  3
 ],
 "bootstrap": 1000,
 "neg_frac": 1.0,
 "panel_start": "2012-01",
 "regions": {
  "sumatra": [
   95.0,
   -6.2,
   106.3,
   6.0
  ],
  "kalimantan": [
   108.5,
   -4.3,
   119.3,
   4.5
  ],
  "papua_selatan": [
   137.5,
   -9.2,
   141.1,
   -5.5
  ]
 }
}
```
