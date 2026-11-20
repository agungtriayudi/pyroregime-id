# Run report

Process report of the registered run and of the three exploratory follow-ups: facts only, without interpretation of
the results. Sections 1–3 are translated from the Indonesian reports written at the time (`RUN_REPORT.md`,
`FOLLOWUP_REPORT.md`, `FOLLOWUP2_REPORT.md`), compiled from `data/provenance.json`, the log files, `CHANGELOG.md` and
`DEVIATIONS.md`; section 4 was added for the public release. The provenance file of the published run
(`data/provenance.json` at the time) is included, translated, as `docs/data_provenance.json`; the log files are not
included. Outputs of the smoke tests (synthetic data) are not quoted. Times are Western Indonesian Time (WIB = UTC+7)
unless marked UTC.

## 1. Registered run (30 September – 1 October 2026)

### 1.1 Status

| Item | Status |
| --- | --- |
| Smoke test (`smoke_test.py`) | passed (2026-09-30 21:21–21:24) |
| Data verification (`fetch_data.py --stage verify`) | passed, "all inputs complete" (2026-10-01 10:24) |
| Resource check | panel of 4,796,169 rows ≈ 1.6 GB (10 % of 16 GB RAM); dry run of `build_panel` peaked at 3.9 GB RSS; analysis code not changed |
| Local registration (`register.py`) | 2026-10-01T03:25:19Z; parameter hash `2fe77e61…bf69e0`, code hash `1d3d9809…43ea86` |
| OSF registration | https://osf.io/b2vc9 (registered 2026-10-01T03:36:09Z); project https://osf.io/5wyek, `REGISTRATION.md` uploaded 03:34:17Z, SHA-256 identical to the local file |
| Analysis (`run_all.py`) | finished 2026-10-01 21:49; the log ends with `46/46 tokens have a value` and `DONE (registered)` |

### 1.2 Environment

- macOS (Apple Silicon, 10 cores, 16 GB RAM); Python 3.12.14 through `uv` in `.venv/`; dependencies from
  `requirements.txt` (pandas 3.0.6, NumPy 2.5.3, scikit-learn 1.9.1, earthengine-api, cdsapi, geopandas, rasterio,
  xarray).
- `cluster_regimes.py --method idec` was not used (not part of `config.json: methods`); PyTorch was not installed.

### 1.3 Data coverage by source

| Source | ID / URL | Coverage used |
| --- | --- | --- |
| FIRMS VIIRS S-NPP SP | FIRMS area API, `VIIRS_SNPP_SP` | 2012-01-20 to 2026-06-30 |
| FIRMS VIIRS S-NPP NRT | `VIIRS_SNPP_NRT` | 2026-07-01 to 2026-09-29 (downloads stop at "yesterday" = 2026-09-29) |
| FIRMS VIIRS NOAA-20 SP | `VIIRS_NOAA20_SP` | requested 2018-01-01 to 2025-12-31; data available from 2018-04-01 |
| ONI | NOAA CPC `oni.ascii.txt` | to the JJA 2026 season |
| DMI | NOAA CPC monthly IOD index, ERSSTv6, 1991–2020 climatology | 1950-01 to 2026-08 (**source changed**, 1.6) |
| FWI | CEMS `cems-fire-historical-v1` through the Copernicus EWDS (`fwinx`, 0.25°, reanalysis v4.1; 2026 = `intermediate_dataset`) | 2012-01 to 2026-09, 27,097 cells of the forecast domain, 1.7 % missing (coastal cells) |
| MODIS MCD64A1 (burned area) | Earth Engine `MODIS/061/MCD64A1` | to 2026-07 (2026-08 empty) |
| MODIS MOD14A1/MYD14A1 | Earth Engine `MODIS/061/MOD14A1`, `MYD14A1` | to 2026-09-21 (used to 2026-08) |
| CHIRPS daily | Earth Engine `UCSB-CHG/CHIRPS/DAILY` | to 2026-08-31 |
| ERA5-Land monthly | Earth Engine `ECMWF/ERA5_LAND/MONTHLY_AGGR` | to 2026-08 |
| Monthly Earth Engine table | `data/cube/gee_monthly.csv.gz` | 18,801,728 cell-month rows, 2012-01 to 2026-08 (106,828 candidate cells) |
| ESA WorldCover | Earth Engine `ESA/WorldCover/v200` | once (static) |
| Country boundaries | Earth Engine `USDOS/LSIB_SIMPLE/2017` | once (static) |
| AlphaEarth | Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | 2017–2025 (**aggregation changed**, 1.6) |
| Peat map | PEATMAP (Xu et al., 2018), DOI 10.5518/252, CC BY 4.0 | `data/raw/peat/peatmap_indonesia.gpkg` (**source changed**, 1.6) |
| Static table | `data/cube/static.csv` | 961,290 rows (106,810 cells × 2017–2025) |

### 1.4 Cells and detections

| Source | Region | Files | Raw detections |
| --- | --- | --- | --- |
| VIIRS_SNPP_SP | sumatra | 1,056 | 1,324,779 |
| VIIRS_SNPP_SP | kalimantan | 1,056 | 1,335,756 |
| VIIRS_SNPP_SP | papua_selatan | 1,056 | 220,257 |
| VIIRS_SNPP_NRT | sumatra | 19 | 46,400 |
| VIIRS_SNPP_NRT | kalimantan | 19 | 214,976 |
| VIIRS_SNPP_NRT | papua_selatan | 19 | 42,109 |
| VIIRS_NOAA20_SP | sumatra | 585 | 356,280 |
| VIIRS_NOAA20_SP | kalimantan | 585 | 486,984 |
| VIIRS_NOAA20_SP | papua_selatan | 585 | 86,106 |

- Retained after the filters of `grid_features.py` (confidence nominal or high, `type` = 0 where available, unique per
  location and time): S-NPP (SP + NRT) 2,926,988 detections; NOAA-20 SP 825,097 detections.
- S-NPP cell-month cube: 1,154,094 rows, 386,308 cells, 2012-01 to 2026-09; 290,558 detections from the NRT months
  (2026-07 to 2026-09).
- NOAA-20 cell-month cube: 395,270 rows, 204,062 cells, 2018-04 to 2025-12.
- 106,828 candidate cells; forecast domain of 27,097 cells (before the LSIB mask); fingerprints 2012–2025 after the
  LSIB mask: 26,026 cells.

### 1.5 SP/NRT ratio

Not computable: the S-NPP NRT archive starts on 2026-07-01 and the SP archive ends on 2026-06-30, so no month overlaps
(`data/provenance.json: sp_nrt_overlap`).

### 1.6 Changes of IDs, URLs, data sources and data-retrieval code

Recorded in `data/provenance.json: changes` and `CHANGELOG.md`. All were made before registration; no results had been
seen.

1. **DMI**: `dmi_urls` changed from NOAA PSL (HadISST1.1; ending 2026-05, second URL HTTP 404) to the monthly NOAA CPC
   IOD index (ERSSTv6, 1991–2020 climatology; 1950-01 to 2026-08); parser `parse_cpc_dmi()` added.
2. **FWI**: the dataset `cems-fire-historical-v1` is no longer in the CDS catalogue; it was taken from the Copernicus
   EWDS (`fwi_cds.url`) with the same request schema.
3. **Peat map (user decision, option "global map")**: the 2019 BBSDLP map (Anda et al., 2021) could not be obtained
   legally with a licence; PEATMAP was used (Xu et al., 2018; CC BY 4.0). According to PEATMAP, its Indonesian part
   comes from Ritung et al. (2011), BBSDLP 1:250,000, and its Malaysian part of Borneo from Wetlands International
   (2010). `Asia.zip` SHA-256 `8767e9ba…b8724e`; clipped to the bounding boxes in `regions` (1,186 polygons).
4. **Aggregation of the AlphaEarth embeddings (user decision)**: the cell mean of a systematic sample of 10 × 10
   points (0.001° grid) instead of all 10 m pixels, because the full mean exceeded the memory and quota of the Earth
   Engine project (`computePixels`, `reduceRegions`, batch export to Drive with `tileScale` 8; the project asset root
   was not available). Validation on 800 cells (Riau and Central Kalimantan, 2023): per-band correlation median 0.999
   (minimum 0.996), mean error 2.7 % of the between-cell SD, correlation of the 8 principal components 0.997–1.000
   (`logs/emb_sample_validation.txt`).
5. Other data-retrieval code (details in `CHANGELOG.md`): parallel FIRMS downloads with a rate limiter and handling of
   the "Invalid MAP_KEY" answer when the quota is full; the SP/NRT overlap test skipped when no month overlaps;
   `gee_fetch.py` (months of the last year up to `panel_end` − 1, `reduceResolution` of MCD64A1 only where data
   exist, masked pixels −inf → NaN, option `--parts`, request deadline); `fetch_data.stage_grid` creates `data/cube/`;
   `peat_fraction.py` drops 162 static rows without embeddings (18 open-sea cells, `in_indonesia` = 0).

### 1.7 Deviations and registration notes

- No deviation from the registered design (`DEVIATIONS.md`); the analysis parameters and the `ANALYSIS_CODE` files
  were not changed.
- Inconsistency between documents (recorded 2026-10-01 10:59 WIB, before any analysis output existed): the text of
  `REGISTRATION.md` states the selection of k as "highest silhouette among stable solutions", whereas the registered
  code (`cluster_regimes.py: select_k`) uses the minimum GMM BIC among stable GMM solutions (bootstrap ARI ≥ 0.80), the
  knee of the BIC curve if the minimum lies at the upper end, and the silhouette only as a fallback. The registered code
  was run.

### 1.8 Warnings

- Data retrieval:
  - `climate_fetch.py`: "WARNING: dmi only until 2026-08, panel until 2026-09"; predictors are needed only until
    2026-08 (`verify` passed).
  - The FIRMS NRT files have no `type` column, so the filter of static thermal sources (volcanoes, gas flares) in
    `grid_features.py` does not apply to the detections of 2026-07 to 2026-09.
  - FIRMS September 2026 only until 2026-09-29.
  - MCD64A1 not yet available for 2026-08 (`burned_frac` empty in that month).
  - FWI missing in 1.7 % of cell-month rows (coastal cells outside the FWI land grid).
  - Static Earth Engine layers: 189 embedding parts failed with "User memory limit exceeded" and the process hung
    (stopped at 03:55) before the aggregation method was changed (1.6, item 4); in the final run 0 parts failed.
  - The `gee_static` process was interrupted once because the Claude Code session ended (06:15) and was resumed at 08:56
    from the saved parts.
- Analysis (`run_all.py`, warning lines of `logs/run_all.log`):
  - `sklearn/neural_network/_multilayer_perceptron.py: ConvergenceWarning: Stochastic Optimizer: Maximum iterations
    (400) reached and the optimization hasn't converged yet.` (× 148)
  - `sklearn/mixture/_base.py: ConvergenceWarning: Best performing initialization did not converge.` (× 3)
  - `sklearn/cluster/_hdbscan/hdbscan.py: FutureWarning: The default value of 'copy' will change from False to True in
    1.10.`
  - "WARNING: static year 2014 missing; land attributes averaged, embeddings left empty" (and the same for 2016).
- Sleep of the computer during the analysis (lid closed): 12:52–13:03, 13:19–13:28 and 15:39–17:28; `run_all.py` was
  delayed by 11 min, 8 min and 1 h 49 min.

### 1.9 Duration of each stage

| Stage | Start | End | Duration | Note |
| --- | --- | --- | --- | --- |
| smoke_test | 09-30 21:21 | 09-30 21:24 | 3 min | passed |
| indices (ONI, DMI) | 09-30 21:23 | 09-30 21:23 | < 1 min | |
| firms | 09-30 21:28 | 09-30 23:08 | 1 h 40 min | failed once at 21:43 (quota), resumed 21:44 |
| grid | 09-30 23:08 | 09-30 23:11 | 3 min | failed once (folder `data/cube/` missing) |
| candidates | 09-30 23:11 | 09-30 23:11 | 3 s | |
| gee_monthly | 09-30 23:11 | 10-01 00:02 | 51 min | |
| fwi (EWDS) | 09-30 23:11 | 10-01 00:09 | 58 min | in parallel with gee_monthly |
| gee_static | 10-01 00:02 | 10-01 10:21 | 10 h 18 min (wall clock) | includes the failed attempts 00:02–03:55, method tests and the session gap 06:15–08:56 |
| peat | 10-01 10:21 | 10-01 10:21 | 31 s | repeated 10:23–10:24 after the static fix |
| verify | 10-01 10:21 | 10-01 10:24 | 4 s | repeated 10:24 |
| register | 10-01 10:25 | 10-01 10:25 | < 1 s | |
| run_all | 10-01 10:40 | 10-01 21:49 | 11 h 9 min (wall clock); 8 h 59 min without the sleep periods | |

## 2. Follow-up 1 (2 October 2026, exploratory)

Design: `registration/ADDENDUM_followup.md`. All analyses are exploratory and outside the registered analysis.

| Item | Status |
| --- | --- |
| Smoke test `followup/smoke_followup.py` | passed (2026-10-02 05:45–05:46) |
| Registered hashes | still matched `registration/registration.json` (checked before and after the change of `config.json`) |
| Registered `run_all.py` and `results/` | not re-run; new outputs only in `results/followup/` |
| Check of `fig_redraw.json` | the numbers of appeared and disappeared cells 2023→2026 **equal** those in `results.json` (`shift.2023_2026`); the province table sums to the number of appeared cells |
| Addendum | `registration/ADDENDUM_followup.md` with the SHA-256 of the scripts filled in on 2026-10-02; uploaded by the user to the OSF project https://osf.io/5wyek as a file (OSF Storage version 1, 2026-10-01T23:12:30Z = 06:12 WIB), SHA-256 of the OSF file identical to the local file. **Not a separate OSF registration**; registration b2vc9 was not updated (user decision). Link in `registration/OSF_URL_addendum.txt` |
| Order | `supplement_s1.py --forecast` (started 06:17) and `susceptibility.py` (started 07:13) were run **after** the upload of the addendum (06:12) |
| Script hashes when run | `followup/susceptibility.py` `bc4c799d…3eb90`, `followup/supplement_s1.py` `082b81e0…ef8fd0`, `followup/fig_redraw.py` `715ae177…26608`, equal to the addendum; scripts not changed |
| Check of S1 | `check_global_matches_registered.abs_diff` = 0.0 (< 1e-6) in `results/followup/s1/forecast_holdout2026_L1.json` |
| `susceptibility.py` | finished 2026-10-02 07:17 without error; 179,860 land cells (peat cells, `peat_frac` ≥ 0.5: 36,163); 15 physical predictors, 64 embedding bands, 2 fire-history features; 7 predictor sets; GroupKFold(5) over 0.5° blocks; 1,000 bootstrap replicates |

Data:

- Province boundaries: geoBoundaries gbOpen IDN ADM1, `boundaryID` IDN-ADM1-65028918, `boundaryYearRepresented` 2017,
  build of 12 December 2023, source OpenStreetMap/Wambacher, licence ODbL 1.0; 34 provinces; SHA-256 `9a2c93ed…cbb3af`
  → `data/raw/boundaries/idn_adm1.geojson`. The provinces created in 2022 (Papua Selatan, Papua Tengah, Papua
  Pegunungan, Papua Barat Daya) are **not** included; their area belongs to the polygon "Papua". In the province table,
  cells in the `papua_selatan` box are labelled with the region of that box (column `region`) and the ADM1 province
  "Papua".
- Static susceptibility data (`data/followup/susc_cells.csv`), from `gee_fetch.py --kind susc` (0.5° tiles, 6 workers)
  and `peat_fraction.py --table-in/--table-out` with the same PEATMAP as the main run: 342,299 cells = 106,828
  candidate cells (all included) + 235,471 cells of a hash sample (`((cell × 2654435761) mod 2^32) mod 10 = 0`,
  `np.uint64`); 0 cells outside the rule. Indonesian land cells (`in_indonesia` ≥ 0.5 and `wc_water` < 0.5): 179,860
  (101,531 candidates, 78,329 sample). Earth Engine: 1,035 tiles × 4 parts = 4,140 parts, **0 failed**.

| Column | Source | Aggregation |
| --- | --- | --- |
| `elev`, `slope` | `USGS/SRTMGL1_003`, `ee.Terrain.slope` | mean per cell (`reduceResolution` mean, as for WorldCover) |
| `rain_annual` | `UCSB-CHG/CHIRPS/DAILY` 2012–2018 | bilinear resampling (as in the monthly table of the main run); total / 7 |
| `dry_days_aso` | CHIRPS daily, August–October 2012–2018 (644 days) | number of days < 1 mm / 7 |
| `swvl1_aso` | `ECMWF/ERA5_LAND/MONTHLY_AGGR`, August–October 2012–2018 (21 months) | bilinear resampling; mean |
| `t2m_annual` | ERA5-Land monthly 2012–2018 | bilinear resampling; mean |
| `wc_*`, `in_indonesia` | `ESA/WorldCover/v200`, `USDOS/LSIB_SIMPLE/2017` | as in the main run (`static_once_image`) |
| `emb_A00`–`emb_A63` | AlphaEarth 2017 and 2018 | as in the main run (systematic 10 × 10 sample per cell), mean of the two years |
| `peat_frac` | PEATMAP (`data/raw/peat/peatmap_indonesia.gpkg`) | `peat_fraction.py` (rasterised at 0.001°, block mean) |

- Checks (`logs/followup_susc_checks.txt`): all required columns present; missing values in land cells only for
  `swvl1_aso` and `t2m_annual` (0.08 %, coastal cells outside the ERA5-Land land grid); no land cell without
  embeddings. Candidate cells compared with `data/cube/static.csv` of the main run: mean embedding 2017–2018 differs
  by at most 5e-06 (rounding to 5 significant digits); `in_indonesia`, `wc_water`, `wc_tree` and `peat_frac` identical.
- Outputs (`results/followup/`): `figures/` (Figures 3, 5 and 6 redrawn), `tables/appeared_2026_by_province.{md,csv}`,
  `fig_redraw.json`; `s1/` (k-means, k = 4: labels, profile, cross-tabulation, maps, `s1.json`,
  `forecast_holdout2026_L1.json`); `susceptibility/` (`susceptibility.json`, `susceptibility.md`, figure).
- Warnings: none in the logs of `fig_redraw.py`, `supplement_s1.py`, `gee_fetch.py --kind susc`, `peat_fraction.py`
  and `susceptibility.py`. In `fig6_transitions.png` (map 2023→2026) the "same regime" and "changed regime" points are
  drawn first and are mostly hidden by the "appeared"/"disappeared" points; the counts in the legend and the table are
  not affected (`fig_redraw.py` not changed, its hash being fixed in the addendum).
- Deviations: none; the three follow-up scripts were not changed.
- Durations: smoke test about 1 min; ADM1 download and check < 1 min; `fig_redraw` about 2 s; susceptibility data from
  Earth Engine 1 h 22 min (05:49–07:11); peat fraction 11 s; `supplement_s1 --forecast` 3 min (06:17–06:20);
  `susceptibility` 3.5 min (07:13–07:17).

## 3. Follow-up 2 (17 November 2026, exploratory)

Design: `registration/ADDENDUM_followup2.md`. All analyses are exploratory and outside the registered analysis.

| Item | Status |
| --- | --- |
| Smoke test `followup2/smoke_followup2.py` | passed (2026-11-17 13:06:24–13:07:22) |
| Registered hashes | matched `registration/registration.json` (checked before the work, before `intersensor.py` and when the report was written) |
| Registered `run_all.py` and `results/` | not re-run; new outputs only in `results/followup2/` |
| Prerequisites | complete (`results.json`, `labels_selected.csv`, `selection.json`, `regimes_noaa20/labels_selected.csv`, `preds_holdout2026.csv.gz`, cubes `cell_month`, `cell_month_noaa20`, `static`, `oni`) |
| Addendum | SHA-256 of the scripts filled in `registration/ADDENDUM_followup2.md` on 2026-11-17; uploaded by the user to the OSF project https://osf.io/5wyek (https://osf.io/8vr7f; OSF Storage version 1, 2026-11-17T06:36:00Z = 13:36 WIB). SHA-256 of the OSF file `bef11c409a321d2a53e23f28da0402cb675817835d460b468e2e3c9342ee51f6`, identical to the local file |
| Upload note | the user's first upload (https://osf.io/5wyek/files/r9qa2, 2026-11-17T06:28:48Z) contained the task instructions (`FOLLOWUP2.md`), not the addendum; that file remains in the project. The correct addendum was uploaded afterwards |
| Order | `intersensor.py` started 2026-11-17 13:36:34, **after** the upload of the addendum (13:36 WIB); finished 13:39:10 |
| Script hashes when run | `followup2/intersensor.py` `0516e746f389b6356b51d13d5e969eda717a009731b66a045da426272cb08395` (recorded in `intersensor.json: meta`), `followup2/smoke_followup2.py` `99faf9d09f7d9529456a27a9d9fb38bb5add2b2d5e773c6ab16df17b9db495de`, equal to the addendum; scripts not changed |
| Check A (`checks.registered_sensor_ari`) | recomputed 0.29297691095343115 against registered 0.29297691095343115, `abs_diff` = 0.0 (< 1e-9); 5,319 cells (registered 5,319) |
| `check_noaa20_refit_matches_registered` | `cells_differ` = 0, `ari` = 1.0 |
| Exit with "CHECK FAILED" | no |

Hold-out data (part E):

- Hold-out target months (unique `target_month` in `results/forecast/preds_holdout2026.csv.gz`): 2026-08, 2026-09.
- Archive coverage according to the FIRMS data-availability service (queried 2026-11-17T13:08:22): `VIIRS_SNPP_SP`
  2012-01-20 to 2026-07-31; `VIIRS_SNPP_NRT` 2026-08-01 to 2026-11-02; `VIIRS_NOAA20_SP` 2018-04-01 to 2026-07-31;
  `VIIRS_NOAA20_NRT` 2026-08-01 to 2026-11-17.
- **S-NPP SP**: the `VIIRS_SNPP_SP` archive did not yet cover any hold-out month; this part was skipped
  (`intersensor.py` run without `--snpp-sp-2026`) and not replaced by another source.
- **NOAA-20**: `VIIRS_NOAA20_SP` did not cover the hold-out months either, so `VIIRS_NOAA20_NRT` was used for both
  months (one processing level, not mixed).
- Download: FIRMS area API with the code of the main run (`fetch_data.download` → `firms_download.fetch`), the three
  regions of `config.json: regions`, 2026-08-01 to 2026-09-30, downloaded 2026-11-17, to
  `data/raw/firms_followup2/VIIRS_NOAA20_NRT/<region>/`; 39 files, 287,850 raw detection rows (before filtering).
- Cube (`grid_features.py` unchanged): `data/cube/followup2/cell_month_noaa20_2026.csv.gz`, 82,205 cell-month rows
  (2026-08: 37,673; 2026-09: 44,532). Retrieval and cube building took 40.7 s.
- Arguments of `intersensor.py`: `--noaa20-2026 data/cube/followup2/cell_month_noaa20_2026.csv.gz --noaa20-months
  2026-08 2026-09 --noaa20-level NRT`.
- As stated in the addendum, the registered S-NPP NRT record ends on 29 September 2026, whereas the NOAA-20 target
  covers the whole month.

Domains and numbers of pairs:

- Overlap period 2018-04 to 2025-12 (93 months); cell-months with detections in that period: S-NPP 381,544,
  NOAA-20 395,270.
- Union domain: 6,905 cells before static attributes, 6,905 with static attributes, 6,571 within Indonesia.

| Level | Forecast domain | Union domain |
| --- | --- | --- |
| B. Occurrence: cells | 26,026 | 6,571 |
| B. Occurrence: cell-months (2 × 2 table) | 2,420,418 | 611,103 |
| B. Occurrence: cells, stratum `peat` | 15,685 | 4,419 |
| B. Occurrence: cells, stratum `non_peat` | 10,341 | 2,152 |
| C. Indicators: cell-month pairs (≥ 1 detection from both sensors) | 41,318 | 21,591 |

- D. Fingerprint features (2018-04 to 2025-12, both sensors): 4,448 common cells (`enso_rho`: 4,331 cells with a
  defined value).
- D. Regime classes, for both the registered partition (GMM, k = 11) and the alternative (k-means, k = 4): cells with
  fingerprints S-NPP 2018–2025 5,491, NOAA-20 2018–2025 5,528; common cells per comparison:
  `ref_vs_noaa20_2018_2025` 5,319, `ref_vs_snpp_2018_2025` 5,491, `snpp_vs_noaa20_2018_2025` 4,448,
  `snpp_model_applied_to_noaa20` 4,448, `noaa20_model_applied_to_snpp` 4,448.
- E. NOAA-20 hold-out: 52,052 cell-month pairs. Block bootstrap over 0.5° blocks: 1,000 replicates.
- Durations: smoke test 13:06:24–13:07:22; data retrieval 13:08:21–13:09:02; `intersensor.py` 13:36:34–13:39:10.
- Warnings: one scikit-learn `ConvergenceWarning` ("Best performing initialization did not converge") from
  `GaussianMixture`; no other errors. Part E for S-NPP SP was not available.
- Deviations: none; `followup2/intersensor.py` and `followup2/smoke_followup2.py` were not changed.
- Outputs: `results/followup2/intersensor.json`, `results/followup2/tables/*.md`, `results/followup2/figures/*.png`.

## 4. Follow-up 3 (20 November 2026, exploratory and post hoc)

Design: `registration/ADDENDUM_followup3.md`, written after the results of the registered run and of follow-ups 1 and
2 were known.

| Item | Status |
| --- | --- |
| Script | `followup3/disagreement.py`, SHA-256 `e233a1059b5d5d8208d1d635987ed082955f82fbb98fb8d6015cd61e9a72d4c7` (recorded in `results/followup3/disagreement.json: meta`) |
| Run | 2026-11-20T14:50:12Z to 14:50:13Z (Python 3.10.12, NumPy 2.2.6, pandas 2.3.3) |
| Inputs | the 2 × 2 occurrence tables of follow-up 2 rebuilt from the same cubes and domains; `results/followup2/intersensor.json` (SHA-256 `81b775d5…ac54683`, recorded in `meta`) |
| Replication check | the follow-up-2 intervals of kappa, positive agreement and proportion of agreement recomputed from the same seed (20261117) and order of draws: 198 interval bounds compared, maximum absolute difference 0.0 |
| Hold-out comparison | separate fixed seed (20261120); 52,052 cell-months, months 2026-08 and 2026-09 |
| Output | `results/followup3/disagreement.json`; the script changed nothing else under `results/` |
| Map helper | `results/followup3/cells_peat_regime.csv` (cell centre, registered regime and peat fraction of the 26,026 cells) was extracted separately for the maps of the paper; it is not an analysis output (see `results/README.md`) |
