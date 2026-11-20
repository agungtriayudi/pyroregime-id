# User guide

PyroRegime-ID is a set of command-line Python scripts. A study is defined by one JSON configuration file; every
script reads it (or the paths it names) and writes its outputs to files, so stages can be run, resumed and checked one
at a time. This guide describes the stages, the configuration, the inputs and outputs of each script, and how the
diagnostics can be reused. `docs/TUTORIAL.md` runs the whole chain on synthetic data; `docs/REPRODUCE.md` lists the
exact commands of the published study.

## 1. Stages

| Stage | Command | Reads | Writes |
| --- | --- | --- | --- |
| Data | `python fetch_data.py --config config.json [--stage S]` | public archives (FIRMS, Earth Engine, Copernicus EWDS, NOAA CPC), local peat map | `data/raw/`, `data/cube/`, `data/provenance.json` |
| Registration | `python register.py --config config.json` | `config.json`, the six analysis modules | `registration/registration.json`, `registration/REGISTRATION.md` |
| Analysis | `python run_all.py --config config.json [--allow-deviation]` | datacube, registration | `results/` (regimes, shift, forecast, tables, figures, `results.json`) |
| Reporting | called by `run_all.py` (`report.py`, `fill_manuscript.py`) | `results/` | `results/results.json`, `results/tables/*.md`, `results/figures/*.png` |
| Follow-ups | `python followup*/<script>.py --config config.json` | datacube, registered outputs | `results/followup*/` |

`fetch_data.py` runs its stages in the order `firms → grid → candidates → indices → gee_monthly → gee_static → peat →
fwi → verify`. Every stage writes resumable parts; rerunning a stage continues where it stopped. `--stage verify`
only checks that every input exists and covers the configured period.

`run_all.py` runs: cell-month aggregation (`grid_features.py`) → fingerprints (`regime_features.py`) → clustering and
selection of k (`cluster_regimes.py`, plus the ablation without embeddings) → fingerprints of the event years → regime
shift (`regime_shift.py`) → forecasting and evaluation (`forecast.py`) → `report.py`. Before it starts it recomputes
the parameter hash and the analysis-code hash and stops if either differs from `registration/registration.json`;
`--allow-deviation` overrides this and records `registration_status = "deviation"` in `results.json`.

## 2. Configuration file

Keys hashed at registration (`register.py: ANALYSIS_KEYS`) define the analysis; changing any of them after
registration is a deviation. All other keys describe data sources and paths and may be corrected (record the change in
`data/provenance.json` and `CHANGELOG.md`).

Analysis keys:

| Key | Meaning | Published study |
| --- | --- | --- |
| `regions` | bounding boxes `[west, south, east, north]` in degrees, one per region | Sumatra, Kalimantan, South Papua (`papua_selatan`) |
| `panel_start` | first month of the datacube | `2012-01` |
| `ref_years` | reference period for fingerprints and clustering | `[2012, 2025]` |
| `min_det` | minimum retained detections per cell in the reference period | `20` |
| `k_range` | candidate numbers of regimes | `"3-12"` |
| `methods` | clustering algorithms (`kmeans`, `gmm`, `ae_kmeans`, `hdbscan`) | all four |
| `cluster_boot` | subsamples (80 %) for resampling stability | `50` |
| `shift_years`, `neutral_years` | El Niño years and ENSO-neutral baseline years for the shift analysis | `[2015, 2019, 2023, 2026]`, `[2017, 2018, 2021, 2022]` |
| `event_months`, `event_min_det` | calendar months and minimum detections of event-year fingerprints | `"1-10"`, `3` |
| `enso_years` | years left out in leave-one-El-Niño-out | `[2015, 2019, 2023]` |
| `holdout_train_end` | last training target month of the temporal hold-out | `"2026-07"` |
| `leads` | forecast lead times (months) | `[1, 2, 3]` |
| `bootstrap` | block-bootstrap replicates (0.5° blocks) | `1000` |
| `neg_frac` | fraction of cell-months without fire kept for training (probabilities are corrected for the prior shift when < 1) | `1.0` |

Data-source and path keys (selection):

| Key | Meaning |
| --- | --- |
| `panel_end` | last month of detections; predictors are used up to `panel_end` − 1 |
| `firms_start`, `firms_day_range`, `firms_workers`, `firms_tx_per_10min` | FIRMS download period, request length (days), parallel workers and rate limit |
| `firms_dir`, `firms_glob` | where FIRMS files are written and which files form the cube (S-NPP SP + NRT) |
| `gee_project`, `gee_workers`, `tile_deg_monthly`, `tile_deg_static`, `gee_deadline_ms` | Earth Engine project, parallel requests, tile sizes and request deadline |
| `gee_ids` | Earth Engine collection IDs (MODIS, CHIRPS, ERA5-Land, WorldCover, LSIB, AlphaEarth, SRTM) |
| `embedding_years` | years of AlphaEarth embeddings to fetch |
| `peat_map` | local peat (or organic-soil) map, vector (GeoPackage/Shapefile) or raster |
| `oni_urls`, `dmi_urls`, `fwi_cds` | climate-index URLs and the FWI request to the Copernicus EWDS |
| `cell_month`, `cell_month_noaa20`, `gee_monthly`, `static`, `fwi`, `oni`, `dmi`, `fingerprint`, ... | paths of the datacube tables |
| `out`, `registration`, `provenance`, `manuscript` | output folder, registration file, provenance file, optional manuscript to fill |
| `followup_*` | inputs of follow-up 1 (static susceptibility) |

## 3. Datacube

All tables use the same 0.01° grid. A cell is identified by the integer `cell` = iy × 100,000 + ix, where
iy = ⌊(lat + 90)/0.01⌋ and ix = ⌊(lon + 180)/0.01⌋ (`grid_features.cell_of`; `grid_features.cell_center` returns
the cell centre). Earth Engine pixels are requested on the same grid, so pixel edges coincide with cell edges
(checked by `smoke_test.py`).

| Table | Rows | Main columns |
| --- | --- | --- |
| `cell_month.csv.gz` | cell-months with ≥ 1 retained detection | `cell`, `month`, `n_det`, `frp_sum`, `frp_med`, `frp_max`, `night_frac`, `active_days`, `persist_max`, `lat`, `lon` |
| `gee_monthly.csv.gz` | candidate cell × month | burned fraction, MODIS fire, precipitation, dry days, soil moisture, 2 m temperature and dew point (vapour-pressure deficit is derived in `forecast.py`) |
| `static.csv` | candidate cell × year | WorldCover fractions, `in_indonesia`, `peat_frac`, `emb_A00`–`emb_A63` |
| `fwi_monthly.csv` | forecast cell × month | monthly mean FWI |
| `fp_2012_2025.csv` | cells with ≥ `min_det` detections | ten fire features, peat fraction, embedding components |

## 4. Diagnostics and how to reuse them

| Diagnostic | Implementation | Reusable functions |
| --- | --- | --- |
| Predictor-block substitution test | `forecast.py` (feature sets F, F + H, F + P + E, full; same learner and folds) | `forecast.py` feature-set loop; deltas in `results.json: featureset` |
| Block bootstrap (0.5° blocks) | `forecast.py`, `followup2/intersensor.py`, `followup3/disagreement.py` | `boot_sums(M, B, rng)` (sums of resampled blocks), `q(v)` (percentile interval) |
| Factor-wise decomposition of regime agreement | `followup2/intersensor.py` (part D) | `fit_partition`, `compare` (refitted), `transfer` (model held fixed), `ari_ci` |
| Occurrence agreement and disagreement components | `followup2/intersensor.py` (part B), `followup3/disagreement.py` | `agree_stats(a, b, c, d)`, `disagreement(a, b, c, d)` → D, Q, A, A/D, D_fire |
| Agreement of paired continuous quantities | `followup2/intersensor.py` (parts C and D) | `paired_stats(x, y, block, B, rng)` (CCC, bias, limits of agreement), `ccc_from_sums` |
| Reading rule | `followup2/intersensor.py` | `band(lo, hi)`: reproducible if the lower bound ≥ 0.80, not reproducible if the upper bound < 0.60 |

The functions take NumPy arrays and have no dependence on the fire data, so they can be imported for other
categorical maps or partitions. Example: disagreement components of two binary maps.

```python
import sys
sys.path[:0] = [".", "followup3"]
from disagreement import disagreement
r = disagreement(a=41318, b=18413, c=17219, d=2343468)   # 2 x 2 counts
print(r["D"], r["Q"], r["A"], r["A_share"])
```

## 5. Credentials

| Service | How it is read | Where to get it |
| --- | --- | --- |
| NASA FIRMS | environment variable `FIRMS_MAP_KEY` | https://firms.modaps.eosdis.nasa.gov/api/map_key/ |
| Google Earth Engine | `earthengine authenticate`; project in `config.json: gee_project` | https://earthengine.google.com |
| Copernicus EWDS (FWI) | `~/.cdsapirc` (key); URL in `config.json: fwi_cds.url` | https://ewds.climate.copernicus.eu |

Credentials are never written to project files; `.gitignore` excludes the usual credential files.

## 6. Applying the workflow to another region

1. Copy `config.json`; set `regions`, `panel_start`, `panel_end`, `ref_years`, `shift_years`, `neutral_years`,
   `enso_years` and `holdout_train_end` for the new study; set `peat_map` to a local peat or soil map; check the FIRMS
   sources available for the period.
2. Run `python smoke_test.py` to check the environment.
3. Run `python fetch_data.py --config my_config.json`, then `--stage verify`.
4. Register before looking at any evaluation data: `python register.py --config my_config.json` and upload
   `registration/REGISTRATION.md` to a registry such as OSF. Use a new `registration` path for a new study.
5. Run `python run_all.py --config my_config.json`. For the comparison of sensors, set `cell_month_noaa20` (cube built
   from a second sensor with `grid_features.py`) and run `followup2/intersensor.py` and `followup3/disagreement.py`.

The grid, the FIRMS and Earth Engine sources and the climate indices are global; only the peat map is regional.

## 7. Troubleshooting

| Message | Cause and remedy |
| --- | --- |
| FIRMS answers HTTP 400 "Invalid MAP_KEY" during a long download | the 10-minute transaction quota is used up; the code checks `mapkey_status` and waits (lower `firms_tx_per_10min` if it recurs) |
| Earth Engine "User memory limit exceeded" | use smaller tiles (`tile_deg_static`), fewer workers, or fetch the parts separately with `gee_fetch.py --parts once` and `--parts emb` |
| FWI request rejected by the Climate Data Store | the dataset is served by the Early Warning Data Store; keep `fwi_cds.url` = https://ewds.climate.copernicus.eu/api |
| `run_all.py`: "Analysis parameters/code differ from the registration" | an analysis key or module changed; revert it, or use `--allow-deviation` and record the reason in `DEVIATIONS.md`; to repeat the published run see `provenance/README.md` |
| scikit-learn `ConvergenceWarning` from the autoencoder or the Gaussian mixture | expected (recorded in `docs/RUN_REPORT.md`); the settings are fixed by the registration |
