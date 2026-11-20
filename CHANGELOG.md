# Change log

Changes to the data-retrieval code, dataset URLs and identifiers, `panel_end` and the environment, and changes made for
the public release. Entries up to 17 November 2026 are translated from the Indonesian original. The Google Earth Engine
project identifier is replaced by `<gee_project>`. Analysis parameters and the six registered analysis modules were
never changed (see `DEVIATIONS.md`).

## Public release (Computers & Geosciences submission)

- 2026-11-21: repository prepared for publication under the MIT licence.
  - Comments, docstrings and messages of all scripts translated from Indonesian into English. No code changed: every
    English script has the same abstract syntax tree as the script that was run once string constants are masked
    (`provenance/verify_registered.py`); byte-identical originals are kept in `provenance/original_code/`, and every
    changed string constant outside docstrings is listed in `provenance/string_changes.txt`. Messages that referred to
    the project notes now refer to `docs/REPRODUCE.md`.
  - Added: English `README.md`, `docs/USER_GUIDE.md`, `docs/TUTORIAL.md`, `docs/REPRODUCE.md`, `docs/RUN_REPORT.md`
    (translation of the run reports), `LICENSE`, `CITATION.cff`, `requirements-lock.txt`, `tests/test_metrics.py`,
    `tools/paper_tables.py` and `tools/paper_figures.py` (rebuild the tables and figures of the paper from `results/`).
  - `config.json`: the Earth Engine project identifier replaced by a placeholder; explanatory `_comment` fields
    translated. Neither is an analysis parameter, and the registered parameter hash is unchanged.
- 2026-11-20 (follow-up 3, exploratory and post hoc): `followup3/disagreement.py` re-expresses the 2 × 2 occurrence
  tables of follow-up 2 as total, quantity and allocation disagreement (`registration/ADDENDUM_followup3.md`). Output:
  `results/followup3/disagreement.json`. Nothing else under `results/` changed.

## 2026-11-17: follow-up 2, step 2 (hold-out data, exploratory)

- New source: FIRMS `VIIRS_NOAA20_NRT`, 2026-08-01 to 2026-09-30 (three regions) → `data/raw/firms_followup2/`,
  cube `data/cube/followup2/cell_month_noaa20_2026.csv.gz` (`grid_features.py` unchanged). Details in
  `data/provenance.json`, key `followup2`.
- `VIIRS_SNPP_SP` (archive ending 2026-07-31) and `VIIRS_NOAA20_SP` (ending 2026-07-31) did not yet cover the hold-out
  months; S-NPP standard processing was skipped and NOAA-20 used near-real-time data. The download used a local helper
  script that calls `fetch_data.download` → `firms_download.fetch`; `docs/REPRODUCE.md` gives the equivalent commands.

## 2026-09-30 to 2026-10-02: data stage and follow-up 1

- 2026-09-30: autonomous package prepared (`gee_fetch.py` via `computePixels`, `climate_fetch.py`, `peat_fraction.py`,
  `fetch_data.py`, `register.py`, `smoke_test.py`).
- 2026-09-30: `dmi_urls` changed to the monthly NOAA CPC IOD index (ERSSTv6, 1991–2020 climatology; 1950-01 to
  2026-08). The PSL HadISST series ended in 2026-05 and the second PSL URL returned 404. Parser `parse_cpc_dmi()` added
  to `climate_fetch.py`. Recorded in `data/provenance.json: changes`. No results had been seen.
- 2026-09-30: environment: Python 3.12.14 through uv (`.venv/`), dependencies from `requirements.txt`.
- 2026-09-30: `fetch_data.py` / `firms_download.py`: parallel FIRMS downloads (`firms_workers` = 4, below the quota of
  5,000 transactions per 10 minutes), atomic file writes (`.part` → `.csv`), retries on network and 5xx errors, the API
  key masked in error messages. The SP/NRT overlap test is skipped when the NRT archive does not cover a full month of
  the last SP month (FIRMS at the time: SP until 2026-06-30, NRT from 2026-07-01), so that a spurious ratio of 0 is not
  recorded.
- 2026-09-30: FIRMS counts about one transaction per day of data (a 5-day request ≈ 5 transactions) and answers
  HTTP 400 "Invalid MAP_KEY" when the 10-minute quota is used up (the key remains valid). A rate limiter
  (`firms_tx_per_10min` = 4000) was added to `fetch_data.download` and a `mapkey_status` check to
  `firms_download.fetch`, so that this answer is waited out (at most 15 × 2 minutes) instead of being treated as a
  rejected credential.
- 2026-09-30: `config.json: gee_project` set to the project used (`<gee_project>`); `ee.Initialize` and the 8 collection
  IDs in `gee_ids` passed a test.
- 2026-09-30: `gee_fetch.py` (tested directly on one 20 × 20-pixel tile):
  - monthly layers of the last year are requested only up to the last predictor month (`panel_end` − 1 = 2026-08),
    because monthly ERA5-Land was available only until 2026-08; parts of a partial year carry the suffix `mMM`, so that
    they are not reused if `panel_end` changes;
  - the `reduceResolution` of MCD64A1 moved to the branch with data (the substitute image for months without MCD64A1
    has no projection, which raised an error);
  - masked pixels returned by `computePixels` (−inf) are recorded as NaN in `to_rows`.
  Availability in Earth Engine when checked: MCD64A1 until 2026-07, MOD/MYD14A1 until 2026-09-21, CHIRPS until
  2026-08-31, monthly ERA5-Land until 2026-08, AlphaEarth until 2025.
- 2026-09-30: the FWI dataset `cems-fire-historical-v1` moved from the Climate Data Store to the Copernicus Early
  Warning Data Store (EWDS). `config.json: fwi_cds.url` = https://ewds.climate.copernicus.eu/api; `climate_fetch.py`
  uses this URL with the key from `~/.cdsapirc`. The EWDS request schema matches `fwi_cds.request`. Recorded in
  `data/provenance.json: changes`.
- 2026-09-30: `~/.cdsapirc` (EWDS URL) created; a one-day FWI test download through EWDS succeeded (variable in the
  file: `fwinx`).
- 2026-09-30: **peat map (user decision, option 3 "global map")**: `config.json: peat_map` =
  `data/raw/peat/peatmap_indonesia.gpkg`, from PEATMAP (Xu et al., 2018; DOI 10.5518/252; CC BY 4.0), `Asia.zip` →
  `SEA_Peatland.shp`, clipped to the bounding boxes in `regions` (1,186 polygons). According to PEATMAP, its Indonesian
  part comes from Ritung et al. (2011), BBSDLP, 1:250,000. The 2019 BBSDLP map (Anda et al., 2021) could not be
  obtained legally with a licence. Table 1 of the manuscript was adjusted accordingly.
- 2026-09-30: `fetch_data.stage_grid` creates the folder `data/cube/` before calling `grid_features.py` (saving failed
  before).
- 2026-10-01: static Earth Engine layers through `computePixels` failed for the AlphaEarth embeddings ("User memory
  limit exceeded", 21 tiles × 9 years), and the process hung without log output from 01:25 (stopped at 03:55). Tests: a
  two-step mean (10 m → 0.001° → 0.01°) was identical to the direct mean, but 0.5° tiles timed out; evaluating only at
  the centres of candidate cells (`reduceRegions`) was identical but still exceeded the interactive memory limit; export
  to project assets failed (the asset root `projects/<gee_project>/assets` was not available). `gee_fetch.py`: option
  `--parts once|emb|all` and `ee.data.setDeadline` (`gee_deadline_ms`, default 10 minutes).
- 2026-10-01: batch export to Google Drive (a table per candidate cell) also failed with "User memory limit exceeded"
  (also with `tileScale` 8); interactive requests of 250 cells succeeded, but in parallel reached only about 0.9 cells
  per second (several days). **User decision (before registration):** the AlphaEarth embedding of a cell = the mean of a
  systematic sample of 10 × 10 points (0.001° grid, nearest neighbour) instead of all 10 m pixels. Validation against
  the full mean on 800 cells (Riau and Central Kalimantan, 2023): per-band correlation median 0.999 (minimum 0.996),
  mean error 2.7 % of the between-cell SD, correlation of the 8 principal components 0.997–1.000
  (`logs/emb_sample_validation.txt`). A 0.5° tile then finished in about 5 seconds. The methods text of the manuscript
  was adjusted. The Drive test script was deleted; 40 old full-mean embedding parts were kept in
  `data/gee_parts/static_emb_computepixels/` and not used.
- 2026-10-01: `peat_fraction.py` drops static rows without any embedding (162 rows = 18 cells × 9 years, all on the
  open sea with `in_indonesia` = 0, probably offshore gas flares), because the PCA in `forecast.py` does not accept NaN.
  These cells were already removed by the LSIB mask in the fingerprints and are not in the forecast domain. The
  verification passed again.
- 2026-10-02 (follow-up 1, outside the registered analysis): `gee_fetch.py --kind susc` (SRTM, CHIRPS and ERA5-Land
  2012–2018, WorldCover/LSIB and embeddings 2017–2018 with the method of the main run; cells = candidates + a 10 % hash
  sample), `peat_fraction.py --table-in/--table-out`, `config.json` keys `gee_ids.srtm` and `followup_*` (not analysis
  keys; the registration hash still matched). Province boundaries from geoBoundaries gbOpen IDN ADM1 (2017,
  34 provinces) in `data/raw/boundaries/idn_adm1.geojson`.
