# Reproducing the published analysis from public data

This guide repeats the analysis of the paper from the public archives. It needs credentials for three services,
about 30 GB of free disk and, on a laptop with 10 cores and 16 GB RAM, about two days of mostly unattended computing
(downloads about 13 h wall clock in the published run, analysis 9 h). To check the numbers of the paper without
downloading anything, use the stored results instead (`tools/paper_tables.py`, `tools/paper_figures.py`).

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-lock.txt for the versions of the published run
python smoke_test.py                     # must end with "SMOKE TEST PASSED"
```

## 2. Credentials (never commit them)

| Service | Setup |
| --- | --- |
| NASA FIRMS | request a free map key at https://firms.modaps.eosdis.nasa.gov/api/map_key/ and `export FIRMS_MAP_KEY=...` |
| Google Earth Engine | register a Cloud project for Earth Engine, run `earthengine authenticate`, and set `gee_project` in `config.json` |
| Copernicus Early Warning Data Store | create an account at https://ewds.climate.copernicus.eu, accept the licence of `cems-fire-historical-v1`, and write the key to `~/.cdsapirc` |

## 3. Inputs that are not downloaded by the scripts

**Peat map.** The published run used PEATMAP (Xu et al., 2018; CC BY 4.0; https://doi.org/10.5518/252), whose
Indonesian part comes from the national peat map of Ritung et al. (2011):

1. Download `Asia.zip` from https://archive.researchdata.leeds.ac.uk/251/5/Asia.zip (SHA-256 of the file used:
   `8767e9ba8250770420cc555250f80b79b67410d00ae13ac629f4fe2ae9b8724e`).
2. Take `Asia/SEA_Peatland.shp` (ESRI:54034), reproject to EPSG:4326, repair the geometries, clip to the bounding
   boxes in `config.json: regions` and explode multipart polygons (1,186 polygons: Sumatra 446, Kalimantan 633,
   South Papua 107), then save as `data/raw/peat/peatmap_indonesia.gpkg`. The file used had SHA-256
   `948b4f310047bb6f8a1f073809ef6109a4f446571ea6a232851ec301f3431ed3`; a file written again will have another hash,
   because a GeoPackage stores write metadata. A sketch of the same steps:

```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
g = gpd.read_file("Asia/SEA_Peatland.shp").to_crs(4326)
g["geometry"] = g.geometry.make_valid()
boxes = [box(95.0, -6.2, 106.3, 6.0), box(108.5, -4.3, 119.3, 4.5), box(137.5, -9.2, 141.1, -5.5)]
out = gpd.GeoDataFrame(pd.concat([gpd.clip(g, b) for b in boxes]), crs=4326).explode(index_parts=False)
out.to_file("data/raw/peat/peatmap_indonesia.gpkg", driver="GPKG")
```

**Province boundaries** (follow-up 1 and the maps; optional): geoBoundaries gbOpen IDN ADM1, boundary year 2017,
https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM1/geoBoundaries-IDN-ADM1.geojson
(SHA-256 `9a2c93ed5c956d7b9afeab4882e9aab9c3fceb8be4cafaae6e198223c2cbb3af`), saved as
`data/raw/boundaries/idn_adm1.geojson`.

## 4. Build the datacube

```bash
python fetch_data.py --config config.json                  # all stages, resumable
python fetch_data.py --config config.json --stage verify   # must report "all inputs complete"
```

Keep `panel_end` = `2026-09` to repeat the period of the paper. The data stage asks the FIRMS availability service
which archive covers which dates and downloads standard processing (SP) where it exists and near-real-time (NRT)
data afterwards. When the published cube was built, S-NPP SP ended on 30 June 2026, so the detections of 1 July to
29 September 2026 were NRT; FIRMS later replaces NRT by SP data, so a new download of these months will differ
slightly from the published cube (the near-real-time files also lack the `type` field that removes static thermal
sources). The coverage of every download is recorded in `data/provenance.json`; the provenance file of the published
run is included as `docs/data_provenance.json`. The NOAA-20 cube for the comparison of sensors (`cell_month_noaa20`)
is built from `VIIRS_NOAA20_SP`, 2018–2025.

## 5. Register, then run the registered analysis

The published registration is in `registration/`. `run_all.py` checks it before running. Because the comments of the
analysis modules were translated after the run, install the byte-identical originals first (see
`provenance/README.md`):

```bash
python provenance/verify_registered.py --install
python run_all.py --config config.json                     # writes results/; about 9 h on 10 cores
git checkout -- .                                          # back to the English scripts
```

For a new study, use your own configuration and registration path instead and run `python register.py` before any
evaluation data are examined.

## 6. Exploratory follow-ups (in the order in which they were fixed and run)

```bash
# follow-up 1 (registration/ADDENDUM_followup.md)
python followup/fig_redraw.py --config config.json --adm1 data/raw/boundaries/idn_adm1.geojson
python gee_fetch.py --config config.json --kind susc
python peat_fraction.py --config config.json --table-in data/followup/susc_gee.csv --table-out data/followup/susc_cells.csv
python followup/supplement_s1.py --config config.json --forecast --adm1 data/raw/boundaries/idn_adm1.geojson
python followup/susceptibility.py --config config.json

# follow-up 2 (registration/ADDENDUM_followup2.md); NOAA-20 detections for the hold-out months
# (NRT in the published run; use VIIRS_NOAA20_SP once the standard-processing archive covers August-September 2026)
python firms_download.py --source VIIRS_NOAA20_NRT --start 2026-08-01 --end 2026-09-30 --out data/raw/firms_followup2
mkdir -p data/cube/followup2
python grid_features.py --firms "data/raw/firms_followup2/VIIRS_NOAA20_NRT/**/*.csv" \
    --out data/cube/followup2/cell_month_noaa20_2026.csv.gz
python followup2/intersensor.py --config config.json \
    --noaa20-2026 data/cube/followup2/cell_month_noaa20_2026.csv.gz --noaa20-months 2026-08 2026-09 --noaa20-level NRT

# follow-up 3 (registration/ADDENDUM_followup3.md)
python followup3/disagreement.py --config config.json
```

## 7. Compare with the published results

```bash
python tools/paper_tables.py --results results --out paper_tables
git diff --stat results/                                     # files that changed relative to the published run
```

Machine-learning steps (gradient boosting, Gaussian mixtures, autoencoder) use fixed seeds, but results can differ
slightly across library versions and processors; `requirements-lock.txt` lists the versions of the published run.
