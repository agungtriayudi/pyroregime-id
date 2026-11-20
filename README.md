# PyroRegime-ID

An open, pre-registered Python workflow for (i) building a monthly 0.01° datacube of satellite fire detections and
their predictors from public archives, (ii) testing whether one predictor block (e.g. Earth-observation foundation-model
embeddings) can replace another (e.g. monthly hydroclimate) with the learner and the validation design held fixed,
(iii) decomposing the reproducibility of unsupervised fire-regime classes into the effects of resampling, algorithm,
record length and sensor, and (iv) measuring the agreement of two satellite sensors along the processing chain with
quantity and allocation disagreement. The workflow is demonstrated on the peatlands of Sumatra, Kalimantan and South
Papua, Indonesia (2012–2026).

This repository accompanies the manuscript

> Triayudi, A., Wedha, B.Y., Nathasia, N.D. PyroRegime-ID: an open, pre-registered workflow for testing satellite
> embeddings against hydroclimate and decomposing the reproducibility of fire-regime classes in Indonesian peatlands.
> Submitted to *Computers & Geosciences*.

Authors: Agung Triayudi, Bayu Yasa Wedha, Novi Dian Nathasia (Faculty of Communication and Information Technology,
Universitas Nasional, Jakarta, Indonesia). Contact: agungtriayudi@civitas.unas.ac.id. Licence: MIT (see `LICENSE`).

## Contents

| Path | Content |
| --- | --- |
| `fetch_data.py`, `firms_download.py`, `gee_fetch.py`, `climate_fetch.py`, `peat_fraction.py` | Data stage: download and grid all inputs (resumable); `gee_export.py` and `raster_to_table.py` are a fallback through Google Drive |
| `grid_features.py`, `regime_features.py`, `cluster_regimes.py`, `regime_shift.py`, `forecast.py`, `report.py` | The six registered analysis modules |
| `run_all.py` | Runs the registered analysis from one configuration; refuses to run if the registered hashes differ |
| `register.py` | Freezes the design: SHA-256 of the analysis parameters and of the six analysis modules |
| `fill_manuscript.py`, `manuscript_tokens.txt` | Write every number from `results.json` into the manuscript, so that no value is transcribed by hand |
| `followup/`, `followup2/`, `followup3/` | Exploratory follow-ups, each fixed in a time-stamped addendum (`registration/`) |
| `make_synthetic.py`, `smoke_test.py`, `check_synthetic.py`, `followup*/smoke_*.py` | Synthetic data with planted regimes and smoke tests (no internet needed) |
| `tests/` | Unit tests of the agreement statistics and of the registration hashes |
| `tools/` | `paper_tables.py` and `paper_figures.py` rebuild the tables and figures of the paper from `results/` |
| `config.json` | Configuration of the published study (analysis parameters as registered) |
| `registration/` | Registration document, hashes and the three addenda (OSF links in `OSF_URL*.txt`) |
| `results/` | Result files of the registered run and of the follow-ups (see `results/README.md`) |
| `provenance/` | Byte-identical originals of the scripts as they were run, and a script that verifies them |
| `docs/` | User guide, tutorial, reproduction guide, run report and the data provenance of the published run |
| `CHANGELOG.md`, `DEVIATIONS.md` | Changes of data sources and code; departures from the registration |

## Installation

Python ≥ 3.10. The analysis, the tests and the tools need NumPy, pandas, SciPy, scikit-learn and matplotlib; data
retrieval also needs earthengine-api, cdsapi, geopandas, rasterio, xarray and netCDF4.

```bash
git clone https://github.com/agungtriayudi/pyroregime-id.git
cd pyroregime-id
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements-lock.txt` lists the package versions of the environment of the registered run (Python 3.12.14, macOS,
Apple Silicon). The code has also been tested with Python 3.13.

## Quick start without internet (synthetic data)

```bash
python tests/test_metrics.py               # unit tests (seconds)
python smoke_test.py                       # full pipeline on synthetic data with planted regimes
python followup/smoke_followup.py          # follow-up 1 (static susceptibility, alternative partition)
python followup2/smoke_followup2.py        # follow-up 2 (agreement of two sensors)
```

Each smoke test takes a few minutes and ends with a line containing `PASSED`. Synthetic outputs are software tests and
are never reported as results. A step-by-step walk-through is in `docs/TUTORIAL.md`.

## Rebuilding the tables and figures of the paper

```bash
python tools/paper_tables.py --results results --out paper_tables
python tools/paper_figures.py --results results --out paper_figures \
    --boundaries geoBoundaries-IDN-ADM1.geojson      # optional, adds province outlines to the maps
```

The tables are written as Markdown and match Tables 2–5, S12 and S13 of the paper; the figures match Figures 1, 2, 4
and 5 and Figures S2, S5 and S8. Figure 3 is drawn by `followup2/intersensor.py`
(`results/followup2/figures/fig_f2_regime_decomposition.png`; the journal version is the same figure at a higher
resolution). The province outlines come from geoBoundaries
(gbOpen, IDN ADM1, boundary year 2017):
https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM1/geoBoundaries-IDN-ADM1.geojson

## Full reproduction from public data

`docs/REPRODUCE.md` lists the credentials (a NASA FIRMS map key, a Google Earth Engine project, a Copernicus Early
Warning Data Store account), the peat map, the disk space (≥ 30 GB recommended) and every command, in order. In short:

```bash
python fetch_data.py --config config.json      # build the datacube (resumable; hours)
python register.py --config config.json        # hash the design (a new study); upload registration/REGISTRATION.md
python run_all.py --config config.json         # registered analysis -> results/
python followup2/intersensor.py --config config.json
python followup3/disagreement.py --config config.json
```

The registered run took 8 h 59 min for `run_all.py` on a laptop with 10 cores and 16 GB RAM (`docs/RUN_REPORT.md`).

## Applying the workflow to another region

A study is defined by `config.json` alone: `regions` (bounding boxes), `ref_years`, `min_det`, `k_range`, `leads`,
`bootstrap`, the data-source identifiers and the path of a local peat or soil map. The agreement statistics
(`followup2/intersensor.py`, `followup3/disagreement.py`) are plain functions on arrays and can be reused for other
maps and partitions. See `docs/USER_GUIDE.md`.

## Pre-registration and verification

The design was registered on the Open Science Framework before the 2026 evaluation was run
(https://osf.io/b2vc9; hashes in `registration/registration.json`). The follow-ups were fixed in addenda
(https://osf.io/5wyek, https://osf.io/8vr7f and `registration/ADDENDUM_followup3.md`).

The comments, docstrings and messages of the code were translated from Indonesian into English after the analyses
were run. The translation changed no code: `provenance/verify_registered.py` checks that the originals in
`provenance/original_code/` reproduce every recorded SHA-256 hash and that every English script has the same abstract
syntax tree as its original once string constants are masked. All changed strings are listed in
`provenance/string_changes.txt`.

```bash
python provenance/verify_registered.py
```

## Citation

See `CITATION.cff`. Please cite the paper once it is published.
