# Results of the published run

Outputs of the registered run (1 October 2026) and of the three exploratory follow-ups, as written by the scripts.
They are the source of every number, table and figure of the paper; nothing here was edited by hand. Field names are
in English; a few descriptive strings inside the JSON files are in Indonesian, because they were written by the
scripts before their messages were translated (see `provenance/README.md`). Not included are `results/work/`
(intermediate fingerprints of the event years and of NOAA-20, and a copy of the configuration that names the private
Earth Engine project) and the upload bundles (`*_bundle.zip`), which duplicate files listed here.

| Path | Written by | Content |
| --- | --- | --- |
| `results.json` | `report.py` (via `run_all.py`) | every number of the registered run: data, clustering, shift, forecast, feature sets; `meta` holds the registration status, the two hashes and the data provenance |
| `fill_values.json` | `fill_manuscript.py` | the values of the manuscript tokens listed in `manuscript_tokens.txt` |
| `embedding.json`, `sensor.json` | `run_all.py` | agreement (ARI) of the selected partition with the partition fitted without embeddings, and with the partition fitted to NOAA-20 fingerprints (the registered comparison of sensors) |
| `tables/` | `report.py` | selection of k, regime profiles, shift, forecast skill, feature sets, importance |
| `figures/` | `report.py` | figures of the registered run (study area, selection of k, regime map, fingerprints, seasonality, transitions, skill, reliability, importance) |
| `regimes/` | `cluster_regimes.py` | labels of each algorithm, selected labels (GMM, k = 11), profile, selection grid, `selection.json` (rule applied) |
| `regimes_noaa20/`, `regimes_noemb/` | `cluster_regimes.py` | partition fitted to NOAA-20 fingerprints (2018–2025); partition without embeddings |
| `shift/`, `shift_neutral/` | `regime_shift.py` | assignments of event-year fingerprints to the reference regimes and transitions (El Niño years; ENSO-neutral baseline) |
| `forecast/` | `forecast.py` | skill by model, lead and design with bootstrap intervals (`results_forecast.json`), predictor importance, predictions for the 2026 hold-out |
| `followup/` | follow-up 1 | redrawn figures and province table (`fig_redraw.py`), k-means alternative (`s1/`), static susceptibility test (`susceptibility/`) |
| `followup2/` | follow-up 2 | agreement of S-NPP and NOAA-20 along the processing chain (`intersensor.json`, tables, figures) |
| `followup3/` | follow-up 3 | disagreement components (`disagreement.json`) |
| `followup3/cells_peat_regime.csv` | extracted for the maps | cell centre, registered regime (`regimes/labels_selected.csv`) and peat fraction (static table of the datacube) of the 26,026 cells; a convenience file for Figures 2 and S2, not an analysis output |

## Where the tables and figures of the paper come from

| Paper | Source | Rebuilt by |
| --- | --- | --- |
| Table 2 | `results.json: featureset` | `tools/paper_tables.py` |
| Table 3 | `results.json: clustering`, `followup/s1/s1.json`, `followup2/intersensor.json: regime`, `regimes/selection.json` | `tools/paper_tables.py` |
| Table 4 | `followup3/disagreement.json: occurrence`, `followup2/intersensor.json: occurrence` | `tools/paper_tables.py` |
| Table 5 | `forecast/results_forecast.json: holdout2026`, `followup2/intersensor.json: holdout2026` | `tools/paper_tables.py` |
| Tables S12, S13 | `followup2/intersensor.json`, `followup3/disagreement.json` | `tools/paper_tables.py` |
| Figure 1 | schematic of the workflow | `tools/paper_figures.py` |
| Figure 2, Figure S2 | `followup3/cells_peat_regime.csv` | `tools/paper_figures.py` |
| Figure 3 | `followup2/figures/fig_f2_regime_decomposition.png` (journal version: same figure at a higher resolution) | `followup2/intersensor.py` |
| Figure 4 | `followup3/disagreement.json` | `tools/paper_figures.py` |
| Figure 5 | `forecast/results_forecast.json` | `tools/paper_figures.py` |
| Figure S5 | `followup/s1/labels_kmeans4.csv` | `tools/paper_figures.py` |
| Figure S8 | `shift/assigned_2023.csv`, `shift/assigned_2026.csv` | `tools/paper_figures.py` |
| Other supplementary figures | `figures/`, `followup/figures/`, `followup/s1/`, `followup/susceptibility/`, `followup2/figures/` | `report.py` and the follow-up scripts |
