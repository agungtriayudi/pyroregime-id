# Tutorial: the whole workflow on synthetic data

This tutorial needs no internet connection, no credentials and no satellite data. It takes about 10 minutes on an
ordinary laptop (each smoke test runs for one to a few minutes). Every output written here comes from synthetic data
and is a software test, not a research result.

## 1. Install

```bash
git clone https://github.com/agungtriayudi/pyroregime-id.git
cd pyroregime-id
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib
```

These five packages are enough for the tutorial; `requirements.txt` adds the packages needed to download real data.

## 2. Unit tests of the agreement statistics

```bash
python tests/test_metrics.py
```

The tests check the disagreement components D, Q and A against values computed by hand, kappa and positive agreement on
a textbook table, Lin's concordance correlation computed from block sums, the block bootstrap, and that `config.json`
and the registered modules reproduce the two hashes of the registration. The run ends with `ALL 6 TESTS PASSED`.

## 3. The full pipeline on synthetic data

```bash
python smoke_test.py
```

What happens:

1. `check_grids()` verifies that an Earth Engine pixel and a FIRMS point at the same location fall in the same
   0.01° cell, and that a 0.001° peat raster is averaged into the right cell.
2. `make_synthetic.py` writes, in a temporary folder, inputs in the format of the real archives for 60 synthetic
   cells per region: FIRMS detections from three planted fire regimes (smouldering peat, land clearing, savanna) with
   a drought response and a planted regime shift in 2026, monthly predictors, static attributes with 64-dimensional
   embeddings, FWI, ONI and DMI, and the true regime of every cell (`truth.csv`).
3. A copy of `config.json` is pointed at these files, with fewer bootstrap replicates and a smaller range of k so that
   the test is quick, and `run_all.py` runs the complete analysis: aggregation, fingerprints, clustering and selection
   of k, regime shift, forecasting under the three validation designs, the predictor-block substitution test and
   the reporting.
4. The test fails unless the selected partition recovers the planted regimes with an adjusted Rand index of at least
   0.9. The last lines look like:

```
ARI against the planted regimes = 1.000 (n = 173)

SMOKE TEST PASSED. Temporary output in /tmp/pyro_smoke_xxxxxxxx (not research results).
```

Open the temporary folder to see the structure of a real run:

```
results/
  results.json          every number of the run (registration status, clustering, shift, forecast, feature sets)
  fill_values.json      the values substituted for the [[tokens]] of a manuscript
  tables/*.md           selection of k, regime profiles, shift, forecast skill, feature sets, importance
  figures/*.png         study area, selection of k, regime map, fingerprints, seasonality, transitions, skill, ...
  regimes/              labels of every algorithm, profile and selection grid
  shift/                event-year assignments and transitions
  forecast/             skill by model, lead and design; hold-out predictions; importance
```

For example, the result of the substitution test:

```bash
python -c "import json; r = json.load(open('/tmp/pyro_smoke_xxxxxxxx/results/results.json')); print(r['featureset']['peat'])"
```

## 4. The follow-up diagnostics on synthetic data

```bash
python followup/smoke_followup.py      # static susceptibility test and the alternative partition
python followup2/smoke_followup2.py    # agreement of two sensors along the processing chain
```

`smoke_followup2.py` creates a synthetic second sensor by thinning and perturbing the synthetic detections, runs the
analysis to obtain mock registered outputs, and then runs `followup2/intersensor.py`: occurrence agreement, cell-month
indicators, fingerprint features, and the regime classes refitted and with the model held fixed. Each test ends with a
line containing `PASSED`.

## 5. Rebuilding the tables and figures of the paper

The repository contains the result files of the published run, so the tables and figures of the paper can be rebuilt
in seconds:

```bash
python tools/paper_tables.py --results results --out paper_tables
python tools/paper_figures.py --results results --out paper_figures
```

`paper_tables/table3_decomposition.md`, for instance, is Table 3 of the paper (agreement of regime classes when one
factor changes at a time), and `paper_figures/fig04_occurrence_disagreement.png` is Figure 4. Province outlines are
added to the maps when a boundary file is given with `--boundaries` (see `README.md`).

## 6. Using the diagnostics on your own maps or partitions

The agreement functions take plain arrays:

```python
import sys
import numpy as np
sys.path[:0] = [".", "followup2", "followup3"]
from disagreement import disagreement
from intersensor import ari_ci

# two binary maps -> 2 x 2 counts -> total, quantity and allocation disagreement
m1 = np.random.default_rng(0).random(10_000) < 0.2
m2 = m1 ^ (np.random.default_rng(1).random(10_000) < 0.05)
a, b, c, d = (m1 & m2).sum(), (m1 & ~m2).sum(), (~m1 & m2).sum(), (~m1 & ~m2).sum()
print(disagreement(a, b, c, d))

# two partitions of the same units, with a block bootstrap over spatial blocks
labels1 = np.random.default_rng(2).integers(0, 4, 2_000)
labels2 = np.where(np.random.default_rng(3).random(2_000) < 0.8, labels1, 0)
blocks = np.arange(2_000) // 50
print(ari_ci(labels1, labels2, blocks, B=200, rng=np.random.default_rng(4)))
```

## 7. Next step

`docs/REPRODUCE.md` explains how to download the real data and repeat the published analysis; `docs/USER_GUIDE.md`
explains the configuration and how to set up a study of another region.
