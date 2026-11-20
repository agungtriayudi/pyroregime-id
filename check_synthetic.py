"""Compare clustering labels with the planted regimes (synthetic data only).

    python check_synthetic.py --truth data/synthetic/truth.csv --labels results/regimes/labels_kmeans.csv
"""
import argparse

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from grid_features import cell_of

ap = argparse.ArgumentParser()
ap.add_argument("--truth", required=True)
ap.add_argument("--labels", nargs="+", required=True)
a = ap.parse_args()
t = pd.read_csv(a.truth)
t["cell"] = cell_of(t["lat0"].to_numpy() + 0.005, t["lon0"].to_numpy() + 0.005)
t = t.drop_duplicates("cell")
for f in a.labels:
    j = pd.read_csv(f).merge(t[["cell", "true_regime"]], on="cell")
    print(f"{f}: ARI vs truth = {adjusted_rand_score(j['true_regime'], j['regime']):.3f} (n={len(j)})")
    print(pd.crosstab(j["true_regime"], j["regime"]).to_string())
