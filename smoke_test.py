"""Smoke test without internet: checks that the code and the Python environment work before real data are downloaded.

    python smoke_test.py            # ~5-15 minutes on an ordinary CPU
Checks: (1) alignment of the GEE/peat-map grid with the FIRMS grid; (2) the full run_all.py pipeline on synthetic
data with planted regimes; (3) all manuscript tokens filled; (4) planted regimes recovered (ARI >= 0.9).
Synthetic results must NOT be reported.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from gee_fetch import cell_ids
from grid_features import cell_of
from peat_fraction import _block_mean

PY = sys.executable


def check_grids():
    # a GEE pixel (rows from the north, columns from -180) must fall in the same cell as a point at the pixel centre
    r, c, ny, nx = 9400, 29800, 7, 9
    ids = cell_ids(r, c, ny, nx)
    lat = 90 - (np.arange(r, r + ny) + 0.5) * 0.01
    lon = -180 + (np.arange(c, c + nx) + 0.5) * 0.01
    ref = cell_of(np.repeat(lat, nx), np.tile(lon, ny)).reshape(ny, nx)
    assert (ids == ref).all(), "GEE cell_ids do not match the FIRMS grid"
    # peat map: 0.001-degree blocks -> 0.01-degree cells
    arr = np.zeros((1000, 1000), np.uint8); arr[:10, :10] = 1          # upper-left corner cell fully peat
    w, n = 101.0, 1.0
    top_left = cell_of(np.array([n - 0.005]), np.array([w + 0.005]))[0]
    df = _block_mean(arr, w, n, np.array([top_left]))
    assert len(df) == 1 and df["peat_frac"].iloc[0] == 1.0, "peat-map alignment is wrong"
    print("grid: OK")


def main():
    check_grids()
    tmp = Path(tempfile.mkdtemp(prefix="pyro_smoke_"))
    subprocess.run([PY, "make_synthetic.py", "--out", tmp / "data", "--cells", "60"], check=True)
    d = tmp / "data"
    cfg = json.load(open("config.json"))
    cfg.update({"firms_glob": str(d / "firms" / "*.csv"), "cell_month": str(tmp / "cell_month.csv.gz"),
                "gee_monthly": str(d / "gee_monthly.csv.gz"), "static": str(d / "static.csv"), "fwi": str(d / "fwi.csv"),
                "oni": str(d / "oni.ascii.txt"), "dmi": str(d / "dmi.csv"), "fingerprint": str(tmp / "fp.csv"),
                "panel_end": "2026-10", "k_range": "3-5", "cluster_boot": 5, "bootstrap": 30, "leads": [1],
                "out": str(tmp / "results"), "registration": str(tmp / "no_registration.json"),
                "provenance": str(tmp / "no_provenance.json"), "manuscript": "", "synthetic": True})
    cfg.pop("cell_month_noaa20", None)
    (tmp / "config.json").write_text(json.dumps(cfg, indent=1))
    subprocess.run([PY, "run_all.py", "--config", tmp / "config.json"], check=True)
    res = json.load(open(tmp / "results" / "results.json"))
    assert res["meta"]["synthetic"] is True
    truth = pd.read_csv(d / "truth.csv")
    lab = pd.read_csv(tmp / "results" / "regimes" / "labels_selected.csv")
    key = "cell" if "cell" in truth else None
    if key is None:
        truth["cell"] = cell_of(truth["lat0"].to_numpy() + 0.005, truth["lon0"].to_numpy() + 0.005)
    j = lab.merge(truth[["cell", "true_regime"]], on="cell")
    ari = adjusted_rand_score(j["true_regime"], j["regime"])
    print(f"\nARI against the planted regimes = {ari:.3f} (n = {len(j)})")
    assert ari >= 0.9, "planted regimes not recovered"
    print(f"\nSMOKE TEST PASSED. Temporary output in {tmp} (not research results).")


if __name__ == "__main__":
    main()
