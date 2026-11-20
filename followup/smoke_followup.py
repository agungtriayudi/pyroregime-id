"""Smoke test of the follow-up scripts on synthetic data (no internet). Synthetic results must NOT be reported.

    python followup/smoke_followup.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(*args):
    subprocess.run([PY, *map(str, args)], check=True, cwd=ROOT)


def synthetic_susc(d, tmp):
    """Cell table for the susceptibility test from synthetic static data + random unburned land cells."""
    st = pd.read_csv(d / "static.csv")
    emb = [c for c in st if c.startswith("emb_")]
    base = st[st["year"].isin([2017, 2018])].groupby("cell")[emb].mean() if "year" in st else st.groupby("cell")[emb].mean()
    land = st.groupby("cell")[[c for c in st if c.startswith("wc_") or c in ("peat_frac", "in_indonesia")]].mean()
    t = base.join(land).reset_index()
    rng = np.random.default_rng(3)
    extra = t.sample(len(t), replace=True, random_state=1).copy()
    extra["cell"] = extra["cell"] + rng.integers(200, 400, len(extra)) * 100000      # other cells, without detections
    extra[emb] = extra[emb].to_numpy() + rng.normal(0, 0.05, (len(extra), len(emb)))
    t = pd.concat([t, extra], ignore_index=True).drop_duplicates("cell")
    for c, (m, s) in {"elev": (50, 30), "slope": (2, 1), "rain_annual": (2500, 300), "dry_days_aso": (40, 10),
                      "swvl1_aso": (0.35, 0.05), "t2m_annual": (300, 1)}.items():
        t[c] = rng.normal(m, s, len(t))
    t["in_indonesia"] = t.get("in_indonesia", 1.0)
    p = tmp / "susc_cells.csv"; t.to_csv(p, index=False)
    return p


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pyro_followup_smoke_"))
    run("make_synthetic.py", "--out", tmp / "data", "--cells", "60")
    d = tmp / "data"
    cfg = json.load(open(ROOT / "config.json"))
    cfg.update({"firms_glob": str(d / "firms" / "*.csv"), "cell_month": str(tmp / "cell_month.csv.gz"),
                "gee_monthly": str(d / "gee_monthly.csv.gz"), "static": str(d / "static.csv"), "fwi": str(d / "fwi.csv"),
                "oni": str(d / "oni.ascii.txt"), "dmi": str(d / "dmi.csv"), "fingerprint": str(tmp / "fp.csv"),
                "panel_end": "2026-10", "k_range": "3-5", "cluster_boot": 5, "bootstrap": 30, "leads": [1],
                "out": str(tmp / "results"), "registration": str(tmp / "no_registration.json"),
                "provenance": str(tmp / "no_provenance.json"), "manuscript": "", "synthetic": True})
    cfg.pop("cell_month_noaa20", None)
    (tmp / "config.json").write_text(json.dumps(cfg, indent=1))
    run("run_all.py", "--config", tmp / "config.json")
    run("followup/fig_redraw.py", "--config", tmp / "config.json")
    run("followup/supplement_s1.py", "--config", tmp / "config.json", "--forecast", "--boot", "3")
    s1 = json.load(open(tmp / "results" / "followup" / "s1" / "s1.json"))
    diff = s1["forecast_holdout2026_L1"]["check_global_matches_registered"]["abs_diff"]
    assert diff < 1e-9, f"S1 global model differs from the registered result (difference {diff})"
    cells = synthetic_susc(d, tmp)
    run("followup/susceptibility.py", "--config", tmp / "config.json", "--cells", cells, "--boot", "30")
    for f in ["followup/figures/fig3_regime_map.png", "followup/figures/fig5_seasonality.png",
              "followup/figures/fig6_transitions.png", "followup/tables/appeared_2026_by_province.md",
              "followup/s1/profile.md", "followup/susceptibility/susceptibility.json"]:
        assert (tmp / "results" / f).exists(), f
    print(f"\nSMOKE FOLLOWUP PASSED ({tmp}; synthetic results, not research results)")


if __name__ == "__main__":
    main()
