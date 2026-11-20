"""Uji asap followup2/intersensor.py pada data sintetis (tanpa internet). Hasil sintetis TIDAK boleh dilaporkan.

    python followup2/smoke_followup2.py

Membuat "NOAA-20" sintetis dengan menipiskan dan mengganggu deteksi S-NPP sintetis, menjalankan run_all.py
(agar keluaran terdaftar tiruan tersedia), lalu intersensor.py dengan kedua target alternatif hold-out.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "followup2"))


def run(*args):
    subprocess.run([PY, *map(str, args)], check=True, cwd=ROOT)


def unit_checks():
    from intersensor import agree_stats, ccc_from_sums, paired_stats
    s = agree_stats(20, 5, 10, 15)                                     # contoh buku teks: po 0,70; pe 0,50; kappa 0,40
    assert abs(float(s["po"]) - 0.70) < 1e-12 and abs(float(s["kappa"]) - 0.40) < 1e-12, s
    assert abs(float(s["ppos"]) - 40 / 55) < 1e-12
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000); blk = rng.integers(0, 50, 5000)
    r = paired_stats(x, x, blk, 50, rng)
    assert abs(r["ccc"] - 1) < 1e-9 and abs(r["bias"]) < 1e-12
    r = paired_stats(x, x + 1.0, blk, 50, rng)                         # pergeseran 1 SD: CCC = 2v/(2v+1)
    v = x.var()
    assert abs(r["ccc"] - 2 * v / (2 * v + 1)) < 1e-9 and abs(r["bias"] - 1) < 1e-9, r
    S = np.array([[4.0, 10.0, 14.0, 30.0, 54.0, 40.0]])
    assert np.isfinite(ccc_from_sums(S)[0]).all()
    print("cek statistik: OK")


def thin(src, dst, rng, keep=0.8, start="2018-04-01", end="2025-12-31"):
    """Sensor kedua sintetis: buang 20 % deteksi, geser waktu, ganggu FRP dan tanda siang/malam."""
    f = pd.read_csv(src)
    f = f[(f["acq_date"] >= start) & (f["acq_date"] <= end)]
    f = f[rng.random(len(f)) < keep].copy()
    f["acq_time"] = (f["acq_time"] + 50) % 2400
    f["frp"] = f["frp"] * rng.lognormal(0, 0.3, len(f))
    flip = rng.random(len(f)) < 0.1
    f.loc[flip, "daynight"] = np.where(f.loc[flip, "daynight"] == "N", "D", "N")
    dst.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(dst, index=False)


def main():
    unit_checks()
    tmp = Path(tempfile.mkdtemp(prefix="pyro_followup2_smoke_"))
    run("make_synthetic.py", "--out", tmp / "data", "--cells", "140")
    d = tmp / "data"
    rng = np.random.default_rng(11)
    src = d / "firms" / "firms_synthetic.csv"
    thin(src, d / "firms_n20" / "n20.csv", rng)
    run("grid_features.py", "--firms", d / "firms_n20" / "*.csv", "--out", tmp / "cell_month_noaa20.csv.gz")
    thin(src, d / "firms_n20_2026" / "n20.csv", rng, start="2026-08-01", end="2026-10-31")
    run("grid_features.py", "--firms", d / "firms_n20_2026" / "*.csv", "--out", tmp / "cm_noaa20_2026.csv.gz")
    thin(src, d / "firms_sp_2026" / "sp.csv", rng, keep=0.95, start="2026-08-01", end="2026-10-31")
    run("grid_features.py", "--firms", d / "firms_sp_2026" / "*.csv", "--out", tmp / "cm_snpp_sp_2026.csv.gz")

    cfg = json.load(open(ROOT / "config.json"))
    cfg.update({"firms_glob": str(d / "firms" / "*.csv"), "cell_month": str(tmp / "cell_month.csv.gz"),
                "cell_month_noaa20": str(tmp / "cell_month_noaa20.csv.gz"),
                "gee_monthly": str(d / "gee_monthly.csv.gz"), "static": str(d / "static.csv"), "fwi": str(d / "fwi.csv"),
                "oni": str(d / "oni.ascii.txt"), "dmi": str(d / "dmi.csv"), "fingerprint": str(tmp / "fp.csv"),
                "panel_end": "2026-10", "k_range": "3-5", "cluster_boot": 5, "bootstrap": 30, "leads": [1],
                "out": str(tmp / "results"), "registration": str(tmp / "no_registration.json"),
                "provenance": str(tmp / "no_provenance.json"), "manuscript": "", "synthetic": True})
    (tmp / "config.json").write_text(json.dumps(cfg, indent=1))
    run("run_all.py", "--config", tmp / "config.json")
    run("followup2/intersensor.py", "--config", tmp / "config.json", "--boot", "30",
        "--snpp-sp-2026", tmp / "cm_snpp_sp_2026.csv.gz", "--snpp-sp-months", "2026-08", "2026-09", "2026-10",
        "--noaa20-2026", tmp / "cm_noaa20_2026.csv.gz", "--noaa20-months", "2026-08", "2026-09", "2026-10",
        "--noaa20-level", "SP")
    out = tmp / "results" / "followup2"
    R = json.load(open(out / "intersensor.json"))
    assert R["meta"]["synthetic"] is True and R["meta"]["exploratory"] is True
    assert R["checks"]["registered_sensor_ari"]["abs_diff"] < 1e-9
    o = R["occurrence"]["forecast_domain"]["all"]
    assert o["a_both"] + o["b_ref_only"] + o["c_other_only"] + o["d_neither"] == o["n"]
    assert 0 < o["ratio"] < 1, "sensor sintetis kedua membuang deteksi, jadi rasio harus < 1"
    assert R["regime"]["selected"]["check_noaa20_refit_matches_registered"]["ari"] > 0.999
    for k in ("ref_vs_snpp_2018_2025", "snpp_vs_noaa20_2018_2025", "snpp_model_applied_to_noaa20"):
        assert "ari" in R["regime"]["selected"][k], k
    assert "ccc" in R["features"]["night_frac"] and "ccc" in R["indicators"]["forecast_domain"]["n_det"]
    for nm in ("snpp_sp", "noaa20"):
        h = R["holdout2026"][nm]
        assert "delta_moe_global_alt_target" in h and h["agreement"]["n"] == h["n"], nm
    for f in ["tables/occurrence.md", "tables/indicators.md", "tables/features.md", "tables/regime_decomposition.md",
              "tables/holdout_alt_targets.md", "figures/fig_f2_occurrence_by_year.png",
              "figures/fig_f2_indicator_differences.png", "figures/fig_f2_regime_decomposition.png"]:
        assert (out / f).exists(), f
    print(f"\nSMOKE FOLLOWUP2 LULUS ({tmp}; hasil sintetis, bukan hasil penelitian)")


if __name__ == "__main__":
    main()
