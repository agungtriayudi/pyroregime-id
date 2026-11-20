"""Suplemen S1 (EKSPLORATIF): solusi k-means k = 4 yang akan dipilih oleh ringkasan teks registrasi.

    python followup/supplement_s1.py --config config.json [--forecast]

Memakai fitur dan praproses yang sama persis dengan cluster_regimes.py (Prep, fit_one, seed 0).
Keluaran di results/followup/s1/: labels_kmeans4.csv, profile.md, s1.json, fig_s1_map.png, fig_s1_seasonality.png
--forecast: hold-out 2026, L = 1, model global vs MoE dengan 4 rezim (bootstrap blok seperti analisis utama).
           Model global tidak memakai rezim, jadi AUC-PR-nya harus sama dengan hasil terdaftar (cek otomatis).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "followup"))

from cluster_regimes import Prep, fit_one, scores, stability  # noqa: E402
from fig_redraw import fig3, fig5, load_adm1  # noqa: E402
from report import peak_month  # noqa: E402

K = 4


def profile_md(fp, lab):
    d = fp.merge(lab[["cell", "regime"]], on="cell")
    g = d.groupby("regime")
    t = pd.DataFrame({
        "Regime": g.size().index, "Cells": g.size().to_numpy(),
        "Detections / yr": np.expm1(g["log_det_per_year"].mean()).to_numpy(),
        "Recurrence": g["recurrence"].mean().to_numpy(), "Persistence (d)": g["persist_mean"].mean().to_numpy(),
        "Night fraction": g["night_frac"].mean().to_numpy(), "Median FRP (MW)": np.expm1(g["log_frp_med"].mean()).to_numpy(),
        "Peak month": [peak_month(s, c) for s, c in zip(g["season_sin"].mean(), g["season_cos"].mean())],
        "Season concentration": g["season_conc"].mean().to_numpy(), "ENSO ρ": g["enso_rho"].mean().to_numpy(),
        "Peat fraction": g["peat_frac"].mean().to_numpy()})
    lines = ["| " + " | ".join(t.columns) + " |", "| " + " | ".join(["---"] * len(t.columns)) + " |"]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(f"{int(v)}" if c in ("Regime", "Cells", "Peak month") else f"{v:.2f}"
                                       for c, v in r.items()) + " |")
    return "\n".join(lines) + "\n"


def forecast_s1(res, labels_path, out):
    import forecast as F
    cfg = json.load(open(res / "work" / "forecast_config.json"))
    cfg = dict(cfg, labels=str(labels_path))
    P0 = F.build_panel(cfg)
    regimes = sorted(P0["regime"].unique())
    for r in regimes:
        P0[f"reg_{r}"] = (P0["regime"] == r).astype(float)
    cols = F.feat_cols(P0, F.FEATURESETS["full"]) + ["clim"]
    P = F.add_targets(P0, 1)
    models = ["global", "global_regime", "moe"]
    pred = F.run_design(P, "holdout2026", cfg, models, cols, regimes)
    s = F.summarize(pred, models, cfg.get("bootstrap", 1000))
    reg = json.load(open(res / "results.json"))["forecast"]["holdout2026"]
    s["check_global_matches_registered"] = {
        "s1_global_auc_pr": s["global"]["auc_pr"], "registered_global_auc_pr": reg["global"]["L1"]["auc_pr"],
        "abs_diff": abs(s["global"]["auc_pr"] - reg["global"]["L1"]["auc_pr"])}
    json.dump(s, open(out / "forecast_holdout2026_L1.json", "w"), indent=1, default=float)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forecast", action="store_true")
    ap.add_argument("--adm1", default=None)
    ap.add_argument("--boot", type=int, default=None, help="default: cluster_boot di config (registrasi: 50)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    res = Path(cfg["out"]); out = res / "followup" / "s1"; out.mkdir(parents=True, exist_ok=True)
    fp = pd.read_csv(cfg["fingerprint"])
    prep = Prep(emb_weight=1.0).fit(fp)
    X = prep.transform(fp)
    lab, _ = fit_one("kmeans", X, K, seed=0)
    rng = np.random.default_rng(0)
    sc = scores(X, lab, rng)
    B = a.boot if a.boot is not None else cfg.get("cluster_boot", 50)
    st, sd = stability("kmeans", X, K, lab, B, rng)
    labels = fp[["cell", "lat", "lon"]].assign(regime=lab)
    labels.to_csv(out / "labels_kmeans4.csv", index=False)
    sel = pd.read_csv(res / "regimes" / "labels_selected.csv")
    j = labels.merge(sel[["cell", "regime"]], on="cell", suffixes=("", "_sel"))
    ari = adjusted_rand_score(j["regime"], j["regime_sel"])
    xt = pd.crosstab(j["regime"], j["regime_sel"])
    xt.to_csv(out / "crosstab_kmeans4_vs_selected.csv")
    (out / "profile.md").write_text(profile_md(fp, labels))
    adm = load_adm1(a.adm1)
    fig3(labels, adm, out / "fig_s1_map.png")
    fig5(fp, labels, out / "fig_s1_seasonality.png")
    s1 = {"k": K, "method": "kmeans", **sc, "stab_ari": st, "stab_sd": sd, "ari_vs_selected": float(ari),
          "n_cells": int(len(labels)), "exploratory": True}
    if a.forecast:
        s1["forecast_holdout2026_L1"] = forecast_s1(res, out / "labels_kmeans4.csv", out)
    json.dump(s1, open(out / "s1.json", "w"), indent=1, default=float)
    print(json.dumps(s1, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
