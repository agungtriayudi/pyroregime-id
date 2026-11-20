"""Kumpulkan semua keluaran menjadi results.json (token naskah), tabel Markdown, dan Gambar 1-9.

    python report.py --config config.json
Keluaran: <out>/results.json, <out>/tables/*.md, <out>/figures/fig*.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

MODELS = ["clim", "fwi", "global", "global_regime", "moe"]
MODEL_NAME = {"clim": "Climatology", "fwi": "FWI", "global": "Global", "global_regime": "Global + regime", "moe": "Mixture of experts"}
DESIGN_NAME = {"spatial_cv": "Spatial block CV", "loeno": "Leave-one-El-Niño-out", "holdout2026": "2026 hold-out"}


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                     else ("—" if pd.isna(v) else str(v)) for v in r) + " |")
    return "\n".join(out)


def peak_month(s, c):
    return int(np.round((np.degrees(np.arctan2(s, c)) % 360) / 30)) % 12 + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    out = Path(cfg["out"]); tab, fig = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)
    R = {}

    # ---- data
    cube = pd.read_csv(cfg["cell_month"]); fp = pd.read_csv(cfg["fingerprint"])
    y0, y1 = cfg["ref_years"]
    yr = pd.PeriodIndex(cube["month"], freq="M").year
    ref = cube[(yr >= y0) & (yr <= y1) & cube["cell"].isin(fp["cell"])]
    R["data"] = {"n_detections": int(ref["n_det"].sum()), "n_cells": int(len(fp)),
                 "n_cell_months": int((cube["n_det"] > 0).sum()),
                 "period": f"{cube['month'].min()[:4]}–{cube['month'].max()[:4]}"}

    # ---- clustering
    sel = json.load(open(out / "regimes" / "selection.json"))
    R["clustering"] = {k: sel[k] for k in ("k", "method", "rule", "silhouette", "stab_ari", "stab_sd", "agreement_ari")}
    R["clustering"].update(json.load(open(out / "sensor.json")))
    R["embedding"] = json.load(open(out / "embedding.json"))
    grid = pd.read_csv(out / "regimes" / "selection_grid.csv")
    g = grid.copy()
    g["stability"] = g.apply(lambda r: "—" if pd.isna(r.get("stab_ari")) else f"{r['stab_ari']:.3f} ± {r['stab_sd']:.3f}", axis=1)
    g["selected"] = np.where((g["k"] == sel["k"]) & (g["method"] == sel["method"]), "✓", "")
    keep = [c for c in ["method", "k", "silhouette", "calinski_harabasz", "davies_bouldin", "bic", "stability", "selected"] if c in g]
    (tab / "selection.md").write_text(md_table(g[keep]))

    prof = pd.read_csv(out / "regimes" / "profile_selected.csv")
    t3 = pd.DataFrame({
        "Regime": prof["regime"], "Cells": prof["n_cells"],
        "Detections / yr": np.expm1(prof["log_det_per_year"]), "Recurrence": prof["recurrence"],
        "Persistence (d)": prof["persist_mean"], "Night fraction": prof["night_frac"],
        "Median FRP (MW)": np.expm1(prof["log_frp_med"]),
        "Peak month": [peak_month(s, c) for s, c in zip(prof["season_sin"], prof["season_cos"])],
        "Season concentration": prof["season_conc"],
        "ENSO ρ": prof.get("enso_rho", np.nan), "Peat fraction": prof.get("peat_frac", np.nan)})
    for c in ("wc_tree", "wc_grass", "wc_shrub", "wc_crop"):
        if c in prof:
            t3[c.replace("wc_", "WC ")] = prof[c]
    (tab / "regimes.md").write_text(md_table(t3))

    # ---- shift
    sh = json.load(open(out / "shift" / "shift.json"))
    base = json.load(open(out / "shift_neutral" / "shift.json"))
    pairs = {k: v for k, v in sh.items() if isinstance(v, dict)}
    R["shift"] = {"assign_acc": sh["assign_acc"], **{k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in pairs.items()}}
    bvals = [v["frac_changed"] for v in base.values() if isinstance(v, dict)]
    R["shift"]["baseline_mean"] = float(np.mean(bvals)) if bvals else None
    rows = [{"Transition": f"{v['from']} → {v['to']}", "Cells in both": v["n_both"],
             "Changed (%)": 100 * v["frac_changed"], "95 % CI (%)": f"{100 * v['ci_low']:.1f}–{100 * v['ci_high']:.1f}",
             "Appeared": v["appeared"], "Disappeared": v["disappeared"]} for v in pairs.values()]
    rows += [{"Transition": f"{v['from']} → {v['to']} (neutral baseline)", "Cells in both": v["n_both"],
              "Changed (%)": 100 * v["frac_changed"], "95 % CI (%)": f"{100 * v['ci_low']:.1f}–{100 * v['ci_high']:.1f}",
              "Appeared": v["appeared"], "Disappeared": v["disappeared"]} for v in base.values() if isinstance(v, dict)]
    (tab / "shift.md").write_text(md_table(pd.DataFrame(rows), "{:.1f}"))

    # ---- forecast
    rf = json.load(open(out / "forecast" / "results_forecast.json"))
    R["featureset"] = rf.pop("featureset")
    R["forecast"] = rf
    fs = R["featureset"]
    t4 = pd.DataFrame([{"Predictor set": n, "AUC-PR (all)": fs["all"].get(n, {}).get("auc_pr"),
                        "AUC-PR (peat ≥ 0.5)": fs["peat"].get(n, {}).get("auc_pr")} for n in ("fire_history", "hydro", "emb", "full")])
    t4.loc[len(t4)] = ["Δ full − emb", fs["all"].get("delta_full_emb"), fs["peat"].get("delta_full_emb")]
    (tab / "featuresets.md").write_text(md_table(t4))
    leads = sorted({L for d in rf.values() for m in d.values() for L in m})
    rows = []
    for m in MODELS:
        r = {"Model": MODEL_NAME[m]}
        for d in DESIGN_NAME:
            for L in leads:
                v = rf.get(d, {}).get(m, {}).get(L, {})
                r[f"{DESIGN_NAME[d]} {L}"] = v.get("auc_pr")
        v = rf.get("holdout2026", {}).get(m, {}).get("L1", {})
        r["BSS 2026 L1"] = v.get("bss")
        rows.append(r)
    (tab / "forecast.md").write_text(md_table(pd.DataFrame(rows)))
    imp = pd.read_csv(out / "forecast" / "importance.csv", index_col=0)
    (tab / "importance.md").write_text(md_table(imp.reset_index().rename(columns={"regime": "Regime"})))
    json.dump(R, open(out / "results.json", "w"), indent=1, default=float)

    # ---- figures
    lab = pd.read_csv(out / "regimes" / "labels_selected.csv").merge(fp, on=["cell", "lat", "lon"])
    k = sel["k"]; cmap = plt.get_cmap("tab10")

    f, ax = plt.subplots(figsize=(10, 3.6))
    sc = ax.scatter(lab["lon"], lab["lat"], c=lab.get("peat_frac", 0), s=3, cmap="YlOrBr", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Peat fraction"); ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Recurrently burning cells"); ax.set_aspect("equal"); f.tight_layout(); f.savefig(fig / "fig1_study_area.png", dpi=300); plt.close(f)

    f, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for m, d in grid[grid["method"] != "hdbscan"].groupby("method"):
        axs[0].plot(d["k"], d["silhouette"], "o-", label=m)
        if "stab_ari" in d:
            axs[1].errorbar(d["k"], d["stab_ari"], yerr=d["stab_sd"], fmt="o-", label=m, capsize=2)
    gm = grid[grid["method"] == "gmm"]
    axs[2].plot(gm["k"], gm["bic"], "o-", color="k")
    for a, t in zip(axs, ["Silhouette", "Bootstrap stability (ARI)", "GMM BIC"]):
        a.axvline(k, ls="--", color="grey"); a.set_xlabel("k"); a.set_title(t)
    axs[1].axhline(0.8, ls=":", color="red"); axs[0].legend(fontsize=8)
    f.tight_layout(); f.savefig(fig / "fig2_kselection.png", dpi=300); plt.close(f)

    f, ax = plt.subplots(figsize=(10, 3.6))
    for r in range(k):
        d = lab[lab["regime"] == r]
        ax.scatter(d["lon"], d["lat"], s=3, color=cmap(r), label=f"Regime {r}")
    ax.legend(markerscale=4, fontsize=8, ncol=min(k, 6)); ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°)"); f.tight_layout()
    f.savefig(fig / "fig3_regime_map.png", dpi=300); plt.close(f)

    feats = [c for c in ["log_det_per_year", "recurrence", "persist_mean", "night_frac", "log_frp_med",
                         "season_conc", "enso_rho", "peat_frac"] if c in prof]
    Z = (prof[feats] - lab[feats].mean()) / lab[feats].std()
    f, ax = plt.subplots(figsize=(8, 0.5 * k + 1.5))
    im = ax.imshow(Z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=40, ha="right"); ax.set_yticks(range(k), [f"Regime {r}" for r in range(k)])
    plt.colorbar(im, ax=ax, label="z-score"); f.tight_layout(); f.savefig(fig / "fig4_fingerprints.png", dpi=300); plt.close(f)

    f, ax = plt.subplots(figsize=(7, 3.5))
    sc_cols = [f"season_{m:02d}" for m in range(1, 13)]
    for r in range(k):
        ax.plot(range(1, 13), prof.loc[prof["regime"] == r, sc_cols].to_numpy().ravel(), "o-", color=cmap(r), label=f"Regime {r}")
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylabel("Share of detections"); ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(fig / "fig5_seasonality.png", dpi=300); plt.close(f)

    tf = sorted((out / "shift").glob("transition_*.csv"))
    f, axs = plt.subplots(1, max(1, len(tf)), figsize=(3.6 * max(1, len(tf)), 3.2), squeeze=False)
    for a, p in zip(axs[0], tf):
        T = pd.read_csv(p, index_col=0)
        a.imshow(T.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        a.set_xticks(range(T.shape[1]), T.columns); a.set_yticks(range(T.shape[0]), T.index)
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                a.text(j, i, f"{T.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        a.set_title(p.stem.replace("transition_", "").replace("_", " → ")); a.set_xlabel("to"); a.set_ylabel("from")
    f.tight_layout(); f.savefig(fig / "fig6_transitions.png", dpi=300); plt.close(f)

    f, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for a, d in zip(axs, DESIGN_NAME):
        for i, m in enumerate(MODELS):
            v = [rf.get(d, {}).get(m, {}).get(L, {}) for L in leads]
            y = [x.get("auc_pr", np.nan) for x in v]
            lo = [x.get("auc_pr", np.nan) - x.get("auc_pr_ci_low", np.nan) for x in v]
            hi = [x.get("auc_pr_ci_high", np.nan) - x.get("auc_pr", np.nan) for x in v]
            a.bar(np.arange(len(leads)) + (i - 2) * 0.16, y, 0.16, yerr=[lo, hi], capsize=2, label=MODEL_NAME[m])
        a.set_xticks(range(len(leads)), leads); a.set_title(DESIGN_NAME[d]); a.set_xlabel("Lead")
    axs[0].set_ylabel("AUC-PR"); axs[0].legend(fontsize=7)
    f.tight_layout(); f.savefig(fig / "fig7_skill.png", dpi=300); plt.close(f)

    pr = pd.read_csv(out / "forecast" / "preds_holdout2026.csv.gz")
    f, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k:")
    for m in MODELS:
        if f"p_{m}" in pr:
            fy, fx = calibration_curve(pr["y"], pr[f"p_{m}"], n_bins=10, strategy="quantile")
            ax.plot(fx, fy, "o-", ms=3, label=MODEL_NAME[m])
    ax.set_xlabel("Forecast probability"); ax.set_ylabel("Observed frequency"); ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(fig / "fig8_reliability.png", dpi=300); plt.close(f)

    f, ax = plt.subplots(figsize=(7, 0.5 * len(imp) + 1.5))
    im = ax.imshow(imp.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(imp.shape[1]), imp.columns, rotation=30, ha="right"); ax.set_yticks(range(len(imp)), [f"Regime {r}" for r in imp.index])
    plt.colorbar(im, ax=ax, label="ΔAUC-PR when permuted"); f.tight_layout(); f.savefig(fig / "fig9_importance.png", dpi=300); plt.close(f)
    print(f"results.json, {len(list(tab.glob('*.md')))} tabel, {len(list(fig.glob('*.png')))} gambar -> {out}")


if __name__ == "__main__":
    main()
