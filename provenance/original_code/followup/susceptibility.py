"""Uji EKSPLORATIF kerentanan statis (setara tugas Zhuang dkk. 2026): apakah embedding AlphaEarth menyamai
prediktor fisik statis untuk memetakan DI MANA api terjadi? Desain ditetapkan di registration/ADDENDUM_followup.md
sebelum dijalankan; jangan diubah setelah melihat hasil.

    python followup/susceptibility.py --config config.json [--cells data/followup/susc_cells.csv]

Satuan: sel 0,01 derajat (darat, in_indonesia >= 0,5). Target: y = 1 bila ada >= 1 deteksi VIIRS tersaring
pada Jan 2019 - Des 2025. Prediktor (semuanya mendahului periode target):
  P  fisik statis : peat_frac, wc_* , elev, slope, rain_annual, dry_days_aso, swvl1_aso, t2m_annual (iklim 2012-2018)
  E  embedding    : emb_A00..emb_A63 = rerata AlphaEarth 2017 dan 2018
  F  riwayat api  : log1p(deteksi 2012-2018), jumlah tahun terbakar 2012-2018
Set: P, E, P+E, F, F+P, F+E, F+P+E. Learner sama dengan analisis utama (HGB, 300 iterasi, lr 0,05, 31 daun,
L2 1,0, early stopping 10 %), validasi silang blok spasial 0,5 derajat (5 lipatan), bootstrap blok 1.000 replikasi.
Kriteria yang dipra-tetapkan: embedding dinilai setara bila 95 % CI AUC-PR(E) - AUC-PR(P) mencakup nol atau di atas
nol, dihitung terpisah untuk semua sel dan sel gambut (peat_frac >= 0,5).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

PHYS = ["peat_frac", "wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built", "wc_bare", "wc_wetland", "wc_mangrove",
        "elev", "slope", "rain_annual", "dry_days_aso", "swvl1_aso", "t2m_annual"]
SETS = {"P": ["P"], "E": ["E"], "P+E": ["P", "E"], "F": ["F"], "F+P": ["F", "P"], "F+E": ["F", "E"],
        "F+P+E": ["F", "P", "E"]}
DELTAS = [("E", "P"), ("P+E", "P"), ("P+E", "E"), ("F+P+E", "F+P"), ("F+E", "F+P")]
TARGET = ("2019-01", "2025-12")
HIST = ("2012-01", "2018-12")


def block_of(lat, lon, deg=0.5):
    return (np.floor((lat + 90) / deg) * 1000 + np.floor((lon + 180) / deg)).astype(np.int64)


def build(cells_path, cube_path):
    d = pd.read_csv(cells_path)
    if "in_indonesia" in d:
        d = d[d["in_indonesia"] >= 0.5]
    if "wc_water" in d:
        d = d[d["wc_water"] < 0.5]
    cube = pd.read_csv(cube_path, usecols=["cell", "month", "n_det"])
    tgt = cube[(cube["month"] >= TARGET[0]) & (cube["month"] <= TARGET[1])].groupby("cell")["n_det"].sum()
    hist = cube[(cube["month"] >= HIST[0]) & (cube["month"] <= HIST[1])].copy()
    hist["year"] = hist["month"].str[:4]
    d["y"] = d["cell"].map(tgt).fillna(0).gt(0).astype(int)
    d["F_log_det"] = np.log1p(d["cell"].map(hist.groupby("cell")["n_det"].sum()).fillna(0))
    d["F_years"] = d["cell"].map(hist[hist["n_det"] > 0].groupby("cell")["year"].nunique()).fillna(0)
    lat = (d["cell"] // 100000) * 0.01 - 90 + 0.005; lon = (d["cell"] % 100000) * 0.01 - 180 + 0.005
    d["block"] = block_of(lat.to_numpy(), lon.to_numpy())
    return d.reset_index(drop=True)


def columns(d):
    return {"P": [c for c in PHYS if c in d], "E": sorted(c for c in d if c.startswith("emb_")), "F": ["F_log_det", "F_years"]}


def oof(d, cols, seed=0):
    p = np.zeros(len(d))
    for i, (tr, te) in enumerate(GroupKFold(5).split(d, groups=d["block"])):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.1, random_state=seed + i)
        m.fit(d.iloc[tr][cols], d.iloc[tr]["y"])
        p[te] = m.predict_proba(d.iloc[te][cols])[:, 1]
    return p


def boot_deltas(d, preds, B, seed=0):
    rng = np.random.default_rng(seed)
    idx = d.groupby("block").indices
    keys = np.array(list(idx))
    y = d["y"].to_numpy()
    out = {f"{a}-{b}": [] for a, b in DELTAS}
    per_set = {s: [] for s in preds}
    for _ in range(B):
        take = np.concatenate([idx[k] for k in rng.choice(keys, len(keys), replace=True)])
        if y[take].min() == y[take].max():
            continue
        ap = {s: average_precision_score(y[take], p[take]) for s, p in preds.items()}
        for s, v in ap.items():
            per_set[s].append(v)
        for a, b in DELTAS:
            out[f"{a}-{b}"].append(ap[a] - ap[b])
    return per_set, out


def summarize(d, preds, B):
    y = d["y"].to_numpy()
    per_set, deltas = boot_deltas(d, preds, B)
    res = {"n": int(len(d)), "base_rate": float(y.mean()), "sets": {}, "deltas": {}}
    for s, p in preds.items():
        bs = np.array(per_set[s])
        res["sets"][s] = {"auc_pr": float(average_precision_score(y, p)), "auc_roc": float(roc_auc_score(y, p)),
                          "auc_pr_ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    for a, b in DELTAS:
        v = np.array(deltas[f"{a}-{b}"])
        obs = res["sets"][a]["auc_pr"] - res["sets"][b]["auc_pr"]
        res["deltas"][f"{a} − {b}"] = {"mean": obs, "ci_low": float(np.quantile(v, .025)), "ci_high": float(np.quantile(v, .975))}
    e_p = res["deltas"]["E − P"]
    res["prespecified_embeddings_comparable"] = bool(e_p["ci_high"] >= 0)   # CI mencakup nol atau di atas nol
    return res


def table(res):
    lines = ["| Predictor set | AUC-PR all (95 % CI) | ROC-AUC all | AUC-PR peat (95 % CI) | ROC-AUC peat |",
             "| --- | --- | --- | --- | --- |"]
    for s in SETS:
        A, Pt = res["all"]["sets"][s], res["peat"]["sets"][s]
        lines.append(f"| {s} | {A['auc_pr']:.3f} ({A['auc_pr_ci'][0]:.3f}–{A['auc_pr_ci'][1]:.3f}) | {A['auc_roc']:.3f} | "
                     f"{Pt['auc_pr']:.3f} ({Pt['auc_pr_ci'][0]:.3f}–{Pt['auc_pr_ci'][1]:.3f}) | {Pt['auc_roc']:.3f} |")
    lines += ["", "| Difference | All cells (95 % CI) | Peat cells (95 % CI) |", "| --- | --- | --- |"]
    for k in res["all"]["deltas"]:
        A, Pt = res["all"]["deltas"][k], res["peat"]["deltas"][k]
        lines.append(f"| {k} | {A['mean']:.3f} ({A['ci_low']:.3f} to {A['ci_high']:.3f}) | "
                     f"{Pt['mean']:.3f} ({Pt['ci_low']:.3f} to {Pt['ci_high']:.3f}) |")
    return "\n".join(lines) + "\n"


def figure(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, sub in zip(axes, ("all", "peat")):
        S = list(SETS)
        v = [res[sub]["sets"][s]["auc_pr"] for s in S]
        lo = [v[i] - res[sub]["sets"][s]["auc_pr_ci"][0] for i, s in enumerate(S)]
        hi = [res[sub]["sets"][s]["auc_pr_ci"][1] - v[i] for i, s in enumerate(S)]
        ax.bar(S, v, yerr=[lo, hi], color=["#1f77b4", "#ff7f0e", "#2ca02c", "0.6", "#9467bd", "#8c564b", "#d62728"], capsize=3)
        ax.axhline(res[sub]["base_rate"], color="k", lw=0.8, ls=":", label="base rate")
        ax.set_title("All cells" if sub == "all" else "Peat cells (peat fraction ≥ 0.5)", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("AUC-PR (static susceptibility, 2019–2025)"); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--boot", type=int, default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = Path(cfg["out"]) / "followup" / "susceptibility"; out.mkdir(parents=True, exist_ok=True)
    d = build(a.cells or cfg.get("followup_susc_cells", "data/followup/susc_cells.csv"), cfg["cell_month"])
    C = columns(d)
    if len(C["E"]) != 64 or len(C["P"]) < 5:
        raise SystemExit(f"kolom tidak lengkap: P={C['P']}, E={len(C['E'])} kolom")
    preds = {s: oof(d, [c for g in groups for c in C[g]]) for s, groups in SETS.items()}
    B = a.boot if a.boot is not None else cfg.get("bootstrap", 1000)
    peat = d["peat_frac"].to_numpy() >= 0.5
    res = {"design": {"target": TARGET, "history": HIST, "embedding_years": [2017, 2018], "physical": C["P"],
                      "cv": "GroupKFold(5) on 0.5° blocks", "bootstrap": B, "exploratory": True},
           "all": summarize(d, preds, B),
           "peat": summarize(d[peat].reset_index(drop=True), {s: p[peat] for s, p in preds.items()}, B)}
    json.dump(res, open(out / "susceptibility.json", "w"), indent=1)
    (out / "susceptibility.md").write_text(table(res))
    figure(res, out / "fig_susceptibility.png")
    print(table(res))
    print("Kriteria pra-tetapkan, embedding setara fisik — semua sel:", res["all"]["prespecified_embeddings_comparable"],
          "| sel gambut:", res["peat"]["prespecified_embeddings_comparable"])


if __name__ == "__main__":
    main()
