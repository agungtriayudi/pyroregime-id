"""Prakiraan kejadian api 1-3 bulan terkondisi-rezim + evaluasi (Bagian 4.5-4.8 naskah).

    python forecast.py --config config.json
Keluaran di <out>/forecast/: results_forecast.json, preds_holdout2026.csv.gz, importance.csv, tables *.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from grid_features import cell_center

HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
           early_stopping=True, validation_fraction=0.1, random_state=0)
SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


# ---------------------------------------------------------------- masukan
def load_oni_monthly(path):
    df = pd.read_csv(path, sep=r"\s+"); df.columns = [c.upper() for c in df.columns]
    m = df["SEAS"].map({s: i + 1 for i, s in enumerate(SEAS)})          # DJF -> Jan (bulan tengah)
    idx = pd.PeriodIndex.from_fields(year=df["YR"], month=m, freq="M")
    return pd.Series(df["ANOM"].astype(float).to_numpy(), index=idx.astype(str))


def vpd_kpa(t2m_k, d2m_k):
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es(t2m_k - 273.15) - es(d2m_k - 273.15)


def block_of(cell, deg=0.5):
    lat, lon = cell_center(np.asarray(cell))
    return (np.floor((lat + 90) / deg) * 1000 + np.floor((lon + 180) / deg)).astype(np.int64)


GROUPS = {
    "fire_history": ["det0", "det1", "det12", "pers0", "clim"],
    "rainfall": ["precip", "p3_anom", "dry_days"],
    "moisture_vpd": ["swvl1", "sw_anom", "vpd"],
    "enso_iod": ["oni", "dmi"],
    "land_peat": ["peat_frac", "wc_tree", "wc_shrub", "wc_grass", "wc_crop"],
    "embeddings": [f"pc{i}" for i in range(8)],
}
FEATURESETS = {
    "fire_history": ["fire_history"],
    "hydro": ["fire_history", "rainfall", "moisture_vpd", "enso_iod"],
    "emb": ["fire_history", "land_peat", "embeddings"],
    "full": ["fire_history", "rainfall", "moisture_vpd", "enso_iod", "land_peat", "embeddings"],
}


def build_panel(cfg):
    cube = pd.read_csv(cfg["cell_month"])
    lab = pd.read_csv(cfg["labels"])[["cell", "regime"]]
    lab = lab[lab["regime"] >= 0]
    cells = lab["cell"].to_numpy()
    months = pd.period_range(cfg["panel_start"], cfg["panel_end"], freq="M").astype(str)
    P = pd.MultiIndex.from_product([cells, months], names=["cell", "month"]).to_frame(index=False)
    P = P.merge(cube[["cell", "month", "n_det", "persist_max"]], on=["cell", "month"], how="left").fillna({"n_det": 0, "persist_max": 0})
    P = P.merge(lab, on="cell")
    per = pd.PeriodIndex(P["month"], freq="M"); P["year"], P["mon"] = per.year, per.month
    P = P.sort_values(["cell", "month"]).reset_index(drop=True)
    g = P.groupby("cell")
    P["det0"] = np.log1p(P["n_det"])
    P["det1"] = g["det0"].shift(1)
    P["det12"] = np.log1p(g["n_det"].transform(lambda s: s.rolling(12, min_periods=1).sum()))
    P["pers0"] = P["persist_max"]
    # hidroklimat
    cov = pd.read_csv(cfg["gee_monthly"])
    P = P.merge(cov, on=["cell", "month"], how="left")
    P["vpd"] = vpd_kpa(P["t2m"], P["d2m"])
    P["p3"] = P.groupby("cell")["precip_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    ref = P[P["year"].between(*cfg["ref_years"])]
    c3 = ref.groupby(["cell", "mon"])["p3"].mean().rename("p3_clim")
    csw = ref.groupby(["cell", "mon"])["swvl1"].mean().rename("sw_clim")
    P = P.join(c3, on=["cell", "mon"]).join(csw, on=["cell", "mon"])
    P["p3_anom"] = np.log((P["p3"] + 1) / (P["p3_clim"] + 1))
    P["precip"], P["sw_anom"] = P["precip_mm"], P["swvl1"] - P["sw_clim"]
    oni = load_oni_monthly(cfg["oni"])
    P["oni"] = P["month"].map(oni)
    if cfg.get("dmi"):
        dmi = pd.read_csv(cfg["dmi"]).set_index("month")["dmi"]
        P["dmi"] = P["month"].map(dmi)
    if cfg.get("fwi"):
        P = P.merge(pd.read_csv(cfg["fwi"]), on=["cell", "month"], how="left")
    # statis + embedding tahun-1 (aturan t+1; sebelum 2018 = hilang)
    st = pd.read_csv(cfg["static"])
    land = st.groupby("cell")[[c for c in ["peat_frac", "wc_tree", "wc_shrub", "wc_grass", "wc_crop"] if c in st]].mean()
    P = P.join(land, on="cell")
    emb_cols = [c for c in st if c.startswith("emb_")]
    if emb_cols:
        pca = PCA(8, random_state=0).fit(st[emb_cols].to_numpy())
        E = pd.DataFrame(pca.transform(st[emb_cols].to_numpy()), columns=[f"pc{i}" for i in range(8)])
        E["cell"], E["emb_year"] = st["cell"].to_numpy(), st["year"].to_numpy() + 1   # embedding y dipakai untuk y+1
        P = P.merge(E, left_on=["cell", "year"], right_on=["cell", "emb_year"], how="left").drop(columns="emb_year")
    P["block"] = block_of(P["cell"])
    return P


def add_targets(P, L):
    P = P.copy()
    g = P.groupby("cell")
    P["y"] = (g["n_det"].shift(-L) > 0).astype(float)
    P["target_month"] = g["month"].shift(-L)
    P = P.dropna(subset=["target_month"])
    tp = pd.PeriodIndex(P["target_month"], freq="M")
    P["t_year"], P["t_mon"] = tp.year, tp.month
    return P


# ---------------------------------------------------------------- klimatologi (dihitung dari data latih saja)
def climatology(train, apply, level):
    key = ["cell", "t_mon"] if level == "cell" else ["regime", "t_mon"]
    agg = train.groupby(key)["y"].agg(["sum", "count"])
    rate = ((agg["sum"] + 1) / (agg["count"] + 2)).rename("clim")
    out = apply.join(rate, on=key)["clim"]
    if level == "cell":                                                  # sel tanpa riwayat -> rezim x bulan
        fb = climatology(train, apply, "regime")
        out = out.fillna(pd.Series(fb, index=out.index))
    return out.fillna((train["y"].sum() + 1) / (len(train) + 2)).to_numpy()


# ---------------------------------------------------------------- model
def feat_cols(P, groups):
    return [c for g in groups for c in GROUPS[g] if c in P]


def fit_predict(model, tr, te, cols, regimes, neg_frac=1.0, seed=0):
    rng = np.random.default_rng(seed)
    if neg_frac < 1.0:                                                   # subsampel negatif (data besar)
        keep = (tr["y"] == 1) | (rng.random(len(tr)) < neg_frac)
        tr = tr[keep]

    def correct(p):                                                      # koreksi prior shift
        if neg_frac >= 1.0:
            return p
        odds = p / (1 - p + 1e-12) * neg_frac
        return odds / (1 + odds)

    if model == "clim":
        return te["clim"].to_numpy()
    if model == "fwi":
        X = lambda d: np.c_[d["fwi"].fillna(d["fwi"].median()), np.log(d["clim"] / (1 - d["clim"]))]
        m = LogisticRegression(max_iter=1000).fit(X(tr), tr["y"])
        return correct(m.predict_proba(X(te))[:, 1])
    if model in ("global", "global_regime"):
        c = cols + ([f"reg_{r}" for r in regimes] if model == "global_regime" else [])
        m = HistGradientBoostingClassifier(**HGB).fit(tr[c], tr["y"])
        return correct(m.predict_proba(te[c])[:, 1])
    if model == "moe":
        p = np.zeros(len(te))
        for r in regimes:
            a, b = tr["regime"] == r, (te["regime"] == r).to_numpy()
            if not b.any():
                continue
            yr = tr.loc[a, "y"]
            if yr.nunique() < 2:
                p[b] = yr.mean() if len(yr) else te.loc[b, "clim"]
                continue
            m = HistGradientBoostingClassifier(**HGB).fit(tr.loc[a, cols], yr)
            p[b] = m.predict_proba(te.loc[b, cols])[:, 1]
        return correct(p)
    raise ValueError(model)


def metrics(y, p, clim):
    if y.sum() == 0 or y.sum() == len(y):
        return {}
    bs, bc = brier_score_loss(y, p), brier_score_loss(y, clim)
    return {"auc_pr": average_precision_score(y, p), "auc_roc": roc_auc_score(y, p),
            "brier": bs, "bss": 1 - bs / bc, "base_rate": float(y.mean()), "n": int(len(y))}


def block_boot(df, fn, B, seed=0):
    """fn(sub) -> skalar; bootstrap blok 0,5 derajat."""
    rng = np.random.default_rng(seed)
    idx = df.groupby("block").indices
    keys = np.array(list(idx))
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[k] for k in rng.choice(keys, len(keys), replace=True)])
        try:
            vals.append(fn(df.iloc[take]))
        except ValueError:
            continue
    return np.array(vals)


# ---------------------------------------------------------------- desain evaluasi
def run_design(P, design, cfg, models, cols, regimes):
    """Kembalikan DataFrame prediksi out-of-sample dengan kolom p_<model>."""
    ref = P[P["t_year"] <= cfg["ref_years"][1]]
    if design == "spatial_cv":
        folds = [(ref.index[tr], ref.index[te]) for tr, te in
                 GroupKFold(5).split(ref, groups=ref["block"])]
        level = "regime"                                                 # sel uji tak terlihat -> klimatologi rezim
        pool = ref
    elif design == "loeno":
        folds, level, pool = [], "cell", ref
        for Y in cfg["enso_years"]:
            te = ref.index[ref["t_year"] == Y]
            tr = ref.index[(ref["t_year"] != Y) & (ref["year"] != Y)]
            if len(te):
                folds.append((tr, te))
    elif design == "holdout2026":
        cut = pd.Period(cfg["holdout_train_end"], freq="M")
        tp = pd.PeriodIndex(P["target_month"], freq="M")
        tr, te = P.index[tp <= cut], P.index[tp > cut]
        folds, level, pool = [(tr, te)], "cell", P
    else:
        raise ValueError(design)
    outs = []
    for i, (tr_i, te_i) in enumerate(folds):
        tr, te = pool.loc[tr_i].copy(), pool.loc[te_i].copy()
        tr["clim"] = climatology(tr, tr, level); te["clim"] = climatology(tr, te, level)
        for m in models:
            if m == "fwi" and "fwi" not in P:
                continue
            te[f"p_{m}"] = fit_predict(m, tr, te, cols, regimes, cfg.get("neg_frac", 1.0), seed=i)
        te["fold"] = i
        outs.append(te)
    return pd.concat(outs)


def summarize(pred, models, B, key_pairs=(("moe", "global"),)):
    res = {}
    for m in models:
        if f"p_{m}" not in pred:
            continue
        r = metrics(pred["y"].to_numpy(), pred[f"p_{m}"].to_numpy(), pred["clim"].to_numpy())
        bs = block_boot(pred, lambda d: average_precision_score(d["y"], d[f"p_{m}"]), B)
        if len(bs):
            r["auc_pr_ci_low"], r["auc_pr_ci_high"] = np.quantile(bs, [0.025, 0.975])
        res[m] = r
    for a, b in key_pairs:
        if f"p_{a}" in pred and f"p_{b}" in pred:
            d = block_boot(pred, lambda s: average_precision_score(s["y"], s[f"p_{a}"]) - average_precision_score(s["y"], s[f"p_{b}"]), B, seed=1)
            obs = average_precision_score(pred["y"], pred[f"p_{a}"]) - average_precision_score(pred["y"], pred[f"p_{b}"])
            res[f"delta_{a}_{b}"] = {"mean": obs, "ci_low": np.quantile(d, .025), "ci_high": np.quantile(d, .975),
                                     "p": float((d <= 0).mean())}
    return res


def featureset_experiment(P, cfg, regimes, B):
    ref = P[P["t_year"] <= cfg["ref_years"][1]]
    out, preds = {}, {}
    for name, groups in FEATURESETS.items():
        cols = feat_cols(P, groups) + ["clim"]                             # klimatologi selalu tersedia
        parts = []
        for tr_i, te_i in GroupKFold(5).split(ref, groups=ref["block"]):
            tr, te = ref.iloc[tr_i].copy(), ref.iloc[te_i].copy()
            tr["clim"] = climatology(tr, tr, "regime"); te["clim"] = climatology(tr, te, "regime")
            te["p"] = fit_predict("global", tr, te, cols, regimes, cfg.get("neg_frac", 1.0))
            parts.append(te[["y", "p", "peat_frac", "block"]])
        preds[name] = pd.concat(parts).sort_index()
    for subset, mask in (("all", lambda d: np.ones(len(d), bool)), ("peat", lambda d: (d["peat_frac"] >= 0.5).to_numpy())):
        out[subset] = {}
        for name, d in preds.items():
            s = d[mask(d)]
            if s["y"].nunique() == 2:
                out[subset][name] = {"auc_pr": average_precision_score(s["y"], s["p"])}
        full, emb = preds["full"], preds["emb"]
        j = full[["y", "block", "peat_frac"]].assign(pf=full["p"], pe=emb["p"])
        j = j[mask(j)]
        if j["y"].nunique() == 2:
            obs = average_precision_score(j["y"], j["pf"]) - average_precision_score(j["y"], j["pe"])
            bs = block_boot(j, lambda s: average_precision_score(s["y"], s["pf"]) - average_precision_score(s["y"], s["pe"]), B, seed=2)
            out[subset]["delta_full_emb"] = obs
            out[subset]["delta_ci_low"], out[subset]["delta_ci_high"] = np.quantile(bs, [.025, .975])
    return out


def grouped_importance(P, cfg, regimes, cols, repeats=3):
    """Penurunan AUC-PR saat semua prediktor satu grup dipermutasi bersama, per rezim (spatial CV, L=1)."""
    ref = P[P["t_year"] <= cfg["ref_years"][1]]
    rng = np.random.default_rng(0)
    rows = []
    for fold, (tr_i, te_i) in enumerate(GroupKFold(5).split(ref, groups=ref["block"])):
        tr, te = ref.iloc[tr_i].copy(), ref.iloc[te_i].copy()
        tr["clim"] = climatology(tr, tr, "regime"); te["clim"] = climatology(tr, te, "regime")
        for r in regimes:
            a, b = tr["regime"] == r, te["regime"] == r
            if tr.loc[a, "y"].nunique() < 2 or te.loc[b, "y"].nunique() < 2:
                continue
            m = HistGradientBoostingClassifier(**HGB).fit(tr.loc[a, cols], tr.loc[a, "y"])
            X = te.loc[b, cols].copy(); y = te.loc[b, "y"]
            base = average_precision_score(y, m.predict_proba(X)[:, 1])
            for g, gc in GROUPS.items():
                gc = [c for c in gc if c in cols]
                if not gc:
                    continue
                drops = []
                for _ in range(repeats):
                    Xp = X.copy(); Xp[gc] = X[gc].to_numpy()[rng.permutation(len(X))]
                    drops.append(base - average_precision_score(y, m.predict_proba(Xp)[:, 1]))
                rows.append({"fold": fold, "regime": r, "group": g, "delta_auc_pr": float(np.mean(drops))})
    return pd.DataFrame(rows).groupby(["regime", "group"])["delta_auc_pr"].mean().unstack()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = Path(cfg["out"]) / "forecast"; out.mkdir(parents=True, exist_ok=True)
    B = cfg.get("bootstrap", 1000)
    P0 = build_panel(cfg)
    regimes = sorted(P0["regime"].unique())
    for r in regimes:
        P0[f"reg_{r}"] = (P0["regime"] == r).astype(float)
    models = ["clim", "fwi", "global", "global_regime", "moe"]
    cols = feat_cols(P0, FEATURESETS["full"]) + ["clim"]
    results = {}
    for L in cfg.get("leads", [1, 2, 3]):
        P = add_targets(P0, L)
        for design in ("spatial_cv", "loeno", "holdout2026"):
            pred = run_design(P, design, cfg, models, cols, regimes)
            s = summarize(pred, models, B)
            for k, v in s.items():
                results.setdefault(design, {}).setdefault(k, {})[f"L{L}"] = v
            if design == "holdout2026" and L == 1:
                pred[["cell", "target_month", "regime", "y", "clim"] + [c for c in pred if c.startswith("p_")]].to_csv(
                    out / "preds_holdout2026.csv.gz", index=False)
            print(f"L{L} {design}: " + ", ".join(f"{m}={s[m]['auc_pr']:.3f}" for m in models if m in s and "auc_pr" in s[m]))
    P1 = add_targets(P0, 1)
    results["featureset"] = featureset_experiment(P1, cfg, regimes, B)
    imp = grouped_importance(P1, cfg, regimes, cols)
    imp.to_csv(out / "importance.csv")
    json.dump(results, open(out / "results_forecast.json", "w"), indent=1, default=float)
    print("featureset:", json.dumps(results["featureset"], default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
