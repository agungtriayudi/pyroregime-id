"""Clustering sidik jari rezim api + pemilihan jumlah rezim + stabilitas bootstrap.

    python cluster_regimes.py --fp data/cube/fingerprint_2012_2025.csv --method kmeans gmm ae_kmeans \
        --k 3-12 --boot 20 --out results/regimes
    python cluster_regimes.py ... --method hdbscan
    python cluster_regimes.py ... --method idec --k 6     # butuh PyTorch

Keluaran: labels_selected.csv (cell, lat, lon, regime), labels_<metode>.csv pada k terpilih,
profile_selected.csv, selection_grid.csv (semua metode x k), selection.json (k, aturan, metode, kesepakatan).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CORE = ["log_det_per_year", "recurrence", "persist_mean", "persist_max", "night_frac", "log_frp_med",
        "season_sin", "season_cos", "season_conc", "enso_rho", "peat_frac"]


# ---------------------------------------------------------------- praproses
class Prep:
    def __init__(self, n_emb_pca=8, emb_weight=1.0, core=None):
        self.n_emb_pca, self.emb_weight, self.core_list = n_emb_pca, emb_weight, core or CORE

    def fit(self, df):
        self.core = [c for c in self.core_list if c in df]
        self.emb = sorted(c for c in df if c.startswith("emb_"))
        self.med = df[self.core + self.emb].median()
        X = self._core(df)
        self.sc = StandardScaler().fit(X)
        if self.emb:
            E = df[self.emb].fillna(self.med[self.emb]).to_numpy()
            self.pca = PCA(min(self.n_emb_pca, len(self.emb)), random_state=0).fit(E)
            self.sc_e = StandardScaler().fit(self.pca.transform(E))
        return self

    def _core(self, df):
        X = df.reindex(columns=self.core).fillna(self.med[self.core]).copy()  # kolom hilang -> median latih
        if "persist_max" in X:
            X["persist_max"] = np.log1p(X["persist_max"])
        return X.to_numpy()

    def transform(self, df):
        Z = self.sc.transform(self._core(df))
        if self.emb:
            E = df.reindex(columns=self.emb).fillna(self.med[self.emb]).to_numpy()
            Z = np.hstack([Z, self.emb_weight * self.sc_e.transform(self.pca.transform(E))])
        return Z


# ---------------------------------------------------------------- autoencoder (sklearn, tanpa GPU)
def ae_latent(X, latent=6, seed=0):
    ae = MLPRegressor(hidden_layer_sizes=(32, latent, 32), activation="tanh", max_iter=400,
                      early_stopping=True, random_state=seed).fit(X, X)
    h = np.tanh(X @ ae.coefs_[0] + ae.intercepts_[0])
    return ae, np.tanh(h @ ae.coefs_[1] + ae.intercepts_[1])


def ae_encode(ae, X):
    h = np.tanh(X @ ae.coefs_[0] + ae.intercepts_[0])
    return np.tanh(h @ ae.coefs_[1] + ae.intercepts_[1])


# ---------------------------------------------------------------- IDEC (PyTorch, opsional)
def idec(X, k, latent=6, epochs_pre=200, epochs=150, gamma=0.1, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    enc = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, latent))
    dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, X.shape[1]))
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], 1e-3)
    for _ in range(epochs_pre):                                   # 1) pretrain AE
        opt.zero_grad(); loss = ((dec(enc(Xt)) - Xt) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        z = enc(Xt).numpy()
    mu = nn.Parameter(torch.tensor(KMeans(k, n_init=10, random_state=seed).fit(z).cluster_centers_,
                                   dtype=torch.float32))
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters(), mu], 1e-3)

    def soft(z):                                                  # Student-t kernel (DEC)
        q = 1.0 / (1.0 + torch.cdist(z, mu) ** 2)
        return q / q.sum(1, keepdim=True)

    for ep in range(epochs):                                      # 2) KL(P||Q) + rekonstruksi
        if ep % 10 == 0:
            with torch.no_grad():
                q = soft(enc(Xt)); p = q ** 2 / q.sum(0); p = p / p.sum(1, keepdim=True)
        opt.zero_grad()
        z = enc(Xt)
        loss = gamma * nn.functional.kl_div(soft(z).log(), p, reduction="batchmean") + ((dec(z) - Xt) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return soft(enc(Xt)).argmax(1).numpy()


# ---------------------------------------------------------------- fit satu konfigurasi
def fit_one(method, X, k, seed=0):
    """Kembalikan (labels, predictor) dengan predictor(Xnew) -> labels, atau None."""
    if method == "kmeans":
        m = KMeans(k, n_init=10, random_state=seed).fit(X)
        return m.labels_, m.predict
    if method == "gmm":
        m = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed).fit(X)
        return m.predict(X), m.predict
    if method == "ae_kmeans":
        ae, z = ae_latent(X, seed=seed)
        km = KMeans(k, n_init=10, random_state=seed).fit(z)
        return km.labels_, lambda Xn: km.predict(ae_encode(ae, Xn))
    if method == "hdbscan":
        m = HDBSCAN(min_cluster_size=max(20, len(X) // 100)).fit(X)
        return m.labels_, None
    if method == "idec":
        return idec(X, k, seed=seed), None
    raise ValueError(method)


def scores(X, lab, rng, n=20000):
    ok = lab >= 0
    if len(np.unique(lab[ok])) < 2:
        return {}
    idx = np.flatnonzero(ok)
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    return {"silhouette": float(silhouette_score(X[idx], lab[idx])),
            "calinski_harabasz": float(calinski_harabasz_score(X[idx], lab[idx])),
            "davies_bouldin": float(davies_bouldin_score(X[idx], lab[idx])),
            "noise_frac": float(1 - ok.mean())}


def stability(method, X, k, ref, B, rng, frac=0.8):
    aris = []
    for b in range(B):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        lab_b, pred = fit_one(method, X[idx], k, seed=b + 1)
        if pred is not None:
            aris.append(adjusted_rand_score(ref, pred(X)))
        else:
            aris.append(adjusted_rand_score(ref[idx], lab_b))
    return float(np.mean(aris)), float(np.std(aris))


def select_k(rows, kmax):
    """Aturan pra-registrasi: k = argmin BIC GMM di antara k yang stabil (ARI bootstrap >= 0.8).
    Bila minimum jatuh di batas atas rentang, pakai titik lutut (selisih kedua terbesar) kurva BIC."""
    g = sorted((r for r in rows if r["method"] == "gmm" and r.get("stab_ari", 0) >= 0.8), key=lambda r: r["k"])
    if not g:
        return None, "no stable GMM solution"
    ks, bic = np.array([r["k"] for r in g]), np.array([r["bic"] for r in g])
    k = int(ks[bic.argmin()])
    if k == kmax and len(ks) >= 3:
        k = int(ks[1:-1][np.argmax(bic[:-2] - 2 * bic[1:-1] + bic[2:])])
        return k, "BIC knee (minimum at upper bound)"
    return k, "BIC minimum among stable solutions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", required=True)
    ap.add_argument("--method", nargs="+", default=["kmeans", "gmm", "ae_kmeans"])
    ap.add_argument("--k", default="3-12")
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--emb-weight", type=float, default=1.0, help="0 = abaikan embedding (ablasi E2)")
    ap.add_argument("--out", default="results/regimes")
    args = ap.parse_args()

    df = pd.read_csv(args.fp)
    prep = Prep(emb_weight=args.emb_weight).fit(df)
    X = prep.transform(df)
    ks = list(range(int(args.k.split("-")[0]), int(args.k.split("-")[-1]) + 1))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    print(f"{len(df):,} sel, {X.shape[1]} fitur ({len(prep.core)} inti, {len(prep.emb)} embedding)")
    if "gmm" not in args.method:
        args.method.append("gmm")                                        # dibutuhkan untuk aturan pemilihan k
    rows, labels = [], {}
    for method in args.method:
        for k in ([None] if method == "hdbscan" else ks):
            lab, _ = fit_one(method, X, k)
            kk = k if k else int(lab.max() + 1)
            row = {"method": method, "k": kk, **scores(X, lab, rng)}
            if method == "gmm":
                row["bic"] = float(GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit(X).bic(X))
            if args.boot and method in ("kmeans", "gmm", "ae_kmeans"):
                row["stab_ari"], row["stab_sd"] = stability(method, X, k, lab, args.boot, rng)
            rows.append(row); labels[(method, kk)] = lab
            print({a: round(b, 3) if isinstance(b, float) else b for a, b in row.items()})
    k_sel, rule = select_k(rows, max(ks))
    if k_sel is None:                                                    # cadangan: silhouette tertinggi
        k_sel = max((r for r in rows if r["method"] != "hdbscan"), key=lambda r: r.get("silhouette", -1))["k"]
        rule = "fallback: highest silhouette"
    cand = [r for r in rows if r["k"] == k_sel and r["method"] != "hdbscan"]
    best = max(cand, key=lambda r: (round(r.get("stab_ari", 0), 2), r.get("silhouette", -1)))
    lab = labels[(best["method"], k_sel)]
    others = [r["method"] for r in cand if r["method"] != best["method"]]
    agree = {m: float(adjusted_rand_score(lab, labels[(m, k_sel)])) for m in others}
    res = df[["cell", "lat", "lon"]].assign(regime=lab)
    res.to_csv(out / "labels_selected.csv", index=False)
    for (m, kk), l in labels.items():
        if kk == k_sel:
            df[["cell", "lat", "lon"]].assign(regime=l).to_csv(out / f"labels_{m}.csv", index=False)
    prof = df.drop(columns=["cell"]).assign(regime=lab).groupby("regime").mean(numeric_only=True)
    prof.insert(0, "n_cells", pd.Series(lab).value_counts().sort_index())
    prof.to_csv(out / "profile_selected.csv")
    pd.DataFrame(rows).to_csv(out / "selection_grid.csv", index=False)
    sel = {"k": int(k_sel), "rule": rule, "method": best["method"], "silhouette": best.get("silhouette"),
           "stab_ari": best.get("stab_ari"), "stab_sd": best.get("stab_sd"), "agreement": agree,
           "agreement_ari": max(agree.values()) if agree else None, "n_cells": int(len(df))}
    json.dump(sel, open(out / "selection.json", "w"), indent=1)
    print(f"-> k={k_sel} ({rule}); partisi dari {best['method']}; kesepakatan {agree}")


if __name__ == "__main__":
    main()
