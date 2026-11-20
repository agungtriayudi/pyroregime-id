"""Pergeseran rezim antar tahun El Nino (Bagian 4.4 naskah).

    python regime_shift.py --ref-fp fp_2012_2025.csv --labels results/regimes/labels_selected.csv \
        --windows 2015=fp_ev2015.csv 2019=fp_ev2019.csv 2023=fp_ev2023.csv 2026=fp_ev2026.csv --out results/shift
Sidik jari tahun kejadian dibuat dengan regime_features.py --years Y-Y --months 1-10 --static-year Y-1.
Penugasan: LDA dilatih pada sidik jari periode acuan (hanya fitur yang terdefinisi dalam satu tahun, tanpa
embedding karena embedding tidak tersedia sebelum 2017) + label klaster.
Keluaran: assigned_<Y>.csv, transition_<a>_<b>.csv, shift_summary.csv, shift.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold

from cluster_regimes import Prep
from forecast import block_of

EVENT = ["log_det_per_year", "persist_mean", "persist_max", "night_frac", "log_frp_med",
         "season_sin", "season_cos", "season_conc", "peat_frac"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-fp", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--windows", nargs="+", required=True, help="Y=path.csv, urut waktu")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results/shift")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(a.ref_fp).merge(pd.read_csv(a.labels)[["cell", "regime"]], on="cell")
    # embedding sengaja dikeluarkan: tidak tersedia sebelum 2017, sehingga tahun kejadian jadi tak sebanding
    ref = ref.drop(columns=[c for c in ref if c.startswith("emb_")])
    prep = Prep(core=EVENT).fit(ref)
    X, y = prep.transform(ref), ref["regime"].to_numpy()
    # akurasi penugasan: CV 10 lipatan berblok 0,5 derajat
    blocks = block_of(ref["cell"])
    acc = []
    for tr, te in GroupKFold(10).split(X, y, blocks):
        acc.append((LinearDiscriminantAnalysis().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
    lda = LinearDiscriminantAnalysis().fit(X, y)
    lab = {}
    for w in a.windows:
        name, path = w.split("=", 1)
        d = pd.read_csv(path)
        lab[name] = pd.Series(lda.predict(prep.transform(d)), index=d["cell"], name=name)
        lab[name].rename("regime").to_csv(out / f"assigned_{name}.csv")
    rng = np.random.default_rng(0)
    names, rows = list(lab), []
    for A, Bn in zip(names, names[1:]):
        j = pd.concat([lab[A], lab[Bn]], axis=1, join="inner")
        pd.crosstab(j[A], j[Bn], normalize="index").to_csv(out / f"transition_{A}_{Bn}.csv")
        moved = (j[A] != j[Bn]).to_numpy()
        blk = block_of(j.index.to_numpy())
        idx = pd.Series(np.arange(len(j))).groupby(blk).indices
        keys = np.array(list(idx))
        bs = [moved[np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(a.boot)]
        appeared = lab[Bn][~lab[Bn].index.isin(lab[A].index)].value_counts().to_dict()
        gone = lab[A][~lab[A].index.isin(lab[Bn].index)].value_counts().to_dict()
        rows.append({"from": A, "to": Bn, "n_both": len(j), "frac_changed": float(moved.mean()),
                     "ci_low": float(np.quantile(bs, .025)), "ci_high": float(np.quantile(bs, .975)),
                     "appeared": int(sum(appeared.values())), "disappeared": int(sum(gone.values())),
                     "appeared_by_regime": {int(k): int(v) for k, v in appeared.items()},
                     "disappeared_by_regime": {int(k): int(v) for k, v in gone.items()}})
    pd.DataFrame(rows).to_csv(out / "shift_summary.csv", index=False)
    res = {"assign_acc": float(np.mean(acc)), **{f"{r['from']}_{r['to']}": r for r in rows}}
    json.dump(res, open(out / "shift.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)})
                      for k, v in res.items()}, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
