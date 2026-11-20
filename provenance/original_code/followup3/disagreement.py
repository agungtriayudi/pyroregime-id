"""Follow-up 3 (EXPLORATORY, POST HOC): total, quantity and allocation disagreement between S-NPP and NOAA-20.

    python followup3/disagreement.py --config config.json

Why. Follow-up 2 (followup2/intersensor.py, registration/ADDENDUM_followup2.md) summarised the agreement of the two
sensors on monthly fire occurrence with Cohen's kappa. Kappa is unsuitable for comparing maps (Pontius and Millones,
2011; Foody, 2020), so the same 2 x 2 tables are re-expressed as total disagreement D, quantity disagreement Q and
allocation disagreement A (Pontius and Millones, 2011). The change was made after the follow-up-2 results were known;
it is post hoc and is reported as such (registration/ADDENDUM_followup3.md). No data, domain or stratum is added.

How. The 2 x 2 tables are built exactly as in follow-up 2, part B (same domains, strata and 0.5-degree blocks), and
the block bootstrap uses the same seed and the same order of random draws. Each replicate is therefore the replicate
of follow-up 2 as long as the NumPy random stream is unchanged. The script checks this by recomputing the follow-up-2
intervals of kappa, positive agreement and proportion of agreement and comparing them with
results/followup2/intersensor.json. For the 2026 hold-out (follow-up 2, part E) the follow-up-2 replicates cannot be
matched without re-running parts C and D, so a separate fixed seed is used there.

Output: results/followup3/disagreement.json. Nothing else under results/ is changed.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grid_features import cell_center  # noqa: E402  (registered code, imported unchanged)

OVERLAP = ("2018-04", "2025-12")   # overlap period of follow-up 2
SEED_FU2 = 20261117                # random seed of followup2/intersensor.py
SEED_HOLDOUT = 20261120            # separate fixed seed for the 2026 hold-out part
STRATA_KEYS = ("kappa", "ppos", "po")


def block_of(cell, deg=0.5):
    """0.5-degree block of each cell. Same code as forecast.block_of (registered); copied here so that this script
    does not need scikit-learn."""
    lat, lon = cell_center(np.asarray(cell))
    return (np.floor((lat + 90) / deg) * 1000 + np.floor((lon + 180) / deg)).astype(np.int64)


def agree_stats(a, b, c, d):
    """Agreement statistics of follow-up 2 (same code). Used only to check that the bootstrap is replicated."""
    a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))
    n = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (a + d) / n
        pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
        return {"po": po, "kappa": (po - pe) / (1 - pe), "ppos": 2 * a / (2 * a + b + c)}


def disagreement(a, b, c, d):
    """Disagreement components for a 2 x 2 table (Pontius and Millones, 2011).

    a = fire in both records, b = fire in the reference record only, c = fire in the other record only,
    d = fire in neither. D, Q and A are proportions of all cell-months, with D = Q + A:
      D = (b + c) / n, Q = |b - c| / n, A = 2 min(b, c) / n.
    A_share = A / D is the part of the disagreement due to allocation; D_fire = (b + c) / (a + b + c) is the
    disagreement among cell-months in which at least one record shows fire.
    """
    a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))
    n = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        D = (b + c) / n
        Q = np.abs(b - c) / n
        A = 2 * np.minimum(b, c) / n
        return {"D": D, "Q": Q, "A": A, "A_share": A / D, "D_fire": (b + c) / (a + b + c)}


def boot_sums(M, B, rng, chunk=100):
    """Column sums of M (n_blocks x p) over blocks resampled with replacement -> (B x p). Same code as follow-up 2."""
    nb = len(M)
    out = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, nb, (min(chunk, B - s), nb))
        out.append(M[idx].sum(axis=1))
    return np.vstack(out)


def q(v):
    """2.5th and 97.5th percentiles, ignoring NaN. Same code as follow-up 2."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if len(v) else (float("nan"), float("nan"))


def summarize(M, B, rng):
    """Point values and 95 % block-bootstrap intervals of D, Q, A; one bootstrap call per stratum, as in follow-up 2."""
    tot = M.sum(axis=0)
    bs = boot_sums(M, B, rng).T
    res = {"a_both": int(tot[0]), "b_ref_only": int(tot[1]), "c_other_only": int(tot[2]), "d_neither": int(tot[3]),
           "n": int(tot.sum())}
    pt, bd = disagreement(*tot), disagreement(*bs)
    for k in pt:
        res[k] = float(pt[k])
        res[f"{k}_ci_low"], res[f"{k}_ci_high"] = q(bd[k])
    ba = agree_stats(*bs)
    res["check_followup2"] = {k: list(q(ba[k])) for k in STRATA_KEYS}
    return res


def occ_table(S, N, cells):
    """Cell-months with at least one detection from S-NPP (s) or NOAA-20 (n). Same code as follow-up 2."""
    s = S.loc[S["cell"].isin(cells) & (S["n_det"] > 0), ["cell", "month"]].assign(s=1)
    n = N.loc[N["cell"].isin(cells) & (N["n_det"] > 0), ["cell", "month"]].assign(n=1)
    o = s.merge(n, on=["cell", "month"], how="outer").fillna({"s": 0, "n": 0})
    o["year"] = o["month"].str[:4].astype(int)
    return o


def block_counts(o, info, months):
    """Block x (a, b, c, d) matrix for a subset of cells and months. Same code as follow-up 2."""
    sub = o[o["cell"].isin(info["cell"]) & o["month"].isin(months)].merge(info[["cell", "block"]], on="cell")
    g = pd.DataFrame({"block": sub["block"], "a": (sub["s"] == 1) & (sub["n"] == 1),
                      "b": (sub["s"] == 1) & (sub["n"] == 0), "c": (sub["s"] == 0) & (sub["n"] == 1)}
                     ).groupby("block").sum()
    tot = info.groupby("block").size() * len(months)
    g = g.reindex(tot.index, fill_value=0).astype(float)
    g["d"] = tot - g.sum(axis=1)
    return g[["a", "b", "c", "d"]].to_numpy()


def part_occurrence(S, N, domains, months, B, rng):
    """Same domains, strata and order of bootstrap calls as follow-up 2, part B."""
    res = {}
    for name, info in domains.items():
        o = occ_table(S, N, info["cell"])
        r = {"n_cells": int(len(info)), "n_months": len(months),
             "all": summarize(block_counts(o, info, months), B, rng)}
        for lab, sub in (("peat", info[info["peat_frac"] >= 0.5]), ("non_peat", info[info["peat_frac"] < 0.5])):
            if len(sub) >= 50:
                r[lab] = dict(summarize(block_counts(o, sub, months), B, rng), n_cells=int(len(sub)))
        r["by_year"] = {}
        for y in sorted({m[:4] for m in months}):
            ms = [m for m in months if m.startswith(y)]
            r["by_year"][y] = dict(summarize(block_counts(o, info, ms), B, rng), n_months=len(ms))
        if "regime" in info:
            r["by_regime"] = {}
            for k, sub in info.groupby("regime"):
                if len(sub) >= 50:
                    r["by_regime"][str(int(k))] = dict(summarize(block_counts(o, sub, months), B, rng),
                                                       n_cells=int(len(sub)))
        res[name] = r
    return res


def strata(r):
    """(label, stratum dict) pairs of one domain, in a fixed order."""
    out = [("all", r["all"])] + [(k, r[k]) for k in ("peat", "non_peat") if k in r]
    out += [(f"year {y}", v) for y, v in r.get("by_year", {}).items()]
    out += [(f"regime {k}", v) for k, v in r.get("by_regime", {}).items()]
    return out


def replication_check(occ, fu2):
    """Largest absolute difference between the recomputed and the follow-up-2 intervals and counts."""
    worst, n = 0.0, 0
    for dom, r in occ.items():
        old = dict(strata(fu2["occurrence"][dom]))
        for lab, s in strata(r):
            o = old[lab]
            for k in ("a_both", "b_ref_only", "c_other_only", "d_neither"):
                if s[k] != o[k]:
                    raise SystemExit(f"CHECK FAILED: count {k} differs in {dom} / {lab}")
            for k in STRATA_KEYS:
                lo, hi = s["check_followup2"][k]
                worst = max(worst, abs(lo - o[f"{k}_ci_low"]), abs(hi - o[f"{k}_ci_high"]))
                n += 2
    return {"n_interval_bounds_compared": n, "max_abs_diff": worst}


def part_holdout(res_dir, cube_path, months, B):
    """2026 hold-out: registered target (S-NPP NRT) against NOAA-20, as in follow-up 2, part E."""
    pred = pd.read_csv(res_dir / "forecast" / "preds_holdout2026.csv.gz")
    pred = pred[pred["target_month"].isin(months)].copy()
    alt = pd.read_csv(cube_path)
    alt = alt[(alt["n_det"] > 0) & alt["month"].isin(months)]
    key = set(zip(alt["cell"].to_numpy(), alt["month"].to_numpy()))
    pred["y_alt"] = [float((c, m) in key) for c, m in zip(pred["cell"].to_numpy(), pred["target_month"].to_numpy())]
    pred["block"] = block_of(pred["cell"].to_numpy())
    y, ya = pred["y"].to_numpy() == 1, pred["y_alt"].to_numpy() == 1
    M = pd.DataFrame({"block": pred["block"], "a": y & ya, "b": y & ~ya, "c": ~y & ya, "d": ~y & ~ya}
                     ).groupby("block").sum().to_numpy(float)
    r = summarize(M, B, np.random.default_rng(SEED_HOLDOUT))
    r["months"] = list(months)
    return r


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--noaa20-2026", default="data/cube/followup2/cell_month_noaa20_2026.csv.gz")
    ap.add_argument("--noaa20-months", nargs="*", default=["2026-08", "2026-09"])
    a = ap.parse_args()
    t0 = datetime.now(timezone.utc)
    cfg = json.load(open(a.config))
    res_dir = Path(cfg["out"])
    out = res_dir / "followup3"
    out.mkdir(parents=True, exist_ok=True)
    fu2 = json.load(open(res_dir / "followup2" / "intersensor.json"))
    B = int(fu2["meta"]["bootstrap"])
    rng = np.random.default_rng(SEED_FU2)
    months = pd.period_range(*OVERLAP, freq="M").astype(str).tolist()

    S_all = pd.read_csv(cfg["cell_month"])
    N_all = pd.read_csv(cfg["cell_month_noaa20"])
    S = S_all[S_all["month"].between(*OVERLAP)].copy()
    N = N_all[N_all["month"].between(*OVERLAP)].copy()
    cols = pd.read_csv(cfg["static"], nrows=0).columns
    st = pd.read_csv(cfg["static"], usecols=["cell"] + [c for c in ("in_indonesia", "peat_frac") if c in cols])
    land = st.groupby("cell")[[c for c in ("in_indonesia", "peat_frac") if c in st]].mean()
    if "in_indonesia" not in land:
        land["in_indonesia"] = 1.0

    # domains exactly as in follow-up 2
    lab = pd.read_csv(res_dir / "regimes" / "labels_selected.csv")
    dom1 = lab[["cell", "regime"]].join(land["peat_frac"], on="cell")
    dom1["block"] = block_of(dom1["cell"].to_numpy())
    md = int(cfg.get("min_det", 20))
    tS, tN = S.groupby("cell")["n_det"].sum(), N.groupby("cell")["n_det"].sum()
    union = pd.Index(tS[tS >= md].index).union(tN[tN >= md].index)
    dom2 = land.loc[land.index.intersection(union)]
    dom2 = dom2[dom2["in_indonesia"] > 0.5].reset_index()[["cell", "peat_frac"]]
    dom2["block"] = block_of(dom2["cell"].to_numpy())
    domains = {"forecast_domain": dom1, "union_domain": dom2}

    occ = part_occurrence(S, N, domains, months, B, rng)
    check = replication_check(occ, fu2)
    hold = part_holdout(res_dir, Path(a.noaa20_2026), a.noaa20_months, B)
    old = fu2["holdout2026"]["noaa20"]["agreement"]
    for k in ("a_both", "b_ref_only", "c_other_only", "d_neither"):
        if hold[k] != old[k]:
            raise SystemExit(f"CHECK FAILED: hold-out count {k} differs from follow-up 2")

    R = {"meta": {"exploratory": True, "post_hoc": True,
                  "reason": "kappa replaced by disagreement components (Pontius and Millones, 2011; Foody, 2020) "
                            "after the follow-up-2 results were known",
                  "overlap": list(OVERLAP), "n_months": len(months), "bootstrap": B,
                  "seed_occurrence": SEED_FU2, "seed_holdout": SEED_HOLDOUT,
                  "definitions": {"D": "(b + c) / n", "Q": "|b - c| / n", "A": "2 min(b, c) / n",
                                  "A_share": "A / D", "D_fire": "(b + c) / (a + b + c)"},
                  "script_sha256": sha(__file__), "followup2_json_sha256": sha(res_dir / "followup2" / "intersensor.json"),
                  "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                  "started_utc": t0.isoformat(timespec="seconds"),
                  "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
         "replication_check": check, "occurrence": occ, "holdout2026_noaa20": hold}
    (out / "disagreement.json").write_text(json.dumps(R, indent=1))
    print(f"replication check: {check}")
    print(f"written {out / 'disagreement.json'}")


if __name__ == "__main__":
    main()
