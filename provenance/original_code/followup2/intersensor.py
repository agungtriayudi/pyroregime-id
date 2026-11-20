"""Tindak lanjut 2 (EKSPLORATIF): kesepakatan antar-sensor S-NPP vs NOAA-20 di sepanjang rantai pengukuran.

    python followup2/intersensor.py --config config.json
    python followup2/intersensor.py --config config.json \
        --snpp-sp-2026 data/cube/followup2/cell_month_snpp_sp_2026.csv.gz --snpp-sp-months 2026-08 2026-09 \
        --noaa20-2026 data/cube/followup2/cell_month_noaa20_2026.csv.gz --noaa20-months 2026-08 2026-09 --noaa20-level SP

Desain dibekukan di registration/ADDENDUM_followup2.md SEBELUM skrip ini dijalankan pada data nyata.
Tidak mengubah apa pun di results/ selain results/followup2/. Memakai ulang kode terdaftar tanpa mengubahnya
(regime_features.fingerprint, cluster_regimes.Prep/fit_one/stability, forecast.block_of/block_boot).

Bagian:
  A  cek konsistensi dengan run terdaftar (ARI sensor terdaftar harus terulang persis)
  B  kesepakatan KEJADIAN api per sel-bulan (besaran nominal) pada periode tumpang tindih
  C  kesepakatan INDIKATOR sel-bulan (jumlah deteksi, hari aktif, persistensi, fraksi malam, FRP)
  D  kesepakatan FITUR sidik jari dan KELAS REZIM; pemisahan efek periode dari efek sensor
  E  hold-out 2026 dengan definisi besaran ukur alternatif (S-NPP standard processing; NOAA-20) -- opsional
Keluaran: results/followup2/intersensor.json, tables/*.md, figures/*.png
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cluster_regimes import Prep, fit_one, stability  # noqa: E402
from forecast import block_boot, block_of  # noqa: E402
from regime_features import attach_static, fingerprint, load_oni  # noqa: E402

OVERLAP = ("2018-04", "2025-12")            # NOAA-20 standard processing tersedia mulai April 2018
FP_YEARS = (2018, 2025)                     # sama dengan run terdaftar untuk sidik jari NOAA-20
INDICATORS = {"n_det": "log", "active_days": "raw", "persist_max": "raw", "night_frac": "raw",
              "frp_med": "log1p", "frp_sum": "log1p"}
FEATURES = ["log_det_per_year", "recurrence", "persist_mean", "persist_max", "night_frac", "log_frp_med",
            "season_sin", "season_cos", "season_conc", "enso_rho"]
BAND_HIGH, BAND_LOW = 0.80, 0.60            # aturan baca pra-tetapkan (ADDENDUM_followup2.md)
ALT_K = ("kmeans", 4)                       # partisi Suplemen S1


# ---------------------------------------------------------------- statistik dasar
def band(lo, hi):
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return "undetermined"
    if lo >= BAND_HIGH:
        return "reproducible"
    if hi < BAND_LOW:
        return "not reproducible"
    return "partly reproducible"


def agree_stats(a, b, c, d):
    """Kesepakatan dua pengukuran biner. a = keduanya 1, b = hanya S-NPP, c = hanya pembanding, d = keduanya 0."""
    a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))
    n = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (a + d) / n
        pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
        return {"po": po, "kappa": (po - pe) / (1 - pe), "ppos": 2 * a / (2 * a + b + c),
                "pneg": 2 * d / (2 * d + b + c), "ratio": (a + c) / (a + b),
                "frac_ref_also_other": a / (a + b), "frac_other_also_ref": a / (a + c)}


def boot_sums(M, B, rng, chunk=100):
    """Jumlah kolom M (n_blok x p) atas blok yang disampel ulang dengan pengembalian -> (B x p)."""
    nb = len(M)
    out = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, nb, (min(chunk, B - s), nb))
        out.append(M[idx].sum(axis=1))
    return np.vstack(out)


def q(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if len(v) else (float("nan"), float("nan"))


def summarize_counts(M, B, rng):
    tot = M.sum(axis=0)
    pt = {k: float(v) for k, v in agree_stats(*tot).items()}
    bs = agree_stats(*boot_sums(M, B, rng).T)
    res = {"a_both": int(tot[0]), "b_ref_only": int(tot[1]), "c_other_only": int(tot[2]), "d_neither": int(tot[3]),
           "n": int(tot.sum()), "prev_ref": float((tot[0] + tot[1]) / tot.sum()),
           "prev_other": float((tot[0] + tot[2]) / tot.sum()), **pt}
    for k in ("kappa", "ppos", "pneg", "ratio", "po"):
        res[f"{k}_ci_low"], res[f"{k}_ci_high"] = q(bs[k])
    res["band_kappa"] = band(res["kappa_ci_low"], res["kappa_ci_high"])
    res["band_ppos"] = band(res["ppos_ci_low"], res["ppos_ci_high"])
    return res


def ccc_from_sums(S):
    """S kolom: n, sx, sy, sxx, syy, sxy -> (ccc, bias = rerata y - rerata x)."""
    n, sx, sy, sxx, syy, sxy = (S[..., i] for i in range(6))
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = sx / n, sy / n
        vx, vy, cov = sxx / n - mx ** 2, syy / n - my ** 2, sxy / n - mx * my
        return 2 * cov / (vx + vy + (mx - my) ** 2), my - mx


def paired_stats(x, y, block, B, rng):
    """Kesepakatan dua pengukuran kontinu berpasangan (x = S-NPP, y = NOAA-20) dengan bootstrap blok."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, block = x[ok], y[ok], np.asarray(block)[ok]
    if len(x) < 30:
        return {"n": int(len(x))}
    d = y - x
    g = pd.DataFrame({"block": block, "n": 1.0, "sx": x, "sy": y, "sxx": x * x, "syy": y * y, "sxy": x * y}
                     ).groupby("block").sum().to_numpy()
    ccc, bias = ccc_from_sums(g.sum(axis=0))
    bc, bb = ccc_from_sums(boot_sums(g, B, rng))
    res = {"n": int(len(x)), "mean_snpp": float(x.mean()), "mean_noaa20": float(y.mean()),
           "bias": float(bias), "sd_diff": float(d.std(ddof=1)),
           "loa_low": float(np.quantile(d, .025)), "loa_high": float(np.quantile(d, .975)),
           "pearson_r": float(np.corrcoef(x, y)[0, 1]), "spearman_rho": float(spearmanr(x, y).statistic),
           "ccc": float(ccc)}
    res["bias_ci_low"], res["bias_ci_high"] = q(bb)
    res["ccc_ci_low"], res["ccc_ci_high"] = q(bc)
    res["band"] = band(res["ccc_ci_low"], res["ccc_ci_high"])
    return res


def ari_ci(l1, l2, block, B, rng):
    l1, l2, block = np.asarray(l1), np.asarray(l2), np.asarray(block)
    idx = pd.Series(np.arange(len(block))).groupby(block).indices
    keys = list(idx)
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        vals.append(adjusted_rand_score(l1[take], l2[take]))
    lo, hi = q(vals)
    return {"ari": float(adjusted_rand_score(l1, l2)), "ci_low": lo, "ci_high": hi, "n_cells": int(len(l1)),
            "band": band(lo, hi)}


# ---------------------------------------------------------------- B: kejadian
def occ_table(S, N, cells):
    s = S.loc[S["cell"].isin(cells) & (S["n_det"] > 0), ["cell", "month"]].assign(s=1)
    n = N.loc[N["cell"].isin(cells) & (N["n_det"] > 0), ["cell", "month"]].assign(n=1)
    o = s.merge(n, on=["cell", "month"], how="outer").fillna({"s": 0, "n": 0})
    o["year"] = o["month"].str[:4].astype(int)
    return o


def block_counts(o, info, months):
    """info: DataFrame (cell, block) untuk subset sel; months: daftar bulan. -> matriks blok x (a, b, c, d)."""
    sub = o[o["cell"].isin(info["cell"]) & o["month"].isin(months)].merge(info[["cell", "block"]], on="cell")
    g = pd.DataFrame({"block": sub["block"], "a": (sub["s"] == 1) & (sub["n"] == 1),
                      "b": (sub["s"] == 1) & (sub["n"] == 0), "c": (sub["s"] == 0) & (sub["n"] == 1)}
                     ).groupby("block").sum()
    tot = info.groupby("block").size() * len(months)
    g = g.reindex(tot.index, fill_value=0).astype(float)
    g["d"] = tot - g.sum(axis=1)
    return g[["a", "b", "c", "d"]].to_numpy()


def part_occurrence(S, N, domains, months, B, rng):
    res = {}
    for name, info in domains.items():
        o = occ_table(S, N, info["cell"])
        r = {"n_cells": int(len(info)), "n_months": len(months),
             "all": summarize_counts(block_counts(o, info, months), B, rng)}
        for lab, sub in (("peat", info[info["peat_frac"] >= 0.5]), ("non_peat", info[info["peat_frac"] < 0.5])):
            if len(sub) >= 50:
                r[lab] = dict(summarize_counts(block_counts(o, sub, months), B, rng), n_cells=int(len(sub)))
        r["by_year"] = {}
        for y in sorted({m[:4] for m in months}):
            ms = [m for m in months if m.startswith(y)]
            r["by_year"][y] = dict(summarize_counts(block_counts(o, info, ms), B, rng), n_months=len(ms))
        if "regime" in info:
            r["by_regime"] = {}
            for k, sub in info.groupby("regime"):
                if len(sub) >= 50:
                    r["by_regime"][str(int(k))] = dict(summarize_counts(block_counts(o, sub, months), B, rng),
                                                       n_cells=int(len(sub)))
        res[name] = r
    return res


# ---------------------------------------------------------------- C: indikator sel-bulan
def transform(v, how):
    v = np.asarray(v, float)
    return np.log(v) if how == "log" else np.log1p(v) if how == "log1p" else v


def part_indicators(S, N, domains, B, rng):
    res, diffs = {}, {}
    for name, info in domains.items():
        j = S[S["cell"].isin(info["cell"]) & (S["n_det"] > 0)].merge(
            N[N["cell"].isin(info["cell"]) & (N["n_det"] > 0)], on=["cell", "month"], suffixes=("_s", "_n"))
        j = j.merge(info[["cell", "block"]], on="cell")
        r = {"n_pairs": int(len(j))}
        for v, how in INDICATORS.items():
            if f"{v}_s" not in j:
                continue
            x, y = transform(j[f"{v}_s"], how), transform(j[f"{v}_n"], how)
            r[v] = dict(paired_stats(x, y, j["block"], B, rng), scale=how)
            if how != "raw" and "bias" in r[v]:
                r[v]["geometric_mean_ratio"] = float(np.exp(r[v]["bias"]))
            if name == "forecast_domain":
                diffs[v] = (y - x)
        res[name] = r
    return res, diffs


# ---------------------------------------------------------------- D: sidik jari dan kelas rezim
def fit_partition(fp, method, k):
    prep = Prep(emb_weight=1.0).fit(fp)
    X = prep.transform(fp)
    lab, pred = fit_one(method, X, k, seed=0)
    return {"fp": fp, "prep": prep, "X": X, "lab": lab, "pred": pred,
            "ser": pd.Series(lab, index=fp["cell"].to_numpy())}


def compare(sa, sb, B, rng):
    cells = sa.index.intersection(sb.index)
    return ari_ci(sa.loc[cells].to_numpy(), sb.loc[cells].to_numpy(), block_of(cells.to_numpy()), B, rng)


def transfer(src, dst, B, rng):
    """Model yang dilatih pada sidik jari `src` diterapkan ke sidik jari `dst` pada sel yang sama."""
    if src["pred"] is None:
        return None
    cells = src["ser"].index.intersection(dst["ser"].index)
    fd = dst["fp"].set_index("cell").loc[cells].reset_index()
    lab_dst = src["pred"](src["prep"].transform(fd))
    return ari_ci(src["ser"].loc[cells].to_numpy(), lab_dst, block_of(cells.to_numpy()), B, rng)


def part_regime(cfg, res_dir, fpS, fpN, B, rng):
    sel = json.load(open(res_dir / "regimes" / "selection.json"))
    fp_ref = pd.read_csv(cfg["fingerprint"])
    out = {}
    for tag, method, k in (("selected", sel["method"], int(sel["k"])), ("alt_kmeans4", *ALT_K)):
        if tag == "selected":
            ref = pd.read_csv(res_dir / "regimes" / "labels_selected.csv").set_index("cell")["regime"]
        else:
            ref = fit_partition(fp_ref, method, k)["ser"]
        PS, PN = fit_partition(fpS, method, k), fit_partition(fpN, method, k)
        r = {"method": method, "k": int(k), "n_cells_snpp_2018_2025": int(len(fpS)), "n_cells_noaa20_2018_2025": int(len(fpN)),
             "ref_vs_noaa20_2018_2025": compare(ref, PN["ser"], B, rng),
             "ref_vs_snpp_2018_2025": compare(ref, PS["ser"], B, rng),
             "snpp_vs_noaa20_2018_2025": compare(PS["ser"], PN["ser"], B, rng),
             "snpp_model_applied_to_noaa20": transfer(PS, PN, B, rng),
             "noaa20_model_applied_to_snpp": transfer(PN, PS, B, rng)}
        nb = int(cfg.get("cluster_boot", 50))
        for nm, P in (("snpp", PS), ("noaa20", PN)):
            m, s = stability(method, P["X"], k, P["lab"], nb, np.random.default_rng(0))
            r[f"repeatability_{nm}_2018_2025"] = {"ari_mean": m, "ari_sd": s, "subsamples": nb}
        if tag == "selected":
            p = res_dir / "regimes_noaa20" / f"labels_{method}.csv"
            if p.exists():
                reg = pd.read_csv(p).set_index("cell")["regime"]
                same = reg.index.symmetric_difference(PN["ser"].index)
                r["check_noaa20_refit_matches_registered"] = {
                    "cells_differ": int(len(same)),
                    "ari": float(adjusted_rand_score(reg.loc[reg.index.intersection(PN["ser"].index)],
                                                     PN["ser"].loc[reg.index.intersection(PN["ser"].index)]))}
        out[tag] = r
    return out


def part_features(fpS, fpN, B, rng):
    j = fpS.merge(fpN, on="cell", suffixes=("_s", "_n"))
    blk = block_of(j["cell"].to_numpy())
    res = {"n_cells": int(len(j))}
    for f in FEATURES:
        if f"{f}_s" in j:
            res[f] = paired_stats(j[f"{f}_s"], j[f"{f}_n"], blk, B, rng)
    return res


# ---------------------------------------------------------------- E: hold-out 2026 dengan target alternatif
def clf_metrics(y, p, clim):
    y = np.asarray(y, float)
    if y.sum() == 0 or y.sum() == len(y):
        return {}
    bs, bc = brier_score_loss(y, p), brier_score_loss(y, clim)
    return {"auc_pr": float(average_precision_score(y, p)), "auc_roc": float(roc_auc_score(y, p)),
            "brier": float(bs), "bss": float(1 - bs / bc), "base_rate": float(y.mean()), "n": int(len(y))}


def part_holdout(res_dir, cube_path, months, level, B, rng):
    pred = pd.read_csv(res_dir / "forecast" / "preds_holdout2026.csv.gz")
    all_months = sorted(pred["target_month"].unique())
    pred = pred[pred["target_month"].isin(months)].copy()
    if not len(pred):
        return {"skipped": f"tidak ada bulan target hold-out di {months}"}
    alt = pd.read_csv(cube_path)
    alt = alt[(alt["n_det"] > 0) & alt["month"].isin(months)]
    key = set(zip(alt["cell"].to_numpy(), alt["month"].to_numpy()))
    pred["y_alt"] = [float((c, m) in key) for c, m in zip(pred["cell"].to_numpy(), pred["target_month"].to_numpy())]
    pred["block"] = block_of(pred["cell"].to_numpy())
    y, ya = pred["y"].to_numpy() == 1, pred["y_alt"].to_numpy() == 1
    M = pd.DataFrame({"block": pred["block"], "a": y & ya, "b": y & ~ya, "c": ~y & ya, "d": ~y & ~ya}
                     ).groupby("block").sum().to_numpy(float)
    out = {"months": list(months), "holdout_months_all": all_months, "processing_level": level,
           "n": int(len(pred)), "agreement": summarize_counts(M, B, rng), "models": {}}
    models = [c[2:] for c in pred if c.startswith("p_")]
    for m in models:
        r = {}
        for tgt in ("y", "y_alt"):
            mm = clf_metrics(pred[tgt], pred[f"p_{m}"], pred["clim"])
            bs = block_boot(pred, lambda d, t=tgt: average_precision_score(d[t], d[f"p_{m}"]), B)
            if len(bs) and mm:
                mm["auc_pr_ci_low"], mm["auc_pr_ci_high"] = q(bs)
            r["registered_target" if tgt == "y" else "alt_target"] = mm
        d = block_boot(pred, lambda s: average_precision_score(s["y_alt"], s[f"p_{m}"])
                       - average_precision_score(s["y"], s[f"p_{m}"]), B, seed=3)
        lo, hi = q(d)
        r["delta_alt_minus_registered"] = {
            "mean": float(average_precision_score(pred["y_alt"], pred[f"p_{m}"])
                          - average_precision_score(pred["y"], pred[f"p_{m}"])), "ci_low": lo, "ci_high": hi}
        out["models"][m] = r
    if "p_moe" in pred and "p_global" in pred:
        for tgt in ("y", "y_alt"):
            d = block_boot(pred, lambda s, t=tgt: average_precision_score(s[t], s["p_moe"])
                           - average_precision_score(s[t], s["p_global"]), B, seed=1)
            lo, hi = q(d)
            out["delta_moe_global_" + ("registered_target" if tgt == "y" else "alt_target")] = {
                "mean": float(average_precision_score(pred[tgt], pred["p_moe"])
                              - average_precision_score(pred[tgt], pred["p_global"])),
                "ci_low": lo, "ci_high": hi, "p": float((d <= 0).mean())}
        a = out["delta_moe_global_alt_target"]
        out["h4_reading"] = ("H4 supported under the alternative target (differs from the registered result)"
                             if a["ci_low"] > 0 else
                             "H4 not supported under the alternative target (same as the registered result)")
    return out


# ---------------------------------------------------------------- tabel dan gambar
def f3(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def ci(r, k):
    return f"{f3(r.get(k))} ({f3(r.get(k + '_ci_low'))} to {f3(r.get(k + '_ci_high'))})"


def write_tables(R, out):
    t = out / "tables"; t.mkdir(parents=True, exist_ok=True)
    L = ["| Domain | Stratum | Cells | Cell-months | Both | S-NPP only | NOAA-20 only | Kappa (95 % interval) | "
         "Positive agreement (95 % interval) | NOAA-20 / S-NPP ratio (95 % interval) | Reading (kappa) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for dom, r in R["occurrence"].items():
        rows = [("all", r["all"], r["n_cells"])] + [(k, r[k], r[k]["n_cells"]) for k in ("peat", "non_peat") if k in r]
        rows += [(f"year {y}", v, r["n_cells"]) for y, v in r["by_year"].items()]
        rows += [(f"regime {k}", v, v["n_cells"]) for k, v in r.get("by_regime", {}).items()]
        for lab, v, nc in rows:
            L.append(f"| {dom} | {lab} | {nc:,} | {v['n']:,} | {v['a_both']:,} | {v['b_ref_only']:,} | {v['c_other_only']:,} | "
                     f"{ci(v, 'kappa')} | {ci(v, 'ppos')} | {ci(v, 'ratio')} | {v['band_kappa']} |")
    (t / "occurrence.md").write_text("\n".join(L) + "\n")

    L = ["| Domain | Indicator | Scale | Pairs | Mean S-NPP | Mean NOAA-20 | Bias NOAA-20 − S-NPP (95 % interval) | SD of differences | "
         "2.5th–97.5th percentile of differences | Pearson r | Spearman ρ | CCC (95 % interval) | Reading |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for dom, r in R["indicators"].items():
        for v in INDICATORS:
            x = r.get(v)
            if not x or "ccc" not in x:
                continue
            L.append(f"| {dom} | {v} | {x['scale']} | {x['n']:,} | {f3(x['mean_snpp'])} | {f3(x['mean_noaa20'])} | {ci(x, 'bias')} | "
                     f"{f3(x['sd_diff'])} | {f3(x['loa_low'])} to {f3(x['loa_high'])} | {f3(x['pearson_r'])} | "
                     f"{f3(x['spearman_rho'])} | {ci(x, 'ccc')} | {x['band']} |")
    (t / "indicators.md").write_text("\n".join(L) + "\n")

    L = ["| Feature | Cells | Mean S-NPP | Mean NOAA-20 | Bias (95 % interval) | SD of differences | Pearson r | CCC (95 % interval) | Reading |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in FEATURES:
        x = R["features"].get(f)
        if x and "ccc" in x:
            L.append(f"| {f} | {x['n']:,} | {f3(x['mean_snpp'])} | {f3(x['mean_noaa20'])} | {ci(x, 'bias')} | {f3(x['sd_diff'])} | "
                     f"{f3(x['pearson_r'])} | {ci(x, 'ccc')} | {x['band']} |")
    (t / "features.md").write_text("\n".join(L) + "\n")

    names = {"ref_vs_noaa20_2018_2025": "Reference (S-NPP, 2012–2025) vs NOAA-20 (2018–2025): sensor and period changed (registered comparison)",
             "ref_vs_snpp_2018_2025": "Reference (S-NPP, 2012–2025) vs S-NPP (2018–2025): period changed, same sensor",
             "snpp_vs_noaa20_2018_2025": "S-NPP vs NOAA-20, both 2018–2025, fitted separately: sensor changed, same period",
             "snpp_model_applied_to_noaa20": "S-NPP model (2018–2025) applied to NOAA-20 fingerprints: sensor changed, same model",
             "noaa20_model_applied_to_snpp": "NOAA-20 model (2018–2025) applied to S-NPP fingerprints: sensor changed, same model"}
    L = ["| Partition | Comparison | Cells | ARI (95 % interval) | Reading |", "| --- | --- | --- | --- | --- |"]
    for tag, r in R["regime"].items():
        p = f"{r['method']}, k = {r['k']}"
        for k, nm in names.items():
            x = r.get(k)
            if x:
                L.append(f"| {p} | {nm} | {x['n_cells']:,} | {f3(x['ari'])} ({f3(x['ci_low'])} to {f3(x['ci_high'])}) | {x['band']} |")
        for s in ("snpp", "noaa20"):
            x = r[f"repeatability_{s}_2018_2025"]
            L.append(f"| {p} | Repeatability under resampling, {'S-NPP' if s == 'snpp' else 'NOAA-20'} 2018–2025 ({x['subsamples']} subsamples of 80 %) | "
                     f"{r[f'n_cells_{s}_2018_2025']:,} | {f3(x['ari_mean'])} ± {f3(x['ari_sd'])} | — |")
    (t / "regime_decomposition.md").write_text("\n".join(L) + "\n")

    if isinstance(R.get("holdout2026"), dict) and any("models" in v for v in R["holdout2026"].values() if isinstance(v, dict)):
        L = ["| Alternative target | Months | Cell-months | Kappa with registered target (95 % interval) | Positive agreement | Model | "
             "AUC-PR, registered target (95 % interval) | AUC-PR, alternative target (95 % interval) | Difference (95 % interval) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        D = ["| Alternative target | ΔAUC-PR MoE − global, registered target (95 % interval; p) | "
             "ΔAUC-PR MoE − global, alternative target (95 % interval; p) | Reading for H4 |", "| --- | --- | --- | --- |"]
        for nm, r in R["holdout2026"].items():
            if not isinstance(r, dict) or "models" not in r:
                continue
            for m, x in r["models"].items():
                a, b, d = x["registered_target"], x["alt_target"], x["delta_alt_minus_registered"]
                L.append(f"| {nm} ({r['processing_level']}) | {', '.join(r['months'])} | {r['n']:,} | {ci(r['agreement'], 'kappa')} | "
                         f"{ci(r['agreement'], 'ppos')} | {m} | {ci(a, 'auc_pr') if a else '—'} | {ci(b, 'auc_pr') if b else '—'} | "
                         f"{f3(d['mean'])} ({f3(d['ci_low'])} to {f3(d['ci_high'])}) |")
            if "delta_moe_global_alt_target" in r:
                a, b = r["delta_moe_global_registered_target"], r["delta_moe_global_alt_target"]
                D.append(f"| {nm} ({r['processing_level']}) | {f3(a['mean'])} ({f3(a['ci_low'])} to {f3(a['ci_high'])}; p = {f3(a['p'])}) | "
                         f"{f3(b['mean'])} ({f3(b['ci_low'])} to {f3(b['ci_high'])}; p = {f3(b['p'])}) | {r['h4_reading']} |")
        (t / "holdout_alt_targets.md").write_text("\n".join(L) + "\n\n" + "\n".join(D) + "\n")


def write_figures(R, diffs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = out / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e3e2dc"
    C1, C2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})

    # 1) kesepakatan kejadian per tahun
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, key, title in zip(axes, ("kappa", "ppos"), ("Cohen's kappa", "Positive agreement")):
        for (dom, r), col, mk in zip(R["occurrence"].items(), (C1, C2), ("o", "s")):
            ys = sorted(r["by_year"])
            v = np.array([r["by_year"][y][key] for y in ys])
            lo = np.array([r["by_year"][y][key + "_ci_low"] for y in ys])
            hi = np.array([r["by_year"][y][key + "_ci_high"] for y in ys])
            ax.errorbar([int(y) for y in ys], v, yerr=[v - lo, hi - v], color=col, marker=mk, ms=5, lw=2, capsize=2,
                        label=dom.replace("_", " "))
        ax.set_title(title, loc="left", color=INK, fontsize=10); ax.set_ylim(0, 1); ax.set_xlabel("Year")
    axes[0].set_ylabel("Agreement, S-NPP vs NOAA-20"); axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(fdir / "fig_f2_occurrence_by_year.png", dpi=200); plt.close(fig)

    # 2) sebaran selisih indikator (NOAA-20 - S-NPP), domain prakiraan
    labs = {"n_det": "Detections (log ratio)", "active_days": "Active days (difference)",
            "persist_max": "Persistence, days (difference)", "night_frac": "Night fraction (difference)",
            "frp_med": "Median FRP (log1p difference)", "frp_sum": "Summed FRP (log1p difference)"}
    keys = [k for k in INDICATORS if k in diffs and len(diffs[k])]
    if keys:
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), squeeze=False)
        for ax, k in zip(axes.ravel(), keys):
            d = np.asarray(diffs[k]); d = d[~np.isnan(d)]
            lo, hi = np.quantile(d, [.005, .995])
            ax.hist(np.clip(d, lo, hi), bins=41, color=C1, edgecolor="white", linewidth=0.4)
            x = R["indicators"]["forecast_domain"][k]
            ax.axvline(x["bias"], color=INK, lw=1.5)
            for v in (x["loa_low"], x["loa_high"]):
                ax.axvline(v, color=INK, lw=1, ls="--")
            ax.set_title(labs[k], loc="left", color=INK, fontsize=9); ax.set_yticks([])
        for ax in axes.ravel()[len(keys):]:
            ax.axis("off")
        fig.suptitle("NOAA-20 minus S-NPP per cell-month; solid line = mean difference, dashed = 2.5th and 97.5th percentiles",
                     x=0.01, ha="left", fontsize=9, color=MUTED)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(fdir / "fig_f2_indicator_differences.png", dpi=200); plt.close(fig)

    # 3) dekomposisi ARI kelas rezim
    order = [("ref_vs_noaa20_2018_2025", "Sensor + period\n(registered comparison)"),
             ("ref_vs_snpp_2018_2025", "Period only\n(same sensor)"),
             ("snpp_vs_noaa20_2018_2025", "Sensor only\n(same period, refitted)"),
             ("snpp_model_applied_to_noaa20", "Sensor only\n(same model applied)")]
    fig, axes = plt.subplots(1, len(R["regime"]), figsize=(4.6 * len(R["regime"]), 3.4), sharey=True, squeeze=False)
    for ax, (tag, r) in zip(axes.ravel(), R["regime"].items()):
        ypos = np.arange(len(order))[::-1]
        for yy, (k, _) in zip(ypos, order):
            x = r.get(k)
            if x:
                ax.errorbar(x["ari"], yy, xerr=[[x["ari"] - x["ci_low"]], [x["ci_high"] - x["ari"]]], color=C1, marker="o",
                            ms=6, lw=2, capsize=3)
                left = x["ci_high"] > 0.85
                ax.text(x["ci_low"] - 0.03 if left else x["ci_high"] + 0.03, yy + 0.22, f"{x['ari']:.2f}",
                        va="center", ha="right" if left else "left", color=INK, fontsize=8)
        rep = r["repeatability_snpp_2018_2025"]["ari_mean"]
        ax.axvline(rep, color=MUTED, lw=1, ls="--")
        ax.text(rep - 0.01 if rep > 0.5 else rep + 0.01, len(order) - 0.45, f"repeatability {rep:.2f}",
                ha="right" if rep > 0.5 else "left", va="bottom", color=MUTED, fontsize=8)
        ax.set_yticks(ypos); ax.set_yticklabels([n for _, n in order]); ax.set_xlim(0, 1.02); ax.set_ylim(-0.6, len(order) - 0.1)
        ax.set_xlabel("Adjusted Rand index"); ax.set_title(f"{r['method']}, k = {r['k']}", loc="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(fdir / "fig_f2_regime_decomposition.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- utama
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--boot", type=int, default=None, help="default: bootstrap di config (registrasi: 1000)")
    ap.add_argument("--snpp-sp-2026", default=None, help="kubus sel-bulan S-NPP standard processing untuk bulan hold-out")
    ap.add_argument("--snpp-sp-months", nargs="*", default=[], help="bulan yang tercakup PENUH, mis. 2026-08 2026-09")
    ap.add_argument("--noaa20-2026", default=None, help="kubus sel-bulan NOAA-20 untuk bulan hold-out")
    ap.add_argument("--noaa20-months", nargs="*", default=[])
    ap.add_argument("--noaa20-level", default="unknown", help="SP | NRT | mixed")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    res_dir = Path(cfg["out"]); out = res_dir / "followup2"; out.mkdir(parents=True, exist_ok=True)
    B = a.boot or int(cfg.get("bootstrap", 1000))
    rng = np.random.default_rng(20261117)
    months = pd.period_range(*OVERLAP, freq="M").astype(str).tolist()

    S_all = pd.read_csv(cfg["cell_month"])
    N_all = pd.read_csv(cfg["cell_month_noaa20"])
    S = S_all[S_all["month"].between(*OVERLAP)].copy()
    N = N_all[N_all["month"].between(*OVERLAP)].copy()
    st = pd.read_csv(cfg["static"])
    land = st.groupby("cell")[[c for c in ("in_indonesia", "peat_frac") if c in st]].mean()
    if "in_indonesia" not in land:
        land["in_indonesia"] = 1.0

    R = {"meta": {"exploratory": True, "overlap": list(OVERLAP), "n_months": len(months), "bootstrap": B,
                  "bands": {"reproducible_if_lower_bound_ge": BAND_HIGH, "not_reproducible_if_upper_bound_lt": BAND_LOW},
                  "script_sha256": sha(__file__), "synthetic": bool(cfg.get("synthetic", False)),
                  "snpp_cell_months_overlap": int(len(S)), "noaa20_cell_months_overlap": int(len(N)),
                  "snpp_first_last_month": [str(S_all["month"].min()), str(S_all["month"].max())],
                  "noaa20_first_last_month": [str(N_all["month"].min()), str(N_all["month"].max())]}}

    # ---- A: cek konsistensi dengan run terdaftar
    sel = json.load(open(res_dir / "regimes" / "selection.json"))
    lab = pd.read_csv(res_dir / "regimes" / "labels_selected.csv")
    checks = {}
    p = res_dir / "regimes_noaa20" / f"labels_{sel['method']}.csv"
    rj = res_dir / "results.json"
    if p.exists() and rj.exists():
        reg = json.load(open(rj))["clustering"]
        j = lab.merge(pd.read_csv(p), on="cell", suffixes=("", "_n20"))
        ari = float(adjusted_rand_score(j["regime"], j["regime_n20"]))
        checks["registered_sensor_ari"] = {"recomputed": ari, "registered": reg.get("sensor_ari"), "n_cells": int(len(j)),
                                           "registered_n_cells": reg.get("sensor_n_cells"),
                                           "abs_diff": abs(ari - (reg.get("sensor_ari") or np.nan))}
        if not checks["registered_sensor_ari"]["abs_diff"] < 1e-9:
            json.dump({"meta": R["meta"], "checks": checks}, open(out / "intersensor.json", "w"), indent=1, default=float)
            raise SystemExit("CEK GAGAL: ARI sensor terdaftar tidak terulang. Berhenti; laporkan ke pengguna.")
    else:
        checks["registered_sensor_ari"] = {"skipped": "results.json atau regimes_noaa20 tidak ada"}
    R["checks"] = checks

    # ---- domain
    dom1 = lab[["cell", "regime"]].join(land["peat_frac"], on="cell")
    dom1["block"] = block_of(dom1["cell"].to_numpy())
    md = int(cfg.get("min_det", 20))
    tS, tN = S.groupby("cell")["n_det"].sum(), N.groupby("cell")["n_det"].sum()
    union = pd.Index(tS[tS >= md].index).union(tN[tN >= md].index)
    dom2 = land.loc[land.index.intersection(union)]
    n_union, n_static = int(len(union)), int(len(dom2))
    dom2 = dom2[dom2["in_indonesia"] > 0.5].reset_index()[["cell", "peat_frac"]]
    dom2["block"] = block_of(dom2["cell"].to_numpy())
    R["meta"]["domains"] = {
        "forecast_domain": "sel bersidik jari pada run terdaftar (>= min_det deteksi S-NPP, 2012-2025)",
        "union_domain": "sel dengan >= min_det deteksi dari SALAH SATU sensor pada periode tumpang tindih, di dalam Indonesia",
        "union_cells_before_static": n_union, "union_cells_with_static": n_static, "union_cells_in_indonesia": int(len(dom2))}
    domains = {"forecast_domain": dom1, "union_domain": dom2}

    # ---- B, C
    R["occurrence"] = part_occurrence(S, N, domains, months, B, rng)
    R["indicators"], diffs = part_indicators(S, N, domains, B, rng)

    # ---- D
    oni = load_oni(cfg["oni"])
    fpS = attach_static(fingerprint(S, oni, *FP_YEARS, md), st)
    fpN = attach_static(fingerprint(N, oni, *FP_YEARS, md), st)
    R["features"] = part_features(fpS, fpN, B, rng)
    R["regime"] = part_regime(cfg, res_dir, fpS, fpN, B, rng)

    # ---- E
    R["holdout2026"] = {}
    for nm, path, ms, level in (("snpp_sp", a.snpp_sp_2026, a.snpp_sp_months, "SP"),
                                ("noaa20", a.noaa20_2026, a.noaa20_months, a.noaa20_level)):
        if path and ms and (res_dir / "forecast" / "preds_holdout2026.csv.gz").exists():
            R["holdout2026"][nm] = part_holdout(res_dir, path, ms, level, B, rng)
        else:
            R["holdout2026"][nm] = {"skipped": "kubus atau daftar bulan tidak diberikan, atau prediksi hold-out tidak ada"}

    json.dump(R, open(out / "intersensor.json", "w"), indent=1, default=float)
    write_tables(R, out)
    write_figures(R, diffs, out)
    o = R["occurrence"]["forecast_domain"]["all"]
    print(f"kejadian (domain prakiraan): kappa {o['kappa']:.3f} [{o['kappa_ci_low']:.3f}, {o['kappa_ci_high']:.3f}]")
    for tag, r in R["regime"].items():
        print(tag, {k: round(v["ari"], 3) for k, v in r.items() if isinstance(v, dict) and "ari" in v})
    print(f"SELESAI (eksploratif) -> {out}")


if __name__ == "__main__":
    main()
