"""Rebuild the tables of the Computers & Geosciences paper from the stored result files.

    python tools/paper_tables.py --results results --out paper_tables

Reads only files under results/ (the registered run and the three follow-ups) and writes Markdown tables:
  table2_substitution.md     Table 2  (one-month AUC-PR by predictor set, spatial block CV)
  table3_decomposition.md    Table 3  (agreement of regime classes when one factor changes at a time)
  table4_occurrence.md       Table 4  (disagreement between S-NPP and NOAA-20 on monthly fire occurrence)
  table5_holdout.md          Table 5  (2026 hold-out at L = 1)
  tableS12_kappa.md          Table S12 (Cohen's kappa, superseded)
  tableS13_strata.md         Table S13 (disagreement components by stratum)
No number is typed by hand: every value is read from the JSON files and formatted here.
"""
import argparse
import json
from pathlib import Path


def f3(v):
    return f"{v:.3f}".replace("-", "−")


def ci3(s, k, lo="_ci_low", hi="_ci_high"):
    return f"{f3(s[k])} ({f3(s[k + lo])}–{f3(s[k + hi])})"


def pct(v, d):
    return f"{100 * v:.{d}f}"


def ci_pct(s, k, d):
    return f"{pct(s[k], d)} ({pct(s[k + '_ci_low'], d)}–{pct(s[k + '_ci_high'], d)})"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_tables")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    res = json.load(open(R / "results.json"))
    fc = json.load(open(R / "forecast" / "results_forecast.json"))
    s1 = json.load(open(R / "followup" / "s1" / "s1.json"))
    f2 = json.load(open(R / "followup2" / "intersensor.json"))
    f3j = json.load(open(R / "followup3" / "disagreement.json"))

    # ---- Table 2: substitution test
    fs = res["featureset"]
    rows = []
    for key, name in (("fire_history", "F"), ("hydro", "F + H"), ("emb", "F + P + E"), ("full", "F + H + P + E")):
        rows.append([name, f3(fs["all"][key]["auc_pr"]), f3(fs["peat"][key]["auc_pr"])])
    rows.append(["Δ(H | E) = (F + H + P + E) − (F + P + E)", f3(fs["all"]["delta_full_emb"]),
                 f3(fs["peat"]["delta_full_emb"])])
    (O / "table2_substitution.md").write_text(table(["Predictor set", "AUC-PR, all cells", "AUC-PR, peat fraction ≥ 0.5"], rows))

    # ---- Table 3: factor-wise decomposition
    cl, g, k = res["clustering"], f2["regime"]["selected"], f2["regime"]["alt_kmeans4"]
    ncells = f"{json.load(open(R / 'regimes' / 'selection.json'))['n_cells']:,}"

    def ari(d):
        return f"{f3(d['ari'])} ({f3(d['ci_low'])}–{f3(d['ci_high'])})"

    def rep(d):
        return f"{f3(d['ari_mean'])} ± {f3(d['ari_sd'])}"

    rows = [
        ["1. Resampling, 2012–2025", "80 % subsamples, refitted", ncells,
         f"{f3(cl['stab_ari'])} ± {f3(cl['stab_sd'])}", f"{f3(s1['stab_ari'])} ± {f3(s1['stab_sd'])}"],
        ["2. Algorithm, 2012–2025", "Best other algorithm at the same k", ncells, f3(cl["agreement_ari"]), "—"],
        ["Registered comparison", "Sensor and record (S-NPP 2012–2025 vs NOAA-20 2018–2025)",
         f"{g['ref_vs_noaa20_2018_2025']['n_cells']:,}", ari(g["ref_vs_noaa20_2018_2025"]), ari(k["ref_vs_noaa20_2018_2025"])],
        ["3. Record", "S-NPP 2012–2025 vs S-NPP 2018–2025, refitted", f"{g['ref_vs_snpp_2018_2025']['n_cells']:,}",
         ari(g["ref_vs_snpp_2018_2025"]), ari(k["ref_vs_snpp_2018_2025"])],
        ["4. Sensor, refitted", "S-NPP vs NOAA-20, both 2018–2025", f"{g['snpp_vs_noaa20_2018_2025']['n_cells']:,}",
         ari(g["snpp_vs_noaa20_2018_2025"]), ari(k["snpp_vs_noaa20_2018_2025"])],
        ["5. Sensor, model held fixed", "S-NPP model applied to NOAA-20 fingerprints",
         f"{g['snpp_model_applied_to_noaa20']['n_cells']:,}", ari(g["snpp_model_applied_to_noaa20"]),
         ari(k["snpp_model_applied_to_noaa20"])],
        ["5b. Sensor, model held fixed (reverse)", "NOAA-20 model applied to S-NPP fingerprints",
         f"{g['noaa20_model_applied_to_snpp']['n_cells']:,}", ari(g["noaa20_model_applied_to_snpp"]),
         ari(k["noaa20_model_applied_to_snpp"])],
        ["1b. Resampling, 2018–2025", "S-NPP / NOAA-20 subsamples, refitted",
         f"{g['n_cells_snpp_2018_2025']:,} / {g['n_cells_noaa20_2018_2025']:,}",
         f"{rep(g['repeatability_snpp_2018_2025'])} / {rep(g['repeatability_noaa20_2018_2025'])}",
         f"{rep(k['repeatability_snpp_2018_2025'])} / {rep(k['repeatability_noaa20_2018_2025'])}"],
    ]
    (O / "table3_decomposition.md").write_text(table(["Condition", "What changes", "Cells", "GMM, k = 11", "k-means, k = 4"], rows))

    # ---- Table 4: occurrence disagreement
    o3, o2 = f3j["occurrence"], f2["occurrence"]
    cols = [("forecast_domain", "all"), ("forecast_domain", "peat"), ("forecast_domain", "non_peat"), ("union_domain", "all")]
    spec = [("Total disagreement D, % of cell-months", "D", 2), ("Quantity disagreement Q, % of cell-months", "Q", 3),
            ("Allocation disagreement A, % of cell-months", "A", 2), ("Share of D due to allocation, A / D (%)", "A_share", 1),
            ("Cell-months with fire in one record only, % of those with fire in either", "D_fire", 1)]
    rows = [[name] + [ci_pct(o3[d][s], key, dd) for d, s in cols] for name, key, dd in spec]
    rows.append(["Positive agreement 2a / (2a + b + c)"] + [ci3(o2[d][s], "ppos") for d, s in cols])
    rows.append(["Ratio of NOAA-20 to S-NPP fire cell-months"] + [ci3(o2[d][s], "ratio") for d, s in cols])
    (O / "table4_occurrence.md").write_text(table(["Statistic", "Forecast domain, all cells", "Forecast domain, peat cells",
                                                   "Forecast domain, non-peat cells", "Union domain, all cells"], rows))

    # ---- Table 5: 2026 hold-out at L = 1
    h, alt = fc["holdout2026"], f2["holdout2026"]["noaa20"]["models"]
    names = {"clim": "Climatology", "fwi": "FWI", "global": "Global", "global_regime": "Global + regime",
             "moe": "Mixture of experts"}
    rows = []
    for m, nm in names.items():
        v = h[m]["L1"]
        rows.append([nm, ci3(v, "auc_pr"), f3(v["auc_roc"]), f3(v["brier"]), f3(v["bss"]),
                     f3(alt[m]["alt_target"]["auc_pr"])])
    (O / "table5_holdout.md").write_text(table(["Model", "AUC-PR (95 % interval)", "ROC-AUC", "Brier score", "BSS",
                                                "AUC-PR, NOAA-20 target"], rows))

    # ---- Table S12 (kappa, superseded) and Table S13 (disagreement by stratum)
    rows12, rows13 = [], []
    for dom, nm in (("forecast_domain", "Forecast"), ("union_domain", "Union")):
        r3, r2 = o3[dom], o2[dom]
        strata = [("All cells", "all", r3["n_cells"])] + [(lab, key, r3[key]["n_cells"]) for key, lab in
                                                          (("peat", "Peat fraction ≥ 0.5"), ("non_peat", "Peat fraction < 0.5"))]
        for lab, key, nc in strata:
            rows12.append([nm, lab, ci3(r2[key], "kappa")])
            s = r3[key]
            rows13.append([nm, lab, f"{nc:,}", f"{s['a_both']:,}", f"{s['b_ref_only']:,}", f"{s['c_other_only']:,}",
                           pct(s["D"], 2), pct(s["Q"], 3), pct(s["A"], 2), pct(s["A_share"], 1), ci3(r2[key], "ppos")])
        for y in r3["by_year"]:
            s, t = r3["by_year"][y], r2["by_year"][y]
            rows12.append([nm, y, ci3(t, "kappa")])
            rows13.append([nm, y, f"{r3['n_cells']:,}", f"{s['a_both']:,}", f"{s['b_ref_only']:,}", f"{s['c_other_only']:,}",
                           pct(s["D"], 2), pct(s["Q"], 3), pct(s["A"], 2), pct(s["A_share"], 1), ci3(t, "ppos")])
        for gk in r3.get("by_regime", {}):
            s, t = r3["by_regime"][gk], r2["by_regime"][gk]
            rows12.append([nm, f"Regime {gk}", ci3(t, "kappa")])
            rows13.append([nm, f"Regime {gk}", f"{s['n_cells']:,}", f"{s['a_both']:,}", f"{s['b_ref_only']:,}",
                           f"{s['c_other_only']:,}", pct(s["D"], 2), pct(s["Q"], 3), pct(s["A"], 2), pct(s["A_share"], 1),
                           ci3(t, "ppos")])
    rows12.append(["2026 hold-out", "S-NPP NRT target vs NOAA-20", ci3(f2["holdout2026"]["noaa20"]["agreement"], "kappa")])
    (O / "tableS12_kappa.md").write_text(table(["Domain", "Stratum", "Cohen's kappa (95 % interval)"], rows12))
    (O / "tableS13_strata.md").write_text(table(["Domain", "Stratum", "Cells", "Both", "S-NPP only", "NOAA-20 only", "D (%)",
                                                 "Q (%)", "A (%)", "A / D (%)", "Positive agreement (95 % interval)"], rows13))
    print("tables ->", O)


if __name__ == "__main__":
    main()
