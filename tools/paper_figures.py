"""Rebuild the figures of the Computers & Geosciences paper from the stored result files.

    python tools/paper_figures.py --results results --out paper_figures \
        [--boundaries data/raw/boundaries/idn_adm1.geojson]

Figures 1, 2, 4 and 5 of the paper and Figures S2, S5 and S8 of the supplement. Figure 3 is written by
followup2/intersensor.py (results/followup2/figures/fig_f2_regime_decomposition.png). Province boundaries are optional;
the file used in the paper is geoBoundaries IDN ADM1 (gbOpen, boundary year 2017):
https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM1/geoBoundaries-IDN-ADM1.geojson
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default="results")
_ap.add_argument("--out", default="paper_figures")
_ap.add_argument("--boundaries", default=None, help="GeoJSON of province boundaries (optional)")
_A = _ap.parse_args()
RES, OUT = Path(_A.results), Path(_A.out)
BOUNDARIES = Path(_A.boundaries) if _A.boundaries else None
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 600})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
REGIONS = {"Sumatra": [95.0, -6.2, 106.3, 6.0], "Kalimantan": [108.5, -4.3, 119.3, 4.5],
           "South Papua": [137.5, -9.2, 141.1, -5.5]}


def save(f, name, tight=False):
    kw = {"bbox_inches": "tight", "pad_inches": 0.03} if tight else {}
    f.savefig(OUT / f"{name}.png", dpi=600, **kw)
    f.savefig(OUT / f"{name}.pdf", **kw)
    plt.close(f)


# ------------------------------------------------------------------ boundaries
def rings(path):
    if path is None or not Path(path).exists():
        return []
    g = json.load(open(path))
    out = []
    for ft in g["features"]:
        geom = ft["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0])
            if len(ring) > 20:
                out.append(ring[:: max(1, len(ring) // 4000)])
    return out


def basemap(ax, R, xlim=(94.5, 141.6), ylim=(-9.8, 6.4)):
    for ring in R:
        if ring[:, 0].max() < xlim[0] or ring[:, 0].min() > xlim[1]:
            continue
        ax.fill(ring[:, 0], ring[:, 1], color="#f2f1ee", lw=0, zorder=0)
        ax.plot(ring[:, 0], ring[:, 1], color="#a8a6a0", lw=0.25, zorder=1)
    for name, (w, s, e, n) in REGIONS.items():
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, lw=0.6, ls=(0, (3, 2)), ec=INK2, zorder=3))
        ax.text(w + 0.2 if name != "South Papua" else w - 0.2, n - 0.25 if name != "South Papua" else n + 0.15,
                name, fontsize=7, color=INK2, va="top" if name != "South Papua" else "bottom",
                ha="left" if name != "South Papua" else "left", zorder=4)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks(range(95, 145, 5)); ax.set_xticklabels([f"{x}°E" for x in range(95, 145, 5)])
    ax.set_yticks([-8, -4, 0, 4]); ax.set_yticklabels(["8°S", "4°S", "0°", "4°N"])
    ax.tick_params(labelsize=7, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    # scale bar (valid at the equator: 1° of longitude = 111.32 km) and north arrow
    x0, y0, km = 121.5, -8.9, 500
    dx = km / 111.32
    ax.add_patch(Rectangle((x0, y0), dx / 2, 0.22, fc=INK, ec=INK, lw=0.4, zorder=5))
    ax.add_patch(Rectangle((x0 + dx / 2, y0), dx / 2, 0.22, fc="white", ec=INK, lw=0.4, zorder=5))
    for x, t, ha in ((x0, "0", "center"), (x0 + dx / 2, "250", "center"), (x0 + dx, "500 km", "left")):
        ax.text(x if ha == "center" else x - 0.3, y0 - 0.25, t, fontsize=5.5, ha=ha, va="top", zorder=5)
    ax.text(x0 + dx / 2, y0 + 0.45, "scale at the equator", fontsize=5.5, ha="center", color=INK2, zorder=5)
    ax.annotate("", xy=(139.9, 5.6), xytext=(139.9, 3.6), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK), zorder=5)
    ax.text(139.9, 5.75, "N", fontsize=8, ha="center", va="bottom", fontweight="bold", zorder=5)


def fig_study_area():
    R = rings(BOUNDARIES)
    cells = pd.read_csv(RES / "followup3" / "cells_peat_regime.csv")
    f, ax = plt.subplots(figsize=(7.1, 3.0), constrained_layout=True)
    basemap(ax, R)
    cmap = LinearSegmentedColormap.from_list("peat", SEQ)
    sc = ax.scatter(cells["lon"], cells["lat"], c=cells["peat_frac"], s=0.35, marker="s", lw=0, cmap=cmap,
                    vmin=0, vmax=1, rasterized=True, zorder=2)
    cb = f.colorbar(sc, ax=ax, shrink=0.75, pad=0.01, aspect=25)
    cb.set_label("Peat fraction of cell", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2)
    ax.set_title(f"{len(cells):,} recurrently burning 0.01° cells (≥ 20 VIIRS detections, 2012–2025)",
                 fontsize=8, loc="left", color=INK)
    save(f, "fig02_study_area")


def fig_regime_map(labels_csv, name, title):
    R = rings(BOUNDARIES)
    lab = pd.read_csv(labels_csv)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#17becf", "#000000", "#6b8e23"]
    f, ax = plt.subplots(figsize=(7.1, 3.3), constrained_layout=True)
    basemap(ax, R)
    for r in sorted(lab["regime"].unique()):
        d = lab[lab["regime"] == r]
        ax.scatter(d["lon"], d["lat"], s=0.35, marker="s", lw=0, color=colors[int(r) % 12], rasterized=True, zorder=2,
                   label=f"{int(r)} (n = {len(d):,})")
    ax.legend(title="Regime", fontsize=6, title_fontsize=6.5, markerscale=9, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), handletextpad=0.2, columnspacing=0.8)
    ax.set_title(f"{title} ({len(lab):,} cells)", fontsize=8, loc="left")
    save(f, name)


# ------------------------------------------------------------------ occurrence disagreement by year
def fig_occurrence_by_year():
    D = json.load(open(RES / "followup3" / "disagreement.json"))["occurrence"]
    f, axs = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)
    for dom, col, name, mk in (("forecast_domain", S1, "Forecast domain", "o"), ("union_domain", S2, "Union domain", "s")):
        lab = f"{name} ({D[dom]['n_cells']:,} cells)"
        by = D[dom]["by_year"]
        yrs = [int(y) for y in by]
        for ax, k in zip(axs, ("D_fire", "A_share")):
            v = np.array([100 * by[y][k] for y in by])
            lo = v - np.array([100 * by[y][k + "_ci_low"] for y in by])
            hi = np.array([100 * by[y][k + "_ci_high"] for y in by]) - v
            off = -0.08 if dom == "forecast_domain" else 0.08
            ax.errorbar(np.array(yrs) + off, v, yerr=[lo, hi], color=col, marker=mk, ms=4, lw=1.4, capsize=2,
                        elinewidth=0.8, label=lab)
    axs[0].set_ylabel("Fire cell-months seen by one\nsensor only (% of fire cell-months)")
    axs[1].set_ylabel("Allocation share of disagreement,\nA / D (%)")
    axs[0].set_ylim(0, 80); axs[1].set_ylim(70, 101)
    for ax, t in zip(axs, ("(a) Disagreement among fire cell-months", "(b) Allocation versus quantity")):
        ax.set_xticks(range(2018, 2026)); ax.tick_params(labelsize=7)
        ax.set_title(t, fontsize=8, loc="left"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axs[0].legend(fontsize=6.5, loc="lower left")
    for y in (2019, 2023):
        for ax in axs:
            ax.axvspan(y - 0.45, y + 0.45, color="#f2f1ee", zorder=0, lw=0)
    axs[0].text(2019, 76, "El Niño", fontsize=6, ha="center", color=INK2)
    axs[0].text(2023, 76, "El Niño", fontsize=6, ha="center", color=INK2)
    save(f, "fig04_occurrence_disagreement")


# ------------------------------------------------------------------ forecast skill (as report.py, Figure 7)
def fig_skill():
    rf = json.load(open(RES / "forecast" / "results_forecast.json"))
    MODELS = ["clim", "fwi", "global", "global_regime", "moe"]
    NAME = {"clim": "Climatology", "fwi": "FWI", "global": "Global", "global_regime": "Global + regime",
            "moe": "Mixture of experts"}
    DES = {"spatial_cv": "(a) Spatial block CV, 2012–2025", "loeno": "(b) Leave-one-El-Niño-out",
           "holdout2026": "(c) 2026 hold-out"}
    cols = [S1, S2, S3, S4, S5]
    leads = ["L1", "L2", "L3"]
    f, axs = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=True, layout="constrained")
    for a, d in zip(axs, DES):
        for i, m in enumerate(MODELS):
            v = [rf.get(d, {}).get(m, {}).get(L, {}) for L in leads]
            y = np.array([x.get("auc_pr", np.nan) for x in v])
            lo = y - np.array([x.get("auc_pr_ci_low", np.nan) for x in v])
            hi = np.array([x.get("auc_pr_ci_high", np.nan) for x in v]) - y
            a.bar(np.arange(3) + (i - 2) * 0.16, y, 0.15, yerr=[lo, hi], capsize=1.2, color=cols[i], label=NAME[m],
                  error_kw=dict(lw=0.6, ecolor=INK2), edgecolor="white", linewidth=0.3)
        a.set_xticks(range(3), ["1 month", "2 months", "3 months"]); a.set_title(DES[d], fontsize=7.5, loc="left")
        a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=7)
    axs[0].set_ylabel("AUC-PR")
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, fontsize=6.5, loc="outside lower center", ncol=5, handlelength=1.2, columnspacing=1.2)
    save(f, "fig05_skill")


# ------------------------------------------------------------------ workflow
def fig_workflow():
    f, ax = plt.subplots(figsize=(7.1, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#ffffff", ec="#52514e", bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=7, fontweight="bold" if bold else None)
        ax.text(x + w / 2, y + h - 4.0, body, ha="center", va="top", fontsize=5.8, color=INK2, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK2))

    # inputs
    box(1, 44, 22, 20, "Public inputs", "VIIRS 375 m active fire\n(S-NPP, NOAA-20; FIRMS API)\nCHIRPS, ERA5-Land, MODIS,\n"
        "WorldCover, AlphaEarth\n(Google Earth Engine)\nFWI (Copernicus EWDS)\nPeat map, ONI, DMI", fc="#f7f7f5")
    box(27, 44, 22, 20, "Datacube builder", "fetch_data.py stages:\nfirms → grid → candidates →\nindices → gee_monthly →\n"
        "gee_static → peat → fwi →\nverify (resumable)\n0.01° cell × month tables", fc="#eef4fd", ec=S1)
    box(53, 44, 22, 20, "Registration", "register.py: SHA-256 of\nanalysis parameters and of\nthe six analysis modules;\n"
        "REGISTRATION.md → OSF\nrun_all.py refuses to run\nif a hash differs", fc="#fdf0ea", ec=S2)
    box(79, 44, 20, 20, "Tests", "make_synthetic.py:\nplanted regimes\nsmoke_test.py and\nfollow-up smoke tests\n"
        "(no internet; outputs\nnever reported)", fc="#f7f7f5")
    arrow(23.6, 54, 26.4, 54); arrow(49.6, 54, 52.4, 54)
    # analysis
    box(1, 20, 30, 20, "Fingerprints and clustering", "regime_features.py: 10 fire\nfeatures + peat fraction +\n8 embedding components\n"
        "cluster_regimes.py: k-means,\nGMM, HDBSCAN, autoencoder;\nk by registered rule;\n50 × 80 % resampling", fc="#eef4fd", ec=S1)
    box(35, 20, 30, 20, "Forecasting and substitution", "forecast.py: 5 models,\n4 predictor sets, 3 leads;\n"
        "0.5° block CV, leave-one-\nEl-Niño-out, 2026 hold-out;\nblock bootstrap (1,000)", fc="#eef4fd", ec=S1)
    box(69, 20, 30, 20, "Reporting", "report.py → results.json,\ntables, figures\nfill_manuscript.py writes\n"
        "every number into the\nmanuscript (no transcription)", fc="#eef4fd", ec=S1)
    arrow(38, 43.6, 16, 40.4); arrow(38, 43.6, 50, 40.4); arrow(31.6, 30, 34.4, 30); arrow(65.6, 30, 68.4, 30)
    arrow(64, 43.6, 64, 40.4)
    # follow-ups
    box(1, 1, 98, 15, "Follow-up diagnostics (exploratory, fixed in time-stamped addenda)",
        "susceptibility.py: static task, same learner      supplement_s1.py: alternative partition (k-means, k = 4)\n"
        "intersensor.py: factor-wise ARI decomposition (record, sensor refitted, sensor with model held fixed, resampling)\n"
        "and agreement along the processing chain      disagreement.py: total, quantity and allocation disagreement",
        fc="#f1fbf7", ec=S3)
    arrow(50, 19.6, 50, 16.4)
    save(f, "fig01_workflow", tight=True)


def fig_change_map(shift_dir):
    """Cells by status between the 2023 and 2026 event years (same regime, changed, appeared, disappeared)."""
    R = rings(BOUNDARIES)
    a = pd.read_csv(shift_dir / "assigned_2023.csv").set_index("cell")["regime"]
    b = pd.read_csv(shift_dir / "assigned_2026.csv").set_index("cell")["regime"]
    both = a.index.intersection(b.index)
    groups = {"same regime": both[(a.loc[both] == b.loc[both]).to_numpy()],
              "changed regime": both[(a.loc[both] != b.loc[both]).to_numpy()],
              "burned only in 2026 (appeared)": b.index.difference(a.index),
              "burned only in 2023 (disappeared)": a.index.difference(b.index)}
    colors = {"same regime": "#8a8984", "changed regime": S5, "burned only in 2026 (appeared)": S2,
              "burned only in 2023 (disappeared)": S1}
    f, ax = plt.subplots(figsize=(7.1, 3.3), constrained_layout=True)
    basemap(ax, R)
    counts = {}
    for k in ("burned only in 2023 (disappeared)", "burned only in 2026 (appeared)", "same regime", "changed regime"):
        cells = np.asarray(groups[k], dtype=np.int64)
        iy, ix = cells // 100000, cells % 100000
        lat, lon = iy * 0.01 - 90 + 0.005, ix * 0.01 - 180 + 0.005
        counts[k] = len(cells)
        ax.scatter(lon, lat, s=0.35, marker="s", lw=0, color=colors[k], rasterized=True, zorder=2,
                   label=f"{k} (n = {len(cells):,})")
    ax.legend(fontsize=6, markerscale=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Cells by status between the 2023 and 2026 event years", fontsize=8, loc="left")
    save(f, "figS8_change_map")
    return counts


if __name__ == "__main__":
    fig_workflow()
    fig_study_area()
    fig_occurrence_by_year()
    fig_skill()
    fig_regime_map(RES / "followup3" / "cells_peat_regime.csv", "figS2_regime_map",
                   "Registered partition: Gaussian mixture, k = 11")
    fig_regime_map(RES / "followup" / "s1" / "labels_kmeans4.csv", "figS5_kmeans4_map",
                   "Alternative partition: k-means, k = 4")
    print("change-map counts:", fig_change_map(RES / "shift"))
    print("figures ->", OUT)
